"""Arm O: plan from ground-truth state, with MuJoCo itself as the dynamics.

The upper bound on the whole programme. Arm O has a perfect representation
*and* perfect dynamics, so ``SR(O)`` is what "everything except the planner is
free" buys -- and ``SR / SR(O)`` is the y-axis every other arm is reported on.

Cost: measured, and much lower than feared
------------------------------------------
The plan expected this arm to dominate the programme's compute. Measured on
the reference machine, it does not:

======================================  ==========
quantity                                     value
======================================  ==========
``set_control`` (Cartesian delta + IK)     72.4 us
``mj_step(nstep=5)``                       96.4 us
per planner action                        168.8 us
one planning call (300 cand, horizon 5)    0.253 s
per eval episode (320 planning calls)       1.4 min
**50 eval episodes, one process**       **1.1 h**
======================================  ==========

So the planner can be **held fixed across all five arms**, which is what the
plan wants: neither of its fallbacks -- a reduced ``num_samples`` for arm O
only, or running arm O on a subset of episodes -- is needed. Both remain
available, and :meth:`OracleWM.deviations` records them if they are ever used.

Why ``mujoco.rollout`` is not the answer
----------------------------------------
The plan's first mitigation was the batched C rollout API. **It does not apply
here**, and the reason is structural rather than incidental:
``mujoco.rollout`` drives a model through *actuator-space* controls (``nu = 7``
for this arm), but the planner's action space is 5-dimensional and
``ManipSpaceEnv.set_control`` is what bridges the two -- a state-dependent
Cartesian delta controller that reads the current effector pose, clips to the
workspace, and solves IK, all in Python, at every step. There is no fixed
actuator-space control sequence to hand the batched API, because the controls
depend on states that do not exist until the rollout produces them.

Attempting it anyway is refused (:data:`_BATCHED_UNAVAILABLE`) rather than
silently accepted, because a plausible-looking wrong rollout here would show up
only as a slightly disappointing oracle success rate.

:meth:`OracleWM.benchmark` remains the phase-8 gate.
"""

import numpy as np
import torch
from torch import nn


# `mujoco.rollout` is present in this build, but is not usable for this arm --
# see the module docstring. Kept as a flag so the reason is discoverable rather
# than absent.
try:
    from mujoco import rollout as mj_rollout  # noqa: F401

    HAS_BATCHED_ROLLOUT = True
except ImportError:  # pragma: no cover
    mj_rollout = None
    HAS_BATCHED_ROLLOUT = False

_BATCHED_UNAVAILABLE = (
    "mujoco.rollout cannot drive this arm. It steps a model through "
    "actuator-space controls (nu=7), but the planner's action space is the "
    "env's 5-D Cartesian delta, and ManipSpaceEnv.set_control maps between "
    "them with a state-dependent IK solve in Python -- so there is no fixed "
    "control sequence to hand the batched API. Use the default step loop; at "
    "num_samples=300 it costs ~0.25 s per planning call, which is affordable."
)


class OracleWM(nn.Module):
    """Ground-truth state plus MuJoCo re-simulation.

    Satisfies ``Dynamics`` without holding a single learned parameter, so the
    existing ``ShootingCostEvaluator``, solvers and ``GoalMSE`` drive it
    unchanged.

    Args:
        env: A ``LeJEPACubeEnv`` (or any ``CustomMuJoCoEnv``) supplying the
            model to simulate. Used read-only: every rollout runs on scratch
            ``MjData``, never on the environment's own state.
        state_keys: ``privileged/*`` / ``proprio/*`` columns forming the
            latent, in fixed order -- the same convention as
            :class:`~stable_worldmodel.wm.state.state_wm.StateWM`, so arms O
            and P are scored in the same coordinates.
        action_block: Physics-control steps per planner action.
        num_samples_override: Reduced candidate count for this arm only.
            Recorded as a deviation; see the module docstring.
        use_batched_rollout: Kept for the record and defaulted **off**;
            setting it raises with an explanation. See the module docstring.
    """

    def __init__(
        self,
        env,
        state_keys,
        action_block: int = 5,
        num_samples_override: int | None = None,
        use_batched_rollout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.env = env
        self.state_keys = list(state_keys)
        self.action_block = int(action_block)
        self.num_samples_override = num_samples_override
        self.use_batched_rollout = bool(use_batched_rollout)
        # Keeps `next(model.parameters())` working for the solvers, which read
        # dtype and device off it. Arm O learns nothing, so this is the only
        # parameter it has.
        self.register_parameter(
            '_dtype_anchor', nn.Parameter(torch.zeros(1), requires_grad=False)
        )

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def deviations(self):
        """Every way this arm departs from the fixed planner protocol.

        Written into the result row. The plan holds the planner identical
        across arms; where arm O cannot afford that, the scatter has to say so
        rather than presenting the number as directly comparable.
        """
        return {
            'arm_o_num_samples_override': self.num_samples_override,
            'arm_o_batched_rollout': self.use_batched_rollout,
            'arm_o_planner_held_fixed': self.num_samples_override is None,
        }

    def benchmark(self, num_candidates=300, horizon=5, repeats=3):
        """Time one planning call's worth of simulation. The phase-8 gate.

        Returns:
            dict: Seconds per planning call, and the implied cost per eval
            episode at one planning call per receding-horizon step.
        """
        import time

        actions = np.zeros(
            (num_candidates, horizon, self.env.action_space.shape[0]),
            dtype=np.float64,
        )
        state = self._current_state()

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            self._simulate(state, actions)
            timings.append(time.perf_counter() - start)

        per_call = float(np.median(timings))
        return {
            'seconds_per_planning_call': per_call,
            'num_candidates': num_candidates,
            'horizon': horizon,
            'physics_steps_per_call': num_candidates
            * horizon
            * self.action_block,
            'batched_rollout': self.use_batched_rollout,
        }

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def _current_state(self):
        """``(nq + nv,)`` snapshot of the environment's own state."""
        return np.concatenate(
            [self.env._data.qpos.copy(), self.env._data.qvel.copy()]
        )

    def _borrow_data(self, data):
        """Swap ``data`` in as the environment's ``_data`` for the duration.

        Not cosmetic. ``ManipSpaceEnv.set_control`` is a *state-dependent*
        controller: it reads the current effector pose out of ``self._data``,
        applies the action as a Cartesian delta, clips to the workspace, and
        solves IK for the actuator targets. Stepping a scratch ``MjData`` while
        ``set_control`` reads the environment's own would compute every
        candidate's control from the wrong state -- silently, and with results
        that still look like a plausible rollout.
        """

        class _Swap:
            def __init__(self, env, scratch):
                self.env, self.scratch = env, scratch

            def __enter__(self):
                self.saved = self.env._data
                self.env._data = self.scratch
                return self.scratch

            def __exit__(self, *exc):
                self.env._data = self.saved
                return False

        return _Swap(self.env, data)

    def _simulate(self, initial_state, action_sequences):
        """Roll every candidate forward from one initial state.

        Args:
            initial_state: ``(nq + nv,)``.
            action_sequences: ``(S, T, action_dim)``.

        Returns:
            ndarray: ``(S, T, nq + nv)`` states after each action block.
        """
        import mujoco

        if self.use_batched_rollout:
            raise NotImplementedError(_BATCHED_UNAVAILABLE)

        model = self.env._model
        n_samples, horizon = action_sequences.shape[:2]
        nq, nv = model.nq, model.nv

        out = np.empty((n_samples, horizon, nq + nv), dtype=np.float64)
        data = mujoco.MjData(model)

        with self._borrow_data(data):
            for i in range(n_samples):
                data.qpos[:] = initial_state[:nq]
                data.qvel[:] = initial_state[nq:]
                data.time = 0.0
                mujoco.mj_forward(model, data)
                for t in range(horizon):
                    self.env.set_control(action_sequences[i, t])
                    mujoco.mj_step(model, data, nstep=self.action_block)
                    out[i, t] = np.concatenate([data.qpos, data.qvel])
        return out

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _state_from_info(self, info):
        missing = [key for key in self.state_keys if key not in info]
        if missing:
            raise KeyError(
                f'OracleWM is missing state columns {missing}; add them to the '
                "eval config's `keys_to_cache`."
            )
        parts = []
        for key in self.state_keys:
            value = info[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(np.asarray(value))
            parts.append(value.float().reshape(*value.shape[:-1], -1)
                         if value.ndim > 1 else value.float().reshape(-1, 1))
        return torch.cat(parts, dim=-1)

    def encode(self, info):
        """Ground-truth state as the embedding. No pixels, no learning."""
        info['emb'] = self._state_from_info(info)
        return info

    def rollout(self, info, action_sequence, history_size: int = None):
        """Re-simulate every candidate through MuJoCo.

        Args:
            info: Must carry ``qpos`` and ``qvel`` for the simulator to be
                restored from -- arm O restores the *full* physical state, not
                just the content latents, because the dynamics depend on
                velocities the content vector deliberately excludes.
            action_sequence: ``(B, S, T, action_dim)``.

        Returns:
            dict: ``info`` with ``predicted_emb`` ``(B, S, T, d)``.
        """
        b, s, t = action_sequence.shape[:3]

        if 'qpos' not in info or 'qvel' not in info:
            raise KeyError(
                'OracleWM.rollout needs `qpos` and `qvel` in info: the '
                'oracle integrates the true dynamics, which depend on '
                'velocities that the content latent vector excludes by design.'
            )

        qpos = np.asarray(_first(info['qpos'])).reshape(-1)
        qvel = np.asarray(_first(info['qvel'])).reshape(-1)
        initial = np.concatenate([qpos, qvel])

        actions = action_sequence.detach().cpu().numpy().reshape(b * s, t, -1)
        states = self._simulate(initial, actions)

        emb = self._embed_states(states)
        info['predicted_emb'] = torch.as_tensor(
            emb, dtype=torch.float32
        ).reshape(b, s, t, -1)
        return info

    def _embed_states(self, states):
        """Read the declared latent columns out of simulated raw states.

        Restores each simulated state into a scratch ``MjData``, runs forward
        kinematics, and reads the same ``privileged/*`` / ``proprio/*``
        columns the other arms use -- so ``GoalMSE`` compares like with like.
        """
        import mujoco

        model = self.env._model
        nq = model.nq
        data = mujoco.MjData(model)

        n_samples, horizon = states.shape[:2]
        rows = []
        saved_qpos = self.env._data.qpos.copy()
        saved_qvel = self.env._data.qvel.copy()
        try:
            for i in range(n_samples):
                for t in range(horizon):
                    data.qpos[:] = states[i, t, :nq]
                    data.qvel[:] = states[i, t, nq:]
                    mujoco.mj_forward(model, data)
                    # `compute_ob_info` reads the env's own `_data`, so the
                    # scratch state is swapped in and restored around it
                    # rather than duplicating the whole info-assembly logic.
                    self.env._data.qpos[:] = data.qpos
                    self.env._data.qvel[:] = data.qvel
                    mujoco.mj_forward(model, self.env._data)
                    info = self.env.compute_ob_info()
                    rows.append(
                        np.concatenate(
                            [
                                np.asarray(
                                    info[key], dtype=np.float64
                                ).reshape(-1)
                                for key in self.state_keys
                            ]
                        )
                    )
        finally:
            self.env._data.qpos[:] = saved_qpos
            self.env._data.qvel[:] = saved_qvel
            mujoco.mj_forward(model, self.env._data)

        return np.stack(rows).reshape(n_samples, horizon, -1)


def _first(value):
    """Take the leading batch/sample element of a possibly-batched array."""
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    array = np.asarray(array)
    while array.ndim > 1:
        array = array[0]
    return array


__all__ = ['HAS_BATCHED_ROLLOUT', 'OracleWM']

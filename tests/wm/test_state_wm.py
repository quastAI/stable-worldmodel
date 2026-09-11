"""Tests for the state-space stage-D arms (P and O).

Both arms share one failure mode that no exception would catch on its own:
they read their latent out of ``privileged/*`` columns that the *eval config*
has to cache. A missing column produces a constant embedding, a flat cost
surface, and a success rate that looks exactly like a catastrophic
identifiability failure. Most of what follows is about making that impossible.
"""

import os

import numpy as np
import pytest
import torch


os.environ.setdefault('MUJOCO_GL', 'glfw')

pytest.importorskip('ogbench')
pytest.importorskip('mujoco')

from stable_worldmodel.protocols import Dynamics  # noqa: E402
from stable_worldmodel.wm.lewm.module import Embedder, Predictor  # noqa: E402
from stable_worldmodel.wm.state import OracleWM, StateWM  # noqa: E402


KEYS = ['privileged/block_0_pos', 'proprio/effector_pos']
D = 6


@pytest.fixture
def state_wm():
    predictor = Predictor(
        num_frames=3, depth=1, heads=2, mlp_dim=16,
        input_dim=D, hidden_dim=16, output_dim=D, dim_head=8,
    )
    return StateWM(
        predictor,
        Embedder(input_dim=5, emb_dim=D),
        KEYS,
        mean=np.zeros(D),
        std=np.ones(D),
    )


@pytest.fixture(scope='module')
def env():
    from stable_worldmodel.envs.ogbench.lejepa_cube_env import LeJEPACubeEnv

    e = LeJEPACubeEnv(
        env_type='single',
        ob_type='states',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=64,
        height=64,
        num_digits=1,
    )
    e.reset(seed=0, options={'variation': ['all']})
    yield e
    e.close()


def info_batch(b=4):
    return {
        'privileged/block_0_pos': torch.randn(b, 3),
        'proprio/effector_pos': torch.randn(b, 3),
    }


# ------------------------------------------------------------------- arm P


def test_state_wm_is_dynamics(state_wm):
    assert isinstance(state_wm, Dynamics)


def test_encode_never_looks_at_pixels(state_wm):
    """Arm P is the perfect-representation ceiling; pixels are not involved."""
    info = info_batch()
    out = state_wm.encode(dict(info))
    assert out['emb'].shape == (4, D)

    # Same state, wildly different pixels -> identical embedding.
    with_pixels = {**info, 'pixels': torch.randn(4, 3, 64, 64)}
    torch.testing.assert_close(
        state_wm.encode(with_pixels)['emb'], out['emb']
    )


def test_missing_column_raises_rather_than_returning_zeros(state_wm):
    """The plan's SS5.E integration risk, made impossible.

    Reading zeros here would be indistinguishable downstream from an encoder
    that learned nothing.
    """
    with pytest.raises(KeyError, match='keys_to_cache'):
        state_wm.encode({'privileged/block_0_pos': torch.randn(4, 3)})


def test_state_column_order_is_fixed(state_wm):
    """A checkpoint must always mean the same thing.

    Dict iteration order deciding the layout would make a saved whitening mean
    silently wrong for a differently-ordered info dict.
    """
    info = info_batch(2)
    forward = state_wm.gather_state(info)
    reordered = state_wm.gather_state(
        {k: info[k] for k in reversed(list(info))}
    )
    torch.testing.assert_close(forward, reordered)


def test_assert_state_varies_is_the_phase_8_gate(state_wm):
    """Columns can be cached and still arrive constant."""
    report = state_wm.assert_state_varies(np.random.randn(50, D))
    assert report['constant_dims'] == 0

    with pytest.raises(ValueError, match='constant across'):
        state_wm.assert_state_varies(np.ones((50, D)))


def test_whitening_uses_stored_buffers():
    """Whitening stats must travel with the checkpoint, not be recomputed."""
    predictor = Predictor(
        num_frames=3, depth=1, heads=2, mlp_dim=16,
        input_dim=D, hidden_dim=16, output_dim=D, dim_head=8,
    )
    model = StateWM(
        predictor,
        Embedder(input_dim=5, emb_dim=D),
        KEYS,
        mean=np.full(D, 2.0),
        std=np.full(D, 4.0),
    )
    assert 'mean' in dict(model.named_buffers())
    torch.testing.assert_close(
        model.whiten(torch.full((1, D), 6.0)), torch.ones(1, D)
    )


def test_state_wm_rollout_shape(state_wm):
    b, s, t = 2, 3, 4
    info = {
        'privileged/block_0_pos': torch.randn(b, 1, 3),
        'proprio/effector_pos': torch.randn(b, 1, 3),
    }
    out = state_wm.rollout(info, torch.randn(b, s, t, 5))
    assert out['predicted_emb'].shape[:3] == (b, s, 1 + t)


# ------------------------------------------------------------------- arm O


def test_oracle_is_dynamics(env):
    assert isinstance(OracleWM(env, KEYS), Dynamics)


def test_oracle_leaves_the_environment_untouched(env):
    """Every rollout must run on scratch state, never the env's own.

    The planner calls this mid-episode; mutating the environment would corrupt
    the very episode being evaluated.
    """
    oracle = OracleWM(env, KEYS, action_block=5)
    before_q = env._data.qpos.copy()
    before_v = env._data.qvel.copy()

    actions = np.zeros((4, 3, env.action_space.shape[0]))
    oracle._simulate(oracle._current_state(), actions)

    np.testing.assert_allclose(env._data.qpos, before_q)
    np.testing.assert_allclose(env._data.qvel, before_v)


def test_oracle_candidates_diverge_with_their_actions(env):
    """The bug this catches: ``set_control`` reading the wrong ``MjData``.

    ``ManipSpaceEnv.set_control`` is a state-dependent Cartesian controller
    that reads ``self._data``. If the scratch state is not swapped in, every
    candidate's control is computed from the *environment's* state instead of
    its own -- and the rollouts still look plausible.
    """
    oracle = OracleWM(env, KEYS, action_block=5)
    actions = np.zeros((5, 3, env.action_space.shape[0]))
    actions[:, :, 0] = np.linspace(-0.05, 0.05, 5)[:, None]

    states = oracle._simulate(oracle._current_state(), actions)
    spread = np.abs(states[:, -1] - states[0, -1]).max(axis=-1)

    assert spread[0] == pytest.approx(0.0)
    assert np.all(np.diff(spread) > 0), (
        f'candidates did not diverge monotonically with action size: {spread}'
    )


def test_oracle_is_deterministic(env):
    oracle = OracleWM(env, KEYS, action_block=5)
    actions = np.full((3, 2, env.action_space.shape[0]), 0.01)
    state = oracle._current_state()
    np.testing.assert_allclose(
        oracle._simulate(state, actions), oracle._simulate(state, actions)
    )


def test_oracle_rollout_needs_full_physical_state(env):
    """Arm O integrates true dynamics, which depend on velocities.

    The content vector excludes ``qvel`` by design -- it has no single-frame
    image correlate -- so the oracle must be handed the raw state instead of
    reconstructing it from the latents.
    """
    oracle = OracleWM(env, KEYS, action_block=5)
    with pytest.raises(KeyError, match='qpos'):
        oracle.rollout({}, torch.zeros(1, 2, 3, env.action_space.shape[0]))


def test_oracle_refuses_the_batched_api_with_an_explanation(env):
    """``mujoco.rollout`` cannot drive a Python Cartesian controller.

    Refused rather than silently accepted: a plausible-looking wrong rollout
    would surface only as a slightly disappointing oracle success rate.
    """
    oracle = OracleWM(env, KEYS, use_batched_rollout=True)
    with pytest.raises(NotImplementedError, match='actuator-space'):
        oracle._simulate(
            oracle._current_state(),
            np.zeros((2, 2, env.action_space.shape[0])),
        )


def test_oracle_reports_its_deviations(env):
    """The planner is held fixed across arms; departures must be recorded."""
    assert OracleWM(env, KEYS).deviations()['arm_o_planner_held_fixed'] is True
    assert (
        OracleWM(env, KEYS, num_samples_override=100)
        .deviations()['arm_o_planner_held_fixed']
        is False
    )


def test_oracle_benchmark_reports_a_usable_cost(env):
    """The phase-8 gate must produce a number before the arm is committed to."""
    oracle = OracleWM(env, KEYS, action_block=5)
    report = oracle.benchmark(num_candidates=8, horizon=2, repeats=1)
    assert report['seconds_per_planning_call'] > 0
    assert report['physics_steps_per_call'] == 8 * 2 * 5


def test_arms_p_and_o_share_state_coordinates(env, state_wm):
    """``GoalMSE`` compares embeddings; the two arms must speak the same ones."""
    assert OracleWM(env, KEYS).state_keys == state_wm.state_keys

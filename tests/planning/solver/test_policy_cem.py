"""Tests for PolicyCEMSolver: CEM seeded from a goal-conditioned policy head.

The contract under test is narrow but easy to break silently:

  * iteration 0 must sample from the head's ``(mu, sigma)`` rather than from
    ``N(0, var_scale)`` — if the mean never reaches the sampler the warm start
    is a no-op that still *looks* like it works, because CEM converges anyway;
  * ``sigma`` must be used as a **standard deviation**, since that is what
    ``CEMSolver``'s misleadingly-named ``var`` actually is;
  * ``var_floor`` must survive a confidently-wrong head, or the search
    collapses onto the proposal;
  * ``'replace'`` must discard a caller-supplied warm start and ``'tail'``
    must keep it, since that distinction is the experiment;
  * device/dtype must line up, the failure mode
    ``tests/planning/solver/test_warm_start_device_mismatch.py`` was written
    for.
"""

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from stable_worldmodel.planning.solver.policy_cem import PolicyCEMSolver


B, HORIZON, ACT_DIM, ACTION_BLOCK = 2, 4, 3, 1
BLOCKED_DIM = ACT_DIM * ACTION_BLOCK


class _PlanConfig:
    horizon = HORIZON
    action_block = ACTION_BLOCK


class _RecordingCost:
    """Cost stub that records the candidates it is asked to score.

    Mirrors the ``ShootingCostEvaluator`` seam: the solver reaches the model
    through ``self.cost.model``.
    """

    def __init__(self, model):
        self.model = model
        self.seen = []

    def get_cost(self, info_dict, candidates):
        self.seen.append(candidates.detach().clone())
        # Cost grows with distance from zero, so the elite set is pulled
        # toward the origin and away from any nonzero proposal -- which makes
        # "the proposal reached iteration 0" a distinguishable claim.
        return candidates.pow(2).sum(dim=(2, 3))


class _StubActor:
    """Model stub exposing only the policy-head surface the solver needs."""

    def __init__(self, mu, std, dtype=torch.float32, device='cpu'):
        self._mu = mu
        self._std = std
        self.dtype = dtype
        self.device = device
        self.calls = []

    def get_action_distribution(self, info_dict, horizon=None):
        self.calls.append(horizon)
        mu = self._mu.to(device=self.device, dtype=self.dtype)
        std = (
            None
            if self._std is None
            else self._std.to(device=self.device, dtype=self.dtype)
        )
        return mu, std


def _solver(actor, n_steps=1, num_samples=64, **kwargs):
    cost = _RecordingCost(actor)
    solver = PolicyCEMSolver(
        cost=cost,
        batch_size=B,
        num_samples=num_samples,
        n_steps=n_steps,
        topk=8,
        device='cpu',
        seed=0,
        **kwargs,
    )
    solver.configure(
        action_space=Box(-1.0, 1.0, shape=(B, ACT_DIM)),
        n_envs=B,
        config=_PlanConfig(),
    )
    return solver, cost


def _info():
    return {'pixels': torch.randn(B, 3, 3, 8, 8)}


def _const_plan(mu_value, std_value=1.0):
    mu = torch.full((B, HORIZON, BLOCKED_DIM), float(mu_value))
    std = (
        None
        if std_value is None
        else torch.full((B, HORIZON, BLOCKED_DIM), float(std_value))
    )
    return mu, std


def test_first_iteration_samples_around_the_policy_mean():
    """The head's mean must reach the sampler, not just be computed."""
    mu, std = _const_plan(5.0, 0.01)
    solver, cost = _solver(_StubActor(mu, std))

    solver.solve(_info())

    first = cost.seen[0]
    assert first.shape == (B, solver.num_samples, HORIZON, BLOCKED_DIM)
    # tight std around 5.0 -> every candidate sits near the proposal
    assert first.mean().item() == pytest.approx(5.0, abs=0.05)


def test_forced_first_candidate_is_exactly_the_policy_mean():
    """CEMSolver pins candidate 0 to the current mean; with a policy warm
    start that candidate is the head's plan verbatim."""
    mu, std = _const_plan(2.5, 0.5)
    solver, cost = _solver(_StubActor(mu, std))

    solver.solve(_info())

    torch.testing.assert_close(cost.seen[0][:, 0], mu)


def test_solve_does_not_mutate_the_head_output():
    """CEMSolver writes its optimized plan back into whatever tensor it was
    handed as `init_action`, so the warm start has to pass a copy -- the
    head's output belongs to the model, not the solver."""
    mu, std = _const_plan(2.5, 1.0)
    before = mu.clone()
    solver, _ = _solver(_StubActor(mu, std), n_steps=3)

    solver.solve(_info())

    torch.testing.assert_close(mu, before)


def test_policy_std_is_used_as_a_standard_deviation():
    """CEM's `var` is really a std (`randn * var + mean`), so the head's
    sigma must be passed through unsquared. A wide sigma has to produce a
    correspondingly wide first population."""
    mu, std = _const_plan(0.0, 3.0)
    solver, cost = _solver(_StubActor(mu, std), num_samples=4096)

    solver.solve(_info())

    assert cost.seen[0].std().item() == pytest.approx(3.0, rel=0.1)


def test_per_dim_std_is_respected():
    """A heteroscedastic sigma must not be collapsed to a scalar."""
    mu = torch.zeros(B, HORIZON, BLOCKED_DIM)
    std = torch.full((B, HORIZON, BLOCKED_DIM), 0.1)
    std[:, 0] = 4.0  # first block much less certain than the rest
    solver, cost = _solver(_StubActor(mu, std), num_samples=4096)

    solver.solve(_info())

    population = cost.seen[0]
    assert population[:, :, 0].std().item() > 2.0
    assert population[:, :, 1:].std().item() < 0.5


def test_var_scale_scales_the_policy_std():
    mu, std = _const_plan(0.0, 1.0)
    solver, cost = _solver(
        _StubActor(mu, std), num_samples=4096, var_scale=0.25
    )

    solver.solve(_info())

    assert cost.seen[0].std().item() == pytest.approx(0.25, rel=0.15)


def test_var_floor_keeps_the_search_alive():
    """A confidently-wrong head would otherwise collapse CEM to a point."""
    mu, std = _const_plan(0.0, 1e-9)
    solver, cost = _solver(
        _StubActor(mu, std), num_samples=4096, var_floor=0.5
    )

    solver.solve(_info())

    assert cost.seen[0].std().item() == pytest.approx(0.5, rel=0.15)


def test_head_without_std_falls_back_to_var_scale():
    """`predict_std=False` still warm-starts the mean; the spread is CEM's."""
    mu, _ = _const_plan(1.0, None)
    solver, cost = _solver(
        _StubActor(mu, None), num_samples=4096, var_scale=2.0
    )

    solver.solve(_info())

    torch.testing.assert_close(cost.seen[0][:, 0], mu)
    assert cost.seen[0].std().item() == pytest.approx(2.0, rel=0.15)


def test_replace_mode_discards_a_caller_warm_start():
    """`replace` is what isolates the head's contribution, so a leftover
    plan from the previous solve must not survive it."""
    mu, std = _const_plan(5.0, 0.01)
    solver, cost = _solver(_StubActor(mu, std))

    leftover = torch.full((B, HORIZON - 1, BLOCKED_DIM), -9.0)
    solver.solve(_info(), init_action=leftover)

    assert cost.seen[0].mean().item() == pytest.approx(5.0, abs=0.05)


def test_tail_mode_keeps_the_caller_prefix():
    """`tail` keeps the previously-optimized prefix and lets the generic
    prepare_init_action path fill the remainder from the actor."""
    mu, std = _const_plan(5.0, 0.01)
    actor = _StubActor(mu, std)
    solver, cost = _solver(actor, warm_start_mode='tail')
    # prepare_init_action reaches the actor through the cost object
    solver.cost.get_action = actor_get_action = (
        lambda info, horizon, prefix_actions=None: mu[:, -horizon:]
    )
    assert actor_get_action is solver.cost.get_action

    leftover = torch.full((B, HORIZON - 1, BLOCKED_DIM), -9.0)
    solver.solve(_info(), init_action=leftover)

    forced = cost.seen[0][:, 0]
    torch.testing.assert_close(forced[:, : HORIZON - 1], leftover)
    torch.testing.assert_close(forced[:, HORIZON - 1 :], mu[:, -1:])


def test_actor_is_asked_for_the_solver_horizon():
    """The solver states its horizon so the model can reject a mismatch."""
    mu, std = _const_plan(0.0)
    actor = _StubActor(mu, std)
    solver, _ = _solver(actor)

    solver.solve(_info())

    assert actor.calls == [HORIZON]


def test_outputs_have_the_usual_cem_shape():
    """A drop-in replacement: WorldModelPolicy reads outputs['actions']."""
    mu, std = _const_plan(0.0)
    solver, _ = _solver(_StubActor(mu, std), n_steps=2)

    out = solver.solve(_info())

    assert out['actions'].shape == (B, HORIZON, BLOCKED_DIM)
    assert len(out['costs']) == B
    assert out['mean'][0].shape == (B, HORIZON, BLOCKED_DIM)
    assert out['var'][0].shape == (B, HORIZON, BLOCKED_DIM)


def test_stashed_std_is_cleared_between_solves():
    """The per-solve std is cached on the instance for init_action_distrib;
    it must not leak into a later solve (or a plain CEMSolver code path)."""
    mu, std = _const_plan(0.0, 2.0)
    solver, _ = _solver(_StubActor(mu, std))

    solver.solve(_info())
    assert solver._policy_std is None

    _, var = solver.init_action_distrib(B)
    torch.testing.assert_close(
        var, solver.var_scale * torch.ones(B, HORIZON, BLOCKED_DIM)
    )


def test_a_model_without_a_policy_head_is_rejected_clearly():
    """Pointing this solver at a plain LeWM/SMWM must say what to use."""

    class _NoHead:
        pass

    solver, _ = _solver(_NoHead())
    with pytest.raises(TypeError, match='get_action_distribution'):
        solver.solve(_info())


def test_numpy_info_entries_survive_expansion():
    """info_dict values are not all tensors; the inherited expansion path
    handles ndarrays and must not be disturbed by the warm start."""
    mu, std = _const_plan(0.0)
    solver, _ = _solver(_StubActor(mu, std))

    info = _info()
    info['variation'] = np.arange(B)
    out = solver.solve(info)

    assert out['actions'].shape == (B, HORIZON, BLOCKED_DIM)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_policy_plan_is_cast_to_the_solver_dtype(dtype):
    """The head may run in a different precision than the solver (bf16 eval);
    mean and std must be cast rather than crashing the sampler."""
    mu, std = _const_plan(1.0, 1.0)
    solver, cost = _solver(_StubActor(mu.to(dtype), std.to(dtype)))
    solver._dtype = dtype

    solver.solve(_info())

    assert cost.seen[0].dtype == dtype


def test_rejects_an_unknown_warm_start_mode():
    mu, std = _const_plan(0.0)
    with pytest.raises(ValueError, match='warm_start_mode'):
        _solver(_StubActor(mu, std), warm_start_mode='nonsense')

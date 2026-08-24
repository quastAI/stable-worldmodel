"""Tests for the GCIDM Dynamics surface, planning seam, and policy head.

GCIDM is LeWM plus a goal-conditioned plan head, so the first two sections
mirror ``tests/wm/test_smwm.py`` / ``tests/wm/test_lewm.py``: GCIDM does not
expose ``get_cost`` (planning goes through
``stable_worldmodel.planning.ShootingCostEvaluator(model, GoalMSE())``), goal
encoding happens once as ``(B, T, D)`` and is reused when pre-injected, and
``rollout`` keeps the ``action_history`` contract the policy feeds it. The
bit-for-bit parity of ``GoalMSE`` with the old ``criterion`` lives in
``tests/planning/test_evaluator.py``.

The last two sections cover what GCIDM adds over SMWM: the
:class:`GoalPolicyHead` itself (shapes, both-endpoint sensitivity, the
parameter budget against the predictor), the
``get_action_distribution``/``get_action`` planning seam including the
short-context left-padding early in an episode, and the fact that the head
sits inside the world model so it survives a
``save_pretrained``/``load_pretrained`` round trip into planning.
"""

import hydra
import pytest
import torch
from torch import nn

from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
from stable_worldmodel.protocols import Actionable, Dynamics
from stable_worldmodel.wm.gcidm.gcidm import GCIDM
from stable_worldmodel.wm.gcidm.module import GoalPolicyHead, Predictor
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

# CEM-like dimensions
B, S, T, D, H, A = 2, 3, 2, 5, 4, 2


def _make_info_dict():
    """Return a CEM-expanded info_dict mimicking what a solver passes to get_cost."""
    return {
        'pixels': torch.randn(B, S, T, 3, 8, 8),
        'goal': torch.randn(B, S, T, 3, 8, 8),
        'action': torch.randn(B, S, H, A),
    }


def _make_action_candidates():
    return torch.randn(B, S, H, A)


def _bare_model():
    """Bypass GCIDM.__init__; we only need the encode/rollout surface."""
    return object.__new__(GCIDM)


def test_gcidm_satisfies_dynamics_protocol():
    """GCIDM exposes encode/rollout, so it is a Dynamics a ShootingCostEvaluator can wrap."""
    assert isinstance(_bare_model(), Dynamics)


def test_gcidm_no_longer_exposes_get_cost():
    """Cost now lives in the ShootingCostEvaluator seam, not on the model."""
    assert not hasattr(_bare_model(), 'get_cost')


def test_cost_evaluator_encodes_goal_once_as_3d():
    """ShootingCostEvaluator(GCIDM, GoalMSE) encodes the goal once and stores it (B, T, D).

    get_cost strips the S axis with v[:, 0] before encoding; GoalMSE then
    re-inserts it via goal_emb[:, None, -1:, :].expand_as(pred_emb).
    """
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    action_candidates = _make_action_candidates()

    encode_calls = []

    def mock_encode(goal_dict):
        encode_calls.append(1)
        assert goal_dict['pixels'].shape == (B, T, 3, 8, 8), (
            f'encode received wrong pixels shape: {goal_dict["pixels"].shape}'
        )
        return {'emb': torch.randn(B, T, D)}

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    model.encode = mock_encode
    model.rollout = mock_rollout

    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info_dict, action_candidates
    )

    assert len(encode_calls) == 1, 'encode should be called exactly once'
    assert cost.shape == (B, S), (
        f'cost shape: expected ({B},{S}), got {cost.shape}'
    )


def test_cost_evaluator_respects_preinjected_goal_emb():
    """A pre-populated goal_emb is reused; encode is not called again."""
    torch.manual_seed(0)
    model = _bare_model()
    info_dict = _make_info_dict()
    info_dict['goal_emb'] = torch.randn(B, T, D)
    action_candidates = _make_action_candidates()

    def encode_must_not_be_called(_):
        raise AssertionError(
            'encode must not be called when goal_emb is already in info_dict'
        )

    def mock_rollout(info, ac):
        info['predicted_emb'] = torch.randn(B, S, T, D)
        return info

    model.encode = encode_must_not_be_called
    model.rollout = mock_rollout

    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info_dict, action_candidates
    )
    assert cost.shape == (B, S)


###############################################
## Real rollout: history / candidate pairing ##
###############################################

# Small dims for the real-rollout tests. The toy predictor adds action
# embeddings to frame embeddings, so action dim == embedding dim.
RB, RS, RD = 2, 3, 4


class _CumsumPredictor(nn.Module):
    """Causal toy predictor: output[t] = sum_{k<=t} (emb[k] + act[k]).

    The last output depends on the whole window, so predictions change
    whenever any past action changes. Records every (emb, act) call.
    """

    def __init__(self, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        self.calls = []

    def forward(self, emb, act_emb):
        self.calls.append((emb.detach().clone(), act_emb.detach().clone()))
        return (emb + act_emb).cumsum(dim=1)


class _RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return x


def _toy_model(num_frames=3):
    return GCIDM(
        encoder=nn.Identity(),  # unused: tests pre-populate info['emb']
        predictor=_CumsumPredictor(num_frames=num_frames),
        action_encoder=_RecordingIdentity(),
    )


def _rollout_info(hist_len, emb=None):
    info = {'pixels': torch.randn(RB, RS, hist_len, 3, 8, 8)}
    info['emb'] = emb if emb is not None else torch.randn(RB, RS, hist_len, RD)
    return info


def _reference_rollout_legacy(emb_init, act_emb_seq, HS):
    """Hand-rolled pre-change rollout semantics (H frames, [H, T-H] split)."""
    H = emb_init.size(1)
    n_steps = act_emb_seq.size(1) - H
    emb_list = list(emb_init.unbind(dim=1))
    for t in range(n_steps + 1):
        lo = max(0, H + t - HS)
        emb_trunc = torch.stack(emb_list[lo:], dim=1)
        act_trunc = act_emb_seq[:, lo : H + t]
        emb_list.append((emb_trunc + act_trunc).cumsum(dim=1)[:, -1])
    return torch.stack(emb_list, dim=1)


def test_rollout_h1_matches_legacy_semantics():
    """With a single context frame the new contract is bit-identical to the
    old one (first candidate = action paired with the current frame)."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(hist_len=1)
    candidates = torch.randn(RB, RS, 5, RD)

    out = model.rollout(dict(info), candidates)

    emb_flat = info['emb'].reshape(RB * RS, 1, RD)
    cand_flat = candidates.reshape(RB * RS, 5, RD)
    expected = _reference_rollout_legacy(emb_flat, cand_flat, HS=3)
    torch.testing.assert_close(
        out['predicted_emb'].reshape(RB * RS, -1, RD), expected
    )
    assert 'action_history' not in out
    torch.testing.assert_close(out['action'], candidates[:, :, :1])


def test_rollout_consumes_past_actions_in_order():
    """action_encoder must receive cat([action_history, candidates])."""
    torch.manual_seed(0)
    model = _toy_model()
    info = _rollout_info(hist_len=3)
    past = torch.randn(RB, RS, 2, RD)
    candidates = torch.randn(RB, RS, 4, RD)
    info['action_history'] = past

    model.rollout(info, candidates)

    seq = model.action_encoder.inputs[-1]  # (BS, H-1+T, A)
    expected = torch.cat([past, candidates], dim=2).reshape(RB * RS, 6, RD)
    torch.testing.assert_close(seq, expected)
    torch.testing.assert_close(
        info['action'], torch.cat([past, candidates[:, :, :1]], dim=2)
    )


def test_rollout_output_length_and_past_sensitivity():
    """Output holds H context + T predicted frames, and predictions react
    to the executed past actions (they are inputs, not dead weight)."""
    torch.manual_seed(0)
    model = _toy_model()
    emb = torch.randn(RB, RS, 3, RD)
    candidates = torch.randn(RB, RS, 4, RD)

    info_a = _rollout_info(hist_len=3, emb=emb)
    info_a['action_history'] = torch.zeros(RB, RS, 2, RD)
    out_a = model.rollout(info_a, candidates)['predicted_emb']

    info_b = _rollout_info(hist_len=3, emb=emb)
    info_b['action_history'] = torch.ones(RB, RS, 2, RD)
    out_b = model.rollout(info_b, candidates)['predicted_emb']

    assert out_a.shape == (RB, RS, 3 + 4, RD)
    torch.testing.assert_close(out_a[:, :, :3], emb)  # context passthrough
    assert not torch.allclose(out_a[:, :, 3:], out_b[:, :, 3:])


def test_rollout_window_arithmetic():
    """First prediction sees the full H-frame context with the past blocks
    plus the first candidate; the window then slides, capped at HS."""
    torch.manual_seed(0)
    model = _toy_model(num_frames=3)
    info = _rollout_info(hist_len=3)
    past = torch.randn(RB, RS, 2, RD)
    candidates = torch.randn(RB, RS, 4, RD)
    info['action_history'] = past

    model.rollout(info, candidates)

    calls = model.predictor.calls
    assert len(calls) == 4  # one prediction per candidate
    emb0, act0 = calls[0]
    assert emb0.shape[1] == 3 and act0.shape[1] == 3
    expected_first = torch.cat([past, candidates[:, :, :1]], dim=2).reshape(
        RB * RS, 3, RD
    )
    torch.testing.assert_close(act0, expected_first)
    for emb_t, act_t in calls[1:]:  # window capped at HS = 3
        assert emb_t.shape[1] == 3 and act_t.shape[1] == 3


def test_rollout_rejects_mismatched_action_history():
    model = _toy_model()
    info = _rollout_info(hist_len=3)
    info['action_history'] = torch.randn(RB, RS, 1, RD)  # needs H-1 = 2
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RD))


def test_rollout_rejects_multiframe_pixels_without_action_history():
    """H > 1 pixels without executed past actions must fail loudly rather
    than silently pairing context frames with optimizer candidates."""
    model = _toy_model()
    info = _rollout_info(hist_len=3)
    with pytest.raises(AssertionError, match='action_history'):
        model.rollout(info, torch.randn(RB, RS, 4, RD))


#################################################
## Goal policy head (the GCIDM addition) ##
#################################################

# Plan dims for the head tests: PH blocks of PA flattened action dims.
PH, PA, PC = 5, 6, 3


def _head(embed_dim=RD, action_dim=PA, predict_std=True, hidden_dim=8):
    return GoalPolicyHead(
        embed_dim=embed_dim,
        action_dim=action_dim,
        horizon=PH,
        context_frames=PC,
        hidden_dim=hidden_dim,
        num_layers=3,
        predict_std=predict_std,
    )


def _plan_model(**kwargs):
    """GCIDM with a real policy head; encode/rollout parts unused here."""
    return GCIDM(
        encoder=nn.Identity(),
        predictor=_CumsumPredictor(),
        action_encoder=_RecordingIdentity(),
        policy_head=_head(**kwargs),
    )


def test_predict_plan_returns_mean_and_positive_std():
    """(z_ctx, z_goal) -> a plan of `horizon` blocks, plus a positive std."""
    model = _plan_model()
    mu, std = model.predict_plan(torch.randn(RB, PC, RD), torch.randn(RB, RD))

    assert mu.shape == (RB, PH, PA)
    assert std.shape == (RB, PH, PA)
    assert bool((std > 0).all()), 'std must be positive (it is exp(log_std))'


def test_predict_plan_without_std_head_returns_none():
    model = _plan_model(predict_std=False)
    mu, std = model.predict_plan(torch.randn(RB, PC, RD), torch.randn(RB, RD))
    assert mu.shape == (RB, PH, PA)
    assert std is None


def test_log_std_is_clamped():
    """An extreme head output cannot produce a degenerate or exploding std."""
    head = _head()
    with torch.no_grad():  # force the output layer to saturate
        head.net[-1].weight.fill_(0.0)
        head.net[-1].bias.fill_(1e4)
    _, std = head(torch.randn(RB, PC, RD), torch.randn(RB, RD))
    torch.testing.assert_close(
        std, torch.full_like(std, float(torch.tensor(head.log_std_max).exp()))
    )


def test_predict_plan_is_sensitive_to_context_and_goal():
    """The head reads both ends: perturbing either changes the plan."""
    torch.manual_seed(0)
    model = _plan_model()
    z_ctx, z_goal = torch.randn(RB, PC, RD), torch.randn(RB, RD)

    base, _ = model.predict_plan(z_ctx, z_goal)
    assert not torch.allclose(base, model.predict_plan(z_ctx, z_goal + 1.0)[0])
    assert not torch.allclose(base, model.predict_plan(z_ctx + 1.0, z_goal)[0])


def test_predict_plan_uses_every_context_frame():
    """The head gets the *predictor's* context, not just the current frame,
    so perturbing any one of the C latents must change the plan."""
    torch.manual_seed(0)
    model = _plan_model()
    z_ctx, z_goal = torch.randn(RB, PC, RD), torch.randn(RB, RD)
    base, _ = model.predict_plan(z_ctx, z_goal)

    for frame in range(PC):
        bumped = z_ctx.clone()
        bumped[:, frame] += 1.0
        assert not torch.allclose(
            base, model.predict_plan(bumped, z_goal)[0]
        ), f'context frame {frame} does not affect the plan'


def test_head_rejects_wrong_context_length():
    """A caller must pad/trim to context_frames rather than silently
    reshaping a different number of latents into the input."""
    model = _plan_model()
    with pytest.raises(ValueError, match='context latents'):
        model.predict_plan(torch.randn(RB, PC + 1, RD), torch.randn(RB, RD))


def test_predict_plan_without_policy_head_raises():
    """A model built without a head fails loudly instead of returning junk."""
    model = _toy_model()
    assert model.policy_head is None
    with pytest.raises(AssertionError, match='No policy head configured'):
        model.predict_plan(torch.randn(RB, PC, RD), torch.randn(RB, RD))


def test_policy_head_is_part_of_the_model_state_dict():
    """The head lives inside GCIDM, so save_pretrained (which serializes the
    model state dict) cannot silently drop it."""
    keys = _plan_model().state_dict().keys()
    assert any(k.startswith('policy_head.') for k in keys)


def test_policy_head_is_smaller_than_the_predictor():
    """The shipped head must stay under the predictor's parameter budget.

    Guards the config: `policy.hidden_dim` / `policy.num_layers` are tunable,
    and this is the constraint they must not silently break. Uses the shipped
    gcidm.yaml numbers, not the toy test dims.
    """
    head = GoalPolicyHead(
        embed_dim=384,
        action_dim=25,
        horizon=5,
        context_frames=3,
        hidden_dim=768,
        num_layers=3,
    )
    predictor = Predictor(
        num_frames=3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=384,
        hidden_dim=384,
        output_dim=384,
        dim_head=64,
    )
    head_params = sum(p.numel() for p in head.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    assert head_params < pred_params, (
        f'policy head ({head_params:,}) must stay under the predictor '
        f'({pred_params:,})'
    )


##########################################
## Planning seam: the solver entry point ##
##########################################


class _MeanEncoder(nn.Module):
    """Stand-in encoder: reduces each frame to a deterministic RD vector."""

    def __init__(self, dim=RD):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(1, dim)

    def forward(self, pixels, **kwargs):
        # pixels: (B*T, C, h, w) -> one scalar per frame, lifted to RD
        flat = pixels.flatten(1).mean(dim=1, keepdim=True)
        out = self.proj(flat)

        class _Out:
            pass

        o = _Out()
        # encode() reads last_hidden_state[:, 0], so add a token axis
        o.last_hidden_state = out[:, None, :].expand(-1, 2, -1)
        return o


def _planning_model():
    model = _plan_model()
    model.encoder = _MeanEncoder()
    return model


def _planning_info(n_frames=PC):
    """The *unexpanded* info_dict a solver holds, per WorldModelPolicy."""
    return {
        'pixels': torch.randn(RB, n_frames, 3, 8, 8),
        'goal': torch.randn(RB, 1, 3, 8, 8),
    }


def test_gcidm_is_actionable_only_with_a_policy_head():
    """`Actionable` is an attribute-presence check that `prepare_init_action`
    and `ShootingCostEvaluator` both branch on, so it has to answer honestly:
    a head-less GCIDM must not claim to be an actor and then raise mid-solve.
    """
    assert isinstance(_planning_model(), Actionable)
    assert not isinstance(_toy_model(), Actionable)


def test_get_action_distribution_shapes():
    model = _planning_model()
    mu, std = model.get_action_distribution(_planning_info(), horizon=PH)
    assert mu.shape == (RB, PH, PA)
    assert std.shape == (RB, PH, PA)


def test_get_action_distribution_pads_a_short_context():
    """Early in an episode the history buffer holds fewer than history_len
    frames; the oldest frame is repeated rather than the call failing."""
    model = _planning_model()
    for n_frames in (1, 2, PC):
        mu, _ = model.get_action_distribution(_planning_info(n_frames))
        assert mu.shape == (RB, PH, PA), f'failed at {n_frames} frames'


def test_get_action_distribution_uses_the_latest_frames():
    """With more frames than the head wants, the *newest* C are used."""
    torch.manual_seed(0)
    model = _planning_model()
    info = _planning_info(n_frames=PC + 2)

    base, _ = model.get_action_distribution(info)

    stale = {k: v.clone() for k, v in info.items()}
    stale['pixels'][:, 0] += 5.0  # a dropped frame must not matter
    torch.testing.assert_close(base, model.get_action_distribution(stale)[0])

    fresh = {k: v.clone() for k, v in info.items()}
    fresh['pixels'][:, -1] += 5.0  # the current frame must matter
    assert not torch.allclose(base, model.get_action_distribution(fresh)[0])


def test_get_action_distribution_reads_the_goal():
    """The goal frame conditions the plan — that is the whole point."""
    torch.manual_seed(0)
    model = _planning_model()
    info = _planning_info()
    base, _ = model.get_action_distribution(info)

    other = {k: v.clone() for k, v in info.items()}
    other['goal'] += 5.0
    assert not torch.allclose(base, model.get_action_distribution(other)[0])


def test_get_action_distribution_accepts_an_unsqueezed_goal():
    """`goal` arrives as (B, 1, C, h, w) from World, but a bare (B, C, h, w)
    is accepted too so the seam is not shape-brittle."""
    model = _planning_model()
    info = _planning_info()
    squeezed = {**info, 'goal': info['goal'][:, 0]}
    torch.testing.assert_close(
        model.get_action_distribution(info)[0],
        model.get_action_distribution(squeezed)[0],
    )


def test_get_action_distribution_rejects_a_horizon_mismatch():
    """A plan_config.horizon that disagrees with the trained goal_horizon is
    a config error, not something to silently pad or truncate."""
    model = _planning_model()
    with pytest.raises(ValueError, match='goal_horizon'):
        model.get_action_distribution(_planning_info(), horizon=PH + 1)


def test_get_action_returns_the_plan_tail():
    """Actionable's contract: `horizon` blocks, taken from the plan's end so
    they line up with the prefix a warm-started solver already holds."""
    model = _planning_model()
    info = _planning_info()
    mu, _ = model.get_action_distribution(info)

    torch.testing.assert_close(model.get_action(info, horizon=PH), mu)
    torch.testing.assert_close(model.get_action(info, horizon=2), mu[:, -2:])
    assert model.get_action(info).shape == (RB, 1, PA)

    with pytest.raises(ValueError, match='cannot fill'):
        model.get_action(info, horizon=PH + 1)


def test_cost_evaluator_forwards_get_action_for_actionable_models():
    """ShootingCostEvaluator delegates get_action when the model has one, so
    the generic prepare_init_action tail-fill reaches the policy head."""
    model = _planning_model()
    cost = ShootingCostEvaluator(model, GoalMSE())
    assert isinstance(cost, Actionable)

    info = _planning_info()
    torch.testing.assert_close(
        cost.get_action(info, horizon=2), model.get_action(info, horizon=2)
    )


def test_cost_evaluator_stays_non_actionable_without_a_head():
    """Behaviour for LeWM/SMWM-style models is unchanged: no actor, so a
    solver zero-pads as before."""
    cost = ShootingCostEvaluator(_toy_model(), GoalMSE())
    assert not hasattr(cost, 'get_action')


###############################
## Checkpoint round trip ##
###############################


def _tiny_hydra_config(embed_dim=RD, action_dim=PA):
    """A hydra-instantiable GCIDM config small enough for a unit test."""
    return {
        '_target_': 'stable_worldmodel.wm.gcidm.gcidm.GCIDM',
        'encoder': {'_target_': 'torch.nn.Identity'},
        'predictor': {
            '_target_': 'stable_worldmodel.wm.gcidm.module.Predictor',
            'num_frames': PC,
            'input_dim': embed_dim,
            'hidden_dim': embed_dim,
            'output_dim': embed_dim,
            'depth': 1,
            'heads': 2,
            'mlp_dim': 8,
            'dim_head': 2,
            'dropout': 0.0,
            'emb_dropout': 0.0,
        },
        'action_encoder': {
            '_target_': 'stable_worldmodel.wm.gcidm.module.Embedder',
            'input_dim': action_dim,
            'emb_dim': embed_dim,
        },
        'policy_head': {
            '_target_': 'stable_worldmodel.wm.gcidm.module.GoalPolicyHead',
            'embed_dim': embed_dim,
            'action_dim': action_dim,
            'horizon': PH,
            'context_frames': PC,
            'hidden_dim': 8,
            'num_layers': 3,
            'predict_std': True,
        },
    }


def test_checkpoint_round_trip_preserves_policy_head(tmp_path):
    """save_pretrained -> load_pretrained rebuilds the head with its weights,
    so a trained GCIDM can warm-start CEM with no eval-side flags."""
    torch.manual_seed(0)
    config = _tiny_hydra_config()
    model = hydra.utils.instantiate(config)

    save_pretrained(
        model, run_name='gcidm_test', config=config, cache_dir=tmp_path
    )
    loaded = load_pretrained('gcidm_test/weights.pt', cache_dir=tmp_path)

    assert loaded.policy_head is not None
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert any(k.startswith('policy_head.') for k in restored)
    for key, value in original.items():
        torch.testing.assert_close(restored[key], value)

    z_ctx, z_goal = torch.randn(RB, PC, RD), torch.randn(RB, RD)
    mu_a, std_a = model.predict_plan(z_ctx, z_goal)
    mu_b, std_b = loaded.predict_plan(z_ctx, z_goal)
    torch.testing.assert_close(mu_b, mu_a)
    torch.testing.assert_close(std_b, std_a)

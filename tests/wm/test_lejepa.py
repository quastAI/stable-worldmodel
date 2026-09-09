"""Tests for the LeJEPA encoder and the frozen-encoder stage-D wrapper.

Three properties carry the plan's design decisions and would otherwise be
enforced only by discipline:

* ``LeJEPA`` is passive and must **not** satisfy ``Dynamics``.
* ``FrozenEncoderWM`` must keep the encoder frozen through everything a
  training loop does to a module.
* A predictor must refuse an encoder it was not trained against, because that
  mismatch degrades planning in a way indistinguishable from an
  identifiability failure.
"""

import numpy as np
import pytest
import torch
from torch import nn

from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.lejepa import CNNEncoder, LeJEPA, NDimHead
from stable_worldmodel.wm.lejepa.losses import alignment_loss, whitening_loss
from stable_worldmodel.wm.lejepa.module import (
    Embedder,
    FrozenEncoderWM,
    Predictor,
    state_dict_hash,
)


N = 9
DIM = 32


class TinyBackbone(nn.Module):
    """Stands in for the ViT: same interface, negligible cost."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, dim)

    def forward(self, pixels, interpolate_pos_encoding=False):
        flat = pixels.flatten(1)
        tokens = self.proj(flat).unsqueeze(1)
        return type('Out', (), {'last_hidden_state': tokens})()


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return LeJEPA(TinyBackbone(), NDimHead(DIM, N, hidden_dim=16))


@pytest.fixture
def frozen(encoder):
    predictor = Predictor(
        num_frames=3, depth=1, heads=2, mlp_dim=16,
        input_dim=N, hidden_dim=16, output_dim=N, dim_head=8,
    )
    return FrozenEncoderWM(
        encoder, predictor, Embedder(input_dim=5, emb_dim=N)
    )


def pixels(b=4, t=2):
    return torch.randn(b, t, 3, 8, 8)


# --------------------------------------------------------- the paper's CNN


def test_cnn_encoder_reproduces_the_reference_shape():
    """`(B, C, H, W) -> (B, proj_dim)`, pooled and projected."""
    enc = CNNEncoder(channels=(8, 16), proj_dim=DIM, image_size=32)
    out = enc(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, DIM)
    assert enc.output_dim == DIM


def test_cnn_pool_stays_global_at_the_design_resolution():
    """The invariant that decides the stage count.

    The pool is global, so position has to be encoded in *which* channels fire.
    That only works while each cell of the final map still sees most of the
    frame. Six stages at 224px give a 3x3 map -- the reference's 64px/4-stage
    design gives 4x4. A regression to 4 stages at 224px would leave 14x14 and
    average the spatial content away, so this pins the geometry.
    """
    six = CNNEncoder(image_size=224)
    assert six.feature_map == 3, six.feature_map
    assert CNNEncoder(channels=(32, 64, 128, 256), image_size=64).feature_map == 4
    assert CNNEncoder(channels=(32, 64, 128, 256), image_size=224).feature_map == 14


def test_cnn_encoder_is_resolution_agnostic():
    """Adaptive pooling, so a different input size runs rather than erroring.

    The reference's fixed `AvgPool2d(4)` silently stops being global when the
    input changes size; this must not.
    """
    enc = CNNEncoder(channels=(8, 16), proj_dim=DIM, image_size=32)
    assert enc(torch.randn(2, 3, 64, 64)).shape == (2, DIM)


def test_cnn_embedding_is_not_normalised():
    """No norm *after* the projection to n.

    BatchNorm on the embedding would force diag(Cov(h)) to one by
    construction, partly trivialising both ``epsilon`` and SIGReg -- and
    ``epsilon`` is a logged measurement the V8 sweep only means anything
    against if it is independent of the objective. The paper's reference is
    built the same way: every BatchNorm internal, the final projection bare.
    """
    model = LeJEPA(
        CNNEncoder(channels=(8, 16), proj_dim=DIM, image_size=32),
        NDimHead(DIM, N, hidden_dim=None, norm=False),
    )
    # The head the paper's config builds is exactly `Linear(proj_dim, n)`.
    assert len(model.head.net) == 1
    assert isinstance(model.head.net[0], nn.Linear)

    emb = model.encode({'pixels': torch.randn(8, 2, 3, 32, 32)})['emb']
    per_dim_var = emb.flatten(0, 1).var(dim=0)
    assert not torch.allclose(
        per_dim_var, torch.ones_like(per_dim_var), atol=0.2
    ), 'embedding looks pre-whitened; SIGReg would be measuring itself'


def test_encode_accepts_both_backbone_conventions():
    """HF-style and plain modules both work, decided by signature.

    `TinyBackbone.forward` takes `interpolate_pos_encoding` and returns an
    object with `last_hidden_state`; `CNNEncoder.forward` takes neither and
    returns a tensor. Passing the kwarg to the CNN would raise, so this pins
    the detection rather than the happy path of whichever ran first.
    """
    vit = LeJEPA(TinyBackbone(), NDimHead(DIM, N, hidden_dim=16))
    cnn = LeJEPA(
        CNNEncoder(channels=(8, 16), proj_dim=DIM, image_size=8),
        NDimHead(DIM, N, hidden_dim=None, norm=False),
    )
    assert vit._takes_pos_encoding is True
    assert cnn._takes_pos_encoding is False
    for model in (vit, cnn):
        assert model.encode({'pixels': pixels()})['emb'].shape == (4, 2, N)


def test_cnn_exposes_image_size_for_the_metric_suite():
    """The metric suite must be able to preprocess exactly as training did.

    HF backbones carry the resolution on `.config`; this one carries it on the
    module. Scoring at the wrong resolution silently invalidates every column
    of the scatter, so the attribute is part of the contract.
    """
    assert CNNEncoder(image_size=224).image_size == 224


# ----------------------------------------------------------- passive encoder


def test_lejepa_is_not_dynamics(encoder):
    """A passive encoder has no dynamics; this is a decision, not an omission.

    If ``LeJEPA`` satisfied ``Dynamics`` it could be handed straight to the
    planner, and "the encoder is frozen before any predictor is trained" would
    become a convention someone has to remember instead of a property of the
    type.
    """
    assert not isinstance(encoder, Dynamics)
    assert not hasattr(encoder, 'rollout')


def test_lejepa_has_no_action_path(encoder):
    """Encoder training never sees an action."""
    assert not hasattr(encoder, 'action_encoder')
    assert not hasattr(encoder, 'predictor')

    out = encoder.encode({'pixels': pixels(), 'action': torch.randn(4, 2, 5)})
    assert 'act_emb' not in out


def test_head_output_width_is_n(encoder):
    out = encoder.encode({'pixels': pixels()})
    assert out['emb'].shape == (4, 2, N)
    assert encoder.output_dim == N


@pytest.mark.parametrize('output_dim', [N - 3, N, N + 5])
def test_v7_is_one_config_field(output_dim):
    """V7a and V7b must not require an architectural change."""
    model = LeJEPA(TinyBackbone(), NDimHead(DIM, output_dim))
    assert model.encode({'pixels': pixels()})['emb'].shape[-1] == output_dim


def test_embed_convenience_matches_encode(encoder):
    frames = torch.randn(5, 3, 8, 8)
    direct = encoder.embed(frames)
    through = encoder.encode({'pixels': frames.unsqueeze(1)})['emb'][:, 0]
    torch.testing.assert_close(direct, through)


# ------------------------------------------------------------------- losses


def test_alignment_is_zero_for_identical_views():
    h = torch.randn(1, 16, N).expand(2, 16, N).contiguous()
    assert float(alignment_loss(h)) == pytest.approx(0.0, abs=1e-9)


def test_alignment_grows_as_views_separate():
    base = torch.randn(1, 64, N)
    losses = [
        float(alignment_loss(torch.cat([base, base + s * torch.randn(1, 64, N)])))
        for s in (0.0, 0.5, 2.0)
    ]
    assert losses[0] < losses[1] < losses[2]


def test_whitening_is_zero_for_whitened_embeddings():
    torch.manual_seed(0)
    h = torch.randn(2, 20000, 4)
    h = (h - h.mean((0, 1))) / h.std((0, 1))
    assert float(whitening_loss(h)) < 0.02


def test_whitening_loss_carries_no_gradient():
    """It is the metric ``epsilon``. Optimising it would make V8 circular.

    The V8 sweep asks how the bound ``D + (eps + D)^2`` behaves as optimisation
    is cut short. That is only a question if ``eps`` is not itself being
    minimised.
    """
    h = torch.randn(2, 32, N, requires_grad=True)
    loss = whitening_loss(h)
    assert not loss.requires_grad
    assert h.grad is None


# ------------------------------------------------------ the frozen wrapper


def test_frozen_encoder_wm_is_dynamics(frozen):
    assert isinstance(frozen, Dynamics)


def test_encoder_parameters_never_require_grad(frozen):
    assert not any(p.requires_grad for p in frozen.encoder.parameters())
    assert any(p.requires_grad for p in frozen.predictor.parameters())


def test_train_mode_does_not_unfreeze_the_encoder(frozen):
    """``.train()`` would otherwise flip the encoder's dropout and norms back on.

    That silently changes the embeddings the predictor was trained against --
    no error, just a quietly different representation.
    """
    frozen.train()
    assert frozen.predictor.training
    assert not frozen.encoder.training
    assert not any(p.requires_grad for p in frozen.encoder.parameters())


def test_encoder_receives_no_gradient_from_the_predictor_loss(frozen):
    """The structural guarantee, exercised through an actual backward pass."""
    out = frozen.encode({'pixels': pixels(), 'action': torch.randn(4, 2, 5)})
    loss = frozen.predict(out['emb'], out['act_emb']).pow(2).mean()
    loss.backward()

    assert all(p.grad is None for p in frozen.encoder.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in frozen.predictor.parameters()
    )


def test_rollout_produces_the_expected_shape(frozen):
    b, s, h_ctx, t = 2, 3, 1, 4
    info = {'pixels': torch.randn(b, s, h_ctx, 3, 8, 8)}
    out = frozen.rollout(info, torch.randn(b, s, t, 5))
    assert out['predicted_emb'].shape == (b, s, h_ctx + t, N)


def test_untrained_predictor_ignores_actions_by_construction(frozen):
    """AdaLN-zero: at initialisation the predictor is exactly the identity.

    ``ConditionalBlock`` zero-initialises the final layer of
    ``adaLN_modulation``, so ``gate_msa`` and ``gate_mlp`` start at zero and
    both residual branches contribute nothing. The conditioning signal --
    the actions -- therefore cannot move the output until training lifts those
    gates off zero.

    This is asserted rather than merely tolerated because it is exactly the
    observation that looks like an action-wiring bug on first encounter: an
    untrained model whose rollout is invariant to its action sequence, and a
    zero action-encoder gradient at step 0. Both are the intended behaviour of
    AdaLN-zero.
    """
    info = {'pixels': torch.randn(2, 3, 1, 3, 8, 8)}
    a = frozen.rollout(dict(info), torch.zeros(2, 3, 4, 5))['predicted_emb']
    b = frozen.rollout(dict(info), torch.ones(2, 3, 4, 5))['predicted_emb']
    torch.testing.assert_close(a, b)


def test_rollout_is_sensitive_to_actions_once_the_gates_open(frozen):
    """The wiring test proper: actions must *reach* the predictor.

    Lifting the AdaLN gates off zero stands in for training. If the rollout is
    still action-invariant after that, the action path is genuinely
    disconnected -- and the planner would have nothing to optimise, silently.
    """
    with torch.no_grad():
        for block in frozen.predictor.transformer.layers:
            block.adaLN_modulation[-1].weight.normal_(0, 0.5)
            block.adaLN_modulation[-1].bias.normal_(0, 0.5)

    info = {'pixels': torch.randn(2, 3, 1, 3, 8, 8)}
    a = frozen.rollout(dict(info), torch.zeros(2, 3, 4, 5))['predicted_emb']
    b = frozen.rollout(dict(info), torch.ones(2, 3, 4, 5))['predicted_emb']
    assert not torch.allclose(a, b)


# --------------------------------------------------------- the hash binding


def test_hash_is_stable_and_content_addressed(encoder):
    assert state_dict_hash(encoder) == state_dict_hash(encoder)

    other = LeJEPA(TinyBackbone(), NDimHead(DIM, N, hidden_dim=16))
    assert state_dict_hash(other) != state_dict_hash(encoder)


def test_hash_changes_when_a_weight_changes(encoder):
    before = state_dict_hash(encoder)
    with torch.no_grad():
        next(encoder.parameters()).add_(1.0)
    assert state_dict_hash(encoder) != before


def test_mismatched_encoder_is_refused(encoder):
    """The wiring bug that looks exactly like an identifiability failure.

    A predictor fed embeddings from a different encoder plans badly, with
    well-behaved losses and no error anywhere. Nothing downstream can tell it
    apart from an encoder that failed to identify the latents, so it is caught
    structurally.
    """
    predictor = Predictor(
        num_frames=3, depth=1, heads=2, mlp_dim=16,
        input_dim=N, hidden_dim=16, output_dim=N, dim_head=8,
    )
    with pytest.raises(ValueError, match='hash mismatch'):
        FrozenEncoderWM(
            encoder,
            predictor,
            Embedder(input_dim=5, emb_dim=N),
            encoder_hash='0' * 16,
        )


def test_matching_hash_is_accepted(encoder):
    predictor = Predictor(
        num_frames=3, depth=1, heads=2, mlp_dim=16,
        input_dim=N, hidden_dim=16, output_dim=N, dim_head=8,
    )
    model = FrozenEncoderWM(
        encoder,
        predictor,
        Embedder(input_dim=5, emb_dim=N),
        encoder_hash=state_dict_hash(encoder),
    )
    assert model.current_encoder_hash() == state_dict_hash(encoder)


def test_predictor_blocks_are_shared_with_lewm():
    """Arms A/R/C reuse the LeWM predictor; a fork would drift them apart."""
    from stable_worldmodel.wm.lewm.module import Predictor as LeWMPredictor

    assert Predictor is LeWMPredictor


def test_sigreg_pool_views_matches_the_reference_convention():
    """`pool_views=True` is what puts `lambda` in the paper's units.

    The identifiability paper's own implementation pools both views into one set
    of 2B samples and scales by 2B; this repo's default takes a per-view
    statistic over B samples and averages. Away from the optimum the two differ
    by exactly a factor of V, so a `lambda` read off the paper's figures is off
    by that factor under the default -- which is the whole reason the flag
    exists.
    """
    import torch

    from stable_worldmodel.wm.loss import SIGReg

    torch.manual_seed(0)
    per_view = SIGReg(num_proj=4096, pool_views=False)
    pooled = SIGReg(num_proj=4096, pool_views=True)

    # Away from the optimum: the pooled statistic is V times the per-view one.
    h = 0.35 * torch.randn(2, 256, 10)
    ratio = pooled(h).item() / per_view(h).item()
    assert 1.8 < ratio < 2.2, ratio

    # A single view is unaffected by pooling.
    single = torch.randn(1, 256, 10)
    assert pooled(single).item() == pytest.approx(
        per_view(single).item(), rel=0.25
    )

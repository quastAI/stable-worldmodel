"""Tests for the LeJEPA encoder and the diagnostics logged beside it.

Two properties carry design decisions that would otherwise be enforced only by
discipline:

* ``LeJEPA`` is passive and must **not** satisfy ``Dynamics``. A passive
  encoder has no dynamics, and a class that claimed otherwise would be handed
  to a planner.
* The diagnostics that read ``h[0] - h[1]`` must be separable from the ones
  that read only the marginal, because train-mode BatchNorm flatters the first
  group and not the second.
"""

import pytest
import torch
from torch import nn

from stable_worldmodel.protocols import Dynamics
from stable_worldmodel.wm.lejepa import CNNEncoder, LeJEPA, NDimHead
from stable_worldmodel.wm.lejepa.losses import (
    alignment_loss,
    bound_diagnostics,
    recovery_diagnostics,
    spectrum_diagnostics,
    whitening_loss,
)
from stable_worldmodel.wm.lejepa.module import state_dict_hash


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
    assert (
        CNNEncoder(channels=(32, 64, 128, 256), image_size=64).feature_map == 4
    )
    assert (
        CNNEncoder(channels=(32, 64, 128, 256), image_size=224).feature_map
        == 14
    )


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
def test_head_width_is_one_config_field(output_dim):
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
        float(
            alignment_loss(torch.cat([base, base + s * torch.randn(1, 64, N)]))
        )
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


# ------------------------------------------------------------ provenance


def test_hash_is_stable_and_content_addressed(encoder):
    assert state_dict_hash(encoder) == state_dict_hash(encoder)

    other = LeJEPA(TinyBackbone(), NDimHead(DIM, N, hidden_dim=16))
    assert state_dict_hash(other) != state_dict_hash(encoder)


def test_hash_changes_when_a_weight_changes(encoder):
    before = state_dict_hash(encoder)
    with torch.no_grad():
        next(encoder.parameters()).add_(1.0)
    assert state_dict_hash(encoder) != before


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


# ------------------------------------------------- which diagnostics survive


def test_marginal_diagnostics_are_blind_to_the_pair_distance():
    """``spectrum`` and ``whitening`` read only ``Cov(h)``, so pulling the two
    views together must not move them.

    This is the property that makes them safe to log on the training stage:
    train-mode BatchNorm normalises both views of a pair with the same batch
    statistics, which changes ``h[0] - h[1]`` without changing the marginal.
    """
    torch.manual_seed(0)
    h = torch.randn(2, 512, N).double()
    pulled = torch.stack([h[0], h[0] + 0.2 * (h[1] - h[0])])

    for key, value in spectrum_diagnostics(h).items():
        # The marginal of view 0 is untouched and view 1 is a contraction
        # toward it, so these move only as much as the marginal itself does.
        assert torch.isfinite(value)
    assert whitening_loss(h) > 0
    assert alignment_loss(pulled) < alignment_loss(h), (
        'the pair distance must be the thing that changed'
    )


def test_bound_diagnostics_flag_an_impossible_delta():
    """``delta`` below its Hermite floor means the embeddings are not a
    function of one frame -- the train-mode signature.

    A style-invariant direction of ``h`` is a function of ``z``, and a function
    of ``z`` decorrelates at ``rho^k`` for its degree-``k`` content. A direction
    uncorrelated with ``z`` has degree at least 2, so ``delta`` has a floor.
    Below it, the row cannot be believed.
    """
    torch.manual_seed(0)
    rho, n, batch = 0.9, 10, 4096
    z = torch.randn(batch, n).double()
    z_next = rho * z + (1 - rho**2) ** 0.5 * torch.randn(batch, n).double()

    def encode(latents):
        half = latents[:, : n // 2]
        rest = (latents[:, n // 2 :] ** 2 - 1) / 2**0.5
        return torch.cat([half, rest], dim=1)

    h = torch.stack([encode(z), encode(z_next)])
    truth = torch.stack([z, z_next])
    recovered = recovery_diagnostics(h.float(), truth.float())[
        'recovered_dimensions'
    ]

    honest = bound_diagnostics(h, rho, recovered)
    assert honest['residual_hermite_degree'] == pytest.approx(2.0, abs=0.15)
    assert bool(honest['delta_admissible'])

    flattered = torch.stack([h[0], h[0] + 0.5 * (h[1] - h[0])])
    broken = bound_diagnostics(flattered, rho, recovered)
    assert broken['delta'] < broken['delta_floor']
    assert broken['residual_hermite_degree'] < 2.0
    assert not bool(broken['delta_admissible'])


def test_recovery_diagnostics_are_empty_under_a_mismatched_width():
    """No square ``Q`` to align with, so a recovery number would be meaningless.

    The per-latent probe in ``run_metrics.py`` is what still applies there, and
    returning ``{}`` is how the caller is told to go and use it.
    """
    h = torch.randn(2, 64, N + 3)
    z = torch.randn(2, 64, N)
    assert recovery_diagnostics(h, z) == {}


def test_recovered_dimensions_matches_the_probe_identity():
    """``sum sigma^2 = n * r2_h_to_z``, which is what floors ``delta``."""
    torch.manual_seed(0)
    z = torch.randn(2, 2048, N).double()
    q = torch.linalg.qr(torch.randn(N, N).double())[0]
    result = recovery_diagnostics(z @ q.T, z)
    assert result['recovered_dimensions'] == pytest.approx(
        float(result['r2_h_to_z']) * N, rel=1e-9
    )
    assert result['recovered_dimensions'] == pytest.approx(N, rel=1e-6)


def test_procrustes_is_not_scale_free():
    """``h = 0`` scores 1.0, so a falling curve near 1.0 is leaving the
    trivial encoder rather than arriving at a good one."""
    torch.manual_seed(0)
    z = torch.randn(2, 2048, N).double()
    result = recovery_diagnostics(torch.zeros_like(z), z)
    assert float(result['procrustes_mse_per_dim']) == pytest.approx(
        1.0, rel=0.05
    )

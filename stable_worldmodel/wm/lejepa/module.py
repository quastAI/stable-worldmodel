"""Building blocks for the LeJEPA encoder: the pixel CNN and the n-dim head.

Two classes, and nothing else. The transformer ``Predictor`` this module used
to re-export from ``wm/lewm`` is gone along with the predictor stage, which is
what makes the LeJEPA encoder path independent of every sibling model in the
repo: nothing under ``wm/lejepa`` imports from another ``wm/*`` package.
"""

import hashlib

import torch
from torch import nn


def state_dict_hash(module: nn.Module) -> str:
    """Content hash of a module's parameters, for provenance.

    ``seed`` does not make a trained encoder reproducible -- nondeterministic
    kernels, dataloader order and library versions all move the weights -- so
    the only durable identifier for a checkpoint is its own contents. The
    training run writes this next to the weights and every results row records
    it, which is what lets a number be traced back to exact parameters after
    the ``.pt`` has been moved, renamed or copied between machines.

    Args:
        module: Module whose parameters to hash.

    Returns:
        str: 16-character hex digest.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().to(torch.float32).numpy().tobytes()
        )
    return digest.hexdigest()[:16]


class CNNEncoder(nn.Module):
    """The LeJEPA paper's pixel encoder, scaled to a larger input.

    Faithful to the reference ``lejepa_id/models.py::make_cnn_encoder`` used
    for the paper's pixel experiment (App. H.11) -- a stack of ``k=4, s=2,
    p=1`` convolutions with BatchNorm and GELU, a **global** average pool, then
    one projection with BatchNorm before the head. Chosen over the LeWM ViT
    because every constant this program freezes (``lambda``, ``rho``, the
    learning rate) was calibrated on it, and because it removes the encoder's
    expressivity as a confound: a from-scratch ViT's failures cannot be told
    apart from optimisation or data-volume failures.

    **What is preserved when scaling from the reference's 64px to 224px.** The
    pool is global, so position has to be encoded in *which* channels fire,
    not in where they fire -- and that only works while each cell of the final
    map still sees most of the image. The reference gets a 4x4 map with a 46px
    receptive field at 64px input, i.e. ~72% of the frame. Four stages at 224px
    would leave a 14x14 map at ~21%, and averaging 196 near-local descriptors
    would wash out exactly the spatial content (cube and effector position)
    that ``z`` is made of. Six stages restore the invariant: a 3x3 map at a
    190px receptive field, ~85% of the frame.

    **What is deliberately *not* normalised**: the head's output. BatchNorm on
    the embedding would force the diagonal of ``Cov(h)`` to one by
    construction, which partly trivialises both ``epsilon`` and SIGReg -- and
    ``epsilon`` is a logged measurement that the V8 sweep only means anything
    against if it is independent of the objective. The reference is built the
    same way: every BatchNorm is internal and the final projection is bare.

    Args:
        channels: Output width of each stride-2 stage. The default
            ``(32, 64, 128, 256, 256, 256)`` is the reference's
            ``32, 64, 128, 256`` progression extended by two stages that hold
            at 256, which is what brings a 224px input back to a small map.
        in_channels: Input channels; 3 for RGB.
        proj_dim: Width of the projection after the pool. 256 as in the
            reference, and the ``input_dim`` the head should be given.
        image_size: Only used to record :attr:`feature_map` for diagnosis; the
            pool is adaptive, so the module works at any resolution.
    """

    def __init__(
        self,
        channels=(32, 64, 128, 256, 256, 256),
        in_channels: int = 3,
        proj_dim: int = 256,
        image_size: int = 224,
    ):
        super().__init__()
        self.output_dim = proj_dim
        self.channels = tuple(channels)

        layers = []
        prev = in_channels
        for width in self.channels:
            layers += [
                nn.Conv2d(prev, width, 4, 2, 1),
                nn.BatchNorm2d(width),
                nn.GELU(),
            ]
            prev = width
        # Adaptive rather than the reference's `AvgPool2d(4)`: both are a global
        # mean at the design resolution, but the fixed kernel silently stops
        # being global the moment the input size changes.
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        layers += [
            nn.Linear(prev, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
        ]
        self.net = nn.Sequential(*layers)

        self.image_size = image_size
        """Input resolution this encoder is built for. The pool is adaptive so
        other sizes run, but the metric suite reads this to preprocess exactly
        as training did -- mirroring an HF backbone's ``config.image_size``."""

        size = image_size
        for _ in self.channels:
            size = (size + 2 - 4) // 2 + 1
        self.feature_map = size
        """Spatial extent of the last conv map at ``image_size``, for the
        record: the global pool discards whatever spatial detail survives here,
        so a large value means position is being averaged away."""

    def forward(self, x):
        """``(B, C, H, W) -> (B, proj_dim)``."""
        return self.net(x)


class NDimHead(nn.Module):
    """Project the encoder's CLS token to exactly ``n`` dimensions.

    The plan fixes ``m = n`` by construction, with ``n`` read off the latent
    registry via the data config. That makes dimension misspecification (V7) a
    single field -- ``output_dim`` -- rather than an architectural change, so
    the V7 ladder cannot accidentally vary anything else.

    Args:
        input_dim: Encoder width (384 for ViT-small).
        output_dim: ``n``. ``n - k`` is V7a, ``n + k`` is V7b.
        hidden_dim: Width of the single hidden layer. ``None`` makes the head
            a bare linear map.
        norm: Whether to LayerNorm the CLS token first. On by default: the
            SIGReg term is scale-sensitive, and an unnormalised backbone
            output would let the head trade isotropy against overall scale.
        hidden_norm_fn: Norm applied to the hidden layer, between the two
            ``Linear``s -- ``None`` (bare, the CNN arm's setting; moot there
            since it has no hidden layer) or ``torch.nn.BatchNorm1d`` for the
            ViT arms. Mirrors where :class:`~stable_worldmodel.wm.lejepa.module.CNNEncoder`
            places its own ``BatchNorm1d``, i.e. on a hidden projection, never
            on the head's output -- so this gives the ViT arms the same
            batch-wise anti-collapse signal the CNN gets structurally, without
            normalising ``h`` itself and trivialising SIGReg/epsilon the way a
            norm *after* the last ``Linear`` would.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        norm=True,
        hidden_norm_fn=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [nn.LayerNorm(input_dim)] if norm else []
        if hidden_dim:
            hidden_norm = (
                hidden_norm_fn(hidden_dim)
                if hidden_norm_fn is not None
                else nn.Identity()
            )
            layers += [
                nn.Linear(input_dim, hidden_dim),
                hidden_norm,
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            ]
        else:
            layers += [nn.Linear(input_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """``(..., input_dim) -> (..., output_dim)``."""
        return self.net(x)


__all__ = ['CNNEncoder', 'NDimHead', 'state_dict_hash']

"""Building blocks for LeJEPA: the n-dim head and the frozen-encoder wrapper.

The transformer blocks are **imported** from ``wm/lewm/module.py`` rather than
copied. Arms A, R, C1 and C2 all use the LeWM ``Predictor`` unchanged, and the
plan holds predictor capacity fixed across arms -- a forked copy that drifted
would make those arms quietly incomparable.

Two things are new:

:class:`NDimHead`
    Maps the encoder's CLS token to exactly ``n`` dimensions, so ``m = n``.
    V7a and V7b are a single config field on this.
:class:`FrozenEncoderWM`
    Holds a frozen :class:`~stable_worldmodel.wm.lejepa.lejepa.LeJEPA` and a
    trainable predictor, and satisfies the ``Dynamics`` protocol so the
    existing planner consumes it without modification.
"""

import hashlib

import torch
from einops import rearrange
from torch import nn

# Re-exported so `wm.lejepa.module.Predictor` resolves for hydra configs, and
# so there is visibly one implementation rather than two.
from stable_worldmodel.wm.lewm.module import (  # noqa: F401
    MLP,
    Attention,
    Block,
    ConditionalBlock,
    Embedder,
    FeedForward,
    Predictor,
    Transformer,
    modulate,
)


def state_dict_hash(module: nn.Module) -> str:
    """Content hash of a module's parameters.

    Used to bind a trained predictor to the exact frozen encoder it was
    trained against. An encoder/predictor mismatch produces embeddings the
    predictor has never seen, which degrades planning in a way that looks
    **exactly** like an identifiability failure -- the same low success rate,
    the same well-behaved losses. Nothing downstream could tell the two apart,
    so it is caught structurally instead.

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
        self, input_dim, output_dim, hidden_dim=None, norm=True, hidden_norm_fn=None
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


class FrozenEncoderWM(nn.Module):
    """A frozen LeJEPA encoder plus a trainable predictor: stage-D arms A/R/C.

    This is the class that satisfies ``Dynamics``; :class:`LeJEPA` deliberately
    does not. Splitting them makes "the encoder is frozen before any predictor
    is trained" a **structural property** rather than a discipline someone has
    to remember: there is no code path here that puts encoder parameters into
    an optimiser, because :meth:`encode` runs under ``torch.no_grad()`` and the
    encoder's parameters have ``requires_grad = False``.

    Arm R is the same class with a randomly initialised encoder -- so the arm
    differs only in the checkpoint it loads, not in any code.

    Args:
        encoder: A :class:`LeJEPA`. Frozen on construction.
        predictor: The LeWM ``Predictor``, trained on this encoder's outputs.
        action_encoder: Embeds actions for the predictor's AdaLN conditioning.
        encoder_hash: Hash recorded when the predictor was trained. When given,
            it is checked against the encoder actually supplied and a mismatch
            raises. See :func:`state_dict_hash`.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        encoder_hash=None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder

        self.freeze_encoder()

        self.encoder_hash = encoder_hash
        if encoder_hash is not None:
            actual = state_dict_hash(self.encoder)
            if actual != encoder_hash:
                raise ValueError(
                    f'frozen encoder hash mismatch: predictor was trained '
                    f'against encoder {encoder_hash}, but the encoder supplied '
                    f'hashes to {actual}. Planning with a mismatched pair '
                    'degrades exactly like an identifiability failure, so this '
                    'is refused rather than reported.'
                )

    def freeze_encoder(self):
        """Put the encoder in eval mode and detach it from every optimiser."""
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        """Train the predictor; keep the encoder in eval mode regardless.

        Without this override, ``module.train()`` would flip the encoder's
        dropout and norm layers back on and silently change the embeddings the
        predictor was trained against.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def current_encoder_hash(self):
        """Hash of the encoder as currently loaded."""
        return state_dict_hash(self.encoder)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def encode(self, info):
        """Embed pixels with the frozen encoder, and actions with the trainable one.

        Args:
            info: Dict with ``pixels`` ``(B, T, C, H, W)`` and optionally
                ``action``.

        Returns:
            dict: ``info`` with ``emb`` ``(B, T, n)`` and, when actions were
            supplied, ``act_emb``.
        """
        with torch.no_grad():
            info = self.encoder.encode(info)
            info['emb'] = info['emb'].detach()

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info

    def predict(self, emb, act_emb):
        """One predictor step. ``emb`` ``(B, T, n)``, ``act_emb`` ``(B, T, A)``."""
        return self.predictor(emb, act_emb)

    def rollout(self, info, action_sequence, history_size: int = None):
        """Roll candidate action sequences forward in the frozen latent space.

        Mirrors ``LeWM.rollout`` exactly -- same context/candidate action
        alignment, same truncation window -- because the planner, the solvers
        and ``GoalMSE`` are all held fixed across arms and must see the same
        rollout semantics whichever arm supplied the dynamics.

        Args:
            info: Dict with ``pixels`` ``(B, S, H, C, h, w)`` and, when
                ``H > 1``, ``action_history`` ``(B, S, H - 1, action_dim)``.
            action_sequence: ``(B, S, T, action_dim)`` strictly-future
                candidates.
            history_size: Context window the predictor attends over.

        Returns:
            dict: ``info`` with ``predicted_emb`` ``(B, S, H + T, n)``.
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        h_frames = info['pixels'].size(2)
        b, s, t = action_sequence.shape[:3]

        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                b, s, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == h_frames - 1, (
            f'action_history must hold H-1={h_frames - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            init = self.encode(init)
            info['emb'] = (
                init['emb'].detach().unsqueeze(1).expand(b, s, -1, -1)
            )

        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_past_flat = rearrange(act_past, 'b s ... -> (b s) ...')
        act_cand_flat = rearrange(action_sequence, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_past_flat, act_cand_flat], dim=1)
        )

        emb_list = list(emb_init.unbind(dim=1))
        for step in range(t):
            lo = max(0, h_frames + step - history_size)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : h_frames + step]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)
        info['predicted_emb'] = rearrange(
            emb, '(b s) ... -> b s ...', b=b, s=s
        )
        return info


__all__ = [
    'CNNEncoder',
    'MLP',
    'Attention',
    'Block',
    'ConditionalBlock',
    'Embedder',
    'FeedForward',
    'FrozenEncoderWM',
    'NDimHead',
    'Predictor',
    'Transformer',
    'state_dict_hash',
]

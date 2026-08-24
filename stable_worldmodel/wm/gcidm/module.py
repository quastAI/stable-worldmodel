"""Model components for the goal-conditioned inverse-dynamics world model.

Same components as ``stable_worldmodel.wm.smwm.module`` with its
single-step ``InverseModel`` replaced by :class:`GoalPolicyHead`, which maps
the predictor's own observation context plus a goal latent to the whole
action-block plan that connects them.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(
            3, dim=-1
        )  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (
            rearrange(t, 'b t (h d) -> b h t d', h=self.heads) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = rearrange(out, 'b h t d -> b t (h d)')
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale
        self.patch_embed = nn.Conv1d(
            input_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class Predictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.emb_dropout = emb_dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class GoalPolicyHead(nn.Module):
    """Goal-conditioned plan head: ``(z_{t-C+1..t}, z_goal) -> (mu, sigma)``.

    Regresses the whole ``horizon``-block action plan that carries the agent
    from the current observation context to the goal observation. ``z_goal``
    is the latent of the frame ``horizon`` action-blocks ahead — the same
    quantity the planner is handed as its goal — so the head can be used
    directly to warm-start a solver over that horizon.

    The context is the *same* set of latents the predictor receives
    (``context_frames == wm.history_size``), not just the current frame, so
    the head sees the motion leading into ``t`` rather than a single static
    snapshot.

    An MLP rather than a transformer: the input is a fixed, ordered,
    length-``C+1`` set with no permutation structure and no variable length
    to exploit (a short context is left-padded by the caller), so
    self-attention over those positions would just be a weight-shared MLP.
    The ``Transformer``/``Block`` classes above remain available if a
    transformer variant is ever wanted.

    ``sigma`` is the *spread of this head's own residual*, not a policy
    entropy — see ``scripts/train/gcidm.py::gcidm_forward``, which trains it
    against a detached mean so calibrating it can never reweight the
    encoder-shaping gradient.

    Args:
        embed_dim: Latent width of one frame embedding.
        action_dim: Flattened action-block width
            (``frameskip * env action dim``).
        horizon: Action blocks predicted per call.
        context_frames: Context latents consumed, matching the predictor's
            ``num_frames``.
        hidden_dim: Width of the hidden layers.
        num_layers: Total ``Linear`` layers, including the output layer.
        predict_std: Also emit a per-dim standard deviation.
        log_std_min: Lower clamp on the predicted ``log sigma``.
        log_std_max: Upper clamp on the predicted ``log sigma``.
    """

    def __init__(
        self,
        embed_dim,
        action_dim,
        horizon,
        context_frames=3,
        hidden_dim=768,
        num_layers=3,
        predict_std=True,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f'num_layers must be >= 1, got {num_layers}')

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.context_frames = context_frames
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_std = predict_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        in_dim = (context_frames + 1) * embed_dim
        out_dim = horizon * action_dim * (2 if predict_std else 1)

        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_ctx, z_goal):
        """
        z_ctx:  (B, C, D) context latents, oldest first
        z_goal: (B, D) goal latent
        Returns ``(mu, sigma)`` of shape (B, horizon, action_dim) each;
        ``sigma`` is None when ``predict_std`` is False.
        """
        if z_ctx.size(1) != self.context_frames:
            raise ValueError(
                f'expected {self.context_frames} context latents, '
                f'got {z_ctx.size(1)}'
            )

        x = torch.cat([z_ctx.flatten(1), z_goal], dim=-1)
        out = self.net(x)

        if not self.predict_std:
            mu = out.view(-1, self.horizon, self.action_dim)
            return mu, None

        mu, log_std = out.chunk(2, dim=-1)
        mu = mu.reshape(-1, self.horizon, self.action_dim)
        log_std = log_std.reshape(-1, self.horizon, self.action_dim)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mu, log_std.exp()

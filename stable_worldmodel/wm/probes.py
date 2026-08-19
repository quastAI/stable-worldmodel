"""Helper functions and read-out heads for probing wm latent spaces.

Two pieces live here:

  - :func:`attach_probe` / :func:`get_probe` / :func:`load_probe` — keep a
    fitted probe alongside the world model it was fitted on.
  - :class:`LinearProbe` / :class:`MLPProbe` — the two classical read-out
    heads. Both carry their own input-standardization buffers, so a probe
    saved with ``torch.save`` is self-contained: it expects *raw* frozen
    features and does the whitening itself.
"""

import numpy as np
import torch
from torch import nn


def attach_probe(model, key, probe):
    """Attach a probe to the model under the given key."""
    assert isinstance(probe, nn.Module), 'Probe must be a nn.Module'
    if not hasattr(model, '_probes'):
        model._probes = nn.ModuleDict()
    model._probes[key] = probe


def get_probe(model, key):
    """Get the probe attached to the model under the given key."""
    if hasattr(model, '_probes'):
        return model._probes[key] if key in model._probes else None

    return None


def load_probe(model, key, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    is_module = isinstance(payload, nn.Module)

    if is_module:
        attach_probe(model, key, payload)
    elif isinstance(payload, dict):
        probe = get_probe(model, key)
        if probe is None:
            raise ValueError(f'No probe found for key {key} in model')

        probe.load_state_dict(payload)
    return


class StandardizedProbe(nn.Module):
    """Base class holding the feature-whitening buffers of a probe.

    A probe is only meaningful together with the statistics of the frozen
    features it was fitted on, so they travel *inside* the module rather
    than in a sidecar file. ``forward`` of a subclass must run its input
    through :meth:`standardize` first.

    Args:
        input_dim: Dimensionality of the frozen feature vector.
        output_dim: Number of outputs (target dims for regression, classes
            for classification).
        standardize: Whether :meth:`standardize` whitens its input. When
            ``False`` the buffers stay at mean 0 / std 1 and the call is a
            no-op, which keeps ``state_dict`` shapes identical either way.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.standardize_input = bool(standardize)
        self.register_buffer('feature_mean', torch.zeros(1, self.input_dim))
        self.register_buffer('feature_std', torch.ones(1, self.input_dim))
        self.register_buffer('target_mean', torch.zeros(1, self.output_dim))
        self.register_buffer('target_std', torch.ones(1, self.output_dim))

    @torch.no_grad()
    def set_feature_stats(self, mean, std, eps: float = 1e-6) -> None:
        """Store the per-dim mean/std of the training features.

        Args:
            mean: Per-dim mean, shape ``(input_dim,)`` or ``(1, input_dim)``.
            std: Per-dim std, same shape. Values below ``eps`` are clamped —
                a constant feature dim would otherwise blow up.
            eps: Floor for ``std``.
        """
        mean = torch.as_tensor(np.asarray(mean, dtype=np.float32)).reshape(
            1, -1
        )
        std = torch.as_tensor(np.asarray(std, dtype=np.float32)).reshape(1, -1)
        if mean.shape[1] != self.input_dim or std.shape[1] != self.input_dim:
            raise ValueError(
                f'feature stats must have {self.input_dim} dims, got '
                f'mean={tuple(mean.shape)} std={tuple(std.shape)}'
            )
        self.feature_mean.copy_(mean.to(self.feature_mean.device))
        self.feature_std.copy_(std.clamp(min=eps).to(self.feature_std.device))

    @torch.no_grad()
    def set_target_stats(self, mean, std, eps: float = 1e-8) -> None:
        """Store the per-dim mean/std of the training *targets*.

        Regression probes are fitted against standardized targets so that
        dims in different units (metres, radians) contribute comparably to
        the loss. Keeping the stats on the module means :meth:`unscale` can
        map a prediction back to physical units later.
        """
        mean = torch.as_tensor(np.asarray(mean, dtype=np.float32)).reshape(
            1, -1
        )
        std = torch.as_tensor(np.asarray(std, dtype=np.float32)).reshape(1, -1)
        if mean.shape[1] != self.output_dim or std.shape[1] != self.output_dim:
            raise ValueError(
                f'target stats must have {self.output_dim} dims, got '
                f'mean={tuple(mean.shape)} std={tuple(std.shape)}'
            )
        self.target_mean.copy_(mean.to(self.target_mean.device))
        self.target_std.copy_(std.clamp(min=eps).to(self.target_std.device))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten ``x`` with the stored stats (identity if disabled)."""
        if not self.standardize_input:
            return x
        return (x - self.feature_mean) / self.feature_std

    def unscale(self, y: torch.Tensor) -> torch.Tensor:
        """Map a standardized-target prediction back to physical units."""
        return y * self.target_std + self.target_mean

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in physical units (regression probes only)."""
        return self.unscale(self(x))


class LinearProbe(StandardizedProbe):
    """A single affine read-out — the classical linear probe.

    Args:
        input_dim: Frozen-feature dimensionality.
        output_dim: Target dims (regression) or classes (classification).
        bias: Whether the read-out has a bias term.
        standardize: See :class:`StandardizedProbe`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,
        standardize: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, standardize=standardize)
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.standardize(x))

    @torch.no_grad()
    def set_weights(self, weight, bias=None) -> None:
        """Load a closed-form solution into the read-out.

        Args:
            weight: ``(input_dim, output_dim)`` or ``(output_dim, input_dim)``
                — the orientation is inferred, which is what makes this
                usable straight from a least-squares solve.
            bias: ``(output_dim,)``, or ``None`` to **zero** the bias.
                ``None`` zeroes rather than leaves-as-is because a solve on
                centred variables has no intercept, and inheriting
                ``nn.Linear``'s random init would silently offset every
                prediction.
        """
        w = torch.as_tensor(np.asarray(weight, dtype=np.float32))
        if w.shape == (self.input_dim, self.output_dim):
            w = w.t()
        if w.shape != (self.output_dim, self.input_dim):
            raise ValueError(
                f'weight must be ({self.input_dim}, {self.output_dim}) or its '
                f'transpose, got {tuple(w.shape)}'
            )
        self.fc.weight.copy_(w.to(self.fc.weight.device))

        if bias is None:
            if self.fc.bias is not None:
                self.fc.bias.zero_()
            return

        if self.fc.bias is None:
            raise ValueError('probe was built with bias=False')
        b = torch.as_tensor(np.asarray(bias, dtype=np.float32)).reshape(-1)
        if b.numel() != self.output_dim:
            raise ValueError(
                f'bias must have {self.output_dim} entries, got {b.numel()}'
            )
        self.fc.bias.copy_(b.to(self.fc.bias.device))


class MLPProbe(StandardizedProbe):
    """A small MLP read-out — the non-linear rung of the probing ladder.

    Kept deliberately plain (``Linear → norm → act → dropout`` blocks, then
    a linear read-out) so that the only difference from
    :class:`LinearProbe` is capacity.

    Args:
        input_dim: Frozen-feature dimensionality.
        output_dim: Target dims (regression) or classes (classification).
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers (``0`` degenerates to a linear
            read-out, which is handy for sweeps over depth).
        dropout: Dropout probability applied after each activation.
        norm: Insert a :class:`~torch.nn.LayerNorm` in each hidden block.
        act_fn: Activation module class.
        standardize: See :class:`StandardizedProbe`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm: bool = True,
        act_fn=nn.GELU,
        standardize: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, standardize=standardize)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        layers: list[nn.Module] = []
        dim = self.input_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(dim, self.hidden_dim))
            if norm:
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = self.hidden_dim
        layers.append(nn.Linear(dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.standardize(x))


__all__ = [
    'LinearProbe',
    'MLPProbe',
    'StandardizedProbe',
    'attach_probe',
    'get_probe',
    'load_probe',
]

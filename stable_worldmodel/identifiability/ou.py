"""The OU sampler that generates encoder-dataset latent pairs.

Ported and generalised from ``lejepa-identifiability/experiments/lejepa_id/data.py``.
Where that module hard-codes a scalar ``rho`` and a shared marginal, this one
exposes every departure from the ideal setting as a knob, because those
departures *are* the experiment: violations V1-V5 and V9 are all properties of
this sampler.

The process
-----------
Latents live in **z-space** -- unit-variance standard coordinates, mean zero.
The registry (:mod:`stable_worldmodel.identifiability.latents`) owns the affine
map from z-space to physical units, so the sampler never needs to know that one
coordinate is a metre and another is a radian.

A positive pair is one OU step::

    z'_alpha = rho_alpha * z_alpha + sqrt(1 - rho_alpha^2) * eta_alpha

which is the stationary transition of an Ornstein-Uhlenbeck process observed at
a fixed interval: if ``z`` has the sampler's marginal, so does ``z'``, for any
``rho`` in ``[0, 1)``. That stationarity is what Thm 1 needs, and it is what
every violation here breaks in a different, controlled way.

Which violation lives where
---------------------------
============  =========================  ====================================
Violation     Knob                       Breaks
============  =========================  ====================================
V1            ``dist`` / ``alpha``       Gaussianity of the marginal
V2            ``drift``                  Stationarity across a pseudo-episode
V3            ``noise_coupling``         Additivity -- sigma becomes sigma(z)
V4            ``rho`` as a vector        Isotropy of the transition
V5            ``truncation``             Support -- rejection makes it compact
V9            ``cross_corr``             Independence across coordinates
============  =========================  ====================================

V6 (occlusion) is a property of the renderer, V7 (dimension misspecification)
and V8 (optimisation gap) of the training config. None of the three is a
sampler concern and none appears here.
"""

import math
from dataclasses import dataclass, field

import numpy as np


# Defaults for the z-space -> physical affine. A latent whose physical bounds
# are `[lo, hi]` is centered at the midpoint and scaled so that `+-SIGMA_SPAN`
# standard deviations reach the bounds.
SIGMA_SPAN = 3.0

# Rejection sampling gives up after this many attempts per batch and reports
# how far it got, rather than looping forever on an infeasible predicate.
MAX_REJECTION_ROUNDS = 64


def gennorm_unit_var_scale(alpha: float) -> float:
    """Scale making a generalised-normal density unit-variance.

    ``beta = sqrt(Gamma(1/alpha) / Gamma(3/alpha))``, evaluated in log space so
    it stays finite for the small ``alpha`` the V1 sweep reaches. Taken exactly
    from the reference ``data.py`` so the V1 results are directly comparable to
    the paper's Fig. 4b and 7-8.

    Args:
        alpha: Shape parameter. ``alpha = 2`` is Gaussian, ``1`` is Laplace,
            and ``alpha -> inf`` tends to uniform.

    Returns:
        float: The scale ``beta``.
    """
    if alpha <= 0:
        raise ValueError(f'gennorm alpha must be positive; got {alpha}.')
    return math.exp(
        0.5 * (math.lgamma(1.0 / alpha) - math.lgamma(3.0 / alpha))
    )


def sample_marginal(shape, dist, rng, alpha=None):
    """Draw unit-variance, zero-mean samples from one of the V1 families.

    Args:
        shape: Output shape.
        dist: ``'gaussian'``, ``'laplace'`` or ``'gennorm'``.
        rng: A ``numpy.random.Generator``.
        alpha: Shape parameter, required for ``'gennorm'``.

    Returns:
        ndarray: Samples of the requested shape, unit variance per coordinate.

    Raises:
        ValueError: On an unknown family, or ``'gennorm'`` without ``alpha``.
    """
    if dist == 'gaussian':
        return rng.standard_normal(shape)
    if dist == 'laplace':
        # Laplace(0, b) has variance 2b^2, so b = 1/sqrt(2) gives unit variance.
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), shape)
    if dist == 'gennorm':
        if alpha is None:
            raise ValueError("dist='gennorm' requires alpha.")
        scale = gennorm_unit_var_scale(float(alpha))
        magnitude = rng.gamma(1.0 / float(alpha), 1.0, shape)
        sign = rng.integers(0, 2, shape) * 2.0 - 1.0
        return scale * sign * magnitude ** (1.0 / float(alpha))
    raise ValueError(
        f"Unknown dist {dist!r}; expected 'gaussian', 'laplace' or 'gennorm'."
    )


# rho must stay strictly inside (0, 1): rho = 0 destroys the pair correlation
# the alignment term needs, and rho = 1 makes the two views identical.
RHO_EPS = 1e-3


def feasible_spread(rho_bar, spread):
    """Largest fan half-width that keeps every coordinate inside ``(0, 1)``.

    A fan of half-width ``spread`` about ``rho_bar`` reaches
    ``rho_bar +- spread``, so the feasible half-width is limited by whichever
    endpoint is nearer. At the frozen ``rho_bar = 0.9`` that is the *upper*
    endpoint, and the ceiling is only ``0.099`` -- far below what a naive
    reading of the V4 ladder would suggest.

    Args:
        rho_bar: Declared mean autocorrelation.
        spread: Requested half-width.

    Returns:
        tuple: ``(allowed, was_clamped)``.
    """
    ceiling = min(rho_bar - RHO_EPS, 1.0 - RHO_EPS - rho_bar)
    ceiling = max(ceiling, 0.0)
    allowed = min(float(spread), ceiling)
    return allowed, allowed < float(spread)


def spread_rho(rho_bar, spread, n, rng=None, mode='linear'):
    """Build a per-dimension ``rho`` vector around a declared mean.

    The V4 sweep's severity axis. ``spread = 0`` returns the isotropic vector;
    larger values fan the coordinates out around ``rho_bar`` **while keeping
    the mean exactly at** ``rho_bar``, so severity and the program constant
    stay independent. That last property is the reason this function exists
    rather than a ``clip(rho_bar + offsets)`` one-liner: clipping the *result*
    silently drags the mean down, which would turn a V4 sweep into a joint
    sweep over ``rho`` -- and ``rho`` is a frozen program constant. The
    requested half-width is clamped to :func:`feasible_spread` instead, so the
    fan stays symmetric and the mean is preserved by construction.

    Args:
        rho_bar: Declared mean autocorrelation.
        spread: Requested half-width of the fan. Clamped to what is feasible;
            use :func:`feasible_spread` to detect the clamp in advance.
        n: Number of coordinates.
        rng: Generator, used only when ``mode='random'``.
        mode: ``'linear'`` lays the coordinates on an even ramp -- reproducible
            and monotone in the coordinate index, which is what a sweep wants.
            ``'random'`` draws them uniformly.

    Returns:
        ndarray: ``(n,)`` of per-dimension ``rho``, mean exactly ``rho_bar``.
    """
    rho_bar = float(rho_bar)
    if n <= 0:
        raise ValueError(f'n must be positive; got {n}.')
    if not RHO_EPS <= rho_bar <= 1.0 - RHO_EPS:
        raise ValueError(
            f'rho_bar must lie in [{RHO_EPS}, {1 - RHO_EPS}]; got {rho_bar}.'
        )

    spread, _ = feasible_spread(rho_bar, spread)
    if spread == 0.0 or n == 1:
        return np.full(n, rho_bar, dtype=np.float64)

    if mode == 'linear':
        offsets = np.linspace(-spread, spread, n)
    elif mode == 'random':
        rng = rng or np.random.default_rng()
        offsets = rng.uniform(-spread, spread, n)
    else:
        raise ValueError(f"mode must be 'linear' or 'random'; got {mode!r}.")

    # Exactly mean-preserving: `linspace` is already symmetric, and recentring
    # the random branch makes it so.
    offsets = offsets - offsets.mean()
    return rho_bar + offsets


def isotropy_criterion(rho):
    """The paper's Table-2 boundary: ``min rho_alpha > (max rho_alpha)^2``.

    Below the boundary, the slowest coordinate decorrelates faster than the
    *second Hermite function* of the fastest one, and the encoder can lower the
    objective by substituting ``He2`` of a fast latent for a slow latent it
    should have kept. That substitution is the concrete V4 failure mode, which
    is why this is reported on every dataset rather than only when V4 is swept.

    Args:
        rho: Scalar or per-dimension autocorrelation.

    Returns:
        dict: ``satisfied`` (bool), ``margin`` (``min - max^2``; positive means
        satisfied), ``rho_min``, ``rho_max``.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    rho_min = float(rho.min())
    rho_max = float(rho.max())
    margin = rho_min - rho_max**2
    return {
        'satisfied': bool(margin > 0.0),
        'margin': margin,
        'rho_min': rho_min,
        'rho_max': rho_max,
    }


@dataclass
class Drift:
    """V2: a slow change in the process's own parameters across an episode.

    Stationarity is an assumption about the *process*, not about any one
    sample, so breaking it requires a notion of time. That is what a
    pseudo-episode is: ``period`` consecutive pairs over which these offsets
    ramp linearly from zero to their stated values and then reset.

    Attributes:
        rho: Total change in ``rho`` over one pseudo-episode.
        mean: Total shift of the marginal's mean.
        scale: Total multiplicative change in the marginal's scale.
        period: Pairs per pseudo-episode.
    """

    rho: float = 0.0
    mean: float = 0.0
    scale: float = 0.0
    period: int = 64

    def at(self, index):
        """Drift factors at pair ``index`` within its pseudo-episode.

        Args:
            index: Global pair index, or an array of them.

        Returns:
            tuple: ``(d_rho, d_mean, scale_factor)``, broadcastable against
            ``index``.
        """
        phase = (np.asarray(index) % max(1, int(self.period))) / max(
            1, int(self.period)
        )
        return (
            self.rho * phase,
            self.mean * phase,
            1.0 + self.scale * phase,
        )


@dataclass
class Truncation:
    """V5: rejection of samples that fall outside a feasible region.

    Attributes:
        radius: Samples whose z-space norm exceeds this are rejected. ``None``
            disables the norm predicate.
        box: Per-coordinate ``(low, high)`` in z-space, or ``None``.
        strength: Probability that a violating sample is actually rejected.
            At ``1.0`` the support is hard-truncated; below that the tails are
            thinned rather than removed, which is what makes severity a
            continuous axis instead of a switch.
    """

    radius: float | None = None
    box: tuple | None = None
    strength: float = 1.0

    def accept(self, z, rng):
        """Boolean mask of which rows survive.

        Args:
            z: ``(B, n)`` samples in z-space.
            rng: Generator, for sub-unit ``strength``.

        Returns:
            ndarray: ``(B,)`` boolean mask.
        """
        violating = np.zeros(len(z), dtype=bool)
        if self.radius is not None:
            violating |= np.linalg.norm(z, axis=1) > self.radius
        if self.box is not None:
            low, high = self.box
            violating |= np.any((z < low) | (z > high), axis=1)
        if not violating.any():
            return np.ones(len(z), dtype=bool)
        rejected = violating & (rng.random(len(z)) < self.strength)
        return ~rejected


@dataclass
class OUSampler:
    """Generates ``(z, z')`` positive pairs, with every violation as a knob.

    Attributes:
        n: Number of latent coordinates.
        rho: Scalar, or ``(n,)`` per-coordinate autocorrelation (**V4**).
        dist: Marginal family (**V1**).
        alpha: ``gennorm`` shape (**V1**).
        drift: :class:`Drift`, or ``None`` (**V2**).
        noise_coupling: ``c`` in ``sigma(z) = 1 + c * |z|`` (**V3**).
        cross_corr: Off-diagonal loading of the driving noise (**V9**).
        truncation: :class:`Truncation`, or ``None`` (**V5**).
        seed: Seed for the sampler's own generator.
    """

    n: int
    rho: float | np.ndarray = 0.9
    dist: str = 'gaussian'
    alpha: float | None = None
    drift: Drift | None = None
    noise_coupling: float = 0.0
    cross_corr: float = 0.0
    truncation: Truncation | None = None
    seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _mixing: np.ndarray | None = field(init=False, repr=False, default=None)
    _index: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError(f'n must be positive; got {self.n}.')
        self._rng = np.random.default_rng(self.seed)
        self.rho = self._as_rho_vector(self.rho)
        if not (0.0 < self.rho).all() or not (self.rho < 1.0).all():
            raise ValueError(
                f'rho must lie strictly in (0, 1); got range '
                f'[{self.rho.min()}, {self.rho.max()}].'
            )
        self._mixing = self._build_mixing()

    def _as_rho_vector(self, rho):
        rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        if rho.size == 1:
            return np.full(self.n, float(rho[0]))
        if rho.size != self.n:
            raise ValueError(
                f'rho must be scalar or length n={self.n}; got {rho.size}.'
            )
        return rho.astype(np.float64)

    def _build_mixing(self):
        """V9: the matrix that correlates the driving noise across coordinates.

        A symmetric, unit-diagonal correlation target with every off-diagonal
        equal to ``cross_corr``, realised through its Cholesky factor and then
        rescaled so each coordinate keeps unit variance. Keeping the marginals
        unit-variance is what isolates V9 as a *dependence* violation rather
        than a variance one -- otherwise the whitening error would move for a
        reason that has nothing to do with independence.
        """
        if self.cross_corr == 0.0:
            return None
        c = float(self.cross_corr)
        # A constant-correlation matrix is positive definite exactly on
        # (-1/(n-1), 1); outside that it has no square root at all.
        lower = -1.0 / (self.n - 1) if self.n > 1 else -1.0
        if not lower < c < 1.0:
            raise ValueError(
                f'cross_corr must lie in ({lower:.4f}, 1) for n={self.n}; '
                f'got {c}.'
            )
        target = np.full((self.n, self.n), c, dtype=np.float64)
        np.fill_diagonal(target, 1.0)
        factor = np.linalg.cholesky(target)
        return factor / np.linalg.norm(factor, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def _draw_noise(self, batch):
        """Driving noise: the V1 marginal, optionally mixed across dims (V9)."""
        eta = sample_marginal((batch, self.n), self.dist, self._rng, self.alpha)
        if self._mixing is not None:
            eta = eta @ self._mixing.T
        return eta

    def sample_marginal_batch(self, batch):
        """Draw ``batch`` independent samples from the stationary marginal."""
        z = self._draw_noise(batch)
        if self.truncation is not None:
            z = self._reject(z)
        return z

    def _reject(self, z):
        """Resample rejected rows until they pass, or the budget runs out."""
        for _ in range(MAX_REJECTION_ROUNDS):
            keep = self.truncation.accept(z, self._rng)
            if keep.all():
                return z
            replacement = self._draw_noise(int((~keep).sum()))
            z[~keep] = replacement
        return z

    def sample_pairs(self, batch):
        """Draw ``batch`` positive pairs.

        Args:
            batch: Number of pairs.

        Returns:
            tuple: ``(z, z_next, meta)``. ``z`` and ``z_next`` are ``(batch,
            n)`` in z-space; ``meta`` carries the per-pair ``rho`` actually
            used, which differs from ``self.rho`` under drift.
        """
        indices = np.arange(self._index, self._index + batch)
        self._index += batch

        rho = np.broadcast_to(self.rho, (batch, self.n)).copy()
        mean_shift = np.zeros((batch, 1))
        scale = np.ones((batch, 1))
        if self.drift is not None:
            d_rho, d_mean, d_scale = self.drift.at(indices)
            rho = np.clip(rho + d_rho[:, None], 1e-3, 1.0 - 1e-3)
            mean_shift = d_mean[:, None]
            scale = d_scale[:, None]

        z = self.sample_marginal_batch(batch)
        eta = self._draw_noise(batch)

        # V3: state-dependent noise. The innovation's scale becomes a function
        # of where the process currently is, so the transition kernel is no
        # longer the same everywhere -- which is precisely the additivity
        # assumption the bound rests on.
        if self.noise_coupling != 0.0:
            sigma = 1.0 + self.noise_coupling * np.abs(z)
            # Renormalised to keep the *marginal* variance at one, so V3 is a
            # heteroscedasticity violation and not a scale violation.
            sigma = sigma / np.sqrt((sigma**2).mean(axis=0, keepdims=True))
            eta = eta * sigma

        z_next = rho * z + np.sqrt(1.0 - rho**2) * eta

        if self.truncation is not None:
            keep = self.truncation.accept(z_next, self._rng)
            # A rejected *successor* cannot simply be redrawn: that would
            # condition z' on acceptance while leaving z unconditioned, which
            # is a different violation than the one V5 declares. Redraw the
            # whole pair instead.
            if not keep.all():
                n_bad = int((~keep).sum())
                z_r, z_next_r, _ = self.sample_pairs(n_bad)
                z[~keep] = z_r
                z_next[~keep] = z_next_r

        z = z * scale + mean_shift
        z_next = z_next * scale + mean_shift

        return z, z_next, {'rho': rho, 'index': indices}

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def empirical_rho(self, z, z_next):
        """Per-coordinate lag-1 correlation actually realised in a sample.

        The declared ``rho`` and the achieved one part company under drift, V3
        and V5, so the achieved value is what the metric suite reports and what
        arm C2's stride is tuned against.

        Args:
            z: ``(B, n)`` first elements.
            z_next: ``(B, n)`` second elements.

        Returns:
            ndarray: ``(n,)`` empirical correlations.
        """
        a = z - z.mean(axis=0)
        b = z_next - z_next.mean(axis=0)
        denominator = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
        return np.where(denominator > 0, (a * b).sum(axis=0) / denominator, 0.0)

    def describe(self):
        """The sampler's full configuration, for the dataset manifest.

        Every field that could change a sample is recorded, so a dataset can
        never be mistaken for a differently-configured one.
        """
        return {
            'n': int(self.n),
            'rho': self.rho.tolist(),
            'rho_mean': float(self.rho.mean()),
            'dist': self.dist,
            'alpha': None if self.alpha is None else float(self.alpha),
            'drift': None
            if self.drift is None
            else {
                'rho': self.drift.rho,
                'mean': self.drift.mean,
                'scale': self.drift.scale,
                'period': self.drift.period,
            },
            'noise_coupling': float(self.noise_coupling),
            'cross_corr': float(self.cross_corr),
            'truncation': None
            if self.truncation is None
            else {
                'radius': self.truncation.radius,
                'box': self.truncation.box,
                'strength': self.truncation.strength,
            },
            'seed': self.seed,
            'isotropy': isotropy_criterion(self.rho),
        }


def spectral_gap(rho):
    """``2 rho (1 - rho)`` -- the unit every violation cost is reported in.

    The gap between the leading and second eigenvalue of the OU transition
    operator. Reporting a degradation in these units is what makes costs
    comparable across environments whose ``rho`` differs.

    Args:
        rho: Scalar or per-dimension autocorrelation. A vector is reduced by
            its mean, which is the aggregation rule the plan fixes.

    Returns:
        float: The spectral gap.
    """
    rho = float(np.mean(np.asarray(rho, dtype=np.float64)))
    return 2.0 * rho * (1.0 - rho)


__all__ = [
    'SIGMA_SPAN',
    'Drift',
    'OUSampler',
    'Truncation',
    'feasible_spread',
    'gennorm_unit_var_scale',
    'isotropy_criterion',
    'sample_marginal',
    'spectral_gap',
    'spread_rho',
]

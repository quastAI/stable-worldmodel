"""The OU sampler that generates the encoder dataset's latent pairs.

Latents live in **z-space** -- unit-variance standard coordinates, mean zero.
The registry (:mod:`stable_worldmodel.identifiability.latents`) owns the affine
map from z-space to physical units, so the sampler never needs to know that one
coordinate is a metre and another is a radian.

A positive pair is one step of a stationary Ornstein-Uhlenbeck process observed
at a fixed interval::

    z' = rho * z + sqrt(1 - rho^2) * eta,    eta ~ N(0, I)

If ``z`` has the standard normal marginal then so does ``z'``, for any ``rho``
in ``(0, 1)``. That stationarity is what the identifiability theorem needs.

**Isotropic and Gaussian by construction.** ``rho`` is one scalar shared by
every coordinate, and the marginal is standard normal with no alternative
families. The violation ladder that used to make each of those a knob --
non-Gaussian marginals, a per-coordinate ``rho`` vector, drift, state-dependent
noise, cross-coordinate correlation, rejection sampling -- is gone along with
the rest of the multi-arm program. Re-adding any of them means re-adding a
sampler field, and doing that deliberately is the point.
"""

import math
from dataclasses import dataclass, field

import numpy as np


# Defaults for the z-space -> physical affine. A latent whose physical bounds
# are `[lo, hi]` is centered at the midpoint and scaled so that `+-SIGMA_SPAN`
# standard deviations reach the bounds.
SIGMA_SPAN = 3.0

# rho must stay strictly inside (0, 1): rho = 0 destroys the pair correlation
# the alignment term needs, and rho = 1 makes the two views identical.
RHO_EPS = 1e-3


@dataclass
class OUSampler:
    """Generates ``(z, z')`` positive pairs from a stationary OU process.

    Attributes:
        n: Number of latent coordinates.
        rho: Lag-1 autocorrelation, shared by every coordinate.
        seed: Seed for the sampler's own generator.
    """

    n: int
    rho: float = 0.9
    seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if self.n <= 0:
            raise ValueError(f'n must be positive; got {self.n}.')
        self.rho = float(self.rho)
        if not RHO_EPS <= self.rho <= 1.0 - RHO_EPS:
            raise ValueError(
                f'rho must lie in [{RHO_EPS}, {1 - RHO_EPS}]; got {self.rho}.'
            )
        self._rng = np.random.default_rng(self.seed)

    def sample_marginal_batch(self, batch):
        """Draw ``batch`` independent samples from the stationary marginal."""
        return self._rng.standard_normal((batch, self.n))

    def sample_pairs(self, batch):
        """Draw ``batch`` positive pairs.

        Args:
            batch: Number of pairs.

        Returns:
            tuple: ``(z, z_next)``, each ``(batch, n)`` in z-space.
        """
        z = self.sample_marginal_batch(batch)
        eta = self.sample_marginal_batch(batch)
        z_next = self.rho * z + math.sqrt(1.0 - self.rho**2) * eta
        return z, z_next

    def empirical_rho(self, z, z_next):
        """Per-coordinate lag-1 correlation actually realised in a sample.

        Reported by the dataset audit. It should match :attr:`rho` to sampling
        error; a systematic gap means the collection pipeline altered the
        process it claims to carry.

        Args:
            z: ``(B, n)`` first elements.
            z_next: ``(B, n)`` second elements.

        Returns:
            ndarray: ``(n,)`` empirical correlations.
        """
        a = z - z.mean(axis=0)
        b = z_next - z_next.mean(axis=0)
        denominator = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
        return np.where(
            denominator > 0, (a * b).sum(axis=0) / denominator, 0.0
        )

    def describe(self):
        """The sampler's full configuration, for the dataset manifest.

        Every field that could change a sample is recorded, so a dataset can
        never be mistaken for a differently-configured one.
        """
        return {
            'n': int(self.n),
            'rho': self.rho,
            'dist': 'gaussian',
            'seed': self.seed,
        }


def spectral_gap(rho):
    """``2 rho (1 - rho)``, the gap of the OU transition operator.

    It is the denominator of the bound's ``D = delta / (2 rho (1 - rho))``, and
    the unit that makes a recovery error comparable across environments whose
    ``rho`` differs.
    """
    rho = float(rho)
    return 2.0 * rho * (1.0 - rho)


__all__ = ['RHO_EPS', 'SIGMA_SPAN', 'OUSampler', 'spectral_gap']

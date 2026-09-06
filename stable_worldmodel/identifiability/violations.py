"""The nine violations, as named configurations with severity ladders.

Each violation is one dataclass with a single scalar ``severity`` in ``[0, 1]``
and a method that turns that scalar into concrete knob values. Keeping severity
on a common scale is what makes the global scatter's x-axis comparable across
violations that are otherwise nothing alike -- a distribution shape, a
rejection rate, and an optimiser budget.

Where each one binds
--------------------
Not every violation acts in the same place, and getting this wrong produces a
dataset that is mislabelled rather than merely wrong:

* **V1-V5, V9 bind at the sampler**, so they are properties of the *encoder*
  dataset and are baked into it at generation time.
* **V6 binds at the environment** -- occlusion is a rendering property.
* **V7, V8 bind at the training config**, not at any dataset. The same encoder
  dataset is reused across their whole ladder.

The predictor dataset is exempt from V1-V4 and V9 entirely: it is collected for
action-space coverage and persistent excitation, not to satisfy a marginal.

Induced baselines
-----------------
Two violations have a **non-zero cost even at severity zero**, and both must be
measured before anything is swept on top of them:

* **V6** has a real floor. The yaw marker is invisible for some poses, digit
  distractors overlap the cube, and the oblique camera foreshortens the top
  face. :meth:`Occlusion.baseline_note` records what is known.
* **V5** has a floor of *zero* in stage A by construction, because
  interpenetration and table-clipping are deliberately allowed. That is a
  design decision, not luck, and it stops being true the moment feasibility
  filtering is switched on.
"""

from dataclasses import dataclass, field

import numpy as np

from stable_worldmodel.identifiability.ou import (
    Drift,
    Truncation,
    feasible_spread,
    isotropy_criterion,
    spread_rho,
)


# Severity ladders. Frozen before Env 1 -- the seed and severity budget is the
# single biggest driver of total compute and must not move once the sweep has
# started, or early and late rows of the scatter stop being comparable.
DEFAULT_LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)
COARSE_LADDER = (0.0, 0.33, 0.67, 1.0)

# Violations with quantitative predictions get the finer ladder and more seeds,
# because those are the three where the measurement can actually falsify
# something rather than merely describe it.
QUANTITATIVE = ('v1', 'v4', 'v8')


@dataclass
class Violation:
    """Base class: a named knob with a severity in ``[0, 1]``."""

    severity: float = 0.0

    #: Short identifier, used as the scatter's violation column.
    key: str = field(default='none', init=False)
    #: Where the knob is applied: 'sampler', 'env' or 'training'.
    site: str = field(default='sampler', init=False)

    def __post_init__(self):
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f'{self.key}: severity must lie in [0, 1]; got {self.severity}.'
            )

    def ladder(self):
        """The severities this violation is swept over."""
        return (
            DEFAULT_LADDER if self.key in QUANTITATIVE else COARSE_LADDER
        )

    def sampler_kwargs(self, n, rho_bar):
        """Knob overrides for :class:`~...ou.OUSampler`. Empty by default."""
        return {}

    def env_kwargs(self):
        """Knob overrides for the environment. Empty by default."""
        return {}

    def training_overrides(self, n):
        """Knob overrides for the training config. Empty by default."""
        return {}

    def describe(self):
        return {'key': self.key, 'site': self.site, 'severity': self.severity}


@dataclass
class NoViolation(Violation):
    """The control arm. Every sweep's severity-zero rung."""

    key: str = field(default='none', init=False)
    site: str = field(default='sampler', init=False)


@dataclass
class NonGaussianMarginal(Violation):
    """V1: the latent marginal stops being Gaussian.

    Severity interpolates the generalised-normal shape ``alpha`` away from
    ``2`` (Gaussian) toward ``alpha_min``. The prediction, recorded before
    measurement: recovery peaks at ``alpha = 2`` and degrades in *both*
    directions -- toward Laplace and toward uniform -- because it is
    Gaussianity that the SIGReg term targets, not tail weight as such.

    Attributes:
        alpha_min: Shape reached at severity 1. Below 1 the density is
            spikier than Laplace.
        direction: ``'heavy'`` moves alpha down (toward Laplace and beyond);
            ``'light'`` moves it up (toward uniform).
    """

    alpha_min: float = 0.5
    alpha_max: float = 8.0
    direction: str = 'heavy'
    key: str = field(default='v1', init=False)
    site: str = field(default='sampler', init=False)

    def alpha(self):
        """Shape parameter at this severity."""
        if self.severity == 0.0:
            return 2.0
        if self.direction == 'heavy':
            return 2.0 + self.severity * (self.alpha_min - 2.0)
        if self.direction == 'light':
            return 2.0 + self.severity * (self.alpha_max - 2.0)
        raise ValueError(
            f"direction must be 'heavy' or 'light'; got {self.direction!r}."
        )

    def sampler_kwargs(self, n, rho_bar):
        if self.severity == 0.0:
            return {'dist': 'gaussian', 'alpha': None}
        return {'dist': 'gennorm', 'alpha': self.alpha()}

    def describe(self):
        return {**super().describe(), 'alpha': self.alpha(),
                'direction': self.direction}


@dataclass
class NonStationary(Violation):
    """V2: the process's parameters drift across a pseudo-episode.

    Attributes:
        max_rho_drift: Change in ``rho`` over one pseudo-episode at severity 1.
        max_mean_drift: Shift of the marginal mean, in z-space units.
        max_scale_drift: Fractional change in the marginal scale.
        period: Pairs per pseudo-episode.
    """

    max_rho_drift: float = 0.08
    max_mean_drift: float = 0.5
    max_scale_drift: float = 0.4
    period: int = 64
    key: str = field(default='v2', init=False)
    site: str = field(default='sampler', init=False)

    def sampler_kwargs(self, n, rho_bar):
        if self.severity == 0.0:
            return {'drift': None}
        return {
            'drift': Drift(
                rho=self.severity * self.max_rho_drift,
                mean=self.severity * self.max_mean_drift,
                scale=self.severity * self.max_scale_drift,
                period=self.period,
            )
        }


@dataclass
class StateDependentNoise(Violation):
    """V3: ``sigma`` becomes a function of the state, ``1 + c|z|``.

    The transition kernel stops being the same everywhere, which is the
    additivity assumption the bound rests on.
    """

    max_coupling: float = 1.5
    key: str = field(default='v3', init=False)
    site: str = field(default='sampler', init=False)

    def sampler_kwargs(self, n, rho_bar):
        return {'noise_coupling': self.severity * self.max_coupling}


@dataclass
class AnisotropicTransitions(Violation):
    """V4: per-coordinate ``rho`` fans out around the declared mean.

    The one violation with a published, quantitative boundary: recovery is
    predicted to fail once ``min rho_alpha <= (max rho_alpha)^2``, at which
    point the encoder can substitute the second Hermite function of a fast
    coordinate for a slow one. :meth:`crosses_boundary` reports whether a
    given severity is past it, which is what makes the V4 calibration a
    reproduction rather than an exploration.

    Severity is a fraction of the **feasible** fan, not an absolute spread.
    That matters more than it sounds: a fan about ``rho_bar = 0.9`` can only
    reach half-width ``0.099`` before a coordinate leaves ``(0, 1)``, so an
    absolute ladder of ``0.35`` would clamp at severity 0.28 and hand three of
    its five rungs identical configurations -- 60% of the sweep's compute
    spent re-running one point, with no resolution anywhere near the boundary
    it exists to locate.
    """

    #: Fraction of the feasible ceiling reached at severity 1.
    max_spread_frac: float = 1.0
    key: str = field(default='v4', init=False)
    site: str = field(default='sampler', init=False)

    def spread(self, rho_bar):
        """Absolute fan half-width at this severity."""
        ceiling, _ = feasible_spread(rho_bar, np.inf)
        return self.severity * self.max_spread_frac * ceiling

    def rho_vector(self, n, rho_bar):
        return spread_rho(rho_bar, self.spread(rho_bar), n)

    def sampler_kwargs(self, n, rho_bar):
        return {'rho': self.rho_vector(n, rho_bar)}

    def crosses_boundary(self, n, rho_bar):
        """Whether this severity puts the sweep past the Table-2 boundary."""
        return not isotropy_criterion(
            self.rho_vector(n, rho_bar)
        )['satisfied']

    def critical_severity(self, n, rho_bar, resolution=4001):
        """Smallest severity on ``[0, 1]`` that crosses the boundary.

        The V4 calibration's target: the sign flip in ``R^2(z->h)`` should land
        here. ``None`` means the ladder never crosses it -- itself worth
        knowing *before* committing compute to the sweep.
        """
        for s in np.linspace(0.0, 1.0, resolution):
            probe = AnisotropicTransitions(
                severity=float(s), max_spread_frac=self.max_spread_frac
            )
            if probe.crosses_boundary(n, rho_bar):
                return float(s)
        return None

    def calibration_ladder(self, n, rho_bar, points=9, window=0.6):
        """A ladder that brackets the boundary instead of straddling it coarsely.

        Phase 6 has to show the sign flip *lands on* the criterion, which needs
        rungs on both sides and close to it. A uniform ladder cannot do that:
        at ``n = 9``, ``rho_bar = 0.9`` the crossing sits near severity 0.32,
        so a ``(0, .25, .5, .75, 1)`` ladder brackets it with exactly one point
        either side and nothing nearby.

        Args:
            n: Number of latent coordinates.
            rho_bar: Frozen mean autocorrelation.
            points: Rungs to return, including the endpoints.
            window: Half-width of the bracket, as a fraction of the critical
                severity.

        Returns:
            tuple: Severities, ascending, always including 0.0.
        """
        critical = self.critical_severity(n, rho_bar)
        if critical is None:
            return DEFAULT_LADDER
        lo = max(0.0, critical * (1.0 - window))
        hi = min(1.0, critical * (1.0 + window))
        bracket = np.linspace(lo, hi, max(2, points - 2))
        return tuple(
            sorted({0.0, *(round(float(s), 6) for s in bracket), 1.0})
        )

    def describe(self):
        return {
            **super().describe(),
            'max_spread_frac': self.max_spread_frac,
        }


@dataclass
class SupportTruncation(Violation):
    """V5: samples outside a feasible region are rejected.

    The stage-A induced baseline is **zero by construction**: lifted,
    table-clipped and interpenetrating cubes are all deliberately allowed, so
    nothing is rejected until this violation switches rejection on. That is a
    design decision recorded here, not an accident to be rediscovered.
    """

    min_radius_sigma: float = 1.5
    key: str = field(default='v5', init=False)
    site: str = field(default='sampler', init=False)

    def sampler_kwargs(self, n, rho_bar):
        if self.severity == 0.0:
            return {'truncation': None}
        # Severity tightens the admissible ball from "essentially everything"
        # down to `min_radius_sigma` standard deviations. Expressed in units of
        # sqrt(n) so the same severity means the same *fraction* rejected
        # regardless of how many latents the profile declares -- the norm of an
        # n-dimensional standard normal concentrates at sqrt(n).
        full = 4.0 * np.sqrt(n)
        tight = self.min_radius_sigma * np.sqrt(n)
        return {
            'truncation': Truncation(
                radius=float(full + self.severity * (tight - full)),
                strength=1.0,
            )
        }


@dataclass
class Occlusion(Violation):
    """V6: parts of the scene stop reaching the camera.

    The only violation with a **non-zero induced baseline that must be measured
    before sweeping**. Severity widens the camera-angle range and adds floor
    distractors; both push content out of frame or behind something else.

    Unlike V1-V5, this one binds at the environment, so it changes what the
    renderer does rather than what the sampler draws.
    """

    max_camera_delta: float = 10.0
    max_digits: int = 4
    key: str = field(default='v6', init=False)
    site: str = field(default='env', init=False)

    def env_kwargs(self):
        return {
            'num_digits': int(
                round(1 + self.severity * (self.max_digits - 1))
            ),
            'camera_delta_scale': float(self.severity),
        }

    @staticmethod
    def baseline_note():
        """What is already occluded at severity zero.

        Measured on the stock configuration; see the marker signal table in
        ``envs/ogbench/lejepa_cube_env.py``.
        """
        return (
            'Non-zero at severity 0. The yaw marker on the top face is '
            'foreshortened by the oblique front_pixels camera and contributes '
            'only ~48 differing pixels between yaws pi/2 apart, so the '
            'quadrant of cube.yaw rests on a weak signal before any occlusion '
            'is added. A single floor digit can also overlap the cube. Measure '
            'this floor before attributing any V6 degradation to severity.'
        )


@dataclass
class LatentDependence(Violation):
    """V9: the coordinates of ``z`` stop being independent."""

    max_corr: float = 0.6
    key: str = field(default='v9', init=False)
    site: str = field(default='sampler', init=False)

    def sampler_kwargs(self, n, rho_bar):
        # A constant-correlation matrix is only positive definite for
        # correlations above -1/(n-1) and below 1, so the ladder is capped
        # short of the singular end rather than allowed to walk into it.
        cap = min(self.max_corr, 0.95)
        return {'cross_corr': float(self.severity * cap)}


@dataclass
class DimensionMisspecification(Violation):
    """V7: the embedding width stops matching ``n``.

    Binds at the **training config**, so the whole ladder reuses one encoder
    dataset. ``V7a`` under-parametrises (``m = n - k``), ``V7b`` over-
    parametrises (``m = n + k``).
    """

    mode: str = 'under'
    max_offset_frac: float = 0.5
    key: str = field(default='v7', init=False)
    site: str = field(default='training', init=False)

    def output_dim(self, n):
        offset = int(round(self.severity * self.max_offset_frac * n))
        if self.mode == 'under':
            return max(1, n - offset)
        if self.mode == 'over':
            return n + offset
        raise ValueError(
            f"mode must be 'under' or 'over'; got {self.mode!r}."
        )

    def training_overrides(self, n):
        return {'model.head.output_dim': self.output_dim(n)}

    def describe(self):
        return {**super().describe(), 'mode': self.mode}


@dataclass
class OptimisationGap(Violation):
    """V8: training stops before the objective is minimised.

    Binds at the training config. Severity *shortens* the budget, so severity
    zero is the full schedule. This is the violation whose cost the bound
    ``D + (eps + D)^2`` predicts directly, which makes it the sharpest test of
    the theory in the whole programme.
    """

    full_epochs: int = 100
    min_epochs: int = 5
    key: str = field(default='v8', init=False)
    site: str = field(default='training', init=False)

    def max_epochs(self):
        return int(
            round(
                self.full_epochs
                + self.severity * (self.min_epochs - self.full_epochs)
            )
        )

    def training_overrides(self, n):
        return {'trainer.max_epochs': self.max_epochs()}

    def describe(self):
        return {**super().describe(), 'max_epochs': self.max_epochs()}


VIOLATIONS = {
    'none': NoViolation,
    'v1': NonGaussianMarginal,
    'v2': NonStationary,
    'v3': StateDependentNoise,
    'v4': AnisotropicTransitions,
    'v5': SupportTruncation,
    'v6': Occlusion,
    'v7': DimensionMisspecification,
    'v8': OptimisationGap,
    'v9': LatentDependence,
}


def make_violation(name, severity=0.0, **kwargs):
    """Build a violation by name.

    Args:
        name: A key of :data:`VIOLATIONS`.
        severity: Severity in ``[0, 1]``.
        **kwargs: Violation-specific overrides.

    Returns:
        Violation: The configured violation.

    Raises:
        KeyError: If ``name`` is not a known violation.
    """
    if name not in VIOLATIONS:
        raise KeyError(
            f'unknown violation {name!r}; expected one of '
            f'{sorted(VIOLATIONS)}.'
        )
    return VIOLATIONS[name](severity=severity, **kwargs)


def budget(seeds_quantitative=5, seeds_other=3):
    """The frozen seed x severity budget, as plain data.

    Declared once and frozen before Env 1. Returned as data rather than read
    from a config so that a run which deviates from it is a visible, first-class
    event in the scatter rather than a silent one.

    Returns:
        dict: Violation key -> ``{'severities': (...), 'seeds': int}``.
    """
    out = {}
    for key, cls in VIOLATIONS.items():
        if key == 'none':
            continue
        probe = cls(severity=0.0)
        out[key] = {
            'severities': probe.ladder(),
            'seeds': (
                seeds_quantitative if key in QUANTITATIVE else seeds_other
            ),
        }
    return out


def total_runs(budget_spec=None):
    """How many encoder runs the budget implies. The compute envelope, in one number."""
    budget_spec = budget_spec or budget()
    return sum(
        len(spec['severities']) * spec['seeds']
        for spec in budget_spec.values()
    )


__all__ = [
    'COARSE_LADDER',
    'DEFAULT_LADDER',
    'QUANTITATIVE',
    'VIOLATIONS',
    'AnisotropicTransitions',
    'DimensionMisspecification',
    'LatentDependence',
    'NoViolation',
    'NonGaussianMarginal',
    'NonStationary',
    'Occlusion',
    'OptimisationGap',
    'StateDependentNoise',
    'SupportTruncation',
    'Violation',
    'budget',
    'make_violation',
    'total_runs',
]

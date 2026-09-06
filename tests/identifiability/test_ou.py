"""Tests for the OU sampler.

The sampler is the encoder dataset's ground truth, so an error here does not
produce a crash -- it produces a dataset that quietly measures something other
than what its manifest claims. These tests are therefore about *distributional*
properties, not smoke.
"""

import numpy as np
import pytest

from stable_worldmodel.identifiability.ou import (
    Drift,
    OUSampler,
    Truncation,
    gennorm_unit_var_scale,
    isotropy_criterion,
    sample_marginal,
    spectral_gap,
    spread_rho,
)


BATCH = 200_000


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------- marginals


@pytest.mark.parametrize('dist', ['gaussian', 'laplace'])
def test_marginals_are_unit_variance(dist, rng):
    x = sample_marginal((BATCH, 4), dist, rng)
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose(x.var(axis=0), 1.0, rtol=0.03)


@pytest.mark.parametrize('alpha', [0.5, 1.0, 1.5, 2.0, 4.0, 8.0])
def test_gennorm_is_unit_variance_across_the_v1_ladder(alpha, rng):
    """Unit variance is what makes the V1 sweep a *shape* sweep.

    If variance moved with alpha, the whitening error would change for a
    reason that has nothing to do with Gaussianity, and every V1 result would
    confound the two.
    """
    x = sample_marginal((BATCH, 3), 'gennorm', rng, alpha=alpha)
    np.testing.assert_allclose(x.var(axis=0), 1.0, rtol=0.06)
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=0.03)


def test_gennorm_at_alpha_two_is_gaussian(rng):
    """alpha = 2 must reproduce the Gaussian, or the ladder has no origin."""
    a = sample_marginal((BATCH, 1), 'gennorm', rng, alpha=2.0).ravel()
    b = sample_marginal((BATCH, 1), 'gaussian', rng).ravel()
    for q in (1, 5, 25, 50, 75, 95, 99):
        assert np.percentile(a, q) == pytest.approx(
            np.percentile(b, q), abs=0.05
        )


def test_gennorm_kurtosis_is_monotone_in_alpha(rng):
    """Smaller alpha means heavier tails -- the direction V1's ladder assumes."""
    kurtoses = []
    for alpha in (0.75, 1.0, 2.0, 4.0, 8.0):
        x = sample_marginal((BATCH, 1), 'gennorm', rng, alpha=alpha).ravel()
        kurtoses.append(float((x**4).mean() / (x**2).mean() ** 2))
    assert all(a > b for a, b in zip(kurtoses, kurtoses[1:])), (
        f'kurtosis not decreasing in alpha: {kurtoses}'
    )


def test_gennorm_scale_matches_the_reference_formula():
    """Guards the exact constant the paper's Fig. 4b / 7-8 depend on."""
    for alpha in (0.5, 1.0, 2.0, 5.0):
        from math import lgamma, exp

        expected = exp(0.5 * (lgamma(1 / alpha) - lgamma(3 / alpha)))
        assert gennorm_unit_var_scale(alpha) == pytest.approx(expected)


# ------------------------------------------------------------ the OU step


def test_pairs_recover_the_declared_rho():
    """The whole point: the realised lag-1 correlation is what was asked for."""
    sampler = OUSampler(n=6, rho=0.9, seed=0)
    z, z_next, _ = sampler.sample_pairs(BATCH)
    np.testing.assert_allclose(
        sampler.empirical_rho(z, z_next), 0.9, atol=0.01
    )


def test_the_step_is_stationary():
    """``z`` and ``z'`` must share a marginal, for every rho.

    Stationarity is what Thm 1 assumes. A step that changed the marginal would
    make every downstream metric a measurement of the drift instead.
    """
    for rho in (0.5, 0.8, 0.9, 0.99):
        sampler = OUSampler(n=4, rho=rho, seed=1)
        z, z_next, _ = sampler.sample_pairs(BATCH)
        np.testing.assert_allclose(z.var(axis=0), 1.0, rtol=0.05)
        np.testing.assert_allclose(z_next.var(axis=0), 1.0, rtol=0.05)
        np.testing.assert_allclose(
            z.mean(axis=0), z_next.mean(axis=0), atol=0.02
        )


def test_per_dimension_rho_is_first_class():
    """V4 needs each coordinate to decorrelate at its own rate."""
    rho = np.linspace(0.6, 0.95, 8)
    sampler = OUSampler(n=8, rho=rho, seed=2)
    z, z_next, _ = sampler.sample_pairs(BATCH)
    np.testing.assert_allclose(
        sampler.empirical_rho(z, z_next), rho, atol=0.012
    )


def test_rho_must_be_strictly_inside_the_unit_interval():
    """rho = 0 destroys the pair; rho = 1 makes the views identical."""
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match='rho'):
            OUSampler(n=3, rho=bad, seed=0)


# ---------------------------------------------------------------- V4 knobs


def test_spread_rho_preserves_the_declared_mean():
    """Severity and the program constant must stay independent.

    If spreading also moved the mean, a V4 sweep would silently be a joint
    sweep over rho -- and rho is a frozen program constant.
    """
    for spread in (0.0, 0.1, 0.3):
        rho = spread_rho(0.9, spread, 16)
        assert rho.mean() == pytest.approx(0.9, abs=1e-9)


def test_spread_rho_stays_inside_the_unit_interval():
    rho = spread_rho(0.9, 0.5, 32)
    assert (rho > 0).all() and (rho < 1).all()


@pytest.mark.parametrize(
    ('rho', 'expected'),
    [
        # Isotropic: min == max == rho, and rho > rho^2 for any rho in (0, 1).
        (0.9, True),
        (0.5, True),
        # min <= max^2 -- the boundary the paper's Table 2 tabulates.
        ([0.5, 0.9], False),   # 0.5 <= 0.81
        ([0.85, 0.9], True),   # 0.85 > 0.81
        ([0.81, 0.9], False),  # 0.81 <= 0.81, exactly on the boundary
        ([0.2, 0.95], False),
        # 0.995^2 = 0.990025 > 0.99, so even this narrow pair fails the
        # criterion -- the boundary bites hardest exactly where rho is high.
        ([0.99, 0.995], False),
        ([0.995, 0.996], True),  # 0.996^2 = 0.992016 < 0.995
    ],
)
def test_isotropy_criterion_matches_the_published_boundary(rho, expected):
    """``min rho_alpha > (max rho_alpha)^2``, table-driven.

    The expected verdicts are published, so this is a reproduction rather than
    a characterisation of our own implementation.
    """
    assert isotropy_criterion(rho)['satisfied'] is expected


def test_isotropy_margin_has_the_right_sign():
    assert isotropy_criterion([0.85, 0.9])['margin'] > 0
    assert isotropy_criterion([0.5, 0.9])['margin'] < 0


# -------------------------------------------------------- other violations


def test_v1_changes_the_marginal_but_not_the_correlation():
    """Each violation must move the thing it names and leave the rest alone."""
    base = OUSampler(n=4, rho=0.9, seed=3)
    heavy = OUSampler(n=4, rho=0.9, dist='gennorm', alpha=0.6, seed=3)

    zb, zb_next, _ = base.sample_pairs(BATCH)
    zh, zh_next, _ = heavy.sample_pairs(BATCH)

    np.testing.assert_allclose(
        heavy.empirical_rho(zh, zh_next), 0.9, atol=0.015
    )
    kurt = lambda x: float((x**4).mean() / (x**2).mean() ** 2)  # noqa: E731
    assert kurt(zh) > 2 * kurt(zb)


def test_v3_makes_the_innovation_heteroscedastic():
    """State-dependent noise: residual scale must correlate with |z|."""
    sampler = OUSampler(n=1, rho=0.9, noise_coupling=1.5, seed=4)
    z, z_next, _ = sampler.sample_pairs(BATCH)
    residual = np.abs(z_next - 0.9 * z).ravel()
    magnitude = np.abs(z).ravel()
    corr = np.corrcoef(residual, magnitude)[0, 1]
    assert corr > 0.1, f'no heteroscedasticity induced (corr {corr:.4f})'

    flat = OUSampler(n=1, rho=0.9, noise_coupling=0.0, seed=4)
    z, z_next, _ = flat.sample_pairs(BATCH)
    baseline = np.corrcoef(
        np.abs(z_next - 0.9 * z).ravel(), np.abs(z).ravel()
    )[0, 1]
    assert abs(baseline) < 0.02, 'baseline is already heteroscedastic'


def test_v9_correlates_the_coordinates_without_changing_their_variance():
    """V9 must be a dependence violation, not a variance one."""
    sampler = OUSampler(n=5, rho=0.9, cross_corr=0.5, seed=5)
    z, _, _ = sampler.sample_pairs(BATCH)
    np.testing.assert_allclose(z.var(axis=0), 1.0, rtol=0.05)

    corr = np.corrcoef(z.T)
    off = corr[~np.eye(5, dtype=bool)]
    assert off.mean() > 0.2, f'no dependence induced (mean {off.mean():.3f})'


def test_v9_rejects_a_correlation_with_no_square_root():
    """A constant-correlation matrix stops being positive definite below -1/(n-1)."""
    with pytest.raises(ValueError, match='cross_corr'):
        OUSampler(n=4, rho=0.9, cross_corr=-0.9, seed=0)


def test_v5_truncation_bounds_the_support():
    sampler = OUSampler(
        n=3, rho=0.9, truncation=Truncation(radius=2.0), seed=6
    )
    z, z_next, _ = sampler.sample_pairs(20_000)
    assert np.linalg.norm(z, axis=1).max() <= 2.0 + 1e-9
    assert np.linalg.norm(z_next, axis=1).max() <= 2.0 + 1e-9


def test_v5_truncation_strength_thins_rather_than_cuts():
    """Sub-unit strength must leave a tail, or severity is a switch not a dial."""
    sampler = OUSampler(
        n=2,
        rho=0.9,
        truncation=Truncation(radius=1.5, strength=0.5),
        seed=7,
    )
    z, _, _ = sampler.sample_pairs(40_000)
    outside = (np.linalg.norm(z, axis=1) > 1.5).mean()
    assert 0.0 < outside < 0.5, (
        f'expected a thinned tail, got {outside:.4f} outside the radius'
    )


def test_v2_drift_breaks_stationarity_within_a_pseudo_episode():
    """Early and late pairs of an episode must differ in distribution."""
    period = 200
    sampler = OUSampler(
        n=2,
        rho=0.9,
        drift=Drift(mean=1.5, scale=0.0, rho=0.0, period=period),
        seed=8,
    )
    z, _, _ = sampler.sample_pairs(period * 200)
    phase = np.arange(len(z)) % period
    early = z[phase < period // 8].mean()
    late = z[phase >= 7 * period // 8].mean()
    assert late - early > 0.5, (
        f'drift did not shift the mean (early {early:.3f}, late {late:.3f})'
    )


def test_no_violation_leaves_the_process_ideal():
    """Severity zero must be genuinely clean, not merely small."""
    sampler = OUSampler(n=4, rho=0.9, seed=9)
    z, z_next, _ = sampler.sample_pairs(BATCH)

    np.testing.assert_allclose(z.var(axis=0), 1.0, rtol=0.03)
    corr = np.corrcoef(z.T)
    off = corr[~np.eye(4, dtype=bool)]
    assert np.abs(off).max() < 0.02
    assert isotropy_criterion(sampler.rho)['satisfied']


# ------------------------------------------------------------------ units


def test_spectral_gap():
    assert spectral_gap(0.9) == pytest.approx(2 * 0.9 * 0.1)
    assert spectral_gap([0.8, 1.0 - 1e-9]) == pytest.approx(
        spectral_gap(0.9), abs=1e-6
    )


def test_describe_round_trips_the_configuration():
    """The manifest must be able to distinguish two different datasets."""
    a = OUSampler(n=4, rho=0.9, seed=1).describe()
    b = OUSampler(n=4, rho=0.9, dist='gennorm', alpha=1.0, seed=1).describe()
    assert a != b
    assert a['isotropy']['satisfied']
    assert a['rho_mean'] == pytest.approx(0.9)


def test_sampler_is_reproducible_from_its_seed():
    a, a_next, _ = OUSampler(n=5, rho=0.9, seed=42).sample_pairs(1000)
    b, b_next, _ = OUSampler(n=5, rho=0.9, seed=42).sample_pairs(1000)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a_next, b_next)

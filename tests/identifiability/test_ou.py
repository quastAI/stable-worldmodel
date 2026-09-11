"""Tests for the OU sampler.

The sampler's contract is short now that it is isotropic and Gaussian by
construction: the pair is a *stationary* transition, and the marginal it
advertises is the marginal it draws. Stationarity is the property the
identifiability theorem needs, so it is asserted rather than assumed.
"""

import numpy as np
import pytest

from stable_worldmodel.identifiability.ou import (
    SIGMA_SPAN,
    OUSampler,
    spectral_gap,
)


BATCH = 40000


@pytest.fixture
def sampler():
    return OUSampler(n=6, rho=0.9, seed=0)


def test_marginal_is_standard_normal(sampler):
    z = sampler.sample_marginal_batch(BATCH)
    assert np.abs(z.mean(axis=0)).max() < 0.03
    assert np.abs(z.var(axis=0) - 1.0).max() < 0.05


def test_both_views_share_the_marginal(sampler):
    """Stationarity: ``z'`` has the same law as ``z``, which is what Thm 1 needs."""
    z, z_next = sampler.sample_pairs(BATCH)
    assert np.abs(z_next.mean(axis=0)).max() < 0.03
    assert np.abs(z_next.var(axis=0) - 1.0).max() < 0.05
    assert np.abs(z.var(axis=0) - z_next.var(axis=0)).max() < 0.05


def test_pair_correlation_is_the_declared_rho(sampler):
    z, z_next = sampler.sample_pairs(BATCH)
    achieved = sampler.empirical_rho(z, z_next)
    assert np.abs(achieved - 0.9).max() < 0.02


@pytest.mark.parametrize('rho', [0.5, 0.9, 0.99])
def test_correlation_tracks_rho_across_the_range(rho):
    z, z_next = OUSampler(n=4, rho=rho, seed=1).sample_pairs(BATCH)
    achieved = OUSampler(n=4, rho=rho).empirical_rho(z, z_next)
    assert np.abs(achieved - rho).max() < 0.02


def test_coordinates_are_independent(sampler):
    """No cross-coordinate structure: the correlation matrix is the identity."""
    z, _ = sampler.sample_pairs(BATCH)
    corr = np.corrcoef(z.T)
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.03


def test_seed_reproduces_the_stream():
    a = OUSampler(n=5, rho=0.9, seed=7).sample_pairs(256)
    b = OUSampler(n=5, rho=0.9, seed=7).sample_pairs(256)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_different_seeds_differ():
    a, _ = OUSampler(n=5, rho=0.9, seed=1).sample_pairs(256)
    b, _ = OUSampler(n=5, rho=0.9, seed=2).sample_pairs(256)
    assert not np.array_equal(a, b)


def test_successive_draws_advance_the_stream(sampler):
    """Two calls must not return the same pairs."""
    first, _ = sampler.sample_pairs(128)
    second, _ = sampler.sample_pairs(128)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize('rho', [0.0, 1.0, -0.1, 1.5])
def test_rho_outside_the_open_interval_is_refused(rho):
    """``rho = 0`` destroys the pair correlation; ``rho = 1`` makes the views identical."""
    with pytest.raises(ValueError, match='rho'):
        OUSampler(n=3, rho=rho)


def test_non_positive_width_is_refused():
    with pytest.raises(ValueError, match='n must be positive'):
        OUSampler(n=0)


def test_describe_records_every_field_that_changes_a_sample(sampler):
    described = sampler.describe()
    assert described == {
        'n': 6,
        'rho': 0.9,
        'dist': 'gaussian',
        'seed': 0,
    }


def test_describe_separates_differently_configured_samplers():
    assert (
        OUSampler(n=6, rho=0.9, seed=0).describe()
        != OUSampler(n=6, rho=0.8, seed=0).describe()
    )


@pytest.mark.parametrize(
    'rho,expected', [(0.5, 0.5), (0.9, 0.18000000000000002)]
)
def test_spectral_gap_is_two_rho_one_minus_rho(rho, expected):
    assert spectral_gap(rho) == pytest.approx(expected)


def test_sigma_span_is_three():
    """The registry's affine reaches the declared bounds at +-3 sigma."""
    assert SIGMA_SPAN == 3.0

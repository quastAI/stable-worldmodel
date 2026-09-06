"""Tests for the violation ladders.

Two properties matter and neither is obvious from reading the dataclasses:

1. **Each violation moves the thing it names and nothing else.** A V1 rung that
   also changed ``rho`` would confound every V1 result with a V4 result.
2. **Each ladder actually spans something.** A ladder whose rungs collapse onto
   one configuration spends its compute re-running a single point -- and would
   look, in the scatter, like a violation with no effect.
"""

import numpy as np
import pytest

from stable_worldmodel.identifiability.ou import (
    OUSampler,
    isotropy_criterion,
)
from stable_worldmodel.identifiability.violations import (
    QUANTITATIVE,
    VIOLATIONS,
    AnisotropicTransitions,
    DimensionMisspecification,
    NoViolation,
    Occlusion,
    OptimisationGap,
    budget,
    make_violation,
    total_runs,
)


N = 9
RHO_BAR = 0.9
BATCH = 60_000


def sampler_for(violation, seed=0):
    kwargs = violation.sampler_kwargs(N, RHO_BAR)
    kwargs.setdefault('rho', RHO_BAR)
    return OUSampler(n=N, seed=seed, **kwargs)


# --------------------------------------------------------------- the ladders


@pytest.mark.parametrize('key', sorted(set(VIOLATIONS) - {'none'}))
def test_every_ladder_spans_distinct_configurations(key):
    """A rung that duplicates its neighbour is wasted compute.

    This is not hypothetical: V4's original absolute-spread parametrisation
    clamped at severity 0.28, so three of its five rungs were the same
    configuration.
    """
    violation = make_violation(key)
    seen = []
    for severity in violation.ladder():
        rung = make_violation(key, severity=severity)
        seen.append(
            repr(
                (
                    rung.sampler_kwargs(N, RHO_BAR),
                    rung.env_kwargs(),
                    rung.training_overrides(N),
                )
            )
        )
    assert len(set(seen)) == len(seen), (
        f'{key}: ladder {violation.ladder()} has duplicate rungs'
    )


@pytest.mark.parametrize('key', sorted(set(VIOLATIONS) - {'none'}))
def test_severity_zero_is_the_clean_configuration(key):
    """Every sweep's control rung must genuinely be the unviolated process."""
    rung = make_violation(key, severity=0.0)
    if rung.site != 'sampler':
        return
    sampler = sampler_for(rung)
    z, z_next, _ = sampler.sample_pairs(BATCH)

    np.testing.assert_allclose(z.var(axis=0), 1.0, rtol=0.05)
    np.testing.assert_allclose(
        sampler.empirical_rho(z, z_next), RHO_BAR, atol=0.02
    )
    assert isotropy_criterion(sampler.rho)['satisfied']


@pytest.mark.parametrize('key', sorted(VIOLATIONS))
def test_severity_is_bounded(key):
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match='severity'):
            make_violation(key, severity=bad)


def test_unknown_violation_is_refused():
    with pytest.raises(KeyError, match='unknown violation'):
        make_violation('v42')


# --------------------------------------------------- one knob at a time


@pytest.mark.parametrize('key', ['v1', 'v2', 'v3', 'v5', 'v9'])
def test_sampler_violations_leave_rho_alone(key):
    """``rho`` is a frozen program constant; only V4 may touch it."""
    rung = make_violation(key, severity=1.0)
    assert 'rho' not in rung.sampler_kwargs(N, RHO_BAR)


def test_v1_moves_only_the_marginal_shape():
    rung = make_violation('v1', severity=1.0)
    kwargs = rung.sampler_kwargs(N, RHO_BAR)
    assert set(kwargs) == {'dist', 'alpha'}
    assert kwargs['dist'] == 'gennorm'

    sampler = sampler_for(rung)
    z, z_next, _ = sampler.sample_pairs(BATCH)
    np.testing.assert_allclose(
        sampler.empirical_rho(z, z_next), RHO_BAR, atol=0.02
    )
    np.testing.assert_allclose(z.var(axis=0), 1.0, rtol=0.08)


def test_v1_alpha_is_monotone_and_brackets_gaussian():
    heavy = [
        make_violation('v1', severity=s, direction='heavy').alpha()
        for s in (0.0, 0.5, 1.0)
    ]
    light = [
        make_violation('v1', severity=s, direction='light').alpha()
        for s in (0.0, 0.5, 1.0)
    ]
    assert heavy[0] == 2.0 and light[0] == 2.0
    assert heavy[0] > heavy[1] > heavy[2]
    assert light[0] < light[1] < light[2]


def test_v4_is_the_only_one_that_sets_rho():
    rung = make_violation('v4', severity=1.0)
    assert 'rho' in rung.sampler_kwargs(N, RHO_BAR)


def test_v4_preserves_the_frozen_rho_mean():
    """Severity must not smuggle in a change to the program constant."""
    for severity in AnisotropicTransitions().ladder():
        rho = AnisotropicTransitions(severity=severity).rho_vector(N, RHO_BAR)
        assert rho.mean() == pytest.approx(RHO_BAR, abs=1e-9)


# ------------------------------------------------------- the V4 calibration


def test_v4_ladder_crosses_the_published_boundary():
    """Phase 6 cannot reproduce a boundary its ladder never reaches."""
    violation = AnisotropicTransitions()
    critical = violation.critical_severity(N, RHO_BAR)
    assert critical is not None, 'V4 ladder never crosses the criterion'
    assert 0.0 < critical < 1.0


def test_v4_calibration_ladder_brackets_the_boundary():
    """Rungs on both sides, and close to it -- otherwise the flip is a cliff."""
    violation = AnisotropicTransitions()
    critical = violation.critical_severity(N, RHO_BAR)
    ladder = violation.calibration_ladder(N, RHO_BAR)

    below = [s for s in ladder if s < critical]
    above = [s for s in ladder if s > critical]
    assert len(below) >= 2 and len(above) >= 2, (
        f'ladder {ladder} does not bracket critical severity {critical}'
    )
    nearest = min(abs(s - critical) for s in ladder)
    assert nearest < 0.1 * critical, (
        f'no rung within 10% of the boundary (nearest {nearest:.4f})'
    )


def test_v4_crosses_boundary_agrees_with_the_criterion():
    """The convenience predicate must not drift from the criterion itself."""
    for severity in np.linspace(0, 1, 21):
        rung = AnisotropicTransitions(severity=float(severity))
        expected = not isotropy_criterion(
            rung.rho_vector(N, RHO_BAR)
        )['satisfied']
        assert rung.crosses_boundary(N, RHO_BAR) is expected


# --------------------------------------------------------- non-sampler sites


def test_v6_binds_at_the_environment_not_the_sampler():
    rung = make_violation('v6', severity=1.0)
    assert rung.site == 'env'
    assert rung.sampler_kwargs(N, RHO_BAR) == {}
    assert rung.env_kwargs()['num_digits'] > 1


def test_v6_records_a_non_zero_induced_baseline():
    """The one violation whose severity-zero cost is not zero.

    Sweeping on top of an unmeasured floor attributes the floor to the sweep.
    """
    note = Occlusion.baseline_note()
    assert 'non-zero' in note.lower()
    assert 'measure' in note.lower()


@pytest.mark.parametrize('key', ['v7', 'v8'])
def test_training_violations_touch_no_dataset(key):
    """V7 and V8 reuse one encoder dataset across their whole ladder."""
    rung = make_violation(key, severity=1.0)
    assert rung.site == 'training'
    assert rung.sampler_kwargs(N, RHO_BAR) == {}
    assert rung.env_kwargs() == {}
    assert rung.training_overrides(N)


def test_v7_under_and_over_parametrise_around_n():
    under = DimensionMisspecification(severity=1.0, mode='under')
    over = DimensionMisspecification(severity=1.0, mode='over')
    assert under.output_dim(N) < N < over.output_dim(N)
    assert DimensionMisspecification(severity=0.0).output_dim(N) == N
    assert under.output_dim(N) >= 1


def test_v8_shortens_the_budget_monotonically():
    epochs = [
        OptimisationGap(severity=s).max_epochs() for s in (0.0, 0.5, 1.0)
    ]
    assert epochs[0] > epochs[1] > epochs[2] >= 1


# ------------------------------------------------------------------- budget


def test_budget_gives_quantitative_violations_more_resolution():
    """V1, V4 and V8 are the three with predictions that can be falsified."""
    spec = budget()
    for key in QUANTITATIVE:
        for other in set(spec) - set(QUANTITATIVE):
            assert spec[key]['seeds'] >= spec[other]['seeds']
            assert len(spec[key]['severities']) >= len(
                spec[other]['severities']
            )


def test_budget_covers_every_violation():
    assert set(budget()) == set(VIOLATIONS) - {'none'}


def test_total_runs_is_a_single_number():
    """The compute envelope must be knowable before the sweep starts."""
    total = total_runs()
    assert total > 0
    assert total == sum(
        len(s['severities']) * s['seeds'] for s in budget().values()
    )


def test_no_violation_is_inert():
    rung = NoViolation()
    assert rung.sampler_kwargs(N, RHO_BAR) == {}
    assert rung.env_kwargs() == {}
    assert rung.training_overrides(N) == {}

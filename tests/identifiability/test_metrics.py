"""Tests for the frozen metric suite.

The plan (SS7.1) demands two adversarial tests be written **before** any real
encoder run, because both failure modes are invisible on real data: a suite
that rejects correct encoders looks like an encoder problem, and a decoy metric
that silently tracks the criterion looks like corroboration. They come first
here for the same reason.
"""

import numpy as np
import pytest

from stable_worldmodel.identifiability.metrics import (
    METRIC_SUITE_VERSION,
    alignment_gap,
    bidirectional_r2,
    compute_all,
    hermite2_substitution,
    in_gap_units,
    mcc_unaligned,
    monotone_recovery,
    orthogonality_gap,
    predicted_error,
    probe_accessibility,
    probe_divergence,
    procrustes_recovery,
    sigreg_z_score,
    style_invariance,
    whitening_error,
)


B = 4000
N = 8


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_orthogonal(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


# ====================================================================
#  The two tests the plan demands first
# ====================================================================


def test_orthogonal_map_scores_at_floor_and_need_not_score_on_mcc(rng):
    """Adversarial test 1: the suite must not reject a correct encoder.

    Theory predicts identification **up to an orthogonal transform**, so
    ``h = Qz`` for a random ``Q`` in ``O(n)`` is a *perfect* result. Procrustes
    recovery must therefore sit at floor -- and unaligned MCC must **not** be
    required to, because greedy matching admits permutations but not rotations
    and would penalise exactly this solution.

    An exit criterion built on MCC would reject correct encoders. This test is
    the guard against writing one.
    """
    z = rng.standard_normal((B, N))
    q = random_orthogonal(N, rng)
    h = z @ q.T

    recovery = procrustes_recovery(z, h)
    assert recovery['procrustes_mse_per_dim'] < 1e-12, recovery

    gap = orthogonality_gap(z, h)
    assert gap['orth_err_normalized'] < 1e-9
    assert gap['cond'] == pytest.approx(1.0, abs=1e-6)

    # ...and MCC is free to be poor. Asserting it *is* poor for a generic
    # rotation is what makes the point rather than merely allowing it.
    mcc = mcc_unaligned(z, h)
    assert mcc['mcc_unaligned'] < 0.95
    assert 'never a gate' in mcc['mcc_note']


def test_wide_embedding_is_probeable_while_failing_recovery(rng):
    """Adversarial test 2: the decoy must decouple from the criterion.

    With ``m >> n`` on restricted-support ``z``, a random wide embedding
    retains the latents linearly -- a probe reads them out easily -- while
    being nowhere near an orthogonal image of ``z``. If probe accessibility
    tracked recovery here, it would corroborate rather than contrast, and
    SS2.5's whole argument would be untestable.
    """
    n, m = 4, 96
    z = rng.uniform(-1.0, 1.0, (B, n))  # restricted support
    projection = rng.standard_normal((m, n))
    h = z @ projection.T + 0.01 * rng.standard_normal((B, m))

    probe = probe_accessibility(h, z)
    assert probe['probe_linear_r2'] > 0.95, probe

    gap = orthogonality_gap(z, h)
    assert gap['orth_err_normalized'] > 0.5, gap

    assert probe_divergence(
        probe['probe_linear_r2'], gap['orth_err_normalized']
    ) > 0.2


# ====================================================================
#  Recovery
# ====================================================================


def test_procrustes_is_invariant_to_rotation_of_h(rng):
    """A rotated embedding is the same result, so the score must not move."""
    z = rng.standard_normal((B, N))
    h = z @ random_orthogonal(N, rng).T + 0.1 * rng.standard_normal((B, N))

    base = procrustes_recovery(z, h)['procrustes_mse_per_dim']
    rotated = procrustes_recovery(
        z, h @ random_orthogonal(N, rng).T
    )['procrustes_mse_per_dim']
    assert rotated == pytest.approx(base, rel=1e-6)


def test_procrustes_degrades_with_noise(rng):
    z = rng.standard_normal((B, N))
    q = random_orthogonal(N, rng)
    errors = [
        procrustes_recovery(z, z @ q.T + noise * rng.standard_normal((B, N)))[
            'procrustes_mse_per_dim'
        ]
        for noise in (0.0, 0.1, 0.5, 1.0)
    ]
    assert all(a < b for a, b in zip(errors, errors[1:]))


def test_per_dim_normalisation_makes_widths_comparable(rng):
    """Without it, a wider profile looks worse for no reason but its width."""
    per_dim = []
    for n in (4, 16, 64):
        z = rng.standard_normal((B, n))
        h = z @ random_orthogonal(n, rng).T + 0.3 * rng.standard_normal((B, n))
        per_dim.append(
            procrustes_recovery(z, h)['procrustes_mse_per_dim']
        )
    assert max(per_dim) / min(per_dim) < 1.3, per_dim


def test_condition_number_catches_what_the_frobenius_gap_misses(rng):
    """A well-fitting but ill-conditioned map still ruins planning.

    A near-singular direction means the planner's cost is nearly flat along a
    latent that physically matters. ``cond`` must flag that.
    """
    z = rng.standard_normal((B, N))
    scale = np.ones(N)
    scale[0] = 1e-3
    h = z * scale

    gap = orthogonality_gap(z, h)
    assert gap['cond'] > 100, gap


# ====================================================================
#  Diagnostics
# ====================================================================


def test_bidirectional_r2_separates_the_two_directions(rng):
    """``h`` can contain ``z`` without being a linear image of it."""
    z = rng.standard_normal((B, 1))
    h = np.column_stack([z[:, 0] ** 2 - 1.0, rng.standard_normal(B)])

    scores = bidirectional_r2(z, h)
    assert scores['r2_z_to_h'] < 0.2
    assert scores['r2_h_to_z'] < 0.2

    linear = bidirectional_r2(z, np.column_stack([z[:, 0], z[:, 0] * 2]))
    assert linear['r2_z_to_h'] > 0.99


def test_hermite2_detects_the_v4_substitution(rng):
    """The predicted V4 failure: ``He2`` of a latent in place of the latent."""
    z = rng.standard_normal((B, 3))
    substituted = np.column_stack(
        [z[:, 0] ** 2 - 1.0, z[:, 1], z[:, 2]]
    )
    faithful = z.copy()

    bad = hermite2_substitution(z, substituted, slow_index=0)
    good = hermite2_substitution(z, faithful, slow_index=0)
    assert bad['hermite2_excess'] > good['hermite2_excess']
    assert bad['hermite2_r2'] > 0.3


def test_hermite2_picks_the_slowest_latent_from_rho(rng):
    z = rng.standard_normal((B, 3))
    result = hermite2_substitution(z, z, rho=[0.95, 0.6, 0.9])
    assert result['hermite2_slow_index'] == 1


def test_monotone_recovery_forgives_a_monotone_warp(rng):
    """Env 1's latents are not i.i.d. and some have no Gaussian marginal."""
    z = rng.standard_normal((B, 3))
    warped = np.sign(z) * np.abs(z) ** 1.7

    linear = procrustes_recovery(z, warped)['procrustes_mse_per_dim']
    monotone = monotone_recovery(z, warped)['monotone_mse_per_dim']
    assert monotone < linear


@pytest.mark.parametrize('mode', ['per_latent', 'shared'])
def test_monotone_recovery_reports_both_modes(mode, rng):
    z = rng.standard_normal((500, 3))
    result = monotone_recovery(z, z, mode=mode)
    assert result['monotone_mode'] == mode
    assert np.isfinite(result['monotone_mse_per_dim'])


# ====================================================================
#  The bound
# ====================================================================


def test_whitening_error_is_zero_for_whitened_data(rng):
    z = rng.standard_normal((40_000, 6))
    z = (z - z.mean(0)) / z.std(0)
    assert whitening_error(z) < 0.15


def test_alignment_gap_is_zero_for_an_exact_ou_step(rng):
    """A faithful embedding of the OU process has no *excess* pair distance."""
    rho = 0.9
    z = rng.standard_normal((40_000, 6))
    z_next = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal((40_000, 6))

    assert alignment_gap(z, z_next, rho) < 0.4


def test_alignment_gap_is_clamped_at_zero(rng):
    """A negative delta would produce a negative D and an unreadable bound."""
    z = rng.standard_normal((200, 4))
    assert alignment_gap(z, z, 0.9) >= 0.0


def test_predicted_error_flags_anisotropy(rng):
    """A single number standing in for a spread must say so."""
    iso = predicted_error(0.1, 0.05, 0.9)
    aniso = predicted_error(0.1, 0.05, [0.8, 0.95])
    assert iso['anisotropic'] is False
    assert aniso['anisotropic'] is True
    assert aniso['spectral_gap'] == pytest.approx(2 * 0.875 * 0.125)


def test_predicted_error_grows_with_both_terms():
    base = predicted_error(0.1, 0.05, 0.9)['predicted_error']
    assert predicted_error(0.3, 0.05, 0.9)['predicted_error'] > base
    assert predicted_error(0.1, 0.20, 0.9)['predicted_error'] > base


def test_gap_units_make_costs_comparable_across_rho():
    """Same cost, different rho, different gap -- the point of the unit."""
    assert in_gap_units(0.18, 0.9) == pytest.approx(1.0)
    assert in_gap_units(0.5, 0.5) == pytest.approx(1.0)


# ====================================================================
#  SIGReg audit
# ====================================================================


def test_sigreg_z_score_is_near_zero_for_gaussian_data(rng):
    """The floor must be recomputed at *our* sample size, not assumed."""
    result = sigreg_z_score(rng.standard_normal((5000, 6)), seed=0)
    assert abs(result['sigreg_z']) < 4.0, result


def test_sigreg_z_score_flags_a_non_gaussian_marginal(rng):
    gaussian = sigreg_z_score(rng.standard_normal((5000, 6)), seed=0)
    uniform = sigreg_z_score(rng.uniform(-1.7, 1.7, (5000, 6)), seed=0)
    assert uniform['sigreg_z'] > gaussian['sigreg_z'] + 3.0


# ====================================================================
#  Style
# ====================================================================


def test_style_invariance_is_zero_for_an_invariant_encoder(rng):
    h = rng.standard_normal((500, 8))
    result = style_invariance(h, h.copy())
    assert result['style_sensitivity'] == pytest.approx(0.0)
    assert result['style_vacuous'] is True


def test_style_invariance_grows_with_sensitivity(rng):
    h = rng.standard_normal((2000, 8))
    weak = style_invariance(h, h + 0.01 * rng.standard_normal((2000, 8)))
    strong = style_invariance(h, h + 1.0 * rng.standard_normal((2000, 8)))
    assert strong['style_sensitivity'] > 10 * weak['style_sensitivity']


# ====================================================================
#  Suite-level contracts
# ====================================================================


def test_compute_all_returns_finite_metrics(rng):
    rho = 0.9
    z = rng.standard_normal((2000, 6))
    z_next = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal((2000, 6))
    q = random_orthogonal(6, rng)

    out = compute_all(z, z @ q.T, z_next, z_next @ q.T, rho=rho)
    for key, value in out.items():
        if isinstance(value, float):
            assert np.isfinite(value), f'{key} is {value}'


def test_compute_all_records_the_suite_version(rng):
    """A metric that moves mid-programme must at least become visible."""
    z = rng.standard_normal((500, 4))
    out = compute_all(z, z, rho=0.9)
    assert out['metric_suite_version'] == METRIC_SUITE_VERSION


def test_perfect_recovery_beats_a_random_encoder(rng):
    """The end-to-end sanity check: the criterion orders the obvious cases."""
    z = rng.standard_normal((2000, 6))
    good = compute_all(z, z @ random_orthogonal(6, rng).T, rho=0.9)
    bad = compute_all(z, rng.standard_normal((2000, 6)), rho=0.9)

    assert good['procrustes_mse_per_dim'] < bad['procrustes_mse_per_dim']
    assert good['orth_err_normalized'] < bad['orth_err_normalized']
    assert good['r2_h_to_z'] > bad['r2_h_to_z']

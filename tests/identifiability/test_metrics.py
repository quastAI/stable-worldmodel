"""Tests for the frozen metric suite.

Organised around the three things the suite has to get right, because each
failed in a way that cost real time:

**The identities.** ``procrustes_mse_per_dim`` and the probe R^2 are two
summaries of one canonical spectrum, and the bound's ``D`` has a floor that is
a plain function of them. Those relations are asserted, not trusted: they are
what lets a pair of reported numbers be inverted into "how many directions did
it actually find", and a silent drift in either would make every historical row
uninterpretable.

**The admissibility gate.** ``delta`` below its Hermite floor is impossible for
any style-invariant encoder, so a row reporting it was measured on embeddings
that are not a function of one frame. This is the check that caught a 20x
discrepancy between train-mode and eval-mode diagnostics, and it has to fire on
that case and stay quiet on honest ones.

**The probes.** The linear/non-linear pair only means something if the
non-linear rung is neither weaker than linear (it was, as a random-feature
ridge) nor able to manufacture signal that is not there (it did, at -0.24 R^2,
before best-epoch selection).
"""

import numpy as np
import pytest

from stable_worldmodel.identifiability import metrics
from stable_worldmodel.identifiability.metrics import (
    DEAD_LATENT_R2,
    alignment_floor,
    alignment_gap,
    bound_reach,
    canonical_correlations,
    compute_all,
    delta_admissibility,
    observability_ceiling,
    orthogonality_gap,
    predicted_error,
    probe_accessibility,
    procrustes_recovery,
    r2,
    spectrum,
    split_alignment_gap,
    style_invariance,
    style_variance,
)


N = 10
RHO = 0.9
SAMPLES = 20000


def he2(x):
    """Second Hermite function, normalised to unit variance."""
    return (x**2 - 1.0) / np.sqrt(2.0)


@pytest.fixture
def pair():
    """``(z, z_next)``: one OU step at ``rho``, standard normal marginal."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((SAMPLES, N))
    z_next = RHO * z + np.sqrt(1.0 - RHO**2) * rng.standard_normal(
        (SAMPLES, N)
    )
    return z, z_next


def mixed(z, n_linear):
    """``n_linear`` coordinates carried linearly, the rest as ``He2``.

    The canonical "partially recovered" encoder: linearly readable in
    ``n_linear`` directions, and filling the remaining output dimensions with
    degree-2 functions of the latents it did not keep. Isotropic either way, so
    whitening looks finished while most of ``z`` is not recoverable.
    """
    return np.concatenate([z[:, :n_linear], he2(z[:, n_linear:])], axis=1)


# ------------------------------------------------------------- identities


def test_procrustes_decodes_from_trace_and_scale(pair):
    """``mse_per_dim = (tr Cov(h) + n - 2 sum sigma) / n``.

    The identity that makes the metric readable. It is also why the metric is
    not a fraction-recovered: the decode has ``tr Cov(h)`` in it.
    """
    z, _ = pair
    h = mixed(z, 4)
    result = procrustes_recovery(z, h)
    # Empirical traces, not the nominal n: z is unit-variance *by
    # construction* but not exactly so in any finite sample, and the identity
    # is about what was measured.
    expected = (
        spectrum(h)['trace_cov']
        + spectrum(z)['trace_cov']
        - 2.0 * result['procrustes_scale'] * N
    ) / N
    assert result['procrustes_mse_per_dim'] == pytest.approx(
        expected, rel=1e-3
    )


def test_zero_embedding_scores_one_not_two(pair):
    """``h = 0`` scores 1.0, so a curve starting near 1.0 is leaving the trivial encoder."""
    z, _ = pair
    assert procrustes_recovery(z, np.zeros_like(z))[
        'procrustes_mse_per_dim'
    ] == pytest.approx(1.0, rel=0.02)


def test_isotropic_but_uninformative_embedding_scores_two(pair):
    """An embedding with full variance and no information scores 2.0, not 1.0."""
    z, _ = pair
    noise = np.random.default_rng(1).standard_normal(z.shape)
    assert procrustes_recovery(z, noise)[
        'procrustes_mse_per_dim'
    ] == pytest.approx(2.0, rel=0.02)


def test_perfect_rotation_scores_zero(pair):
    """Theory identifies ``z`` only up to an orthogonal map, so ``h = Qz`` is perfect."""
    z, _ = pair
    q = np.linalg.qr(np.random.default_rng(2).standard_normal((N, N)))[0]
    assert procrustes_recovery(z, z @ q.T)['procrustes_mse_per_dim'] < 1e-9


def test_recovered_dimensions_equals_n_times_mean_probe_r2(pair):
    """``sum sigma^2`` is the same quantity the per-latent probe means."""
    z, _ = pair
    h = mixed(z, 4)
    recovered = canonical_correlations(z, h)['recovered_dimensions']
    assert recovered == pytest.approx(N * r2(h, z), rel=0.02)
    assert recovered == pytest.approx(4.0, abs=0.15)


def test_canonical_participation_counts_directions_not_strength(pair):
    """Six directions at 0.7 and three at 1.0 are different facts."""
    z, _ = pair
    strong = canonical_correlations(z, mixed(z, 3))
    spread = canonical_correlations(
        z, np.sqrt(0.5) * z + np.sqrt(0.5) * he2(z)
    )
    assert strong['canonical_participation'] == pytest.approx(3.0, abs=0.2)
    assert spread['canonical_participation'] == pytest.approx(N, abs=0.5)


def test_orthogonality_gap_is_zero_for_a_rotation(pair):
    z, _ = pair
    q = np.linalg.qr(np.random.default_rng(3).standard_normal((N, N)))[0]
    result = orthogonality_gap(z, z @ q.T)
    assert result['orth_err_normalized'] < 1e-6
    assert result['cond'] == pytest.approx(1.0, rel=1e-6)


def test_cond_blows_up_on_a_starved_direction(pair):
    """The number that matters downstream: a flat direction in the latent space."""
    z, _ = pair
    h = z.copy()
    h[:, 0] *= 1e-4
    assert orthogonality_gap(z, h)['cond'] > 1e3


# --------------------------------------------------------------- spectrum


def test_spectrum_reports_partial_collapse_that_epsilon_hides(pair):
    """One dead direction moves ``epsilon`` little and ``cov_eig_min`` to zero."""
    z, _ = pair
    h = z.copy()
    h[:, 0] = 0.0
    result = spectrum(h)
    assert result['cov_eig_min'] < 1e-9
    assert result['effective_rank'] == pytest.approx(N - 1, abs=0.3)
    # The whole point: the aggregate stays in ordinary early-training territory.
    assert result['epsilon'] == pytest.approx(1.0, abs=0.1)


def test_whitened_embedding_has_near_zero_epsilon(pair):
    z, _ = pair
    result = spectrum(z)
    assert result['epsilon'] < 0.15
    assert result['trace_cov'] == pytest.approx(N, rel=0.02)
    assert result['effective_rank'] == pytest.approx(N, abs=0.3)


def test_sigreg_z_is_small_for_a_gaussian_and_large_otherwise(pair):
    """``Cov = I`` is not isotropy: the statistic sees the higher moments."""
    z, _ = pair
    gaussian = metrics.sigreg_z_score(z, seed=0)['sigreg_z']
    # He2 of a Gaussian is whitened but strongly non-Gaussian.
    skewed = metrics.sigreg_z_score(he2(z), seed=0)['sigreg_z']
    assert abs(gaussian) < 5.0
    assert skewed > 50.0


# ------------------------------------------------------- the bound's floor


def test_delta_floor_is_the_unexplained_variance_in_gap_units(pair):
    """``delta >= 2 rho (1 - rho) (tr Cov(h) - recovered)``, so ``D >= tr - recovered``."""
    z, _ = pair
    h = mixed(z, 4)
    trace = spectrum(h)['trace_cov']
    recovered = canonical_correlations(z, h)['recovered_dimensions']
    gate = delta_admissibility(1.0, trace, recovered, RHO)
    assert gate['delta_floor'] == pytest.approx(
        2.0 * RHO * (1.0 - RHO) * (trace - recovered), rel=1e-9
    )
    assert gate['delta_floor'] / (2.0 * RHO * (1.0 - RHO)) == pytest.approx(
        trace - recovered, rel=1e-9
    )


def test_degree_two_residual_sits_exactly_at_the_floor(pair):
    """A purely degree-2 residual is the best case, so it measures degree 2."""
    z, z_next = pair
    h, h_next = mixed(z, 4), mixed(z_next, 4)
    scores = compute_all(z, h, h_next=h_next, rho=RHO, seed=0)
    assert scores['residual_hermite_degree'] == pytest.approx(2.0, abs=0.1)
    assert scores['delta_admissible']


def test_affine_encoder_has_no_floor_to_violate(pair):
    """Nothing unexplained means no residual, so the gate is trivially satisfied."""
    z, z_next = pair
    scores = compute_all(z, z.copy(), h_next=z_next.copy(), rho=RHO, seed=0)
    assert scores['recovered_dimensions'] == pytest.approx(N, rel=0.01)
    assert scores['delta'] < 0.05
    assert scores['delta_admissible']
    assert not scores['bound_vacuous']


def test_gate_fires_when_the_pair_distance_is_flattered(pair):
    """The train-mode signature: ``delta`` far below a floor it cannot be below.

    Reproduces what train-mode BatchNorm does to ``h[0] - h[1]`` -- it pulls the
    two views together without changing ``Cov(h)`` -- and the gate must call it
    impossible rather than report a small ``delta`` as a converged run.
    """
    z, z_next = pair
    h, h_next = mixed(z, 4), mixed(z_next, 4)
    flattered = h + 0.5 * (h_next - h)
    scores = compute_all(z, h, h_next=flattered, rho=RHO, seed=0)
    assert scores['delta'] < scores['delta_floor']
    assert scores['residual_hermite_degree'] < 2.0
    assert not scores['delta_admissible']


def test_gate_tolerates_sampling_noise_on_an_honest_row():
    """An exactly-degree-2 residual measures ~1.97 at 4k samples; that must pass."""
    rng = np.random.default_rng(4)
    z = rng.standard_normal((4000, N))
    z_next = RHO * z + np.sqrt(1 - RHO**2) * rng.standard_normal((4000, N))
    scores = compute_all(
        z, mixed(z, 4), h_next=mixed(z_next, 4), rho=RHO, seed=0
    )
    assert scores['delta_admissible']


def test_bound_is_vacuous_until_enough_is_recovered(pair):
    """``mean_probe_r2_needed`` is an arithmetic gate, not a target to try harder at."""
    z, z_next = pair
    scores = compute_all(
        z, mixed(z, 4), h_next=mixed(z_next, 4), rho=RHO, seed=0
    )
    assert scores['bound_vacuous']
    assert scores['predicted_error'] > N
    assert 0.6 < scores['mean_probe_r2_needed'] < 0.85
    # And the run is below it, which is *why* the bound says nothing.
    assert scores['probe_linear_r2'] < scores['mean_probe_r2_needed']


def test_bound_reach_inverts_the_vacuity_condition():
    """At the reported requirement, the implied bound lands exactly on ``n``."""
    reach = bound_reach(N, 0.25, N, RHO)
    d = reach['d_max_nonvacuous']
    assert d + (0.25 + d) ** 2 == pytest.approx(float(N), rel=1e-9)
    assert reach['recovered_dimensions_needed'] == pytest.approx(N - d)


def test_predicted_error_uses_content_not_total_delta():
    """Style leakage charged to nonlinearity is squared by the bound."""
    total, sigma_sq = 2.5, 1.0
    split = split_alignment_gap(total, RHO, sigma_sq)
    assert split['delta_content'] == pytest.approx(total - 2 * RHO * sigma_sq)
    honest = predicted_error(0.25, split['delta_content'], RHO)
    naive = predicted_error(0.25, total, RHO)
    assert naive['predicted_error'] > 2 * honest['predicted_error']


def test_alignment_gap_recovers_L_and_its_training_units(pair):
    """``L = 4 n * align_loss`` at two views -- the unit mismatch that bit."""
    z, z_next = pair
    h, h_next = mixed(z, 4), mixed(z_next, 4)
    result = alignment_gap(h, h_next, RHO)
    stacked = np.stack([h, h_next])
    align_loss = ((stacked.mean(0) - stacked) ** 2).mean()
    assert result['align_loss_equivalent'] == pytest.approx(
        result['L'] / (4 * N)
    )
    assert result['align_loss_equivalent'] == pytest.approx(
        align_loss, rel=1e-9
    )


def test_alignment_floor_scales_with_achieved_trace_not_n(pair):
    """A shrinking embedding has a lower floor; comparing against ``n`` reads it as converged."""
    shrunk = alignment_floor(N, RHO, trace_cov=5.0)
    full = alignment_floor(N, RHO, trace_cov=float(N))
    assert (
        shrunk['align_floor_deterministic'] < full['align_floor_deterministic']
    )
    assert full['align_loss_floor'] == pytest.approx((1 - RHO) / 2)


def test_style_term_enters_the_floor_with_its_own_weight():
    """``2 rho`` against the content term's ``2(1 - rho)``: 9:1 at rho = 0.9."""
    result = alignment_floor(N, RHO, trace_cov=float(N), sigma_sq=0.5)
    assert result['align_floor_style_term'] == pytest.approx(2 * RHO * 0.5)
    assert result['align_floor'] > result['align_floor_deterministic']


# ----------------------------------------------------------------- probes


def test_probes_agree_when_the_coding_is_linear(pair):
    """A non-linear probe must never be *worse* than linear on linear coding."""
    z, _ = pair
    q = np.linalg.qr(np.random.default_rng(5).standard_normal((N, N)))[0]
    result = probe_accessibility(z @ q.T, z, seed=0)
    assert result['probe_linear_r2'] > 0.99
    assert result['probe_mlp_r2'] > 0.98
    assert result['probe_gap'] > -0.05


def test_mlp_probe_finds_what_linear_cannot(pair):
    """The distinction the metric exists for: present, but not linearised."""
    z, _ = pair
    h = np.concatenate([np.tanh(2.0 * z[:, :5]), z[:, 5:]], axis=1)
    result = probe_accessibility(h, z, seed=0)
    linear = np.array(result['probe_linear_per_latent'])[:5]
    mlp = np.array(result['probe_mlp_per_latent'])[:5]
    assert (mlp > linear + 0.05).all()


def test_mlp_probe_does_not_manufacture_absent_signal(pair):
    """``He2`` destroys the sign of ``z``; no probe can recover it, so R^2 ~ 0.

    Before best-epoch selection the probe overfit and reported -0.24 here,
    which reads as "worse than knowing nothing" and is not a coherent answer.
    """
    z, _ = pair
    result = probe_accessibility(he2(z), z, seed=0)
    scores = np.array(result['probe_mlp_per_latent'])
    assert (scores < 0.05).all()
    assert (scores > -0.05).all()


def test_probe_names_make_the_row_self_describing(pair):
    z, _ = pair
    names = [f'latent.{i}' for i in range(N)]
    result = probe_accessibility(z, z, names=names, seed=0)
    assert result['probe_latent_names'] == names
    assert len(result['probe_linear_per_latent']) == N


def test_probes_are_deterministic_given_a_seed(pair):
    z, _ = pair
    h = mixed(z, 4)
    a = probe_accessibility(h, z, seed=0)
    b = probe_accessibility(h, z, seed=0)
    assert a['probe_mlp_per_latent'] == b['probe_mlp_per_latent']


# ---------------------------------------------------------------- ceiling


def test_ceiling_excludes_the_latents_the_renderer_starves(pair):
    """An aggregate over latents of wildly different observability needs its floor stated."""
    z, _ = pair
    h = mixed(z, 4)
    per_latent = probe_accessibility(h, z, seed=0)['probe_linear_per_latent']
    trace = spectrum(h)['trace_cov']
    ceiling = observability_ceiling(per_latent, trace, N)
    assert ceiling['n_dead_latents'] == 6
    assert ceiling['r2_ceiling'] == pytest.approx(0.4)
    # The assumption-free floor is a real floor: never crossed.
    measured = procrustes_recovery(z, h)['procrustes_mse_per_dim']
    assert measured >= ceiling['procrustes_floor']
    # The whitened target is tighter, and this encoder is near it -- within the
    # sampling noise that makes it a target rather than a bound.
    assert ceiling['procrustes_floor_whitened'] > ceiling['procrustes_floor']
    assert measured == pytest.approx(
        ceiling['procrustes_floor_whitened'], abs=0.05
    )


def test_ceiling_is_perfect_when_every_latent_is_recovered(pair):
    z, _ = pair
    per_latent = probe_accessibility(z, z, seed=0)['probe_linear_per_latent']
    ceiling = observability_ceiling(per_latent, float(N), N)
    assert ceiling['n_dead_latents'] == 0
    assert ceiling['r2_ceiling'] == pytest.approx(1.0)
    # With every latent live and the trace at n, both agree on zero.
    assert ceiling['procrustes_floor'] == pytest.approx(0.0, abs=1e-9)
    assert ceiling['procrustes_floor_whitened'] == pytest.approx(0.0, abs=1e-9)


def test_dead_latent_threshold_is_not_a_gate_on_anything():
    """It only decides which latents the *reported ceiling* excludes."""
    assert 0.0 < DEAD_LATENT_R2 < 0.1


# ------------------------------------------------------------------ style


def test_style_variance_is_half_the_mean_squared_difference():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((5000, N))
    xi = 0.1 * rng.standard_normal((5000, N))
    result = style_variance(a, a + xi)
    assert result['sigma_sq'] == pytest.approx(
        0.5 * ((xi) ** 2).sum(axis=1).mean(), rel=1e-9
    )
    assert result['sigma_sq_per_dim'] == pytest.approx(result['sigma_sq'] / N)


def test_style_invariance_flags_a_vacuous_probe(pair):
    """Identical views mean there was no style to resample, not perfect invariance."""
    z, _ = pair
    result = style_invariance(z, z.copy())
    assert result['style_vacuous']
    assert result['style_sensitivity'] == 0.0


def test_style_sensitivity_is_scale_free(pair):
    z, _ = pair
    rng = np.random.default_rng(7)
    xi = 0.2 * rng.standard_normal(z.shape)
    small = style_invariance(z, z + xi)
    scaled = style_invariance(10 * z, 10 * (z + xi))
    assert small['style_sensitivity'] == pytest.approx(
        scaled['style_sensitivity'], rel=1e-6
    )


# -------------------------------------------------------------- the whole


def test_compute_all_is_complete_and_self_describing(pair):
    z, z_next = pair
    scores = compute_all(
        z,
        mixed(z, 4),
        h_next=mixed(z_next, 4),
        rho=RHO,
        names=[f'l{i}' for i in range(N)],
        seed=0,
    )
    for key in (
        'metric_suite_version',
        'procrustes_mse_per_dim',
        'procrustes_scale',
        'canonical_corr',
        'recovered_dimensions',
        'probe_linear_per_latent',
        'probe_mlp_per_latent',
        'probe_latent_names',
        'trace_cov',
        'effective_rank',
        'epsilon',
        'delta_floor',
        'residual_hermite_degree',
        'delta_admissible',
        'predicted_error',
        'bound_vacuous',
        'mean_probe_r2_needed',
        'r2_ceiling',
        'procrustes_floor',
        'procrustes_floor_whitened',
    ):
        assert key in scores, key
    assert len(scores['canonical_corr']) == N
    assert scores['has_second_view']
    assert not scores['has_style_probe']


def test_bound_keys_absent_without_a_second_view(pair):
    """Absent rather than zero: a missing measurement must not read as a good one."""
    z, _ = pair
    scores = compute_all(z, mixed(z, 4), rho=RHO, seed=0)
    assert not scores['has_second_view']
    for key in ('delta', 'predicted_error', 'delta_admissible'):
        assert key not in scores


def test_style_keys_absent_without_a_probe(pair):
    z, z_next = pair
    scores = compute_all(
        z, mixed(z, 4), h_next=mixed(z_next, 4), rho=RHO, seed=0
    )
    assert not scores['has_style_probe']
    assert 'sigma_sq' not in scores
    # Without sigma_sq the split is degenerate and says so.
    assert scores['delta_style'] == 0.0
    assert scores['delta_content'] == pytest.approx(scores['delta'])

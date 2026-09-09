"""The frozen identifiability metric suite.

**Versioned separately from the training code, and frozen before Env 1.** A
metric that moves mid-programme silently invalidates every earlier row of the
global scatter, and there is no way to detect that after the fact -- the rows
look fine. :data:`METRIC_SUITE_VERSION` is recorded into every result row so a
change at least becomes visible.

Ported from ``lejepa-identifiability/experiments/lejepa_id/metrics.py`` and
``run_reacher.py::final_eval``, with the gaps in both filled in.

What each metric is for
-----------------------
The suite deliberately contains metrics that *disagree*, because the
disagreements are the measurement:

* :func:`procrustes_recovery` is the criterion. Theory predicts recovery up to
  an orthogonal transform, so this is what "recovered" means.
* :func:`bidirectional_r2` is diagnostic. ``R^2(z->h)`` flipping sign against
  ``R^2(h->z)`` is the V4 signature.
* :func:`probe_accessibility` is the **decoy**. A probe can read a latent out
  of an embedding that has not identified it at all -- most obviously when
  ``m >> n``. Reported alongside recovery precisely so the two can be seen to
  come apart; :func:`probe_divergence` is that gap as a single number.
* :func:`mcc_unaligned` is logged and **never a gate**. Greedy matching admits
  permutations but not rotations, so it penalises exactly the class of
  solutions the theory says are correct.

Every metric is reported on **both** distributions -- the OU training set and
the physics-rollout set -- and the gap between them is itself a logged
quantity, not something to be asserted away.
"""

import numpy as np


#: Bumped whenever any metric's *definition* changes. Recorded into every
#: result row. Changing a metric without bumping this makes old and new rows
#: silently incomparable.
METRIC_SUITE_VERSION = '1.1.0'


def _as2d(x):
    return np.atleast_2d(np.asarray(x, dtype=np.float64))


def _lstsq_with_intercept(x, y):
    """Least squares of ``y`` on ``[x, 1]``. Returns ``(coef, prediction)``."""
    x = _as2d(x)
    y = _as2d(y)
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return solution, design @ solution


def r2(x, y):
    """Coefficient of determination of a linear map ``x -> y``."""
    y = _as2d(y)
    _, prediction = _lstsq_with_intercept(x, y)
    ss_res = ((y - prediction) ** 2).sum()
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def bidirectional_r2(z, h):
    """``R^2(z->h)`` and ``R^2(h->z)``.

    Both directions, because they fail differently. ``R^2(h->z)`` staying high
    while ``R^2(z->h)`` collapses is the second-Hermite substitution V4
    predicts: the embedding still *contains* the latent but is no longer a
    linear image of it.

    Returns:
        dict: ``r2_z_to_h``, ``r2_h_to_z``.
    """
    return {'r2_z_to_h': r2(z, h), 'r2_h_to_z': r2(h, z)}


def procrustes_recovery(z, h):
    """Best orthogonal alignment of ``h`` to ``z``, and its residual.

    The criterion metric. Theory predicts identification **up to an orthogonal
    transform**, so the right question is not "is ``h`` equal to ``z``" but
    "is there a ``Q`` in ``O(n)`` with ``h ~ Qz``".

    Both raw and per-dimension errors are returned. The per-dimension one is
    the comparable number: a raw sum-of-squares grows with ``n``, so an
    ``n = 58`` profile would look worse than an ``n = 9`` one for no reason
    other than width.

    Returns:
        dict: ``procrustes_mse``, ``procrustes_mse_per_dim``, ``scale``.
    """
    z = _as2d(z)
    h = _as2d(h)
    n = z.shape[1]

    zc = z - z.mean(axis=0)
    hc = h - h.mean(axis=0)

    m = (hc.T @ zc) / len(zc)
    u, s, vt = np.linalg.svd(m)
    q = u @ vt

    residual = hc - zc @ q.T
    mse = float((residual**2).sum(axis=1).mean())
    return {
        'procrustes_mse': mse,
        'procrustes_mse_per_dim': mse / n,
        'procrustes_scale': float(s.sum() / n),
    }


def orthogonality_gap(z, h):
    """How far the best-fitting linear map is from orthogonal.

    Two numbers, because they catch different failures:

    ``orth_err_normalized``
        ``||A^T A - I||_F / sqrt(n)``. Zero for any orthogonal map, at any ``n``.
    ``cond``
        The condition number of ``A``. **This is the one that matters for
        planning**: a map can fit well and still be badly conditioned, and a
        near-singular direction means the planner's cost is nearly flat along a
        latent direction that physically matters. The Frobenius gap can stay
        small while ``cond`` blows up, so reporting only the first would hide
        exactly the failure that degrades control.

    Returns:
        dict: ``orth_err``, ``orth_err_normalized``, ``cond``.
    """
    z = _as2d(z)
    h = _as2d(h)
    n = z.shape[1]

    solution, _ = _lstsq_with_intercept(z, h)
    a = solution[:n].T

    gap = float(np.linalg.norm(a.T @ a - np.eye(n), 'fro'))
    singular = np.linalg.svd(a, compute_uv=False)
    smallest = singular.min()
    return {
        'orth_err': gap,
        'orth_err_normalized': gap / np.sqrt(n),
        'cond': float(singular.max() / smallest)
        if smallest > 0
        else float('inf'),
    }


def _isotonic(x, y):
    """Pool-adjacent-violators isotonic regression of ``y`` on ``x``."""
    order = np.argsort(x)
    values = y[order].astype(np.float64).copy()
    weights = np.ones(len(values))

    # Standard PAVA over a stack of (value, weight) blocks.
    value_stack, weight_stack, size_stack = [], [], []
    for value, weight in zip(values, weights, strict=True):
        value_stack.append(value)
        weight_stack.append(weight)
        size_stack.append(1)
        while len(value_stack) > 1 and value_stack[-2] > value_stack[-1]:
            v2, w2, s2 = (
                value_stack.pop(), weight_stack.pop(), size_stack.pop()
            )
            v1, w1, s1 = (
                value_stack.pop(), weight_stack.pop(), size_stack.pop()
            )
            merged = (v1 * w1 + v2 * w2) / (w1 + w2)
            value_stack.append(merged)
            weight_stack.append(w1 + w2)
            size_stack.append(s1 + s2)

    fitted = np.concatenate(
        [np.full(size, value)
         for value, size in zip(value_stack, size_stack, strict=True)]
    )
    out = np.empty_like(fitted)
    out[order] = fitted
    return out


def monotone_recovery(z, h, mode='per_latent'):
    """Recovery after a monotone reparametrisation of each coordinate.

    Env 1's latents are not i.i.d. and not all of them have a Gaussian
    marginal -- the discrete ones have none at all. Linear recovery would
    charge an encoder for a monotone warp it was never asked to avoid, so this
    fits an isotonic map per coordinate first, then measures Procrustes
    recovery on the warped variables.

    Args:
        z: ``(B, n)`` ground truth.
        h: ``(B, m)`` embeddings.
        mode: ``'per_latent'`` fits an independent warp per coordinate.
            ``'shared'`` fits one warp using the pooled coordinates -- weaker,
            but it cannot manufacture agreement by warping each coordinate
            onto its own target.

    Returns:
        dict: ``monotone_mse_per_dim`` and ``monotone_mode``.
    """
    z = _as2d(z)
    h = _as2d(h)

    if mode == 'per_latent':
        warped = np.column_stack(
            [
                _isotonic(h[:, j % h.shape[1]], z[:, j])
                for j in range(z.shape[1])
            ]
        )
    elif mode == 'shared':
        flat = _isotonic(h[:, : z.shape[1]].ravel(), z.ravel())
        warped = flat.reshape(z.shape)
    else:
        raise ValueError(
            f"mode must be 'per_latent' or 'shared'; got {mode!r}."
        )

    result = procrustes_recovery(z, warped)
    return {
        'monotone_mse_per_dim': result['procrustes_mse_per_dim'],
        'monotone_mode': mode,
    }


def whitening_error(h):
    """``eps = ||Cov(h) - I||_F``, the bound's first term."""
    h = _as2d(h)
    cov = np.cov(h.T)
    cov = np.atleast_2d(cov)
    return float(np.linalg.norm(cov - np.eye(len(cov)), 'fro'))


def alignment_gap(h, h_next, rho):
    """``delta = L(h) - 2(1 - rho) tr Cov(h)``, clamped at zero.

    The excess pair distance beyond what the OU step alone accounts for. The
    clamp is not cosmetic: sampling noise can push the estimate slightly
    negative, and a negative ``delta`` would produce a negative ``D`` and an
    uninterpretable bound.
    """
    h = _as2d(h)
    h_next = _as2d(h_next)
    rho = float(np.mean(rho))

    observed = float(((h_next - h) ** 2).sum(axis=1).mean())
    cov = np.atleast_2d(np.cov(h.T))
    expected = 2.0 * (1.0 - rho) * float(np.trace(cov))
    return max(observed - expected, 0.0)


def predicted_error(epsilon, delta, rho, anisotropic=False):
    """The theory's predicted recovery error: ``D + (eps + D)^2``.

    ``D = delta / (2 rho (1 - rho))``, the alignment gap measured in spectral-gap
    units.

    Args:
        epsilon: Whitening error.
        delta: Alignment gap.
        rho: Scalar or per-dimension autocorrelation. A vector is reduced by
            its **mean** -- the aggregation rule the plan fixes -- and
            ``anisotropic`` is set so the row records that a single number is
            standing in for a spread.
        anisotropic: Force the flag on.

    Returns:
        dict: ``D``, ``predicted_error``, ``spectral_gap``, ``anisotropic``.
    """
    rho_array = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    anisotropic = bool(anisotropic or np.ptp(rho_array) > 1e-9)
    rho_mean = float(rho_array.mean())

    gap = 2.0 * rho_mean * (1.0 - rho_mean)
    d = delta / gap if gap > 0 else float('inf')
    return {
        'D': float(d),
        'predicted_error': float(d + (epsilon + d) ** 2),
        'spectral_gap': float(gap),
        'anisotropic': anisotropic,
    }


def style_variance(h_style_a, h_style_b):
    """``sigma^2 = E||xi||^2``, the absolute style variance in the embedding.

    The observation here is not ``x = g(z)``: style is redrawn per view, so
    ``f(x)`` is random given ``z``. Write ``phi(z) = E_S[f(g(z, S))]`` for the
    style-averaged encoder and ``xi = f(g(z, S)) - phi(z)`` for the residual.
    The two views of a style probe share their content, so their difference is
    ``xi_a - xi_b`` with independent residuals, giving
    ``E||a - b||^2 = 2 sigma^2``.

    This is the *absolute* quantity the loss floor and the alignment gap are
    written in; :func:`style_invariance` reports the scale-free ratio, which is
    the right thing for comparing runs but cannot be substituted into either.

    ``sigma^2`` is also the continuous stand-in for the assumption this setup
    does not satisfy. At ``sigma^2 = 0`` the composition ``f . g`` is
    deterministic at the optimum -- ``f(g(z, s)) = Qz`` for every ``s`` -- even
    though ``g`` is not, so the theory's conclusion is recovered without its
    premise. How far ``sigma^2`` sits above zero is how far the run is from
    that.

    Returns:
        dict: ``sigma_sq`` and ``sigma_sq_per_dim``.
    """
    a = _as2d(h_style_a)
    b = _as2d(h_style_b)
    sigma_sq = 0.5 * float(((a - b) ** 2).sum(axis=1).mean())
    return {
        'sigma_sq': sigma_sq,
        'sigma_sq_per_dim': sigma_sq / a.shape[1] if a.shape[1] else 0.0,
    }


def alignment_floor(n, rho, sigma_sq=0.0):
    """The lowest alignment loss any encoder can reach, style included.

    With a deterministic ``g`` the floor is Thm 1's ``2(1 - rho) n``. With style
    redrawn per view it rises:

        ``L >= 2(1 - rho) n + 2 rho sigma^2``

    which follows from splitting ``L`` into ``E||phi(z') - phi(z)||^2 + 2
    sigma^2``, noting that whitening constrains the *total* output so
    ``tr Cov(phi) = n - sigma^2``, and applying the bound to ``phi``.

    Two things worth reading off it. The style term carries ``2 rho`` against
    the content term's ``2(1 - rho)`` -- 9:1 at ``rho = 0.9`` -- so the
    objective does prefer discarding style, which is why the setup works at
    all. And ``sigma^2 = 0`` is the unique minimiser, so a style-invariant
    encoder is not merely permitted but selected.

    Args:
        n: Embedding width ``m`` (equal to the latent dimension by
            construction).
        rho: Scalar or per-dimension autocorrelation, reduced by its mean.
        sigma_sq: Measured style variance from :func:`style_variance`. Zero
            gives the deterministic-``g`` floor.

    Returns:
        dict: ``align_floor``, ``align_floor_deterministic`` and
        ``align_floor_style_term``.
    """
    rho_mean = float(np.mean(np.asarray(rho, dtype=np.float64)))
    deterministic = 2.0 * (1.0 - rho_mean) * float(n)
    style_term = 2.0 * rho_mean * float(sigma_sq)
    return {
        'align_floor': deterministic + style_term,
        'align_floor_deterministic': deterministic,
        'align_floor_style_term': style_term,
    }


def split_alignment_gap(delta, rho, sigma_sq):
    """Separate the nonlinearity in ``phi`` from style leakage in ``delta``.

    :func:`alignment_gap` measures ``delta`` against ``2(1 - rho) tr Cov(h)``,
    and under a stochastic ``g`` that quantity decomposes exactly:

        ``delta = delta_content + 2 rho sigma^2``

    Only ``delta_content`` is the nonlinear energy Thm 3's ``D`` is meant to
    bound. Feeding the total in instead attributes style leakage to
    nonlinearity, and since the bound is ``D + (eps + D)^2`` the error is then
    squared: at ``rho = 0.9`` a style contribution of ``sigma^2`` enters ``D``
    as ``sigma^2 / (1 - rho) = 10 sigma^2``. That reads as "the encoder is
    nonlinear" when the cause is style reaching the output, which is the one
    misreading this whole measurement exists to prevent.

    Clamped at zero for the same reason :func:`alignment_gap` is: sampling
    noise can push the estimate slightly negative, and a negative gap produces
    an uninterpretable bound.

    Returns:
        dict: ``delta_total``, ``delta_content`` and ``delta_style``.
    """
    rho_mean = float(np.mean(np.asarray(rho, dtype=np.float64)))
    style = 2.0 * rho_mean * float(sigma_sq)
    return {
        'delta_total': float(delta),
        'delta_content': max(float(delta) - style, 0.0),
        'delta_style': style,
    }


def bound_is_vacuous(predicted, n):
    """Whether the recovery bound beats predicting ``h = 0``.

    ``E||h - Qz||^2 <= D + (eps + D)^2`` says nothing once it exceeds
    ``E||z||^2 = n``, which the trivial encoder already achieves. Reported
    rather than left implicit, because a bound of 20 against ``n = 10`` looks
    like a number and is not one.
    """
    return {
        'bound_vacuous': bool(predicted >= float(n)),
        'bound_headroom': float(n) - float(predicted),
    }


def in_gap_units(cost, rho):
    """Express a degradation in spectral-gap units.

    The unit every violation cost is reported in, so that costs stay
    comparable across environments whose ``rho`` differs.
    """
    rho_mean = float(np.mean(rho))
    gap = 2.0 * rho_mean * (1.0 - rho_mean)
    return float(cost / gap) if gap > 0 else float('inf')


def sigreg_z_score(samples, num_projections=256, num_null=64, seed=0):
    """SIGReg statistic, z-scored against a matched-i.i.d.-Gaussian floor.

    The raw Epps-Pulley statistic is not comparable across sample sizes or
    dimensions -- it grows with both. The floor is therefore recomputed **at
    our own sample size and width**, by running the same statistic on
    genuinely Gaussian data of identical shape, and the reported number is how
    many null standard deviations away the data sits.

    Args:
        samples: ``(B, d)`` samples to test.
        num_projections: Random 1-d projections per evaluation.
        num_null: Null replicates used to estimate the floor.
        seed: Seed for both the projections and the null.

    Returns:
        dict: ``sigreg``, ``sigreg_null_mean``, ``sigreg_null_std``,
        ``sigreg_z``.
    """
    samples = _as2d(samples)
    rng = np.random.default_rng(seed)
    knots = np.linspace(0.0, 3.0, 17)
    dt = 3.0 / 16
    weights = np.full(17, 2 * dt)
    weights[[0, -1]] = dt
    window = np.exp(-(knots**2) / 2.0)
    weights = weights * window

    def statistic(x, generator):
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
        directions = generator.standard_normal((x.shape[1], num_projections))
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        projected = (x @ directions)[..., None] * knots
        error = (np.cos(projected).mean(0) - window) ** 2 + np.sin(
            projected
        ).mean(0) ** 2
        return float((error @ weights).mean() * len(x))

    observed = statistic(samples, np.random.default_rng(seed))
    null = np.array(
        [
            statistic(
                rng.standard_normal(samples.shape),
                np.random.default_rng(seed),
            )
            for _ in range(num_null)
        ]
    )
    std = null.std()
    return {
        'sigreg': observed,
        'sigreg_null_mean': float(null.mean()),
        'sigreg_null_std': float(std),
        'sigreg_z': float((observed - null.mean()) / std)
        if std > 0
        else 0.0,
    }


def probe_accessibility(h, z, hidden=64, seed=0):
    """Per-latent linear **and** non-linear readout of ``z`` from ``h``.

    The section-2.5 decoy. A probe answers "is the information present", which
    is a strictly weaker question than "has the latent been identified" -- and
    the two come apart most obviously when ``m >> n``, where a random wide
    embedding is linearly probe-able for almost anything while being nowhere
    near an orthogonal image of ``z``.

    Reported ``h -> z`` only, with **no alignment**: aligning first would
    smuggle the recovery criterion into the decoy and destroy the very
    contrast this metric exists to provide.

    Returns:
        dict: ``probe_linear_r2``, ``probe_mlp_r2``, ``probe_gap``
        (``mlp - linear``), and the per-latent linear scores.
    """
    h = _as2d(h)
    z = _as2d(z)

    linear = np.array([r2(h, z[:, [j]]) for j in range(z.shape[1])])

    # A random-feature ridge stands in for the MLP rung: it is non-linear,
    # deterministic given the seed, and has no optimiser to tune -- which
    # keeps the frozen suite free of a training loop whose hyperparameters
    # would themselves become an unversioned degree of freedom.
    rng = np.random.default_rng(seed)
    hs = (h - h.mean(axis=0)) / (h.std(axis=0) + 1e-12)
    weight = rng.standard_normal((hs.shape[1], hidden)) / np.sqrt(
        hs.shape[1]
    )
    features = np.tanh(hs @ weight + rng.standard_normal(hidden))
    nonlinear = np.array(
        [r2(features, z[:, [j]]) for j in range(z.shape[1])]
    )

    return {
        'probe_linear_r2': float(linear.mean()),
        'probe_mlp_r2': float(nonlinear.mean()),
        'probe_gap': float(nonlinear.mean() - linear.mean()),
        'probe_linear_per_latent': linear.tolist(),
    }


def probe_divergence(probe_r2, orth_err_normalized):
    """How far probe accessibility has come apart from orthogonal recovery.

    The scatter's annotation. Large and positive means the embedding is
    probe-able but not identified -- the decoy firing while the criterion does
    not. That combination is a *result*, not an anomaly.
    """
    recovery_score = 1.0 / (1.0 + float(orth_err_normalized))
    return float(probe_r2 - recovery_score)


def mcc_unaligned(z, h):
    """Greedy-matched mean correlation coefficient.

    **Logged, labelled "permutation-only", and never a gate.** Greedy matching
    admits permutations but not rotations, so a perfectly recovered ``h = Qz``
    for a general orthogonal ``Q`` scores poorly here *by construction*. Using
    it as an exit criterion would reject exactly the encoders the theory says
    are correct.
    """
    z = _as2d(z)
    h = _as2d(h)

    zc = (z - z.mean(0)) / (z.std(0) + 1e-12)
    hc = (h - h.mean(0)) / (h.std(0) + 1e-12)
    corr = np.abs(zc.T @ hc) / len(zc)

    corr = corr.copy()
    matched = []
    for _ in range(min(corr.shape)):
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        matched.append(corr[i, j])
        corr[i, :] = -np.inf
        corr[:, j] = -np.inf
    return {
        'mcc_unaligned': float(np.mean(matched)),
        'mcc_note': 'permutation-only, never a gate',
    }


def hermite2_substitution(z, h, slow_index=None, rho=None):
    """How much of ``h`` is explained by ``He2`` of the slowest latent.

    V4's predicted failure mode made concrete. Past the isotropy boundary the
    encoder can lower its objective by representing the *second Hermite
    function* of a fast coordinate in place of a slow one it should have kept.
    ``He2(x) = x^2 - 1``.

    Args:
        z: ``(B, n)`` ground truth in z-space.
        h: ``(B, m)`` embeddings.
        slow_index: Which coordinate is the slow one. Defaults to the
            ``argmin`` of ``rho``, or coordinate 0.
        rho: Per-dimension autocorrelation, used to pick ``slow_index``.

    Returns:
        dict: ``hermite2_r2`` and ``hermite2_excess`` -- how much better
        ``He2`` explains ``h`` than the raw latent does. A positive excess is
        the substitution actually happening.
    """
    z = _as2d(z)
    h = _as2d(h)

    if slow_index is None:
        slow_index = (
            int(np.argmin(np.atleast_1d(rho))) if rho is not None else 0
        )

    raw = z[:, [slow_index]]
    hermite = raw**2 - 1.0
    return {
        'hermite2_r2': r2(hermite, h),
        'hermite2_excess': r2(hermite, h) - r2(raw, h),
        'hermite2_slow_index': int(slow_index),
    }


def style_invariance(h_style_a, h_style_b):
    """Embedding sensitivity to a known style transform.

    The two inputs are embeddings of the **same content** under two
    independent style draws. Normalised by the embedding's own scale, so it is
    comparable across runs whose embeddings differ in magnitude.

    Vacuous -- and reported as such -- under the ``all_content`` profile,
    where there is no style left to resample.

    Returns:
        dict: ``style_sensitivity`` and ``style_vacuous``.
    """
    a = _as2d(h_style_a)
    b = _as2d(h_style_b)
    scale = float(np.sqrt((a**2).sum(axis=1).mean()))
    if scale == 0.0:
        return {'style_sensitivity': 0.0, 'style_vacuous': True}
    sensitivity = float(
        np.sqrt(((a - b) ** 2).sum(axis=1).mean()) / scale
    )
    return {
        'style_sensitivity': sensitivity,
        'style_vacuous': bool(np.allclose(a, b)),
    }


def compute_all(
    z, h, z_next=None, h_next=None, rho=0.9, seed=0,
    h_style_a=None, h_style_b=None,
):
    """Run the whole suite on one (z, h) pair of matrices.

    Args:
        z: ``(B, n)`` ground truth in z-space.
        h: ``(B, m)`` embeddings of the first view.
        z_next: ``(B, n)`` second-view ground truth. Accepted and recorded but
            not used by any current metric -- the bound is a property of the
            *embeddings* alone. Kept in the signature so a caller that has it
            passes it, rather than the suite silently growing a dependency on
            data the call site never supplied.
        h_next: ``(B, m)`` second-view embeddings, needed for the bound.
        rho: Scalar or per-dimension autocorrelation.
        seed: Seed for the stochastic metrics.
        h_style_a: ``(B, m)`` embeddings of a **style probe** -- one view of a
            pair whose two frames share their content exactly and differ only
            in style. Cannot be the ordinary OU pair: those differ by one OU
            step *as well as* style, which would score the transition as if it
            were style leakage. Collect a probe set with ``rho = 1``, where
            ``z' = z`` identically.
        h_style_b: ``(B, m)`` the other view of that same probe set.

    Both style arguments must be given together; when either is missing the
    style-invariance metric is simply absent from the result, and the scatter
    records it as null rather than as a measured zero.

    Returns:
        dict: Every metric, flat, plus ``metric_suite_version``.
    """
    out = {'metric_suite_version': METRIC_SUITE_VERSION}
    out.update(bidirectional_r2(z, h))
    out.update(procrustes_recovery(z, h))
    out.update(orthogonality_gap(z, h))
    out.update(monotone_recovery(z, h, mode='per_latent'))
    out.update(mcc_unaligned(z, h))
    out.update(probe_accessibility(h, z, seed=seed))
    out.update(hermite2_substitution(z, h, rho=rho))
    out.update(sigreg_z_score(h, seed=seed))

    out['epsilon'] = whitening_error(h)
    out['has_second_view'] = z_next is not None and h_next is not None

    # The term the alignment loss is *supposed* to discard. Only measurable
    # against a same-content/different-style probe, so it stays absent rather
    # than defaulting to a number nothing computed -- and `sigma_sq` is
    # resolved before the bound, because the bound depends on it.
    out['has_style_probe'] = h_style_a is not None and h_style_b is not None
    if out['has_style_probe']:
        out.update(style_invariance(h_style_a, h_style_b))
        out.update(style_variance(h_style_a, h_style_b))
    out.update(alignment_floor(h.shape[1], rho, out.get('sigma_sq', 0.0)))

    if h_next is not None:
        out['delta'] = alignment_gap(h, h_next, rho)
        out.update(split_alignment_gap(out['delta'], rho, out.get('sigma_sq', 0.0)))
        # `delta_content`, not `delta`: only the nonlinearity of `phi` is what
        # Thm 3's D bounds. Without a style probe the two coincide, and the
        # `has_style_probe` flag is what says which of the two was reported.
        out.update(
            predicted_error(out['epsilon'], out['delta_content'], rho)
        )
        out.update(bound_is_vacuous(out['predicted_error'], h.shape[1]))

    out['probe_divergence'] = probe_divergence(
        out['probe_linear_r2'], out['orth_err_normalized']
    )
    out['recovery_in_gap_units'] = in_gap_units(
        out['procrustes_mse_per_dim'], rho
    )

    return out


__all__ = [
    'METRIC_SUITE_VERSION',
    'alignment_floor',
    'alignment_gap',
    'bidirectional_r2',
    'bound_is_vacuous',
    'compute_all',
    'hermite2_substitution',
    'in_gap_units',
    'mcc_unaligned',
    'monotone_recovery',
    'orthogonality_gap',
    'predicted_error',
    'probe_accessibility',
    'probe_divergence',
    'procrustes_recovery',
    'r2',
    'sigreg_z_score',
    'split_alignment_gap',
    'style_invariance',
    'style_variance',
    'whitening_error',
]

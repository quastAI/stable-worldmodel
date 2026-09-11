"""The frozen metric suite: does ``h`` recover ``z``, and is the answer usable?

Every function here takes numpy arrays of embeddings that were produced by the
encoder **in eval mode**. That is not a stylistic note. ``CNNEncoder`` carries
BatchNorm after every stage plus a ``BatchNorm1d`` before the head, and
``LeJEPA.encode`` flattens the view axis into the batch, so in train mode ``h``
is a function of *the whole batch* rather than of one frame. Every quantity
below -- and every quantity in the theory -- quantifies over a fixed measurable
``h``, so train-mode values are not noisy estimates of these numbers, they are
a different object. :func:`delta_admissibility` exists to catch it when it
happens anyway.

How the suite is meant to be read
---------------------------------
Three layers, in order of how much they can tell you:

**Per-latent.** :func:`probe_accessibility` reports ``R^2(h -> z_j)`` for every
coordinate separately, linear and non-linear. This is the layer that localises
a failure, and it is the reason the aggregates below are never read alone: a
mean over ten latents whose observability through the renderer spans two orders
of magnitude is not a number about the encoder.

**The spectrum.** :func:`canonical_correlations` is how many directions of
``h`` actually carry ``z``, and at what strength. ``procrustes_mse_per_dim``
and ``r2_*`` are two different summaries of that one spectrum -- see
:func:`procrustes_recovery` for the identity that connects them -- so they
cannot disagree, and their agreement is not corroboration.

**The bound, and whether it can say anything.** ``predicted_error`` is only
meaningful when it is below ``n``, because the trivial encoder ``h = 0`` already
achieves ``n``. :func:`bound_reach` reports the recovery level at which the
bound *could* become non-vacuous, so a vacuous bound reads as arithmetically
inevitable rather than as a bad run.
"""

import numpy as np


METRIC_SUITE_VERSION = '2.0.0'

#: Per-latent linear R^2 below this counts a coordinate as unrecovered when
#: computing the observability ceiling. Not a gate on anything -- it only
#: decides which latents the reported ceiling excludes.
DEAD_LATENT_R2 = 0.01


def _as2d(x):
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 2 else x.reshape(len(x), -1)


def _lstsq_with_intercept(x, y):
    """Least squares of ``y`` on ``[x, 1]``. Returns ``(coef, prediction)``."""
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    solution = np.linalg.pinv(design) @ y
    return solution, design @ solution


def r2(x, y):
    """Coefficient of determination of a linear map ``x -> y``.

    One global sum-of-squares ratio, not a mean of per-column ratios. With
    ``z`` unit-variance by construction the two coincide for ``h -> z``, but
    the global form is the one the Procrustes identity is written in.
    """
    y = _as2d(y)
    _, prediction = _lstsq_with_intercept(x, y)
    ss_res = ((y - prediction) ** 2).sum()
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _holdout(n_rows, fraction, seed):
    """Deterministic train/test split, shared by every probe."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    cut = int(round(n_rows * (1.0 - fraction)))
    return order[:cut], order[cut:]


# ----------------------------------------------------------------- recovery


def bidirectional_r2(z, h):
    """``R^2(z->h)`` and ``R^2(h->z)``.

    Both directions are reported, but note what they can and cannot tell you
    apart: once ``Cov(h)`` is close to ``I`` and ``Cov(z) = I`` by
    construction, **both equal the mean squared canonical correlation and are
    therefore forced to agree**. Their agreement is a check that whitening
    worked, not independent evidence about recovery.

    Returns:
        dict: ``r2_z_to_h``, ``r2_h_to_z``.
    """
    return {'r2_z_to_h': r2(z, h), 'r2_h_to_z': r2(h, z)}


def canonical_correlations(z, h):
    """The spectrum of linear agreement between ``h`` and ``z``.

    Singular values of the cross-covariance after whitening both sides, so
    each lies in ``[0, 1]`` and is invariant to any invertible linear
    reparametrisation of either -- which is exactly the invariance the theory
    grants (identification up to a rotation). This is the most direct answer to
    "how many directions of the latent state did the encoder actually find",
    and the aggregates are functions of it:

    * ``sum(sigma^2)`` equals ``n * mean(probe_linear_per_latent)``
    * ``sum(sigma_raw)`` drives ``procrustes_mse_per_dim``

    Returns:
        dict: ``canonical_corr`` (descending list), ``recovered_dimensions``
        (``sum sigma^2``, reading as "how many latents' worth of information"),
        and ``canonical_participation`` (``(sum s^2)^2 / sum s^4``, the
        effective *number* of recovered directions, which separates "six
        directions at 0.85" from "four at 1.0").
    """
    z = _as2d(z)
    h = _as2d(h)

    def whiten(a):
        a = a - a.mean(axis=0)
        cov = (a.T @ a) / max(len(a) - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        keep = values > 1e-10 * max(values.max(), 1e-30)
        return a @ (vectors[:, keep] / np.sqrt(values[keep]))

    zw, hw = whiten(z), whiten(h)
    if zw.shape[1] == 0 or hw.shape[1] == 0:
        return {
            'canonical_corr': [],
            'recovered_dimensions': 0.0,
            'canonical_participation': 0.0,
        }

    cross = (hw.T @ zw) / max(len(zw) - 1, 1)
    sigma = np.clip(np.linalg.svd(cross, compute_uv=False), 0.0, 1.0)
    squared = sigma**2
    total = squared.sum()
    return {
        'canonical_corr': sigma.tolist(),
        'recovered_dimensions': float(total),
        'canonical_participation': float(total**2 / (squared**2).sum())
        if total > 0
        else 0.0,
    }


def procrustes_recovery(z, h):
    """Best orthogonal alignment of ``h`` to ``z``, and its residual.

    The criterion metric: theory identifies ``z`` only up to an orthogonal
    transform, so the question is whether some ``Q`` in ``O(n)`` has ``h ~ Qz``.

    **This metric is not scale-free, and reading it as a fraction-recovered is
    wrong.** With ``z`` unit-variance by construction it decodes exactly as::

        procrustes_mse_per_dim = ( tr Cov(h) + n - 2 * sum(sigma_raw) ) / n

    so ``h = 0`` scores **1.0**, and an isotropic ``h`` at ``tr Cov(h) = n``
    that carries no information at all scores **2.0**. A randomly initialised
    encoder measures near 1.0 because its embedding is small, not because it is
    halfway to correct. ``procrustes_scale`` is reported alongside for that
    reason: it is ``sum(sigma_raw) / n``, the part of the number that is
    actually about agreement.

    Returns:
        dict: ``procrustes_mse``, ``procrustes_mse_per_dim``,
        ``procrustes_scale``.
    """
    z = _as2d(z)
    h = _as2d(h)
    n = z.shape[1]

    zc = z - z.mean(axis=0)
    hc = h - h.mean(axis=0)

    cross = (hc.T @ zc) / len(zc)
    u, s, vt = np.linalg.svd(cross)
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

    ``orth_err_normalized`` is ``||A^T A - I||_F / sqrt(n)``, zero for any
    orthogonal map at any ``n``. ``cond`` is the condition number of ``A``, and
    **it is the one that matters downstream**: a near-singular direction is a
    latent along which any cost built on ``h`` is nearly flat. The two move
    independently -- the Frobenius gap falls as the recovered block tidies up
    while ``cond`` rises because the unrecovered directions stay null -- so
    reporting only the first hides the failure that breaks control.

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


# ----------------------------------------------------------------- spectrum


def spectrum(h):
    """The embedding's second-moment structure, including what aggregates hide.

    ``epsilon = ||Cov(h) - I||_F`` is the bound's first term and an aggregate,
    and aggregates hide partial collapse: one dead direction out of ``n``
    contributes 1 to ``epsilon^2`` and disappears into ordinary values. The
    smallest eigenvalue and the participation ratio do not average it away.

    Returns:
        dict: ``epsilon``, ``whitening_metric`` (``epsilon^2 / n^2``, the
        training-time key's units), ``trace_cov``, ``cov_eig_min``,
        ``cov_eig_max`` and ``effective_rank`` -- ``(sum lambda)^2 / sum
        lambda^2``, in ``[1, n]``, reading as "how many directions carry the
        variance".
    """
    h = _as2d(h)
    n = h.shape[1]
    centred = h - h.mean(axis=0)
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    values = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    total = values.sum()
    epsilon = float(np.linalg.norm(cov - np.eye(n), 'fro'))
    return {
        'epsilon': epsilon,
        'whitening_metric': epsilon**2 / n**2,
        'trace_cov': float(total),
        'cov_eig_min': float(values.min()),
        'cov_eig_max': float(values.max()),
        'effective_rank': float(total**2 / (values**2).sum())
        if total > 0
        else 0.0,
    }


def sigreg_z_score(
    samples, num_projections=128, num_null=32, max_rows=4096, seed=0
):
    """Isotropy of the *distribution*, not just of the covariance.

    SIGReg's target is an isotropic **Gaussian**, and ``Cov(h) = I`` only
    constrains the second moment. This z-scores the Epps-Pulley statistic
    against a matched i.i.d.-Gaussian null at the same sample size and width,
    so a value near zero means "indistinguishable from the target" while a
    large one means the higher moments are still wrong even though whitening
    looks finished.

    ``max_rows`` subsamples before scoring, and the null is drawn at the same
    size so the comparison stays matched. The statistic converges long before
    20k rows, and the full-sample version dominated the cost of the whole
    suite -- it is a sum over ``rows x projections x knots``, evaluated once
    per null draw.

    Returns:
        dict: ``sigreg_z``, ``sigreg_rows``.
    """
    x = _as2d(samples)
    rng = np.random.default_rng(seed)

    if len(x) > max_rows:
        x = x[rng.choice(len(x), max_rows, replace=False)]
    n_rows, width = x.shape

    knots = np.linspace(-3.0, 3.0, 17)
    target = np.exp(-0.5 * knots**2)

    directions = rng.standard_normal((width, num_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)

    def statistic(a):
        a = a - a.mean(axis=0)
        cov = (a.T @ a) / max(len(a) - 1, 1)
        values, vectors = np.linalg.eigh(cov)
        whitener = vectors / np.sqrt(np.clip(values, 1e-12, None))
        projected = (a @ whitener) @ directions
        # Accumulate one knot at a time: the (rows, projections, knots) array
        # is what made the full-sample version unaffordable.
        total = np.zeros(num_projections)
        for knot, want in zip(knots, target):
            total += (np.cos(projected * knot).mean(axis=0) - want) ** 2
        return float(total.mean())

    observed = statistic(x)
    null = np.array(
        [
            statistic(rng.standard_normal((n_rows, width)))
            for _ in range(num_null)
        ]
    )
    spread = null.std()
    return {
        'sigreg_z': float((observed - null.mean()) / spread)
        if spread > 0
        else 0.0,
        'sigreg_rows': int(n_rows),
    }


# ------------------------------------------------------------------- probes


def probe_accessibility(
    h, z, names=None, hidden=256, epochs=200, holdout=0.2, seed=0
):
    """Per-latent read-out of ``z`` from ``h``, linear and non-linear.

    The question "is the information present at all" is strictly weaker than
    "has the latent been identified", and keeping them apart is the point of
    this metric -- so nothing here is aligned to ``z`` first.

    Both probes are scored on a **held-out split**, which is what makes the
    linear and non-linear numbers comparable: an in-sample R^2 from a flexible
    model measures capacity, not accessibility. The non-linear rung is a small
    trained MLP rather than a random-feature ridge. A ridge of a few dozen
    fixed features cannot separate "absent from ``h``" from "present but
    non-linearly coded", and that distinction is the whole reason to report a
    non-linear probe: a latent that a trained probe recovers and a linear one
    does not is an encoder that found the quantity and failed to linearise it,
    which is a completely different finding from one that never saw it.

    Determinism is by fixed seed and a fixed schedule, not by avoiding an
    optimiser. Torch is imported lazily so the rest of the suite stays numpy.

    Args:
        h: ``(B, m)`` embeddings, produced in eval mode.
        z: ``(B, n)`` ground truth in z-space.
        names: Optional latent names, ``n`` of them, recorded alongside the
            scores so a row is self-describing.
        hidden: Width of the MLP probe's two hidden layers.
        epochs: Fixed number of full passes for the MLP probe.
        holdout: Fraction of rows held out for scoring.
        seed: Seed for the split and the probe initialisation.

    Returns:
        dict: ``probe_linear_r2`` and ``probe_mlp_r2`` (means over latents),
        ``probe_linear_per_latent`` and ``probe_mlp_per_latent``,
        ``probe_gap`` (mlp minus linear), and ``probe_latent_names``.
    """
    h = _as2d(h)
    z = _as2d(z)
    train, test = _holdout(len(h), holdout, seed)

    def scores(predict):
        out = []
        for j in range(z.shape[1]):
            truth = z[test, j]
            ss_tot = ((truth - truth.mean()) ** 2).sum()
            residual = ((truth - predict[:, j]) ** 2).sum()
            out.append(float(1.0 - residual / ss_tot) if ss_tot > 0 else 0.0)
        return np.array(out)

    solution, _ = _lstsq_with_intercept(h[train], z[train])
    design = np.concatenate([h[test], np.ones((len(test), 1))], axis=1)
    linear = scores(design @ solution)

    nonlinear = _mlp_probe(h, z, train, test, hidden, epochs, seed)
    nonlinear = scores(nonlinear)

    result = {
        'probe_linear_r2': float(linear.mean()),
        'probe_mlp_r2': float(nonlinear.mean()),
        'probe_gap': float(nonlinear.mean() - linear.mean()),
        'probe_linear_per_latent': linear.tolist(),
        'probe_mlp_per_latent': nonlinear.tolist(),
    }
    if names is not None:
        result['probe_latent_names'] = list(names)
    return result


def _mlp_probe(h, z, train, test, hidden, epochs, seed):
    """Train one multi-output MLP ``h -> z`` and predict the held-out rows.

    One shared network for all coordinates rather than ``n`` separate ones:
    the question is whether the information is in ``h``, and a shared trunk
    answers it at a fraction of the cost.

    **Best-epoch selection on an inner validation slice**, not a fixed number
    of steps. Without it the probe overfits and reports a *negative* held-out
    R^2 on a coordinate that is genuinely unrecoverable -- measured -0.24 on a
    latent encoded as ``He2(z)``, where the sign of ``z`` is destroyed and the
    only correct answer is 0. Worse, it then scores *below* the linear probe on
    coordinates the linear probe gets right, which inverts the one comparison
    this metric exists to make. Selecting the best inner-validation epoch fixes
    both: the network starts near the mean predictor, so the selected state can
    never be worse than "predict the mean", i.e. R^2 = 0.

    Inputs are standardised on the training rows only.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    # Carve an inner validation slice out of `train`; `test` stays untouched.
    cut = int(round(len(train) * 0.85))
    fit, inner = train[:cut], train[cut:]

    mean, std = h[fit].mean(axis=0), h[fit].std(axis=0) + 1e-8

    def tensor(a):
        return torch.tensor(np.asarray(a, dtype=np.float32))

    x_fit = tensor((h[fit] - mean) / std)
    x_inner = tensor((h[inner] - mean) / std)
    x_test = tensor((h[test] - mean) / std)
    y_fit = tensor(z[fit])
    y_inner = tensor(z[inner])

    net = nn.Sequential(
        nn.Linear(h.shape[1], hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, z.shape[1]),
    )
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    batch = min(1024, len(x_fit))
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for begin in range(0, len(order), batch):
            index = order[begin : begin + batch]
            optimiser.zero_grad()
            nn.functional.mse_loss(net(x_fit[index]), y_fit[index]).backward()
            optimiser.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                loss = float(nn.functional.mse_loss(net(x_inner), y_inner))
            if loss < best_loss:
                best_loss = loss
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(x_test).numpy().astype(np.float64)


# -------------------------------------------------------------------- style


def style_variance(h_style_a, h_style_b):
    """``sigma^2 = E||xi||^2``, the absolute style variance in the embedding.

    The observation is not ``x = g(z)``: style is redrawn per view, so
    ``f(x)`` is random given ``z``. Write ``phi(z) = E_S[f(g(z, S))]`` and
    ``xi = f(g(z, S)) - phi(z)``. The two views of a style probe share their
    content, so their difference is ``xi_a - xi_b`` with independent residuals
    and ``E||a - b||^2 = 2 sigma^2``.

    This is the *absolute* quantity the alignment floor and the alignment gap
    are written in; :func:`style_invariance` reports the scale-free ratio,
    which is right for comparing runs but cannot be substituted into either.

    Returns:
        dict: ``sigma_sq``, ``sigma_sq_per_dim``.
    """
    a = _as2d(h_style_a)
    b = _as2d(h_style_b)
    sigma_sq = 0.5 * float(((a - b) ** 2).sum(axis=1).mean())
    return {
        'sigma_sq': sigma_sq,
        'sigma_sq_per_dim': sigma_sq / a.shape[1] if a.shape[1] else 0.0,
    }


def style_invariance(h_style_a, h_style_b):
    """Scale-free sensitivity to a known style transform.

    ``E||a - b||^2 / tr Cov(h)``: the share of the embedding's own variance
    that style accounts for. ``style_vacuous`` flags the case where there was
    no style to resample, so a perfect-looking zero cannot be mistaken for
    invariance that was earned.

    Returns:
        dict: ``style_sensitivity``, ``style_vacuous``.
    """
    a = _as2d(h_style_a)
    b = _as2d(h_style_b)
    pooled = np.concatenate([a, b], axis=0)
    centred = pooled - pooled.mean(axis=0)
    total = float((centred**2).sum(axis=1).mean())
    difference = float(((a - b) ** 2).sum(axis=1).mean())
    return {
        'style_sensitivity': difference / total if total > 0 else 0.0,
        'style_vacuous': bool(difference == 0.0),
    }


# ---------------------------------------------------------------- the bound


def alignment_gap(h, h_next, rho):
    """``delta = L - 2(1 - rho) tr Cov(h)``, clamped at zero.

    ``L = E||h(z') - h(z)||^2``, the paper's alignment loss -- a *sum* over
    dimensions, which the training key ``align_loss`` is not: that one is a
    mean over ``(V, B, n)``, so ``L = 4 n * align_loss`` at two views.

    The clamp is for sampling noise, and it is also the thing that hides a
    broken measurement: a raw gap that is genuinely negative is impossible for
    any style-invariant function of ``z``, so it means the inputs are not what
    they claim to be. :func:`delta_admissibility` is what turns that from a
    silent clamp into a reported flag.

    Returns:
        dict: ``L``, ``delta``, ``align_loss_equivalent``.
    """
    h = _as2d(h)
    h_next = _as2d(h_next)
    loss = float(((h_next - h) ** 2).sum(axis=1).mean())
    centred = h - h.mean(axis=0)
    trace = float(np.trace((centred.T @ centred) / max(len(centred) - 1, 1)))
    return {
        'L': loss,
        'delta': max(loss - 2.0 * (1.0 - rho) * trace, 0.0),
        'align_loss_equivalent': loss / (4.0 * h.shape[1]),
    }


def split_alignment_gap(delta, rho, sigma_sq):
    """Separate nonlinearity in ``phi`` from style leakage in ``delta``.

    Under a stochastic ``g`` the gap decomposes exactly as ``delta =
    delta_content + 2 rho sigma^2``, and only ``delta_content`` is the
    nonlinear energy the bound's ``D`` is meant to cover. Feeding the total in
    instead attributes style leakage to nonlinearity, and since the bound is
    ``D + (eps + D)^2`` the error is then squared: at ``rho = 0.9`` a style
    contribution of ``sigma^2`` enters ``D`` as ``10 sigma^2``.

    Returns:
        dict: ``delta_total``, ``delta_content``, ``delta_style``.
    """
    style = 2.0 * rho * float(sigma_sq)
    return {
        'delta_total': float(delta),
        'delta_content': max(float(delta) - style, 0.0),
        'delta_style': style,
    }


def alignment_floor(n, rho, trace_cov=None, sigma_sq=0.0):
    """The lowest alignment loss any encoder can reach, style included.

    ``L >= 2(1 - rho) tr Cov(h) + 2 rho sigma^2``. Note the floor is set by the
    embedding's *achieved* trace, not by ``n``: the often-quoted ``2(1 - rho)
    n`` assumes whitening already succeeded, and comparing a run against it
    while ``tr Cov(h) < n`` reads a shrinking embedding as a converged one.

    The style term carries ``2 rho`` against the content term's ``2(1 - rho)``
    -- 9:1 at ``rho = 0.9`` -- so the objective does prefer discarding style,
    and ``sigma^2 = 0`` is its unique minimiser.

    Returns:
        dict: ``align_floor``, ``align_floor_deterministic``,
        ``align_floor_style_term``, and ``align_loss_floor`` in the training
        key's own units.
    """
    trace = float(n if trace_cov is None else trace_cov)
    deterministic = 2.0 * (1.0 - rho) * trace
    style_term = 2.0 * rho * float(sigma_sq)
    return {
        'align_floor': deterministic + style_term,
        'align_floor_deterministic': deterministic,
        'align_floor_style_term': style_term,
        'align_loss_floor': (deterministic + style_term) / (4.0 * n),
    }


def delta_admissibility(delta_content, trace_cov, recovered_dimensions, rho):
    """Is the measured ``delta`` even possible, given how little was recovered?

    This is a **self-consistency gate on the measurement**, not a property of
    the encoder, and it is the cheapest way to catch embeddings that were
    produced under the wrong normalisation mode.

    The argument. A style-invariant direction of ``h`` is a function of ``z``
    alone, and a function of ``z`` decorrelates across one OU step at ``rho^k``
    for its degree-``k`` Hermite content. A direction with *zero* linear
    correlation with ``z`` therefore has degree at least two, so it decorrelates
    at ``rho^2`` or faster. Splitting ``h``'s variance into the part ``z``
    explains linearly (``sum sigma^2``, i.e. ``recovered_dimensions``) and the
    rest, the best case for the remainder is degree exactly two, and the whole
    expression collapses to::

        delta >= 2 rho (1 - rho) * ( tr Cov(h) - recovered_dimensions )
        D     >=                    tr Cov(h) - recovered_dimensions

    So the bound's ``D`` is at least the amount of the embedding's variance
    that ``z`` does not linearly explain -- which is a tidier statement of the
    bound's real content than ``delta`` itself.

    Inverting the same expression gives the degree the residual actually
    behaves like, ``rho_res = rho - delta / (2 * residual_variance)`` and
    ``degree = log(rho_res) / log(rho)``. Below 2 is impossible, so
    ``residual_hermite_degree`` is the single number to read: under 2 means do
    not trust the row.

    Returns:
        dict: ``residual_variance``, ``delta_floor``, ``delta_admissible``,
        ``residual_corr``, ``residual_hermite_degree``.
    """
    residual = float(trace_cov) - float(recovered_dimensions)
    floor = 2.0 * rho * (1.0 - rho) * max(residual, 0.0)

    corr = degree = float('nan')
    if residual > 1e-9:
        corr = rho - float(delta_content) / (2.0 * residual)
        if 0.0 < corr < 1.0:
            degree = float(np.log(corr) / np.log(rho))
        elif corr >= 1.0:
            degree = 0.0

    return {
        'residual_variance': residual,
        'delta_floor': floor,
        # 15% slack, and trivially satisfied when there is no unexplained
        # variance to floor. `delta` is a difference of two nearly-equal large
        # numbers and the floor is built from another estimate, so the ratio is
        # noisy: an exactly-degree-2 residual measures a degree of 1.97 at 4k
        # samples, and a fully affine encoder leaves a residual of ~0.02 whose
        # floor `delta` misses by a hair. The failure this catches is not
        # marginal -- the train-mode run measured degree 1.0 against a floor it
        # missed by 100% -- so a loose gate loses nothing and stops the flag
        # from crying wolf on honest rows.
        'delta_admissible': bool(
            float(delta_content) >= 0.85 * floor or residual < 0.5
        ),
        'residual_corr': corr,
        'residual_hermite_degree': degree,
    }


def predicted_error(epsilon, delta_content, rho):
    """The theory's predicted recovery error: ``D + (eps + D)^2``.

    ``D = delta_content / (2 rho (1 - rho))``. ``delta_content``, not the raw
    gap: only the nonlinearity of ``phi`` is what ``D`` covers.

    Returns:
        dict: ``D``, ``predicted_error``, ``spectral_gap``.
    """
    gap = 2.0 * rho * (1.0 - rho)
    d = float(delta_content) / gap
    return {
        'D': d,
        'predicted_error': d + (float(epsilon) + d) ** 2,
        'spectral_gap': gap,
    }


def bound_reach(n, epsilon, trace_cov, rho):
    """How much recovery the bound needs before it can say anything.

    ``E||h - Qz||^2 <= D + (eps + D)^2`` is vacuous once the right-hand side
    exceeds ``E||z||^2 = n``, which ``h = 0`` already achieves. Since
    :func:`delta_admissibility` puts ``D >= tr Cov(h) - recovered_dimensions``,
    the bound cannot be non-vacuous unless enough of ``h`` is linearly
    explained by ``z`` -- regardless of how well the run was optimised.

    Solving ``D + (eps + D)^2 = n`` for ``D`` and reading the requirement back
    gives the recovery level below which reporting ``predicted_error`` epoch to
    epoch is tracking an arithmetic identity. Reported so that a vacuous bound
    is labelled inevitable rather than disappointing.

    Returns:
        dict: ``d_max_nonvacuous``, ``recovered_dimensions_needed``,
        ``mean_probe_r2_needed``.
    """
    # D + (eps + D)^2 = n  ->  D^2 + (2 eps + 1) D + eps^2 - n = 0
    b = 2.0 * float(epsilon) + 1.0
    c = float(epsilon) ** 2 - float(n)
    d_max = (-b + np.sqrt(b * b - 4.0 * c)) / 2.0
    needed = max(float(trace_cov) - d_max, 0.0)
    return {
        'd_max_nonvacuous': float(d_max),
        'recovered_dimensions_needed': needed,
        'mean_probe_r2_needed': needed / n,
    }


def bound_is_vacuous(predicted, n):
    """Whether the bound beats predicting ``h = 0``.

    Reported rather than left implicit, because a bound of 20 against ``n =
    10`` looks like a number and is not one.
    """
    return {'bound_vacuous': bool(predicted >= float(n))}


# ------------------------------------------------------------------ ceiling


def observability_ceiling(probe_linear_per_latent, trace_cov, n):
    """What the aggregates could reach if the unrecovered latents never move.

    The renderer does not expose every latent equally -- camera framing starves
    some coordinates of pixel travel, and a latent that reaches the sensor
    weakly caps recovery for reasons that have nothing to do with the
    objective. An aggregate that averages such a coordinate in alongside a
    well-observed one is not a number about the encoder, so the ceiling is
    reported next to it.

    ``dead_latents`` are those below :data:`DEAD_LATENT_R2`. Both ceilings
    assume they stay at zero and the rest go to perfect, and both come from the
    Procrustes identity in :func:`procrustes_recovery`.

    **Two floors, because they answer different questions.** Given ``k`` live
    latents and a trace ``T``, the raw cross-covariance singular values obey
    ``sum sigma <= sqrt(k T)`` -- maximised by concentrating all of ``T`` into
    the ``k`` live directions -- so ``procrustes_floor`` is a true lower bound
    that no encoder can beat. It is loose, because concentrating the trace is
    exactly what SIGReg forbids. ``procrustes_floor_whitened`` instead assumes
    whitening holds, ``Cov(h) = (T/n) I``, giving ``sum sigma <= k sqrt(T/n)``;
    that is the realistic target, and it is **not** a hard floor -- sampling
    noise in the unrecovered block adds a little to ``sum sigma`` and a
    measured value can sit just under it.

    Returns:
        dict: ``dead_latents`` (indices), ``n_dead_latents``, ``r2_ceiling``,
        ``procrustes_floor``, ``procrustes_floor_whitened``.
    """
    scores = np.asarray(probe_linear_per_latent, dtype=np.float64)
    dead = np.flatnonzero(scores < DEAD_LATENT_R2)
    live = int(len(scores) - len(dead))
    trace = max(float(trace_cov), 0.0)

    return {
        'dead_latents': dead.tolist(),
        'n_dead_latents': int(len(dead)),
        'r2_ceiling': live / n,
        'procrustes_floor': float(
            (trace + n - 2.0 * np.sqrt(live * trace)) / n
        ),
        'procrustes_floor_whitened': float(
            (trace + n - 2.0 * live * np.sqrt(trace / n)) / n
        ),
    }


# ------------------------------------------------------------------ the run


def compute_all(
    z,
    h,
    h_next=None,
    rho=0.9,
    names=None,
    seed=0,
    h_style_a=None,
    h_style_b=None,
):
    """Run the whole suite on one ``(z, h)`` pair of matrices.

    Args:
        z: ``(B, n)`` ground truth in z-space, first view.
        h: ``(B, m)`` eval-mode embeddings of the first view.
        h_next: ``(B, m)`` embeddings of the second view. Needed for the
            alignment gap and therefore for the bound; without it those keys
            are absent rather than zero.
        rho: The collector's autocorrelation, one scalar.
        names: Latent names, for a self-describing per-latent breakdown.
        seed: Seed for the probes' split and initialisation.
        h_style_a: ``(B, m)`` one view of a **style probe** -- a pair whose two
            frames share their content exactly and differ only in style. It
            cannot be the ordinary OU pair, which differs by one OU step as
            well, and without it ``delta`` cannot be split and the bound
            charges style leakage to nonlinearity.
        h_style_b: The other view of that same probe set.

    Returns:
        dict: Every metric, flat, plus ``metric_suite_version``.
    """
    out = {'metric_suite_version': METRIC_SUITE_VERSION, 'rho': float(rho)}

    out.update(bidirectional_r2(z, h))
    out.update(canonical_correlations(z, h))
    out.update(procrustes_recovery(z, h))
    out.update(orthogonality_gap(z, h))
    out.update(spectrum(h))
    out.update(sigreg_z_score(h, seed=seed))
    out.update(probe_accessibility(h, z, names=names, seed=seed))
    out.update(
        observability_ceiling(
            out['probe_linear_per_latent'], out['trace_cov'], _as2d(z).shape[1]
        )
    )

    # `sigma_sq` is resolved before the bound, because the bound depends on it.
    out['has_style_probe'] = h_style_a is not None and h_style_b is not None
    if out['has_style_probe']:
        out.update(style_variance(h_style_a, h_style_b))
        out.update(style_invariance(h_style_a, h_style_b))
    out.update(
        alignment_floor(
            _as2d(h).shape[1],
            rho,
            trace_cov=out['trace_cov'],
            sigma_sq=out.get('sigma_sq', 0.0),
        )
    )

    out['has_second_view'] = h_next is not None
    if h_next is not None:
        out.update(alignment_gap(h, h_next, rho))
        out.update(
            split_alignment_gap(out['delta'], rho, out.get('sigma_sq', 0.0))
        )
        out.update(
            delta_admissibility(
                out['delta_content'],
                out['trace_cov'],
                out['recovered_dimensions'],
                rho,
            )
        )
        out.update(predicted_error(out['epsilon'], out['delta_content'], rho))
        out.update(bound_is_vacuous(out['predicted_error'], _as2d(z).shape[1]))
        out.update(
            bound_reach(
                _as2d(z).shape[1], out['epsilon'], out['trace_cov'], rho
            )
        )

    return out


__all__ = [
    'DEAD_LATENT_R2',
    'METRIC_SUITE_VERSION',
    'alignment_floor',
    'alignment_gap',
    'bidirectional_r2',
    'bound_is_vacuous',
    'bound_reach',
    'canonical_correlations',
    'compute_all',
    'delta_admissibility',
    'observability_ceiling',
    'orthogonality_gap',
    'predicted_error',
    'probe_accessibility',
    'procrustes_recovery',
    'r2',
    'sigreg_z_score',
    'spectrum',
    'split_alignment_gap',
    'style_invariance',
    'style_variance',
]

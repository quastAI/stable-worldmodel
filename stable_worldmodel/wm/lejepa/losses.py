"""The LeJEPA objective, and the diagnostics logged beside it.

Two of these are optimised (``alignment_loss``, plus ``wm/loss.py::SIGReg``);
everything else is computed under ``no_grad`` and **never** enters the
objective. That separation is load-bearing rather than tidy: ``epsilon`` is
only an honest measurement of the theory's bound because it is independent of
what is being minimised, and ``recovery_diagnostics`` scores against
``latent/z``, which is a *label* -- putting it in the loss would turn an
identifiability claim into supervised regression.

Which diagnostics survive train mode, and which do not
------------------------------------------------------
The encoder is BatchNorm-heavy and ``LeJEPA.encode`` folds the view axis into
the batch, so in **train mode** ``h`` is a function of the whole batch rather
than of one frame. That breaks the premise every quantity in the theory rests
on (a fixed measurable ``h``), and it does not break them all equally:

*Mode-safe* -- :func:`whitening_loss` and :func:`spectrum_diagnostics` read
only the marginal second moment of ``h``, which is measured to agree across
modes.

*Mode-sensitive* -- :func:`bound_diagnostics` reads ``h[0] - h[1]``. Train-mode
BatchNorm normalises both views of a pair with the same batch statistics, which
flatters that difference: measured on a 25-epoch run, ``delta`` read 0.13 in
train mode against 2.69 in eval mode on the same weights. **Log these on the
validation stage only**, and read
``bound_diagnostics(...)['residual_hermite_degree']`` before believing any of
them -- it is below 2 exactly when the measurement is impossible.
"""

import torch


def alignment_loss(h):
    """Pull the views of a positive pair together.

    ``mean over views of ||h_v - mean_v h||^2``. This is the term asked to
    discard style: the two views share their content up to one OU step and
    differ in style entirely, so the only way down is to stop representing
    style.

    Note the units. The paper's ``L`` is ``E||h(z') - h(z)||^2``, a *sum* over
    dimensions; this is a *mean* over ``(V, B, n)``, so ``L = 4 n *
    alignment_loss`` at ``V = 2``. Its floor is therefore
    ``(1 - rho) * tr Cov(h) / (2 n)`` -- 0.048 at ``rho = 0.9``, ``n = 10`` and
    ``tr Cov(h) = 9.6``, not the 0.2 a reading of ``2 (1 - rho) n`` suggests.

    Args:
        h: ``(V, B, N)`` embeddings.

    Returns:
        torch.Tensor: Scalar loss.
    """
    return (h.mean(0) - h).square().mean()


def _pooled_covariance(h):
    """``Cov`` of both views pooled, in float64.

    Pooled rather than view-0-only. Both views are draws from the same
    stationary marginal, so pooling is a lower-variance estimate of the same
    quantity -- and it removes an asymmetry that mattered: the alignment gap
    subtracts ``2 (1 - rho) tr Cov(h)`` from a quantity computed over *both*
    views, so estimating the trace from one view alone biases the gap and can
    push it negative, where the clamp then hides it.
    """
    flat = h.flatten(0, 1).double()
    centred = flat - flat.mean(dim=0)
    return (centred.T @ centred) / max(centred.shape[0] - 1, 1)


@torch.no_grad()
def whitening_loss(h):
    """``||Cov(h) - I||^2_F / n^2``, the metric ``epsilon`` in mean-square form.

    **Computed and logged every step, never optimised.** It appears in the
    bound as a measured quantity, which is only meaningful while it is
    independent of the objective. Mode-safe.

    Args:
        h: ``(V, B, N)`` embeddings.

    Returns:
        torch.Tensor: Scalar, detached.
    """
    cov = _pooled_covariance(h)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return (cov - eye).square().mean()


@torch.no_grad()
def spectrum_diagnostics(h):
    """How many directions of the embedding are actually alive. Mode-safe.

    ``epsilon`` is a Frobenius aggregate, and aggregates hide partial collapse:
    one dead direction out of ``n = 10`` contributes 1 to ``epsilon^2`` and
    sits inside ordinary early-training values without moving them visibly.
    The smallest eigenvalue and the participation ratio do not average it away.

    Args:
        h: ``(V, B, n)`` embeddings.

    Returns:
        dict: ``cov_eig_min`` (0 is a collapsed direction, 1 is the SIGReg
        target), ``cov_eig_max``, ``trace_cov`` (should rise toward ``n``;
        heading to 0 is outright collapse) and ``effective_rank``, the
        participation ratio ``(sum lambda)^2 / sum lambda^2`` in ``[1, n]``,
        which reads directly as "how many directions carry the variance".
    """
    cov = _pooled_covariance(h)
    eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigenvalues.sum()
    squared = eigenvalues.square().sum()
    return {
        'cov_eig_min': eigenvalues.min(),
        'cov_eig_max': eigenvalues.max(),
        'trace_cov': total,
        'effective_rank': total.square() / squared
        if squared > 0
        else torch.zeros_like(total),
    }


def _lstsq_with_intercept(x, y):
    """Least squares of ``y`` on ``[x, 1]``, matching ``metrics``' version.

    ``pinv`` rather than :func:`torch.linalg.lstsq`: numpy's ``lstsq`` returns
    the *minimum-norm* solution, which ``pinv @ y`` reproduces exactly and
    ``lstsq``'s CUDA driver does not. A diagnostic that disagrees with the
    suite it stands in for is worse than no diagnostic.
    """
    design = torch.cat(
        [x, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], dim=1
    )
    solution = torch.linalg.pinv(design) @ y
    return solution, design @ solution


def _r2(x, y):
    """One global sum-of-squares ratio, as ``metrics.r2`` computes it."""
    _, prediction = _lstsq_with_intercept(x, y)
    ss_res = (y - prediction).square().sum()
    ss_tot = (y - y.mean(dim=0)).square().sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else torch.zeros_like(ss_res)


@torch.no_grad()
def recovery_diagnostics(h, z):
    """The criterion metric, live, against the ground-truth latents.

    Ports of ``identifiability.metrics``' ``procrustes_recovery``,
    ``orthogonality_gap`` and ``bidirectional_r2``, in float64 on one batch,
    so the numbers are directly comparable to the results table's columns of
    the same name. They are an estimate at ``B`` samples, not a replacement
    for the frozen suite -- and they are an estimate of the *whole-set*
    aggregate, which a single number cannot localise. The per-latent breakdown
    only exists in ``run_metrics.py``, and it is where a failure is actually
    diagnosed.

    Read ``procrustes_mse_per_dim`` knowing it is **not scale-free**: ``h = 0``
    scores 1.0 and an isotropic uninformative ``h`` scores 2.0, so a falling
    curve that starts near 1.0 is leaving the trivial encoder, not arriving.

    Both views are pooled into one ``(V*B, n)`` sample. That is not ``V*B``
    independent draws -- the views correlate at ``rho`` -- but it lowers the
    variance of the estimate and cannot bias it.

    Args:
        h: ``(V, B, m)`` embeddings.
        z: ``(V, B, n)`` recorded ground-truth latents.

    Returns:
        dict: ``procrustes_mse_per_dim``, ``procrustes_scale``,
        ``orth_err_normalized``, ``cond``, ``r2_z_to_h``, ``r2_h_to_z`` and
        ``recovered_dimensions``. **Empty** when ``m != n``: there is no square
        ``Q`` to align with, and a recovery number against a mismatched width
        would be meaningless rather than merely approximate. Under ``m != n``
        the per-latent probe in ``run_metrics.py`` is the metric that still
        applies.
    """
    n = z.shape[-1]
    if h.shape[-1] != n:
        return {}

    # float64: these are n x n decompositions at n = 10, and a float32 pinv is
    # the one place this and the numpy suite could visibly disagree.
    hh = h.flatten(0, 1).double()
    zz = z.flatten(0, 1).double()

    zc = zz - zz.mean(dim=0)
    hc = hh - hh.mean(dim=0)

    # Procrustes: the best *orthogonal* alignment, because the theory
    # identifies z only up to a rotation.
    cross = (hc.T @ zc) / len(zc)
    u, singular_values, vt = torch.linalg.svd(cross)
    q = u @ vt
    mse = (hc - zc @ q.T).square().sum(dim=-1).mean()

    # The best *linear* map, and how far it is from orthogonal.
    solution, _ = _lstsq_with_intercept(zz, hh)
    a = solution[:n].T
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    gap = torch.linalg.matrix_norm(a.T @ a - eye, ord='fro')
    singular = torch.linalg.svdvals(a)
    smallest = singular.min()

    r2_h_to_z = _r2(hh, zz)
    return {
        'procrustes_mse_per_dim': mse / n,
        'procrustes_scale': singular_values.sum() / n,
        'orth_err_normalized': gap / n**0.5,
        # The one that matters downstream: a map can fit well and still be
        # badly conditioned, and a near-singular direction is a latent any
        # cost built on `h` is nearly flat along.
        'cond': singular.max() / smallest
        if smallest > 0
        else torch.full_like(smallest, float('inf')),
        'r2_z_to_h': _r2(zz, hh),
        'r2_h_to_z': r2_h_to_z,
        # sum of squared canonical correlations. Equals n * r2_h_to_z because
        # every coordinate of z has unit variance by construction, and it is
        # what `bound_diagnostics` needs to floor `delta`.
        'recovered_dimensions': r2_h_to_z * n,
    }


@torch.no_grad()
def bound_diagnostics(h, rho, recovered_dimensions=None):
    """The theory's ``delta`` and the bound it implies. **Validation only.**

    Three unit mismatches make the raw losses unusable against the paper, and
    this closes them: ``L`` is a sum over dimensions where ``align_loss`` is a
    mean, ``epsilon`` is ``||Cov - I||_F`` where ``whitening_loss`` is its
    mean-square, and ``delta`` -- which the paper finds is the *binding* term,
    approximate whitening being "essentially free" -- was not computed at all.

    ``delta`` here is the raw gap; under a stochastic ``g`` it decomposes as
    ``delta_content + 2 rho sigma^2`` and only the first is nonlinearity, but
    ``sigma^2`` needs a same-content/different-style probe that training does
    not have. ``run_metrics.py`` splits it.

    **Do not log this on the training stage.** See the module docstring: the
    difference ``h[0] - h[1]`` is flattered by train-mode BatchNorm, badly
    enough that ``delta`` fell below its own theoretical floor on a real run.

    Args:
        h: ``(V, B, n)`` embeddings, ideally from an eval-mode forward.
        rho: The collector's autocorrelation.
        recovered_dimensions: ``sum sigma^2`` from
            :func:`recovery_diagnostics`. When given, the admissibility gate is
            computed -- the cheapest way to notice that the embeddings were
            produced in a mode these numbers do not apply to.

            **The boolean needs roughly a thousand samples.** ``delta`` is a
            difference of two nearly-equal large numbers, so at a 256-pair
            batch an exactly-degree-2 residual measures a degree of 1.7 and
            the flag false-alarms; by 1024 it reads 2.04 and is reliable. The
            training script therefore logs ``delta`` and ``delta_floor`` as
            separate keys and leaves the comparison to the epoch means, while
            ``run_metrics.py`` evaluates the flag directly at 20k.

    Returns:
        dict: ``L``, ``delta``, ``epsilon``, ``trace_cov``, ``D``,
        ``predicted_error``, ``bound_headroom``, and when
        ``recovered_dimensions`` is given, ``delta_floor``,
        ``residual_hermite_degree`` and ``delta_admissible``.
        ``residual_hermite_degree`` is only interpretable while
        ``trace_cov - recovered_dimensions`` is appreciable; with nothing
        unexplained there is no floor to violate and the flag is trivially
        true.
    """
    n = h.shape[-1]
    loss = (h[0] - h[1]).square().sum(-1).mean()

    cov = _pooled_covariance(h)
    trace = cov.diagonal().sum()
    eye = torch.eye(n, device=cov.device, dtype=cov.dtype)

    delta = (loss - 2.0 * (1.0 - rho) * trace).clamp_min(0.0)
    epsilon = torch.linalg.matrix_norm(cov - eye, ord='fro')

    spectral_gap = 2.0 * rho * (1.0 - rho)
    d = delta / spectral_gap
    predicted = d + (epsilon + d).square()

    out = {
        'L': loss,
        'delta': delta,
        'epsilon': epsilon,
        'trace_cov': trace,
        'D': d,
        'predicted_error': predicted,
        # Negative means the bound exceeds E||z||^2 = n, which `h = 0` already
        # achieves -- at which point the prediction has stopped saying anything.
        'bound_headroom': n - predicted,
    }

    if recovered_dimensions is not None:
        # A style-invariant direction of `h` is a function of `z`, and a
        # function of `z` decorrelates at `rho^k` for its degree-k Hermite
        # content. A direction with zero linear correlation with `z` has degree
        # >= 2, so the residual variance decorrelates at `rho^2` or faster and
        # `delta >= 2 rho (1 - rho) * (tr Cov(h) - recovered)`. Inverting gives
        # the degree the residual actually behaves like; under 2 is impossible.
        residual = (trace - recovered_dimensions).clamp_min(0.0)
        floor = spectral_gap * residual
        corr = rho - delta / (2.0 * residual.clamp_min(1e-9))
        degree = torch.where(
            (corr > 0) & (corr < 1),
            torch.log(corr.clamp(1e-9, 1 - 1e-9))
            / torch.log(
                torch.as_tensor(rho, dtype=corr.dtype, device=corr.device)
            ),
            torch.zeros_like(corr),
        )
        out.update(
            {
                'delta_floor': floor,
                'residual_hermite_degree': degree,
                # 15% slack, and trivially satisfied when there is
                # no unexplained variance to floor. `delta` is a difference of
                # two nearly-equal large numbers, so at batch scale an
                # exactly-degree-2 residual measures ~1.97 and a tight gate
                # false-alarms on honest batches.
                'delta_admissible': (
                    (delta >= 0.85 * floor) | (residual < 0.05 * n)
                ).to(delta.dtype),
            }
        )

    return out


__all__ = [
    'alignment_loss',
    'bound_diagnostics',
    'recovery_diagnostics',
    'spectrum_diagnostics',
    'whitening_loss',
]

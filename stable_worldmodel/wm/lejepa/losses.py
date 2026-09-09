"""The two loss terms LeJEPA adds on top of the repo's existing SIGReg.

Ported from ``lejepa-identifiability/experiments/lejepa_id/losses.py``. The
SIGReg term itself is **not** re-implemented here: ``wm/loss.py::SIGReg`` is
the same Epps-Pulley statistic on the same ``(T, B, D)`` layout, and having two
copies of an isotropy regulariser drift apart is exactly the kind of thing that
would invalidate a violation sweep without anyone noticing.

Both functions take ``h`` of shape ``(V, B, N)`` -- ``V`` views, ``B`` batch,
``N`` embedding dims. For an OU pair ``V = 2``.
"""

import torch


def alignment_loss(h):
    """Pull the views of a positive pair together.

    ``mean over views of ||h_v - mean_v h||^2``. This is the term that is asked
    to discard style: the two views share their content up to one OU step and
    differ in style entirely, so the only way to lower this is to stop
    representing style.

    Args:
        h: ``(V, B, N)`` embeddings.

    Returns:
        torch.Tensor: Scalar loss.
    """
    return (h.mean(0) - h).square().mean()


def whitening_loss(h):
    """``||Cov(h) - I||_F^2``, the metric ``epsilon``.

    **Computed and logged every step, never optimised.** It appears in the
    theory's bound as a measured quantity, and V8 asks how the bound behaves as
    optimisation is cut short -- which is only a question if this term is
    *independent* of what is being minimised. Adding it to the objective would
    make the bound partly self-fulfilling.

    Args:
        h: ``(V, B, N)`` embeddings.

    Returns:
        torch.Tensor: Scalar, detached from the graph.
    """
    with torch.no_grad():
        flat = h.flatten(0, 1)
        flat = flat - flat.mean(dim=0)
        cov = (flat.T @ flat) / (flat.shape[0] - 1)
        eye = torch.eye(cov.shape[0], device=h.device, dtype=cov.dtype)
        return (cov - eye).square().mean()


@torch.no_grad()
def alignment_diagnostics(h, rho):
    """The theory's ``delta`` and ``epsilon``, in the theory's own units.

    **Computed and logged every step, never optimised**, like
    :func:`whitening_loss`.

    Three unit mismatches make the logged losses unusable against the paper as
    they stand, and this closes all three:

    ``L`` vs ``align_loss``
        The paper's alignment loss is ``E||h(z') - h(z)||^2``, a *sum* over
        dimensions. :func:`alignment_loss` is a *mean* over ``(V, B, n)``, so
        ``L = 4 n * align_loss`` at ``V = 2``. Against the exact-recovery floor
        ``2(1 - rho) n``, that puts ``align_loss`` at ``(1 - rho) / 2`` -- 0.05
        at ``rho = 0.9``, not 0.2.
    ``epsilon`` vs ``whitening_metric``
        ``whitening_loss`` returns ``||Cov - I||_F^2 / n^2`` (a mean over the
        matrix), while the bound's ``epsilon`` is ``||Cov - I||_F``. The two
        differ by a factor of ``n``, so they must not be compared.
    ``delta`` was not computed at all
        Yet App. H.8 of the paper finds ``delta`` is the *binding* term and
        approximate whitening "essentially free", so it is the one number worth
        watching. It needs ``tr Cov(h)``, which no logged quantity carried.

    ``delta`` here is the raw gap. Under a stochastic ``g`` it decomposes as
    ``delta_content + 2 rho sigma^2``, and only the first term is nonlinearity
    -- but ``sigma^2`` needs a same-content/different-style probe, which
    training does not have. So this reports the total, and the metric suite
    splits it. A ``delta`` that plateaus well above zero while ``epsilon``
    stays small is the signature of style reaching the output.

    Args:
        h: ``(V, B, n)`` embeddings.
        rho: The collector's autocorrelation. Not otherwise read by training.

    Returns:
        dict: Scalar tensors ``L``, ``delta``, ``epsilon``, ``trace_cov``,
        ``align_floor_deterministic``, and the bound they combine into --
        ``D``, ``predicted_error`` and ``bound_headroom``.
    """
    n = h.shape[-1]
    # The paper's L: a sum over dimensions, a mean over the batch.
    loss = (h[0] - h[1]).square().sum(-1).mean()

    view = h[0]
    centred = view - view.mean(dim=0)
    cov = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    trace = cov.diagonal().sum()
    eye = torch.eye(n, device=h.device, dtype=cov.dtype)

    delta = (loss - 2.0 * (1.0 - rho) * trace).clamp_min(0.0)
    epsilon = torch.linalg.matrix_norm(cov - eye, ord='fro')

    # The bound itself, so it does not have to wait for `run_metrics.py`.
    # `D = delta / (2 rho (1 - rho))` and `predicted = D + (eps + D)^2`, exactly
    # as `metrics.predicted_error` computes them. `bound_headroom = n -
    # predicted` goes negative once the bound exceeds `E||z||^2 = n`, which the
    # trivial encoder `h = 0` already achieves -- at which point the prediction
    # has stopped saying anything and the run is not yet worth continuing on
    # its account.
    spectral_gap = 2.0 * rho * (1.0 - rho)
    d = delta / spectral_gap
    predicted = d + (epsilon + d).square()

    return {
        'L': loss,
        'delta': delta,
        'epsilon': epsilon,
        'trace_cov': trace,
        'align_floor_deterministic': torch.full_like(
            loss, 2.0 * (1.0 - rho) * n
        ),
        'D': d,
        'predicted_error': predicted,
        'bound_headroom': n - predicted,
    }


def _lstsq_with_intercept(x, y):
    """Least squares of ``y`` on ``[x, 1]``, matching ``metrics._lstsq_with_intercept``.

    ``pinv`` rather than :func:`torch.linalg.lstsq`: numpy's ``lstsq`` returns
    the *minimum-norm* solution, which ``pinv @ y`` reproduces exactly and
    ``lstsq``'s CUDA driver does not, and a diagnostic that disagrees with the
    suite it is standing in for is worse than no diagnostic.
    """
    design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], dim=1)
    solution = torch.linalg.pinv(design) @ y
    return solution, design @ solution


def _r2(x, y):
    """``metrics.r2`` in torch: one global SS ratio, not a per-column mean."""
    _, prediction = _lstsq_with_intercept(x, y)
    ss_res = (y - prediction).square().sum()
    ss_tot = (y - y.mean(dim=0)).square().sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else torch.zeros_like(ss_res)


@torch.no_grad()
def recovery_diagnostics(h, z):
    """The criterion metric, live, against the ground-truth latents.

    **Computed and logged every step, never optimised**, like
    :func:`whitening_loss` -- and for the same reason. ``z`` is a *label*: it is
    in every batch because the collector wrote it, and letting it reach the
    objective would turn an identifiability claim into a supervised regression.
    Nothing here is differentiable and nothing here is added to the loss.

    These are ports of ``identifiability.metrics.procrustes_recovery``,
    ``orthogonality_gap`` and ``bidirectional_r2``, computed in float64 on one
    batch instead of numpy on the eval set, so the numbers are directly
    comparable to the scatter's columns of the same name. They are an estimate
    at ``B`` samples, not a replacement for the frozen suite: the scatter still
    comes from ``run_metrics.py``.

    Both views are pooled into one ``(V*B, n)`` sample. That is not ``V*B``
    *independent* draws -- the views are one OU step apart, so they correlate at
    ``rho`` -- but it lowers the variance of the estimate and cannot bias it.

    Args:
        h: ``(V, B, m)`` embeddings.
        z: ``(V, B, n)`` recorded ground-truth latents.

    Returns:
        dict: ``procrustes_mse_per_dim``, ``orth_err_normalized``, ``cond``,
        ``r2_z_to_h``, ``r2_h_to_z``. **Empty** when ``m != n`` -- under a V7
        dimension misspecification there is no square ``Q`` to align with, and
        reporting a recovery number against a mismatched width would be
        meaningless rather than merely approximate.
    """
    n = z.shape[-1]
    if h.shape[-1] != n:
        return {}

    # float64: the suite is numpy, these are n x n decompositions on n = 10, and
    # a float32 pinv is the one place the two could visibly disagree.
    hh = h.flatten(0, 1).double()
    zz = z.flatten(0, 1).double()

    zc = zz - zz.mean(dim=0)
    hc = hh - hh.mean(dim=0)

    # Procrustes: the best *orthogonal* alignment, because the theory identifies
    # z only up to a rotation.
    m = (hc.T @ zc) / len(zc)
    u, _, vt = torch.linalg.svd(m)
    q = u @ vt
    mse = (hc - zc @ q.T).square().sum(dim=-1).mean()

    # The best *linear* map, and how far it is from orthogonal.
    solution, _ = _lstsq_with_intercept(zz, hh)
    a = solution[:n].T
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    gap = torch.linalg.matrix_norm(a.T @ a - eye, ord='fro')
    singular = torch.linalg.svdvals(a)
    smallest = singular.min()

    return {
        'procrustes_mse_per_dim': mse / n,
        'orth_err_normalized': gap / n**0.5,
        # The one that matters for planning: a map can fit well and still be
        # badly conditioned, and a near-singular direction is a latent the
        # planner's cost is nearly flat along.
        'cond': singular.max() / smallest
        if smallest > 0
        else torch.full_like(smallest, float('inf')),
        'r2_z_to_h': _r2(zz, hh),
        'r2_h_to_z': _r2(hh, zz),
    }


@torch.no_grad()
def spectrum_diagnostics(h):
    """How many directions of the embedding are actually alive.

    ``epsilon = ||Cov(h) - I||_F`` is an aggregate, and aggregates hide partial
    collapse: one dead direction out of ``n = 10`` contributes 1 to ``epsilon^2``
    and can sit inside the ordinary early-training value without moving it
    visibly. The smallest eigenvalue does not hide it, and neither does the
    participation ratio.

    Logged, never optimised. Uses both views pooled, as
    :func:`whitening_loss` does.

    Args:
        h: ``(V, B, n)`` embeddings.

    Returns:
        dict: ``cov_eig_min`` (0 is a collapsed direction, 1 is the SIGReg
        target), ``cov_eig_max``, and ``effective_rank`` -- the participation
        ratio ``(sum lambda)^2 / sum lambda^2``, which lies in ``[1, n]`` and
        reads directly as "how many directions carry the variance".
    """
    flat = h.flatten(0, 1).double()
    centred = flat - flat.mean(dim=0)
    cov = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigenvalues.sum()
    squared = eigenvalues.square().sum()
    return {
        'cov_eig_min': eigenvalues.min(),
        'cov_eig_max': eigenvalues.max(),
        'effective_rank': total.square() / squared
        if squared > 0
        else torch.zeros_like(total),
    }


__all__ = [
    'alignment_diagnostics',
    'alignment_loss',
    'recovery_diagnostics',
    'spectrum_diagnostics',
    'whitening_loss',
]

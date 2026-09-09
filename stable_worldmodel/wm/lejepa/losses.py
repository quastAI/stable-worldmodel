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
        dict: Scalar tensors ``L``, ``delta``, ``epsilon``, ``trace_cov`` and
        ``align_floor_deterministic``.
    """
    n = h.shape[-1]
    # The paper's L: a sum over dimensions, a mean over the batch.
    loss = (h[0] - h[1]).square().sum(-1).mean()

    view = h[0]
    centred = view - view.mean(dim=0)
    cov = (centred.T @ centred) / max(centred.shape[0] - 1, 1)
    trace = cov.diagonal().sum()
    eye = torch.eye(n, device=h.device, dtype=cov.dtype)

    return {
        'L': loss,
        'delta': (loss - 2.0 * (1.0 - rho) * trace).clamp_min(0.0),
        'epsilon': torch.linalg.matrix_norm(cov - eye, ord='fro'),
        'trace_cov': trace,
        'align_floor_deterministic': torch.full_like(
            loss, 2.0 * (1.0 - rho) * n
        ),
    }


__all__ = ['alignment_diagnostics', 'alignment_loss', 'whitening_loss']

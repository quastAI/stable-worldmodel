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


__all__ = ['alignment_loss', 'whitening_loss']

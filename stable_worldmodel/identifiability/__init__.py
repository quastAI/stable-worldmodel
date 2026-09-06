"""Identifiability tooling for the LeJEPA Env-1 study.

Four pieces, deliberately separable:

``latents``
    The registry -- what ``z`` is, and the affine between z-space and physical
    units. Everything else reads its notion of ``n`` from here.
``ou``
    The sampler that produces positive pairs, with violations V1-V5 and V9 as
    knobs.
``violations``
    Those knobs as named, severity-laddered configurations, plus the ones that
    live at the environment and the training config instead.
``metrics``
    The frozen metric suite. Versioned separately from the training code, per
    the plan: a metric that moves mid-programme silently invalidates every
    earlier row of the scatter.
``scatter``
    The append-only result table, written from the first run onward because
    several of its columns cannot be reconstructed after the fact.
"""

from .latents import *  # noqa: F403
from .ou import *  # noqa: F403
from .violations import *  # noqa: F403

# `metrics` and `collect` are exposed as submodules rather than star-imported.
# `metrics` is versioned separately and its names (`r2`, `compute_all`) are
# generic enough that flattening them into the package namespace would invite
# an accidental shadow; `collect` is a script-facing helper, not part of the
# library surface.
from . import collect, metrics, scatter  # noqa: F401

"""Identifiability measurement for the LeJEPA encoder study.

Four modules, one per concern:

``latents``
    The registry: what ``z`` *is*, and the affine map between z-space and
    physical units. The single place that decides what coordinate 7 means.
``ou``
    The sampler that draws ``(z, z')`` positive pairs from a stationary,
    isotropic, Gaussian OU process.
``collect``
    Drives the environment to render those pairs into a dataset, and audits
    the result against what the manifest declares.
``metrics``
    The frozen suite: does ``h`` recover ``z``, and is the answer usable.
``results``
    Append-only per-checkpoint results rows.
"""

from . import collect, latents, metrics, ou, results  # noqa: F401
from .latents import PROFILES, Latent, LatentRegistry, build_registry
from .metrics import compute_all
from .ou import SIGMA_SPAN, OUSampler, spectral_gap
from .results import append_row, build_row, load_rows


__all__ = [
    'PROFILES',
    'SIGMA_SPAN',
    'Latent',
    'LatentRegistry',
    'OUSampler',
    'append_row',
    'build_registry',
    'build_row',
    'collect',
    'compute_all',
    'latents',
    'load_rows',
    'metrics',
    'ou',
    'results',
    'spectral_gap',
]

"""Collect the LeJEPA encoder dataset: OU latent pairs on OGBench Cube-single.

Each positive pair is written as a two-step episode, so it reads back through
the stock loader with ``num_steps=2, frameskip=1`` and no new code.

Unlike every other ``collect_*.py`` in this directory, this one does **not**
use ``World.collect``: that drives a policy through stepped physics, and
encoder data must never step physics -- every frame is an independent draw
from the OU sampler. It drives the environment directly through
``LeJEPACubeEnv.render_content`` and writes through ``LanceWriter``, which
``World.collect`` also uses.

Throughput, measured at 224x224 on the reference machine: ~165 frames/s, so
200k pairs (400k frames) is about 0.67 core-hours in one process. The same
pipeline routed through ``env.reset`` would cost 29.2 core-hours, because
``cube.size``, ``agent.color``, ``floor.color`` and ``camera.angle_delta``
force a full MJCF recompile on every reset. See
``stable_worldmodel/identifiability/collect.py`` for the measurements.

Collection is single-process. To use more cores, run several with
``shard=i num_shards=N``; each takes a disjoint slice of ``num_pairs`` and a
disjoint block of sampler seeds, and writes its own ``*_shard{i}`` dataset for
``swm merge`` to concatenate.

Usage::

    python scripts/data/collect_cube_single_ou.py num_pairs=200000
    python scripts/data/collect_cube_single_ou.py latents.profile=all_content
    python scripts/data/collect_cube_single_ou.py violation.name=v4 violation.severity=0.32
    python scripts/data/collect_cube_single_ou.py num_pairs=200000 shard=0 num_shards=8
"""

import os
import sys
from pathlib import Path


# Pick a renderer that will actually start, and never override an explicit
# export. `glfw` needs a display; `egl` needs a GPU and is rejected on macOS;
# `osmesa` is software rasterization and always works. On a Linux node with a
# GPU, export MUJOCO_GL=egl yourself -- rendering is ~90% of this script's
# runtime, so that choice dominates its throughput.
if 'MUJOCO_GL' not in os.environ:
    if sys.platform == 'darwin' or os.environ.get('DISPLAY'):
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger as logging  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.ogbench.lejepa_cube_env import (  # noqa: E402,F401
    LeJEPACubeEnv,
)
from stable_worldmodel.identifiability import collect as ident_collect  # noqa: E402
from stable_worldmodel.identifiability.latents import (  # noqa: E402
    PROFILES,
    build_registry,
)
from stable_worldmodel.identifiability.ou import OUSampler  # noqa: E402
from stable_worldmodel.identifiability.violations import (  # noqa: E402
    make_violation,
)


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='ogb_cube_single_ou',
)
def run(cfg: DictConfig):
    """Collect one shard (by default, all) of the encoder dataset."""

    shard = int(cfg.get('shard') or 0)
    num_shards = int(cfg.get('num_shards') or 1)
    if not 0 <= shard < num_shards:
        raise ValueError(
            f'shard must be in [0, num_shards); got {shard} of {num_shards}.'
        )

    pairs = cfg.num_pairs // num_shards + (
        1 if shard < cfg.num_pairs % num_shards else 0
    )
    if pairs == 0:
        logging.warning(f'shard {shard}: no pairs to collect, exiting.')
        return

    # Shards must not overlap in the sampler's stream, or the "dataset" would
    # silently contain each pair `num_shards` times over.
    base_seed = int(cfg.seed) + shard * 1_000_003

    env = LeJEPACubeEnv(
        env_type=cfg.env.env_type,
        ob_type='pixels',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=cfg.env.image_size,
        height=cfg.env.image_size,
        num_digits=cfg.env.num_digits,
        marker_enabled=cfg.env.marker.enabled,
        marker_face=cfg.env.marker.face,
        marker_scale=cfg.env.marker.scale,
    )
    env.reset(seed=base_seed, options={'variation': ['all']})

    # ---------------------------------------------------------- registry
    registry = build_registry(
        env,
        yaw_half_arc=float(cfg.latents.yaw_half_arc),
        include_roll_pitch=bool(cfg.latents.include_roll_pitch),
        include_discrete=bool(cfg.latents.include_discrete),
        pos_z_max=float(cfg.latents.pos_z_max),
    )
    profile_name = str(cfg.latents.profile)
    if profile_name not in PROFILES:
        raise KeyError(
            f'unknown latent profile {profile_name!r}; expected one of '
            f'{sorted(PROFILES)}.'
        )
    registry = registry.resolve(PROFILES[profile_name])
    logging.info(f'latent profile "{profile_name}":\n{registry.summary()}')

    # --------------------------------------------------------- violation
    violation = make_violation(
        str(cfg.violation.name), severity=float(cfg.violation.severity)
    )
    if violation.site != 'sampler':
        logging.warning(
            f'violation {violation.key} binds at the {violation.site}, not at '
            'the sampler -- this dataset is the unviolated one, and the knob '
            'is applied later.'
        )

    rho_bar = float(cfg.program_constants.rho)
    sampler_kwargs = {
        'rho': rho_bar,
        'dist': cfg.ou.dist,
        'alpha': cfg.ou.alpha,
        'noise_coupling': float(cfg.ou.noise_coupling),
        'cross_corr': float(cfg.ou.cross_corr),
    }
    sampler_kwargs.update(violation.sampler_kwargs(registry.n, rho_bar))
    sampler = OUSampler(n=registry.n, seed=base_seed, **sampler_kwargs)

    isotropy = sampler.describe()['isotropy']
    logging.info(
        f'rho mean {np.mean(sampler.rho):.4f}; isotropy criterion '
        f'{"satisfied" if isotropy["satisfied"] else "VIOLATED"} '
        f'(margin {isotropy["margin"]:+.5f})'
    )

    # ------------------------------------------------------------- write
    dataset_name = str(cfg.dataset_name)
    if num_shards > 1:
        stem, _, suffix = dataset_name.rpartition('.')
        dataset_name = f'{stem}_shard{shard}.{suffix}'

    out_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / dataset_name
    )

    manifest = ident_collect.build_manifest(
        env,
        registry,
        sampler,
        violation,
        pairs,
        base_seed,
        profile_name,
        extra={
            'shard': shard,
            'num_shards': num_shards,
            'program_constants': OmegaConf.to_container(
                cfg.program_constants, resolve=True
            ),
        },
    )

    logging.info(
        f'shard {shard}/{num_shards}: {pairs} pairs, n = {registry.n}, '
        f'config_hash {manifest["config_hash"]} -> {out_path}'
    )

    with swm.data.LanceWriter(out_path, mode=cfg.write_mode) as writer:
        # `write_chunked`, not `write_episodes`: Lance drains a streamed
        # iterable from its own background thread, and this generator renders.
        # A MuJoCo OpenGL context belongs to the thread that created it, so
        # letting the writer pull frames deadlocks. See `write_chunked`.
        ident_collect.write_chunked(
            writer,
            ident_collect.collect_pairs(
                env,
                registry,
                sampler,
                pairs,
                seed=base_seed + 7,
                camera=cfg.env.camera,
            ),
        )

    sidecar = ident_collect.write_manifest(out_path, manifest)
    env.close()

    logging.success(
        f'🎉 encoder dataset written -> {out_path} (manifest {sidecar})'
    )


if __name__ == '__main__':
    run()

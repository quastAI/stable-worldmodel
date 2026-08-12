"""Collect domain-randomized cube-quadruple expert demonstrations.

Runs the OGBench plan oracle on ``swm/OGBCubeDR-v0`` in data-collection mode.
Each episode chains ~11 pick-and-places (matching upstream's
``cube-quadruple-play-v0`` recipe of 1001-step episodes) and resamples the
domain randomization, so the recorded ``variation.*`` columns are what
``DRCubeEnv.reset_options_from_dataset`` later replays at evaluation time.

Usage::

    python scripts/data/collect_cube_quadruple_dr.py num_traj=5000
"""

import os
from pathlib import Path

# glfw needs a display, so headless Linux nodes must export MUJOCO_GL=egl (or
# osmesa) before launching. Only default it, never override an explicit choice.
os.environ.setdefault('MUJOCO_GL', 'glfw')
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='ogb_cube_quadruple_dr',
)
def run(cfg: DictConfig):
    """Run parallel data collection for the DR cube-quadruple env."""

    world = swm.World(
        'swm/OGBCubeDR-v0',
        **cfg.world,
        **cfg.env,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy(policy_type=cfg.policy_type))

    out_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / cfg.dataset_name
    )

    world.collect(
        out_path,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )

    logging.success(f'🎉 Completed DR cube-quadruple collection -> {out_path}')


if __name__ == '__main__':
    run()

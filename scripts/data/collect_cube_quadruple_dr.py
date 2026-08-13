"""Collect domain-randomized cube-quadruple expert demonstrations.

Runs the OGBench plan oracle on ``swm/OGBCubeDR-v0`` in data-collection mode.
Each episode chains pick-and-places -- how many, and how often one lands on
top of another, is set by ``world.max_episode_steps`` and ``p_stack`` (see
below) -- and resamples the domain randomization, so the recorded
``variation.*`` columns are what ``DRCubeEnv.reset_options_from_dataset``
later replays at evaluation time.

The shipped default (``p_stack: null``, 1000-step episodes) matches
upstream's ``cube-quadruple-play-v0`` recipe: ~11 chained pick-and-places per
episode at the stock 10-50% per-episode stacking rate, for general-purpose
representation learning. Raising ``p_stack`` biases collection toward
stacking events, at the cost of matching that recipe -- see ``p_stack``'s
docstring in :class:`~stable_worldmodel.envs.ogbench.ExpertPolicy` for why
that alone still won't produce episodes that end on a *held* 4-tower.

Usage::

    python scripts/data/collect_cube_quadruple_dr.py num_traj=5000
    python scripts/data/collect_cube_quadruple_dr.py p_stack=[0.7,0.8] world.max_episode_steps=500
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
    p_stack = cfg.get('p_stack')
    # `OmegaConf.to_object` turns a `[lo, hi]` YAML entry into a plain list;
    # ExpertPolicy checks `isinstance(..., (list, tuple))`, which the raw
    # `ListConfig` node would fail.
    p_stack = OmegaConf.to_object(p_stack) if p_stack is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy(policy_type=cfg.policy_type, p_stack=p_stack))

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

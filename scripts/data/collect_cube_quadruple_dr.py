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

Collection is single-threaded -- ``EnvPool`` steps its envs in a Python loop,
so ``world.num_envs`` batches but does not parallelize. To use more than one
core, run several *processes* with ``shard=i num_shards=N``; each takes a
disjoint slice of ``num_traj`` and a disjoint block of episode seeds, and
writes its own ``*_shard{i}`` dataset for ``swm merge`` to concatenate.
``collect_cube_quadruple_dr_sharded.py`` drives that for you.

Usage::

    python scripts/data/collect_cube_quadruple_dr.py num_traj=5000
    python scripts/data/collect_cube_quadruple_dr.py p_stack=[0.7,0.8] world.max_episode_steps=500
    python scripts/data/collect_cube_quadruple_dr.py num_traj=5000 shard=0 num_shards=4
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
    """Collect one shard (by default, all) of the DR cube-quadruple dataset."""

    shard = int(cfg.get('shard') or 0)
    num_shards = int(cfg.get('num_shards') or 1)
    if not 0 <= shard < num_shards:
        raise ValueError(
            f'shard must be in [0, num_shards); got {shard} of {num_shards}.'
        )

    # Split the episode budget as evenly as the shard count allows.
    episodes = cfg.num_traj // num_shards + (
        1 if shard < cfg.num_traj % num_shards else 0
    )
    if episodes == 0:
        logging.warning(f'shard {shard}: no episodes to collect, exiting.')
        return

    # `World` seeds episodes from a contiguous integer run starting at the
    # base seed -- `num_envs` for the opening reset, then one per episode --
    # so two shards whose blocks overlap would silently record the *same*
    # episodes twice. Striding by the full budget keeps them disjoint whatever
    # the split, and keeps the whole dataset regenerable from `cfg.seed`.
    stride = int(cfg.num_traj) + int(cfg.world.num_envs)
    base_seed = int(cfg.seed) + shard * stride

    # OGBench's `CubePlanOracle` draws its waypoint and timing jitter straight
    # from the global numpy RNG and exposes no seed hook, so seeding the env
    # and the policy is not enough to pin a trajectory down. This makes the
    # *process* reproducible: identical output requires re-running with the
    # same `shard` / `num_shards` / `world.num_envs`, since the global stream
    # is consumed across episodes rather than reset per episode.
    np.random.seed(base_seed)

    dataset_name = cfg.dataset_name
    if num_shards > 1:
        stem, _, suffix = str(dataset_name).rpartition('.')
        dataset_name = f'{stem}_shard{shard}.{suffix}'

    world = swm.World(
        'swm/OGBCubeDR-v0',
        **cfg.world,
        **cfg.env,
        mode='data_collection',
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    p_stack = cfg.get('p_stack')
    # A `[lo, hi]` YAML entry parses to a `ListConfig`; `OmegaConf.to_object`
    # turns it into a plain list, since ExpertPolicy checks
    # `isinstance(..., (list, tuple))`, which `ListConfig` itself would fail.
    # A scalar entry parses to a plain float already, and `to_object` rejects
    # non-container input outright, so only convert actual config nodes.
    if OmegaConf.is_config(p_stack):
        p_stack = OmegaConf.to_object(p_stack)
    # The policy has its own RNG for action noise and the per-episode `p_stack`
    # draw; left unseeded it defaults to OS entropy, which would make the run
    # irreproducible even though every reset is now seeded.
    world.set_policy(
        ExpertPolicy(
            policy_type=cfg.policy_type, p_stack=p_stack, seed=base_seed
        )
    )

    out_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / dataset_name
    )

    logging.info(
        f'shard {shard}/{num_shards}: {episodes} episodes, '
        f'seeds [{base_seed}, {base_seed + episodes + cfg.world.num_envs}) '
        f'-> {out_path}'
    )

    world.collect(
        out_path,
        episodes=episodes,
        seed=base_seed,
        options=options,
    )

    logging.success(f'🎉 Completed DR cube-quadruple collection -> {out_path}')


if __name__ == '__main__':
    run()

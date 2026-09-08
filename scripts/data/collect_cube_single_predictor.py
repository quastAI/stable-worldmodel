"""Collect the predictor dataset: physics rollouts on Cube-single.

A much thinner script than the encoder collector, and deliberately so. Per plan
SS0.1, **V1-V4 and V9 do not apply here**: this dataset is not trying to
satisfy a marginal, it is collected for its own requirements -- action-space
coverage, persistent excitation, and task-relevant state visitation. It is
``collect_cube_quadruple_dr.py`` with ``env_type='single'``, the LeJEPA env,
and three additions.

**1. A policy mixture.** Excitation is the design target, and a purely expert
action distribution is a poor basis for a transition model that the planner
will drive off-distribution. ``policy_mixture`` blends the OGBench expert with
random actions.

**2. Style range must match the encoder dataset's.** Appearance outside the
range the encoder was trained on is not covered by the alignment loss and leaks
straight into the embedding (plan SS0.1, second caveat). The script reads the
encoder dataset's manifest and **refuses to run on a mismatch** rather than
producing a dataset that would quietly violate the assumption.

**3. A rollout-distribution eval split.** A held-out slice is tagged
``eval/rollout``. Every identifiability metric is reported on *both* this and
the OU eval set, and the gap between them is a logged measurement -- the
encoder trains on OU-set states and the planner only ever sees rollouts.

Usage::

    python scripts/data/collect_cube_single_predictor.py num_traj=5000
    python scripts/data/collect_cube_single_predictor.py policy_mixture=0.5
"""

import json
import os
import sys
from pathlib import Path


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
from stable_worldmodel.envs.ogbench import ExpertPolicy  # noqa: E402
from stable_worldmodel.identifiability.collect import load_manifest  # noqa: E402
from stable_worldmodel.envs.ogbench.lejepa_cube_env import (  # noqa: E402,F401
    LeJEPACubeEnv,
)


class MixturePolicy:
    """Expert actions with a random-action fraction mixed in per episode.

    Persistent excitation, not performance, is what this dataset needs: a
    transition model fitted only to expert trajectories has never seen the
    off-distribution states a planner's candidate rollouts will visit.

    Args:
        expert: The OGBench ``ExpertPolicy``.
        action_space: For drawing the random component.
        random_fraction: Probability that any given step is random.
        seed: RNG seed.
    """

    def __init__(self, expert, action_space, random_fraction=0.25, seed=0):
        self.expert = expert
        self.action_space = action_space
        self.random_fraction = float(random_fraction)
        self.rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        # Everything the World policy protocol needs but this wrapper does not
        # override (reset hooks, etc.) goes straight to the expert.
        return getattr(self.expert, name)

    def get_action(self, *args, **kwargs):
        action = self.expert.get_action(*args, **kwargs)
        action = np.asarray(action)
        mask = self.rng.random(action.shape[:1]) < self.random_fraction
        if mask.any():
            noise = np.stack(
                [self.action_space.sample() for _ in range(int(mask.sum()))]
            )
            action[mask] = noise
        return action


def assert_style_range_matches(encoder_dataset, env, cache_dir=None):
    """Refuse to run if appearance randomisation differs from the encoder's.

    Not a warning. Appearance outside the encoder's training range is exactly
    the condition under which the alignment loss provides no invariance
    guarantee, and the resulting leakage would show up as an unexplained gap
    between the OU and rollout metrics -- attributed, wrongly, to distribution
    shift in the *states*.

    Args:
        encoder_dataset: Name of the encoder dataset this predictor set will be
            paired with, e.g. ``'ogbench/cube_single_ou_physical_content.lance'``.
            Its sidecar manifest is what carries the declared env config.
        env: The live unwrapped env to check against.
        cache_dir: Cache root; defaults to ``STABLEWM_HOME``.
    """
    manifest = load_manifest(encoder_dataset, cache_dir)
    declared = manifest['env']
    mismatches = []
    for key, actual in (
        ('num_cubes', env._num_cubes),
        ('num_digits', env._num_digits),
        ('marker_enabled', env._marker_enabled),
        ('marker_face', env._marker_face),
    ):
        if key in declared and declared[key] != actual:
            mismatches.append(f'{key}: encoder={declared[key]} here={actual}')

    if mismatches:
        raise ValueError(
            'predictor env does not match the encoder dataset it will be '
            'paired with:\n  ' + '\n  '.join(mismatches)
        )
    return manifest


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='ogb_cube_single_predictor',
)
def run(cfg: DictConfig):
    """Collect the predictor dataset."""
    shard = int(cfg.get('shard') or 0)
    num_shards = int(cfg.get('num_shards') or 1)
    episodes = cfg.num_traj // num_shards + (
        1 if shard < cfg.num_traj % num_shards else 0
    )
    if episodes == 0:
        logging.warning(f'shard {shard}: nothing to collect.')
        return

    stride = int(cfg.num_traj) + int(cfg.world.num_envs)
    base_seed = int(cfg.seed) + shard * stride
    np.random.seed(base_seed)

    world = swm.World(
        'swm/LeJEPACube-v0', **cfg.world, **cfg.env, mode='data_collection'
    )

    # The gate: refuse a mismatch rather than record one.
    assert_style_range_matches(
        cfg.encoder_dataset,
        world.envs.envs[0].unwrapped,
        cfg.get('cache_dir'),
    )

    expert = ExpertPolicy(
        policy_type=cfg.policy_type, p_stack=cfg.get('p_stack'), seed=base_seed
    )
    world.set_policy(
        MixturePolicy(
            expert,
            world.envs.single_action_space,
            random_fraction=float(cfg.policy_mixture),
            seed=base_seed,
        )
        if float(cfg.policy_mixture) > 0
        else expert
    )

    dataset_name = str(cfg.dataset_name)
    if num_shards > 1:
        stem, _, suffix = dataset_name.rpartition('.')
        dataset_name = f'{stem}_shard{shard}.{suffix}'
    out_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / dataset_name
    )

    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None

    logging.info(
        f'shard {shard}/{num_shards}: {episodes} episodes, '
        f'{cfg.policy_mixture:.0%} random actions -> {out_path}'
    )
    world.collect(
        out_path, episodes=episodes, seed=base_seed, options=options
    )

    # The rollout-distribution eval split. Recorded as a sidecar rather than a
    # column so the split can be recomputed without rewriting the dataset.
    n_eval = int(episodes * float(cfg.eval_fraction))
    split = {
        'eval_rollout_episodes': list(range(episodes - n_eval, episodes)),
        'num_episodes': episodes,
        'note': 'every identifiability metric is reported on both this and '
        'the OU eval set; the gap between them is itself a measurement.',
    }
    with open(out_path.parent / f'{out_path.stem}_split.json', 'w') as handle:
        json.dump(split, handle, indent=2)

    logging.success(f'🎉 predictor dataset -> {out_path}')


if __name__ == '__main__':
    run()

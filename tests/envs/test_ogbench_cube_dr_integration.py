"""End-to-end check of the DR cube pipeline: collect -> subgoal evaluation.

Small but real: a genuine ``World``, the OGBench plan oracle, a lance dataset
on disk, and ``World.evaluate`` walking an oracle-subgoal chain. This is the
only test that exercises the *dataset column names*, which differ between the
info dict (``variation.a.b``) and the lance schema (``variation_a_b``).
"""

import os

import numpy as np
import pytest


os.environ.setdefault('MUJOCO_GL', 'glfw')

pytest.importorskip('ogbench')
pytest.importorskip('lance')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.ogbench import ExpertPolicy  # noqa: E402
from stable_worldmodel.world.world import _extract_init_goal  # noqa: E402


ENV_KWARGS = dict(
    env_type='quadruple',
    ob_type='states',
    multiview=False,
    width=64,
    height=64,
    visualize_info=False,
    num_digits=3,
    num_bg_materials=5,
)


@pytest.fixture(scope='module')
def collected(tmp_path_factory):
    path = tmp_path_factory.mktemp('dr_data') / 'cube_quadruple_dr.lance'
    world = swm.World(
        'swm/OGBCubeDR-v0',
        num_envs=2,
        image_shape=(64, 64),
        max_episode_steps=30,
        **ENV_KWARGS,
        terminate_at_goal=False,
        mode='data_collection',
    )
    world.set_policy(ExpertPolicy(policy_type='plan_oracle', seed=0))
    world.collect(
        path,
        episodes=2,
        seed=0,
        options={'variation': ['all']},
        progress=False,
    )
    world.close()
    return swm.data.load_dataset(str(path))


@pytest.fixture(scope='module')
def eval_world():
    world = swm.World(
        'swm/OGBCubeDR-v0',
        num_envs=2,
        image_shape=(64, 64),
        max_episode_steps=60,
        **ENV_KWARGS,
        terminate_at_goal=True,
    )
    world.set_policy(swm.policy.RandomPolicy(seed=0))
    yield world
    world.close()


def test_dr_columns_are_recorded(collected):
    cols = set(collected.column_names) | set(
        getattr(collected, '_schema_names', ())
    )
    assert 'qpos' in cols and 'qvel' in cols
    assert any(c.startswith('variation') for c in cols)
    assert 'privileged/digit_0_value' in cols
    assert 'privileged/digit_0_pos' in cols


def test_every_variation_axis_round_trips(collected, eval_world):
    """Guards the info-key vs lance-column naming mismatch."""
    init_rows, goal_rows, _ = _extract_init_goal(
        collected, [0, 1], [0, 0], 5, 2
    )
    env = eval_world.envs.envs[0].unwrapped
    opts = env.reset_options_from_dataset(init_rows[0], goal_rows[0][-1])

    assert len(opts['variation_values']) == len(env.variation_space.names())

    eval_world.envs.envs[0].reset(options=opts)
    for name in ('background.floor_material', 'digit.value', 'cube.color'):
        space = env.variation_space
        for part in name.split('.'):
            space = space[part]
        col = f'variation_{name.replace(".", "_")}'
        np.testing.assert_allclose(
            np.asarray(space.value, dtype=np.float64).reshape(-1),
            np.asarray(init_rows[0][col], dtype=np.float64).reshape(-1),
        )


def test_subgoal_evaluation_runs(collected, eval_world, tmp_path):
    results = eval_world.evaluate(
        dataset=collected,
        episodes_idx=[0, 1],
        start_steps=[0, 0],
        goal_offset=4,
        num_subgoals=3,
        subgoal_budget=3,
        subgoal_advance='both',
        eval_budget=9,
        video=tmp_path,
    )

    assert results['subgoal_index'].shape == (2,)
    assert results['subgoal_index'].max() >= 1, 'chain never advanced'
    assert results['subgoal_index'].max() <= 2
    assert len(list(tmp_path.glob('*.mp4'))) == 2

"""Tests for the domain-randomized OGBench cube environment.

Skipped wholesale when ``ogbench`` is not installed -- it lives in the
``env`` extra, not the base dependency set.
"""

import os

import numpy as np
import pytest


os.environ.setdefault('MUJOCO_GL', 'glfw')

pytest.importorskip('ogbench')
pytest.importorskip('mujoco')

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv  # noqa: E402
from stable_worldmodel.envs.ogbench.dr_cube_env import (  # noqa: E402
    DRCubeEnv,
    render_digit_png,
)


# Every axis DRCubeEnv adds itself; all of these must be applied to the
# already-compiled model, never by rebuilding it.
RECOMPILE_FREE_VARIATIONS = [
    'light.position',
    'light.direction',
    'light.diffuse',
    'light.ambient',
    'light.specular',
    'light.headlight_diffuse',
    'background.floor_material',
    'background.wall_material',
    'digit.value',
    'digit.position',
    'digit.yaw',
    'cube.start_position',
    'cube.start_yaw',
    'cube.goal_position',
    'cube.goal_yaw',
]


def make_env(**kwargs):
    defaults = dict(
        env_type='quadruple',
        ob_type='states',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=64,
        height=64,
        num_digits=3,
        num_bg_materials=5,
    )
    defaults.update(kwargs)
    return DRCubeEnv(**defaults)


@pytest.fixture(scope='module')
def env():
    e = make_env()
    e.reset(seed=0)
    return e


class TestDigitAssets:
    def test_png_is_deterministic_per_digit(self):
        assert render_digit_png(3) == render_digit_png(3)
        assert render_digit_png(3) != render_digit_png(4)

    def test_png_decodes(self):
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(render_digit_png(7, size=64)))
        assert img.size == (64, 64)


class TestVariationSpace:
    def test_new_axes_present(self, env):
        names = set(env.variation_space.names())
        for key in RECOMPILE_FREE_VARIATIONS:
            assert key in names

    def test_inherited_axes_survive(self, env):
        names = set(env.variation_space.names())
        assert {'cube.size', 'cube.color', 'floor.color', 'agent.color'} <= (
            names
        )

    def test_defaults_are_in_bounds(self, env):
        assert env.variation_space.check(debug=True)

    def test_sampling_stays_in_bounds(self):
        e = make_env()
        for seed in range(3):
            e.reset(seed=seed, options={'variation': ['all']})
            assert e.variation_space.check(debug=True)

    def test_digit_space_omitted_when_disabled(self):
        e = make_env(num_digits=0)
        assert 'digit.value' not in set(e.variation_space.names())


class TestNoRecompile:
    def test_dr_axes_never_recompile(self):
        e = make_env()
        e.reset(seed=0)

        calls = []
        original = e.compile_model_and_data
        e.compile_model_and_data = lambda: (calls.append(1), original())[1]

        for seed in range(5):
            e.reset(
                seed=seed, options={'variation': RECOMPILE_FREE_VARIATIONS}
            )
        assert calls == []

    def test_cube_size_does_recompile(self):
        """Sanity check on the counter: mass/inertia genuinely need a rebuild."""
        e = make_env()
        e.reset(seed=0)

        calls = []
        original = e.compile_model_and_data
        e.compile_model_and_data = lambda: (calls.append(1), original())[1]

        e.reset(seed=1, options={'variation': ['cube.size']})
        assert len(calls) == 1


class TestVisualVariationApplied:
    def test_values_actually_reach_the_model(self):
        e = make_env()
        seen_floor, seen_digit_mat, seen_light = set(), set(), set()
        for seed in range(8):
            e.reset(
                seed=seed, options={'variation': RECOMPILE_FREE_VARIATIONS}
            )
            seen_floor.add(int(e._model.geom_matid[e._floor_geom_id]))
            seen_digit_mat.add(int(e._model.geom_matid[e._digit_geom_ids[0]]))
            seen_light.add(tuple(np.round(e._model.light_pos[0], 5)))

        assert len(seen_floor) > 1
        assert len(seen_digit_mat) > 1
        assert len(seen_light) > 1

    def test_light_direction_is_normalized(self):
        e = make_env()
        e.reset(seed=4, options={'variation': ['light.direction']})
        for lid in e._light_ids:
            np.testing.assert_allclose(
                np.linalg.norm(e._model.light_dir[lid]), 1.0, atol=1e-6
            )

    def test_digit_mocaps_follow_the_sampled_positions(self):
        e = make_env()
        e.reset(seed=5, options={'variation': ['digit.position']})
        want = e.variation_space['digit']['position'].value
        for i, mocap_id in enumerate(e._digit_mocap_ids):
            np.testing.assert_allclose(
                e._data.mocap_pos[mocap_id][:2], want[i], atol=1e-9
            )


class TestPrivilegedInfo:
    def test_digit_ground_truth_exported(self, env):
        _, info = env.reset(seed=2, options={'variation': ['digit.value']})
        want = env.variation_space['digit']['value'].value
        for i in range(env._num_digits):
            assert int(info[f'privileged/digit_{i}_value'][0]) == int(want[i])
            assert info[f'privileged/digit_{i}_pos'].shape == (2,)

    def test_background_and_light_exported(self, env):
        _, info = env.reset(seed=2)
        assert info['privileged/floor_material'].shape == (1,)
        assert info['privileged/wall_material'].shape == (1,)
        assert info['privileged/light_pos'].shape == (2 * 3,)

    def test_cube_keys_still_there(self, env):
        _, info = env.reset(seed=2)
        for i in range(env._num_cubes):
            assert f'privileged/block_{i}_pos' in info
        assert info['privileged/target_task'] == 'cube'


class TestSizeAwareGeometry:
    def test_cubes_rest_on_their_own_half_extent(self):
        e = make_env()
        e.reset(
            seed=7,
            options={'variation': ['cube.size', 'cube.start_position']},
        )
        sizes = e.variation_space['cube']['size'].value
        for i in range(e._num_cubes):
            z = e._data.joint(f'object_joint_{i}').qpos[2]
            assert abs(z - sizes[i]) < 5e-3, (i, z, sizes[i])

    def test_matches_parent_at_the_stock_size(self):
        """With size_aware_geometry off, placement matches CubeEnv exactly."""
        e = make_env(size_aware_geometry=False)
        e.reset(seed=7, options={'variation': ['cube.start_position']})
        for i in range(e._num_cubes):
            z = e._data.joint(f'object_joint_{i}').qpos[2]
            assert abs(z - 0.02) < 1e-9

    def test_task_waypoints_rescale(self):
        e = make_env(mode='task', permute_blocks=False)
        e.reset(seed=1, options={'variation': ['cube.size']})
        # task5_stack is a 4-high tower; rebuild the expected ladder.
        stock = CubeEnv.set_tasks
        assert stock is not None  # the ladder lives in the parent
        rescaled = e._rescaled_task_info(e.task_infos[4])
        halves = e.variation_space['cube']['size'].value
        for i, half in enumerate(halves):
            assert rescaled['goal_xyzs'][i][2] == pytest.approx(
                half + 2 * half * i
            )


class TestDatasetReset:
    def _row_from(self, e, info):
        row = {'qpos': info['qpos'].copy(), 'qvel': info['qvel'].copy()}
        for name in e.variation_space.names():
            space = e.variation_space
            for part in name.split('.'):
                space = space[part]
            row[f'variation.{name}'] = np.asarray(space.value)
        return row

    def _goal_from(self, e):
        goal = {}
        for i in range(e._num_cubes):
            goal[f'goal_privileged/block_{i}_pos'] = (
                e._data.joint(f'object_joint_{i}').qpos[:3].copy()
            )
        return goal

    def test_state_and_appearance_round_trip(self):
        src = make_env()
        _, info = src.reset(seed=11, options={'variation': ['all']})
        row = self._row_from(src, info)
        goal = self._goal_from(src)

        dst = make_env()
        opts = dst.reset_options_from_dataset(row, goal)
        assert opts['variation'] == []
        _, info2 = dst.reset(seed=11, options=opts)

        np.testing.assert_allclose(info2['qpos'], row['qpos'], atol=1e-9)
        np.testing.assert_allclose(info2['qvel'], row['qvel'], atol=1e-9)
        for name in ('background.floor_material', 'digit.value'):
            a, b = src.variation_space, dst.variation_space
            for part in name.split('.'):
                a, b = a[part], b[part]
            np.testing.assert_allclose(
                np.asarray(a.value, dtype=np.float64),
                np.asarray(b.value, dtype=np.float64),
            )

    def test_targets_come_from_the_goal_row(self):
        src = make_env()
        _, info = src.reset(seed=12, options={'variation': ['all']})
        row = self._row_from(src, info)
        goal = self._goal_from(src)

        dst = make_env()
        dst.reset(seed=12, options=dst.reset_options_from_dataset(row, goal))
        for i in range(dst._num_cubes):
            np.testing.assert_allclose(
                dst._data.mocap_pos[dst._cube_target_mocap_ids[i]],
                goal[f'goal_privileged/block_{i}_pos'],
                atol=1e-9,
            )

    def test_float32_columns_survive_the_bounds_check(self):
        """Datasets downcast to float32; a pinned/edge value must still load."""
        src = make_env()
        _, info = src.reset(seed=13, options={'variation': ['all']})
        row = self._row_from(src, info)
        row = {
            k: (
                np.asarray(v, dtype=np.float32)
                if k.startswith('variation.')
                else v
            )
            for k, v in row.items()
        }

        dst = make_env()
        dst.reset(
            seed=13,
            options=dst.reset_options_from_dataset(row, self._goal_from(src)),
        )

    def test_underscore_column_names_accepted(self):
        """Some backends flatten `privileged/x` to `privileged_x`."""
        src = make_env()
        _, info = src.reset(seed=14, options={'variation': ['all']})
        row = self._row_from(src, info)
        goal = {
            k.replace('/', '_'): v for k, v in self._goal_from(src).items()
        }

        dst = make_env()
        opts = dst.reset_options_from_dataset(row, goal)
        assert len(opts['targets']) == dst._num_cubes

    def test_flattened_variation_columns_are_found(self):
        """The lance writer stores `variation.a.b` as `variation_a_b`.

        Regression guard: a prefix-based filter matches neither spelling of
        every axis, and would silently restore nothing at all.
        """
        src = make_env()
        _, info = src.reset(seed=16, options={'variation': ['all']})
        row = self._row_from(src, info)
        flat = {
            (
                'variation_' + k[len('variation.') :].replace('.', '_')
                if k.startswith('variation.')
                else k
            ): v
            for k, v in row.items()
        }
        assert not any(k.startswith('variation.') for k in flat)

        dst = make_env()
        opts = dst.reset_options_from_dataset(flat, self._goal_from(src))
        assert len(opts['variation_values']) == len(
            dst.variation_space.names()
        )

    def test_missing_variation_columns_fall_back_to_defaults(self):
        """A dataset collected without DR must still restore cleanly."""
        src = make_env()
        _, info = src.reset(seed=17)
        row = {'qpos': info['qpos'].copy(), 'qvel': info['qvel'].copy()}

        dst = make_env()
        opts = dst.reset_options_from_dataset(row, self._goal_from(src))
        assert opts['variation_values'] == {}
        dst.reset(seed=17, options=opts)


class TestSubgoalReached:
    def test_true_at_the_current_state(self):
        e = make_env()
        e.reset(seed=15)
        goal = {
            f'goal_privileged/block_{i}_pos': e._data.joint(
                f'object_joint_{i}'
            )
            .qpos[:3]
            .copy()
            for i in range(e._num_cubes)
        }
        assert e.subgoal_reached(goal, tol=1e-6) is True

    def test_false_when_a_cube_is_off(self):
        e = make_env()
        e.reset(seed=15)
        goal = {
            f'goal_privileged/block_{i}_pos': e._data.joint(
                f'object_joint_{i}'
            )
            .qpos[:3]
            .copy()
            for i in range(e._num_cubes)
        }
        goal['goal_privileged/block_1_pos'] = goal[
            'goal_privileged/block_1_pos'
        ] + np.array([1.0, 0.0, 0.0])
        assert e.subgoal_reached(goal, tol=0.04) is False

    def test_false_without_cube_positions(self):
        e = make_env()
        e.reset(seed=15)
        assert e.subgoal_reached({}, tol=0.04) is False


class TestRegistration:
    def test_registered_and_makeable(self):
        import gymnasium as gym

        import stable_worldmodel.envs  # noqa: F401

        env = gym.make(
            'swm/OGBCubeDR-v0',
            env_type='double',
            ob_type='states',
            width=64,
            height=64,
            num_digits=2,
        )
        env.reset(seed=0)
        env.close()

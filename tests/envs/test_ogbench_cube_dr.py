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
    DIRECTIONAL_LIGHT_ROWS,
    LIGHT_NAMES,
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
    'background.floor_rgb',
    'background.wall_rgb',
    'digit.value',
    'digit.position',
    'digit.yaw',
    'digit.count',
    'digit.size',
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

    def test_cube_size_stops_at_the_effector_clip_plane(self, env):
        """Below this the oracle would grasp above the cube's center."""
        assert env.variation_space['cube']['size'].low.min() >= 0.015


class TestSeparation:
    """Cube starts and digit decals must not be sampled on top of each other."""

    @staticmethod
    def _min_gap(positions):
        positions = np.asarray(positions).reshape(-1, 2)
        deltas = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        iu = np.triu_indices(len(positions), k=1)
        return distances[iu].min()

    def test_cubes_never_start_interpenetrating(self):
        e = make_env()
        for seed in range(12):
            e.reset(
                seed=seed,
                options={'variation': ['cube.size', 'cube.start_position']},
            )
            sizes = np.asarray(e.variation_space['cube']['size'].value)
            positions = e.variation_space['cube']['start_position'].value
            # Circumradius, so the test holds at any yaw.
            assert self._min_gap(positions) >= 2 * sizes.min() * np.sqrt(2) - 1e-6

    def test_digits_never_cover_each_other(self):
        e = make_env()
        for seed in range(12):
            e.reset(
                seed=seed,
                options={'variation': ['digit.size', 'digit.position']},
            )
            sizes = np.asarray(e.variation_space['digit']['size'].value)
            positions = e.variation_space['digit']['position'].value
            assert self._min_gap(positions) >= 2 * sizes.min() * np.sqrt(2) - 1e-6

    def test_defaults_satisfy_the_predicates(self):
        """The inherited default puts every cube at the same xy."""
        e = make_env()
        assert e.variation_space.check(debug=True)
        positions = e.variation_space['cube']['start_position'].value
        assert self._min_gap(positions) > 0.05

    def test_predicates_are_optional(self):
        e = make_env(enforce_separation=False)
        e.reset(seed=0, options={'variation': ['all']})
        assert e.variation_space.check(debug=True)


class TestBackgroundColor:
    def test_floor_and_wall_take_independent_colors(self):
        e = make_env()
        e.reset(seed=5, options={'variation': ['all']})
        floor_mat = e._model.geom_matid[e._floor_geom_id]
        wall_mat = e._model.geom_matid[e._backdrop_geom_ids[0]]
        np.testing.assert_allclose(
            e._model.mat_rgba[floor_mat][:3],
            e.variation_space['background']['floor_rgb'].value,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            e._model.mat_rgba[wall_mat][:3],
            e.variation_space['background']['wall_rgb'].value,
            atol=1e-6,
        )

    def test_pools_never_share_a_material(self, env):
        """Index 1 used to be one `plain` material in both pools.

        Sharing it silently forced the floor and the backdrop to the same
        color whenever both sampled that index.
        """
        assert not set(env._bg_floor_matids) & set(env._bg_wall_matids)

    def test_colors_reach_the_render(self):
        e = make_env(width=64, height=64)
        seen = set()
        for seed in range(4):
            e.reset(
                seed=seed,
                options={
                    'variation': [
                        'background.floor_rgb',
                        'background.wall_rgb',
                    ]
                },
            )
            seen.add(e.render().mean().round(2))
        assert len(seen) > 1


class TestDigitCountAndSize:
    def test_surplus_decals_are_parked_out_of_frame(self):
        e = make_env()
        e.reset(seed=4, options={'variation': ['digit.count']})
        count = int(e.variation_space['digit']['count'].value)
        for i in range(e._num_digits):
            z = e._data.mocap_pos[e._digit_mocap_ids[i]][2]
            assert (z > 0) == (i < count)

    def test_visibility_is_exported(self):
        e = make_env()
        _, info = e.reset(seed=4, options={'variation': ['digit.count']})
        count = int(info['privileged/digit_count'][0])
        visible = [
            int(info[f'privileged/digit_{i}_visible'][0])
            for i in range(e._num_digits)
        ]
        assert sum(visible) == count

    def test_sizes_reach_the_model(self):
        e = make_env()
        e.reset(seed=6, options={'variation': ['digit.size']})
        sizes = e.variation_space['digit']['size'].value
        for i in range(e._num_digits):
            geom_size = e._model.geom_size[e._digit_geom_ids[i]]
            assert geom_size[0] == pytest.approx(sizes[i])
            assert geom_size[1] == pytest.approx(sizes[i])


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
            # Light 0 is directional and its position is pinned, so watch a
            # light whose position MuJoCo actually uses.
            positioned = next(
                row
                for row in range(len(LIGHT_NAMES))
                if row not in DIRECTIONAL_LIGHT_ROWS
            )
            seen_light.add(tuple(np.round(e._model.light_pos[positioned], 5)))

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
        assert info['privileged/floor_rgb'].shape == (3,)
        assert info['privileged/wall_rgb'].shape == (3,)
        assert info['privileged/light_dir'].shape == (2 * 3,)

    def test_directional_light_position_is_not_exported(self, env):
        """A directional light's position never reaches a pixel.

        Exporting it would hand a latent probe a target it can only fit noise
        to, so only the positioned lights appear.
        """
        _, info = env.reset(seed=2)
        positioned = len(LIGHT_NAMES) - len(DIRECTIONAL_LIGHT_ROWS)
        assert info['privileged/light_pos'].shape == (positioned * 3,)

    def test_directional_light_position_is_pinned(self, env):
        position = env.variation_space['light']['position']
        for row in DIRECTIONAL_LIGHT_ROWS:
            np.testing.assert_allclose(position.low[row], position.high[row])

    def test_declared_directional_rows_match_the_model(self, env):
        env.reset(seed=2)
        env._verify_directional_lights()  # raises if the constant went stale

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
        identity = np.arange(e._num_cubes)
        rescaled = e._rescaled_task_info(e.task_infos[4], identity)
        halves = e.variation_space['cube']['size'].value

        # Each cube rests on the one below it in its column, so its height is
        # the running sum of the *supporting* cubes' full extents, not a
        # multiple of its own.
        bottom = 0.0
        for i, half in enumerate(halves):
            assert rescaled['goal_xyzs'][i][2] == pytest.approx(bottom + half)
            bottom += 2 * half

    def test_stacked_waypoints_use_the_supporting_cube(self):
        """A layer-1 cube sits on layer 0's height, not on its own."""
        e = make_env(mode='task', permute_blocks=False)
        e.reset(seed=1, options={'variation': []})
        e.variation_space['cube']['size'].set_value(
            np.array([0.03, 0.015, 0.02, 0.02][: e._num_cubes])
        )
        identity = np.arange(e._num_cubes)
        rescaled = e._rescaled_task_info(e.task_infos[4], identity)
        # Cube 0 (half 0.03) is the base; cube 1 (half 0.015) rests on top of
        # it, so it sits at 2 * 0.03 + 0.015, not at 3 * 0.015.
        assert rescaled['goal_xyzs'][0][2] == pytest.approx(0.03)
        assert rescaled['goal_xyzs'][1][2] == pytest.approx(0.075)

    def test_rows_are_permuted_before_the_heights_are_built(self):
        """Row i must belong to cube i, or the half-extents get swapped."""
        e = make_env(mode='task', permute_blocks=False)
        e.reset(seed=1, options={'variation': []})
        e.variation_space['cube']['size'].set_value(
            np.array([0.03, 0.015, 0.02, 0.02][: e._num_cubes])
        )
        permutation = np.array([1, 0] + list(range(2, e._num_cubes)))
        rescaled = e._rescaled_task_info(e.task_infos[4], permutation)
        # The tower is the same shape with the two cubes exchanged: cube 1
        # (half 0.015) is now the base and cube 0 (half 0.03) sits on it.
        assert rescaled['goal_xyzs'][1][2] == pytest.approx(0.015)
        assert rescaled['goal_xyzs'][0][2] == pytest.approx(0.06)

    def test_effector_clip_plane_follows_the_smallest_cube(self):
        e = make_env()
        e.reset(seed=3, options={'variation': ['cube.size']})
        smallest = float(np.min(e.variation_space['cube']['size'].value))
        assert e._workspace_bounds[0][2] == pytest.approx(
            min(0.02, smallest)
        )

    def test_clip_plane_does_not_ratchet_down(self):
        e = make_env()
        e.reset(seed=3, options={'variation': ['cube.size']})
        e.variation_space['cube']['size'].set_value(
            np.full(e._num_cubes, 0.03)
        )
        e._apply_size_aware_workspace()
        assert e._workspace_bounds[0][2] == pytest.approx(0.02)


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

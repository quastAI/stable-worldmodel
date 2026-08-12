"""Tests for the oracle-subgoal chain in dataset-driven evaluation.

Reuses the ``CounterEnv`` / ``FakeEvalDataset`` doubles from
``test_new_world`` so these tests need neither MuJoCo nor a real dataset.
"""

import numpy as np
import pytest
from test_new_world import (
    CounterEnv,
    DatasetResetEnv,
    FakeEvalDataset,
    RecordingPolicy,
)

from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.world.world import World


class ReachEnv(DatasetResetEnv):
    """Dataset-restorable env that reports a subgoal reached after N steps."""

    def __init__(self, max_steps: int = 50, reach_after: int = 2):
        super().__init__(max_steps)
        self._reach_after = reach_after
        self.reach_calls = 0

    def subgoal_reached(self, goal_row, tol=0.04):
        self.reach_calls += 1
        return (
            self._step_count > 0 and self._step_count % self._reach_after == 0
        )


def _make_world(env_fns):
    pool = EnvPool(env_fns)
    world = object.__new__(World)
    world.envs = pool
    world.policy = None
    world.infos = {}
    world.rewards = None
    world.terminateds = None
    world.truncateds = None
    policy = RecordingPolicy()
    world.policy = policy
    policy.set_env(world.envs)
    return world


def _evaluate(world, **kwargs):
    defaults = dict(
        dataset=FakeEvalDataset(),
        episodes_idx=[0, 1],
        start_steps=[0, 0],
        goal_offset=1,
        eval_budget=6,
    )
    defaults.update(kwargs)
    return world.evaluate(**defaults)


class TestSingleGoalUnchanged:
    """num_subgoals=1 must behave exactly as the pre-chain implementation."""

    def test_goal_is_pinned_and_no_flush(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        results = _evaluate(world, goal_offset=3)

        # goal at start + goal_offset, held for the whole rollout
        np.testing.assert_array_equal(
            world.infos['goal_proprio'][0, 0], [6.0, 7.0]
        )
        assert '_needs_flush' not in world.infos
        assert not world.policy._flush_log
        np.testing.assert_array_equal(results['subgoal_index'], [0, 0])

    def test_reset_row_is_the_final_goal(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        _evaluate(world, goal_offset=3)

        for i in range(2):
            _, goal_row = world.envs.envs[i].received[0]
            assert set(goal_row) == {
                'goal',
                'goal_proprio',
                'goal_states',
                'goal_seed',
            }


class TestBudgetAdvance:
    def test_goal_walks_the_chain(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        results = _evaluate(
            world,
            goal_offset=1,
            num_subgoals=3,
            subgoal_budget=1,
            eval_budget=5,
        )

        # budget of 1 step per subgoal: after 5 steps we sit on the last one
        np.testing.assert_array_equal(results['subgoal_index'], [2, 2])
        np.testing.assert_array_equal(
            world.infos['goal_proprio'][0, 0], [6.0, 7.0]
        )
        # two advances per env => two flushes each
        assert sorted(world.policy._flush_log) == [0, 0, 1, 1]

    def test_env_target_is_the_last_subgoal(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        _evaluate(world, goal_offset=1, num_subgoals=3, subgoal_budget=1)

        # reset_options_from_dataset must see the *final* waypoint, so that
        # env-side success means the whole segment was completed.
        for i in range(2):
            _, goal_row = world.envs.envs[i].received[0]
            np.testing.assert_array_equal(
                goal_row['goal_proprio'], [6.0 + 10.0 * i, 7.0 + 10.0 * i]
            )

    def test_index_saturates_at_last_subgoal(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        results = _evaluate(
            world,
            goal_offset=1,
            num_subgoals=2,
            subgoal_budget=1,
            eval_budget=6,
        )
        np.testing.assert_array_equal(results['subgoal_index'], [1, 1])


class TestReachedAdvance:
    def test_advances_on_reach(self):
        world = _make_world(
            [lambda: ReachEnv(50, reach_after=2) for _ in range(2)]
        )
        results = _evaluate(
            world,
            goal_offset=1,
            num_subgoals=3,
            subgoal_advance='reached',
            subgoal_budget=100,  # far larger than eval_budget: reach only
            eval_budget=6,
        )

        # reached on steps 2, 4, 6 -> two advances before saturating
        np.testing.assert_array_equal(results['subgoal_index'], [2, 2])
        np.testing.assert_array_equal(results['subgoals_reached'], [3, 3])
        assert world.envs.envs[0].reach_calls > 0

    def test_falls_back_to_budget_without_env_support(self):
        """An env with no subgoal_reached must not silently stall."""
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        results = _evaluate(
            world,
            goal_offset=1,
            num_subgoals=3,
            subgoal_advance='reached',
            subgoal_budget=1,
            eval_budget=5,
        )
        np.testing.assert_array_equal(results['subgoal_index'], [2, 2])
        np.testing.assert_array_equal(results['subgoals_reached'], [0, 0])

    def test_both_takes_whichever_comes_first(self):
        world = _make_world(
            [lambda: ReachEnv(50, reach_after=5) for _ in range(2)]
        )
        results = _evaluate(
            world,
            goal_offset=1,
            num_subgoals=3,
            subgoal_advance='both',
            subgoal_budget=2,
            eval_budget=5,
        )
        # budget (2 steps) fires before the reach signal (5 steps)
        np.testing.assert_array_equal(results['subgoal_index'], [2, 2])


class TestValidation:
    def test_rejects_zero_subgoals(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        with pytest.raises(ValueError, match='num_subgoals'):
            _evaluate(world, num_subgoals=0)

    def test_rejects_unknown_advance_mode(self):
        world = _make_world([lambda: DatasetResetEnv(50) for _ in range(2)])
        with pytest.raises(ValueError, match='subgoal_advance'):
            _evaluate(world, subgoal_advance='whenever')


class TestVideoPanels:
    def test_goal_panel_tracks_the_active_subgoal(self, tmp_path):
        world = _make_world([lambda: CounterEnv(50) for _ in range(2)])
        _evaluate(
            world,
            goal_offset=1,
            num_subgoals=3,
            subgoal_budget=1,
            eval_budget=4,
            video=tmp_path,
        )
        # one mp4 per env, written without raising on the per-step goal panel
        assert len(list(tmp_path.glob('*.mp4'))) == 2

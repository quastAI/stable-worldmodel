"""Tests for the LeJEPA identifiability cube environment.

The gate these exist to guard is :meth:`LeJEPACubeEnv.render_content` -- the
physics-free frame path that the encoder dataset is built on. It skips the
recompile, the IK solve and one of the two renders that ``reset`` performs, so
the burden of proof is on it to show it produces the *same frame* ``reset``
would have. ``test_render_content_matches_reset`` is that proof; everything
else supports it.

Skipped wholesale when ``ogbench`` is not installed -- it lives in the ``env``
extra, not the base dependency set.
"""

import os

import numpy as np
import pytest


os.environ.setdefault('MUJOCO_GL', 'glfw')

pytest.importorskip('ogbench')
pytest.importorskip('mujoco')

import mujoco  # noqa: E402
from ogbench.manipspace import lie  # noqa: E402

from stable_worldmodel.envs.ogbench.lejepa_cube_env import (  # noqa: E402
    GRIPPER_QPOS_SCALE,
    MARKER_FACES,
    LeJEPACubeEnv,
)


# Appearance axes that must be identical between the `reset` reference and the
# `render_content` candidate for the equivalence test to mean anything.
APPEARANCE = {
    'cube.color': np.array([[0.85, 0.25, 0.15]]),
    'cube.size': np.array([0.026]),
    'agent.color': np.array([0.25, 0.45, 0.75]),
    'camera.angle_delta': np.array([[4.0, -3.0]]),
    'light.diffuse': np.array([[0.62, 0.58, 0.71], [0.33, 0.41, 0.29]]),
    'light.ambient': np.array([[0.11, 0.09, 0.13], [0.05, 0.07, 0.04]]),
    'light.specular': np.array([[0.21, 0.25, 0.19], [0.4, 0.31, 0.36]]),
    'light.headlight_diffuse': np.array([0.55, 0.62, 0.48]),
    'background.floor_material': 3,
    'background.wall_material': 5,
    'background.floor_rgb': np.array([0.42, 0.61, 0.38]),
    'background.wall_rgb': np.array([0.19, 0.23, 0.44]),
    'digit.value': np.array([7]),
    'digit.size': np.array([0.048]),
    'digit.position': np.array([[0.44, -0.11]]),
    'digit.yaw': np.array([1.1]),
    'marker.value': np.array([4]),
}


def make_env(**kwargs):
    defaults = dict(
        env_type='single',
        ob_type='pixels',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=128,
        height=128,
        num_digits=1,
    )
    defaults.update(kwargs)
    env = LeJEPACubeEnv(**defaults)
    env.reset(seed=0, options={'variation': ['all']})
    return env


@pytest.fixture(scope='module')
def env():
    e = make_env()
    yield e
    e.close()


def content_state(env, **overrides):
    """A representative physical latent assignment."""
    physical = {
        'cube.pos_xy': np.array([[0.41, -0.06]]),
        'cube.pos_z': np.array([0.026]),
        'cube.yaw': np.array([0.9]),
        'effector.pos': np.array([0.42, 0.11, 0.28]),
        'effector.yaw': 0.6,
        'gripper.opening': 0.35,
    }
    physical.update(overrides)
    return env.set_content_state(physical), physical


# ---------------------------------------------------------------- registration


def test_env_is_registered():
    from gymnasium.envs import registration

    assert 'swm/LeJEPACube-v0' in registration.registry


# ------------------------------------------------------- the equivalence gate


def test_render_content_matches_reset(env):
    """The fast path must render exactly what ``reset`` renders.

    Both are handed the same content state and the same appearance values.
    ``reset`` recompiles the model for ``cube.size``/``agent.color``/
    ``camera.angle_delta``; ``render_content`` writes the compiled fields
    instead. If those two disagree, every frame in the encoder dataset is a
    frame the evaluation path cannot reproduce.

    The goal markers are the one deliberate difference -- ``reset`` runs
    ``set_new_target``, ``render_content`` parks them out of frame -- so they
    are parked on the reference too, rather than being excused in the
    comparison.
    """
    state, _ = content_state(env)

    reference = env.reset(
        options={
            'variation': [],
            'variation_values': APPEARANCE,
            'state': state,
        }
    )
    del reference
    env._hide_goal_markers()
    mujoco.mj_forward(env._model, env._data)
    expected = env.render(camera='front_pixels').copy()

    got = env.render_content(state, APPEARANCE, camera='front_pixels')

    assert got.shape == expected.shape
    diff = np.abs(got.astype(np.int32) - expected.astype(np.int32))
    assert diff.max() == 0, (
        f'render_content diverged from reset: max |delta| = {diff.max()}, '
        f'{int((diff > 0).sum())} differing subpixels'
    )


def test_render_content_is_deterministic(env):
    """Same content in, same pixels out -- no hidden RNG in the fast path."""
    state, _ = content_state(env)
    a = env.render_content(state, APPEARANCE).copy()
    b = env.render_content(state, APPEARANCE).copy()
    assert np.array_equal(a, b)


def test_render_content_does_not_step_physics(env):
    """``qpos`` after a frame is the ``qpos`` that was asked for.

    A cube at ``pos_z`` above the table must stay there. If anything in the
    path integrated the dynamics, it would fall, and the recorded label would
    describe a state that was never rendered.
    """
    state, _ = content_state(env, **{'cube.pos_z': np.array([0.22])})
    env.render_content(state, APPEARANCE)

    nq = env._model.nq
    np.testing.assert_allclose(env._data.qpos, state[:nq], atol=0, rtol=0)
    np.testing.assert_allclose(env._data.qvel, 0.0, atol=0, rtol=0)


# ------------------------------------------------------------ latent round-trip


def test_content_state_round_trip(env):
    """``z -> qpos -> privileged/* + proprio/* -> z`` to float32 tolerance.

    This is the registry's contract: a latent that cannot be read back out of
    a recorded row is a latent the recovery metrics cannot score.
    """
    state, physical = content_state(env)
    env.render_content(state, APPEARANCE)
    info = env.content_info()

    np.testing.assert_allclose(
        info['privileged/block_0_pos'][:2],
        physical['cube.pos_xy'][0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        info['privileged/block_0_pos'][2], physical['cube.pos_z'][0], atol=1e-6
    )
    np.testing.assert_allclose(
        info['privileged/block_0_yaw'][0], physical['cube.yaw'][0], atol=1e-6
    )
    # The effector is placed by IK, so it round-trips to the solver's
    # tolerance rather than to machine precision.
    np.testing.assert_allclose(
        info['proprio/effector_pos'], physical['effector.pos'], atol=1e-3
    )
    np.testing.assert_allclose(
        info['privileged/ee_yaw'][0], physical['effector.yaw'], atol=1e-3
    )
    np.testing.assert_allclose(
        info['proprio/gripper_opening'][0],
        physical['gripper.opening'],
        atol=1e-6,
    )


def test_gripper_opening_is_settable_by_qpos(env):
    """The plan's SS9 gripper risk, as a gate.

    The Robotiq 2F-85 is a coupled linkage. If setting the opening does not
    propagate through it, ``gripper.opening`` is not a settable latent and must
    be demoted in the registry rather than silently recorded wrong.
    """
    lo, hi = env.gripper_opening_bounds
    seen = []
    for opening in (lo, 0.5 * (lo + hi), hi):
        state, _ = content_state(env, **{'gripper.opening': opening})
        env.render_content(state, APPEARANCE)
        info = env.content_info()
        seen.append(float(info['proprio/gripper_opening'][0]))
        assert info['proprio/gripper_opening'][0] == pytest.approx(
            opening, abs=1e-6
        )
        assert env._data.qpos[
            env._gripper_opening_joint_id
        ] == pytest.approx(GRIPPER_QPOS_SCALE * opening, abs=1e-9)

    assert len(set(seen)) == 3, 'gripper opening did not actually vary'


def test_gripper_opening_moves_the_pads(env):
    """Round-tripping the label is not enough -- the geometry must move too.

    ``proprio/gripper_opening`` is read straight back off the driver joint, so
    it would round-trip perfectly even if the coupled pads never responded --
    which is exactly what happens if only the driver is written. Comparing the
    pad bodies is what actually tests the linkage.
    """
    lo, hi = env.gripper_opening_bounds
    positions = []
    for opening in (lo, hi):
        state, _ = content_state(env, **{'gripper.opening': opening})
        env.render_content(state, APPEARANCE)
        positions.append(
            env._data.body('ur5e/robotiq/right_pad').xpos.copy()
        )

    assert np.linalg.norm(positions[0] - positions[1]) > 1e-3, (
        'gripper pads did not move between fully closed and fully open; '
        'demote gripper.opening in the registry'
    )


def test_gripper_opening_lands_on_the_constraint_manifold(env):
    """The interpolated configuration must satisfy the equality constraints.

    A configuration off the manifold renders a visibly broken linkage -- pads
    detached from the fingers -- while every scalar label still round-trips.
    ``mj_forward`` populates ``efc_pos`` for each active equality constraint;
    on the manifold those residuals are zero.
    """
    lo, hi = env.gripper_opening_bounds
    for opening in np.linspace(lo, hi, 7):
        state, _ = content_state(env, **{'gripper.opening': float(opening)})
        env.render_content(state, APPEARANCE)

        eq_rows = env._data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        residual = np.abs(env._data.efc_pos[eq_rows])
        assert residual.max() < 1e-3, (
            f'opening {opening:.3f} sits off the constraint manifold: '
            f'max equality residual {residual.max():.2e}'
        )


def test_gripper_pads_move_monotonically(env):
    """Opening is monotone in pad separation, so the latent is invertible.

    A non-monotone map would make two distinct openings render identically,
    which is an injectivity failure the recovery metrics would report as an
    encoder problem.
    """
    lo, hi = env.gripper_opening_bounds
    separations = []
    for opening in np.linspace(lo, hi, 9):
        state, _ = content_state(env, **{'gripper.opening': float(opening)})
        env.render_content(state, APPEARANCE)
        left = env._data.body('ur5e/robotiq/left_pad').xpos
        right = env._data.body('ur5e/robotiq/right_pad').xpos
        separations.append(float(np.linalg.norm(left - right)))

    deltas = np.diff(separations)
    assert np.all(deltas > 0) or np.all(deltas < 0), (
        f'pad separation is not monotone in gripper.opening: {separations}'
    )


def test_gripper_bounds_are_narrower_than_unit(env):
    """The declared truncation is real and recorded, not an oversight.

    The actuator cannot reach either joint limit, so the axis is declared over
    the achievable range. If this ever became ``[0, 1]`` exactly, the endpoints
    would sit off the constraint manifold.
    """
    lo, hi = env.gripper_opening_bounds
    assert 0.0 < lo < hi < 1.0
    space = env.variation_space['agent']['gripper_opening']
    assert float(space.low[0]) == pytest.approx(lo)
    assert float(space.high[0]) == pytest.approx(hi)


# ----------------------------------------------------------------- yaw marker


def _yaw_rms(e, appearance, a, b):
    """RMS pixel difference between two yaws, everything else held fixed."""
    frames = []
    for yaw in (a, b):
        state, _ = content_state(e, **{'cube.yaw': np.array([yaw])})
        frames.append(
            e.render_content(state, appearance).astype(np.float64).copy()
        )
    return float(np.sqrt(((frames[0] - frames[1]) ** 2).mean())), frames


def test_cube_is_exactly_four_fold_symmetric_without_a_marker():
    """The premise the marker exists to fix, asserted exactly.

    A bare cube is invariant under yaw -> yaw + pi/2, so those two states carry
    no distinguishable information. Yaw is then unidentifiable in principle,
    not approximately -- no encoder can recover what the renderer discarded.

    The assertion is at the rasterizer's own precision rather than at exactly
    zero: a handful of subpixels land one LSB apart because the two
    quaternions are different float32 values that rasterize to the same
    geometry. That is quantization, not signal. What matters is the contrast
    with a marked cube, which differs by ~48 whole pixels at magnitudes well
    above this floor.
    """
    e = make_env(marker_enabled=False, width=224, height=224)
    appearance = {k: v for k, v in APPEARANCE.items() if k != 'marker.value'}
    try:
        _, frames = _yaw_rms(e, appearance, 0.0, np.pi / 2)
        delta = np.abs(frames[0] - frames[1])
        assert delta.max() <= 1, (
            f'unmarked cube differs by up to {delta.max():.0f} between yaws '
            'pi/2 apart; it is not 4-fold symmetric and the marker rationale '
            'needs revisiting'
        )
        assert int((delta > 1).sum()) == 0
    finally:
        e.close()


def test_marker_makes_yaw_observable():
    """The marker must break the 4-fold symmetry by a decisive margin.

    Stated as a ratio against the unmarked baseline rather than as an absolute
    rms, because the absolute number scales with render resolution and cube
    size while the claim -- "the marker is what makes yaw identifiable" --
    does not. Run at 224x224, the resolution the dataset is actually collected
    at; at 128x128 the marker covers ~12 pixels and this signal is genuinely
    marginal.
    """
    marked = make_env(width=224, height=224)
    unmarked = make_env(marker_enabled=False, width=224, height=224)
    bare = {k: v for k, v in APPEARANCE.items() if k != 'marker.value'}
    try:
        with_marker, _ = _yaw_rms(marked, APPEARANCE, 0.0, np.pi / 2)
        without, _ = _yaw_rms(unmarked, bare, 0.0, np.pi / 2)

        assert with_marker > 1.0, (
            f'marker yaw signal is only rms {with_marker:.3f}'
        )
        assert with_marker > 50 * max(without, 1e-6), (
            f'marker adds little over the unmarked baseline '
            f'({with_marker:.4f} vs {without:.4f})'
        )
    finally:
        marked.close()
        unmarked.close()


def test_yaw_is_observable_mod_quarter_turn_without_a_marker():
    """Records what the marker is *not* responsible for.

    Yaws pi/4 apart change the cube's silhouette outright, so they are
    strongly distinguishable with no marker at all -- more strongly, in fact,
    than the marker distinguishes yaws pi/2 apart. Yaw recovery therefore has
    two regimes: the within-quadrant angle, which is easy, and the quadrant
    itself, which rests entirely on a ~48-pixel marker.

    This is a property of the rendering, not of any encoder. A yaw metric that
    reports a single number will average the two and misattribute the result.
    """
    e = make_env(marker_enabled=False, width=224, height=224)
    bare = {k: v for k, v in APPEARANCE.items() if k != 'marker.value'}
    try:
        quarter, _ = _yaw_rms(e, bare, 0.0, np.pi / 2)
        eighth, _ = _yaw_rms(e, bare, 0.0, np.pi / 4)
        assert eighth > 100 * max(quarter, 1e-6), (
            f'expected the silhouette to dominate at pi/4 (got {eighth:.4f}) '
            f'while pi/2 is symmetric (got {quarter:.4f})'
        )
    finally:
        e.close()


def test_side_marker_is_stronger_but_occludable():
    """Both horns of the SS9 marker-face tradeoff, measured rather than assumed.

    A front-face marker presents more pixels to the camera than a foreshortened
    top-face one -- but only while it faces the camera. The plan defaults to
    ``top`` for exactly this reason; this test is what makes that a measured
    choice.
    """
    top = make_env(marker_face='top', width=224, height=224)
    front = make_env(marker_face='front', width=224, height=224)
    try:
        top_rms, _ = _yaw_rms(top, APPEARANCE, 0.0, np.pi / 2)
        front_rms, _ = _yaw_rms(front, APPEARANCE, 0.0, np.pi / 2)
        assert front_rms > top_rms, (
            f'front-face marker ({front_rms:.3f}) was expected to present a '
            f'stronger signal than top-face ({top_rms:.3f})'
        )

        # ...and the cost: turned away from the camera it contributes nothing.
        away, frames = _yaw_rms(front, APPEARANCE, np.pi, np.pi + np.pi / 2)
        assert away < front_rms, (
            'a front-face marker facing away should carry less yaw signal '
            f'than one facing the camera ({away:.3f} vs {front_rms:.3f})'
        )
    finally:
        top.close()
        front.close()


@pytest.mark.parametrize('face', sorted(MARKER_FACES))
def test_marker_face_is_a_knob(face):
    """Every declared face builds and renders."""
    e = make_env(marker_face=face)
    try:
        state, _ = content_state(e)
        frame = e.render_content(state, APPEARANCE)
        assert frame.shape == (128, 128, 3)
    finally:
        e.close()


def test_marker_does_not_collide(env):
    """The decal must be collision-free, or it changes the physics it decorates.

    The predictor dataset is collected in the same environment, so a marker
    with contacts would alter the very rollouts the planner is evaluated on.
    """
    for gid in env._marker_geom_ids:
        assert env._model.geom_contype[gid] == 0
        assert env._model.geom_conaffinity[gid] == 0


def test_marker_value_is_recorded(env):
    state, _ = content_state(env)
    env.render_content(state, {**APPEARANCE, 'marker.value': np.array([6])})
    info = env.content_info()
    assert int(info['privileged/marker_0_value'][0]) == 6


def test_marker_scales_with_cube_size(env):
    """A marker sized for the stock cube would clip through a smaller one."""
    for size in (0.016, 0.03):
        state, _ = content_state(env)
        env.render_content(state, {**APPEARANCE, 'cube.size': np.array([size])})
        gid = env._marker_geom_ids[0]
        axis, _ = MARKER_FACES[env._marker_face]
        tile = [d for i, d in enumerate(env._model.geom_size[gid]) if i != axis]
        assert all(t < size for t in tile), (
            f'marker tile {tile} is not inside a cube of half-extent {size}'
        )


# ------------------------------------------------------------ new arm latents


def test_ee_start_yaw_is_a_variation_axis(env):
    """The parent draws effector yaw from its own RNG; we must not.

    Two ``reset`` calls with the same requested yaw and different seeds have
    to agree, or ``effector.yaw`` is not a controlled latent no matter what
    the sampler asks for.
    """
    yaws = []
    for seed in (11, 12):
        env.reset(
            seed=seed,
            options={
                'variation': [],
                'variation_values': {
                    **APPEARANCE,
                    'agent.ee_start_yaw': np.array([0.75]),
                },
            },
        )
        yaws.append(float(env.compute_ob_info()['proprio/effector_yaw'][0]))

    assert yaws[0] == pytest.approx(yaws[1], abs=1e-6), (
        'effector yaw still depends on the seed, not on ee_start_yaw'
    )
    assert yaws[0] == pytest.approx(0.75, abs=1e-3)


def test_floor_color_is_pinned(env):
    """``floor.color`` is the one axis that would force a per-frame recompile.

    Pinned by construction (low == high), so it cannot be resampled even by
    ``variation: ['all']``.
    """
    space = env.variation_space['floor']['color']
    np.testing.assert_array_equal(space.low, space.high)


def test_floor_colour_control_survives_via_background_rgb(env):
    """Pinning ``floor.color`` must not cost the floor-color latent itself."""
    frames = []
    for rgb in (np.array([0.9, 0.2, 0.2]), np.array([0.2, 0.3, 0.9])):
        state, _ = content_state(env)
        frames.append(
            env.render_content(
                state, {**APPEARANCE, 'background.floor_rgb': rgb}
            ).astype(np.float64).copy()
        )
    rms = float(np.sqrt(((frames[0] - frames[1]) ** 2).mean()))
    assert rms > 1.0, 'background.floor_rgb does not change the floor'


# ---------------------------------------------------------- no goal-marker leak


def test_goal_markers_never_render(env):
    """A visible goal marker is a latent the registry does not account for."""
    state, _ = content_state(env)
    env.render_content(state, APPEARANCE)
    for gid in env._cube_target_geom_ids_list[0]:
        assert env._model.geom_rgba[gid, 3] == 0.0


def test_goal_marker_pose_does_not_reach_the_frame(env):
    """Moving the goal mocap must not change a single pixel."""
    state, _ = content_state(env)
    a = env.render_content(state, APPEARANCE).copy()

    env._data.mocap_pos[env._cube_target_mocap_ids[0]] = (0.45, 0.0, 0.05)
    env._data.mocap_quat[env._cube_target_mocap_ids[0]] = (
        lie.SO3.from_z_radians(1.0).wxyz.tolist()
    )
    b = env.render_content(state, APPEARANCE).copy()

    assert np.array_equal(a, b)


# -------------------------------------------------------------------- physics


def test_reset_path_still_steps_physics(env):
    """The fast path is an addition, not a replacement.

    Evaluation, replay and the predictor dataset all go through ``reset`` and
    ``step``; those must keep working with a stepped, recompiled model.
    """
    env.reset(seed=3, options={'variation': ['all']})
    before = env._data.qpos.copy()
    env.step(env.action_space.sample())
    assert not np.array_equal(before, env._data.qpos), (
        'step() did not advance the simulator'
    )

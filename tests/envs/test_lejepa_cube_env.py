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
#
# Note what is *absent*: no `digit.*` and no `marker.value`. The environment
# defaults to `num_digits=0` and `marker_enabled=False`, so those sub-spaces do
# not exist and naming them would raise. Tests that want them opt in explicitly.
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
}

# Extra axes that exist only when the yaw marker and floor digits are enabled.
MARKER_APPEARANCE = {**APPEARANCE, 'marker.value': np.array([4])}
DIGIT_APPEARANCE = {
    **APPEARANCE,
    'digit.value': np.array([7]),
    'digit.size': np.array([0.048]),
    'digit.position': np.array([[0.44, -0.11]]),
    'digit.yaw': np.array([1.1]),
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


# ------------------------------------------- yaw: the quotient, not the circle


def _yaw_rms(e, appearance, a, b):
    """RMS pixel difference between two yaws, everything else held fixed."""
    frames = []
    for yaw in (a, b):
        state, _ = content_state(e, **{'cube.yaw': np.array([yaw])})
        frames.append(
            e.render_content(state, appearance).astype(np.float64).copy()
        )
    return float(np.sqrt(((frames[0] - frames[1]) ** 2).mean())), frames


def test_cube_is_exactly_four_fold_symmetric():
    """The fact the whole yaw convention rests on.

    A rotation of pi/2 about z maps the cube onto itself, so those two states
    are not "hard to tell apart" -- they are one state under two names, for the
    renderer and for the grasp alike. That is why the fix is to sample a
    fundamental domain rather than to bolt on a marker that manufactures a
    distinction the task does not have.

    Asserted at the rasterizer's precision: a handful of subpixels land one LSB
    apart because the two quaternions are different float32 values that
    rasterize to the same geometry. That is quantization, not signal.
    """
    e = make_env(width=224, height=224)
    try:
        _, frames = _yaw_rms(e, APPEARANCE, 0.0, np.pi / 2)
        delta = np.abs(frames[0] - frames[1])
        assert delta.max() <= 1, (
            f'cube differs by up to {delta.max():.0f} between yaws pi/2 apart; '
            'it is not 4-fold symmetric and the quotient argument is wrong'
        )
        assert int((delta > 1).sum()) == 0
    finally:
        e.close()


def test_yaw_is_identifiable_on_the_fundamental_domain():
    """Within one fundamental domain, yaw is strongly observable unaided.

    This is what makes the marker unnecessary. Compared at half the domain's
    width apart (~43 deg) rather than at its extremes -- see
    :func:`test_yaw_separability_is_non_monotone_in_angle` for why the extremes
    are the *worst* possible probe of identifiability, not the best.
    """
    from stable_worldmodel.identifiability.latents import DEFAULT_YAW_HALF_ARC

    e = make_env(width=224, height=224)
    try:
        for a, b in (
            (0.0, DEFAULT_YAW_HALF_ARC),
            (-DEFAULT_YAW_HALF_ARC, 0.0),
            (-DEFAULT_YAW_HALF_ARC / 2, DEFAULT_YAW_HALF_ARC / 2),
        ):
            rms, _ = _yaw_rms(e, APPEARANCE, a, b)
            assert rms > 3.0, (
                f'yaws {np.degrees(a):.1f} and {np.degrees(b):.1f} deg differ '
                f'by only rms {rms:.3f}; yaw would not be recoverable even on '
                'the quotient'
            )
    finally:
        e.close()


def test_yaw_separability_is_non_monotone_in_angle():
    """Pixel separability peaks near 45 deg and *falls* toward 90 deg.

    A direct consequence of the C4 symmetry, and a trap for anyone reading a
    yaw-recovery curve: two cubes 85 deg apart look far more alike (rms ~1.2)
    than two 43 deg apart (rms ~4.4), because 85 deg is nearly the 90 deg
    identification. Measured at 224x224:

        separation   rms
          21 deg     3.79
          43 deg     4.36
          86 deg     1.17

    So "larger angular error" does not mean "more visibly wrong", and a metric
    that assumes monotonicity in raw angle will misread the result. Recovery
    should be scored on the quotient coordinate, which is what the registry
    hands it.
    """
    from stable_worldmodel.identifiability.latents import DEFAULT_YAW_HALF_ARC

    e = make_env(width=224, height=224)
    try:
        near, _ = _yaw_rms(e, APPEARANCE, 0.0, DEFAULT_YAW_HALF_ARC)
        far, _ = _yaw_rms(
            e, APPEARANCE, -DEFAULT_YAW_HALF_ARC, DEFAULT_YAW_HALF_ARC
        )
        assert far < near, (
            f'expected the wider separation to look MORE alike under C4 '
            f'({far:.3f} vs {near:.3f})'
        )
    finally:
        e.close()


def test_fundamental_domain_contains_no_duplicate_configuration():
    """No two sampled yaws may denote the same physical state.

    The endpoints -pi/4 and +pi/4 differ by exactly pi/2 and are therefore the
    same configuration. The default arc carries a margin so that even clipped
    draws cannot land on both, which is what keeps the map from z to pixels
    injective. Without the margin this test fails.
    """
    from stable_worldmodel.identifiability.latents import (
        DEFAULT_YAW_HALF_ARC,
        YAW_FUNDAMENTAL_HALF_ARC,
    )

    assert DEFAULT_YAW_HALF_ARC < YAW_FUNDAMENTAL_HALF_ARC

    e = make_env(width=224, height=224)
    try:
        rms, _ = _yaw_rms(
            e, APPEARANCE, -DEFAULT_YAW_HALF_ARC, DEFAULT_YAW_HALF_ARC
        )
        # If the arc reached the full domain these two would be identical.
        assert rms > 1.0, (
            'the two ends of the sampled arc render near-identically, so the '
            'arc has reached the symmetry boundary and yaw is not injective'
        )
    finally:
        e.close()


def test_marker_is_off_by_default():
    """A decal scores the encoder on a quadrant index no grasp depends on."""
    e = make_env()
    try:
        assert e._marker_enabled is False
        assert 'marker' not in e.variation_space.spaces
        assert e._marker_geom_ids == []
    finally:
        e.close()


def test_marker_still_works_when_explicitly_enabled():
    """The machinery is kept and tested -- it is the way to study the full circle."""
    e = make_env(marker_enabled=True, width=224, height=224)
    try:
        assert 'marker' in e.variation_space.spaces
        marked, _ = _yaw_rms(e, MARKER_APPEARANCE, 0.0, np.pi / 2)
        assert marked > 1.0, (
            f'marker enabled but yaws pi/2 apart still differ by only '
            f'rms {marked:.3f}'
        )
    finally:
        e.close()


@pytest.mark.parametrize('face', sorted(MARKER_FACES))
def test_marker_face_is_a_knob(face):
    """Every declared face builds and renders, when the marker is enabled."""
    e = make_env(marker_enabled=True, marker_face=face)
    try:
        state, _ = content_state(e)
        assert e.render_content(state, MARKER_APPEARANCE).shape == (128, 128, 3)
    finally:
        e.close()


def test_marker_does_not_collide():
    """The decal must be collision-free, or it changes the physics it decorates."""
    e = make_env(marker_enabled=True)
    try:
        for gid in e._marker_geom_ids:
            assert e._model.geom_contype[gid] == 0
            assert e._model.geom_conaffinity[gid] == 0
    finally:
        e.close()


def test_marker_value_is_recorded():
    e = make_env(marker_enabled=True)
    try:
        state, _ = content_state(e)
        e.render_content(state, {**MARKER_APPEARANCE, 'marker.value': np.array([6])})
        assert int(e.content_info()['privileged/marker_0_value'][0]) == 6
    finally:
        e.close()


def test_marker_scales_with_cube_size():
    """A marker sized for the stock cube would clip through a smaller one."""
    e = make_env(marker_enabled=True)
    try:
        for size in (0.016, 0.03):
            state, _ = content_state(e)
            e.render_content(
                state, {**MARKER_APPEARANCE, 'cube.size': np.array([size])}
            )
            gid = e._marker_geom_ids[0]
            axis, _ = MARKER_FACES[e._marker_face]
            tile = [
                d for i, d in enumerate(e._model.geom_size[gid]) if i != axis
            ]
            assert all(t < size for t in tile), (
                f'marker tile {tile} is not inside a cube of half-extent {size}'
            )
    finally:
        e.close()


# ------------------------------------------- floor decals and arm visibility


def test_no_floor_digits_by_default():
    """The distractors exist for a DR probe; nothing here reads them.

    Against a scene whose content latents already occupy few pixels they are
    pure occlusion risk -- a V6 contribution nobody asked for.
    """
    e = make_env()
    try:
        assert e._num_digits == 0
        assert 'digit' not in e.variation_space.spaces
        state, _ = content_state(e)
        e.render_content(state, APPEARANCE)
        info = e.content_info()
        assert not [k for k in info if k.startswith('privileged/digit')]
    finally:
        e.close()


def test_floor_digits_still_available_when_asked_for():
    e = make_env(num_digits=1)
    try:
        assert 'digit' in e.variation_space.spaces
        state, _ = content_state(e)
        assert e.render_content(state, DIGIT_APPEARANCE).shape == (128, 128, 3)
    finally:
        e.close()


def test_arm_is_opaque_by_default():
    """The gripper is a content latent; it must not be rendered at alpha 0.1.

    Upstream OGBench fades the arm so it does not occlude the object in a
    manipulation benchmark. Here ``effector.pos``, ``effector.yaw`` and
    ``gripper.opening`` are latents the encoder is *required* to recover, so
    fading them would be asking it to recover something deliberately hidden.
    """
    e = make_env()
    try:
        assert e._pixel_transparent_arm is False
        for name in ('ur5e/robotiq/black', 'ur5e/robotiq/pad_gray',
                     'ur5e/robotiq/metal', 'ur5e/black'):
            alpha = e._model.mat_rgba[e._model.material(name).id, 3]
            assert alpha == pytest.approx(1.0), (
                f'{name} renders at alpha {alpha}, not opaque'
            )
    finally:
        e.close()


def test_opaque_arm_is_more_visible_than_the_faded_one():
    """The change must actually put gripper pixels in the frame.

    Measured rather than asserted from the flag: comparing the two renders is
    what shows the fade was materially hiding the content latent.
    """
    opaque = make_env(width=224, height=224)
    faded = make_env(width=224, height=224, pixel_transparent_arm=True)
    try:
        state, _ = content_state(opaque)
        a = opaque.render_content(state, APPEARANCE).astype(np.float64)
        state_b, _ = content_state(faded)
        b = faded.render_content(state_b, APPEARANCE).astype(np.float64)
        rms = float(np.sqrt(((a - b) ** 2).mean()))
        assert rms > 5.0, (
            f'opaque and faded arms differ by only rms {rms:.3f}; the '
            'transparency setting is not reaching the render'
        )
    finally:
        opaque.close()
        faded.close()


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

"""Tests for the latent registry.

The registry decides what ``z`` is. Every failure mode here is silent: a
mis-declared bound produces a dataset whose latents are systematically clipped,
a mis-ordered concatenation produces recovery metrics that score the wrong
coordinate against the wrong target, and neither raises.
"""

import os

import numpy as np
import pytest


os.environ.setdefault('MUJOCO_GL', 'glfw')

pytest.importorskip('ogbench')
pytest.importorskip('mujoco')

from stable_worldmodel.envs.ogbench.lejepa_cube_env import (  # noqa: E402
    LeJEPACubeEnv,
)
from stable_worldmodel.identifiability.latents import (  # noqa: E402
    PHYSICAL_CONTENT_PROFILE,
    PROFILES,
    LatentRegistry,
    build_registry,
)


# Role dicts declared *here*, not shipped by the library. The study ships one
# profile; these exist only to exercise properties that must hold for any
# assignment of roles -- that the slices tile `z`, that every consumer agrees
# on the order, that `describe` separates two different z's. Keeping them local
# means the library's profile list stays the list of profiles anyone runs.
ALL_CONTENT = {
    '*': 'content',
    'camera.angle_delta': 'excluded',
    'background.floor_material': 'style',
    'background.wall_material': 'style',
    '?digit.value': 'style',
    '?marker.value': 'style',
}

#: `physical_content` minus `cube.size`: every content latent has a
#: `privileged/*` or `proprio/*` readback, so a simulator round-trip can be
#: checked coordinate by coordinate.
READABLE_ONLY = {
    '*': 'style',
    'camera.angle_delta': 'excluded',
    'cube.pos_xy': 'content',
    'cube.pos_z': 'content',
    'cube.yaw': 'content',
    'effector.pos': 'content',
    'effector.yaw': 'content',
    'gripper.opening': 'content',
}


@pytest.fixture(scope='module')
def env():
    e = LeJEPACubeEnv(
        env_type='single',
        ob_type='pixels',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=128,
        height=128,
        num_digits=1,
    )
    e.reset(seed=0, options={'variation': ['all']})
    yield e
    e.close()


@pytest.fixture(scope='module')
def registry(env):
    return build_registry(env)


# ------------------------------------------------------------------ profiles


def test_physical_content_is_ten_dims_with_cube_size(registry):
    """The stage-A exit-criterion profile: 9 physical DOFs + ``cube.size``.

    ``cube.size`` is content, not style, because it shares the
    apparent-footprint cue with ``cube.pos_z`` and dominates it: sweeping size
    moves the cube's footprint 153 -> 585 px while sweeping ``pos_z`` over its
    *entire* range moves it only 272 -> 428 px. As style it would demand a
    size-invariant height estimate from shading and contact cues alone, on a
    20x20 px patch. Pinned here because moving it back to style silently
    reintroduces that confound.
    """
    resolved = registry.resolve(PHYSICAL_CONTENT_PROFILE)
    assert resolved.n == 10, resolved.summary()
    assert 'cube.size' in {latent.name for latent in resolved.content}


def test_camera_angle_is_excluded_from_every_shipped_profile(registry):
    """Camera DR is off, and stays off by construction.

    At +-10 deg, ``camera.angle_delta`` moves the cube's image centroid 32-36
    px, while ``cube.pos_xy``'s x component moves only 24-31 px across its
    entire range -- a nuisance larger than the signal it hides. Excluded rather
    than promoted to content because eval holds the camera fixed; if that
    changes, promote it to content rather than returning it to style.
    """
    for name, profile in PROFILES.items():
        resolved = registry.resolve(profile)
        excluded = {latent.name for latent in resolved.by_role('excluded')}
        assert 'camera.angle_delta' in excluded, (
            f'{name}: {resolved.summary()}'
        )


def test_profile_rejects_an_unknown_latent(registry):
    """A typo must not silently change ``n``."""
    with pytest.raises(KeyError, match='not in the registry'):
        registry.resolve({'cube.postion_xy': 'content'})


def test_profile_rejects_promoting_an_incapable_latent(registry):
    """Guards against putting an unrecoverable target into ``z``.

    A latent with no image correlate would drive the recovery floor down for a
    reason that has nothing to do with the encoder.
    """
    incapable = LatentRegistry(
        [
            latent.__class__(
                **{
                    **latent.__dict__,
                    'content_capable': False,
                    'notes': 'no image correlate',
                }
            )
            for latent in registry.latents[:1]
        ]
    )
    name = incapable.latents[0].name
    with pytest.raises(ValueError, match='cannot be content'):
        incapable.resolve({name: 'content'})


def test_resolve_does_not_mutate_the_original(registry):
    before = dict(registry.roles)
    registry.resolve(PHYSICAL_CONTENT_PROFILE)
    assert registry.roles == before


# ---------------------------------------------------------- the z <-> physical map


def test_z_to_physical_round_trips(registry):
    """The affine must be exactly invertible inside the bounds."""
    resolved = registry.resolve(PHYSICAL_CONTENT_PROFILE)
    rng = np.random.default_rng(0)
    # Inside +-sigma_span, so nothing clips.
    z = rng.uniform(-2.5, 2.5, (64, resolved.n))

    values, clipped = resolved.to_physical(z)
    assert clipped == 0.0
    np.testing.assert_allclose(resolved.to_z(values), z, atol=1e-9)


def test_z_to_physical_respects_declared_bounds(registry):
    """Values must land inside what the environment will accept."""
    resolved = registry.resolve(PHYSICAL_CONTENT_PROFILE)
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4096, resolved.n)) * 3.0

    values, _ = resolved.to_physical(z)
    for latent in resolved.content:
        got = values[latent.name]
        assert (got >= latent.low - 1e-12).all()
        assert (got <= latent.high + 1e-12).all()


def test_clipping_is_reported_not_hidden(registry):
    """A support truncation must be visible to the caller.

    At sigma_span = 3 roughly 0.27% of Gaussian draws land outside the bounds,
    so the clip is not free -- and an unreported clip is an undeclared V5.
    """
    resolved = registry.resolve(PHYSICAL_CONTENT_PROFILE)
    rng = np.random.default_rng(2)

    _, none = resolved.to_physical(rng.uniform(-1, 1, (2048, resolved.n)))
    assert none == 0.0

    _, some = resolved.to_physical(rng.standard_normal((20_000, resolved.n)))
    assert 0.0 < some < 0.02, f'unexpected clip fraction {some}'


def test_sigma_span_puts_the_bounds_at_three_sigma(registry):
    """The declared convention, checked rather than trusted."""
    resolved = registry.resolve(PHYSICAL_CONTENT_PROFILE)
    at_three = np.full((1, resolved.n), resolved.sigma_span)
    values, clipped = resolved.to_physical(at_three)
    assert clipped == 0.0
    for latent in resolved.content:
        np.testing.assert_allclose(
            values[latent.name][0], latent.high, atol=1e-9
        )


# ------------------------------------------------------------------ ordering


def test_slices_tile_z_exactly(registry):
    """``z`` is the concatenation of the content latents, with no gaps."""
    resolved = registry.resolve(ALL_CONTENT)
    slices = resolved.slices()
    covered = np.zeros(resolved.n, dtype=int)
    for latent in resolved.content:
        covered[slices[latent.name]] += 1
    np.testing.assert_array_equal(covered, 1)


def test_slice_order_follows_registry_order(registry):
    """Every consumer walks the registry in order; they must all agree."""
    resolved = registry.resolve(ALL_CONTENT)
    starts = [
        resolved.slices()[latent.name].start for latent in resolved.content
    ]
    assert starts == sorted(starts)


# ------------------------------------------------------------ reading back


def test_read_info_recovers_the_physical_latents(env, registry):
    """Ground truth must come from the simulator, not from the request.

    The two part company wherever a value was clipped, IK failed to converge,
    or a coupled joint did not track its driver -- which is exactly when the
    difference matters.
    """
    resolved = registry.resolve(PHYSICAL_CONTENT_PROFILE)
    rng = np.random.default_rng(3)
    z = rng.uniform(-2.0, 2.0, (1, resolved.n))
    values, _ = resolved.to_physical(z)

    physical = {
        'cube.pos_xy': values['cube.pos_xy'].reshape(1, 2),
        'cube.pos_z': values['cube.pos_z'].reshape(1),
        'cube.yaw': values['cube.yaw'].reshape(1),
        'effector.pos': values['effector.pos'].reshape(3),
        'effector.yaw': float(values['effector.yaw'][0, 0]),
        'gripper.opening': float(values['gripper.opening'][0, 0]),
    }
    state = env.set_content_state(physical)
    env.render_content(state, {})
    info = env.content_info()

    read = resolved.read_info(info, num_cubes=1)
    for name in ('cube.pos_xy', 'cube.pos_z', 'cube.yaw'):
        np.testing.assert_allclose(
            read[name], values[name].reshape(-1), atol=1e-6
        )
    # IK-reached quantities round-trip to the solver's tolerance.
    np.testing.assert_allclose(
        read['effector.pos'], values['effector.pos'].reshape(-1), atol=1e-3
    )
    np.testing.assert_allclose(
        read['gripper.opening'],
        values['gripper.opening'].reshape(-1),
        atol=1e-6,
    )


def test_readback_z_matches_requested_z(env, registry):
    """The full loop: z -> physical -> sim -> info -> z.

    On ``arm_c_content``, not ``task_content`` or ``physical_content``: this is
    a *readback* loop, and it closes only for latents the simulator reports
    back through ``content_info()``. ``cube.size`` is excluded from the role
    dict used here because it is a variation axis with no ``privileged/*``
    entry, so it has no place in a round-trip that goes through the simulator
    at all -- and under the shipped profile that makes it the one content
    latent whose presence in the render no recorded column can confirm.
    """
    resolved = registry.resolve(READABLE_ONLY)
    rng = np.random.default_rng(4)
    z = rng.uniform(-2.0, 2.0, (1, resolved.n))
    values, _ = resolved.to_physical(z)

    state = env.set_content_state(
        {
            'cube.pos_xy': values['cube.pos_xy'].reshape(1, 2),
            'cube.pos_z': values['cube.pos_z'].reshape(1),
            'cube.yaw': values['cube.yaw'].reshape(1),
            'effector.pos': values['effector.pos'].reshape(3),
            'effector.yaw': float(values['effector.yaw'][0, 0]),
            'gripper.opening': float(values['gripper.opening'][0, 0]),
        }
    )
    env.render_content(state, {})
    read = resolved.read_info(env.content_info(), num_cubes=1)

    recovered = resolved.to_z(
        {name: read[name].reshape(1, -1) for name in resolved.slices()}
    )
    # Everything but the IK-reached coordinates is exact; those carry the
    # solver's residual, scaled into z-space by the registry's own affine.
    np.testing.assert_allclose(recovered, z, atol=0.05)


# ------------------------------------------------------------------ manifest


def test_describe_distinguishes_profiles(registry):
    """A dataset must not be mistakable for a differently-configured one."""
    a = registry.resolve(PHYSICAL_CONTENT_PROFILE).describe()
    b = registry.resolve(ALL_CONTENT).describe()
    assert a != b
    assert a['n'] != b['n']


def test_every_latent_records_why_its_bounds_are_what_they_are(registry):
    """Bounds without a rationale become folklore. Physical latents at least."""
    for latent in registry.latents:
        if latent.kind == 'physical':
            assert latent.notes, f'{latent.name} has no notes'


def test_gripper_bounds_come_from_the_measured_linkage(env, registry):
    """The registry must not re-declare what the environment measured."""
    latent = next(
        latent
        for latent in registry.latents
        if latent.name == 'gripper.opening'
    )
    lo, hi = env.gripper_opening_bounds
    assert float(latent.low[0]) == pytest.approx(lo)
    assert float(latent.high[0]) == pytest.approx(hi)


def test_floor_color_is_not_content_capable(registry):
    """It would force an MJCF recompile per frame; background.floor_rgb wins."""
    names = {latent.name for latent in registry.latents}
    assert 'floor.color' not in names
    assert 'background.floor_rgb' in names


def test_yaw_arc_is_a_declared_truncation(env):
    """The SS9 wraparound mitigation must be a knob with a recorded rationale."""
    narrow = build_registry(env, yaw_half_arc=0.4)
    latent = next(
        latent for latent in narrow.latents if latent.name == 'cube.yaw'
    )
    assert float(latent.high[0]) == pytest.approx(0.4)
    assert 'truncation' in latent.notes.lower()

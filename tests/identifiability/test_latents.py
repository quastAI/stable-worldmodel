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
    ALL_CONTENT_PROFILE,
    TASK_CONTENT_PROFILE,
    LatentRegistry,
    build_registry,
)


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


def test_task_content_profile_has_nine_physical_dims(registry):
    """The plan's SS3.1 subtotal, as a gate.

    ``task_content`` is the stage-A exit-criterion profile, and its ``n`` is
    quoted throughout the plan. If the registry drifts, every stage-A number
    silently refers to a different problem.
    """
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
    assert resolved.n == 9, resolved.summary()
    assert {latent.kind for latent in resolved.content} == {'physical'}


def test_task_content_leaves_style_to_discard(registry):
    """The profile that actually exercises the alignment loss must have style."""
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
    assert len(resolved.style) > 0


def test_all_content_leaves_no_continuous_style(registry):
    """The plan's SS3.5 consequence, asserted rather than assumed.

    Under ``all_content`` every content-capable latent is content, so
    within-pair style resampling is empty and the style-invariance metric is
    vacuous. That is coherent with the theory but means this profile does not
    test the discard behaviour at all -- which is exactly why both profiles
    are run.
    """
    resolved = registry.resolve(ALL_CONTENT_PROFILE)
    continuous_style = [
        latent for latent in resolved.style if latent.kind != 'discrete'
    ]
    assert continuous_style == [], (
        f'all_content still has continuous style latents: '
        f'{[latent.name for latent in continuous_style]}'
    )
    assert resolved.n > registry.resolve(TASK_CONTENT_PROFILE).n


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
    registry.resolve(TASK_CONTENT_PROFILE)
    assert registry.roles == before


# ---------------------------------------------------------- the z <-> physical map


def test_z_to_physical_round_trips(registry):
    """The affine must be exactly invertible inside the bounds."""
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
    rng = np.random.default_rng(0)
    # Inside +-sigma_span, so nothing clips.
    z = rng.uniform(-2.5, 2.5, (64, resolved.n))

    values, clipped = resolved.to_physical(z)
    assert clipped == 0.0
    np.testing.assert_allclose(resolved.to_z(values), z, atol=1e-9)


def test_z_to_physical_respects_declared_bounds(registry):
    """Values must land inside what the environment will accept."""
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
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
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
    rng = np.random.default_rng(2)

    _, none = resolved.to_physical(rng.uniform(-1, 1, (2048, resolved.n)))
    assert none == 0.0

    _, some = resolved.to_physical(rng.standard_normal((20_000, resolved.n)))
    assert 0.0 < some < 0.02, f'unexpected clip fraction {some}'


def test_sigma_span_puts_the_bounds_at_three_sigma(registry):
    """The declared convention, checked rather than trusted."""
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
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
    resolved = registry.resolve(ALL_CONTENT_PROFILE)
    slices = resolved.slices()
    covered = np.zeros(resolved.n, dtype=int)
    for latent in resolved.content:
        covered[slices[latent.name]] += 1
    np.testing.assert_array_equal(covered, 1)


def test_slice_order_follows_registry_order(registry):
    """Every consumer walks the registry in order; they must all agree."""
    resolved = registry.resolve(ALL_CONTENT_PROFILE)
    starts = [resolved.slices()[latent.name].start
              for latent in resolved.content]
    assert starts == sorted(starts)


# ------------------------------------------------------------ reading back


def test_read_info_recovers_the_physical_latents(env, registry):
    """Ground truth must come from the simulator, not from the request.

    The two part company wherever a value was clipped, IK failed to converge,
    or a coupled joint did not track its driver -- which is exactly when the
    difference matters.
    """
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
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
    """The full loop: z -> physical -> sim -> info -> z."""
    resolved = registry.resolve(TASK_CONTENT_PROFILE)
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
    a = registry.resolve(TASK_CONTENT_PROFILE).describe()
    b = registry.resolve(ALL_CONTENT_PROFILE).describe()
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
        latent for latent in registry.latents
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

"""The latent registry: every true latent of Cube-single, and its role.

This is the single place that decides what ``z`` *is*. The sampler draws in
z-space, the environment renders from physical units, and the metric suite
reads ground truth back out of recorded columns -- all three go through the
maps defined here, so they cannot disagree about what coordinate 17 means.

Roles
-----
``content``
    Part of ``z``. Held fixed within a positive pair up to the OU step, and
    scored by the recovery metrics.
``style``
    Resampled *independently* within a positive pair. The alignment term is
    supposed to discard it; the style-invariance metric measures whether it did.
``excluded``
    Neither. Pinned to a constant, and recorded so the exclusion is a decision
    on the record rather than an oversight.

A latent may be ``content`` only if it (a) has an image correlate in the
recorded frame and (b) is settable frame-by-frame without stepping physics.
:data:`CONTENT_CAPABLE` records that judgement per latent, and
:meth:`LatentRegistry.resolve` refuses a profile that promotes an incapable
latent -- silently accepting one would put a target into ``z`` that no encoder
could recover, and the resulting floor would look exactly like an encoder
failure.

``cube.size`` is content in every shipped profile, and ``camera.angle_delta``
is excluded from all of them -- both under one rule: **a nuisance that shares a
rendering channel with a content latent must not be style**, because style
demands exact invariance along a direction the content itself depends on. The
per-latent measurements behind that are recorded with the profiles below.

The profile
-----------
``physical_content``
    The only shipped profile. The 9 physical DOFs plus ``cube.size``, so
    ``n = 10`` at one cube. Everything else with a rendering channel is
    ``style`` and is resampled independently within a pair;
    ``camera.angle_delta`` is ``excluded`` and pinned.

    The profiles that used to sit beside this one -- ``task_content`` (with
    ``cube.color`` promoted), ``arm_c_content`` (physical readbacks only) and
    ``all_content`` -- served comparisons between experimental arms that no
    longer exist. Re-adding one is a dict of roles plus a line in
    :data:`PROFILES`, which is the point of keeping the mechanism.
"""

from dataclasses import dataclass, field

import numpy as np

from stable_worldmodel.identifiability.ou import SIGMA_SPAN


@dataclass(frozen=True)
class Latent:
    """One true latent variable of the environment.

    Attributes:
        name: Dotted key, unique in the registry.
        dim: Number of scalar coordinates.
        kind: ``'physical'``, ``'appearance'`` or ``'discrete'``.
        low: ``(dim,)`` lower bound in physical units.
        high: ``(dim,)`` upper bound in physical units.
        channel: How the value reaches the environment -- ``'state'`` for
            anything written into ``qpos``, or ``'variation:<axis>'`` for a
            variation-space axis.
        readback: Info key the ground truth is read back from, with
            ``slice_`` selecting the coordinates within it.
        slice_: Slice into the readback array.
        default_role: Role under the ``all_content`` profile.
        content_capable: Whether this latent may ever be content. ``False``
            means no profile can promote it; ``notes`` says why.
        notes: Why the bounds, the capability verdict or the default are what
            they are.
    """

    name: str
    dim: int
    kind: str
    low: np.ndarray
    high: np.ndarray
    channel: str
    readback: str | None
    slice_: slice
    default_role: str
    content_capable: bool
    notes: str = ''
    axis_rows: tuple | None = None
    """Rows of the variation axis this latent occupies, when it covers only
    some of them.

    ``light.position`` is the case that forces this to exist: the axis is
    ``(n_lights, 3)`` but the directional light's row has no image correlate
    and is pinned, so the latent is 3-dimensional over a 6-dimensional axis.
    Without an explicit row map the collector would either reshape-error or,
    worse, silently write the latent into the wrong rows.
    """

    def center(self):
        """Midpoint of the physical range."""
        return 0.5 * (self.low + self.high)

    def half_span(self):
        """Half-width of the physical range."""
        return 0.5 * (self.high - self.low)


# Latents that cannot be content, and the reason. Kept as data rather than as
# prose so `resolve` can enforce it.
NOT_CONTENT_CAPABLE = {
    'light.position_global': 'Directional light -- MuJoCo shades from '
    'light_dir alone, so the position never reaches a pixel.',
    'light.intensity': 'Pinned to PINNED_LIGHT_INTENSITY and superseded by '
    'light.diffuse.',
    'digit.count': 'Pinned to Discrete(1, start=1) at num_digits == 1.',
    'cube.goal_position': 'Target mocaps are invisible at '
    'visualize_info=False, and render_content parks them out of frame.',
    'cube.goal_yaw': 'As cube.goal_position.',
    'qvel': 'No single-frame image correlate; the encoder is applied per '
    'frame and every encoder frame pins qvel to zero.',
    'floor.color': 'The rgb1/rgb2 of a builtin procedural texture, '
    'rasterized at compile time. Superseded by background.floor_rgb, which '
    'reaches the same pixels without a recompile.',
}


# The cube's yaw symmetry group is C4: a rotation of pi/2 about z maps the cube
# exactly onto itself -- zero differing pixels, and the same grasp. A
# fundamental domain is therefore an arc of width pi/2, and every physically
# distinct orientation has exactly one representative in it.
YAW_FUNDAMENTAL_HALF_ARC = np.pi / 4

# ...but the two endpoints of that arc, -pi/4 and +pi/4, denote the SAME
# configuration (they differ by exactly pi/2). Sampling the full closed domain
# would pile the ~0.27% of clipped draws onto both ends, and those two piles
# are one physical state under two z-values -- a genuine, if small, injectivity
# failure of precisely the kind these metrics exist to detect. The margin costs
# 5% of the arc and removes it outright.
DEFAULT_YAW_HALF_ARC = 0.95 * YAW_FUNDAMENTAL_HALF_ARC

# The gripper's jaws are symmetric under a pi rotation about the approach axis,
# so effector yaw is meaningful only modulo pi -- a fundamental domain of width
# pi, i.e. [-pi/2, pi/2].
EFFECTOR_YAW_HALF_ARC = np.pi / 2

#: Symmetry order of each circular latent: how many times its physical
#: configuration repeats around one full turn, i.e. ``2 * pi / period``.
#:
#: This is what a *representation* of the angle has to respect, and it is why a
#: plain ``(cos theta, sin theta)`` read-out is the wrong probe here. Because
#: ``theta`` and ``theta + period`` are the same physical state -- identical
#: pixels -- any function of the image is periodic with that period, so the
#: lowest harmonics available to the encoder are ``cos(m theta)`` and
#: ``sin(m theta)``, not ``cos theta`` and ``sin theta``.
#:
#: Over the sampled fundamental domain ``m * theta`` covers (almost) exactly one
#: full circle, so the harmonic pair is a non-degenerate coordinate there:
#: ``cube.yaw`` spans ``m*theta`` in ``+-0.95 pi`` and ``effector.yaw`` in
#: ``+-pi``. ``theta`` itself is *also* single-valued on that domain, which is
#: why the ordinary linear probe is not wrong -- it just asks a different
#: question. Scoring both is what separates "the angle is not represented" from
#: "the angle is represented in its natural circular coding".
CIRCULAR_ORDER = {
    'cube.yaw': 4,  # C4 about z: period pi/2
    'effector.yaw': 2,  # jaws symmetric under pi: period pi
}


def _box(low, high, dim):
    return (
        np.broadcast_to(np.asarray(low, dtype=np.float64), (dim,)).copy(),
        np.broadcast_to(np.asarray(high, dtype=np.float64), (dim,)).copy(),
    )


def build_registry(
    env,
    yaw_half_arc=DEFAULT_YAW_HALF_ARC,
    pos_z_max=0.15,
):
    """Enumerate the latents of a configured :class:`LeJEPACubeEnv`.

    Bounds are read off the environment wherever it owns them -- sampling
    bounds, variation-space ranges, the measured gripper band -- rather than
    duplicated here, so the registry cannot drift away from what the
    environment will actually accept.

    Args:
        env: A ``LeJEPACubeEnv`` that has been reset at least once.
        yaw_half_arc: Half-width of the sampled ``cube.yaw`` arc, in radians.
            Defaults to just inside :data:`YAW_FUNDAMENTAL_HALF_ARC`, one
            fundamental domain of the cube's 4-fold symmetry. Every physically
            distinct orientation has exactly one representative there, so yaw
            is identifiable from the silhouette alone and no marker is needed
            -- and no coordinate is scored on a quadrant index that no grasp
            depends on. Widening this past ``pi/4`` reintroduces the 4-fold
            ambiguity and requires ``marker_enabled=True`` to stay
            identifiable at all.
        pos_z_max: Upper bound on cube height. Above the table by design:
            lifted and interpenetrating cubes are deliberately allowed, so
            that the induced V5 baseline is zero by construction in stage A.

    Returns:
        LatentRegistry: The full inventory, roles at their defaults.
    """
    n_cubes = env._num_cubes
    latents = []

    # ---------------------------------------------------------- physical
    obj_lo, obj_hi = env._object_sampling_bounds
    latents.append(
        Latent(
            'cube.pos_xy',
            2 * n_cubes,
            'physical',
            *_box(
                np.tile(obj_lo, n_cubes), np.tile(obj_hi, n_cubes), 2 * n_cubes
            ),
            channel='state',
            readback='privileged/block_{i}_pos',
            slice_=slice(0, 2),
            default_role='content',
            content_capable=True,
            notes="Bounded by the environment's own object sampling bounds.",
        )
    )
    latents.append(
        Latent(
            'cube.pos_z',
            n_cubes,
            'physical',
            *_box(0.0, pos_z_max, n_cubes),
            channel='state',
            readback='privileged/block_{i}_pos',
            slice_=slice(2, 3),
            default_role='content',
            content_capable=True,
            notes='Lifted and table-clipped cubes are allowed on purpose, so '
            'the induced V5 baseline is zero in stage A.',
        )
    )
    latents.append(
        Latent(
            'cube.yaw',
            n_cubes,
            'physical',
            *_box(-yaw_half_arc, yaw_half_arc, n_cubes),
            channel='state',
            readback='privileged/block_{i}_yaw',
            slice_=slice(0, 1),
            default_role='content',
            content_capable=True,
            notes=f'Sampled on a half-arc of {yaw_half_arc:.4f} rad. A '
            f'rotation of pi/2 maps the cube exactly onto itself -- same '
            f'pixels, same grasp -- so this is one fundamental domain of the '
            f'C4 symmetry, not an arbitrary truncation: it is a coordinate on '
            f'the quotient. Identifiable from the silhouette alone (rms 6.1 '
            f'between the arc extremes); no marker required.',
        )
    )
    arm_lo, arm_hi = env._arm_sampling_bounds
    latents.append(
        Latent(
            'effector.pos',
            3,
            'physical',
            *_box(arm_lo, arm_hi, 3),
            channel='state',
            readback='proprio/effector_pos',
            slice_=slice(0, 3),
            default_role='content',
            content_capable=True,
            notes="Reached by IK; round-trips to the solver's tolerance.",
        )
    )
    latents.append(
        Latent(
            'effector.yaw',
            1,
            'physical',
            *_box(-EFFECTOR_YAW_HALF_ARC, EFFECTOR_YAW_HALF_ARC, 1),
            channel='state',
            readback='privileged/ee_yaw',
            slice_=slice(0, 1),
            default_role='content',
            content_capable=True,
            notes='Needs LeJEPACubeEnv.agent.ee_start_yaw; the stock env draws '
            'this from its own RNG. Restricted to [-pi/2, pi/2]: the jaws are '
            'symmetric under a pi rotation about the approach axis, so this '
            "is one fundamental domain of the jaws' 2-fold symmetry -- the "
            'same quotient argument as cube.yaw.',
        )
    )
    grip_lo, grip_hi = env.gripper_opening_bounds
    latents.append(
        Latent(
            'gripper.opening',
            1,
            'physical',
            *_box(grip_lo, grip_hi, 1),
            channel='state',
            readback='proprio/gripper_opening',
            slice_=slice(0, 1),
            default_role='content',
            content_capable=True,
            notes=f"Bounds are the linkage's *measured* achievable range "
            f'[{grip_lo:.4f}, {grip_hi:.4f}], not [0, 1]: the actuator cannot '
            f'reach either joint limit, and sampling outside the achievable '
            f'band would put point masses at the endpoints.',
        )
    )

    # -------------------------------------------------------- appearance
    space = env.variation_space

    def var_latent(name, axis_path, dim, readback, notes='', role='content'):
        axis = space
        for part in axis_path.split('.'):
            axis = axis[part]
        return Latent(
            name,
            dim,
            'appearance',
            *_box(
                np.asarray(axis.low).reshape(-1),
                np.asarray(axis.high).reshape(-1),
                dim,
            ),
            channel=f'variation:{axis_path}',
            readback=readback,
            slice_=slice(0, dim),
            default_role=role,
            content_capable=True,
            notes=notes,
        )

    latents += [
        var_latent(
            'cube.color',
            'cube.color',
            3 * n_cubes,
            'variation.cube.color',
            'Applied by a direct geom.rgba write.',
        ),
        var_latent(
            'cube.size',
            'cube.size',
            n_cubes,
            'variation.cube.size',
            'Written post-compilation to model.geom_size. Mass and '
            'inertia go stale, which is sound only because the encoder '
            'path never steps physics.',
        ),
        var_latent(
            'agent.color',
            'agent.color',
            3,
            'variation.agent.color',
            'Post-compilation write to model.mat_rgba; renders '
            'bit-identically to the recompiled model.',
        ),
        var_latent(
            'camera.angle_delta',
            'camera.angle_delta',
            2,
            'variation.camera.angle_delta',
            'Post-compilation write to model.cam_quat; renders '
            'bit-identically. Also the V6 occlusion knob.',
        ),
        var_latent(
            'light.direction', 'light.direction', 6, 'privileged/light_dir'
        ),
        var_latent(
            'light.diffuse', 'light.diffuse', 6, 'variation.light.diffuse'
        ),
        var_latent(
            'light.ambient', 'light.ambient', 6, 'variation.light.ambient'
        ),
        var_latent(
            'light.specular', 'light.specular', 6, 'variation.light.specular'
        ),
        var_latent(
            'light.headlight_diffuse',
            'light.headlight_diffuse',
            3,
            'variation.light.headlight_diffuse',
        ),
        var_latent(
            'background.floor_rgb',
            'background.floor_rgb',
            3,
            'privileged/floor_rgb',
            'Supersedes the inherited floor.color, which is pinned '
            'because it would force an MJCF recompile per frame.',
        ),
        var_latent(
            'background.wall_rgb',
            'background.wall_rgb',
            3,
            'privileged/wall_rgb',
        ),
    ]

    # Floor digit distractors, only when the environment actually has them.
    # `LeJEPACubeEnv` defaults to `num_digits=0`: the decals exist to give a
    # linear probe a nuisance target in the DR benchmark, nothing here reads
    # them, and against a scene whose content latents are already small in the
    # frame they are pure occlusion risk. Guarded rather than assumed, because
    # at `num_digits=0` the `digit` sub-space does not exist at all and
    # `var_latent` would raise on the lookup.
    if env._num_digits > 0:
        latents += [
            var_latent(
                'digit.size',
                'digit.size',
                env._num_digits,
                'privileged/digit_0_size',
            ),
            var_latent(
                'digit.position',
                'digit.position',
                2 * env._num_digits,
                'privileged/digit_0_pos',
            ),
            var_latent(
                'digit.yaw',
                'digit.yaw',
                env._num_digits,
                'variation.digit.yaw',
            ),
        ]

    # `light.position` -- only the rows that actually reach a pixel. The
    # directional row is excluded by the environment itself, and duplicating
    # that judgement here is what keeps a probe from being handed an
    # unlearnable target.
    positioned = [
        row
        for row in range(len(np.asarray(space['light']['position'].low)))
        if row not in env._directional_light_rows
    ]
    if positioned:
        low = np.asarray(space['light']['position'].low)[positioned].reshape(
            -1
        )
        high = np.asarray(space['light']['position'].high)[positioned].reshape(
            -1
        )
        latents.append(
            Latent(
                'light.position',
                low.size,
                'appearance',
                low.copy(),
                high.copy(),
                channel='variation:light.position',
                readback='privileged/light_pos',
                slice_=slice(0, low.size),
                default_role='content',
                content_capable=True,
                notes=f'Directional rows {tuple(env._directional_light_rows)} '
                f'excluded -- they have no image correlate.',
                axis_rows=tuple(positioned),
            )
        )

    # ---------------------------------------------------------- discrete
    # Discrete latents are always style: they are renderable, and so
    # content-capable, but a categorical has no Gaussian marginal and the OU
    # sampler draws one.
    discrete_role = 'style'
    discrete_specs = [
        (
            'background.floor_material',
            'background.floor_material',
            env._num_bg_materials,
            'privileged/floor_material',
        ),
        (
            'background.wall_material',
            'background.wall_material',
            env._num_bg_materials,
            'privileged/wall_material',
        ),
    ]
    if env._num_digits > 0:
        discrete_specs.append(
            ('digit.value', 'digit.value', 10, 'privileged/digit_0_value')
        )
    for name, axis_path, card, readback in discrete_specs:
        latents.append(
            Latent(
                name,
                1,
                'discrete',
                *_box(0, card - 1, 1),
                channel=f'variation:{axis_path}',
                readback=readback,
                slice_=slice(0, 1),
                default_role=discrete_role,
                content_capable=True,
                notes='Renderable and therefore content-capable, but has no '
                'Gaussian marginal -- a structural V1. Reported with the '
                'monotone-recovery metric rather than the linear one.',
            )
        )
    if getattr(env, '_marker_enabled', False):
        latents.append(
            Latent(
                'marker.value',
                1,
                'discrete',
                *_box(0, 9, 1),
                channel='variation:marker.value',
                readback='privileged/marker_0_value',
                slice_=slice(0, 1),
                default_role=discrete_role,
                content_capable=True,
                notes="The yaw marker's digit. Structural V1, as above.",
            )
        )

    return LatentRegistry(latents)


@dataclass
class LatentRegistry:
    """An ordered inventory of latents, with roles resolved from a profile.

    ``z`` is the concatenation of every ``content`` latent **in registry
    order**, so ``n`` is a function of the profile and every consumer that
    walks the registry in order agrees about which coordinate is which.
    """

    latents: list
    roles: dict = field(default_factory=dict)
    sigma_span: float = SIGMA_SPAN

    def __post_init__(self):
        names = [latent.name for latent in self.latents]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f'duplicate latent names: {sorted(duplicates)}')
        if not self.roles:
            self.roles = {
                latent.name: latent.default_role for latent in self.latents
            }

    # ------------------------------------------------------------------
    # roles
    # ------------------------------------------------------------------

    def resolve(self, profile):
        """Apply a role profile, returning a new registry.

        Args:
            profile: Mapping from latent name to ``'content'``, ``'style'`` or
                ``'excluded'``. Names absent from the profile keep their
                default role. Two special forms:

                * ``'*'`` sets the default for every name the profile does not
                  mention.
                * a leading ``'?'`` marks the entry **conditional** -- it is
                  applied when the latent exists and ignored when it does not,
                  without weakening the typo check on every other name. This
                  exists because some latents are present only for certain
                  environment configurations: ``digit.*`` only when
                  ``num_digits > 0``, ``marker.value`` only when the yaw marker
                  is enabled. Both default off.

        Returns:
            LatentRegistry: A new registry; this one is unchanged.

        Raises:
            KeyError: If the profile names an unconditional latent that does
                not exist -- usually a typo, and silently ignoring it would
                change ``n`` without saying so.
            ValueError: If it promotes a latent that cannot be content.
        """
        profile = dict(profile or {})
        fallback = profile.pop('*', None)

        # Conditional entries are stripped of their marker and merged in only
        # where the latent actually exists.
        known = {latent.name for latent in self.latents}
        optional = {
            key[1:]: role
            for key, role in profile.items()
            if key.startswith('?')
        }
        profile = {
            key: role
            for key, role in profile.items()
            if not key.startswith('?')
        }
        profile.update(
            {name: role for name, role in optional.items() if name in known}
        )

        unknown = set(profile) - known
        if unknown:
            raise KeyError(
                f'profile names latents that are not in the registry: '
                f'{sorted(unknown)}. Prefix a name with "?" if it is '
                'conditional on the environment configuration.'
            )

        roles = {}
        for latent in self.latents:
            role = profile.get(latent.name, fallback or latent.default_role)
            if role not in ('content', 'style', 'excluded'):
                raise ValueError(
                    f'{latent.name}: role must be content, style or excluded; '
                    f'got {role!r}.'
                )
            if role == 'content' and not latent.content_capable:
                raise ValueError(
                    f'{latent.name} cannot be content: {latent.notes} '
                    'Promoting it would put a target into z that no encoder '
                    'can recover.'
                )
            roles[latent.name] = role

        return LatentRegistry(
            list(self.latents), roles, sigma_span=self.sigma_span
        )

    def by_role(self, role):
        """Latents with the given role, in registry order."""
        return [
            latent
            for latent in self.latents
            if self.roles[latent.name] == role
        ]

    @property
    def content(self):
        """Content latents, in registry order. Their concatenation is ``z``."""
        return self.by_role('content')

    @property
    def style(self):
        """Style latents -- resampled independently within a positive pair."""
        return self.by_role('style')

    @property
    def n(self):
        """Dimension of ``z``: the total width of the content latents."""
        return sum(latent.dim for latent in self.content)

    def slices(self):
        """Map from content-latent name to its slice of ``z``."""
        out = {}
        offset = 0
        for latent in self.content:
            out[latent.name] = slice(offset, offset + latent.dim)
            offset += latent.dim
        return out

    # ------------------------------------------------------------------
    # the z-space <-> physical affine
    # ------------------------------------------------------------------

    def to_physical(self, z, latents=None):
        """Map z-space coordinates to physical units, per latent.

        The affine is ``physical = center + (half_span / sigma_span) * z``,
        clipped to the declared bounds. Clipping is a *support truncation* and
        is reported as one: at the default ``sigma_span = 3`` roughly 0.27% of
        Gaussian draws land outside, so the clip is not free and the caller is
        told how often it bit.

        Args:
            z: ``(B, n)`` in z-space.
            latents: Which latents to decode; defaults to the content set.

        Returns:
            tuple: ``(values, clipped_fraction)``. ``values`` maps latent name
            to a ``(B, dim)`` array in physical units.
        """
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        latents = latents if latents is not None else self.content
        offsets = self.slices()

        values = {}
        clipped = 0
        total = 0
        for latent in latents:
            chunk = z[:, offsets[latent.name]]
            raw = (
                latent.center()
                + (latent.half_span() / self.sigma_span) * chunk
            )
            bounded = np.clip(raw, latent.low, latent.high)
            clipped += int((raw != bounded).sum())
            total += bounded.size
            values[latent.name] = bounded

        return values, (clipped / total if total else 0.0)

    def to_z(self, values, missing='raise'):
        """Invert :meth:`to_physical`, for reading ground truth back.

        Args:
            values: Map from latent name to ``(B, dim)`` physical values. It
                may cover only some content latents -- see ``missing``.
            missing: What to do about a content latent absent from ``values``.
                ``'raise'`` (the default) refuses, because a silently-zeroed
                coordinate would be scored as a recovered one. ``'nan'`` fills
                its columns with NaN, which is what a caller comparing only the
                readable coordinates wants: under the shipped profile
                ``cube.size`` has no ``privileged/*`` / ``proprio/*`` readback,
                so any round-trip through the simulator is necessarily partial.

        Returns:
            ndarray: ``(B, n)`` in z-space, content latents in registry order.

        Raises:
            KeyError: On a missing latent when ``missing='raise'``.
        """
        if missing not in ('raise', 'nan'):
            raise ValueError(
                f"missing must be 'raise' or 'nan'; got {missing!r}."
            )

        rows = max(
            (np.atleast_2d(np.asarray(v)).shape[0] for v in values.values()),
            default=1,
        )
        columns = []
        for latent in self.content:
            if latent.name not in values:
                if missing == 'raise':
                    raise KeyError(
                        f'{latent.name} is a content latent but was not '
                        f'supplied. Pass missing="nan" to read back only the '
                        'coordinates you have; filling it silently would '
                        'score an absent coordinate as a recovered one.'
                    )
                columns.append(np.full((rows, latent.dim), np.nan))
                continue
            physical = np.atleast_2d(
                np.asarray(values[latent.name], dtype=np.float64)
            )
            scale = latent.half_span() / self.sigma_span
            safe = np.where(scale > 0, scale, 1.0)
            columns.append((physical - latent.center()) / safe)
        return np.concatenate(columns, axis=1)

    # ------------------------------------------------------------------
    # reading ground truth out of a recorded row
    # ------------------------------------------------------------------

    def read_info(self, info, num_cubes=1):
        """Assemble physical values from an environment info dict.

        The recovery target must come from the *simulator*, not from what the
        sampler asked for: the two part company wherever a value was clipped,
        IK failed to converge, or a coupled joint did not track its driver.

        Args:
            info: One ``content_info()`` / ``get_reset_info()`` dict.
            num_cubes: Number of cubes, for the per-cube readback keys.

        Returns:
            dict: Latent name -> ``(dim,)`` physical values.
        """
        values = {}
        for latent in self.latents:
            if latent.readback is None:
                continue
            if '{i}' in latent.readback:
                parts = [
                    np.asarray(
                        info[latent.readback.format(i=i)], dtype=np.float64
                    ).reshape(-1)[latent.slice_]
                    for i in range(num_cubes)
                ]
                values[latent.name] = np.concatenate(parts)
            elif latent.readback in info:
                values[latent.name] = np.asarray(
                    info[latent.readback], dtype=np.float64
                ).reshape(-1)[: latent.dim]
        return values

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def describe(self):
        """The registry as plain data, for the dataset manifest."""
        return {
            'n': self.n,
            'sigma_span': self.sigma_span,
            'latents': [
                {
                    'name': latent.name,
                    'dim': latent.dim,
                    'kind': latent.kind,
                    'role': self.roles[latent.name],
                    'content_capable': latent.content_capable,
                    'low': latent.low.tolist(),
                    'high': latent.high.tolist(),
                    'channel': latent.channel,
                    'readback': latent.readback,
                    'axis_rows': latent.axis_rows,
                    'notes': latent.notes,
                }
                for latent in self.latents
            ],
        }

    def summary(self):
        """One-line-per-latent table, for a collection script's log."""
        lines = [f'{"latent":<28}{"dim":>5}{"kind":>12}{"role":>10}']
        lines.append('-' * 55)
        for latent in self.latents:
            lines.append(
                f'{latent.name:<28}{latent.dim:>5}{latent.kind:>12}'
                f'{self.roles[latent.name]:>10}'
            )
        lines.append('-' * 55)
        lines.append(
            f'n = {self.n} content dims across '
            f'{len(self.content)} latents; {len(self.style)} style, '
            f'{len(self.by_role("excluded"))} excluded'
        )
        return '\n'.join(lines)


# `cube.size` is content in every shipped profile, and `camera.angle_delta` is
# excluded from all of them. Both follow one rule: **a nuisance that shares a
# rendering channel with a content latent must not be style.**
#
# As style, a latent is resampled independently within a pair, so the alignment
# term asks the encoder to project that direction *out* -- to be exactly
# invariant to it. That is only cheap when the direction is orthogonal to
# everything being recovered. It is not, for these two:
#
# * ``cube.size`` and ``cube.pos_z`` share the apparent-footprint cue, and size
#   dominates it. Measured at 224px: sweeping size over its range moves the
#   cube's footprint 153 -> 585 px, while sweeping ``pos_z`` over its *entire*
#   +-3 sigma moves it only 272 -> 428 px. The content interval sits strictly
#   inside the style interval, so footprint carries no information about height
#   and a size-invariant height estimate has to come from shading and contact
#   cues alone, on a 20x20 px patch. As *content* the requirement collapses
#   from "project size out" to "span the 2-D subspace size and height
#   generate" -- and since identifiability is only up to a rotation *within*
#   the content block, a mixture is an acceptable optimum that a linear probe
#   separates afterwards. Same cues, far weaker requirement.
#
# * ``camera.angle_delta`` shares the projection channel with every positional
#   latent. Measured: +-10 deg on one axis moves the cube's image centroid
#   32-36 px, while ``cube.pos_xy``'s x component moves only 24-31 px across
#   its whole range -- the nuisance is larger than the entire signal. Recovering
#   world-frame position under it means inferring camera pose from the floor
#   grid, whose material and colour are themselves style. Excluded rather than
#   promoted to content because the eval side is ours to choose and holds the
#   camera fixed; if that ever changes, promote it to content (n += 2) rather
#   than returning it to style.
#
# Excluded latents are pinned to their axis's canonical ``init_value`` by
# `identifiability.collect.excluded_payload`, not merely left unwritten -- see
# that function for why "unwritten" was a shard-dependent nuisance.

# The study's profile. n = 10 at n_cubes = 1: the 9 physical DOFs plus
# `cube.size`.
#
# `cube.size` has no `privileged/*` / `proprio/*` readback -- it reads back from
# `variation.cube.size` -- so it is scored from the recorded `latent/z` column,
# from the recorded column alone. That is sound because it
# is a direct post-compilation `model.geom_size` write with nothing in between
# to clip or decouple, so there is no readback divergence for a physical
# readback to detect -- and it is the one latent whose presence in the *render*
# no recorded column can confirm, which is worth knowing when it scores near
# zero.
PHYSICAL_CONTENT_PROFILE = {
    '*': 'style',
    'camera.angle_delta': 'excluded',
    'cube.pos_xy': 'content',
    'cube.pos_z': 'content',
    'cube.yaw': 'content',
    'cube.size': 'content',
    'effector.pos': 'content',
    'effector.yaw': 'content',
    'gripper.opening': 'content',
}

#: Every shipped role assignment, by name. One entry: the study trains one
#: encoder on one profile.
PROFILES = {'physical_content': PHYSICAL_CONTENT_PROFILE}


__all__ = [
    'CIRCULAR_ORDER',
    'NOT_CONTENT_CAPABLE',
    'PHYSICAL_CONTENT_PROFILE',
    'PROFILES',
    'Latent',
    'LatentRegistry',
    'build_registry',
]

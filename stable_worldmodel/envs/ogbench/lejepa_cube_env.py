"""Cube environment for LeJEPA identifiability data generation.

:class:`LeJEPACubeEnv` extends :class:`~stable_worldmodel.envs.ogbench.dr_cube_env.DRCubeEnv`
with what the encoder dataset (see the Env-1 implementation plan, SS5.B) needs
and the stock domain-randomized environment does not have:

* **A grasp-relevant yaw convention** -- ``cube.yaw`` is sampled on one
  fundamental domain of the cube's own symmetry group, so it is identifiable
  from the silhouette with no marker at all. See below.
* **``agent.ee_start_yaw``** -- a variation axis for the effector's yaw, which
  the parent draws from ``self.np_random`` inside ``initialize_arm`` and
  therefore does not expose as a controllable latent.
* **``agent.gripper_opening``** -- a variation axis for the gripper driver
  joint, likewise not exposed upstream.
* **A physics-free frame path** (:meth:`render_content`) -- the reason this
  class exists at all. See below.

Why a separate frame path
-------------------------
The encoder dataset is a set of *designed* latent states, not a trajectory:
every frame is an independent draw from the OU sampler, and physics is never
stepped. ``CubeEnv.reset(options={'state': ...})`` can produce such a frame,
but it pays for machinery this use case does not want:

* it **renders twice** -- once in ``CustomMuJoCoEnv.reset`` via
  ``compute_observation``, and again in ``CubeEnv.reset``'s ``state`` branch
  after ``set_state``;
* it runs a full ``initialize_arm`` IK solve whose result ``options['state']``
  then overwrites; and
* it routes ``cube.size``, ``agent.color``, ``floor.color`` and
  ``camera.angle_delta`` through ``modify_mjcf_model``, which calls
  ``mark_dirty()`` and forces ``compile_model_and_data`` -- re-serializing the
  MJCF, re-parsing ~33 MiB of arm meshes, and tearing down and rebuilding the
  ``mujoco.Renderer``.

Measured on the reference machine at 224x224, single cube, one recorded frame:

===================================================  ============  ============
path                                                   ms/frame     frames/s
===================================================  ============  ============
``reset()``, no recompiling axes                            21.4          46.7
``reset()``, recompiling axes live (all-content)           263.1           3.8
:meth:`render_content`                                       5.1         196.3
===================================================  ============  ============

Of that 5.1 ms, 4.6 ms is ``render()`` itself, so the fast path is within ~10%
of the cost of drawing the pixels at all.

Three of the four recompiling axes are reproduced **exactly** -- zero differing
pixels against the recompiled render -- by a post-compilation field write:

* ``agent.color`` -> ``model.mat_rgba``
* ``camera.angle_delta`` -> ``model.cam_quat``
* ``cube.size`` -> ``model.geom_size``

``cube.size`` is the interesting one: the recompile exists so MuJoCo can
recompute mass and inertia, and *nothing in this dataset integrates the
equations of motion*, so those quantities never reach a pixel or a label. The
render depends on ``geom_size`` alone.

``floor.color`` is the exception. It is the ``rgb1``/``rgb2`` of a *builtin*
procedural checker texture, rasterized at compile time; changing it after
compilation means repainting ``model.tex_data`` and re-uploading, which does
not reproduce the compiled texture faithfully. It is therefore pinned here,
and floor color is driven instead by the parent's ``background.floor_rgb``,
which writes ``mat_rgba`` on the selected pool material, costs nothing, and is
already a content latent in its own right (plan SS3.2).

:meth:`render_content` is an addition, not a replacement: the inherited
``reset`` is untouched and remains the path used by evaluation, replay and
every stepped-physics consumer.

The orientation quantity that matters
-------------------------------------
A cube grasped by a parallel-jaw gripper has a symmetry group, and the
identifiability target should be a coordinate on the *quotient*, not on the
raw circle.

Rotating the cube by ``pi/2`` about z maps it exactly onto itself -- the
renderer produces zero differing pixels (measured; see the table below) and
the grasp is the same grasp. So ``cube.yaw`` and ``cube.yaw + pi/2`` are not
two states that an encoder is failing to distinguish; they are **one state
with two names**. The gripper carries the matching symmetry: its jaws are
symmetric under a ``pi`` rotation about the approach axis, so ``effector.yaw``
is likewise only meaningful modulo ``pi``.

Two ways to make yaw identifiable follow from that, and they are not equally
good:

*Break the symmetry* (a decal on a face). This makes all of ``[0, 2pi)``
distinguishable -- but it does so by inventing a distinction the *task* does
not have, and paying for it with a ~48-pixel signal on a 50k-pixel frame. The
encoder is then scored on recovering a quadrant index that no grasp depends
on, and the resulting failure mode dominates the yaw metric while being
irrelevant to control.

*Quotient the symmetry out* (what this class does). Sample ``cube.yaw`` on a
single fundamental domain of the 4-fold group -- an arc of width ``pi/2``,
centred at zero. Every physically distinct orientation has exactly one
representative there, the silhouette determines it strongly (rms 6.1 between
the arc's own extremes), and no marker is needed. ``effector.yaw`` is already
declared on ``[-pi/2, pi/2]``, one fundamental domain of the gripper's 2-fold
symmetry, for the same reason.

The marker machinery is kept and tested but defaults **off**; enabling it is
how one would study the full circle deliberately, not the default posture.

One consequence worth stating before anyone reads a yaw-recovery curve:
**pixel separability is not monotone in angular error.** Measured at 224x224
on an unmarked cube::

    separation    rms
      21 deg     3.79
      43 deg     4.36
      86 deg     1.17

Separability peaks near 45 deg and falls back toward zero at 90 deg, because
90 deg *is* the identification. So a larger raw angular error can look far
*less* wrong than a smaller one, and any metric that assumes monotonicity in
raw angle will misread the result. Scoring on the quotient coordinate -- which
is what the registry hands the metric suite -- is what avoids this.

What is deliberately absent
---------------------------
``num_digits`` defaults to ``0`` here, against :class:`DRCubeEnv`'s ``1``. The
floor digit decals exist to give a linear probe a nuisance target in the DR
benchmark. Nothing in the identifiability study reads them, and against a
scene whose content latents already occupy few pixels they are pure occlusion
risk -- a V6 contribution nobody asked for.

``pixel_transparent_arm`` defaults to ``False``, against OGBench's ``True``.
Upstream fades the arm to alpha 0.1 so it does not occlude the object in a
manipulation benchmark. Here ``effector.pos``, ``effector.yaw`` and
``gripper.opening`` are *content latents the encoder must recover*, so fading
them to near-invisibility would be asking for the recovery of something the
renderer was told to hide.
"""

import mujoco
import numpy as np
from ogbench.manipspace import lie

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.ogbench.dr_cube_env import (
    DRCubeEnv,
    png_asset,
    render_digit_png,
)
from stable_worldmodel.envs.utils import perturb_camera_angle


# Faces a marker can be baked onto, as (axis, sign) into the cube's local
# frame. 'top' is the default: the `front_pixels` camera is oblique and looks
# down on the table, so a top-face marker is visible at every yaw. A side-face
# marker is hidden for roughly half of all yaws, which makes yaw genuinely
# unrecoverable on those frames -- a real injectivity failure, and the V6
# occlusion floor discussed in the plan's risk table.
MARKER_FACES = {
    'top': (2, +1.0),
    'bottom': (2, -1.0),
    'front': (0, +1.0),
    'back': (0, -1.0),
    'left': (1, -1.0),
    'right': (1, +1.0),
}

# How strong the yaw signal actually is, measured at 224x224 on the stock
# `front_pixels` camera with a 0.026 m half-extent cube:
#
#   config                       rms(0, pi/2)   rms(0, pi/4)   px differing
#   no marker                          0.0089         6.1182              0
#   marker top,   scale 0.72           1.5008         7.0115             32
#   marker top,   scale 0.95           2.0936         7.4526             48
#   marker front, scale 0.95           5.9023         6.8466            224
#
# The first row is the whole argument. An unmarked cube is *exactly* 4-fold
# symmetric about z -- yaws pi/2 apart render to zero differing pixels -- but
# yaws pi/4 apart differ strongly (rms 6.1) from the silhouette alone. So yaw
# is already strongly observable **within a quadrant**; only the quadrant
# itself is ambiguous, and a marker buys ~48 pixels out of 50k to resolve it.
MARKER_YAW_SIGNAL_RMS_TOP = 2.09
MARKER_YAW_SIGNAL_RMS_NONE = 0.009

# Thickness of the marker slab, in units of the cube half-extent. It has to
# stand proud of the face to avoid z-fighting with the cube geom, but stay thin
# enough not to change the cube's apparent silhouette.
MARKER_THICKNESS_FRAC = 0.04

# Fraction of the face the marker tile covers. Sized from the table above:
# bigger is a strictly stronger yaw signal, bounded by leaving enough of the
# face showing that `cube.color` stays readable on it.
DEFAULT_MARKER_SCALE = 0.85

# The gripper driver joint's qpos is scaled by this before it is written, so
# `agent.gripper_opening` can be declared on a clean [0, 1] axis. Matches the
# scaling `ManipSpaceEnv` uses when it reports `proprio/gripper_opening`.
GRIPPER_QPOS_SCALE = 0.8

# The seven joints that follow `right_driver_joint` around the 2F-85's closed
# linkage, in the order the coupling table stores them.
GRIPPER_COUPLED_JOINTS = (
    'ur5e/robotiq/right_coupler_joint',
    'ur5e/robotiq/right_spring_link_joint',
    'ur5e/robotiq/right_follower_joint',
    'ur5e/robotiq/left_driver_joint',
    'ur5e/robotiq/left_coupler_joint',
    'ur5e/robotiq/left_spring_link_joint',
    'ur5e/robotiq/left_follower_joint',
)

# Resolution of the coupling table. The manifold is smooth and near-linear, so
# 33 knots put the interpolation error ~1e-6 rad -- four orders of magnitude
# below anything that reaches a pixel.
GRIPPER_TABLE_KNOTS = 33

# Physics steps used to build the table. The sweep is warm-started from the
# previous knot, so each one only has to settle a small increment; the initial
# settle gets the larger budget.
GRIPPER_SETTLE_STEPS = 2000
GRIPPER_KNOT_STEPS = 400

# Built once per distinct gripper linkage, then shared by every environment in
# the process. Keyed on the linkage's own model layout (see
# `_coupling_cache_key`), not on the `MjModel` object, because `cube.size`
# recompiles produce a new model with an identical gripper.
_GRIPPER_COUPLING_CACHE = {}


def _coupling_cache_key(model, driver_id, coupled_ids, actuator_id):
    """Identity of the gripper linkage, as far as the coupling table cares."""
    return (
        int(model.njnt),
        int(driver_id),
        tuple(int(j) for j in coupled_ids),
        int(actuator_id),
        tuple(model.jnt_range[driver_id].tolist()),
        tuple(model.actuator_ctrlrange[actuator_id].tolist()),
    )


def gripper_coupling_table(model, home_qpos, arm_joint_ids):
    """Sample the 2F-85's constraint manifold, once per process.

    Writing ``right_driver_joint`` and calling ``mj_forward`` does **not** move
    the gripper. The 2F-85 is a closed four-bar: the pads hang off
    ``spring_link``/``follower``, which reach the driver only through two
    ``mjEQ_CONNECT`` constraints and one ``mjEQ_JOINT`` constraint. MuJoCo
    enforces equality constraints with forces *during integration*; it does not
    project ``qpos`` onto the constraint manifold in ``mj_forward``. So a
    physics-free write of the driver alone leaves the seven coupled joints
    stale and the rendered gripper frozen -- while ``proprio/gripper_opening``,
    which reads the driver straight back, reports the requested value perfectly.

    That combination -- a label that round-trips and pixels that do not move --
    is exactly the silent failure the plan's SS9 gripper risk describes, and it
    is why this table exists rather than a linear approximation. The linkage is
    *not* linear: ``follower`` fits ``-0.9645 * driver`` with a residual of
    4.6e-3 rad, which is a real geometric error rather than solver noise.

    Rather than re-derive the closure algebra, this settles the linkage with
    MuJoCo's own solver on a throwaway ``MjData`` across a grid of actuator
    commands and records where each joint lands. Interpolating that table is
    then exact to ~1e-6 rad, verified against independent cold settles at
    off-grid points.

    Physics is stepped **here and only here**, at setup, on a scratch
    ``MjData`` -- never on the environment's own state and never per frame.

    Args:
        model: Compiled ``MjModel`` carrying the gripper.
        home_qpos: Arm home configuration, used to hold the arm still while
            the gripper settles.
        arm_joint_ids: Joint ids the home configuration applies to.

    Returns:
        tuple: ``(driver, coupled)`` where ``driver`` is ``(K,)`` of achievable
        driver-joint positions, ascending, and ``coupled`` is ``(K, 7)`` of the
        matching :data:`GRIPPER_COUPLED_JOINTS` positions.
    """
    driver_id = model.joint('ur5e/robotiq/right_driver_joint').id
    coupled_ids = [model.joint(n).id for n in GRIPPER_COUPLED_JOINTS]
    actuator_id = model.actuator('ur5e/robotiq/fingers_actuator').id

    key = _coupling_cache_key(model, driver_id, coupled_ids, actuator_id)
    if key in _GRIPPER_COUPLING_CACHE:
        return _GRIPPER_COUPLING_CACHE[key]

    driver_adr = model.jnt_qposadr[driver_id]
    coupled_adr = [model.jnt_qposadr[j] for j in coupled_ids]

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[arm_joint_ids] = home_qpos

    lo, hi = model.actuator_ctrlrange[actuator_id]
    commands = np.linspace(lo, hi, GRIPPER_TABLE_KNOTS)

    data.ctrl[actuator_id] = commands[0]
    for _ in range(GRIPPER_SETTLE_STEPS):
        mujoco.mj_step(model, data)

    driver = np.empty(GRIPPER_TABLE_KNOTS, dtype=np.float64)
    coupled = np.empty(
        (GRIPPER_TABLE_KNOTS, len(coupled_adr)), dtype=np.float64
    )
    for k, command in enumerate(commands):
        data.ctrl[actuator_id] = command
        for _ in range(GRIPPER_KNOT_STEPS):
            mujoco.mj_step(model, data)
        # Zero the velocities so the next knot settles from rest rather than
        # carrying momentum across the sweep.
        data.qvel[:] = 0.0
        driver[k] = data.qpos[driver_adr]
        coupled[k] = [data.qpos[a] for a in coupled_adr]

    order = np.argsort(driver)
    table = (driver[order].copy(), coupled[order].copy())
    _GRIPPER_COUPLING_CACHE[key] = table
    return table


class LeJEPACubeEnv(DRCubeEnv):
    """Cube environment with a yaw marker, arm latents and a fast frame path.

    Attributes:
        variation_space (swm.spaces.Dict): Everything :class:`DRCubeEnv`
            declares, plus ``agent.ee_start_yaw``, ``agent.gripper_opening``
            and ``marker.{value,scale}``. ``floor.color`` is pinned to its
            default -- see the module docstring.
        _marker_enabled (bool): Whether the cube carries a marker decal.
        _marker_face (str): Which face it sits on. A key of
            :data:`MARKER_FACES`.
    """

    def __init__(
        self,
        marker_enabled: bool = False,
        marker_face: str = 'top',
        marker_scale: float = DEFAULT_MARKER_SCALE,
        pin_floor_color: bool = True,
        num_digits: int = 0,
        pixel_transparent_arm: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the LeJEPA cube environment.

        Args:
            marker_enabled: Whether to bake an asymmetric decal onto each cube.
                **Off by default.** A decal breaks the cube's 4-fold yaw
                symmetry, but that symmetry is a symmetry of the *grasping
                task* too -- see the module docstring's "The orientation
                quantity that matters" section. Restricting ``cube.yaw`` to one
                fundamental domain achieves identifiability without inventing a
                distinction the task does not have.
            marker_face: Which cube face carries the decal, when enabled. One
                of :data:`MARKER_FACES`.
            marker_scale: Marker tile size as a fraction of the cube face.
            pin_floor_color: Whether to pin the inherited ``floor.color`` axis
                to its default. Leaving it live is the one axis that forces an
                MJCF recompile per frame; ``background.floor_rgb`` supersedes
                it. See the module docstring.
            num_digits: Floor digit distractors. **Zero by default**, against
                :class:`DRCubeEnv`'s 1. Those decals exist to give a linear
                probe a nuisance target in the DR benchmark; here they are pure
                occlusion risk over a scene whose content latents are already
                small in the frame, and nothing in the identifiability study
                reads them.
            pixel_transparent_arm: Whether to render the arm at alpha 0.1.
                **False by default**, against OGBench's ``True``. Upstream
                fades the arm so it does not occlude the object in a
                *manipulation benchmark*. Here ``effector.pos``,
                ``effector.yaw`` and ``gripper.opening`` are content latents the
                encoder is required to recover, so rendering them at alpha 0.1
                asks for the recovery of something deliberately made almost
                invisible.
            *args: Forwarded to :class:`DRCubeEnv`.
            **kwargs: Forwarded to :class:`DRCubeEnv`.
        """
        self._marker_enabled = bool(marker_enabled)
        if marker_face not in MARKER_FACES:
            raise ValueError(
                f'marker_face must be one of {sorted(MARKER_FACES)}; '
                f'got {marker_face!r}.'
            )
        self._marker_face = marker_face
        self._marker_scale = float(marker_scale)
        self._pin_floor_color = bool(pin_floor_color)
        kwargs.setdefault('num_digits', num_digits)
        kwargs.setdefault('pixel_transparent_arm', pixel_transparent_arm)

        # Populated by `add_objects` / `post_compilation_objects`.
        self._marker_geom_elems = []
        self._marker_geom_ids = []
        self._marker_material_elems = []
        self._marker_matids = np.zeros(0, dtype=np.int32)

        super().__init__(*args, **kwargs)

        self.env_name = 'LeJEPACube'
        # `agent.gripper_opening`'s bounds are a measured property of the
        # linkage, so the model has to exist before the axis can be declared.
        # `CustomMuJoCoEnv` compiles lazily on first `reset`, which is too
        # late: the variation space is sampled at the top of that same reset,
        # before compilation, so the first episode would sample against
        # provisional bounds.
        self._ensure_compiled()
        self._extend_lejepa_variation_space()

    def _ensure_compiled(self):
        """Build and compile the model now, if it has not been already.

        Idempotent, and leaves the environment in exactly the state the first
        ``reset`` would find it in -- ``reset`` re-runs ``modify_mjcf_model``
        and only recompiles if that marks the model dirty.
        """
        if self._mjcf_model is None:
            self._mjcf_model = self.build_mjcf_model()
        if self._model is None:
            self.compile_model_and_data()
            self._never_compiled = False

    # ------------------------------------------------------------------
    # variation space
    # ------------------------------------------------------------------

    def _extend_lejepa_variation_space(self):
        """Add the arm and marker axes, and pin ``floor.color``."""
        spaces_dict = dict(self.variation_space.spaces)

        agent = dict(spaces_dict['agent'].spaces)
        agent['ee_start_yaw'] = swm_spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(1,),
            dtype=np.float64,
            init_value=np.zeros(1),
        )
        # Not [0, 1]: the actuator cannot drive the linkage to either joint
        # limit -- the achievable driver range is about [0.003, 0.978] in
        # these units. Declaring the axis over the achievable range keeps every
        # sampled value *on* the constraint manifold; declaring it over [0, 1]
        # would put the endpoints off it, where the coupling table can only
        # extrapolate. The narrowing is a declared support truncation, recorded
        # in the registry rather than discovered later as an anomaly.
        lo, hi = self.gripper_opening_bounds
        agent['gripper_opening'] = swm_spaces.Box(
            low=lo,
            high=hi,
            shape=(1,),
            dtype=np.float64,
            init_value=np.array([lo]),
        )
        spaces_dict['agent'] = swm_spaces.Dict(agent)

        if self._pin_floor_color:
            # Pinning by construction rather than by convention: an axis whose
            # low equals its high can be sampled, recorded and replayed like
            # any other, and can never mark the model dirty.
            stock = np.asarray(
                spaces_dict['floor']['color'].init_value, dtype=np.float64
            )
            floor = dict(spaces_dict['floor'].spaces)
            floor['color'] = swm_spaces.Box(
                low=stock,
                high=stock,
                shape=stock.shape,
                dtype=np.float64,
                init_value=stock,
            )
            spaces_dict['floor'] = swm_spaces.Dict(floor)

        if self._marker_enabled:
            spaces_dict['marker'] = swm_spaces.Dict(
                {
                    'value': swm_spaces.MultiDiscrete(
                        np.full(self._num_cubes, 10, dtype=np.int64),
                        init_value=np.arange(
                            self._num_cubes, dtype=np.int64
                        )
                        % 10,
                    ),
                }
            )

        self.variation_space = swm_spaces.Dict(spaces_dict)

    # ------------------------------------------------------------------
    # gripper linkage
    # ------------------------------------------------------------------

    @property
    def _coupling_table(self):
        """The 2F-85 coupling manifold, built on first use.

        See :func:`gripper_coupling_table` for why writing the driver joint
        alone is not enough.
        """
        return gripper_coupling_table(
            self._model, self._home_qpos, self._arm_joint_ids
        )

    @property
    def gripper_opening_bounds(self):
        """``(lo, hi)`` opening values the linkage can actually reach.

        In the same normalized units as ``proprio/gripper_opening`` -- driver
        joint position divided by :data:`GRIPPER_QPOS_SCALE`. Narrower than
        ``[0, 1]``; see :meth:`_extend_lejepa_variation_space`.
        """
        driver, _ = self._coupling_table
        return (
            float(driver[0] / GRIPPER_QPOS_SCALE),
            float(driver[-1] / GRIPPER_QPOS_SCALE),
        )

    def _write_gripper_opening(self, qpos, opening):
        """Write a gripper opening and every joint coupled to it.

        Args:
            qpos: ``(nq,)`` configuration to write into, modified in place.
            opening: Requested opening, in ``proprio/gripper_opening`` units.
                Clipped into :attr:`gripper_opening_bounds`.

        Returns:
            float: The opening actually written, after clipping.
        """
        driver_table, coupled_table = self._coupling_table
        lo, hi = self.gripper_opening_bounds
        opening = float(np.clip(opening, lo, hi))

        target = opening * GRIPPER_QPOS_SCALE
        # `ManipSpaceEnv.compute_ob_info` indexes `qpos` with the driver's
        # *joint id*, which is only the right address while it coincides with
        # the joint's `qposadr`. It does here, and `post_compilation_objects`
        # asserts it still does -- writing the correct address while the label
        # is read from a different one would put the two silently out of step.
        qpos[self._model.jnt_qposadr[self._gripper_opening_joint_id]] = target
        for name, column in zip(
            GRIPPER_COUPLED_JOINTS, coupled_table.T, strict=True
        ):
            adr = self._model.jnt_qposadr[self._model.joint(name).id]
            qpos[adr] = np.interp(target, driver_table, column)

        return opening

    # ------------------------------------------------------------------
    # scene construction (build time, once)
    # ------------------------------------------------------------------

    def add_objects(self, arena_mjcf):
        """Add the DR scene, then bake the marker decals onto the cubes.

        The marker is a child geom of ``object_{i}``, not a mocap body, so it
        follows the cube through every pose without any per-frame bookkeeping.
        It is collision-free (``contype=0, conaffinity=0``), so adding it
        cannot change the physics that the predictor dataset records.

        Args:
            arena_mjcf (mjcf.RootElement): Arena being built.
        """
        super().add_objects(arena_mjcf)

        if not self._marker_enabled:
            return

        # The digit textures are baked by `DRCubeEnv._bake_digit_assets` as
        # `type='2d'`, which is what a flat decal needs -- but those materials
        # are shared with the floor distractors, and `_apply_visual_variations`
        # rewrites `geom_matid` on those every frame. A separate pool keeps the
        # marker's material independent of the distractors'.
        for digit in range(10):
            arena_mjcf.asset.add(
                'texture',
                name=f'marker_tex_{digit}',
                type='2d',
                file=png_asset(
                    render_digit_png(digit), f'marker_tex_{digit}'
                ),
            )
            self._marker_material_elems.append(
                arena_mjcf.asset.add(
                    'material',
                    name=f'marker_mat_{digit}',
                    texture=f'marker_tex_{digit}',
                    texuniform=False,
                    texrepeat=(1.0, 1.0),
                    specular=0.0,
                    shininess=0.0,
                    reflectance=0.0,
                )
            )

        axis, sign = MARKER_FACES[self._marker_face]
        for i in range(self._num_cubes):
            body = arena_mjcf.find('body', f'object_{i}')
            # Sized against the stock half-extent; `_apply_marker_geometry`
            # rescales it to the episode's actual `cube.size` post-compilation.
            half = 0.02
            self._marker_geom_elems.append(
                body.add(
                    'geom',
                    name=f'marker_geom_{i}',
                    type='box',
                    size=self._marker_size(half, axis),
                    pos=self._marker_pos(half, axis, sign),
                    material='marker_mat_0',
                    contype=0,
                    conaffinity=0,
                    group=1,
                )
            )

    def _marker_size(self, half: float, axis: int) -> tuple:
        """Half-extents of the marker slab on a cube of half-extent ``half``."""
        tile = half * self._marker_scale
        size = [tile, tile, tile]
        size[axis] = half * MARKER_THICKNESS_FRAC
        return tuple(size)

    @staticmethod
    def _marker_pos(half: float, axis: int, sign: float) -> tuple:
        """Center of the marker slab, sitting just proud of the chosen face."""
        pos = [0.0, 0.0, 0.0]
        pos[axis] = sign * half * (1.0 + MARKER_THICKNESS_FRAC)
        return tuple(pos)

    def post_compilation_objects(self):
        """Cache the marker geom and material ids alongside the DR ones."""
        super().post_compilation_objects()

        if not self._marker_enabled:
            return

        self._marker_geom_ids = [
            self._model.geom(elem.full_identifier).id
            for elem in self._marker_geom_elems
        ]
        self._marker_matids = np.array(
            [
                self._model.material(elem.full_identifier).id
                for elem in self._marker_material_elems
            ],
            dtype=np.int32,
        )

    def post_compilation(self):
        """Cache ids, then check the one indexing assumption upstream makes."""
        super().post_compilation()

        driver_id = self._gripper_opening_joint_id
        if self._model.jnt_qposadr[driver_id] != driver_id:
            raise RuntimeError(
                'ManipSpaceEnv reads the gripper opening as '
                f'qpos[{driver_id}] (the joint id), but that joint now lives '
                f'at qposadr {self._model.jnt_qposadr[driver_id]}. Writing and '
                'reading the opening would address different degrees of '
                'freedom.'
            )

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------

    def initialize_arm(self):
        """Place the effector, taking yaw from the variation space.

        Identical to :meth:`CubeEnv.initialize_arm` except that the yaw comes
        from ``agent.ee_start_yaw`` instead of ``self.np_random.uniform``.
        Without this, ``effector.yaw`` cannot be a controlled latent: it would
        be redrawn from the environment's own RNG on every reset, uncorrelated
        with the value the sampler asked for.
        """
        eff_pos = self.variation_space['agent']['ee_start_position'].value
        yaw = float(
            np.asarray(
                self.variation_space['agent']['ee_start_yaw'].value
            ).reshape(-1)[0]
        )
        eff_ori = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation

        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        self._write_gripper_opening(
            self._data.qpos,
            float(
                np.asarray(
                    self.variation_space['agent']['gripper_opening'].value
                ).reshape(-1)[0]
            ),
        )
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        """Run the DR episode setup, then apply the marker appearance."""
        super().initialize_episode()
        self._apply_marker_variations()

    # ------------------------------------------------------------------
    # post-compilation appearance writes
    # ------------------------------------------------------------------

    def _apply_marker_variations(self):
        """Point each marker geom at its digit material and rescale it.

        Both writes target the compiled ``MjModel``, so neither marks the
        model dirty.
        """
        if not self._marker_enabled or not self._marker_geom_ids:
            return

        axis, sign = MARKER_FACES[self._marker_face]
        values = np.asarray(
            self.variation_space['marker']['value'].value, dtype=np.int64
        ).reshape(-1)

        for i, gid in enumerate(self._marker_geom_ids):
            self._model.geom_matid[gid] = self._marker_matids[values[i]]
            half = self._half(i)
            self._model.geom_size[gid] = self._marker_size(half, axis)
            self._model.geom_pos[gid] = self._marker_pos(half, axis, sign)

    def _apply_cube_size(self):
        """Write ``cube.size`` straight onto the compiled geoms.

        The parent reaches this axis through ``modify_mjcf_model``, which marks
        the model dirty and forces a full recompile so MuJoCo can recompute
        mass and inertia from the new extents. This method is the
        physics-free counterpart: it writes ``geom_size`` and leaves the
        inertial properties stale, which is sound **only** because
        :meth:`render_content` never integrates the equations of motion.

        Verified to render bit-identically to the recompiled model (zero
        differing pixels), once the goal-marker mocap -- which the recompile
        path repositions as a side effect and which
        :meth:`render_content` parks out of frame -- is accounted for.

        Do not call this on any path that steps physics.
        """
        sizes = np.asarray(
            self.variation_space['cube']['size'].value, dtype=np.float64
        ).reshape(-1)
        for i in range(self._num_cubes):
            extents = (sizes[i], sizes[i], sizes[i])
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom_size[gid] = extents
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom_size[gid] = extents

    def _apply_agent_color(self):
        """Write ``agent.color`` onto the compiled materials.

        Renders bit-identically to the recompiled model.
        """
        rgb = np.asarray(
            self.variation_space['agent']['color'].value, dtype=np.float64
        ).reshape(-1)[:3]
        for name in ('ur5e/robotiq/black', 'ur5e/robotiq/pad_gray'):
            self._model.mat_rgba[self._model.material(name).id, :3] = rgb

    def _apply_cube_color(self):
        """Write ``cube.color`` onto the compiled cube geoms.

        The parent does this inside its ``initialize_episode`` branches, which
        :meth:`render_content` does not run.
        """
        colors = np.asarray(
            self.variation_space['cube']['color'].value, dtype=np.float64
        ).reshape(self._num_cubes, -1)
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom_rgba[gid, :3] = colors[i][:3]
                self._model.geom_rgba[gid, 3] = 1.0

    def _apply_camera_angle(self):
        """Write ``camera.angle_delta`` onto the compiled camera frame.

        The parent perturbs the camera's ``xyaxes`` in the MJCF and
        recompiles. A fixed camera's world pose is derived by ``mj_forward``
        from ``model.cam_pos``/``model.cam_quat``, so writing the rotation
        directly reaches the renderer by the same route -- and renders
        bit-identically to the recompiled model.
        """
        deltas = np.asarray(
            self.variation_space['camera']['angle_delta'].value,
            dtype=np.float64,
        ).reshape(-1, 2)

        names = (
            ['front_pixels', 'side_pixels']
            if self._multiview
            else ['front_pixels']
        )
        for i, cam_name in enumerate(names):
            xyaxes = perturb_camera_angle(
                self.cameras[cam_name]['xyaxes'], deltas[i]
            )
            self._model.cam_quat[self._model.camera(cam_name).id] = (
                _xyaxes_to_quat(xyaxes)
            )

    def _hide_goal_markers(self):
        """Park every goal mocap out of frame.

        The content vector describes the *scene*, and a visible goal marker is
        a latent nothing in the registry accounts for. ``CubeEnv.set_new_target``
        normally places these; :meth:`render_content` does not run it, but the
        mocaps keep whatever pose the last ``reset`` left behind, so they are
        parked explicitly rather than left to chance.
        """
        for i in range(self._num_cubes):
            self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (
                0.0,
                0.0,
                -0.3,
            )
            self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
                lie.SO3.identity().wxyz.tolist()
            )
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom_rgba[gid, 3] = 0.0

    # ------------------------------------------------------------------
    # the physics-free frame path
    # ------------------------------------------------------------------

    def render_content(self, state, variation_values, camera='front_pixels'):
        """Render one designed frame without stepping, resetting or recompiling.

        This is the encoder dataset's inner loop. It sets every appearance axis
        by writing the compiled model, sets the physical state by writing
        ``qpos``/``qvel``, runs forward kinematics once, and renders once.

        The contract it deliberately does **not** honour, relative to
        :meth:`reset`:

        * no ``mj_resetData``, no ``initialize_episode``, no
          ``initialize_arm`` IK solve -- ``state`` already carries the arm
          pose, and re-deriving it would be thrown away;
        * no task sampling and no goal markers (see
          :meth:`_hide_goal_markers`);
        * no recompilation, so mass and inertia do not track ``cube.size``.

        Every one of those is safe *because physics is never stepped here* and
        unsafe anywhere else. The environment must have been reset at least
        once before this is called, so that a compiled model exists.

        Args:
            state: ``(nq + nv,)`` concatenation of ``qpos`` and ``qvel``,
                as produced by :meth:`set_content_state`.
            variation_values: Mapping from dotted variation name to value,
                covering every axis the caller wants set this frame. Values
                are validated against the variation space, exactly as
                ``reset(options={'variation_values': ...})`` would.
            camera: Which camera to render.

        Returns:
            ndarray: ``(H, W, 3)`` uint8 frame.

        Raises:
            RuntimeError: If called before the model has been compiled.
        """
        if self._model is None or self._data is None:
            raise RuntimeError(
                'render_content requires a compiled model; call reset() once '
                'before using the fast frame path.'
            )

        # Deliberately not `swm_spaces.reset_variation_space`: that re-seeds
        # every sub-space from OS entropy on the way in, which costs ~0.24 ms
        # of a ~6 ms frame and buys nothing here, because this path samples
        # nothing -- every value arrives explicitly. The three steps that do
        # matter are kept, validation included.
        self.variation_space.reset()
        self.variation_space.set_value(variation_values)
        if not self.variation_space.check(debug=True):
            raise ValueError(
                'variation_values fall outside the variation space.'
            )

        # Appearance: everything post-compilation, in the order the parent
        # would have applied it.
        self._apply_cube_size()
        self._apply_cube_color()
        self._apply_agent_color()
        self._apply_camera_angle()
        self._apply_visual_variations()
        self._apply_marker_variations()
        self._hide_goal_markers()

        # Physical state.
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        nq, nv = self._model.nq, self._model.nv
        if state.shape[0] != nq + nv:
            raise ValueError(
                f'state must have shape ({nq + nv},); got {state.shape}.'
            )
        self._data.qpos[:] = state[:nq]
        self._data.qvel[:] = state[nq:]

        # `compute_ob_info` reports `prev_qpos`/`prev_qvel`, which `pre_step`
        # would normally have populated. Frames here are independent draws,
        # not consecutive states, so "the previous state" is this state --
        # anything else would be a stale value from whenever `reset` last ran.
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()

        mujoco.mj_forward(self._model, self._data)

        return self.render(camera=camera)

    def content_info(self):
        """Ground-truth info for the frame currently in ``_data``.

        The same ``privileged/*`` and ``proprio/*`` columns
        :meth:`get_reset_info` would return, without re-rendering. This is what
        the recovery metrics' ``z`` is read from at eval time, so it must come
        from the simulator rather than from the sampler's request -- the two
        differ wherever a value was clipped, IK failed to converge, or a
        coupled joint did not track its driver.

        Deliberately does *not* call ``pre_step``/``post_step``:
        ``post_step`` re-derives goal-marker visibility from ``_target_block``
        and would undo :meth:`_hide_goal_markers`, putting a marker into the
        next frame that no latent accounts for. ``compute_ob_info`` reads the
        simulator directly and needs neither.

        Returns:
            dict: Observation info for the current simulator state.
        """
        return self.compute_ob_info()

    # ------------------------------------------------------------------
    # content vector -> simulator state
    # ------------------------------------------------------------------

    def set_content_state(self, physical):
        """Assemble a ``qpos || qvel`` vector from physical latent values.

        Physics-free: the arm pose is solved by IK and written directly, the
        cube pose is written as a free-joint qpos, and ``qvel`` is zero
        throughout. Nothing is stepped and the simulator state is not mutated
        -- the array is returned for :meth:`render_content` to apply.

        Args:
            physical: Mapping with any of the keys ``cube.pos_xy`` ``(n, 2)``,
                ``cube.pos_z`` ``(n,)``, ``cube.yaw`` ``(n,)``,
                ``cube.quat`` ``(n, 4)`` (overrides ``cube.yaw``),
                ``effector.pos`` ``(3,)``, ``effector.yaw`` scalar,
                ``gripper.opening`` scalar in ``[0, 1]``. Anything omitted
                keeps the value currently in ``_data``.

        Returns:
            ndarray: ``(nq + nv,)`` state vector, ``qvel`` all zero.

        Raises:
            RuntimeError: If called before the model has been compiled.
        """
        if self._model is None or self._data is None:
            raise RuntimeError(
                'set_content_state requires a compiled model; call reset() '
                'once first.'
            )

        qpos = self._data.qpos.copy()
        qvel = np.zeros(self._model.nv, dtype=np.float64)

        # --- arm ---------------------------------------------------------
        if 'effector.pos' in physical or 'effector.yaw' in physical:
            eff_pos = np.asarray(
                physical.get(
                    'effector.pos',
                    self._data.site_xpos[self._pinch_site_id],
                ),
                dtype=np.float64,
            ).reshape(3)
            yaw = float(np.asarray(physical.get('effector.yaw', 0.0)).reshape(-1)[0])
            eff_ori = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation
            T_wa = (
                lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
                @ self._T_pa
            )
            qpos[self._arm_joint_ids] = self._ik.solve(
                pos=T_wa.translation(),
                quat=T_wa.rotation().wxyz,
                curr_qpos=self._home_qpos,
            )

        if 'gripper.opening' in physical:
            # Writes the driver *and* the seven joints coupled to it. Writing
            # the driver alone leaves the pads where they were -- see
            # `gripper_coupling_table`.
            self._write_gripper_opening(
                qpos,
                float(np.asarray(physical['gripper.opening']).reshape(-1)[0]),
            )

        # --- cubes -------------------------------------------------------
        for i in range(self._num_cubes):
            adr = self._model.joint(f'object_joint_{i}').qposadr[0]

            if 'cube.pos_xy' in physical:
                xy = np.asarray(
                    physical['cube.pos_xy'], dtype=np.float64
                ).reshape(self._num_cubes, 2)[i]
                qpos[adr : adr + 2] = xy
            if 'cube.pos_z' in physical:
                z = np.asarray(
                    physical['cube.pos_z'], dtype=np.float64
                ).reshape(self._num_cubes)[i]
                qpos[adr + 2] = z

            if 'cube.quat' in physical:
                quat = np.asarray(
                    physical['cube.quat'], dtype=np.float64
                ).reshape(self._num_cubes, 4)[i]
                qpos[adr + 3 : adr + 7] = quat / np.linalg.norm(quat)
            elif 'cube.yaw' in physical:
                yaw = np.asarray(
                    physical['cube.yaw'], dtype=np.float64
                ).reshape(self._num_cubes)[i]
                qpos[adr + 3 : adr + 7] = lie.SO3.from_z_radians(
                    float(yaw)
                ).wxyz

        return np.concatenate([qpos, qvel])

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    def add_object_info(self, ob_info):
        """Add the DR ground truth, plus the marker value and effector yaw.

        Adds on top of everything :class:`DRCubeEnv` adds:
            - ``privileged/marker_{i}_value``: digit on cube ``i``'s marker.
            - ``privileged/ee_yaw``: the effector's yaw, read back from the
              simulator rather than from the requested variation value.

        ``privileged/ee_yaw`` duplicates ``proprio/effector_yaw`` by design:
        the recovery target vector is assembled from ``privileged/*`` alone,
        and a latent that is only reachable through a different prefix is a
        latent that will eventually be left out of it.
        """
        super().add_object_info(ob_info)

        if self._marker_enabled:
            values = np.asarray(
                self.variation_space['marker']['value'].value, dtype=np.int64
            ).reshape(-1)
            for i in range(self._num_cubes):
                ob_info[f'privileged/marker_{i}_value'] = np.array(
                    [values[i]], dtype=np.int64
                )

        ob_info['privileged/ee_yaw'] = np.asarray(
            ob_info['proprio/effector_yaw'], dtype=np.float64
        ).reshape(-1).copy()


def _xyaxes_to_quat(xyaxes) -> np.ndarray:
    """Convert a MuJoCo ``xyaxes`` camera spec to a body quaternion.

    ``xyaxes`` gives the camera's x and y axes in world coordinates; MuJoCo
    orthonormalizes them at compile time and stores the result as ``cam_quat``.
    Reproducing that here is what lets ``camera.angle_delta`` be applied
    post-compilation.

    Args:
        xyaxes: Length-6 sequence, ``(x_x, x_y, x_z, y_x, y_y, y_z)``.

    Returns:
        ndarray: ``(4,)`` quaternion in MuJoCo's ``wxyz`` order.
    """
    x = np.asarray(xyaxes[:3], dtype=np.float64)
    y = np.asarray(xyaxes[3:], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)

    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.stack([x, y, z], axis=1).reshape(-1))
    return quat


# Registered here rather than in `envs/__init__.py` so that no existing file
# has to be edited; importing this module is what makes the id resolvable.
from gymnasium.envs import registration  # noqa: E402


if 'swm/LeJEPACube-v0' not in registration.registry:
    registration.register(
        id='swm/LeJEPACube-v0',
        entry_point=(
            'stable_worldmodel.envs.ogbench.lejepa_cube_env:LeJEPACubeEnv'
        ),
    )


__all__ = ['LeJEPACubeEnv', 'MARKER_FACES']

"""Domain-randomized OGBench cube environment.

:class:`DRCubeEnv` extends :class:`~stable_worldmodel.envs.ogbench.cube_env.CubeEnv`
with the visual-nuisance axes that the stock environment does not vary, and
with the geometry fixes that the inherited ``cube.size`` axis needs in order to
stay physically consistent.

What it adds on top of ``CubeEnv``:

* **Full light randomization** -- per-light position, direction, diffuse,
  ambient and specular, plus the global headlight. All applied to the compiled
  ``MjModel`` at episode start, so **no recompilation** is triggered.
* **Background / floor / wall materials** -- a pool of procedural (and,
  optionally, image-backed) materials baked once at scene-construction time;
  each episode picks an index and swaps ``geom_matid``. Optional opaque
  backdrop panels give the camera a randomizable background instead of the
  static skybox.
* **Floor digit distractors** -- ``num_digits`` non-colliding mocap decals, each
  showing one of ten pre-baked digit textures at a randomized floor position and
  yaw. Their ground-truth value and position are exported under
  ``privileged/digit_{i}_*`` so a linear probe can ask whether the learned
  representation retained them.
* **Size-consistent geometry** -- resting heights, stacking offsets and task
  waypoints scale with the per-cube half-extent instead of assuming ``0.02``.

Recompilation cost
------------------
Every axis this class *adds* is applied post-compilation and therefore costs
nothing beyond a few array writes. The axes it *inherits* are not all free:
``cube.size``, ``floor.color``, ``agent.color`` and ``camera.angle_delta`` go
through :meth:`CubeEnv.modify_mjcf_model`, which calls ``mark_dirty()`` and
forces a full MJCF recompile on every reset. That is unavoidable (and correct)
for ``cube.size``, since mass and inertia must be recomputed.

``light.intensity`` is the one inherited axis this class deliberately disables:
it is pinned to its default so the parent never marks the model dirty for it,
and per-light RGB control is exposed through ``light.diffuse`` instead.

Example:
    Collect domain-randomized quadruple-cube demonstrations::

        import stable_worldmodel as swm

        world = swm.World(
            'swm/OGBCubeDR-v0',
            num_envs=8,
            image_shape=(224, 224),
            env_type='quadruple',
            mode='data_collection',
            terminate_at_goal=False,
        )
        world.set_policy(swm.envs.ogbench.ExpertPolicy())
        world.collect('cube_quadruple_dr.lance', episodes=100,
                      options={'variation': ['all']})
"""

import functools
import io

import mujoco
import numpy as np
from dm_control import mjcf
from ogbench.manipspace import lie

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


# Lights declared by `descriptions/floor_wall.xml`. The arm's own spotlight is
# removed by `ManipSpaceEnv.build_mjcf_model` before `add_objects` runs, so
# these two are exactly the lights present after compilation.
LIGHT_NAMES = ('global', 'spotlight')

# `light.intensity` is inherited from CubeEnv but disabled here: the parent
# applies it by editing the MJCF and recompiling, which this class replaces
# with the recompile-free `light.diffuse` axis. Pinning it to the stock value
# keeps `CubeEnv.modify_mjcf_model` working without ever marking the model
# dirty. See the module docstring.
PINNED_LIGHT_INTENSITY = 0.7

DEFAULT_VARIATIONS = (
    'cube.start_position',
    'cube.start_yaw',
    'cube.goal_position',
    'cube.goal_yaw',
)

# Digit decals sit this far above the floor plane to avoid z-fighting.
DIGIT_Z = 0.0015
DIGIT_HALF_EXTENT = 0.03


@functools.lru_cache(maxsize=16)
def render_digit_png(digit: int, size: int = 128) -> bytes:
    """Render a single digit to an in-memory PNG.

    Deterministic: the same digit always produces the same bytes, so a dataset
    that stores only the digit *index* can be replayed exactly.

    Args:
        digit: Digit to draw, ``0``-``9``.
        size: Side length in pixels of the square output image.

    Returns:
        PNG-encoded image bytes.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype('DejaVuSans.ttf', int(size * 0.8))
        canvas = size
    except OSError:
        # Bitmap fallback: draw small, then upscale to the requested size.
        font = ImageFont.load_default()
        canvas = 32

    img = Image.new('RGB', (canvas, canvas), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    text = str(int(digit))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((-left, -top), text, fill=(25, 25, 25), font=font)

    # A decal is only ~6 cm wide on screen, so crop to the glyph's ink and
    # rescale it to fill the tile -- the font's own side bearings would
    # otherwise leave the digit too small to read (or to probe for).
    glyph = img.crop((0, 0, max(1, right - left), max(1, bottom - top)))
    side = max(glyph.size)
    pad = max(1, int(side * 0.12))
    tile = Image.new('RGB', (side + 2 * pad, side + 2 * pad), (250, 250, 250))
    tile.paste(
        glyph,
        (
            pad + (side - glyph.size[0]) // 2,
            pad + (side - glyph.size[1]) // 2,
        ),
    )
    img = tile.resize((size, size), Image.LANCZOS)
    # MuJoCo maps the texture's u axis to the decal's local x, which the
    # `front_pixels` camera sees as "up". Pre-rotate so a decal at yaw=0 reads
    # upright from that camera; `digit.yaw` randomizes it from there.
    img = img.transpose(Image.ROTATE_90)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def png_asset(payload: bytes, name: str) -> mjcf.Asset:
    """Wrap PNG bytes as an MJCF asset that survives OGBench's name mangling.

    ``ogbench.manipspace.mjcf_utils`` strips a trailing ``-<40 char sha1>``
    from every asset filename (``name[:-41]``) in both ``to_string`` and
    ``get_assets``. dm_control names a prefix-less in-memory asset with the
    bare 40-char hash, so that slice leaves an **empty** stem: every such
    asset collapses to the single filename ``.png`` and they all silently
    resolve to the same image. Passing a ``prefix`` restores the
    ``<prefix>-<hash>`` layout the stripping expects, keeping each asset
    distinct.

    Args:
        payload: PNG-encoded image bytes.
        name: Unique prefix, recovered verbatim as the mangled filename.

    Returns:
        An ``mjcf.Asset`` safe to assign to a ``<texture file=...>``.
    """
    return mjcf.Asset(payload, '.png', prefix=name)


def _row_get(row: dict, key: str):
    """Fetch ``key`` from a dataset row, tolerating ``/`` vs ``_`` naming.

    Info keys use ``privileged/block_0_pos`` while some dataset backends
    flatten the separator to ``privileged_block_0_pos``. Try both rather than
    hard-coding one convention.

    Raises:
        KeyError: If no spelling of ``key`` is present in the row.
    """
    for candidate in (key, key.replace('/', '_'), key.replace('_', '/')):
        if candidate in row:
            return row[candidate]
    raise KeyError(key)


class DRCubeEnv(CubeEnv):
    """Cube environment with extended visual domain randomization.

    Adds ``light.*`` (position/direction/diffuse/ambient/specular/headlight),
    ``background.{floor_material,wall_material}`` and
    ``digit.{value,position,yaw}`` to the inherited variation space, and makes
    the environment's geometry consistent under ``cube.size`` randomization.

    Attributes:
        variation_space (swm.spaces.Dict): Inherited cube/agent/floor/camera
            axes plus the light, background and digit axes documented above.
        _num_digits (int): Number of floor digit decals.
        _bg_floor_material_elems (list): MJCF material handles of the floor
            pool (2d textures); index 0 is the stock checkerboard, index 1 a
            plain dark material.
        _bg_wall_material_elems (list): The cube-textured counterparts used by
            the backdrop panels, indexed identically.
    """

    def __init__(
        self,
        num_digits: int = 5,
        num_bg_materials: int = 8,
        bg_image_dir=None,
        digit_bounds=((0.22, -0.36), (0.62, 0.36)),
        add_backdrop: bool = True,
        size_aware_geometry: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the domain-randomized cube environment.

        Args:
            num_digits: Number of floor digit decals. ``0`` disables them.
            num_bg_materials: Size of the background material pool. Index 0 is
                always the stock checkerboard and index 1 a plain dark
                material; the rest are procedurally generated from a fixed
                seed, so the pool is identical across processes.
            bg_image_dir: Optional directory of ``.png`` files to bake into the
                pool as image textures, replacing procedural entries from
                index 2 onward.
            digit_bounds: ``((x_lo, y_lo), (x_hi, y_hi))`` sampling box for
                digit positions on the floor.
            add_backdrop: Whether to add opaque, collision-free backdrop panels
                behind and beside the workspace. Disable to keep the stock
                skybox background.
            size_aware_geometry: Whether resting heights, stack offsets and
                task waypoints scale with the per-cube half-extent. Disable to
                reproduce ``CubeEnv`` behavior exactly.
            *args: Forwarded to :class:`CubeEnv`.
            **kwargs: Forwarded to :class:`CubeEnv`.
        """
        self._num_digits = int(num_digits)
        self._num_bg_materials = max(2, int(num_bg_materials))
        self._bg_image_dir = bg_image_dir
        self._digit_bounds = np.asarray(digit_bounds, dtype=np.float64)
        self._add_backdrop = bool(add_backdrop)
        self._size_aware_geometry = bool(size_aware_geometry)

        super().__init__(*args, **kwargs)

        self.env_name = 'CubeDR'
        self._extend_variation_space()

    # ------------------------------------------------------------------
    # variation space
    # ------------------------------------------------------------------

    def _extend_variation_space(self):
        """Replace ``light`` and add ``background`` / ``digit`` sub-spaces."""
        n_lights = len(LIGHT_NAMES)

        light_space = swm_spaces.Dict(
            {
                # Pinned: superseded by `light.diffuse`, see module docstring.
                'intensity': swm_spaces.Box(
                    low=PINNED_LIGHT_INTENSITY,
                    high=PINNED_LIGHT_INTENSITY,
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([PINNED_LIGHT_INTENSITY]),
                ),
                'position': swm_spaces.Box(
                    low=np.array([[-0.6, -0.6, 1.2], [0.05, -0.35, 0.30]]),
                    high=np.array([[0.6, 0.6, 2.6], [0.65, 0.35, 0.85]]),
                    shape=(n_lights, 3),
                    dtype=np.float64,
                    init_value=np.array([[0.0, 0.0, 2.0], [0.25, 0.0, 0.5]]),
                ),
                'direction': swm_spaces.Box(
                    low=np.tile([-0.6, -0.6, -1.0], (n_lights, 1)),
                    high=np.tile([0.6, 0.6, -0.2], (n_lights, 1)),
                    shape=(n_lights, 3),
                    dtype=np.float64,
                    init_value=np.tile([0.0, 0.0, -1.0], (n_lights, 1)),
                ),
                'diffuse': swm_spaces.Box(
                    low=0.15,
                    high=1.0,
                    shape=(n_lights, 3),
                    dtype=np.float64,
                    init_value=np.array([[0.7] * 3, [0.3] * 3]),
                ),
                'ambient': swm_spaces.Box(
                    low=0.0,
                    high=0.35,
                    shape=(n_lights, 3),
                    dtype=np.float64,
                    init_value=np.zeros((n_lights, 3)),
                ),
                'specular': swm_spaces.Box(
                    low=0.0,
                    high=0.6,
                    shape=(n_lights, 3),
                    dtype=np.float64,
                    init_value=np.full((n_lights, 3), 0.3),
                ),
                'headlight_diffuse': swm_spaces.Box(
                    low=0.1,
                    high=0.9,
                    shape=(3,),
                    dtype=np.float64,
                    init_value=np.full((3,), 0.6),
                ),
            }
        )

        background_space = swm_spaces.Dict(
            {
                'floor_material': swm_spaces.Discrete(
                    self._num_bg_materials, init_value=0
                ),
                'wall_material': swm_spaces.Discrete(
                    self._num_bg_materials, init_value=1
                ),
            }
        )

        spaces_dict = dict(self.variation_space.spaces)
        spaces_dict['light'] = light_space
        spaces_dict['background'] = background_space

        if self._num_digits > 0:
            lo, hi = self._digit_bounds
            spaces_dict['digit'] = swm_spaces.Dict(
                {
                    'value': swm_spaces.MultiDiscrete(
                        np.full(self._num_digits, 10, dtype=np.int64),
                        init_value=np.arange(self._num_digits, dtype=np.int64)
                        % 10,
                    ),
                    'position': swm_spaces.Box(
                        low=np.tile(lo, (self._num_digits, 1)),
                        high=np.tile(hi, (self._num_digits, 1)),
                        shape=(self._num_digits, 2),
                        dtype=np.float64,
                        init_value=self._default_digit_positions(),
                    ),
                    'yaw': swm_spaces.Box(
                        low=0.0,
                        high=2 * np.pi,
                        shape=(self._num_digits,),
                        dtype=np.float64,
                        init_value=np.zeros(self._num_digits),
                    ),
                }
            )

        self.variation_space = swm_spaces.Dict(spaces_dict)

    def _default_digit_positions(self) -> np.ndarray:
        """Deterministic, evenly spread default positions inside the bounds."""
        lo, hi = self._digit_bounds
        frac = (np.arange(self._num_digits) + 0.5) / self._num_digits
        # Zig-zag across y so the decals do not all sit on one line.
        y = lo[1] + frac * (hi[1] - lo[1])
        x = lo[0] + ((np.arange(self._num_digits) % 2) * 0.7 + 0.15) * (
            hi[0] - lo[0]
        )
        return np.stack([x, y], axis=1)

    # ------------------------------------------------------------------
    # scene construction (build time, once)
    # ------------------------------------------------------------------

    def add_objects(self, arena_mjcf):
        """Add cubes, cameras, then the DR assets.

        Everything added here is baked once, at model-construction time: digit
        textures/materials, the background material pool, the digit mocap
        decals and the optional backdrop panels. Per-episode randomization only
        swaps indices and writes mocap poses, so it never needs a recompile.

        Args:
            arena_mjcf (mjcf.RootElement): Arena being built.
        """
        super().add_objects(arena_mjcf)

        self._light_elems = [
            arena_mjcf.find('light', name) for name in LIGHT_NAMES
        ]
        self._light_elems = [e for e in self._light_elems if e is not None]

        self._bake_digit_assets(arena_mjcf)
        self._bake_background_pool(arena_mjcf)
        self._add_digit_decals(arena_mjcf)
        self._add_backdrop_panels(arena_mjcf)

    def _bake_digit_assets(self, arena_mjcf):
        """Bake ten digit PNGs into ``<texture>``/``<material>`` pairs.

        Mapping matters here. The decals are very flat boxes, and a
        ``type='cube'`` texture projects from the geom center outwards -- on a
        1 mm-thick box almost the whole top face points sideways, so it samples
        the side images and the digit collapses to a smudge. A ``type='2d'``
        texture with ``texuniform=False`` and ``texrepeat=1`` maps one copy of
        the image flat across the face, which is what a decal wants.
        """
        self._digit_material_elems = []
        if self._num_digits == 0:
            return

        for digit in range(10):
            arena_mjcf.asset.add(
                'texture',
                name=f'digit_tex_{digit}',
                type='2d',
                file=png_asset(render_digit_png(digit), f'digit_tex_{digit}'),
            )
            self._digit_material_elems.append(
                arena_mjcf.asset.add(
                    'material',
                    name=f'digit_mat_{digit}',
                    texture=f'digit_tex_{digit}',
                    texuniform=False,
                    texrepeat=(1.0, 1.0),
                    specular=0.0,
                    shininess=0.0,
                    reflectance=0.0,
                )
            )

    def _bake_background_pool(self, arena_mjcf):
        """Bake the floor and wall material pools.

        Two parallel pools share one index, because MuJoCo texture types are
        geom-specific: the floor is a plane and needs ``type='2d'``, the
        backdrop panels are boxes and need ``type='cube'``. Sampling one index
        and looking it up in the matching pool keeps a single variation axis
        while rendering correctly on both.

        Index 0 reuses the stock ``grid`` checkerboard so the inherited
        ``floor.color`` axis keeps working; index 1 is a plain dark material
        matching the skybox. Remaining entries are procedural, generated from a
        fixed seed so the pool is byte-identical in every process -- a dataset
        that stores only the index can therefore be replayed exactly.
        """
        # Cube-mapped twin of the stock `grid` checker, for the box panels.
        arena_mjcf.asset.add(
            'texture',
            name='bg_tex_grid_cube',
            type='cube',
            builtin='checker',
            rgb1=(0.08, 0.11, 0.16),
            rgb2=(0.15, 0.18, 0.25),
            mark='cross',
            markrgb=(0.8, 0.8, 0.8),
            width=256,
            height=256,
        )
        grid_cube = arena_mjcf.asset.add(
            'material',
            name='bg_mat_grid_cube',
            texture='bg_tex_grid_cube',
            texuniform=True,
        )
        # An untextured material renders the same on any geom type.
        plain = arena_mjcf.asset.add(
            'material',
            name='bg_mat_plain',
            rgba=(0.15, 0.18, 0.25, 1.0),
            specular=0.0,
            shininess=0.0,
        )

        self._bg_floor_material_elems = [
            arena_mjcf.find('material', 'grid'),
            plain,
        ]
        self._bg_wall_material_elems = [grid_cube, plain]

        images = []
        if self._bg_image_dir is not None:
            from pathlib import Path

            images = sorted(Path(self._bg_image_dir).glob('*.png'))

        builtins = ('checker', 'gradient', 'flat')
        marks = ('none', 'edge', 'cross')

        for i in range(2, self._num_bg_materials):
            rng = np.random.default_rng(9_000 + i)
            shared = dict(
                builtin=builtins[i % len(builtins)],
                rgb1=rng.uniform(0.05, 0.9, size=3),
                rgb2=rng.uniform(0.05, 0.9, size=3),
                mark=marks[i % len(marks)],
                markrgb=rng.uniform(0.0, 1.0, size=3),
                width=256,
                height=256,
            )
            payload = None
            if images:
                payload = images[(i - 2) % len(images)].read_bytes()

            repeat = (float(rng.integers(1, 5)), float(rng.integers(1, 5)))
            specular = float(rng.uniform(0.0, 0.3))
            shininess = float(rng.uniform(0.0, 0.3))

            for kind, pool in (
                ('2d', self._bg_floor_material_elems),
                ('cube', self._bg_wall_material_elems),
            ):
                tex_name = f'bg_tex_{kind}_{i}'
                if payload is not None:
                    arena_mjcf.asset.add(
                        'texture',
                        name=tex_name,
                        type=kind,
                        file=png_asset(payload, tex_name),
                    )
                else:
                    arena_mjcf.asset.add(
                        'texture', name=tex_name, type=kind, **shared
                    )
                pool.append(
                    arena_mjcf.asset.add(
                        'material',
                        name=f'bg_mat_{kind}_{i}',
                        texture=tex_name,
                        texrepeat=repeat,
                        texuniform=True,
                        specular=specular,
                        shininess=shininess,
                    )
                )

    def _add_digit_decals(self, arena_mjcf):
        """Add non-colliding mocap decals, one per digit distractor."""
        self._digit_geom_elems = []
        self._digit_body_names = []
        init_pos = self._default_digit_positions()

        for i in range(self._num_digits):
            body = arena_mjcf.worldbody.add(
                'body',
                name=f'digit_{i}',
                mocap=True,
                pos=(init_pos[i][0], init_pos[i][1], DIGIT_Z),
            )
            self._digit_geom_elems.append(
                body.add(
                    'geom',
                    name=f'digit_geom_{i}',
                    type='box',
                    size=(DIGIT_HALF_EXTENT, DIGIT_HALF_EXTENT, 0.001),
                    material='digit_mat_0',
                    contype=0,
                    conaffinity=0,
                    group=1,
                )
            )
            self._digit_body_names.append(f'digit_{i}')

    def _add_backdrop_panels(self, arena_mjcf):
        """Add opaque, collision-free panels for a randomizable background.

        The existing ``wall_*`` geoms in ``floor_wall.xml`` are near-invisible
        collision proxies (``rgba`` alpha 0.1, ``conaffinity=2``) that keep the
        cubes on the table -- they are deliberately left untouched.
        """
        self._backdrop_geom_elems = []
        if not self._add_backdrop:
            return

        panels = [
            ('backdrop_back', (-0.30, 0.0, 0.45), (0.01, 1.0, 0.45)),
            ('backdrop_left', (0.35, -0.95, 0.45), (0.95, 0.01, 0.45)),
            ('backdrop_right', (0.35, 0.95, 0.45), (0.95, 0.01, 0.45)),
        ]
        for name, pos, size in panels:
            self._backdrop_geom_elems.append(
                arena_mjcf.worldbody.add(
                    'geom',
                    name=name,
                    type='box',
                    pos=pos,
                    size=size,
                    material='bg_mat_plain',
                    contype=0,
                    conaffinity=0,
                    group=1,
                )
            )

    def post_compilation_objects(self):
        """Cache MuJoCo ids for the cubes and every DR-controlled element."""
        super().post_compilation_objects()

        self._digit_mocap_ids = [
            self._model.body(name).mocapid[0]
            for name in self._digit_body_names
        ]
        self._digit_geom_ids = [
            self._model.geom(elem.full_identifier).id
            for elem in self._digit_geom_elems
        ]
        self._digit_matids = np.array(
            [
                self._model.material(elem.full_identifier).id
                for elem in self._digit_material_elems
            ],
            dtype=np.int32,
        )
        self._bg_floor_matids = np.array(
            [
                self._model.material(elem.full_identifier).id
                for elem in self._bg_floor_material_elems
            ],
            dtype=np.int32,
        )
        self._bg_wall_matids = np.array(
            [
                self._model.material(elem.full_identifier).id
                for elem in self._bg_wall_material_elems
            ],
            dtype=np.int32,
        )
        self._floor_geom_id = self._model.geom('floor').id
        self._backdrop_geom_ids = [
            self._model.geom(elem.full_identifier).id
            for elem in self._backdrop_geom_elems
        ]
        self._light_ids = [
            self._model.light(elem.full_identifier).id
            for elem in self._light_elems
        ]

    # ------------------------------------------------------------------
    # per-episode visual randomization (post-compilation, no recompile)
    # ------------------------------------------------------------------

    def _apply_visual_variations(self):
        """Write the sampled light/background/digit values into the sim.

        Every write targets an already-compiled ``MjModel``/``MjData`` field,
        so this never sets the dirty flag.
        """
        light = self.variation_space['light']
        n = len(self._light_ids)
        if n:
            ids = np.asarray(self._light_ids)
            directions = np.asarray(light['direction'].value, dtype=np.float64)
            norms = np.linalg.norm(directions, axis=-1, keepdims=True)
            directions = directions / np.maximum(norms, 1e-8)

            self._model.light_pos[ids] = light['position'].value[:n]
            self._model.light_dir[ids] = directions[:n]
            self._model.light_diffuse[ids] = light['diffuse'].value[:n]
            self._model.light_ambient[ids] = light['ambient'].value[:n]
            self._model.light_specular[ids] = light['specular'].value[:n]

        self._model.vis.headlight.diffuse = light['headlight_diffuse'].value

        background = self.variation_space['background']
        floor_idx = int(background['floor_material'].value)
        wall_idx = int(background['wall_material'].value)
        self._model.geom_matid[self._floor_geom_id] = self._bg_floor_matids[
            floor_idx
        ]
        for gid in self._backdrop_geom_ids:
            self._model.geom_matid[gid] = self._bg_wall_matids[wall_idx]

        if self._num_digits:
            digit = self.variation_space['digit']
            values = np.asarray(digit['value'].value, dtype=np.int64)
            positions = np.asarray(digit['position'].value, dtype=np.float64)
            yaws = np.asarray(digit['yaw'].value, dtype=np.float64)
            for i in range(self._num_digits):
                self._model.geom_matid[self._digit_geom_ids[i]] = (
                    self._digit_matids[values[i]]
                )
                mocap_id = self._digit_mocap_ids[i]
                self._data.mocap_pos[mocap_id] = (
                    positions[i][0],
                    positions[i][1],
                    DIGIT_Z,
                )
                self._data.mocap_quat[mocap_id] = lie.SO3.from_z_radians(
                    yaws[i]
                ).wxyz.tolist()

    # ------------------------------------------------------------------
    # size-aware geometry
    # ------------------------------------------------------------------

    def _half(self, cube_idx: int) -> float:
        """Half-extent of cube ``cube_idx`` for the current episode.

        Returns the stock ``0.02`` when ``size_aware_geometry`` is disabled, so
        the environment degrades exactly to :class:`CubeEnv`.
        """
        if not self._size_aware_geometry:
            return 0.02
        return float(self.variation_space['cube']['size'].value[cube_idx])

    def _rescaled_task_info(self, task_info: dict) -> dict:
        """Rescale a task's z waypoints to the current cube sizes.

        ``CubeEnv.set_tasks`` hard-codes the stacking ladder as
        ``0.02 + 0.04 * layer``. With randomized sizes that ladder is wrong, so
        recover the layer index and rebuild it as ``half + 2 * half * layer``.

        ``set_tasks`` itself cannot do this: it runs inside ``__init__``,
        before the variation space exists.
        """
        if not self._size_aware_geometry or task_info is None:
            return task_info

        out = dict(task_info)
        for key in ('init_xyzs', 'goal_xyzs'):
            xyzs = np.asarray(task_info[key], dtype=np.float64).copy()
            layers = np.rint((xyzs[:, 2] - 0.02) / 0.04).astype(int)
            halves = np.array(
                [self._half(i) for i in range(len(xyzs))], dtype=np.float64
            )
            xyzs[:, 2] = halves + 2.0 * halves * layers
            out[key] = xyzs
        return out

    # ------------------------------------------------------------------
    # episode lifecycle
    # ------------------------------------------------------------------

    def initialize_episode(self):
        """Randomize the scene, then run the inherited episode setup.

        The visual variations are applied *first*: in task mode the parent
        teleports the sim to the goal state and renders the goal image midway
        through its own setup, so the scene must already look final by then.
        """
        self._apply_visual_variations()

        if self._mode == 'data_collection':
            self._initialize_episode_data_collection()
            return

        saved = self.cur_task_info
        self.cur_task_info = self._rescaled_task_info(saved)
        try:
            super().initialize_episode()
        finally:
            self.cur_task_info = saved

    def _initialize_episode_data_collection(self):
        """Size-aware version of the parent's data-collection branch.

        Mirrors :meth:`CubeEnv.initialize_episode` for ``mode ==
        'data_collection'``, replacing the hard-coded ``0.02`` resting height
        with the per-cube half-extent. Identical to the parent when every cube
        is the stock size.
        """
        if not hasattr(self, '_prev_qpos'):
            self._prev_qpos = self._data.qpos.copy()
            self._prev_qvel = self._data.qvel.copy()

        colors = self.variation_space['cube']['color'].value
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = colors[i]
                self._model.geom(gid).rgba[3] = 1.0
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = colors[i]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        self.initialize_arm()

        for i in range(self._num_cubes):
            xy = self.variation_space['cube']['start_position'].value[i]
            yaw = self.variation_space['cube']['start_yaw'].value[i]
            self._data.joint(f'object_joint_{i}').qpos[:3] = (
                *xy,
                self._half(i),
            )
            self._data.joint(f'object_joint_{i}').qpos[3:] = (
                lie.SO3.from_z_radians(yaw).wxyz.tolist()
            )

        self.set_new_target(return_info=False)

        # NOTE: Goal observation is not used in data collection mode.
        self._cur_goal_ob = np.zeros_like(self.compute_observation())

        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Size-aware version of :meth:`CubeEnv.set_new_target`.

        Same target-selection logic as the parent, with three constants made
        size-dependent: the resting height (``0.02``), the stacking offset
        (``0.04``) and the xy radius used to decide whether one cube sits on
        top of another (``0.02``). Identical to the parent at the stock size.

        Args:
            return_info: Whether to return ``(observation, reset_info)``.
            p_stack: Probability of stacking on another cube when possible.

        Returns:
            ``(observation, reset_info)`` if ``return_info`` else ``None``.
        """
        assert self._mode == 'data_collection'

        block_xyzs = np.array(
            [
                self._data.joint(f'object_joint_{i}').qpos[:3]
                for i in range(self._num_cubes)
            ]
        )

        # Compute the top blocks.
        top_blocks = []
        for i in range(self._num_cubes):
            for j in range(self._num_cubes):
                if i == j:
                    continue
                radius = max(self._half(i), self._half(j))
                if (
                    block_xyzs[j][2] > block_xyzs[i][2]
                    and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2])
                    < radius
                ):
                    break
            else:
                top_blocks.append(i)

        self._target_block = self.np_random.choice(top_blocks)

        stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
        if stack:
            block_idx = self.np_random.choice(
                list(set(top_blocks) - {self._target_block})
            )
            block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
            tar_pos = np.array(
                [
                    block_pos[0],
                    block_pos[1],
                    block_pos[2]
                    + self._half(block_idx)
                    + self._half(self._target_block),
                ]
            )
        else:
            xy = self.variation_space['cube']['goal_position'].value[0]
            tar_pos = (*xy, self._half(self._target_block))
        yaw = self.variation_space['cube']['goal_yaw'].value[0]
        tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()

        # Only show the target block.
        for i in range(self._num_cubes):
            if i == self._target_block:
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
            else:
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (
                    0,
                    0,
                    -0.3,
                )
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = (
                    lie.SO3.identity().wxyz.tolist()
                )

        for i in range(self._num_cubes):
            alpha = (
                0.2
                if self._visualize_info and i == self._target_block
                else 0.0
            )
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[3] = alpha

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    # ------------------------------------------------------------------
    # observation / info
    # ------------------------------------------------------------------

    def add_object_info(self, ob_info):
        """Add cube info plus the DR ground truth used by latent probes.

        Adds to ``ob_info`` (on top of everything :class:`CubeEnv` adds):
            - ``privileged/digit_{i}_value``: digit shown by decal ``i``.
            - ``privileged/digit_{i}_pos``: its ``(x, y)`` floor position.
            - ``privileged/floor_material`` / ``privileged/wall_material``:
              background pool indices.
            - ``privileged/light_pos``: flattened per-light positions.
        """
        super().add_object_info(ob_info)

        if self._num_digits:
            digit = self.variation_space['digit']
            values = np.asarray(digit['value'].value, dtype=np.int64)
            positions = np.asarray(digit['position'].value, dtype=np.float64)
            for i in range(self._num_digits):
                ob_info[f'privileged/digit_{i}_value'] = np.array(
                    [values[i]], dtype=np.int64
                )
                ob_info[f'privileged/digit_{i}_pos'] = positions[i].copy()

        background = self.variation_space['background']
        ob_info['privileged/floor_material'] = np.array(
            [int(background['floor_material'].value)], dtype=np.int64
        )
        ob_info['privileged/wall_material'] = np.array(
            [int(background['wall_material'].value)], dtype=np.int64
        )
        ob_info['privileged/light_pos'] = np.asarray(
            self.variation_space['light']['position'].value, dtype=np.float64
        ).reshape(-1)

    # ------------------------------------------------------------------
    # dataset-driven reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None, *args, **kwargs):
        """Reset the environment, optionally restoring dataset cube targets.

        Extends :meth:`CubeEnv.reset` with one additional option:

        - ``'targets'``: sequence of ``(pos, quat)`` (``quat`` may be ``None``)
          applied to the cube target mocaps *after* the simulator state has
          been restored. This is what makes success in a dataset-driven
          evaluation refer to the recorded goal rather than to whatever task
          the environment sampled at reset.

        Args:
            seed: Seed forwarded to the parent reset.
            options: Reset options. See :meth:`CubeEnv.reset` plus ``targets``.
            *args: Forwarded to the parent reset.
            **kwargs: Forwarded to the parent reset.

        Returns:
            tuple: ``(observation, info)``.
        """
        options = options or {}
        ob, info = super().reset(seed=seed, options=options, *args, **kwargs)

        targets = options.get('targets')
        if targets is not None:
            for cube_id, target in enumerate(targets):
                pos, quat = target if len(target) == 2 else (target, None)
                self.set_target_pos(cube_id, pos, quat)
            mujoco.mj_forward(self._model, self._data)
            self.post_step()
            ob = self.compute_observation()
            info = self.get_reset_info()

        return ob, info

    def reset_options_from_dataset(self, init_row: dict, goal_row: dict):
        """Build reset options that reproduce a recorded dataset step.

        ``World._evaluate_from_dataset`` calls this in preference to the older
        ``callables`` config, and it is the only path that restores the
        domain-randomized *appearance* as well as the physical state -- the
        variation values were recorded into every row as ``variation.<key>``
        by ``EverythingToInfoWrapper``.

        Args:
            init_row: Per-column values at the episode's start step.
            goal_row: Per-column values at the goal step, keys prefixed with
                ``goal_``.

        Returns:
            dict: Options for :meth:`reset` -- ``variation`` (empty, so nothing
            is resampled), ``variation_values``, ``state`` and ``targets``.
        """
        variation_values = {}
        for name in self.variation_space.names():
            for key in self._variation_columns(name):
                if key in init_row:
                    variation_values[name] = self._cast_variation(
                        name, init_row[key]
                    )
                    break

        state = np.concatenate(
            [
                np.asarray(_row_get(init_row, 'qpos')).reshape(-1),
                np.asarray(_row_get(init_row, 'qvel')).reshape(-1),
            ]
        )

        targets = []
        for i in range(self._num_cubes):
            pos = np.asarray(
                _row_get(goal_row, f'goal_privileged/block_{i}_pos')
            ).reshape(-1)
            try:
                quat = np.asarray(
                    _row_get(goal_row, f'goal_privileged/block_{i}_quat')
                ).reshape(-1)
            except KeyError:
                quat = None
            targets.append((pos, quat))

        return {
            'variation': [],
            'variation_values': variation_values,
            'state': state,
            'targets': targets,
        }

    def subgoal_reached(self, goal_row: dict, tol: float = 0.04) -> bool:
        """Whether every cube currently sits at its position in ``goal_row``.

        Used by ``World._evaluate_from_dataset`` to decide when to advance the
        oracle-subgoal chain, so a planner that solves a waypoint early does
        not idle until its step budget runs out. Compares cube centers only,
        matching :meth:`CubeEnv._compute_successes`.

        Args:
            goal_row: One subgoal dict, holding
                ``goal_privileged/block_{i}_pos`` for each cube.
            tol: Position tolerance in meters.

        Returns:
            True if all cubes are within ``tol`` of their subgoal positions.
            False if the row does not carry cube positions at all.
        """
        for i in range(self._num_cubes):
            try:
                target = np.asarray(
                    _row_get(goal_row, f'goal_privileged/block_{i}_pos')
                ).reshape(-1)
            except KeyError:
                return False
            pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            if np.linalg.norm(pos - target) > tol:
                return False
        return True

    @staticmethod
    def _variation_columns(name: str) -> tuple[str, ...]:
        """Column spellings a dataset may use for variation ``name``.

        ``EverythingToInfoWrapper`` emits the info key ``variation.<dotted>``,
        but the lance writer flattens ``.`` to ``_``, so the same axis is
        stored as ``variation_<flattened>``. Names themselves contain
        underscores (``agent.ee_start_position``), so the flattening is not
        invertible -- always match forwards, from the known axis name to the
        column, never the other way round.
        """
        flat = name.replace('.', '_')
        return (
            f'variation.{name}',
            f'variation_{flat}',
            f'variation.{flat}',
            f'variation_{name}',
        )

    def _cast_variation(self, name: str, value):
        """Coerce a stored variation value back into its space.

        ``reset_variation_space`` validates with ``space.contains(...)`` before
        assigning, and dataset backends round-trip every column through float
        arrays -- so discrete axes come back as floats, and a box value that
        was sampled at (or pinned to) a bound can land a few ULPs outside it.
        Cast to the declared dtype and clip back into range; the clip only ever
        moves a value by float32 rounding error.
        """
        from stable_worldmodel import utils as swm_utils

        space = swm_utils.get_in(self.variation_space, name.split('.'))
        arr = np.asarray(value)

        if isinstance(space, swm_spaces.Discrete):
            return int(np.clip(np.rint(arr.reshape(-1)[0]), 0, space.n - 1))
        if isinstance(space, swm_spaces.MultiDiscrete):
            clipped = np.clip(
                np.rint(arr).reshape(space.shape), 0, space.nvec - 1
            )
            return clipped.astype(space.dtype)

        cast = arr.astype(space.dtype).reshape(space.shape)
        return np.clip(cast, space.low, space.high)


__all__ = ['DRCubeEnv', 'render_digit_png']

"""Turning OU latent pairs into a recorded encoder dataset.

The loop the plan's SS5.B describes, with one deliberate departure: it drives
:meth:`~stable_worldmodel.envs.ogbench.lejepa_cube_env.LeJEPACubeEnv.render_content`
rather than ``env.reset(options={'state': ...})``.

Why not ``reset``
-----------------
Measured at 224x224, single cube, per recorded frame:

===================================================  ============
path                                                   ms/frame
===================================================  ============
``reset()``, no recompiling axes                            21.4
``reset()``, all-content (recompiles every frame)          263.1
``render_content``, all-content                              6.1
===================================================  ============

At the plan's proposed scale -- 200k pairs, so 400k frames -- that is 29.2
core-hours against 0.67. The saving comes from three places, all of which
``reset`` pays for machinery this pipeline does not use: a full MJCF recompile
for ``cube.size`` / ``agent.color`` / ``camera.angle_delta`` (verified to
render bit-identically when written post-compilation instead), a second render
inside the ``state`` branch, and an ``initialize_arm`` IK solve whose result
``options['state']`` immediately discards.

``freeze_recompile_axes`` -- the plan's SS9 fallback, which would have dropped
the four recompiling axes out of the OU-driven set once per shard -- is
therefore **not needed and not implemented**. The all-content profile keeps
every axis live at the same cost as ``task_content``.

What a pair looks like on disk
------------------------------
One positive pair is written as a **two-step episode**, so it reads back
through the stock loader with no new code::

    swm.data.load_dataset(path, num_steps=2, frameskip=1)

yields ``pixels`` of shape ``(2, H, W, 3)`` -- the two views -- and
``latent/z`` of shape ``(2, n)``.

Ground truth is read back out of the *simulator* (``content_info()``), never
copied from what the sampler requested. The two differ wherever a value was
clipped to its bounds, IK failed to converge, or a coupled joint did not track
its driver -- which is precisely when the difference matters.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from loguru import logger as logging

from stable_worldmodel import utils as swm_utils
from stable_worldmodel.data.format import EPISODE_DATA_KEY
from stable_worldmodel.data.utils import get_cache_dir


def variation_shape(env, dotted):
    """Shape the variation space expects for axis ``dotted``."""
    space = swm_utils.get_in(env.variation_space, dotted.split('.'))
    return tuple(np.asarray(space.low).shape) or (1,)


#: Axes whose collision makes the cube undetectable. Both must be **style**
#: for the contrast floor to apply -- see :func:`contrast_is_legal`.
CONTRAST_PAIR = ('cube.color', 'background.floor_rgb')

#: Attempts before the contrast floor gives up on a draw and accepts it. A 0.2
#: L-infinity floor rejects about 5% of draws, so exhausting this is a sign the
#: bounds have changed, not bad luck.
MAX_CONTRAST_ROUNDS = 32


def contrast_is_legal(registry):
    """Whether a cube/floor contrast floor can be applied without breaking a
    stronger assumption.

    Rejecting draws where the cube's colour is close to the floor's removes the
    tail where the cube is invisible -- a genuine failure of injectivity, since
    two different cube positions then render the same image. It is safe only
    while **both** axes are ``style``:

    * Two style axes may be made dependent on each other freely. Nothing asks
      style to be internally independent; the assumption that matters is that
      style is independent of *content*, and a predicate reading only style
      preserves that.
    * If ``cube.color`` is **content** (``task_content``), the same rejection
      either truncates the content marginal -- a support truncation, on a
      coordinate that is supposed to be Gaussian -- or, if applied to the floor
      alone given the cube's colour, makes style a function of content and
      breaks the one independence assumption the theory rests on. Both are
      worse than the collision, so the floor is skipped and the caller warned.

    Returns:
        bool: True when both axes of :data:`CONTRAST_PAIR` are style.
    """
    style_names = {latent.name for latent in registry.style}
    return all(name in style_names for name in CONTRAST_PAIR)


def _contrast(payload):
    """L-infinity distance between the cube's colour and the floor's."""
    cube = np.asarray(payload['cube.color'], dtype=np.float64).reshape(-1)
    floor = np.asarray(
        payload['background.floor_rgb'], dtype=np.float64
    ).reshape(-1)
    channels = min(cube.size, floor.size)
    return float(np.abs(cube[:channels] - floor[:channels]).max())


def sample_style(env, registry, rng, min_contrast=0.0):
    """Draw one independent value for every ``style`` latent.

    Style is resampled **within** a pair -- the two views of a positive pair
    get different draws. That is what the alignment term is asked to discard,
    and what the style-invariance metric later measures.

    Args:
        env: The environment, for its variation space.
        registry: A resolved registry.
        rng: A ``numpy.random.Generator``.
        min_contrast: Smallest L-infinity distance allowed between
            ``cube.color`` and ``background.floor_rgb``. Both are drawn
            uniformly on ``[0, 1]^3`` and independently, so without a floor the
            cube occasionally renders the same colour as what is behind it and
            vanishes -- an occlusion nobody declared and nothing records. At
            ``0.2`` roughly 5% of draws are redrawn, which removes the tail
            without meaningfully reshaping the style distribution. ``0.0``
            disables it. Applied only when :func:`contrast_is_legal`.

            The floor is a proxy, not a guarantee: ``background.floor_rgb`` is
            modulated by whichever of the 8 ``floor_material`` textures is
            active, so the rendered background is not exactly this colour.

    Returns:
        tuple: ``(variation_values, flat)`` -- the payload for
        ``render_content`` and a flat vector for the ``latent/style`` column.
    """
    axes = []
    for latent in registry.style:
        if not latent.channel.startswith('variation:'):
            continue
        axis = latent.channel.split(':', 1)[1]
        axes.append(
            (axis, swm_utils.get_in(env.variation_space, axis.split('.')))
        )

    check = min_contrast > 0.0 and all(
        name in dict(axes) for name in CONTRAST_PAIR
    )

    for _ in range(MAX_CONTRAST_ROUNDS):
        payload = {}
        for axis, space in axes:
            space.seed(int(rng.integers(0, 2**31 - 1)))
            payload[axis] = space.sample()
        if not check or _contrast(payload) >= min_contrast:
            break

    flat = [
        np.asarray(payload[axis], dtype=np.float64).reshape(-1)
        for axis, _ in axes
    ]
    return payload, (
        np.concatenate(flat) if flat else np.zeros(0, dtype=np.float64)
    )


def excluded_payload(env, registry):
    """Pin every ``excluded`` variation axis to its canonical ``init_value``.

    Excluded latents used to be simply *unwritten*: ``content_payload`` handles
    content axes and :func:`sample_style` handles style axes, so an excluded
    axis was left holding whatever the opening
    ``env.reset(options={'variation': ['all']})`` happened to draw for it.

    That is not "pinned" -- it is pinned to a random value, and ``base_seed``
    differs per shard, so a sharded collection would hold a *different*
    constant in every shard. The result is a nuisance perfectly correlated with
    shard identity: invisible within a shard, and worse than either style or
    content once the shards are merged, because nothing downstream records it.

    Writing the axis's own ``init_value`` on every frame makes the exclusion
    deterministic, shard-independent, and identical to what the manifest says
    it is.

    Returns:
        dict: Axis path -> pinned value, for ``render_content``.
    """
    payload = {}
    for latent in registry.by_role('excluded'):
        if not latent.channel.startswith('variation:'):
            continue
        axis = latent.channel.split(':', 1)[1]
        space = swm_utils.get_in(env.variation_space, axis.split('.'))
        pinned = space.init_value
        if pinned is None:
            continue
        payload[axis] = np.array(pinned)
    return payload


def content_payload(env, registry, values, index):
    """Split one row of decoded content latents into state and appearance.

    Args:
        env: The environment, for variation-space shapes.
        registry: A resolved :class:`~...latents.LatentRegistry`.
        values: Output of ``registry.to_physical``, mapping name to
            ``(B, dim)``.
        index: Which row of ``values`` to take.

    Returns:
        tuple: ``(physical, variation_values)``, ready for
        ``set_content_state`` and ``render_content`` respectively.
    """
    n_cubes = env._num_cubes
    physical = {}
    variation = {}

    for latent in registry.content:
        row = np.asarray(values[latent.name][index], dtype=np.float64)

        if latent.channel == 'state':
            if latent.name == 'cube.pos_xy':
                physical['cube.pos_xy'] = row.reshape(n_cubes, 2)
            elif latent.name == 'cube.pos_z':
                physical['cube.pos_z'] = row.reshape(n_cubes)
            elif latent.name == 'cube.yaw':
                physical['cube.yaw'] = row.reshape(n_cubes)
            elif latent.name == 'effector.pos':
                physical['effector.pos'] = row.reshape(3)
            elif latent.name == 'effector.yaw':
                physical['effector.yaw'] = float(row.reshape(-1)[0])
            elif latent.name == 'gripper.opening':
                physical['gripper.opening'] = float(row.reshape(-1)[0])
            else:
                raise KeyError(
                    f'{latent.name} declares channel "state" but '
                    'content_payload does not know how to write it. A latent '
                    'the collector cannot set would be recorded as a label '
                    'with no corresponding pixel change.'
                )
        elif latent.channel.startswith('variation:'):
            axis = latent.channel.split(':', 1)[1]
            shape = variation_shape(env, axis)
            if latent.kind == 'discrete':
                variation[axis] = int(np.rint(row.reshape(-1)[0]))
            elif latent.axis_rows is not None:
                # A latent that covers only some rows of its axis -- the
                # directional light's position is pinned because it never
                # reaches a pixel, so `light.position` is 3-dimensional over a
                # (2, 3) axis. Start from the axis's own pinned value and
                # overwrite just the rows this latent owns; reshaping the flat
                # row across the whole axis would scatter it into the wrong
                # lights.
                space = swm_utils.get_in(env.variation_space, axis.split('.'))
                full = np.array(
                    space.value
                    if space.value is not None
                    else space.init_value,
                    dtype=np.float64,
                ).reshape(shape)
                full[list(latent.axis_rows)] = row.reshape(
                    len(latent.axis_rows), -1
                )
                variation[axis] = full
            else:
                variation[axis] = row.reshape(shape)
        else:
            raise KeyError(
                f'{latent.name}: unknown channel {latent.channel!r}.'
            )

    return physical, variation


def config_hash(manifest):
    """Stable digest of a dataset's configuration.

    Recorded into every result row so a metric row can always be traced to
    the exact dataset it came from -- and so two datasets that differ in any
    knob can never be conflated.
    """
    payload = json.dumps(manifest, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def collect_pairs(
    env,
    registry,
    sampler,
    num_pairs,
    seed=0,
    camera='front_pixels',
    log_every=2000,
    resample_style_within_pair=True,
    min_contrast=0.0,
):
    """Generate encoder-dataset episodes, one two-step episode per pair.

    Yields rather than returns, so :meth:`LanceWriter.write_episodes` can
    stream the whole run through a single Lance version without ever holding
    more than one episode in memory.

    Args:
        env: A reset :class:`LeJEPACubeEnv`.
        registry: A resolved latent registry.
        sampler: An :class:`~...ou.OUSampler` with ``n == registry.n``.
        num_pairs: How many positive pairs to write.
        seed: Seed for the style stream, kept separate from the sampler's own.
        camera: Camera to record.
        log_every: Progress cadence, in pairs.
        min_contrast: Cube/floor contrast floor, passed to
            :func:`sample_style`. Disabled with a warning under any profile
            where ``cube.color`` is content.
        resample_style_within_pair: Whether the two views of a pair get
            *independent* style draws (the default) or share one.

            Independent draws are what makes style discardable, and the reason
            is spectral rather than incidental. Style redrawn per view is
            independent across the pair, so the transition operator annihilates
            every function of it -- eigenvalue 0 -- while the content
            coordinates sit at ``rho`` and their degree-2 Hermite terms at
            ``rho^2``. At the frozen ``rho = 0.9`` the ordering is 0.90
            (content, linear) > 0.81 (content, quadratic) > ... > 0 (anything
            touching style), so the top-``n`` eigenspace is exactly the content
            block and slowness alone selects it.

            Sharing one draw per pair does **not** recover the theory's
            setting -- it is strictly worse than the default. A shared draw is
            *perfectly* correlated across the pair, so ``rho_style = 1``, not
            ``rho``: every function of style sits at eigenvalue 1, strictly
            above content's 0.90. Alignment is then minimised *exactly* by
            encoding style and ignoring content, because the two views agree on
            style by construction. With ~40 dims of style variation to draw
            ``n`` independent functions from, the encoder can reach alignment 0
            while satisfying SIGReg and representing no content at all.

            So this flag is only meaningful *together with* a profile that
            pins style -- one where every non-content latent is ``excluded``.
            That pair of settings is the theory's literal ``x = g(z)``; this
            flag alone is a degenerate configuration.

    Yields:
        dict: One episode, two steps deep.

    Raises:
        ValueError: If the sampler's width does not match the registry's.
    """
    if sampler.n != registry.n:
        raise ValueError(
            f'sampler.n = {sampler.n} but the registry declares n = '
            f'{registry.n}. A mismatch here silently mislabels every row.'
        )

    rng = np.random.default_rng(seed)
    n_cubes = env._num_cubes
    clip_total = 0.0
    batch = 256

    # Constant for the whole run, so an excluded axis holds the same value in
    # every frame of every shard. See `excluded_payload`.
    pinned = excluded_payload(env, registry)

    # Skipped rather than silently misapplied when `cube.color` is content:
    # rejecting on a content coordinate truncates its marginal.
    if min_contrast > 0.0 and not contrast_is_legal(registry):
        logging.warning(
            f'min_contrast={min_contrast} requested but '
            f'{CONTRAST_PAIR[0]} is not style under this profile, so the '
            'cube/floor contrast floor is DISABLED. Rejecting on a content '
            'coordinate would truncate its marginal, or make '
            'style depend on content; both are worse than the collision.'
        )
        min_contrast = 0.0

    written = 0
    while written < num_pairs:
        size = min(batch, num_pairs - written)
        z, z_next = sampler.sample_pairs(size)

        values_a, clip_a = registry.to_physical(z)
        values_b, clip_b = registry.to_physical(z_next)
        clip_total += 0.5 * (clip_a + clip_b) * size

        for k in range(size):
            steps = {}
            # One draw per pair when style is not resampled within it, so both
            # views share it and `x = g(z)` becomes deterministic.
            shared_style = (
                None
                if resample_style_within_pair
                else sample_style(env, registry, rng, min_contrast)
            )
            for view, (zz, values) in enumerate(
                ((z, values_a), (z_next, values_b))
            ):
                physical, variation = content_payload(env, registry, values, k)
                style, style_flat = (
                    sample_style(env, registry, rng, min_contrast)
                    if shared_style is None
                    else shared_style
                )

                state = env.set_content_state(physical)
                frame = env.render_content(
                    state,
                    {**pinned, **variation, **style},
                    camera=camera,
                )
                info = env.content_info()

                row = {
                    'pixels': np.asarray(frame),
                    'latent/z': zz[k].astype(np.float32),
                    'latent/style': style_flat.astype(np.float32),
                    'latent/view': np.array([view], dtype=np.int64),
                }
                # Every privileged/proprio column the environment emits, so
                # the recovery target can be rebuilt from a recorded row
                # without re-running the environment.
                for key, value in info.items():
                    if key.startswith(('privileged/', 'proprio/')) or key in (
                        'qpos',
                        'qvel',
                    ):
                        arr = np.asarray(value)
                        if arr.dtype.kind in 'iufb':
                            row[key] = arr.reshape(-1)

                for key, value in row.items():
                    steps.setdefault(key, []).append(value)

            yield {
                **steps,
                EPISODE_DATA_KEY: {
                    'pair_index': written + k,
                    'num_cubes': int(n_cubes),
                },
            }

        written += size
        if log_every and written % log_every < batch:
            logging.info(f'  {written}/{num_pairs} pairs')

    logging.info(
        f'clipped {clip_total / max(1, num_pairs) * 100:.4f}% of latent '
        'coordinates to their declared bounds'
    )


#: Episodes materialised per `write_episodes` call. Bounds peak memory to one
#: chunk (at 224x224 a pair is ~300 KiB, so 2000 pairs is ~600 MiB) while
#: keeping the number of Lance versions small.
WRITE_CHUNK = 2000


def write_chunked(writer, episodes, chunk_size=WRITE_CHUNK):
    """Write an episode stream without letting the writer pull it.

    ``LanceWriter.write_episodes`` hands the caller's iterable to Lance as a
    ``RecordBatchReader``, and Lance drains that reader from its **own
    background thread**. That is fine for a generator that only shuffles
    arrays around -- and a deadlock for this one, because our generator
    renders.

    A MuJoCo ``Renderer`` owns an OpenGL context bound to the thread that
    created it. Under ``MUJOCO_GL=glfw`` on macOS, touching that context from
    Lance's background thread hangs: the main thread blocks in
    ``create_table`` waiting on a future, and the background loop blocks in
    the render call, forever. The symptom is a test that consumes no CPU and
    never returns.

    So the frames are rendered here, on the calling thread, a chunk at a time,
    and each finished chunk is handed over as a materialised list. Memory
    stays bounded to one chunk, and the writer never pulls a frame into a
    thread that cannot draw it.

    Args:
        writer: An open ``LanceWriter`` (or anything with ``write_episodes``).
        episodes: Iterable of episode dicts -- typically :func:`collect_pairs`.
        chunk_size: Episodes materialised per write.

    Returns:
        int: Episodes written.
    """
    chunk = []
    written = 0
    for episode in episodes:
        chunk.append(episode)
        if len(chunk) >= chunk_size:
            writer.write_episodes(chunk)
            written += len(chunk)
            chunk = []
    if chunk:
        writer.write_episodes(chunk)
        written += len(chunk)
    return written


def build_manifest(
    env,
    registry,
    sampler,
    num_pairs,
    seed,
    profile_name,
    extra=None,
    resample_style_within_pair=True,
    min_contrast=0.0,
):
    """Everything needed to tell this dataset apart from any other.

    Written both into ``EPISODE_DATA_KEY``-adjacent sidecar JSON and hashed
    into ``config_hash``, so a dataset can never be mistaken for a
    differently-configured one -- the failure the plan's SS5.B.3 calls out.
    """
    manifest = {
        'env': {
            'id': 'swm/LeJEPACube-v0',
            'env_type': env._env_type,
            'num_cubes': int(env._num_cubes),
            'num_digits': int(env._num_digits),
            'marker_enabled': bool(env._marker_enabled),
            'marker_face': env._marker_face,
            'render': [int(env._render_height), int(env._render_width)],
        },
        'latents': registry.describe(),
        'ou': sampler.describe(),
        'num_pairs': int(num_pairs),
        'seed': int(seed),
        'profile': profile_name,
        # The appearance randomisation range. Recorded because appearance
        # outside the encoder's training range is not covered by the alignment
        # loss and leaks straight into the embedding, so any later dataset
        # scored against this encoder has to be checked against it.
        'style_range': {
            latent.name: {
                'low': latent.low.tolist(),
                'high': latent.high.tolist(),
            }
            for latent in registry.style
        },
        # Independent style per view is what puts style at autocorrelation 0
        # and so below content's degree-2 Hermite terms in the transition
        # operator's spectrum. Flipping it changes which latents the objective
        # can distinguish at all, so it is part of the dataset's identity.
        'resample_style_within_pair': bool(resample_style_within_pair),
        # The cube/floor contrast floor actually in force. Recorded because it
        # reshapes the style distribution, and because it is silently disabled
        # under profiles where `cube.color` is content.
        'min_contrast': float(
            min_contrast if contrast_is_legal(registry) else 0.0
        ),
        'contrast_floor_applies': bool(
            min_contrast > 0.0 and contrast_is_legal(registry)
        ),
        # The value every excluded axis was actually held at. Recorded because
        # "excluded" is a claim about a constant, and an unrecorded constant
        # that silently differed per shard is exactly the bug this replaced.
        'excluded_pinned': {
            axis: np.asarray(value).tolist()
            for axis, value in excluded_payload(env, registry).items()
        },
        **(extra or {}),
    }
    manifest['config_hash'] = config_hash(manifest)
    return manifest


def write_manifest(path, manifest):
    """Drop the sidecar next to the dataset."""
    path = Path(path)
    sidecar = path.parent / f'{path.stem}_manifest.json'
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, 'w') as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return sidecar


def load_manifest(dataset_name, cache_dir=None):
    """Read the sidecar manifest for a dataset name.

    Args:
        dataset_name: Dataset name or path, e.g.
            ``'ogbench/cube_single_ou.lance'``. The manifest is looked up as
            ``<cache>/datasets/<stem>_manifest.json``.
        cache_dir: Cache root; defaults to ``STABLEWM_HOME``.

    Returns:
        dict: The manifest.

    Raises:
        FileNotFoundError: If no manifest is there. Every consumer of a
            manifest needs the dataset config it came from, and guessing a
            default is how two runs end up indistinguishable in the scatter.
    """
    name = Path(dataset_name)
    # `name.parent` matters: datasets are namespaced (`ogbench/...`) and
    # `write_manifest` drops the sidecar *next to* the dataset. Using only the
    # stem would look in `datasets/` for a file that lives in
    # `datasets/ogbench/`.
    path = (
        Path(cache_dir or get_cache_dir())
        / 'datasets'
        / name.parent
        / f'{name.stem}_manifest.json'
    )
    if not path.exists():
        raise FileNotFoundError(
            f'no manifest at {path}. It is written next to the dataset by the '
            'OU and arm-C collectors; a sharded run writes one per shard, and '
            '`swm merge` does not merge them.'
        )
    with open(path) as handle:
        return json.load(handle)


def audit_dataset(
    registry, sampler, z_recorded, z_next_recorded, readback=None
):
    """Post-hoc check that the dataset carries the process it claims to.

    Two checks, and they answer different questions. Reading them as one is
    how a pipeline bug survives collection.

    **Storage round-trip** (always). The recorded ``latent/z`` column is the
    sampler's own draw, so comparing its marginal and lag-1 correlation against
    the declared ``rho`` cannot detect anything the *environment* did -- it
    compares the sampler with itself. What it does catch is real but narrow: a
    reshape that scrambled a coordinate, a shard written with the wrong width,
    a column that lost its pairing with ``step_idx``. Expect agreement to
    sampling error, and treat a clean result as "storage is intact", not as
    "the frames match the labels".

    **Render round-trip** (when ``readback`` is given). This is the check that
    actually bears on recovery. ``collect_pairs`` *commands* a physical state
    and records the pre-clip ``z`` that asked for it, so any gap between the
    commanded state and the one the simulator ended up in -- a clipped
    coordinate, an IK solve that did not converge, a coupled joint that did not
    track its driver -- puts the label out of step with the pixels and caps
    recovery for a reason no encoder can fix. Pass the recorded
    ``privileged/*`` / ``proprio/*`` columns as physical values and this
    reports the residual per latent, in z-space units so it is comparable
    across coordinates.

    A latent with no physical readback cannot appear here. Under
    ``physical_content`` that is exactly ``cube.size``, whose presence in the
    render no recorded column can confirm.

    Args:
        registry: The resolved registry.
        sampler: The sampler that produced the data.
        z_recorded: ``(B, n)`` first views, from ``latent/z``.
        z_next_recorded: ``(B, n)`` second views.
        readback: Optional mapping from latent name to ``(B, dim)`` *physical*
            values read back out of the recorded columns.

    Returns:
        dict: The declared-versus-achieved report. ``render`` is present only
        when ``readback`` was supplied, and carries ``max_abs_z_error`` per
        latent plus the overall worst, ``render_max_abs_z_error``.
    """
    achieved = sampler.empirical_rho(z_recorded, z_next_recorded)
    declared = np.full(registry.n, sampler.rho)
    report = {
        'n': registry.n,
        'rho_declared': sampler.rho,
        'rho_achieved': achieved.tolist(),
        'rho_max_abs_error': float(np.abs(achieved - declared).max()),
        'marginal_mean': z_recorded.mean(axis=0).tolist(),
        'marginal_var': z_recorded.var(axis=0).tolist(),
        'marginal_var_max_dev': float(
            np.abs(z_recorded.var(axis=0) - 1.0).max()
        ),
    }

    if readback:
        offsets = registry.slices()
        commanded = registry.to_z(
            {
                latent.name: z_recorded[:, offsets[latent.name]]
                for latent in registry.content
            }
        )
        observed = registry.to_z(readback, missing='nan')
        per_latent = {}
        for latent in registry.content:
            if latent.name not in readback:
                continue
            index = offsets[latent.name]
            gap = np.abs(commanded[:, index] - observed[:, index])
            per_latent[latent.name] = float(gap.max())
        report['render'] = {
            'max_abs_z_error': per_latent,
            'unreadable': [
                latent.name
                for latent in registry.content
                if latent.name not in readback
            ],
        }
        report['render_max_abs_z_error'] = (
            max(per_latent.values()) if per_latent else None
        )

    return report


__all__ = [
    'load_manifest',
    'audit_dataset',
    'build_manifest',
    'collect_pairs',
    'config_hash',
    'content_payload',
    'sample_style',
    'variation_shape',
    'write_chunked',
    'write_manifest',
]

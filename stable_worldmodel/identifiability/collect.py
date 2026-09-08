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


def sample_style(env, registry, rng):
    """Draw one independent value for every ``style`` latent.

    Style is resampled **within** a pair -- the two views of a positive pair
    get different draws. That is what the alignment term is asked to discard,
    and what the style-invariance metric later measures.

    Returns:
        tuple: ``(variation_values, flat)`` -- the payload for
        ``render_content`` and a flat vector for the ``latent/style`` column.
    """
    payload = {}
    flat = []
    for latent in registry.style:
        if not latent.channel.startswith('variation:'):
            continue
        axis = latent.channel.split(':', 1)[1]
        space = swm_utils.get_in(env.variation_space, axis.split('.'))
        space.seed(int(rng.integers(0, 2**31 - 1)))
        value = space.sample()
        payload[axis] = value
        flat.append(np.asarray(value, dtype=np.float64).reshape(-1))
    return payload, (
        np.concatenate(flat) if flat else np.zeros(0, dtype=np.float64)
    )


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
                space = swm_utils.get_in(
                    env.variation_space, axis.split('.')
                )
                full = np.array(
                    space.value if space.value is not None else space.init_value,
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

    written = 0
    while written < num_pairs:
        size = min(batch, num_pairs - written)
        z, z_next, meta = sampler.sample_pairs(size)

        values_a, clip_a = registry.to_physical(z)
        values_b, clip_b = registry.to_physical(z_next)
        clip_total += 0.5 * (clip_a + clip_b) * size

        for k in range(size):
            steps = {}
            for view, (zz, values) in enumerate(
                ((z, values_a), (z_next, values_b))
            ):
                physical, variation = content_payload(
                    env, registry, values, k
                )
                style, style_flat = sample_style(env, registry, rng)

                state = env.set_content_state(physical)
                frame = env.render_content(
                    state, {**variation, **style}, camera=camera
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

            steps['ou/rho_per_dim'] = [
                meta['rho'][k].astype(np.float32),
                meta['rho'][k].astype(np.float32),
            ]

            yield {
                **steps,
                EPISODE_DATA_KEY: {
                    'pair_index': int(meta['index'][k]),
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
    env, registry, sampler, violation, num_pairs, seed, profile_name, extra=None
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
        'violation': violation.describe(),
        'num_pairs': int(num_pairs),
        'seed': int(seed),
        'profile': profile_name,
        # The appearance randomisation range, recorded so the predictor
        # dataset can assert it matches. Appearance outside the encoder's
        # training range is not covered by the alignment loss and leaks
        # straight into the embedding.
        'style_range': {
            latent.name: {
                'low': latent.low.tolist(),
                'high': latent.high.tolist(),
            }
            for latent in registry.style
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


def audit_dataset(registry, sampler, z_recorded, z_next_recorded):
    """Post-hoc check that the dataset carries the process it claims to.

    The plan's phase-3 gate. Comparing the *recorded* marginal and
    autocorrelation against what the manifest declares is the only way to
    catch a pipeline that quietly altered the process -- through clipping,
    rejection, or a reshape that scrambled a coordinate.

    Args:
        registry: The resolved registry.
        sampler: The sampler that produced the data.
        z_recorded: ``(B, n)`` first views.
        z_next_recorded: ``(B, n)`` second views.

    Returns:
        dict: Declared-versus-achieved report.
    """
    achieved = sampler.empirical_rho(z_recorded, z_next_recorded)
    declared = np.broadcast_to(sampler.rho, (registry.n,))
    return {
        'n': registry.n,
        'rho_declared': declared.tolist(),
        'rho_achieved': achieved.tolist(),
        'rho_max_abs_error': float(np.abs(achieved - declared).max()),
        'marginal_mean': z_recorded.mean(axis=0).tolist(),
        'marginal_var': z_recorded.var(axis=0).tolist(),
        'marginal_var_max_dev': float(
            np.abs(z_recorded.var(axis=0) - 1.0).max()
        ),
    }


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

"""End-to-end tests for the encoder-dataset pipeline.

The plan's phase-3 gate: a dataset whose *recorded* marginal and
per-dimension autocorrelation match what its manifest declares. Everything
upstream can be individually correct and still produce a dataset that carries a
different process than it claims -- through clipping, rejection, or a reshape
that scrambles a coordinate -- and none of that raises.
"""

import os

import numpy as np
import pytest


os.environ.setdefault('MUJOCO_GL', 'glfw')

pytest.importorskip('ogbench')
pytest.importorskip('mujoco')
pytest.importorskip('lancedb')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.ogbench.lejepa_cube_env import (  # noqa: E402
    LeJEPACubeEnv,
)
from stable_worldmodel.identifiability import collect as ident  # noqa: E402
from stable_worldmodel.identifiability.latents import (  # noqa: E402
    PROFILES,
    build_registry,
)
from stable_worldmodel.identifiability.ou import OUSampler  # noqa: E402


IMAGE = 64


@pytest.fixture(scope='module')
def env():
    e = LeJEPACubeEnv(
        env_type='single',
        ob_type='pixels',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=IMAGE,
        height=IMAGE,
        num_digits=1,
    )
    e.reset(seed=0, options={'variation': ['all']})
    yield e
    e.close()


def make(env, profile='physical_content', **sampler_kwargs):
    registry = build_registry(env).resolve(PROFILES[profile])
    kwargs = {'rho': 0.9, **sampler_kwargs}
    sampler = OUSampler(n=registry.n, seed=0, **kwargs)
    return registry, sampler


# ------------------------------------------------------------ episode shape


def test_a_pair_is_a_two_step_episode(env):
    """Two steps is what makes the stock loader yield a positive pair."""
    registry, sampler = make(env)
    episodes = list(
        ident.collect_pairs(env, registry, sampler, 3, log_every=0)
    )
    assert len(episodes) == 3

    for episode in episodes:
        assert len(episode['pixels']) == 2
        assert len(episode['latent/z']) == 2
        assert episode['pixels'][0].shape == (IMAGE, IMAGE, 3)


def test_ground_truth_columns_are_present(env):
    """The recovery target must be rebuildable from a recorded row alone."""
    registry, sampler = make(env)
    episode = next(ident.collect_pairs(env, registry, sampler, 1, log_every=0))

    for latent in registry.content:
        if latent.readback and latent.readback.startswith(
            ('privileged/', 'proprio/')
        ):
            key = latent.readback.format(i=0)
            assert key in episode, f'{latent.name} readback {key} missing'
    assert 'qpos' in episode and 'qvel' in episode


def test_the_two_views_differ(env):
    """A pair whose views are identical carries no signal for the OU step."""
    registry, sampler = make(env)
    episode = next(ident.collect_pairs(env, registry, sampler, 1, log_every=0))
    a, b = episode['pixels']
    assert not np.array_equal(a, b)


def test_style_is_resampled_within_a_pair(env):
    """The alignment term can only discard style that actually varies."""
    registry, sampler = make(env)
    episode = next(ident.collect_pairs(env, registry, sampler, 1, log_every=0))
    style_a, style_b = episode['latent/style']
    assert style_a.size > 0
    assert not np.allclose(style_a, style_b)


def test_qvel_is_zero_in_every_frame(env):
    """No single-frame image correlate, so it is pinned rather than sampled."""
    registry, sampler = make(env)
    for episode in ident.collect_pairs(env, registry, sampler, 3, log_every=0):
        for qvel in episode['qvel']:
            np.testing.assert_allclose(qvel, 0.0)


# --------------------------------------------------------- the phase-3 gate


def test_recorded_process_matches_the_declared_one(env):
    """The gate: recorded rho and marginal must match the manifest.

    Run on ``latent/z`` as *written to disk*, not on the sampler's output, so
    it covers the whole path including the physical round-trip.

    1200 pairs is enough to pin each per-dimension rho to about +-0.03 at
    rho = 0.9, comfortably inside the tolerance below. Every pair costs two
    renders and a render is ~4.6 ms of fixed GL overhead regardless of
    resolution, so this is the whole budget of the test.
    """
    registry, sampler = make(env)
    episodes = list(
        ident.collect_pairs(env, registry, sampler, 1200, log_every=0)
    )

    z = np.stack([ep['latent/z'][0] for ep in episodes])
    z_next = np.stack([ep['latent/z'][1] for ep in episodes])

    report = ident.audit_dataset(registry, sampler, z, z_next)
    assert report['rho_max_abs_error'] < 0.06, report['rho_achieved']
    assert report['marginal_var_max_dev'] < 0.15, report['marginal_var']
    assert np.abs(report['marginal_mean']).max() < 0.1


def test_excluded_axes_are_pinned_to_a_recorded_constant(env):
    """An excluded axis must hold one value, and the manifest must say which.

    Excluded latents used to be simply unwritten, which left them holding
    whatever the opening ``reset(options={'variation': ['all']})`` drew --
    a random constant, and a *different* one per shard, since ``base_seed``
    differs. That is a nuisance perfectly correlated with shard identity:
    invisible within a shard and unrecorded anywhere.
    """
    registry, sampler = make(env, profile='physical_content')
    pinned = ident.excluded_payload(env, registry)

    assert 'camera.angle_delta' in pinned
    assert np.allclose(pinned['camera.angle_delta'], 0.0)

    # Independent of the reset draw: re-randomise everything, and the pin holds.
    env.reset(seed=12345, options={'variation': ['all']})
    again = ident.excluded_payload(env, registry)
    assert np.allclose(
        again['camera.angle_delta'], pinned['camera.angle_delta']
    )

    manifest = ident.build_manifest(
        env,
        registry,
        sampler,
        4,
        0,
        'physical_content',
    )
    assert 'camera.angle_delta' in manifest['excluded_pinned']


def test_style_may_be_shared_across_a_pair(env):
    """``resample_within_pair=False`` shares one style draw across a pair.

    Only meaningful together with a profile that pins style: a shared draw is
    perfectly correlated across the pair, so rho_style = 1, *above* content's
    rho, and alignment is then minimised by encoding style and ignoring content
    entirely. This test pins the mechanics of the flag, not a recommendation to
    use it alone.
    """
    registry, sampler = make(env, profile='physical_content')

    shared = list(
        ident.collect_pairs(
            env,
            registry,
            sampler,
            6,
            log_every=0,
            resample_style_within_pair=False,
        )
    )
    for episode in shared:
        a, b = episode['latent/style']
        assert np.allclose(a, b), 'views should share one style draw'

    independent = list(
        ident.collect_pairs(env, registry, sampler, 6, log_every=0)
    )
    assert any(
        not np.allclose(*episode['latent/style']) for episode in independent
    ), 'default must redraw style per view'


def test_readback_recovers_the_latents_that_were_written(env):
    """Simulator ground truth must track the requested z, not drift from it.

    The drift this guards against -- a clipped value, an unconverged IK solve,
    a coupled joint that did not track its driver -- is a property of the
    latents written into ``qpos``, and it matters because ``latent/z`` records
    what was *asked for*: any gap puts the label out of step with the pixels
    and caps recovery for a reason no encoder can fix.

    ``cube.size`` is skipped, and it is the one latent that cannot be checked
    this way: it is a direct ``model.geom_size`` write with nothing in between
    to diverge, and it has no ``privileged/*`` column to read back from. Its
    presence in the *render* is therefore unverifiable from a recorded row,
    which is worth remembering when it scores near zero.
    """
    registry, sampler = make(env)
    episodes = list(
        ident.collect_pairs(env, registry, sampler, 120, log_every=0)
    )

    offsets = registry.slices()
    readable = [
        latent for latent in registry.content if latent.name != 'cube.size'
    ]

    requested, observed = [], []
    for episode in episodes:
        for view in (0, 1):
            info = {
                key: np.asarray(values[view])
                for key, values in episode.items()
                if isinstance(values, list)
            }
            values = registry.read_info(info, num_cubes=1)
            assert 'cube.size' not in values, (
                'cube.size gained a physical readback; this test skips it '
                'precisely because it has none'
            )
            observed.append(
                registry.to_z(
                    {
                        latent.name: values[latent.name].reshape(1, -1)
                        for latent in readable
                    },
                    missing='nan',
                )[0]
            )
            requested.append(episode['latent/z'][view])

    requested = np.stack(requested)
    observed = np.stack(observed)
    # Per-coordinate correlation, which is what the recovery metrics consume.
    for latent in readable:
        for j in range(offsets[latent.name].start, offsets[latent.name].stop):
            corr = np.corrcoef(requested[:, j], observed[:, j])[0, 1]
            assert corr > 0.99, (
                f'{latent.name} coordinate {j} reads back at correlation '
                f'{corr:.4f} against what was written'
            )


# --------------------------------------------------------------- manifest


def test_manifest_distinguishes_configurations(env):
    """A dataset must never be mistakable for a differently-configured one."""
    registry, sampler = make(env)
    a = ident.build_manifest(env, registry, sampler, 10, 0, 'physical_content')
    other = OUSampler(n=registry.n, rho=0.8, seed=0)
    b = ident.build_manifest(env, registry, other, 10, 0, 'physical_content')
    assert a['config_hash'] != b['config_hash']


def test_manifest_records_the_style_range(env):
    """The predictor dataset asserts against this.

    Appearance outside the encoder's randomisation range is not covered by the
    alignment loss and leaks straight into the embedding.
    """
    registry, sampler = make(env)
    manifest = ident.build_manifest(
        env, registry, sampler, 10, 0, 'physical_content'
    )
    assert manifest['style_range']
    for spec in manifest['style_range'].values():
        assert len(spec['low']) == len(spec['high'])


def test_config_hash_is_stable(env):
    registry, sampler = make(env)
    args = (env, registry, sampler, 10, 0, 'physical_content')
    assert (
        ident.build_manifest(*args)['config_hash']
        == ident.build_manifest(*args)['config_hash']
    )


def test_sampler_registry_width_mismatch_is_refused(env):
    """A silent mismatch here mislabels every row in the dataset."""
    registry, _ = make(env)
    wrong = OUSampler(n=registry.n + 1, rho=0.9, seed=0)
    with pytest.raises(ValueError, match='silently mislabels'):
        next(ident.collect_pairs(env, registry, wrong, 1, log_every=0))


# ------------------------------------------------------- the loader contract


def test_dataset_round_trips_through_the_stock_loader(env, tmp_path):
    """No new loader: ``num_steps=2, frameskip=1`` yields the positive pair."""
    registry, sampler = make(env)
    path = tmp_path / 'cube_single_ou.lance'

    with swm.data.LanceWriter(path, mode='overwrite') as writer:
        ident.write_chunked(
            writer,
            ident.collect_pairs(env, registry, sampler, 24, log_every=0),
            chunk_size=8,
        )

    dataset = swm.data.LanceDataset(
        path, num_steps=2, frameskip=1, keys_to_load=['pixels', 'latent/z']
    )
    assert len(dataset) > 0

    sample = dataset[0]
    pixels = np.asarray(sample['pixels'])
    assert pixels.shape[0] == 2, pixels.shape
    assert np.asarray(sample['latent/z']).shape == (2, registry.n)


def test_contrast_floor_removes_the_invisible_cube_tail(env):
    """cube.color and floor_rgb collide often enough to matter.

    Both are uniform on [0,1]^3 and drawn independently, so without a floor the
    cube sometimes renders the same colour as the floor behind it and vanishes
    -- and two different cube positions then produce the same image, which is a
    genuine injectivity failure rather than merely a hard example.
    """
    registry, _ = make(env, profile='physical_content')
    rng = np.random.default_rng(0)

    unconstrained = np.array(
        [
            ident._contrast(ident.sample_style(env, registry, rng, 0.0)[0])
            for _ in range(600)
        ]
    )
    assert (unconstrained < 0.2).mean() > 0.01, 'tail should be non-trivial'

    floored = np.array(
        [
            ident._contrast(ident.sample_style(env, registry, rng, 0.2)[0])
            for _ in range(600)
        ]
    )
    assert floored.min() >= 0.2
    # The floor removes a tail; it must not reshape the bulk.
    assert abs(floored.mean() - unconstrained.mean()) < 0.1


def test_manifest_records_the_contrast_floor_in_force(env):
    """A knob that reshapes the style distribution cannot be unrecorded."""
    registry, sampler = make(env, profile='physical_content')
    manifest = ident.build_manifest(
        env,
        registry,
        sampler,
        4,
        0,
        'physical_content',
        min_contrast=0.2,
    )
    assert manifest['contrast_floor_applies'] is True
    assert manifest['min_contrast'] == pytest.approx(0.2)

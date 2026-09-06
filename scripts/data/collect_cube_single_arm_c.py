"""Arms C1 / C2: encoder datasets built from trajectories, not designed states.

The control that decides whether designed data is *necessary* or merely
sufficient. Everything the OU pipeline does deliberately, this one does not:
no state resets, no per-frame appearance randomisation, a fixed camera, and
whatever marginal the physics happens to produce.

Why C2 exists
-------------
Comparing arm A against C1 alone confounds two changes at once: the latent
marginal stops being Gaussian **and** the pair correlation stops being ``rho``.
Either could explain a gap. C2 removes the second by choosing a frame stride
whose empirical per-dimension ``rho_alpha`` lands near the declared ``rho``, so
the A-to-C1 gap decomposes into a ``rho`` mismatch plus a marginal-shape
problem instead of being a single unexplained number.

Both arms **report the achieved ``rho_alpha`` vector** either way, which is
what makes that decomposition possible after the fact.

Usage::

    python scripts/data/collect_cube_single_arm_c.py arm=c1
    python scripts/data/collect_cube_single_arm_c.py arm=c2          # searches the stride
    python scripts/data/collect_cube_single_arm_c.py arm=c2 frame_stride=7
"""

import os
import sys
from pathlib import Path


if 'MUJOCO_GL' not in os.environ:
    if sys.platform == 'darwin' or os.environ.get('DISPLAY'):
        os.environ['MUJOCO_GL'] = 'glfw'
    else:
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger as logging  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data.format import EPISODE_DATA_KEY  # noqa: E402
from stable_worldmodel.envs.ogbench import ExpertPolicy  # noqa: E402
from stable_worldmodel.envs.ogbench.lejepa_cube_env import (  # noqa: E402
    LeJEPACubeEnv,
)
from stable_worldmodel.identifiability import collect as ident  # noqa: E402
from stable_worldmodel.identifiability.latents import (  # noqa: E402
    PROFILES,
    build_registry,
)


def latents_along_trajectory(env, registry, policy, episodes, max_steps, seed):
    """Roll the policy and record the content latents at every step.

    Physics *is* stepped here -- that is the whole point of arm C. The frames
    are whatever the trajectory visits, in whatever order it visits them.

    Returns:
        list: One ``(T, n)`` array of z-space latents per episode, plus the
        matching rendered frames.
    """
    trajectories = []
    for episode in range(episodes):
        env.reset(seed=seed + episode, options={'variation': ['all']})
        policy.reset() if hasattr(policy, 'reset') else None

        latents, frames = [], []
        for _ in range(max_steps):
            info = env.compute_ob_info()
            values = registry.read_info(info, num_cubes=env._num_cubes)
            if all(name in values for name in registry.slices()):
                latents.append(
                    registry.to_z(
                        {
                            name: values[name].reshape(1, -1)
                            for name in registry.slices()
                        }
                    )[0]
                )
                frames.append(env.render(camera='front_pixels').copy())

            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        if len(latents) > 1:
            trajectories.append(
                (np.stack(latents), np.stack(frames))
            )
    return trajectories


def achieved_rho(trajectories, stride):
    """Empirical per-dimension lag-``stride`` correlation across trajectories.

    The number arm C is judged on. Reported for every arm C dataset so the
    A-to-C gap can be attributed rather than merely observed.
    """
    firsts, seconds = [], []
    for latents, _ in trajectories:
        if len(latents) <= stride:
            continue
        firsts.append(latents[:-stride])
        seconds.append(latents[stride:])
    if not firsts:
        return None

    a = np.concatenate(firsts)
    b = np.concatenate(seconds)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denom = np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
    return np.where(denom > 0, (a * b).sum(axis=0) / denom, 0.0)


def search_stride(trajectories, target_rho, max_stride=40):
    """Pick the stride whose achieved ``rho`` is closest to the declared one.

    This is what makes C2 *stride-matched* rather than merely strided: the
    stride is not a guess, it is fitted to the same ``rho`` the OU sampler was
    given, so the only remaining difference from arm A is the marginal.

    Returns:
        tuple: ``(stride, achieved_rho_vector, table)``.
    """
    table = []
    best, best_error = 1, np.inf
    for stride in range(1, max_stride + 1):
        rho = achieved_rho(trajectories, stride)
        if rho is None:
            break
        error = float(abs(rho.mean() - target_rho))
        table.append(
            {
                'stride': stride,
                'rho_mean': float(rho.mean()),
                'rho_min': float(rho.min()),
                'rho_max': float(rho.max()),
                'error': error,
            }
        )
        if error < best_error:
            best, best_error = stride, error
    return best, achieved_rho(trajectories, best), table


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='ogb_cube_single_arm_c',
)
def run(cfg: DictConfig):
    """Collect one arm-C encoder dataset."""
    arm = str(cfg.arm).lower()
    if arm not in ('c1', 'c2'):
        raise ValueError(f"arm must be 'c1' or 'c2'; got {cfg.arm!r}.")

    env = LeJEPACubeEnv(
        env_type=cfg.env.env_type,
        ob_type='pixels',
        mode='data_collection',
        terminate_at_goal=False,
        visualize_info=False,
        width=cfg.env.image_size,
        height=cfg.env.image_size,
        num_digits=cfg.env.num_digits,
        marker_enabled=cfg.env.marker.enabled,
        marker_face=cfg.env.marker.face,
        marker_scale=cfg.env.marker.scale,
    )
    env.reset(seed=cfg.seed, options={'variation': ['all']})

    registry = build_registry(
        env,
        yaw_half_arc=float(cfg.latents.yaw_half_arc),
        include_roll_pitch=bool(cfg.latents.include_roll_pitch),
        include_discrete=bool(cfg.latents.include_discrete),
        pos_z_max=float(cfg.latents.pos_z_max),
    ).resolve(PROFILES[str(cfg.latents.profile)])

    policy = ExpertPolicy(
        policy_type=cfg.policy_type, p_stack=cfg.get('p_stack'), seed=cfg.seed
    )

    logging.info(f'arm {arm}: rolling {cfg.num_episodes} trajectories')
    trajectories = latents_along_trajectory(
        env, registry, policy, int(cfg.num_episodes),
        int(cfg.max_episode_steps), int(cfg.seed),
    )
    if not trajectories:
        raise RuntimeError('no usable trajectories were collected.')

    target = float(cfg.program_constants.rho)
    if arm == 'c1':
        stride = 1
        rho = achieved_rho(trajectories, stride)
        table = None
    elif cfg.get('frame_stride'):
        stride = int(cfg.frame_stride)
        rho = achieved_rho(trajectories, stride)
        table = None
    else:
        stride, rho, table = search_stride(trajectories, target)

    logging.info(
        f'arm {arm}: stride {stride}, achieved rho mean {rho.mean():.4f} '
        f'(declared {target:.4f}), per-dim range '
        f'[{rho.min():.4f}, {rho.max():.4f}]'
    )
    if arm == 'c1' and abs(rho.mean() - target) > 0.05:
        logging.warning(
            f'C1 rho ({rho.mean():.4f}) differs from the declared rho '
            f'({target:.4f}). The A->C1 gap therefore confounds a rho '
            'mismatch with a marginal-shape problem -- which is exactly what '
            'arm C2 is for. Run it before attributing the gap.'
        )

    # ------------------------------------------------------------- write
    out_path = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / str(cfg.dataset_name)
    )

    def episodes():
        for latents, frames in trajectories:
            for t in range(len(latents) - stride):
                yield {
                    'pixels': [frames[t], frames[t + stride]],
                    'latent/z': [
                        latents[t].astype(np.float32),
                        latents[t + stride].astype(np.float32),
                    ],
                    'latent/view': [
                        np.array([0], dtype=np.int64),
                        np.array([1], dtype=np.int64),
                    ],
                    EPISODE_DATA_KEY: {'stride': int(stride)},
                }

    with swm.data.LanceWriter(out_path, mode=cfg.write_mode) as writer:
        # `write_chunked` for the same reason the OU collector uses it: Lance
        # drains a streamed iterable from a background thread, and these
        # frames came off a GL context bound to this one.
        ident.write_chunked(writer, episodes())

    manifest = {
        'arm': arm,
        'stride': int(stride),
        'rho_achieved': rho.tolist(),
        'rho_declared': target,
        'rho_mean_achieved': float(rho.mean()),
        'stride_search': table,
        'latents': registry.describe(),
        'num_episodes': int(cfg.num_episodes),
        'seed': int(cfg.seed),
        'program_constants': OmegaConf.to_container(
            cfg.program_constants, resolve=True
        ),
        'note': 'trajectory-derived encoder data: no state resets, fixed '
        'appearance, natural marginal.',
    }
    manifest['config_hash'] = ident.config_hash(manifest)
    sidecar = ident.write_manifest(out_path, manifest)

    env.close()
    logging.success(f'🎉 arm {arm} dataset -> {out_path} (manifest {sidecar})')


if __name__ == '__main__':
    run()

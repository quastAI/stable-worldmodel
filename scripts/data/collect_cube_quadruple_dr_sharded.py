"""Run DR cube-quadruple collection as N parallel processes, then merge.

Collection is single-threaded: :class:`~stable_worldmodel.world.EnvPool` steps
its envs in a plain Python loop, so ``world.num_envs`` controls batching and
memory, not parallelism -- one collection process saturates one core and leaves
the rest of the machine idle. This script fans the episode budget across
``--shards`` independent processes and concatenates the results with
``swm merge``.

Each shard runs ``collect_cube_quadruple_dr.py shard=i num_shards=N``, which
takes an even slice of ``num_traj`` and a **disjoint block of episode seeds**,
and writes its own ``<name>_shard{i}.lance``. Shards never write to a shared
table, so there is no concurrent-writer hazard; the merge is a separate,
sequential pass at the end.

Measured on an M1 Max (10 cores): 3.47 s/episode serial, 1.22 s/episode at
``--shards 4`` -- 2.8x. Returns diminish past that (the render path contends),
and each process peaks near 6.7 GB at ``world.num_envs=10``, so lower
``world.num_envs`` rather than raising the shard count if memory is tight.

Usage::

    # 5000 episodes across 4 processes, merged into the default dataset name
    python scripts/data/collect_cube_quadruple_dr_sharded.py --shards 4

    # Overrides after `--` go to every shard verbatim
    python scripts/data/collect_cube_quadruple_dr_sharded.py --shards 4 \
        -- num_traj=1000 world.num_envs=5 world.max_episode_steps=500

    # Keep the per-shard tables around instead of deleting them after merge
    python scripts/data/collect_cube_quadruple_dr_sharded.py --shards 4 --keep-shards
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

COLLECT_SCRIPT = Path(__file__).with_name('collect_cube_quadruple_dr.py')
CONFIG = Path(__file__).parent / 'config' / 'ogb_cube_quadruple_dr.yaml'


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--shards',
        type=int,
        default=4,
        help='Number of collection processes (default: 4).',
    )
    parser.add_argument(
        '--keep-shards',
        action='store_true',
        help='Keep the per-shard datasets after merging.',
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Collect the shards but skip the merge step.',
    )
    parser.add_argument(
        '--hydra-dir',
        default='/tmp/hydra',
        help='Hydra sweep dir prefix, kept out of the repo (default: /tmp/hydra).',
    )
    return parser.parse_known_args()


def resolve(key: str, overrides: list[str], default):
    """Read ``key`` from the shard overrides, falling back to the YAML config."""
    prefix = f'{key}='
    for override in reversed(overrides):
        if override.startswith(prefix):
            return override[len(prefix) :]
    cfg = OmegaConf.load(CONFIG)
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def main() -> int:
    args, overrides = parse_args()
    if args.shards < 1:
        raise SystemExit('--shards must be >= 1')

    dataset_name = str(resolve('dataset_name', overrides, ''))
    num_traj = int(resolve('num_traj', overrides, 0))
    if args.shards > num_traj:
        raise SystemExit(
            f'--shards {args.shards} exceeds num_traj {num_traj}; '
            'some shards would have nothing to collect.'
        )

    stem, _, suffix = dataset_name.rpartition('.')
    shard_names = [f'{stem}_shard{i}.{suffix}' for i in range(args.shards)]

    procs = []
    start = time.monotonic()
    for i in range(args.shards):
        cmd = [
            sys.executable,
            str(COLLECT_SCRIPT),
            f'shard={i}',
            f'num_shards={args.shards}',
            f'hydra.sweep.dir={args.hydra_dir}{i}',
            *overrides,
        ]
        print(f'[shard {i}] {" ".join(cmd)}', flush=True)
        procs.append(subprocess.Popen(cmd))

    codes = [p.wait() for p in procs]
    elapsed = time.monotonic() - start
    failed = [i for i, code in enumerate(codes) if code != 0]
    if failed:
        # The surviving shards are complete datasets in their own right, so
        # leave them on disk: re-running only the failed indices is cheaper
        # than redoing the whole budget.
        print(
            f'shards {failed} failed (exit codes {codes}); '
            'skipping merge. Completed shards were left in place.',
            file=sys.stderr,
        )
        return 1

    print(
        f'All {args.shards} shards finished in {elapsed / 60:.1f} min '
        f'({elapsed / num_traj:.2f} s/episode).',
        flush=True,
    )

    if args.no_merge:
        print('--no-merge: shards left as', ', '.join(shard_names))
        return 0

    # `swm merge` renumbers episodes into one contiguous range and rejects a
    # column mismatch before writing anything.
    merge_target = stem  # `swm merge` appends the format suffix itself.
    merge = [
        'swm',
        'merge',
        *shard_names,
        '--output',
        merge_target,
        '--overwrite',
    ]
    print(f'[merge] {" ".join(merge)}', flush=True)
    if subprocess.run(merge).returncode != 0:
        print('merge failed; shards left in place.', file=sys.stderr)
        return 1

    if not args.keep_shards:
        from stable_worldmodel.data.utils import get_cache_dir

        root = Path(get_cache_dir()) / 'datasets'
        for name in shard_names:
            shutil.rmtree(root / name, ignore_errors=True)
        print(f'Removed {args.shards} shard datasets.')

    print(f'🎉 Merged {num_traj} episodes -> {merge_target}.{suffix}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

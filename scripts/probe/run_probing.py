"""Run the probing experiment end to end: extract features, fit read-outs.

    # one checkpoint, plus the untrained-encoder control
    python scripts/probe/run_probing.py \
        --checkpoint lewm_q4_dr/weights_epoch_11.pt \
        --with-random-init \
        --out $STABLEWM_HOME/probing/lewm_q4_dr

    # reuse the cached features (skips the encoder entirely)
    python scripts/probe/run_probing.py \
        --checkpoint lewm_q4_dr/weights_epoch_11.pt \
        --out $STABLEWM_HOME/probing/lewm_q4_dr --reuse-cache

    # representation quality over training
    python scripts/probe/run_probing.py \
        --checkpoint lewm_q4_dr/weights_epoch_{1,4,8,11}.pt \
        --groups state --probes baseline linear \
        --out $STABLEWM_HOME/probing/epoch_sweep

Writes ``results.json`` (rows + metadata + the feature-cache manifest) and
``results.csv`` per run directory, and one ``features_<tag>.npz`` per
configuration. The notebook ``scripts/notebooks/probe_lewm_ogbcubedr.ipynb``
drives the same functions interactively.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

# Sibling modules; `python scripts/probe/run_probing.py` already puts this
# directory first on sys.path, but an explicit insert also covers `-m` and
# notebook use.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as ft  # noqa: E402
import fit as fitting  # noqa: E402
import targets as tg  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    grp = p.add_argument_group('what to probe')
    grp.add_argument(
        '--checkpoint',
        nargs='+',
        required=True,
        help='One or more checkpoints relative to <root>/checkpoints/, '
        'e.g. lewm_q4_dr/weights_epoch_11.pt',
    )
    grp.add_argument(
        '--with-random-init',
        action='store_true',
        help='Also probe an untrained encoder of the same architecture — '
        'the control that says how much of the score is the ViT prior.',
    )
    grp.add_argument(
        '--checkpoint-root',
        default=None,
        help='Overrides STABLEWM_HOME for the checkpoint lookup.',
    )
    grp.add_argument(
        '--dataset',
        default='ogbench/cube_quadruple_dr_expert.lance',
        help='Dataset name or path.',
    )
    grp.add_argument(
        '--dataset-root',
        default=None,
        help='Overrides STABLEWM_HOME for the dataset lookup.',
    )

    grp = p.add_argument_group('targets and features')
    grp.add_argument('--targets', nargs='+', default=None)
    grp.add_argument(
        '--groups', nargs='+', default=None, choices=list(tg.GROUPS)
    )
    grp.add_argument(
        '--variants', nargs='+', default=list(ft.DEFAULT_VARIANTS)
    )
    grp.add_argument(
        '--probes',
        nargs='+',
        default=['baseline', 'linear', 'mlp'],
        choices=['baseline', 'linear', 'mlp'],
    )

    grp = p.add_argument_group('sampling')
    # Episodes, not windows, are the unit that matters: every episode
    # re-draws lighting, camera angle, cube colours and materials, so
    # episode-level appearance dominates the features. Below ~1000 training
    # episodes a linear probe fits episode identity and every within-episode
    # target reads R2 ~ 0 on held-out episodes. Raise --train-episodes before
    # raising --windows-per-episode, which adds correlated samples.
    grp.add_argument('--train-episodes', type=int, default=1000)
    grp.add_argument('--val-episodes', type=int, default=150)
    grp.add_argument('--test-episodes', type=int, default=250)
    grp.add_argument('--windows-per-episode', type=int, default=20)
    grp.add_argument('--seed', type=int, default=0)

    grp = p.add_argument_group('window layout (must match training)')
    grp.add_argument('--history-size', type=int, default=3)
    grp.add_argument('--num-preds', type=int, default=1)
    grp.add_argument('--frameskip', type=int, default=5)
    grp.add_argument('--img-size', type=int, default=224)

    grp = p.add_argument_group('compute')
    grp.add_argument('--batch-size', type=int, default=64)
    grp.add_argument('--num-workers', type=int, default=4)
    grp.add_argument('--device', default=None)
    grp.add_argument(
        '--dtype', default='float32', choices=['float32', 'bfloat16']
    )

    grp = p.add_argument_group('probe hyperparameters')
    grp.add_argument('--mlp-hidden-dim', type=int, default=512)
    grp.add_argument('--mlp-layers', type=int, default=1)
    grp.add_argument('--mlp-dropout', type=float, default=0.0)
    grp.add_argument('--probe-epochs', type=int, default=200)
    grp.add_argument('--probe-patience', type=int, default=25)
    grp.add_argument('--probe-lr', type=float, default=3e-3)

    grp = p.add_argument_group('output')
    grp.add_argument('--out', required=True, help='Run directory.')
    grp.add_argument(
        '--reuse-cache',
        action='store_true',
        help='Load features_<tag>.npz instead of re-encoding, when present.',
    )
    grp.add_argument(
        '--save-probes',
        action='store_true',
        help='Also torch.save the fitted probes under <out>/probes/.',
    )
    return p


def run_tag(checkpoint: str, random_init: bool) -> str:
    """Filesystem-safe label for one (checkpoint, control) configuration."""
    stem = re.sub(r'[^A-Za-z0-9]+', '_', checkpoint.replace('.pt', '')).strip(
        '_'
    )
    return f'{stem}__randinit' if random_init else stem


def configs_from_args(args):
    """Every (checkpoint, random_init) configuration the run covers."""
    for checkpoint in args.checkpoint:
        yield checkpoint, False
        if args.with_random_init:
            yield checkpoint, True


def extract_config(args, checkpoint: str, random_init: bool):
    return ft.ExtractConfig(
        dataset_name=args.dataset,
        checkpoint=checkpoint,
        checkpoint_root=args.checkpoint_root,
        dataset_cache_dir=args.dataset_root,
        random_init=random_init,
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        img_size=args.img_size,
        episodes={
            'train': args.train_episodes,
            'val': args.val_episodes,
            'test': args.test_episodes,
        },
        windows_per_episode=args.windows_per_episode,
        variants=tuple(args.variants),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dtype=args.dtype,
    )


def fit_config(args):
    return fitting.FitConfig(
        probes=tuple(args.probes),
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        mlp_dropout=args.mlp_dropout,
        epochs=args.probe_epochs,
        patience=args.probe_patience,
        lr=args.probe_lr,
        device=args.device,
        seed=args.seed,
    )


def get_features(cfg, label_columns, cache_path: Path, reuse: bool):
    """Load the cache when asked and available, otherwise encode and save."""
    if reuse and cache_path.exists():
        payload = ft.load_features(cache_path)
        missing = [
            c
            for c in label_columns
            if c not in payload['meta']['label_columns']
        ]
        if missing:
            raise ValueError(
                f'{cache_path.name} lacks label columns {missing} — drop '
                '--reuse-cache (or delete the file) to re-extract.'
            )
        print(f'Reusing {cache_path}')
        return payload

    payload = ft.extract(cfg, label_columns)
    ft.save_features(cache_path, payload)
    return payload


def summarize(rows, probe_targets) -> str:
    """A compact table: score per (target, variant) for each rung."""
    from tabulate import tabulate

    variants, rungs = [], []
    for row in rows:
        if row['variant'] not in variants:
            variants.append(row['variant'])
        if row['probe'] not in rungs:
            rungs.append(row['probe'])

    lookup = {
        (r['run'], r['variant'], r['target'], r['probe']): r['score']
        for r in rows
    }
    runs = list(dict.fromkeys(r['run'] for r in rows))

    table, headers = [], ['run', 'target', 'group', 'metric']
    headers += [f'{v}/{p}' for v in variants for p in rungs]
    for run in runs:
        for target in probe_targets:
            metric = 'acc' if target.kind == 'classification' else 'R2'
            line = [run, target.name, target.group, metric]
            for variant in variants:
                for rung in rungs:
                    score = lookup.get((run, variant, target.name, rung))
                    line.append('-' if score is None else f'{score:.3f}')
            table.append(line)
    return tabulate(table, headers=headers, tablefmt='github')


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    probe_targets = tg.select_targets(args.targets, args.groups)
    label_columns = tg.required_columns(probe_targets)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'{len(probe_targets)} targets, {len(args.variants)} feature '
        f'variants, {len(args.probes)} probe rungs'
    )

    all_rows: list[dict] = []
    manifests: dict[str, dict] = {}
    fidelity: dict[str, dict] = {}

    for checkpoint, random_init in configs_from_args(args):
        tag = run_tag(checkpoint, random_init)
        print(f'\n=== {tag} ===')

        cfg = extract_config(args, checkpoint, random_init)
        payload = get_features(
            cfg,
            label_columns,
            out_dir / f'features_{tag}.npz',
            args.reuse_cache,
        )
        manifests[tag] = payload['meta']
        fidelity[tag] = fitting.prediction_fidelity(payload)

        rows, probes = fitting.fit_all(
            payload,
            probe_targets,
            fit_config(args),
            variants=args.variants,
            keep_probes=args.save_probes,
        )
        for row in rows:
            row['run'] = tag
            row['checkpoint'] = checkpoint
            row['random_init'] = random_init
        all_rows.extend(rows)

        if args.save_probes:
            import torch

            probe_dir = out_dir / 'probes' / tag
            probe_dir.mkdir(parents=True, exist_ok=True)
            for (variant, target, rung), probe in probes.items():
                torch.save(
                    probe, probe_dir / f'{variant}__{target}__{rung}.pt'
                )

    payload = {
        'args': vars(args),
        'targets': [asdict(t) for t in probe_targets],
        'manifests': manifests,
        'prediction_fidelity': fidelity,
        'rows': all_rows,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(payload, f, indent=2)
    _write_csv(out_dir / 'results.csv', all_rows)

    print()
    print(summarize(all_rows, probe_targets))
    print(f'\nWrote {out_dir / "results.json"} and results.csv')
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()
                }
            )


if __name__ == '__main__':
    # Lance opens its own thread pool; keep the DataLoader workers from
    # oversubscribing the box on a many-core node.
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    raise SystemExit(main())

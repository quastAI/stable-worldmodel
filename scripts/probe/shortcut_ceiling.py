"""How much of the GCIDM plan target is predictable *without* cube state?

GCIDM's bet is that regressing the action-block plan between an observation
and a goal observation forces object state into the latent. There is a
shortcut that would defeat it. The oracle's action is literally an effector
delta::

    action[:3] = plan[i][:3] - proprio/effector_pos
    action[3]  = plan[i][3]  - proprio/effector_yaw
    action[4]  = plan[i][4]  - proprio/gripper_opening

so the *sum* of the predicted blocks is roughly (effector pose at the goal) -
(effector pose now) -- computable from gripper features alone, with no idea
where any cube is. What genuinely needs cube state is the path *curvature*
(the oracle approaches above a cube, descends, grasps, lifts, routes through
a clearance waypoint) and the *timing* of the grasp. Both are real, both are
a minority of the label variance.

This script measures the two ceilings that bracket the experiment, using
ground-truth state rather than any learned encoder -- so it runs in minutes,
on CPU, with no checkpoint:

``proprio``
    Every ``proprio/*`` column: joint angles and velocities, effector pose,
    gripper opening/velocity/contact. Everything about the robot and nothing
    about the cubes. Its R2 is the **shortcut ceiling**.
``proprio_cubes``
    The same columns plus the ground-truth block poses -- a **strict
    superset**, which is what makes the difference interpretable. Its R2 is
    the **information ceiling**.

The pair is deliberately built from raw columns rather than from the env's
own ``observation`` vector. ``observation`` is *not* a concatenation of these
columns: only ``proprio/joint_pos`` appears in it verbatim, while effector
and block positions are in some other encoding, so it is nested with neither
set and differences against it would not mean "what cube state adds". It is
still available as ``--features all`` for a cross-check.

The gap between the nested pair is the entire headroom the objective could
ever buy at this horizon, and it is the go/no-go:

  * **gap < ~0.05** -- a goal-conditioned action objective cannot force cube
    encoding at this horizon whatever the encoder does, so training GCIDM at
    this ``goal_horizon`` would be uninformative. Sweep ``--goal-horizon``
    (see below) and use the offset where the gap is widest.
  * **gap large** -- the experiment is well posed. Record both numbers: once
    GCIDM trains, ``train/policy_loss`` sits on the same axis (labels are
    z-scored, so ``R2 ~= 1 - policy_loss``) and its position between the two
    ceilings says how much of the available cube information the encoder
    actually picked up.

The gap is compared **within** a rung, never across rungs: the superset can
only beat the subset at equal fit quality, so a negative per-rung gap means
that rung underfit the wider feature set (more inputs, same probe budget)
rather than that cube state hurts.

Second job -- picking the long-horizon fallback empirically::

    for H in 5 10 20; do
      python scripts/probe/shortcut_ceiling.py --goal-horizon $H \
             --out runs/shortcut_ceiling_h$H
    done

The fitting is the same three-rung protocol as ``run_probing.py``
(``baseline`` / ``linear`` closed-form ridge / ``mlp``), with the same
episode-level splits, train-statistic whitening and single test-split touch.
The only inversion is which side is which: here the *features* are
ground-truth state and the *target* is the action plan.

Usage::

    python scripts/probe/shortcut_ceiling.py --out runs/shortcut_ceiling_h5
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

import fit as fitting
import targets as tg
from features import SPLITS, episode_split


# The plan target. Features/labels are built here rather than by
# `features.extract` (which needs a checkpoint), but the target goes through
# the same `build_labels` -> reducer path as every other probing target.
PLAN_COLUMN = 'action_plan'
PLAN_TARGET = tg.ProbeTarget(
    name=PLAN_COLUMN,
    columns=(PLAN_COLUMN,),
    kind='regression',
    reduce='concat',
    group='state',
    units='normalized action',
    note=(
        'the action-block plan joining the current frame to the goal frame; '
        'GCIDM regresses exactly this from latents'
    ),
)

PROPRIO_COLUMNS = (
    'proprio/joint_pos',
    'proprio/joint_vel',
    'proprio/effector_pos',
    'proprio/effector_yaw',
    'proprio/gripper_opening',
    'proprio/gripper_vel',
    'proprio/gripper_contact',
)

CUBE_COLUMNS = tuple(
    f'privileged/block_{i}_{field}'
    for i in range(4)
    for field in ('pos', 'quat', 'yaw')
)

FEATURE_SETS = {
    # Robot-only. Deliberately generous -- it gets joint angles AND
    # velocities, so a weak score cannot be blamed on missing
    # proprioception. 19 dims per step.
    'proprio': PROPRIO_COLUMNS,
    # Robot + ground-truth cube poses: 51 dims per step, and a *strict
    # superset* of `proprio` by construction. This nesting is what makes the
    # gap interpretable, and it is why the pair is built from raw columns
    # rather than from `observation` -- see below.
    'proprio_cubes': PROPRIO_COLUMNS + CUBE_COLUMNS,
    # The env's own 55-dim state vector, as a cross-check. NOTE it is *not* a
    # concatenation of the columns above: only `proprio/joint_pos` appears
    # verbatim (obs[0:6]); effector and block positions are in some other
    # encoding. So `full_state` is not nested with either set above and its
    # score must not be differenced against them -- it is here to confirm the
    # nested pair, not to define the ceiling.
    'full_state': ('observation',),
}

# The two sets the go/no-go verdict is computed from: nested, so their
# difference is exactly "what ground-truth cube state adds".
SHORTCUT_SET, CEILING_SET = 'proprio', 'proprio_cubes'


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    grp = p.add_argument_group('what to measure')
    grp.add_argument(
        '--features',
        nargs='+',
        default=[SHORTCUT_SET, CEILING_SET],
        choices=[*FEATURE_SETS, 'all'],
        help=(
            'Feature sets to fit. Default is the nested pair the verdict '
            "needs; 'all' adds the `observation` cross-check."
        ),
    )
    grp.add_argument(
        '--probes',
        nargs='+',
        default=['baseline', 'linear', 'mlp'],
        choices=['baseline', 'linear', 'mlp'],
    )

    grp = p.add_argument_group('plan geometry (must match the train config)')
    grp.add_argument(
        '--goal-horizon',
        type=int,
        default=5,
        help='Action blocks predicted, i.e. wm.goal_horizon (default 5).',
    )
    grp.add_argument(
        '--frameskip',
        type=int,
        default=5,
        help='Env steps per action block, i.e. data.dataset.frameskip.',
    )
    grp.add_argument(
        '--context-frames',
        type=int,
        default=3,
        help=(
            'Context frames the features see, i.e. wm.history_size. Matches '
            'what the policy head gets so the ceiling is not handicapped.'
        ),
    )

    grp = p.add_argument_group('sampling')
    grp.add_argument(
        '--dataset',
        default='ogbench/cube_quadruple_dr_expert.lance',
    )
    grp.add_argument('--cache-dir', default=None)
    grp.add_argument('--train-episodes', type=int, default=1000)
    grp.add_argument('--val-episodes', type=int, default=150)
    grp.add_argument('--test-episodes', type=int, default=250)
    grp.add_argument('--windows-per-episode', type=int, default=20)
    grp.add_argument('--seed', type=int, default=0)

    grp = p.add_argument_group('probe hyperparameters')
    grp.add_argument(
        '--mlp-hidden-dim',
        type=int,
        default=768,
        help='Matches the shipped GoalPolicyHead width, so the mlp rung is '
        'a fair ceiling rather than an underfit one.',
    )
    grp.add_argument('--mlp-layers', type=int, default=2)
    grp.add_argument('--probe-epochs', type=int, default=200)
    grp.add_argument('--probe-patience', type=int, default=25)
    grp.add_argument('--probe-lr', type=float, default=3e-3)
    grp.add_argument('--device', default=None)

    p.add_argument('--out', required=True, help='Run directory.')
    return p


def resolve_feature_sets(names) -> list[str]:
    if 'all' in names:
        return list(FEATURE_SETS)
    # dedupe, preserve order
    return list(dict.fromkeys(names))


def load_columns(dataset, columns):
    """Whole-column reads, as ``(rows, dim)`` float32 arrays."""
    out = {}
    for col in columns:
        arr = np.asarray(dataset.get_col_data(col), dtype=np.float32)
        out[col] = arr.reshape(len(arr), -1)
    return out


def sample_windows(
    lengths, episodes, per_episode, span_after, span_before, seed
):
    """Draw window anchors inside the given episodes.

    The anchor ``t0`` is the *current* frame's local step. A window is valid
    when its context reaches back ``span_before`` steps and its goal frame
    (plus the action blocks up to it) reaches forward ``span_after`` steps.

    Returns ``(episode_idx, t0)``, sorted by episode for read locality.
    """
    rng = np.random.default_rng(seed)
    eps, anchors = [], []

    for ep in episodes:
        ep = int(ep)
        lo = span_before
        hi = int(lengths[ep]) - 1 - span_after
        if hi < lo:
            continue
        take = min(per_episode, hi - lo + 1)
        chosen = rng.choice(np.arange(lo, hi + 1), size=take, replace=False)
        eps.append(np.full(take, ep, dtype=np.int64))
        anchors.append(chosen.astype(np.int64))

    if not anchors:
        raise ValueError(
            'no valid windows in the requested episodes -- goal_horizon '
            '* frameskip may exceed the episode length'
        )

    eps = np.concatenate(eps)
    anchors = np.concatenate(anchors)
    order = np.lexsort((anchors, eps))
    return eps[order], anchors[order]


def build_payload(dataset, cols, feature_columns, args, split_episodes):
    """Assemble a `fit_all`-shaped payload with state features and a plan
    target, entirely from cached numeric columns (no pixels, no model)."""
    fs = args.frameskip
    horizon = args.goal_horizon
    n_ctx = args.context_frames

    span_before = (n_ctx - 1) * fs
    span_after = horizon * fs
    # context steps relative to the anchor, oldest first, then the goal step
    ctx_offsets = [(-(n_ctx - 1) + k) * fs for k in range(n_ctx)]
    goal_offset = horizon * fs

    offsets = dataset.offsets
    action = cols['action']
    act_dim = action.shape[1]

    payload = {
        'features': {},
        'labels': {s: {} for s in SPLITS},
        'windows': {s: {} for s in SPLITS},
        'meta': {
            'source': 'shortcut_ceiling',
            'dataset_name': args.dataset,
            'feature_columns': list(feature_columns),
            'goal_horizon': horizon,
            'frameskip': fs,
            'context_frames': n_ctx,
            'plan_dim': horizon * fs * act_dim,
            'label_columns': [PLAN_COLUMN],
            'seed': args.seed,
        },
    }

    for split in SPLITS:
        eps, t0 = sample_windows(
            dataset.lengths,
            split_episodes[split],
            args.windows_per_episode,
            span_after,
            span_before,
            # distinct stream per split; the episode sets are already disjoint
            seed=args.seed + SPLITS.index(split),
        )
        base = offsets[eps] + t0  # global row of the current frame

        # -- features: state at each context step, plus the goal step
        parts = []
        for off in (*ctx_offsets, goal_offset):
            rows = base + off
            parts.extend(cols[c][rows] for c in feature_columns)
        payload['features'][split] = np.concatenate(parts, axis=1).astype(
            np.float32
        )

        # -- label: the action blocks from the current frame to the goal.
        # `action[k]` is the block leaving frame k, so the plan is the raw
        # actions on [t0, t0 + horizon * frameskip), grouped per block.
        step = np.arange(goal_offset)
        rows = base[:, None] + step[None, :]
        plan = action[rows.reshape(-1)].reshape(
            len(base), horizon, fs * act_dim
        )
        # (N, 1, D): build_labels slices a timestep axis
        payload['labels'][split][PLAN_COLUMN] = plan.reshape(
            len(base), 1, horizon * fs * act_dim
        )

        payload['windows'][split]['episode_idx'] = eps
        payload['windows'][split]['frame_idx'] = t0

    return payload


def fit_config(args):
    return fitting.FitConfig(
        probes=tuple(args.probes),
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_layers=args.mlp_layers,
        epochs=args.probe_epochs,
        patience=args.probe_patience,
        lr=args.probe_lr,
        device=args.device,
        seed=args.seed,
    )


def summarize(rows, feature_sets) -> str:
    from tabulate import tabulate

    rungs = list(dict.fromkeys(r['probe'] for r in rows))
    lookup = {(r['run'], r['probe']): r['score'] for r in rows}

    table = []
    for name in feature_sets:
        line = [name]
        for rung in rungs:
            score = lookup.get((name, rung))
            line.append('-' if score is None else f'{score:.3f}')
        table.append(line)
    return tabulate(table, headers=['features', *rungs], tablefmt='github')


def verdict(rows, gap_threshold=0.05) -> str:
    """The go/no-go: how much R2 ground-truth cube state adds over
    proprioception alone.

    Compared **per rung**, never across rungs. ``proprio_cubes`` is a strict
    superset of ``proprio``, so within a rung its score can only be higher if
    the extra columns carry usable information -- and a *negative* per-rung
    gap therefore says that rung underfit the wider feature set (more inputs,
    same probe budget), not that cube state hurts. Differencing the best
    scores across *different* rungs would silently mix those two effects, so
    the headline gap is the best gap achieved at any single rung: the most
    generous honest reading, which is the right bias for a go/no-go that can
    cancel an experiment.
    """
    scores = {
        (row['run'], row['probe']): row['score']
        for row in rows
        if row['probe'] != 'baseline'
    }
    rungs = list(dict.fromkeys(p for _, p in scores))
    have = {run for run, _ in scores}

    if not {SHORTCUT_SET, CEILING_SET} <= have:
        return (
            f'Need --features {SHORTCUT_SET} {CEILING_SET} for the go/no-go '
            'verdict; got ' + ', '.join(sorted(have))
        )

    lines, gaps = ['per-rung gap (cube state minus proprio only):'], {}
    for rung in rungs:
        lo = scores.get((SHORTCUT_SET, rung))
        hi = scores.get((CEILING_SET, rung))
        if lo is None or hi is None:
            continue
        gaps[rung] = hi - lo
        flag = '   <- underfit at this rung' if hi < lo else ''
        lines.append(
            f'  {rung:8s} {lo:.3f} -> {hi:.3f}   gap {hi - lo:+.3f}{flag}'
        )

    if not gaps:
        return 'No rung produced both feature sets; nothing to compare.'

    best_rung = max(gaps, key=gaps.get)
    gap = gaps[best_rung]
    shortcut = scores[(SHORTCUT_SET, best_rung)]
    ceiling = scores[(CEILING_SET, best_rung)]

    lines += [
        '',
        f'headroom cube state can buy = {gap:+.3f} (best rung: {best_rung})',
        '',
    ]

    if all(g < 0 for g in gaps.values()):
        return '\n'.join(
            lines
            + [
                'INCONCLUSIVE: every rung scored the superset BELOW the',
                'subset, which cannot be a property of the data. The wider',
                'feature set is underfit everywhere. Raise --train-episodes',
                '/ --windows-per-episode (the shipped defaults give ~20k',
                'train windows) and re-run.',
            ]
        )

    if gap < gap_threshold:
        lines += [
            f'VERDICT: NO-GO at this horizon (gap {gap:+.3f} < '
            f'{gap_threshold}).',
            'Proprioception alone already explains this plan, so no encoder',
            'can be pushed to represent cube state by this objective at this',
            'horizon. Sweep --goal-horizon and use the widest-gap offset for',
            'the long-horizon shaping head instead of training at this one.',
        ]
    else:
        lines += [
            f'VERDICT: GO (gap {gap:+.3f}). The experiment is well posed.',
            'Once GCIDM trains, read train/policy_loss as R2 ~= 1 - loss:',
            f'  ~{1 - shortcut:.3f} loss means it took the gripper shortcut,',
            f'  ~{1 - ceiling:.3f} loss means it used all available cube info.',
        ]
    return '\n'.join(lines)


def main(argv=None) -> int:
    import stable_worldmodel as swm

    args = build_parser().parse_args(argv)
    feature_sets = resolve_feature_sets(args.features)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = {'action'}
    for name in feature_sets:
        needed.update(FEATURE_SETS[name])

    print(f'Loading {args.dataset} (columns only, no pixels)')
    # num_steps=1 / frameskip=1: every clip index is one frame. The window
    # layout is done here by arithmetic on whole columns, which is far
    # cheaper than decoding 8 frames per sample through __getitem__.
    dataset = swm.data.load_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        transform=None,
        num_steps=1,
        frameskip=1,
        keys_to_load=sorted(needed),
    )
    cols = load_columns(dataset, sorted(needed))

    counts = {
        'train': args.train_episodes,
        'val': args.val_episodes,
        'test': args.test_episodes,
    }
    split_episodes = episode_split(len(dataset.lengths), counts, args.seed)

    plan_blocks = args.goal_horizon
    print(
        f'plan = {plan_blocks} blocks x {args.frameskip} steps '
        f'x {cols["action"].shape[1]} dims '
        f'= {plan_blocks * args.frameskip * cols["action"].shape[1]} outputs, '
        f'goal at +{plan_blocks * args.frameskip} env steps'
    )

    all_rows: list[dict] = []
    manifests: dict[str, dict] = {}

    for name in feature_sets:
        print(f'\n=== {name} ===')
        payload = build_payload(
            dataset, cols, FEATURE_SETS[name], args, split_episodes
        )
        manifests[name] = payload['meta']
        print(
            f'features {payload["features"]["train"].shape[1]} dims, '
            f'{len(payload["features"]["train"])} train windows'
        )

        rows, _ = fitting.fit_all(payload, [PLAN_TARGET], fit_config(args))
        for row in rows:
            row['run'] = name
        all_rows.extend(rows)

    result = {
        'args': vars(args),
        'targets': [asdict(PLAN_TARGET)],
        'manifests': manifests,
        'rows': all_rows,
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)

    print()
    print(summarize(all_rows, feature_sets))
    print()
    print(verdict(all_rows))
    print(f'\nWrote {out_dir / "results.json"}')
    return 0


if __name__ == '__main__':
    # Lance opens its own thread pool; keep it from oversubscribing the box.
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    raise SystemExit(main())

"""The global scatter: one append-only row per (arm x violation x severity x seed).

**Written from the very first run.** The plan is explicit that retrofitting
this is not possible, and the reason is worth stating plainly: several of the
columns cannot be recovered after the fact. The per-dimension ``rho`` vector,
the isotropy margin, the program constants in force, and the dataset's
``config_hash`` all describe a run's *configuration*, and a run that did not
record them cannot be told apart later from one that was configured
differently. A scatter with those columns missing for early rows is not a
partial scatter -- it is a scatter whose early rows are uninterpretable.

Both axes, twice
----------------
Every row carries **two y-axes** and **two x-axes**, because the pairs
disagree and the disagreement is the measurement:

* ``sr_over_o`` -- success relative to the oracle (perfect representation *and*
  perfect dynamics). The absolute ceiling.
* ``sr_over_p`` -- success relative to arm P (perfect representation, *learned*
  dynamics). Isolates the representation from the transition model.
* ``measured_recovery_error`` -- what the encoder actually did.
* ``predicted_error`` -- what the theory said it would do, ``D + (eps + D)^2``.

Reporting only one of each pair would make an ordinary predictor-capacity
shortfall indistinguishable from an identifiability failure.
"""

import json
import os
from pathlib import Path

import numpy as np


#: Bumped when a column's *meaning* changes. Recorded per row, so a schema
#: change is visible rather than silently mixing incompatible rows.
SCATTER_SCHEMA_VERSION = '1.0.0'

#: Columns every row must carry. A row missing any of these is refused rather
#: than written, because a partially-populated scatter is worse than none: the
#: gaps look like data.
REQUIRED_COLUMNS = (
    'arm',
    'violation',
    'severity',
    'seed',
    'distribution',
    'config_hash',
    'metric_suite_version',
)


def _jsonable(value):
    """Coerce numpy scalars/arrays into something json and parquet accept."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def build_row(
    arm,
    violation,
    severity,
    seed,
    distribution,
    metrics,
    manifest,
    program_constants,
    success_rate=None,
    success_rate_oracle=None,
    success_rate_state=None,
    deviations=None,
    **extra,
):
    """Assemble one scatter row.

    Args:
        arm: ``'A'``, ``'R'``, ``'P'``, ``'O'``, ``'C1'`` or ``'C2'``.
        violation: Violation key, or ``'none'``.
        severity: Severity in ``[0, 1]``.
        seed: Run seed.
        distribution: ``'ou'`` or ``'rollout'`` -- **every metric is reported
            on both**, and the gap between them is a logged quantity rather
            than something to be asserted away.
        metrics: Output of
            :func:`~stable_worldmodel.identifiability.metrics.compute_all`.
        manifest: The dataset manifest, for ``config_hash``, the per-dimension
            ``rho`` vector and the isotropy margin.
        program_constants: The frozen ``rho`` / ``lambda`` / seed budget,
            copied verbatim so a deviating run is a visible event.
        success_rate: This arm's success rate.
        success_rate_oracle: ``SR(O)``, for the first y-axis.
        success_rate_state: ``SR(P)``, for the second.
        deviations: Any departure from the fixed protocol -- e.g. arm O run at
            a reduced ``num_samples``.
        **extra: Additional columns.

    Returns:
        dict: One row.
    """
    ou = manifest.get('ou', {})
    rho = np.atleast_1d(np.asarray(ou.get('rho', [np.nan]), dtype=float))
    isotropy = ou.get('isotropy', {})

    row = {
        'schema_version': SCATTER_SCHEMA_VERSION,
        'arm': arm,
        'violation': violation,
        'severity': float(severity),
        'seed': int(seed),
        'distribution': distribution,
        # --- both y-axes ---
        'success_rate': success_rate,
        'success_rate_oracle': success_rate_oracle,
        'success_rate_state': success_rate_state,
        'sr_over_o': _ratio(success_rate, success_rate_oracle),
        'sr_over_p': _ratio(success_rate, success_rate_state),
        # --- both x-axes ---
        'measured_recovery_error': metrics.get('procrustes_mse_per_dim'),
        'orth_err_normalized': metrics.get('orth_err_normalized'),
        'cond': metrics.get('cond'),
        'predicted_error': metrics.get('predicted_error'),
        'epsilon': metrics.get('epsilon'),
        'delta': metrics.get('delta'),
        'D': metrics.get('D'),
        'spectral_gap': metrics.get('spectral_gap'),
        'recovery_in_gap_units': metrics.get('recovery_in_gap_units'),
        # --- the decoy annotation ---
        'probe_linear_r2': metrics.get('probe_linear_r2'),
        'probe_mlp_r2': metrics.get('probe_mlp_r2'),
        'probe_divergence': metrics.get('probe_divergence'),
        # --- diagnostics ---
        'r2_z_to_h': metrics.get('r2_z_to_h'),
        'r2_h_to_z': metrics.get('r2_h_to_z'),
        'monotone_mse_per_dim': metrics.get('monotone_mse_per_dim'),
        'mcc_unaligned': metrics.get('mcc_unaligned'),
        'hermite2_excess': metrics.get('hermite2_excess'),
        'sigreg_z': metrics.get('sigreg_z'),
        'style_sensitivity': metrics.get('style_sensitivity'),
        # --- configuration that cannot be recovered later ---
        'rho_per_dim': rho.tolist(),
        'rho_mean': float(np.mean(rho)),
        'anisotropic': bool(metrics.get('anisotropic', False)),
        'isotropy_satisfied': isotropy.get('satisfied'),
        'isotropy_margin': isotropy.get('margin'),
        'n': manifest.get('latents', {}).get('n'),
        'profile': manifest.get('profile'),
        'config_hash': manifest.get('config_hash'),
        'metric_suite_version': metrics.get('metric_suite_version'),
        'program_constants': json.dumps(
            _jsonable(program_constants), sort_keys=True
        ),
        'deviations': json.dumps(_jsonable(deviations or {}), sort_keys=True),
    }
    row.update({k: _jsonable(v) for k, v in extra.items()})

    missing = [c for c in REQUIRED_COLUMNS if row.get(c) is None]
    if missing:
        raise ValueError(
            f'scatter row is missing required columns {missing}. A row with '
            'gaps is worse than no row: the gaps look like data.'
        )
    return {k: _jsonable(v) for k, v in row.items()}


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def append_rows(path, rows):
    """Append rows to the scatter, creating it if absent.

    Append-only by construction: existing rows are read, the new ones are
    concatenated, and the file is replaced atomically. Rewriting history is
    never a supported operation, because a scatter that can be edited is a
    scatter whose earlier rows cannot be trusted.

    Parquet when ``pyarrow`` is available, JSONL otherwise -- the schema is the
    same either way and the JSONL fallback keeps the reporting path from
    depending on an optional install.

    Args:
        path: Destination. Suffix decides the format.
        rows: Iterable of rows from :func:`build_row`.

    Returns:
        int: Total rows in the file after the append.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        return count_rows(path)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == '.parquet':
        import pandas as pd

        frame = pd.DataFrame(rows)
        if path.exists():
            frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
        tmp = path.with_suffix('.parquet.tmp')
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return len(frame)

    with open(path, 'a') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    return count_rows(path)


def count_rows(path):
    """How many rows the scatter currently holds."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.suffix == '.parquet':
        import pandas as pd

        return len(pd.read_parquet(path))
    with open(path) as handle:
        return sum(1 for line in handle if line.strip())


def read_rows(path):
    """Read the scatter back as a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == '.parquet':
        import pandas as pd

        return pd.read_parquet(path).to_dict('records')
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def violation_cost_table(rows, y_axis='sr_over_o'):
    """One degradation curve per violation -- the plan's SS7 cost table.

    This is the artifact Env 2+ predictions are checked against, so it is
    built from the scatter rather than maintained separately: a cost model
    that can drift from the rows it summarises is not a cost model.

    Args:
        rows: Rows from :func:`read_rows`.
        y_axis: ``'sr_over_o'`` or ``'sr_over_p'``.

    Returns:
        dict: ``violation -> [{severity, mean, std, n_seeds, gap_units}]``.
    """
    table = {}
    for row in rows:
        table.setdefault(row['violation'], {}).setdefault(
            row['severity'], []
        ).append(row)

    out = {}
    for violation, by_severity in table.items():
        curve = []
        for severity in sorted(by_severity):
            group = by_severity[severity]
            values = [
                r[y_axis] for r in group if r.get(y_axis) is not None
            ]
            gap = [
                r['recovery_in_gap_units']
                for r in group
                if r.get('recovery_in_gap_units') is not None
            ]
            curve.append(
                {
                    'severity': severity,
                    'mean': float(np.mean(values)) if values else None,
                    'std': float(np.std(values)) if values else None,
                    'n_seeds': len({r['seed'] for r in group}),
                    'cost_in_gap_units': float(np.mean(gap)) if gap else None,
                }
            )
        out[violation] = curve
    return out


def distribution_gap(rows):
    """The OU-versus-rollout gap, per arm. A measurement, not a nuisance.

    The encoder trains on OU-set states and the planner sees only physics
    rollouts. The plan refuses to assert that gap away, so it is computed and
    reported.
    """
    by_key = {}
    for row in rows:
        key = (row['arm'], row['violation'], row['severity'], row['seed'])
        by_key.setdefault(key, {})[row['distribution']] = row

    gaps = []
    for key, both in by_key.items():
        if {'ou', 'rollout'} <= set(both):
            ou = both['ou'].get('measured_recovery_error')
            rollout = both['rollout'].get('measured_recovery_error')
            if ou is not None and rollout is not None:
                gaps.append(
                    {
                        'arm': key[0],
                        'violation': key[1],
                        'severity': key[2],
                        'seed': key[3],
                        'ou': ou,
                        'rollout': rollout,
                        'gap': rollout - ou,
                    }
                )
    return gaps


__all__ = [
    'REQUIRED_COLUMNS',
    'SCATTER_SCHEMA_VERSION',
    'append_rows',
    'build_row',
    'count_rows',
    'distribution_gap',
    'read_rows',
    'violation_cost_table',
]

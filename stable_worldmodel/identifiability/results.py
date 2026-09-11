"""The append-only results table: one row per scored checkpoint.

What this replaced, and why
---------------------------
This used to be a "global scatter" keyed by ``(arm, violation, severity)`` with
planning success rates as its y-axes, because the programme it served compared
many experimental arms across a violation ladder. With the study reduced to
encoder training, the only axes left are *which checkpoint* and *which
dataset*, so the row is a plain results record.

One property is kept deliberately and one is deliberately dropped.

**Kept: append-only.** Several fields cannot be reconstructed after the fact --
the dataset's ``config_hash``, the ``rho`` the metrics were computed against,
the encoder hash -- so rows are written from the first run onward and never
rewritten. A re-score appends; the reader takes the last row per key.

**Dropped: the column whitelist.** The old ``build_row`` enumerated every
column it would keep, which silently discarded anything the metric suite grew
later -- ``probe_linear_per_latent`` and ``procrustes_scale`` were computed on
every single run of the old suite and thrown away here, which is exactly the
per-latent breakdown the study turned out to need. Metrics are now passed
through whole, and the *record* fields are the short, explicit part.
"""

import json
from pathlib import Path

import numpy as np


#: Bumped when a field's *meaning* changes. Recorded per row, so a schema
#: change is visible rather than silently mixing incompatible rows.
SCHEMA_VERSION = '2.0.0'

#: Fields every row must carry. A row missing any of these is refused rather
#: than written: a partially-populated table is worse than none, because the
#: gaps look like data.
REQUIRED_FIELDS = (
    'checkpoint',
    'dataset',
    'seed',
    'config_hash',
    'metric_suite_version',
)


def _jsonable(value):
    """Coerce numpy scalars/arrays into something json accepts."""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def build_row(
    checkpoint,
    dataset,
    seed,
    metrics,
    manifest,
    program_constants,
    epoch=None,
    encoder_hash=None,
    bn_recalibrated=None,
    n_samples=None,
    eval_mode=True,
    deviations=None,
    **extra,
):
    """Assemble one results row.

    Args:
        checkpoint: Path or name of the scored weights, e.g.
            ``'lejepa/weights_epoch_25.pt'``.
        dataset: Which dataset the embeddings came from.
        seed: The training run's seed.
        metrics: Output of
            :func:`~stable_worldmodel.identifiability.metrics.compute_all`,
            passed through in full.
        manifest: The dataset manifest, for ``config_hash``, ``rho`` and ``n``.
        program_constants: The run's frozen ``rho`` / ``lambda``, copied
            verbatim so a deviating run is a visible event rather than a
            footnote.
        epoch: Parsed epoch number, for sorting without re-parsing paths.
        encoder_hash: Parameter hash of the scored encoder, so a row can be
            tied back to exact weights even if the file is lost.
        bn_recalibrated: Whether BatchNorm running statistics were refreshed
            over the training split before scoring. Recorded because it changes
            the measured numbers materially and is otherwise invisible.
        n_samples: How many pairs were embedded.
        eval_mode: Whether the encoder was in eval mode. Always true in the
            shipped path; recorded so a row produced any other way is
            self-identifying rather than silently incomparable.
        deviations: Any departure from the fixed protocol.
        **extra: Additional fields.

    Returns:
        dict: One row.

    Raises:
        ValueError: If a required field is missing.
    """
    # Validated BEFORE coercion, not after. `str(None)` is the perfectly
    # valid-looking string "None" and `int(None)` raises a TypeError from deep
    # inside the constructor, so a post-hoc `is None` check on the assembled
    # row cannot catch either -- which would put exactly the kind of
    # gap-that-looks-like-data into the table that this function exists to
    # refuse.
    supplied = {
        'checkpoint': checkpoint,
        'dataset': dataset,
        'seed': seed,
        'config_hash': manifest.get('config_hash'),
        'metric_suite_version': metrics.get('metric_suite_version'),
    }
    absent = sorted(k for k, v in supplied.items() if v is None)
    if absent:
        raise ValueError(
            f'refusing to build a results row missing {absent}. These fields '
            'cannot be reconstructed later, and a gap in the table looks like '
            'data.'
        )

    ou = manifest.get('ou', {})

    row = {
        'schema_version': SCHEMA_VERSION,
        'checkpoint': str(checkpoint),
        'epoch': epoch,
        'dataset': str(dataset),
        'seed': int(seed),
        'encoder_hash': encoder_hash,
        # --- how the embeddings were produced ---
        'eval_mode': bool(eval_mode),
        'bn_recalibrated': bn_recalibrated,
        'n_samples': n_samples,
        # --- configuration that cannot be recovered later ---
        'rho': ou.get('rho'),
        'n': manifest.get('latents', {}).get('n'),
        'profile': manifest.get('profile'),
        'config_hash': manifest.get('config_hash'),
        'metric_suite_version': metrics.get('metric_suite_version'),
        'program_constants': json.dumps(
            _jsonable(program_constants), sort_keys=True
        ),
        'deviations': json.dumps(_jsonable(deviations or {}), sort_keys=True),
    }

    # Metrics pass through whole. Record fields win a name collision, since
    # they are what identifies the row.
    row = {**{k: _jsonable(v) for k, v in metrics.items()}, **row}
    row.update({k: _jsonable(v) for k, v in extra.items()})

    # Belt and braces: catches a required field added to REQUIRED_FIELDS
    # without a matching entry in the pre-coercion check above.
    missing = [f for f in REQUIRED_FIELDS if row.get(f) is None]
    if missing:
        raise ValueError(
            f'refusing to write a results row missing {missing}. These '
            'fields cannot be reconstructed later, and a gap in the table '
            'looks like data.'
        )
    return row


def append_row(path, row):
    """Append one row to a JSONL table, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(_jsonable(row), sort_keys=True) + '\n')
    return path


def load_rows(path, latest_only=True):
    """Read a results table back.

    Args:
        path: The JSONL file.
        latest_only: Keep only the last row per identity key -- see below --
            which is what an append-only table means when something has been
            re-scored.

    The identity key is ``(checkpoint, dataset, probe_kind, bn_recalibrated)``,
    not just ``(checkpoint, dataset)``. Two rows can legitimately share a
    checkpoint and dataset while measuring different things:
    ``run_oracle.py`` writes to the same file as ``run_metrics.py`` (that is
    what makes a ceiling row joinable to the rows it bounds -- see
    ``run_oracle.py``'s module docstring), distinguished only by
    ``probe_kind``; and a checkpoint scored once with BatchNorm recalibrated
    and once without -- the comparison that catches recalibration itself
    distorting a reading -- is distinguished only by ``bn_recalibrated``.
    Keying on ``(checkpoint, dataset)`` alone made the later-appended row
    silently overwrite the earlier one whenever those coincided, which is how
    an oracle row erased its own metrics row here on a live run: both rows
    survived in the file, but only the oracle row survived this function.

    Returns:
        list: Rows, in file order (or last-per-key order when deduplicated).
    """
    path = Path(path)
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not latest_only:
        return rows

    seen = {}
    for row in rows:
        key = (
            row.get('checkpoint'),
            row.get('dataset'),
            row.get('probe_kind'),
            row.get('bn_recalibrated'),
        )
        seen[key] = row
    return list(seen.values())


__all__ = [
    'REQUIRED_FIELDS',
    'SCHEMA_VERSION',
    'append_row',
    'build_row',
    'load_rows',
]

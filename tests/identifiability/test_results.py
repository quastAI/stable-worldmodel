"""Tests for the results table.

Two properties, and the second is the one this table exists to fix.

**A row is refused rather than written incomplete.** Several fields cannot be
reconstructed after the fact -- the dataset's ``config_hash``, the ``rho`` the
metrics were computed against, the encoder's parameter hash -- and a gap in the
table looks like data.

**Metrics pass through whole.** The store this replaced enumerated the columns
it would keep, so anything the metric suite grew later was silently discarded:
``probe_linear_per_latent`` and ``procrustes_scale`` were computed on every run
and thrown away here, and they are the per-latent breakdown the study turned
out to need. A whitelist cannot be allowed back.
"""

import json

import numpy as np
import pytest

from stable_worldmodel.identifiability import results


MANIFEST = {
    'ou': {'rho': 0.9, 'dist': 'gaussian', 'seed': 3072},
    'latents': {'n': 10},
    'profile': 'physical_content',
    'config_hash': '045077e815b426d0',
}

METRICS = {
    'metric_suite_version': '2.0.0',
    'procrustes_mse_per_dim': 0.9728,
    'probe_linear_r2': 0.4264,
    'probe_linear_per_latent': [
        0.78,
        0.73,
        0.72,
        0.58,
        0.56,
        0.49,
        0.24,
        0.17,
        0.002,
        0.0005,
    ],
    'canonical_corr': np.linspace(0.9, 0.0, 10),
    'delta_admissible': np.bool_(True),
    'bound_vacuous': np.True_,
    'trace_cov': np.float64(9.87),
}


def build(**overrides):
    kwargs = {
        'checkpoint': 'lejepa/weights_epoch_25.pt',
        'dataset': 'ogbench/cube_single_ou_physical_content.lance',
        'seed': 3072,
        'metrics': METRICS,
        'manifest': MANIFEST,
        'program_constants': {'rho': 0.9, 'lambda': 0.05},
        'epoch': 25,
        'encoder_hash': '6d03995d12a8fa0d',
        'bn_recalibrated': True,
        'n_samples': 20000,
    }
    kwargs.update(overrides)
    return results.build_row(**kwargs)


# ----------------------------------------------------------- pass-through


def test_every_metric_reaches_the_row():
    """No whitelist: a metric added to the suite lands in the table unchanged."""
    row = build()
    for key, value in METRICS.items():
        assert key in row, f'{key} was dropped'
    assert row['probe_linear_per_latent'] == METRICS['probe_linear_per_latent']
    assert row['probe_linear_r2'] == 0.4264


def test_a_newly_added_metric_is_not_dropped():
    row = build(metrics={**METRICS, 'some_future_metric': 1.25})
    assert row['some_future_metric'] == 1.25


def test_numpy_values_survive_json():
    row = build()
    text = json.dumps(row)
    back = json.loads(text)
    assert back['trace_cov'] == pytest.approx(9.87)
    assert back['delta_admissible'] is True
    assert back['bound_vacuous'] is True
    assert len(back['canonical_corr']) == 10


def test_record_fields_win_a_name_collision():
    """The fields that identify the row must not be overwritable by a metric."""
    row = build(metrics={**METRICS, 'seed': 999, 'epoch': -1})
    assert row['seed'] == 3072
    assert row['epoch'] == 25


# ------------------------------------------------------------- refusal


@pytest.mark.parametrize('field', results.REQUIRED_FIELDS)
def test_a_row_missing_an_unreconstructable_field_is_refused(field):
    overrides = {}
    if field == 'config_hash':
        overrides['manifest'] = {
            k: v for k, v in MANIFEST.items() if k != 'config_hash'
        }
    elif field == 'metric_suite_version':
        overrides['metrics'] = {
            k: v for k, v in METRICS.items() if k != 'metric_suite_version'
        }
    else:
        overrides[field] = None

    with pytest.raises(ValueError, match=field):
        build(**overrides)


def test_seed_zero_is_not_treated_as_missing():
    """``0`` is a legitimate seed; a falsy check here would refuse it."""
    assert build(seed=0)['seed'] == 0


# ------------------------------------------------------------ provenance


def test_row_records_how_the_embeddings_were_produced():
    """BatchNorm recalibration changes the numbers materially and is otherwise
    invisible, so it is part of the row's identity."""
    row = build()
    assert row['bn_recalibrated'] is True
    assert row['eval_mode'] is True
    assert row['n_samples'] == 20000
    assert row['encoder_hash'] == '6d03995d12a8fa0d'


def test_row_carries_the_rho_the_metrics_used():
    """Not the config's rho: the manifest's, since that is what was measured."""
    assert build()['rho'] == 0.9


def test_program_constants_are_recorded_verbatim():
    row = build(program_constants={'rho': 0.9, 'lambda': 3.0e-3})
    assert json.loads(row['program_constants'])['lambda'] == 3.0e-3


# ------------------------------------------------------------ append-only


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / 'results.jsonl'
    results.append_row(path, build())
    rows = results.load_rows(path)
    assert len(rows) == 1
    assert rows[0]['checkpoint'] == 'lejepa/weights_epoch_25.pt'


def test_append_creates_missing_directories(tmp_path):
    path = tmp_path / 'nested' / 'deeper' / 'results.jsonl'
    results.append_row(path, build())
    assert path.exists()


def test_rescoring_appends_and_the_reader_takes_the_latest(tmp_path):
    """Append-only: a re-score must not rewrite history, but must win on read."""
    path = tmp_path / 'results.jsonl'
    results.append_row(
        path, build(metrics={**METRICS, 'probe_linear_r2': 0.1})
    )
    results.append_row(
        path, build(metrics={**METRICS, 'probe_linear_r2': 0.9})
    )

    assert len(results.load_rows(path, latest_only=False)) == 2
    latest = results.load_rows(path)
    assert len(latest) == 1
    assert latest[0]['probe_linear_r2'] == 0.9


def test_distinct_checkpoints_are_distinct_rows(tmp_path):
    path = tmp_path / 'results.jsonl'
    results.append_row(
        path, build(epoch=24, checkpoint='a/weights_epoch_24.pt')
    )
    results.append_row(
        path, build(epoch=25, checkpoint='a/weights_epoch_25.pt')
    )
    assert len(results.load_rows(path)) == 2


def test_loading_a_missing_table_is_empty_not_an_error(tmp_path):
    assert results.load_rows(tmp_path / 'nope.jsonl') == []


def test_oracle_row_does_not_erase_the_metrics_row_it_bounds(tmp_path):
    """A live regression: `run_oracle.py` writes to the same file, same
    (checkpoint, dataset), as the `run_metrics.py` row it joins to. Keying
    the dedup on `(checkpoint, dataset)` alone made the later-appended oracle
    row silently overwrite the metrics row -- both rows survived in the file,
    only one survived `load_rows`.
    """
    path = tmp_path / 'results.jsonl'
    results.append_row(
        path, build(metrics={**METRICS, 'probe_linear_r2': 0.36})
    )
    results.append_row(
        path,
        build(
            metrics={
                'metric_suite_version': '2.1.0',
                'probe_kind': 'oracle_supervised_scratch',
                'oracle_r2': 0.83,
            },
            bn_recalibrated=False,
        ),
    )

    rows = results.load_rows(path)
    assert len(rows) == 2
    kinds = {r.get('probe_kind') for r in rows}
    assert kinds == {None, 'oracle_supervised_scratch'}
    metrics_row = next(r for r in rows if r.get('probe_kind') is None)
    assert metrics_row['probe_linear_r2'] == 0.36


def test_recalibrated_and_non_recalibrated_rows_both_survive(tmp_path):
    """The comparison that catches BN recalibration distorting a reading
    needs both variants to survive scoring the same checkpoint twice.
    """
    path = tmp_path / 'results.jsonl'
    results.append_row(
        path, build(bn_recalibrated=True, metrics={**METRICS, 'L': 0.98})
    )
    results.append_row(
        path, build(bn_recalibrated=False, metrics={**METRICS, 'L': 5.45})
    )

    rows = results.load_rows(path)
    assert len(rows) == 2
    by_recal = {r['bn_recalibrated']: r['L'] for r in rows}
    assert by_recal == {True: 0.98, False: 5.45}


def test_true_rescore_at_the_same_identity_still_dedups(tmp_path):
    """The original intent is preserved: an exact re-run (same checkpoint,
    dataset, probe_kind, bn_recalibrated) is a correction, not a new
    measurement, and only the latest should read back.
    """
    path = tmp_path / 'results.jsonl'
    results.append_row(
        path, build(bn_recalibrated=True, metrics={**METRICS, 'L': 1.0})
    )
    results.append_row(
        path, build(bn_recalibrated=True, metrics={**METRICS, 'L': 2.0})
    )
    rows = results.load_rows(path)
    assert len(rows) == 1
    assert rows[0]['L'] == 2.0


def test_schema_version_is_on_every_row():
    """A schema change must be visible rather than silently mixing rows."""
    assert build()['schema_version'] == results.SCHEMA_VERSION

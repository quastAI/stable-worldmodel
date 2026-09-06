"""Tests for the global scatter.

The scatter's defining property is that it cannot be repaired later. A row
written without its ``rho`` vector, isotropy margin, program constants or
dataset ``config_hash`` is not a partial row -- it is a row that can never
again be told apart from one produced under a different configuration. These
tests are mostly about refusing to write such a row.
"""

import numpy as np
import pytest

from stable_worldmodel.identifiability import scatter as sc


def manifest(rho=(0.9,) * 4, config_hash='abc123', severity=0.0, key='none'):
    rho = list(rho)
    return {
        'config_hash': config_hash,
        'profile': 'task_content',
        'latents': {'n': len(rho)},
        'violation': {'key': key, 'severity': severity},
        'ou': {
            'rho': rho,
            'isotropy': {
                'satisfied': min(rho) > max(rho) ** 2,
                'margin': min(rho) - max(rho) ** 2,
            },
        },
    }


def metrics(**overrides):
    base = {
        'metric_suite_version': '1.0.0',
        'procrustes_mse_per_dim': 0.05,
        'orth_err_normalized': 0.1,
        'cond': 1.2,
        'predicted_error': 0.08,
        'epsilon': 0.2,
        'delta': 0.01,
        'D': 0.055,
        'spectral_gap': 0.18,
        'recovery_in_gap_units': 0.28,
        'probe_linear_r2': 0.9,
        'probe_mlp_r2': 0.93,
        'probe_divergence': 0.01,
        'r2_z_to_h': 0.95,
        'r2_h_to_z': 0.94,
        'mcc_unaligned': 0.6,
        'hermite2_excess': -0.1,
        'sigreg_z': 0.4,
    }
    base.update(overrides)
    return base


CONSTANTS = {'rho': 0.9, 'lambda': 3e-3, 'seed_budget': 5}


def row(**kwargs):
    params = {
        'arm': 'A',
        'violation': 'none',
        'severity': 0.0,
        'seed': 1,
        'distribution': 'ou',
        'metrics': metrics(),
        'manifest': manifest(),
        'program_constants': CONSTANTS,
        'success_rate': 0.6,
        'success_rate_oracle': 0.8,
        'success_rate_state': 0.75,
    }
    params.update(kwargs)
    return sc.build_row(**params)


# ------------------------------------------------------------- both axes


def test_row_carries_both_y_axes():
    """``SR/SR(O)`` and ``SR/SR(P)`` disagree, and the disagreement matters.

    Reporting only one would make an ordinary predictor-capacity shortfall
    indistinguishable from an identifiability failure.
    """
    r = row()
    assert r['sr_over_o'] == pytest.approx(0.6 / 0.8)
    assert r['sr_over_p'] == pytest.approx(0.6 / 0.75)


def test_row_carries_both_x_axes():
    r = row()
    assert r['measured_recovery_error'] == 0.05
    assert r['predicted_error'] == 0.08


def test_missing_denominator_yields_none_not_a_crash():
    r = row(success_rate_oracle=None)
    assert r['sr_over_o'] is None
    assert r['sr_over_p'] is not None


# ------------------------------------------- the unrecoverable columns


def test_row_records_the_full_rho_vector():
    """A scalar mean cannot be un-averaged later."""
    r = row(manifest=manifest(rho=(0.8, 0.85, 0.9, 0.95)))
    assert r['rho_per_dim'] == [0.8, 0.85, 0.9, 0.95]
    assert r['rho_mean'] == pytest.approx(0.875)


def test_row_records_the_isotropy_margin():
    r = row(manifest=manifest(rho=(0.5, 0.9, 0.9, 0.9)))
    assert r['isotropy_satisfied'] is False
    assert r['isotropy_margin'] < 0


def test_row_records_program_constants_verbatim():
    """A deviating run must be a visible event, not a footnote."""
    r = row()
    assert '"rho": 0.9' in r['program_constants']
    assert '"lambda": 0.003' in r['program_constants']


def test_row_is_refused_when_a_required_column_is_missing():
    """A gap in the scatter looks like data. Better to refuse the write."""
    with pytest.raises(ValueError, match='missing required columns'):
        row(manifest=manifest(config_hash=None))


def test_row_records_the_metric_suite_version():
    """A metric that moves mid-programme must become visible."""
    assert row()['metric_suite_version'] == '1.0.0'


def test_row_records_deviations():
    r = row(deviations={'arm_o_num_samples_override': 100})
    assert '100' in r['deviations']


# --------------------------------------------------------------- the file


def test_append_is_append_only(tmp_path):
    path = tmp_path / 'scatter.jsonl'
    assert sc.append_rows(path, [row(seed=1)]) == 1
    assert sc.append_rows(path, [row(seed=2), row(seed=3)]) == 3

    rows = sc.read_rows(path)
    assert [r['seed'] for r in rows] == [1, 2, 3]


def test_append_of_nothing_is_a_no_op(tmp_path):
    path = tmp_path / 'scatter.jsonl'
    sc.append_rows(path, [row()])
    assert sc.append_rows(path, []) == 1


def test_numpy_scalars_survive_the_round_trip(tmp_path):
    """Metrics arrive as numpy types; json must not choke on them."""
    path = tmp_path / 'scatter.jsonl'
    sc.append_rows(
        path,
        [
            row(
                metrics=metrics(
                    procrustes_mse_per_dim=np.float64(0.25),
                    cond=np.float32(3.5),
                )
            )
        ],
    )
    assert sc.read_rows(path)[0]['measured_recovery_error'] == 0.25


# ----------------------------------------------------------- the artifacts


def test_violation_cost_table_averages_over_seeds():
    rows = [
        row(
            violation='v1',
            severity=s,
            seed=seed,
            success_rate=sr,
            manifest=manifest(key='v1', severity=s),
        )
        for s, srs in ((0.0, [0.8, 0.78]), (1.0, [0.4, 0.42]))
        for seed, sr in enumerate(srs)
    ]
    table = sc.violation_cost_table(rows)

    curve = table['v1']
    assert [point['severity'] for point in curve] == [0.0, 1.0]
    assert curve[0]['mean'] > curve[1]['mean']
    assert curve[0]['n_seeds'] == 2


def test_violation_cost_table_reports_gap_units():
    """Costs must be comparable across environments whose rho differs."""
    table = sc.violation_cost_table([row(violation='v1', severity=0.5)])
    assert table['v1'][0]['cost_in_gap_units'] == pytest.approx(0.28)


def test_distribution_gap_pairs_ou_with_rollout():
    """The OU-versus-rollout gap is a measurement, not a nuisance."""
    rows = [
        row(distribution='ou', metrics=metrics(procrustes_mse_per_dim=0.05)),
        row(
            distribution='rollout',
            metrics=metrics(procrustes_mse_per_dim=0.12),
        ),
    ]
    gaps = sc.distribution_gap(rows)
    assert len(gaps) == 1
    assert gaps[0]['gap'] == pytest.approx(0.07)


def test_distribution_gap_ignores_unpaired_rows():
    gaps = sc.distribution_gap([row(distribution='ou')])
    assert gaps == []


def test_both_y_axes_can_be_selected_in_the_cost_table():
    rows = [row(violation='v1', severity=0.0)]
    by_o = sc.violation_cost_table(rows, y_axis='sr_over_o')
    by_p = sc.violation_cost_table(rows, y_axis='sr_over_p')
    assert by_o['v1'][0]['mean'] != by_p['v1'][0]['mean']

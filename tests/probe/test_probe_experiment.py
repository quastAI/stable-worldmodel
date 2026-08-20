"""Tests for the probing experiment under ``scripts/probe/``.

The modules live in ``scripts/`` rather than the installed package, so this
file puts that directory on ``sys.path`` instead of importing through
``stable_worldmodel``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'probe'
sys.path.insert(0, str(SCRIPTS))

import features as ft  # noqa: E402
import fit as fitting  # noqa: E402
import targets as tg  # noqa: E402

from stable_worldmodel.data.dataset import Dataset  # noqa: E402


#####################
## target registry ##
#####################


def test_registry_names_are_unique():
    names = [t.name for t in tg.TARGETS]
    assert len(names) == len(set(names))


def test_every_group_is_populated():
    for group in tg.GROUPS:
        assert any(t.group == group for t in tg.TARGETS)


def test_select_targets_defaults_to_everything():
    assert len(tg.select_targets()) == len(tg.TARGETS)


def test_select_targets_by_group():
    picked = tg.select_targets(groups=['nuisance'])
    assert picked and all(t.group == 'nuisance' for t in picked)


def test_select_targets_by_name_keeps_registry_order():
    picked = tg.select_targets(names=['digit_value', 'effector_pos'])
    assert [t.name for t in picked] == ['effector_pos', 'digit_value']


def test_select_targets_rejects_unknown_name():
    with pytest.raises(KeyError, match='Unknown probe targets'):
        tg.select_targets(names=['not_a_target'])


def test_select_targets_rejects_unknown_group():
    with pytest.raises(KeyError, match='Unknown groups'):
        tg.select_targets(groups=['nope'])


def test_required_columns_is_deduplicated():
    cols = tg.required_columns(tg.select_targets())
    assert len(cols) == len(set(cols))


def test_classification_targets_declare_num_classes():
    for target in tg.TARGETS:
        if target.kind == 'classification':
            assert target.num_classes and target.num_classes > 1


def test_probe_target_validates_kind():
    with pytest.raises(ValueError, match='bad kind'):
        tg.ProbeTarget(name='x', columns=('a',), kind='ranking')


def test_probe_target_validates_group():
    with pytest.raises(ValueError, match='bad group'):
        tg.ProbeTarget(
            name='x', columns=('a',), kind='regression', group='other'
        )


def test_probe_target_validates_reducer():
    with pytest.raises(ValueError, match='bad reducer'):
        tg.ProbeTarget(name='x', columns=('a',), kind='regression', reduce='?')


def test_probe_target_classification_needs_classes():
    with pytest.raises(ValueError, match='num_classes'):
        tg.ProbeTarget(name='x', columns=('a',), kind='classification')


##############
## reducers ##
##############


def _cols(**kwargs):
    """Build the ``(N, num_steps, dim)`` layout the extractor stores."""
    return {k: np.asarray(v, dtype=np.float32) for k, v in kwargs.items()}


def test_build_labels_concat_reads_the_requested_step():
    target = tg.TARGETS_BY_NAME['effector_pos']
    cols = _cols(
        **{
            'proprio/effector_pos': [
                [[0, 0, 0], [1, 2, 3], [9, 9, 9], [4, 4, 4]]
            ]
        }
    )
    out = tg.build_labels(target, cols, step=1)
    assert out.shape == (1, 3)
    assert np.allclose(out, [[1, 2, 3]])


def test_build_labels_sincos_maps_an_angle():
    target = tg.TARGETS_BY_NAME['effector_yaw']
    cols = _cols(**{'proprio/effector_yaw': [[[np.pi / 2]]]})
    out = tg.build_labels(target, cols, step=0)
    assert out.shape == (1, 2)
    assert np.allclose(out, [[1.0, 0.0]], atol=1e-6)


def test_sincos_is_continuous_across_the_pi_wrap():
    """The whole reason yaw is not regressed raw."""
    target = tg.TARGETS_BY_NAME['effector_yaw']
    just_under = tg.build_labels(
        target, _cols(**{'proprio/effector_yaw': [[[np.pi - 1e-4]]]}), 0
    )
    just_over = tg.build_labels(
        target, _cols(**{'proprio/effector_yaw': [[[-np.pi + 1e-4]]]}), 0
    )
    assert np.allclose(just_under, just_over, atol=1e-3)


def test_build_labels_classification_returns_int_indices():
    target = tg.TARGETS_BY_NAME['digit_value']
    cols = _cols(**{'privileged/digit_0_value': [[[7.0]], [[0.0]]]})
    out = tg.build_labels(target, cols, step=0)
    assert out.dtype == np.int64
    assert out.tolist() == [7, 0]


def test_build_labels_rejects_out_of_range_class():
    target = tg.TARGETS_BY_NAME['digit_value']
    cols = _cols(**{'privileged/digit_0_value': [[[11.0]]]})
    with pytest.raises(ValueError, match='class indices'):
        tg.build_labels(target, cols, step=0)


def _block_cols(positions):
    """positions: (num_steps, 4, 3) -> the four block_i_pos columns."""
    positions = np.asarray(positions, dtype=np.float32)
    return {
        col: positions[None, :, i, :]
        for i, col in enumerate(tg.BLOCK_POS_COLUMNS)
    }


def test_blocks_z_sorted_is_ascending():
    cols = _block_cols([[[0, 0, 0.3], [0, 0, 0.1], [0, 0, 0.4], [0, 0, 0.2]]])
    out = tg.build_labels(tg.TARGETS_BY_NAME['cube_z_sorted'], cols, 0)
    assert np.allclose(out, [[0.1, 0.2, 0.3, 0.4]])


def test_blocks_pos_sorted_is_permutation_invariant():
    """The point of the sorted reducer: relabelling the cubes cannot change it."""
    layout = [[1.0, 0, 0], [3.0, 1, 0], [2.0, 2, 0], [4.0, 3, 0]]
    shuffled = [layout[i] for i in (2, 0, 3, 1)]
    target = tg.TARGETS_BY_NAME['cube_pos_sorted']
    a = tg.build_labels(target, _block_cols([layout]), 0)
    b = tg.build_labels(target, _block_cols([shuffled]), 0)
    assert a.shape == (1, 12)
    assert np.allclose(a, b)


def test_build_labels_reports_a_missing_column():
    target = tg.TARGETS_BY_NAME['effector_pos']
    with pytest.raises(KeyError, match='needs column'):
        tg.build_labels(target, {}, step=0)


def test_build_labels_rejects_a_flat_column():
    target = tg.TARGETS_BY_NAME['effector_pos']
    cols = {'proprio/effector_pos': np.zeros((4, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match='num_steps'):
        tg.build_labels(target, cols, step=0)


def test_label_dim_matches_build_labels_for_every_target():
    dims = {
        'proprio/effector_pos': 3,
        'proprio/effector_yaw': 1,
        'proprio/gripper_opening': 1,
        'proprio/gripper_contact': 1,
        'privileged/digit_0_value': 1,
        'privileged/floor_material': 1,
        'privileged/wall_material': 1,
        'privileged/floor_rgb': 3,
        'privileged/light_pos': 3,
        **{c: 3 for c in tg.BLOCK_POS_COLUMNS},
    }
    rng = np.random.default_rng(0)
    columns = {
        col: rng.random((2, 4, dim)).astype(np.float32)
        for col, dim in dims.items()
    }
    # Class labels must stay inside their declared range.
    columns['privileged/digit_0_value'] = np.zeros((2, 4, 1), np.float32)
    columns['privileged/floor_material'] = np.zeros((2, 4, 1), np.float32)
    columns['privileged/wall_material'] = np.zeros((2, 4, 1), np.float32)

    for target in tg.TARGETS:
        built = tg.build_labels(target, columns, step=0)
        expected = tg.label_dim(target, dims)
        if target.kind == 'classification':
            assert built.ndim == 1
            assert expected == target.num_classes
        else:
            assert built.shape[1] == expected, target.name


###########
## split ##
###########


def test_episode_split_is_disjoint_and_sized():
    counts = {'train': 10, 'val': 3, 'test': 5}
    splits = ft.episode_split(40, counts, seed=0)
    for name, n in counts.items():
        assert len(splits[name]) == n
    all_eps = np.concatenate([splits[s] for s in ft.SPLITS])
    assert len(np.unique(all_eps)) == len(all_eps)


def test_episode_split_is_seed_reproducible():
    counts = {'train': 5, 'val': 2, 'test': 2}
    a = ft.episode_split(20, counts, seed=7)
    b = ft.episode_split(20, counts, seed=7)
    c = ft.episode_split(20, counts, seed=8)
    assert all(np.array_equal(a[s], b[s]) for s in ft.SPLITS)
    assert not all(np.array_equal(a[s], c[s]) for s in ft.SPLITS)


def test_episode_split_rejects_an_oversized_request():
    with pytest.raises(ValueError, match='asked for'):
        ft.episode_split(5, {'train': 4, 'val': 1, 'test': 1}, seed=0)


def test_episode_split_requires_all_splits():
    with pytest.raises(ValueError, match='missing splits'):
        ft.ExtractConfig(episodes={'train': 1}, checkpoint='r/w.pt')


######################
## frame sampling   ##
######################


class _StubDataset(Dataset):
    """Just enough of a Dataset to expose real ``clip_indices``."""

    def __init__(self, lengths):
        super().__init__(
            np.asarray(lengths, dtype=np.int64),
            np.zeros(len(lengths), dtype=np.int64),
            frameskip=1,
            num_steps=1,
        )


def test_sample_frames_indices_match_real_clip_indices():
    """The arithmetic shortcut must agree with Dataset.clip_indices."""
    lengths = [40, 25, 60, 12, 33]
    dataset = _StubDataset(lengths)

    clips, eps, frame_idx = ft.sample_frames(
        dataset.lengths, np.arange(len(lengths)), 3, seed=1
    )
    for clip, ep, frame in zip(clips, eps, frame_idx):
        assert dataset.clip_indices[int(clip)] == (int(ep), int(frame))


def test_sample_frames_is_capped_by_episode_length():
    lengths = [5, 100]
    _, eps, _ = ft.sample_frames(np.asarray(lengths), np.arange(2), 50, seed=0)
    assert (eps == 0).sum() == 5
    assert (eps == 1).sum() == 50


def test_sample_frames_draws_without_replacement():
    clips, _, _ = ft.sample_frames(np.asarray([100]), np.array([0]), 40, seed=3)
    assert len(np.unique(clips)) == len(clips)


def test_sample_frames_raises_when_nothing_fits():
    with pytest.raises(ValueError, match='no valid frames'):
        ft.sample_frames(np.asarray([0, 0]), np.arange(2), 5, seed=0)


def test_sample_frames_is_sorted():
    clips, _, _ = ft.sample_frames(np.asarray([100, 100, 100]), np.arange(3), 10, seed=0)
    assert np.all(np.diff(clips) > 0)


######################
## extract config   ##
######################


def test_extract_config_requires_a_checkpoint():
    with pytest.raises(ValueError, match='checkpoint is required'):
        ft.ExtractConfig(checkpoint=None)


#############
## metrics ##
#############


def test_r2_is_one_for_a_perfect_prediction():
    y = np.random.default_rng(0).normal(size=(50, 3))
    assert np.allclose(fitting.r2_per_dim(y, y), 1.0)


def test_r2_is_zero_for_the_test_mean_predictor():
    y = np.random.default_rng(0).normal(size=(50, 3))
    pred = np.broadcast_to(y.mean(axis=0, keepdims=True), y.shape)
    assert np.allclose(fitting.r2_per_dim(y, pred), 0.0)


def test_r2_of_a_constant_target_dim_is_zero_not_nan():
    y = np.zeros((10, 2))
    y[:, 1] = np.arange(10)
    out = fitting.r2_per_dim(y, y)
    assert out[0] == 0.0
    assert np.isfinite(out).all()


def test_regression_metrics_reports_physical_units():
    y = np.zeros((10, 2))
    pred = np.full((10, 2), 0.5)
    metrics = fitting.regression_metrics(y, pred)
    assert metrics['mae'] == pytest.approx(0.5)
    assert metrics['rmse'] == pytest.approx(0.5)


def test_classification_metrics_perfect_accuracy():
    y = np.array([0, 1, 2])
    logits = np.eye(3) * 10
    metrics = fitting.classification_metrics(y, logits, 3)
    assert metrics['acc'] == pytest.approx(1.0)
    assert metrics['balanced_acc'] == pytest.approx(1.0)


def test_classification_metrics_balanced_acc_penalizes_a_majority_guess():
    y = np.array([0, 0, 0, 0, 1])
    logits = np.tile([10.0, 0.0], (5, 1))
    metrics = fitting.classification_metrics(y, logits, 2)
    assert metrics['acc'] == pytest.approx(0.8)
    assert metrics['balanced_acc'] == pytest.approx(0.5)


##############
## baseline ##
##############


def test_regression_baseline_lands_near_zero():
    rng = np.random.default_rng(0)
    y = {s: rng.normal(size=(200, 2)) for s in ft.SPLITS}
    score, _ = fitting.fit_baseline(tg.TARGETS_BY_NAME['effector_pos'], y)
    assert abs(score) < 0.1


def test_classification_baseline_is_the_majority_rate():
    y = {
        'train': np.array([0] * 90 + [1] * 10),
        'val': np.array([0] * 9 + [1]),
        'test': np.array([0] * 80 + [1] * 20),
    }
    score, _ = fitting.fit_baseline(tg.TARGETS_BY_NAME['floor_material'], y)
    assert score == pytest.approx(0.8)


#################
## linear fits ##
#################


def _linear_problem(n=400, dim=12, out=3, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.normal(size=(dim, out))
    data = {}
    for split, size in (('train', n), ('val', n // 4), ('test', n // 2)):
        x = rng.normal(size=(size, dim))
        y = x @ weight + noise * rng.normal(size=(size, out))
        data[split] = (x.astype(np.float32), y.astype(np.float32))
    return (
        {s: v[0] for s, v in data.items()},
        {s: v[1] for s, v in data.items()},
    )


def test_ridge_recovers_a_noiseless_linear_map():
    x, y = _linear_problem(noise=0.0)
    cfg = fitting.FitConfig(seed=0)
    _, val_score, metrics, hyper = fitting.fit_ridge(x, y, cfg, 'cpu')
    assert metrics['r2'] > 0.99
    assert val_score > 0.99
    assert hyper['alpha'] in cfg.ridge_alphas


def test_ridge_probe_predicts_in_physical_units():
    x, y = _linear_problem(noise=0.0)
    probe, _, _, _ = fitting.fit_ridge(x, y, fitting.FitConfig(), 'cpu')
    pred = probe.predict(torch.tensor(x['test'])).detach().numpy()
    assert np.allclose(pred, y['test'], atol=0.05)


def test_ridge_finds_no_signal_in_noise():
    rng = np.random.default_rng(1)
    x = {
        s: rng.normal(size=(n, 8)).astype(np.float32)
        for s, n in (('train', 300), ('val', 80), ('test', 150))
    }
    y = {
        s: rng.normal(size=(len(v), 2)).astype(np.float32)
        for s, v in x.items()
    }
    _, _, metrics, _ = fitting.fit_ridge(x, y, fitting.FitConfig(), 'cpu')
    assert metrics['r2'] < 0.1


def test_ridge_flags_an_alpha_at_the_edge_of_the_sweep():
    x, y = _linear_problem(noise=0.0)
    cfg = fitting.FitConfig(ridge_alphas=(1e3, 1e4))
    _, _, _, hyper = fitting.fit_ridge(x, y, cfg, 'cpu')
    assert hyper['warning']


####################
## gradient  fits ##
####################


def test_mlp_probe_beats_a_linear_one_on_a_nonlinear_target():
    """The linear -> MLP gap must be able to appear when it should."""
    rng = np.random.default_rng(0)
    dim = 6
    data = {}
    for split, n in (('train', 2000), ('val', 400), ('test', 800)):
        x = rng.normal(size=(n, dim))
        # XOR-like: not linearly decodable, easy for one hidden layer.
        y = (x[:, 0] * x[:, 1])[:, None]
        data[split] = (x.astype(np.float32), y.astype(np.float32))
    x = {s: v[0] for s, v in data.items()}
    y = {s: v[1] for s, v in data.items()}

    target = tg.TARGETS_BY_NAME['gripper_opening']
    cfg = fitting.FitConfig(epochs=120, patience=30, seed=0)
    _, _, linear_metrics, _ = fitting.fit_ridge(x, y, cfg, 'cpu')
    _, _, mlp_metrics, _ = fitting.fit_gradient_probe(
        'mlp', x, y, target, cfg, 'cpu'
    )
    assert linear_metrics['r2'] < 0.15
    assert mlp_metrics['r2'] > 0.6


def test_logistic_probe_separates_two_classes():
    rng = np.random.default_rng(0)
    data = {}
    for split, n in (('train', 600), ('val', 150), ('test', 300)):
        labels = rng.integers(0, 4, size=n)
        x = np.eye(4)[labels] * 3 + rng.normal(scale=0.3, size=(n, 4))
        data[split] = (x.astype(np.float32), labels.astype(np.int64))
    x = {s: v[0] for s, v in data.items()}
    y = {s: v[1] for s, v in data.items()}

    target = tg.TARGETS_BY_NAME['floor_material']
    cfg = fitting.FitConfig(epochs=80, patience=20, seed=0)
    _, _, metrics, hyper = fitting.fit_gradient_probe(
        'linear', x, y, target, cfg, 'cpu'
    )
    assert metrics['acc'] > 0.95
    assert hyper['weight_decay'] in cfg.weight_decays


def test_fit_one_routes_linear_regression_to_the_closed_form():
    x, y = _linear_problem(noise=0.0)
    target = tg.TARGETS_BY_NAME['effector_pos']
    _, _, _, _, hyper = fitting.fit_one(
        'linear', x, y, target, fitting.FitConfig(), 'cpu'
    )
    assert 'alpha' in hyper


def test_fit_one_routes_linear_classification_to_gradients():
    rng = np.random.default_rng(0)
    x = {
        s: rng.normal(size=(n, 4)).astype(np.float32)
        for s, n in (('train', 200), ('val', 50), ('test', 100))
    }
    y = {
        s: rng.integers(0, 8, size=len(v)).astype(np.int64)
        for s, v in x.items()
    }
    cfg = fitting.FitConfig(epochs=5, patience=2, weight_decays=(1e-2,))
    _, _, _, _, hyper = fitting.fit_one(
        'linear', x, y, tg.TARGETS_BY_NAME['floor_material'], cfg, 'cpu'
    )
    assert 'weight_decay' in hyper


def test_fit_config_rejects_an_unknown_rung():
    with pytest.raises(ValueError, match='unknown probe kinds'):
        fitting.FitConfig(probes=('linear', 'transformer'))


##################
## fit_all glue ##
##################


def _fake_payload(n_train=200, n_eval=60, dim=16, num_steps=1, seed=0):
    """A cache-shaped payload with a decodable and an undecodable target."""
    rng = np.random.default_rng(seed)
    columns = tg.required_columns(
        [
            tg.TARGETS_BY_NAME['effector_pos'],
            tg.TARGETS_BY_NAME['digit_value'],
        ]
    )
    weight = rng.normal(size=(dim, 3))

    features, labels, windows = {}, {}, {}
    for split, n in (
        ('train', n_train),
        ('val', n_eval),
        ('test', n_eval),
    ):
        x = rng.normal(size=(n, dim)).astype(np.float32)
        effector = np.zeros((n, num_steps, 3), dtype=np.float32)
        effector[:, 0] = x @ weight
        digit = rng.integers(0, 10, size=(n, num_steps, 1)).astype(np.float32)
        features[split] = x
        labels[split] = {
            'proprio/effector_pos': effector,
            'privileged/digit_0_value': digit,
        }
        windows[split] = {
            'episode_idx': np.repeat(np.arange(n // 10 or 1), 10)[:n],
            'frame_idx': np.arange(n),
            'clip_index': np.arange(n),
        }
    meta = {
        'feature_dim': dim,
        'label_columns': columns,
        'column_dims': {
            'proprio/effector_pos': 3,
            'privileged/digit_0_value': 1,
        },
        'episodes': {s: [] for s in ft.SPLITS},
        'num_frames': {'train': n_train, 'val': n_eval, 'test': n_eval},
    }
    return {
        'features': features,
        'labels': labels,
        'windows': windows,
        'meta': meta,
    }


def test_fit_all_produces_one_row_per_combination():
    payload = _fake_payload()
    probe_targets = tg.select_targets(names=['effector_pos', 'digit_value'])
    cfg = fitting.FitConfig(
        probes=('baseline', 'linear'),
        epochs=10,
        patience=3,
        weight_decays=(1e-2,),
    )
    rows, _ = fitting.fit_all(payload, probe_targets, cfg, progress=False)
    assert len(rows) == 2 * 2
    assert {r['probe'] for r in rows} == {'baseline', 'linear'}


def test_fit_all_recovers_the_planted_signal_and_not_the_noise():
    payload = _fake_payload()
    probe_targets = tg.select_targets(names=['effector_pos', 'digit_value'])
    cfg = fitting.FitConfig(
        probes=('baseline', 'linear'),
        epochs=20,
        patience=5,
        weight_decays=(1e-2,),
    )
    rows, _ = fitting.fit_all(payload, probe_targets, cfg, progress=False)
    scores = {(r['target'], r['probe']): r['score'] for r in rows}
    assert scores[('effector_pos', 'linear')] > 0.9
    assert scores[('digit_value', 'linear')] < 0.35


def test_fit_all_reports_effective_n_for_episode_constant_targets():
    payload = _fake_payload()
    cfg = fitting.FitConfig(probes=('baseline',))
    rows, _ = fitting.fit_all(
        payload,
        tg.select_targets(names=['effector_pos', 'digit_value']),
        cfg,
        progress=False,
    )
    by_target = {r['target']: r for r in rows}
    n_episodes = len(np.unique(payload['windows']['train']['episode_idx']))
    assert by_target['digit_value']['n_train_effective'] == n_episodes
    assert by_target['digit_value']['episode_constant'] is True
    assert by_target['effector_pos']['n_train_effective'] == 200
    assert by_target['effector_pos']['episode_constant'] is False


def test_fit_all_can_return_the_fitted_probes():
    payload = _fake_payload()
    cfg = fitting.FitConfig(probes=('linear',))
    _, probes = fitting.fit_all(
        payload,
        tg.select_targets(names=['effector_pos']),
        cfg,
        keep_probes=True,
        progress=False,
    )
    assert ('effector_pos', 'linear') in probes


###########
## cache ##
###########


def test_feature_cache_roundtrips(tmp_path):
    payload = _fake_payload(n_train=20, n_eval=8, dim=5)
    path = ft.save_features(tmp_path / 'features.npz', payload)
    loaded = ft.load_features(path)

    assert loaded['meta'] == payload['meta']
    for split in ft.SPLITS:
        assert np.allclose(
            loaded['features'][split], payload['features'][split]
        )
        for col in payload['meta']['label_columns']:
            assert np.allclose(
                loaded['labels'][split][col], payload['labels'][split][col]
            )
        assert np.array_equal(
            loaded['windows'][split]['clip_index'],
            payload['windows'][split]['clip_index'],
        )


def test_feature_cache_survives_slashes_in_column_names(tmp_path):
    payload = _fake_payload(n_train=10, n_eval=4, dim=3)
    assert any('/' in c for c in payload['meta']['label_columns'])
    loaded = ft.load_features(ft.save_features(tmp_path / 'f.npz', payload))
    assert set(loaded['labels']['train']) == set(payload['labels']['train'])

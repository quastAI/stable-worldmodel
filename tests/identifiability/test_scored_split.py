"""Scoring the held-out split, and reproducing it exactly.

`run_metrics.py` scored the whole dataset until 2026-09-11, and `lejepa.py`
trains on 90% of that same dataset -- so ~90% of every row ever written was
measured on data the encoder had fit. That put `L` at 0.98 against a floor of
1.87 on the s3072 50-epoch run: below the floor, impossible for a function of
`z` under the declared rho, and possible only because the Hermite floor bounds
the population rather than a finite fitted sample. The same weights, scored
held-out by the run's own validation, gave an admissible `delta = 3.6`.

The split must therefore be reproduced *exactly*, not re-drawn -- a different
draw silently scores a blend of both halves, which looks like neither.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import stable_pretraining as spt


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'identifiability'


def _load_run_metrics():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        'run_metrics', SCRIPTS / 'run_metrics.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_metrics = _load_run_metrics()


def cfg(**overrides):
    base = {'seed': 3072, 'train_split': 0.9, 'split': 'val'}
    base.update(overrides)
    return OmegaConf.create(base)


def training_split(dataset, seed=3072, train_split=0.9):
    """Exactly what lejepa.py does, as the reference to match."""
    generator = torch.Generator().manual_seed(seed)
    return spt.data.random_split(
        dataset,
        lengths=[train_split, 1.0 - train_split],
        generator=generator,
    )


# --------------------------------------------------------------------------
# the property that matters: identical indices to the training run
# --------------------------------------------------------------------------


def test_val_subset_matches_the_training_runs_val_split():
    dataset = list(range(2000))
    _, reference_val = training_split(dataset)
    subset, _ = run_metrics.scored_subset(dataset, cfg())
    assert list(subset.indices) == list(reference_val.indices)


def test_train_subset_matches_the_training_runs_train_split():
    dataset = list(range(2000))
    reference_train, _ = training_split(dataset)
    subset, _ = run_metrics.scored_subset(dataset, cfg(split='train'))
    assert list(subset.indices) == list(reference_train.indices)


def test_val_and_train_subsets_are_disjoint_and_exhaustive():
    dataset = list(range(2000))
    val, _ = run_metrics.scored_subset(dataset, cfg())
    train, _ = run_metrics.scored_subset(dataset, cfg(split='train'))
    assert set(val.indices) & set(train.indices) == set()
    assert set(val.indices) | set(train.indices) == set(range(2000))


def test_reproduces_at_the_real_dataset_size():
    """200k pairs, 90/10, seed 3072 -- the actual run's configuration.

    19,999 rather than a round 20,000: `1 - 0.9` is 0.09999999999999998, so
    `floor(0.0999... * 200000)` is 19999 and the leftover row goes to train.
    `lejepa.py` computes its lengths the same way, so this matches it exactly
    -- which is the point, and is what the index assertion actually proves.
    """
    dataset = list(range(200_000))
    _, reference_val = training_split(dataset)
    subset, description = run_metrics.scored_subset(dataset, cfg())
    assert len(subset) == 19_999
    assert list(subset.indices) == list(reference_val.indices)
    assert 'val split: 19999 of 200000' in description


def test_a_wrong_train_split_does_not_silently_reproduce():
    """The guard-rail: mismatching train_split must change the indices, so a
    misconfigured row cannot pass for a held-out one."""
    dataset = list(range(2000))
    _, reference_val = training_split(dataset, train_split=0.9)
    subset, _ = run_metrics.scored_subset(dataset, cfg(train_split=0.8))
    assert list(subset.indices) != list(reference_val.indices)


def test_a_wrong_seed_does_not_silently_reproduce():
    dataset = list(range(2000))
    _, reference_val = training_split(dataset, seed=3072)
    subset, _ = run_metrics.scored_subset(dataset, cfg(seed=1))
    assert list(subset.indices) != list(reference_val.indices)


# --------------------------------------------------------------------------
# the escape hatches, and refusing anything else
# --------------------------------------------------------------------------


def test_all_returns_the_whole_dataset_and_says_so():
    dataset = list(range(2000))
    subset, description = run_metrics.scored_subset(dataset, cfg(split='all'))
    assert len(subset) == 2000
    assert 'INCLUDES training data' in description


def test_subset_is_indexable_so_embed_dataset_needs_no_change():
    """embed_dataset/recalibrate_batchnorm take it as an ordinary dataset."""
    dataset = list(range(100, 2100))
    subset, _ = run_metrics.scored_subset(dataset, cfg())
    assert subset[0] == dataset[subset.indices[0]]
    assert len(subset) == 199


@pytest.mark.parametrize('bad', ['validation', 'holdout', '', 'VAL'])
def test_unknown_split_is_refused(bad):
    with pytest.raises(ValueError, match='split must be'):
        run_metrics.scored_subset(list(range(100)), cfg(split=bad))


def test_defaults_to_val_when_unset():
    dataset = list(range(2000))
    _, reference_val = training_split(dataset)
    bare = OmegaConf.create({'seed': 3072, 'train_split': 0.9})
    subset, _ = run_metrics.scored_subset(dataset, bare)
    assert list(subset.indices) == list(reference_val.indices)

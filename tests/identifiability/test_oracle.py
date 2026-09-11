"""The supervised ceiling: does it actually separate observable from not?

The script's whole claim is that a latent it cannot read is a latent no
self-supervised objective could read either. That claim is only worth anything
if the procedure *can* read a latent that is genuinely in the pixels, so both
directions are tested here on synthetic frames where the answer is known by
construction.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from stable_worldmodel.wm.lejepa.module import CNNEncoder, NDimHead


SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'identifiability'


def _load_oracle():
    """Import run_oracle.py, which lives outside the package."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        'run_oracle', SCRIPTS / 'run_oracle.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load_oracle()


class _Model(torch.nn.Module):
    """The shape `run_oracle` expects: `.encoder` then `.head`."""

    def __init__(self, n):
        super().__init__()
        self.encoder = CNNEncoder(channels=(8, 16), proj_dim=16, image_size=32)
        self.head = NDimHead(input_dim=16, output_dim=n)


def _frames(n_rows, seed=0):
    """z[0] is painted into the frame; z[1] is never rendered.

    So the ceiling must be high for the first coordinate and ~0 for the
    second -- the synthetic analogue of a latent the renderer drops.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_rows, 2)).astype(np.float32)

    pixels = rng.normal(scale=0.1, size=(n_rows, 3, 32, 32)).astype(np.float32)
    # A bar whose brightness is z[0]. Nothing anywhere depends on z[1].
    pixels[:, 0, 8:24, 8:24] += z[:, 0, None, None]
    return torch.from_numpy(pixels), torch.from_numpy(z)


# --------------------------------------------------------------------------
# per_latent_r2
# --------------------------------------------------------------------------


def test_per_latent_r2_perfect_and_useless():
    truth = np.array([[1.0, 5.0], [2.0, 5.5], [3.0, 6.0], [4.0, 7.0]])
    perfect = oracle.per_latent_r2(truth, truth.copy())
    assert perfect == pytest.approx([1.0, 1.0])

    mean_only = np.repeat(truth.mean(axis=0, keepdims=True), 4, axis=0)
    assert oracle.per_latent_r2(truth, mean_only) == pytest.approx([0.0, 0.0])


def test_per_latent_r2_constant_column_is_zero_not_nan():
    truth = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    scores = oracle.per_latent_r2(truth, truth.copy())
    assert scores[1] == 0.0


# --------------------------------------------------------------------------
# reinitialise -- the claim that no pretrained feature survives
# --------------------------------------------------------------------------


def test_reinitialise_discards_learned_weights():
    """Every learned weight is replaced; normalisation affines go back to 1/0.

    Split this way because the two behave differently and only one of them is
    random: a conv or linear weight is redrawn, while a BatchNorm/LayerNorm
    affine resets to exactly ``(1, 0)`` -- deterministic, and equally free of
    anything the checkpoint had learned. Asserting "every tensor changed" would
    fail on the second group for the wrong reason.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    model = _Model(2)
    # Drift the affines away from their canonical init, so resetting them back
    # is an observable event rather than a no-op.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (_BatchNorm, torch.nn.LayerNorm)):
                module.weight.add_(0.37)
                module.bias.add_(-0.21)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    count = oracle.reinitialise(model)
    assert count > 0

    normed = set()
    for name, module in model.named_modules():
        if isinstance(module, (_BatchNorm, torch.nn.LayerNorm)):
            assert torch.all(module.weight == 1.0), f'{name} weight not reset'
            assert torch.all(module.bias == 0.0), f'{name} bias not reset'
            normed |= {f'{name}.weight', f'{name}.bias'}

    learned = [
        name
        for name, p in model.named_parameters()
        if name not in normed and p.numel() > 1
    ]
    assert learned, 'no learned weights found -- the test proves nothing'
    for name in learned:
        assert not torch.equal(
            before[name], dict(model.named_parameters())[name]
        ), f'{name} survived the reset'


def test_reinitialise_resets_batchnorm_running_stats():
    from torch.nn.modules.batchnorm import _BatchNorm

    model = _Model(2)
    model.train()
    model.head(model.encoder(_frames(8)[0]))
    norms = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    assert any(m.num_batches_tracked.item() > 0 for m in norms)

    oracle.reinitialise(model)
    assert all(m.num_batches_tracked.item() == 0 for m in norms)


# --------------------------------------------------------------------------
# end to end: observable vs not
# --------------------------------------------------------------------------


def test_ceiling_separates_rendered_from_unrendered_latent():
    """The interpretation table, verified: high for z[0], ~0 for z[1]."""
    from omegaconf import OmegaConf

    pixels, z = _frames(768, seed=1)
    cfg = OmegaConf.create(
        {
            'lr': 3.0e-3,
            'weight_decay': 1.0e-4,
            'batch_size': 64,
            'epochs': 12,
            'warmup_steps': 10,
            'constant_frac': 0.5,
            'seed': 0,
        }
    )

    rng = np.random.default_rng(0)
    order = rng.permutation(len(pixels))
    train_idx, test_idx = order[:640], order[640:]

    model = _Model(2)
    oracle.reinitialise(model)
    model = oracle.train_supervised(model, pixels, z, train_idx, cfg, 'cpu')
    prediction = oracle.predict(model, pixels, test_idx, 'cpu')
    scores = oracle.per_latent_r2(z[test_idx].numpy(), prediction)

    assert scores[0] > 0.5, f'rendered latent unreadable: {scores[0]:.3f}'
    assert scores[1] < 0.2, f'unrendered latent read anyway: {scores[1]:.3f}'
    assert scores[0] > scores[1] + 0.4


# --------------------------------------------------------------------------
# load_views -- the RAM-blowup this module exists to prevent
# --------------------------------------------------------------------------
#
# The bug this guards against: a prior version built a Python list of every
# decoded frame and fed it to `torch.cat`, so the list and the concatenated
# output coexisted in host RAM. At max_samples=20000 and 224px that peaked
# around 48 GB and SIGKILLed the process (OOM) before a single training epoch
# ran. `load_views` now streams into a disk-backed `np.memmap` instead.


class _FakeDataset:
    """Just enough of the real dataset's interface for `load_views`."""

    def __init__(self, n_rows, n_latent=2, frame_shape=(3, 8, 8), seed=0):
        rng = np.random.default_rng(seed)
        self.transform = None
        self.pixels = rng.normal(size=(n_rows, 2, *frame_shape)).astype(
            np.float32
        )
        self.z = rng.normal(size=(n_rows, 2, n_latent)).astype(np.float32)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, i):
        return {'pixels': self.pixels[i], 'latent/z': self.z[i]}


def test_load_views_is_memmap_backed_not_a_ram_tensor():
    """The core regression guard: pixels must never be one dense RAM array."""
    dataset = _FakeDataset(n_rows=6)
    with tempfile.TemporaryDirectory() as cache_dir:
        pixels, z = oracle.load_views(dataset, 6, None, cache_dir)
        assert isinstance(pixels, np.memmap)
        assert (Path(cache_dir) / 'oracle_pixels.f32').exists()
        assert pixels.shape == (12, 3, 8, 8)
        assert z.shape == (12, 2)


def test_load_views_flattens_both_views_correctly():
    dataset = _FakeDataset(n_rows=3, n_latent=2, frame_shape=(1, 2, 2))
    with tempfile.TemporaryDirectory() as cache_dir:
        pixels, z = oracle.load_views(dataset, 3, None, cache_dir)
        np.testing.assert_allclose(np.asarray(pixels[0:2]), dataset.pixels[0])
        np.testing.assert_allclose(np.asarray(pixels[2:4]), dataset.pixels[1])
        np.testing.assert_allclose(z[0:2].numpy(), dataset.z[0])


def test_load_views_respects_max_samples():
    dataset = _FakeDataset(n_rows=10)
    with tempfile.TemporaryDirectory() as cache_dir:
        pixels, z = oracle.load_views(dataset, 4, None, cache_dir)
        assert pixels.shape[0] == 8  # 4 pairs x 2 views
        assert z.shape[0] == 8


def test_load_views_cache_dir_is_freed_on_exit():
    """The caller's TemporaryDirectory pattern must actually clean up."""
    dataset = _FakeDataset(n_rows=4)
    with tempfile.TemporaryDirectory() as cache_dir:
        oracle.load_views(dataset, 4, None, cache_dir)
        cache_file = Path(cache_dir) / 'oracle_pixels.f32'
        assert cache_file.exists()
    assert not cache_file.exists()


# --------------------------------------------------------------------------
# _as_batch -- the indexing shim between a memmap and a torch.Tensor
# --------------------------------------------------------------------------


def test_as_batch_indexes_a_memmap_and_returns_only_that_slice():
    with tempfile.TemporaryDirectory() as cache_dir:
        arr = np.memmap(
            Path(cache_dir) / 'x.f32',
            dtype=np.float32,
            mode='w+',
            shape=(100, 4),
        )
        arr[:] = np.arange(400, dtype=np.float32).reshape(100, 4)
        idx = np.array([5, 7, 9])
        out = oracle._as_batch(arr, idx)
        assert torch.is_tensor(out)
        assert out.shape == (3, 4)
        assert torch.equal(out, torch.as_tensor(arr[idx]))


def test_as_batch_passes_a_tensor_through_unchanged():
    t = torch.arange(20.0).reshape(5, 4)
    idx = np.array([1, 3])
    out = oracle._as_batch(t, idx)
    assert torch.is_tensor(out)
    assert torch.equal(out, t[idx])

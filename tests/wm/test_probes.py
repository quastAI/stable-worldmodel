import pytest
import torch
from torch import nn

from stable_worldmodel.wm.probes import attach_probe, get_probe, load_probe


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class SimpleProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


######################
## attach_probe tests ##
######################


def test_attach_probe_creates_probes_dict():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'my_probe', probe)
    assert hasattr(model, '_probes')
    assert 'my_probe' in model._probes


def test_attach_probe_stores_correct_module():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)
    assert model._probes['p'] is probe


def test_attach_probe_multiple_probes():
    model = DummyModel()
    p1 = SimpleProbe()
    p2 = SimpleProbe()
    attach_probe(model, 'a', p1)
    attach_probe(model, 'b', p2)
    assert 'a' in model._probes
    assert 'b' in model._probes


def test_attach_probe_requires_nn_module():
    model = DummyModel()
    with pytest.raises(AssertionError):
        attach_probe(model, 'bad', lambda x: x)


def test_attach_probe_overwrite():
    model = DummyModel()
    p1 = SimpleProbe()
    p2 = SimpleProbe()
    attach_probe(model, 'p', p1)
    attach_probe(model, 'p', p2)
    assert model._probes['p'] is p2


####################
## get_probe tests ##
####################


def test_get_probe_returns_none_without_probes():
    model = DummyModel()
    assert get_probe(model, 'missing') is None


def test_get_probe_returns_none_for_missing_key():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'a', probe)
    assert get_probe(model, 'b') is None


def test_get_probe_returns_attached_probe():
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)
    assert get_probe(model, 'p') is probe


#####################
## load_probe tests ##
#####################


def test_load_probe_from_module(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    path = tmp_path / 'probe.pt'
    torch.save(probe, path)

    load_probe(model, 'p', path)

    loaded = get_probe(model, 'p')
    assert loaded is not None
    assert isinstance(loaded, SimpleProbe)


def test_load_probe_from_state_dict(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)

    path = tmp_path / 'state.pt'
    torch.save(probe.state_dict(), path)

    load_probe(model, 'p', path)

    loaded = get_probe(model, 'p')
    assert loaded is not None


def test_load_probe_state_dict_no_probe_raises(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    path = tmp_path / 'state.pt'
    torch.save(probe.state_dict(), path)

    with pytest.raises(ValueError, match='No probe found'):
        load_probe(model, 'missing', path)


def test_load_probe_state_dict_updates_weights(tmp_path):
    model = DummyModel()
    probe = SimpleProbe()
    attach_probe(model, 'p', probe)

    # Save modified weights
    new_probe = SimpleProbe()
    with torch.no_grad():
        new_probe.fc.weight.fill_(99.0)
    path = tmp_path / 'state.pt'
    torch.save(new_probe.state_dict(), path)

    load_probe(model, 'p', path)

    assert torch.allclose(
        get_probe(model, 'p').fc.weight, torch.full_like(probe.fc.weight, 99.0)
    )


###############################
## LinearProbe / MLPProbe tests ##
###############################

import numpy as np  # noqa: E402

from stable_worldmodel.wm.probes import (  # noqa: E402
    LinearProbe,
    MLPProbe,
    StandardizedProbe,
)


def test_linear_probe_shapes():
    probe = LinearProbe(8, 3)
    assert probe(torch.randn(5, 8)).shape == (5, 3)


def test_mlp_probe_shapes():
    probe = MLPProbe(8, 3, hidden_dim=16, num_layers=2)
    assert probe(torch.randn(5, 8)).shape == (5, 3)


def test_mlp_probe_zero_layers_is_linear_width():
    probe = MLPProbe(8, 3, num_layers=0)
    assert probe(torch.randn(5, 8)).shape == (5, 3)
    assert len(probe.net) == 1


def test_probe_stats_buffers_are_in_state_dict():
    probe = LinearProbe(4, 2)
    keys = probe.state_dict().keys()
    for name in ('feature_mean', 'feature_std', 'target_mean', 'target_std'):
        assert name in keys


def test_set_feature_stats_standardizes():
    probe = LinearProbe(3, 1, standardize=True)
    probe.set_feature_stats([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    x = torch.tensor([[3.0, 6.0, 9.0]])
    assert torch.allclose(probe.standardize(x), torch.ones(1, 3))


def test_set_feature_stats_floors_zero_std():
    probe = LinearProbe(2, 1)
    probe.set_feature_stats([0.0, 0.0], [0.0, 1.0])
    assert probe.feature_std.min() > 0


def test_set_feature_stats_rejects_wrong_dim():
    probe = LinearProbe(3, 1)
    with pytest.raises(ValueError, match='feature stats'):
        probe.set_feature_stats([1.0, 2.0], [1.0, 1.0])


def test_standardize_disabled_is_identity():
    probe = LinearProbe(3, 1, standardize=False)
    probe.set_feature_stats([5.0, 5.0, 5.0], [2.0, 2.0, 2.0])
    x = torch.randn(4, 3)
    assert torch.allclose(probe.standardize(x), x)


def test_set_target_stats_and_unscale_roundtrip():
    probe = LinearProbe(3, 2)
    probe.set_target_stats([1.0, -1.0], [2.0, 0.5])
    y = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(probe.unscale(y), torch.tensor([[3.0, 0.0]]))


def test_set_target_stats_rejects_wrong_dim():
    probe = LinearProbe(3, 2)
    with pytest.raises(ValueError, match='target stats'):
        probe.set_target_stats([0.0], [1.0])


def test_predict_applies_unscale():
    probe = LinearProbe(2, 1)
    probe.set_target_stats([10.0], [1.0])
    x = torch.randn(3, 2)
    assert torch.allclose(probe.predict(x), probe(x) + 10.0)


def test_set_weights_accepts_both_orientations():
    probe = LinearProbe(4, 2)
    weight = np.arange(8, dtype=np.float32).reshape(4, 2)  # (in, out)
    probe.set_weights(weight)
    assert torch.allclose(probe.fc.weight, torch.tensor(weight).t())

    probe2 = LinearProbe(4, 2)
    probe2.set_weights(weight.T)  # (out, in)
    assert torch.allclose(probe2.fc.weight, torch.tensor(weight).t())


def test_set_weights_with_bias():
    probe = LinearProbe(3, 2)
    probe.set_weights(np.zeros((3, 2), dtype=np.float32), bias=[1.0, 2.0])
    assert torch.allclose(probe.fc.bias, torch.tensor([1.0, 2.0]))


def test_set_weights_rejects_bad_shape():
    probe = LinearProbe(3, 2)
    with pytest.raises(ValueError, match='weight must be'):
        probe.set_weights(np.zeros((5, 2), dtype=np.float32))


def test_set_weights_rejects_bad_bias_length():
    probe = LinearProbe(3, 2)
    with pytest.raises(ValueError, match='bias must have'):
        probe.set_weights(np.zeros((3, 2), dtype=np.float32), bias=[1.0])


def test_set_weights_without_bias_term_raises():
    probe = LinearProbe(3, 2, bias=False)
    with pytest.raises(ValueError, match='bias=False'):
        probe.set_weights(np.zeros((3, 2), dtype=np.float32), bias=[1.0, 2.0])


def test_closed_form_ridge_recovers_a_linear_map():
    """A LinearProbe loaded from a least-squares solve must reproduce it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    true_w = rng.normal(size=(5, 2)).astype(np.float32)
    y = x @ true_w

    weight = np.linalg.lstsq(x, y, rcond=None)[0]
    probe = LinearProbe(5, 2, bias=False, standardize=False)
    probe.set_weights(weight)

    pred = probe(torch.tensor(x)).detach().numpy()
    assert np.allclose(pred, y, atol=1e-3)


def test_probe_roundtrips_through_attach_and_save(tmp_path):
    model = DummyModel()
    probe = MLPProbe(4, 2, hidden_dim=8)
    probe.set_feature_stats([0.0] * 4, [1.0] * 4)
    attach_probe(model, 'mlp', probe)

    path = tmp_path / 'probe.pt'
    torch.save(probe, path)

    fresh = DummyModel()
    load_probe(fresh, 'mlp', path)
    loaded = get_probe(fresh, 'mlp')
    assert isinstance(loaded, MLPProbe)
    x = torch.randn(3, 4)
    assert torch.allclose(loaded(x), probe(x))


def test_standardized_probe_is_an_nn_module():
    assert issubclass(StandardizedProbe, nn.Module)


def test_set_weights_zeroes_the_bias_by_default():
    """A closed-form solve on centred variables has no intercept; inheriting
    nn.Linear's random bias would offset every prediction."""
    probe = LinearProbe(3, 2)
    probe.fc.bias.data.fill_(5.0)
    probe.set_weights(np.zeros((3, 2), dtype=np.float32))
    assert torch.allclose(probe.fc.bias, torch.zeros(2))


def test_set_weights_default_bias_gives_an_exact_linear_map():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4)).astype(np.float32)
    weight = rng.normal(size=(4, 2)).astype(np.float32)
    probe = LinearProbe(4, 2, standardize=False)
    probe.set_weights(weight)
    pred = probe(torch.tensor(x)).detach().numpy()
    assert np.allclose(pred, x @ weight, atol=1e-4)

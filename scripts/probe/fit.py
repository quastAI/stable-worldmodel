"""Fitting the read-outs: the cheap half, run on cached frozen features.

Three rungs per (feature variant, target), so a number is only ever read as
a difference against the rung below it:

``baseline``
    Predict the training mean (regression) or the training majority class.
    Defines "learned nothing". Its test R² is ~0 by construction and can go
    slightly negative when the test split's mean differs from the train
    split's — which is exactly what an episode-level split makes possible,
    and worth seeing.
``linear``
    Regression: **closed-form ridge**, with the penalty chosen on the
    validation split. Closed form on purpose — an SGD-fitted linear probe
    conflates "the information is not linearly decodable" with "the
    optimizer did not converge", and a probing result must not depend on
    that. Classification: multinomial logistic regression, weight decay
    chosen on validation.
``mlp``
    One hidden layer by default, same optimizer and schedule as the logistic
    probe. The linear→MLP gap is the interesting quantity: information that
    is *present but not linearly decodable*.

Conventions that make the numbers comparable:

  * Features are whitened with **training-split** statistics only.
  * Regression targets are standardized with training statistics for the
    fit, and predictions are mapped back before any metric is computed, so
    R² and MAE are in physical units.
  * R² is the uniform average over target dims, each against that dim's
    test-split variance — so a 12-dim target is not dominated by its
    highest-variance coordinate.
  * Model selection (ridge penalty, weight decay, early stopping) reads the
    validation split. The test split is touched once, at the end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from stable_worldmodel.wm.probes import LinearProbe, MLPProbe

import targets as tg
from features import default_device


@dataclass
class FitConfig:
    """Hyperparameters of the probe fits.

    Args:
        probes: Which rungs to run, from ``('baseline', 'linear', 'mlp')``.
        ridge_alphas: Penalties swept for the closed-form linear regression
            probe; selected on validation R².
        weight_decays: Penalties swept for the logistic (linear
            classification) probe; selected on validation accuracy.
        mlp_hidden_dim: Hidden width of the MLP probe.
        mlp_layers: Hidden layers in the MLP probe.
        mlp_dropout: Dropout inside the MLP probe.
        epochs: Maximum epochs for gradient-fitted probes.
        patience: Stop after this many epochs without a validation
            improvement.
        batch_size: Minibatch size for gradient-fitted probes. Capped so
            that each epoch takes at least ``min_steps_per_epoch``
            optimizer steps — otherwise a small split would give one step
            per epoch and every gradient-fitted score would be
            optimizer-limited rather than capacity-limited.
        min_steps_per_epoch: Floor on optimizer steps per epoch.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay for the MLP probe.
        device: ``None`` autodetects.
        seed: Seeds probe init and minibatch order.
    """

    probes: tuple[str, ...] = ('baseline', 'linear', 'mlp')
    ridge_alphas: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        1e1,
        1e2,
        1e3,
        1e4,
    )
    weight_decays: tuple[float, ...] = (1e-4, 1e-2, 1.0)

    mlp_hidden_dim: int = 512
    mlp_layers: int = 1
    mlp_dropout: float = 0.0

    epochs: int = 200
    patience: int = 25
    batch_size: int = 1024
    min_steps_per_epoch: int = 8
    lr: float = 3e-3
    weight_decay: float = 1e-4

    device: str | None = None
    seed: int = 0

    def __post_init__(self):
        bad = [p for p in self.probes if p not in ('baseline', 'linear', 'mlp')]
        if bad:
            raise ValueError(f'unknown probe kinds {bad}')


@dataclass
class ProbeResult:
    """One row of the results table."""

    variant: str
    target: str
    group: str
    kind: str
    probe: str
    feature_dim: int
    output_dim: int
    label_step: int
    n_train: int
    n_val: int
    n_test: int
    n_train_effective: int = 0
    episode_constant: bool = False
    score: float = float('nan')
    val_score: float = float('nan')
    metrics: dict = field(default_factory=dict)
    hyper: dict = field(default_factory=dict)
    fit_seconds: float = 0.0

    def as_row(self) -> dict:
        row = {
            k: getattr(self, k)
            for k in (
                'variant',
                'target',
                'group',
                'kind',
                'probe',
                'feature_dim',
                'output_dim',
                'label_step',
                'n_train',
                'n_val',
                'n_test',
                'n_train_effective',
                'episode_constant',
                'score',
                'val_score',
                'fit_seconds',
            )
        }
        row.update(self.metrics)
        row.update({f'hp_{k}': v for k, v in self.hyper.items()})
        return row


###################
##    metrics    ##
###################


def r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-dim ``1 - SS_res / SS_tot``, ``SS_tot`` from ``y_true``'s mean."""
    resid = ((y_true - y_pred) ** 2).sum(axis=0)
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    # A dim with no variance in this split carries no information about the
    # probe; report it as 0 rather than -inf.
    out = np.ones_like(total, dtype=np.float64)
    nonzero = total > 0
    out[nonzero] = 1.0 - resid[nonzero] / total[nonzero]
    out[~nonzero] = 0.0
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R² (uniform average), variance-weighted R², MAE and RMSE."""
    per_dim = r2_per_dim(y_true, y_pred)
    resid = ((y_true - y_pred) ** 2).sum()
    total = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return {
        'r2': float(per_dim.mean()),
        'r2_weighted': float(1.0 - resid / total) if total > 0 else 0.0,
        'r2_min_dim': float(per_dim.min()),
        'mae': float(np.abs(y_true - y_pred).mean()),
        'rmse': float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        'r2_dims': [float(v) for v in per_dim],
    }


def classification_metrics(
    y_true: np.ndarray, logits: np.ndarray, num_classes: int
) -> dict:
    """Accuracy, macro-averaged recall, and cross-entropy."""
    pred = logits.argmax(axis=1)
    acc = float((pred == y_true).mean())

    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            recalls.append(float((pred[mask] == c).mean()))
    log_probs = logits - logits.max(axis=1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
    return {
        'acc': acc,
        'balanced_acc': float(np.mean(recalls)) if recalls else float('nan'),
        'nll': float(-log_probs[np.arange(len(y_true)), y_true].mean()),
    }


###################
##   baselines   ##
###################


def _stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim mean and std of ``arr``, with a floor on the std."""
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return mean, np.maximum(std, 1e-6)


def fit_baseline(target: tg.ProbeTarget, y: dict) -> tuple[float, dict]:
    """Constant predictor fitted on train, scored on test."""
    if target.kind == 'regression':
        train_mean = y['train'].mean(axis=0, keepdims=True)
        pred = np.broadcast_to(train_mean, y['test'].shape)
        metrics = regression_metrics(y['test'], pred)
        return metrics['r2'], metrics
    counts = np.bincount(y['train'], minlength=target.num_classes)
    logits = np.tile(
        np.log(np.maximum(counts, 1e-12)).astype(np.float64),
        (len(y['test']), 1),
    )
    metrics = classification_metrics(y['test'], logits, target.num_classes)
    return metrics['acc'], metrics


###################
##  linear probe ##
###################


def _ridge_weights(
    xs: torch.Tensor, ys: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Solve ``(XᵀX + αI) W = XᵀY`` in float64. ``xs``/``ys`` are whitened."""
    dim = xs.shape[1]
    gram = xs.T @ xs
    gram = gram + alpha * torch.eye(dim, dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram, xs.T @ ys)


def fit_ridge(
    x: dict, y: dict, cfg: FitConfig, device: str
) -> tuple[LinearProbe, float, dict, dict]:
    """Closed-form linear regression probe with a validation-chosen penalty.

    Args:
        x: ``{split: (N, D)}`` raw features.
        y: ``{split: (N, K)}`` raw targets.
        cfg: Fit config (supplies ``ridge_alphas``).
        device: Where to run the solve.

    Returns:
        ``(probe, val_r2, test_metrics, hyper)``. The probe predicts
        *standardized* targets; ``probe.predict`` returns physical units.
    """
    xm, xsd = _stats(x['train'])
    ym, ysd = _stats(y['train'])

    # float64 for the Gram matrix; MPS has no float64, so anything that is
    # not CUDA solves on the CPU (the solve is milliseconds either way).
    solve_device = device if device == 'cuda' else 'cpu'
    xs = {
        s: torch.as_tensor(
            (x[s] - xm) / xsd, dtype=torch.float64, device=solve_device
        )
        for s in x
    }
    ys = {
        s: torch.as_tensor(
            (y[s] - ym) / ysd, dtype=torch.float64, device=solve_device
        )
        for s in y
    }

    best = (-np.inf, None, None)
    for alpha in cfg.ridge_alphas:
        weight = _ridge_weights(xs['train'], ys['train'], float(alpha))
        pred = (xs['val'] @ weight).cpu().numpy() * ysd + ym
        score = regression_metrics(y['val'], pred)['r2']
        if score > best[0]:
            best = (score, float(alpha), weight)

    val_score, alpha, weight = best
    # Not fatal, but a sweep should bracket its own optimum.
    edges = {cfg.ridge_alphas[0]: 'lower', cfg.ridge_alphas[-1]: 'upper'}
    edge = (
        f'alpha hit the {edges[alpha]} end of the sweep'
        if alpha in edges
        else ''
    )

    pred_test = (xs['test'] @ weight).cpu().numpy() * ysd + ym
    metrics = regression_metrics(y['test'], pred_test)

    probe = LinearProbe(x['train'].shape[1], y['train'].shape[1])
    probe.set_feature_stats(xm, xsd)
    probe.set_target_stats(ym, ysd)
    probe.set_weights(weight.float().cpu().numpy())
    return probe, val_score, metrics, {'alpha': alpha, 'warning': edge}


###################
## gradient fits ##
###################


def _train_torch_probe(
    probe: nn.Module,
    x: dict,
    y: dict,
    target: tg.ProbeTarget,
    cfg: FitConfig,
    device: str,
    weight_decay: float,
) -> tuple[float, np.ndarray, int]:
    """Fit ``probe`` with AdamW, early-stopping on the validation score.

    Returns:
        ``(best_val_score, test_predictions, steps)`` — logits for
        classification, standardized targets for regression, plus the number
        of optimizer steps actually taken (recorded so an under-converged
        fit is visible in the results rather than mistaken for a low-capacity
        one).
    """
    is_cls = target.kind == 'classification'
    loss_fn = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()

    xt = {
        s: torch.as_tensor(v, dtype=torch.float32, device=device)
        for s, v in x.items()
    }
    if is_cls:
        yt = {
            s: torch.as_tensor(v, dtype=torch.long, device=device)
            for s, v in y.items()
        }
    else:
        yt = {
            s: torch.as_tensor(v, dtype=torch.float32, device=device)
            for s, v in y.items()
        }

    probe = probe.to(device)
    opt = torch.optim.AdamW(
        probe.parameters(), lr=cfg.lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    generator = torch.Generator(device='cpu').manual_seed(cfg.seed)
    n_train = len(xt['train'])
    # Keep the optimizer-step count from collapsing on a small split.
    batch_size = max(
        1, min(cfg.batch_size, n_train // max(cfg.min_steps_per_epoch, 1))
    )
    best_score, best_state, stale, steps = -np.inf, None, 0, 0

    for _ in range(cfg.epochs):
        probe.train()
        perm = torch.randperm(n_train, generator=generator).to(device)
        for lo in range(0, n_train, batch_size):
            idx = perm[lo : lo + batch_size]
            opt.zero_grad(set_to_none=True)
            loss_fn(probe(xt['train'][idx]), yt['train'][idx]).backward()
            opt.step()
            steps += 1
        sched.step()

        probe.eval()
        with torch.no_grad():
            out = probe(xt['val']).cpu().numpy()
        score = _score(target, y['val'], out)

        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().clone() for k, v in probe.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_out = probe(xt['test']).cpu().numpy()
    return float(best_score), test_out, steps


def _score(target: tg.ProbeTarget, y_true: np.ndarray, out: np.ndarray):
    """Primary validation score: accuracy or R² (both higher-is-better).

    Regression predictions arrive standardized; R² is invariant to a shared
    affine rescaling of prediction and truth, so scoring in standardized
    space matches scoring in physical units.
    """
    if target.kind == 'classification':
        return float((out.argmax(axis=1) == y_true).mean())
    return float(r2_per_dim(y_true, out).mean())


def fit_gradient_probe(
    kind: str,
    x: dict,
    y: dict,
    target: tg.ProbeTarget,
    cfg: FitConfig,
    device: str,
) -> tuple[nn.Module, float, dict, dict]:
    """Fit a linear (logistic) or MLP probe by gradient descent.

    Args:
        kind: ``'linear'`` or ``'mlp'``.
        x: ``{split: (N, D)}`` raw features.
        y: ``{split: (N, K)}`` regression targets or ``{split: (N,)}``
            class indices.
        target: The probing target.
        cfg: Fit config.
        device: Torch device.

    Returns:
        ``(probe, val_score, test_metrics, hyper)``.
    """
    feature_dim = x['train'].shape[1]
    is_cls = target.kind == 'classification'
    output_dim = target.num_classes if is_cls else y['train'].shape[1]

    xm, xsd = _stats(x['train'])
    if is_cls:
        ym = np.zeros((1, output_dim), dtype=np.float32)
        ysd = np.ones((1, output_dim), dtype=np.float32)
        y_fit = y
    else:
        ym, ysd = _stats(y['train'])
        y_fit = {s: (v - ym) / ysd for s, v in y.items()}

    # A linear probe fitted by gradient descent only makes sense for
    # classification; regression goes through the closed form.
    decays = cfg.weight_decays if kind == 'linear' else (cfg.weight_decay,)

    best = (-np.inf, None, None, None, 0)
    for weight_decay in decays:
        torch.manual_seed(cfg.seed)
        if kind == 'linear':
            probe = LinearProbe(feature_dim, output_dim)
        else:
            probe = MLPProbe(
                feature_dim,
                output_dim,
                hidden_dim=cfg.mlp_hidden_dim,
                num_layers=cfg.mlp_layers,
                dropout=cfg.mlp_dropout,
            )
        probe.set_feature_stats(xm, xsd)
        probe.set_target_stats(ym, ysd)

        val_score, test_out, steps = _train_torch_probe(
            probe, x, y_fit, target, cfg, device, float(weight_decay)
        )
        if val_score > best[0]:
            best = (val_score, probe, test_out, float(weight_decay), steps)

    val_score, probe, test_out, weight_decay, steps = best
    if is_cls:
        metrics = classification_metrics(
            y['test'], test_out.astype(np.float64), target.num_classes
        )
    else:
        metrics = regression_metrics(y['test'], test_out * ysd + ym)
    return (
        probe,
        val_score,
        metrics,
        {'weight_decay': weight_decay, 'steps': steps},
    )


###################
##   the sweep   ##
###################


def _split_labels(target: tg.ProbeTarget, payload: dict, step: int) -> dict:
    return {
        split: tg.build_labels(target, payload['labels'][split], step)
        for split in payload['labels']
    }


def fit_one(
    rung: str,
    x: dict,
    y: dict,
    target: tg.ProbeTarget,
    cfg: FitConfig,
    device: str,
):
    """Fit a single rung and return ``(probe, score, val_score, metrics, hp)``.

    The linear rung splits by target kind: closed-form ridge for regression,
    gradient-fitted logistic regression for classification.
    """
    if rung == 'baseline':
        score, metrics = fit_baseline(target, y)
        return None, score, float('nan'), metrics, {}

    if rung == 'linear' and target.kind == 'regression':
        probe, val_score, metrics, hyper = fit_ridge(x, y, cfg, device)
        return probe, metrics['r2'], val_score, metrics, hyper

    probe, val_score, metrics, hyper = fit_gradient_probe(
        rung, x, y, target, cfg, device
    )
    key = 'acc' if target.kind == 'classification' else 'r2'
    return probe, metrics[key], val_score, metrics, hyper


def fit_all(
    payload: dict,
    probe_targets,
    cfg: FitConfig,
    variants=None,
    keep_probes: bool = False,
    progress: bool = True,
) -> tuple[list[dict], dict]:
    """Fit every (variant, target, rung) combination on a feature cache.

    Args:
        payload: A cache from :func:`features.extract` / ``load_features``.
        probe_targets: Targets from :func:`targets.select_targets`.
        cfg: Fit config.
        variants: Feature variants to probe; ``None`` uses every variant in
            the cache.
        keep_probes: Also return the fitted modules, keyed by
            ``(variant, target, probe)``.
        progress: Print a line per (variant, target).

    Returns:
        ``(rows, probes)`` — result rows ready for a table, and the fitted
        probes (empty unless ``keep_probes``).
    """
    device = cfg.device or default_device()
    meta = payload['meta']
    variants = list(variants or meta['feature_dims'])

    unknown = [v for v in variants if v not in payload['features']['train']]
    if unknown:
        raise KeyError(
            f'variants {unknown} are not in this cache; it holds '
            f'{sorted(payload["features"]["train"])}'
        )

    rows: list[dict] = []
    probes: dict = {}

    # An episode-constant label repeats identically across every window of
    # its episode, so its effective training size is the episode count.
    n_train_episodes = len(
        np.unique(payload['windows']['train']['episode_idx'])
    )

    for variant in variants:
        x = {s: payload['features'][s][variant] for s in payload['features']}
        step = meta['variant_label_step'][variant]

        for target in probe_targets:
            y = _split_labels(target, payload, step)
            shared = dict(
                variant=variant,
                target=target.name,
                group=target.group,
                kind=target.kind,
                feature_dim=x['train'].shape[1],
                output_dim=(
                    target.num_classes
                    if target.kind == 'classification'
                    else y['train'].shape[1]
                ),
                label_step=step,
                n_train=len(x['train']),
                n_val=len(x['val']),
                n_test=len(x['test']),
                n_train_effective=(
                    n_train_episodes
                    if target.episode_constant
                    else len(x['train'])
                ),
                episode_constant=target.episode_constant,
            )

            for rung in cfg.probes:
                started = time.time()
                probe, score, val_score, metrics, hyper = fit_one(
                    rung, x, y, target, cfg, device
                )

                result = ProbeResult(
                    probe=rung,
                    score=float(score),
                    val_score=float(val_score),
                    metrics=metrics,
                    hyper=hyper,
                    fit_seconds=time.time() - started,
                    **shared,
                )
                rows.append(result.as_row())
                if keep_probes and probe is not None:
                    probes[(variant, target.name, rung)] = probe

            if progress:
                summary = ' '.join(
                    f'{r["probe"]}={r["score"]:.3f}'
                    for r in rows[-len(cfg.probes) :]
                )
                print(f'{variant:>14s} / {target.name:<18s} {summary}')

    return rows, probes


def prediction_fidelity(payload: dict) -> dict:
    """How close ``pred_emb`` sits to ``emb_next_true``, per split.

    Not a probing result — a sanity check on the world model itself. A
    ``pred_emb`` that probes far worse than ``emb_next_true`` while sitting
    right on top of it here would mean the two variants got mismatched.
    """
    out = {}
    for split, feats in payload['features'].items():
        if 'pred_emb' not in feats or 'emb_next_true' not in feats:
            continue
        pred = feats['pred_emb'].astype(np.float64)
        true = feats['emb_next_true'].astype(np.float64)
        cos = (pred * true).sum(1) / (
            np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-12
        )
        out[split] = {
            'cosine_mean': float(cos.mean()),
            'cosine_std': float(cos.std()),
            'mse': float(((pred - true) ** 2).mean()),
            'relative_mse': float(
                ((pred - true) ** 2).mean() / (true**2).mean()
            ),
        }
    return out


__all__ = [
    'FitConfig',
    'ProbeResult',
    'classification_metrics',
    'fit_all',
    'fit_baseline',
    'fit_gradient_probe',
    'fit_one',
    'fit_ridge',
    'prediction_fidelity',
    'r2_per_dim',
    'regression_metrics',
]

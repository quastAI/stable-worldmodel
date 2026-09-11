"""Supervised ceiling: what could this architecture read from these pixels?

**The one question the rest of the suite cannot answer.** Every metric in
``run_metrics.py`` is conditional on the encoder it is handed. When
``cube.size`` scores an R^2 of 0.002, three explanations are consistent with
that number and the suite cannot separate them:

1. LeJEPA's objective did not pick the latent up;
2. this CNN cannot represent it -- six stride-2 stages take a 224px frame to a
   3x3 map and then average it globally, which is a lot of downsampling for a
   20px cube's *size*;
3. the renderer does not put it in the pixels at all, in which case no ``f``
   exists with ``f(g(z)) = Qz`` and the theory's optimum is unreachable by
   construction rather than by any failure of the method.

``observability_ceiling`` looks like it addresses this, but it takes the probe
scores as an *input*: it declares a latent dead because the probe read ~0, then
reports what the aggregate could reach assuming it stays there. That is a
correct aggregate correction and a circular attribution.

This script closes the loop by training the **same architecture** on the
**same pixels** with the labels handed to it. If a supervised network cannot
read ``cube.size``, then no self-supervised objective on this data can, and the
0.002 is a fact about the renderer or the backbone rather than a result about
LeJEPA. Each per-latent score is an upper bound for every row in
``run_metrics.py`` scored on the same dataset.

**Why it re-initialises the checkpoint rather than declaring an architecture.**
The comparison is only sound if the architecture is identical, and a duplicated
spec in a second config drifts. So the checkpoint is loaded for its *shape*,
every parameter is re-initialised, and the training is supervised from scratch.
``init=pretrained`` skips the reset to ask the different question of how far the
learned features get with a trained head on top.

Read the result as a ceiling, not as a target:

===================  =================  ==============================
supervised R^2       LeJEPA R^2         reading
===================  =================  ==============================
~0                   ~0                 not in the pixels, or not
                                        representable by this backbone.
                                        Not a LeJEPA failure
high                 ~0                 the information is there and
                                        reachable; the objective did
                                        not select it
high                 high               recovered
===================  =================  ==============================

Usage::

    python scripts/identifiability/run_oracle.py \\
        checkpoint=lejepa/weights_epoch_100.pt
"""

import os
import sys
from pathlib import Path


if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = (
        'glfw'
        if sys.platform == 'darwin' or os.environ.get('DISPLAY')
        else 'osmesa'
    )

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger as logging  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.identifiability import metrics as ident_metrics  # noqa: E402
from stable_worldmodel.identifiability import results as ident_results  # noqa: E402
from stable_worldmodel.identifiability.collect import load_manifest  # noqa: E402
from stable_worldmodel.wm.lejepa.schedule import (  # noqa: E402
    WarmupHoldCosineLR,
    schedule_kwargs,
)
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_metrics import (  # noqa: E402
    _epoch_of,
    circular_targets,
    encoder_preprocessor,
    latent_names,
)


def reinitialise(model):
    """Reset every parameter, so no pretrained feature survives.

    The claim this script makes is that its scores are an upper bound reachable
    *from the pixels*, which is only true if the run starts from scratch. Any
    module exposing ``reset_parameters`` is reset; BatchNorm running statistics
    are reset too, since those are state the checkpoint carries outside
    ``parameters()``.

    Returns:
        int: How many modules were reset -- logged, because a silent zero here
        would turn the ceiling into a fine-tune of the very encoder it is
        supposed to bound.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    count = 0
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.reset_running_stats()
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            count += 1
    return count


def load_views(dataset, max_samples, transform):
    """Both views of every pair, flattened into independent labelled frames.

    The OU pair structure is irrelevant here -- there is no alignment term to
    compute -- so each frame is its own training example and the two views
    double the supervised sample at no rendering cost.

    Returns:
        tuple: ``(pixels, z)`` with ``pixels`` a ``(2B, C, H, W)`` float tensor
        and ``z`` a ``(2B, n)`` float tensor.
    """
    dataset.transform = transform
    n = min(len(dataset), max_samples)

    pixels, z = [], []
    for i in range(n):
        row = dataset[i]
        frames = torch.as_tensor(np.asarray(row['pixels'])).float()
        latents = torch.as_tensor(
            np.asarray(row['latent/z']), dtype=torch.float32
        )
        pixels.append(frames)
        z.append(latents)

    return torch.cat(pixels, dim=0), torch.cat(z, dim=0)


def train_supervised(model, pixels, z, train_idx, cfg, device):
    """Fit ``pixels -> z`` by least squares, end to end.

    Plain MSE on z-space coordinates. ``z`` is unit-variance per coordinate by
    construction (the collector's design), so the coordinates are already
    commensurate and no per-latent weighting is needed -- which matters,
    because a weighting would silently decide which latent the ceiling is
    generous to.

    The schedule is the training script's own
    :class:`~stable_worldmodel.wm.lejepa.schedule.WarmupHoldCosineLR`, so the
    ceiling is not handicapped by a worse optimiser than the thing it bounds.
    """
    model = model.to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    batch = int(cfg.batch_size)
    steps_per_epoch = max(1, len(train_idx) // batch)
    total_steps = int(cfg.epochs) * steps_per_epoch
    scheduler = WarmupHoldCosineLR(
        optimiser,
        **schedule_kwargs(
            min(int(cfg.warmup_steps), max(1, total_steps - 1)),
            total_steps,
            float(cfg.constant_frac),
        ),
    )

    generator = torch.Generator().manual_seed(int(cfg.seed))
    for epoch in range(int(cfg.epochs)):
        order = train_idx[
            torch.randperm(len(train_idx), generator=generator).numpy()
        ]
        running = 0.0
        for step in range(steps_per_epoch):
            rows = order[step * batch : (step + 1) * batch]
            x = pixels[rows].to(device)
            y = z[rows].to(device)

            prediction = model.head(model.encoder(x))
            loss = torch.nn.functional.mse_loss(prediction, y)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.detach())

        logging.info(
            f'  epoch {epoch + 1:>3d}/{int(cfg.epochs)}  '
            f'mse {running / steps_per_epoch:.5f}  '
            f'lr {optimiser.param_groups[0]["lr"]:.2e}'
        )
    return model


@torch.no_grad()
def predict(model, pixels, rows, device, batch_size=64):
    """Held-out predictions in eval mode."""
    model = model.to(device).eval()
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        x = pixels[chunk].to(device)
        out.append(model.head(model.encoder(x)).cpu().numpy())
    return np.concatenate(out)


def per_latent_r2(truth, prediction):
    """One R^2 per coordinate, on the held-out rows."""
    scores = []
    for j in range(truth.shape[1]):
        column = truth[:, j]
        ss_tot = ((column - column.mean()) ** 2).sum()
        ss_res = ((column - prediction[:, j]) ** 2).sum()
        scores.append(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return np.array(scores)


@hydra.main(version_base=None, config_path='./config', config_name='oracle')
def run(cfg: DictConfig):
    """Train the supervised ceiling and append one results row."""
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    model = load_pretrained(cfg.checkpoint)
    img_size, transform = encoder_preprocessor(model)
    if cfg.init == 'scratch':
        reset = reinitialise(model)
        logging.info(
            f're-initialised {reset} modules (supervised from scratch)'
        )
    elif cfg.init == 'pretrained':
        logging.warning(
            'init=pretrained keeps the LeJEPA weights, so this is a fine-tune '
            'of the encoder under test, NOT an independent ceiling on it'
        )
    else:
        raise ValueError(
            f"init must be 'scratch' or 'pretrained', got {cfg.init!r}"
        )

    dataset = swm.data.load_dataset(
        cfg.ou_dataset,
        num_steps=2,
        frameskip=1,
        keys_to_load=['pixels', 'latent/z'],
    )
    manifest = load_manifest(cfg.ou_dataset, cfg.get('cache_dir'))
    names = latent_names(manifest)

    pixels, z = load_views(dataset, int(cfg.max_samples), transform)
    logging.info(
        f'{len(pixels)} labelled frames at {img_size}px, n = {z.shape[1]}'
    )

    rng = np.random.default_rng(int(cfg.seed))
    order = rng.permutation(len(pixels))
    cut = int(round(len(order) * (1.0 - float(cfg.holdout))))
    train_idx, test_idx = order[:cut], order[cut:]

    model = train_supervised(model, pixels, z, train_idx, cfg, device)

    truth = z[test_idx].numpy()
    prediction = predict(model, pixels, test_idx, device)
    scores = per_latent_r2(truth, prediction)

    result = {
        'metric_suite_version': ident_metrics.METRIC_SUITE_VERSION,
        'probe_kind': f'oracle_supervised_{cfg.init}',
        'oracle_r2': float(scores.mean()),
        'oracle_per_latent': scores.tolist(),
        'probe_latent_names': names,
        'oracle_epochs': int(cfg.epochs),
        'n_train_frames': int(len(train_idx)),
    }

    angles, orders, circ_names = circular_targets(manifest, truth)
    if angles is not None:
        harmonic = ident_metrics.probe_circular(
            prediction, angles, orders, names=circ_names, seed=int(cfg.seed)
        )
        result.update({f'oracle_{k}': v for k, v in harmonic.items()})

    logging.info('supervised ceiling, per latent:')
    for i in np.argsort(scores)[::-1]:
        flag = '  <-- not readable from the pixels' if scores[i] < 0.05 else ''
        logging.info(f'  {names[i]:<20s} {scores[i]:+.4f}{flag}')
    logging.info(f'mean {scores.mean():+.4f}')

    row = ident_results.build_row(
        checkpoint=str(cfg.checkpoint),
        dataset=str(cfg.ou_dataset),
        seed=int(cfg.seed),
        metrics=result,
        manifest=manifest,
        program_constants=OmegaConf.to_container(
            cfg.program_constants, resolve=True
        ),
        epoch=_epoch_of(cfg.checkpoint),
        n_samples=int(len(test_idx)),
    )
    out = Path(cfg.results_path)
    ident_results.append_row(out, row)
    logging.success(f'appended 1 oracle row -> {out}')


if __name__ == '__main__':
    run()

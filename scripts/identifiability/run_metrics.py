"""Score a checkpoint with the frozen metric suite and append to the scatter.

Takes an encoder checkpoint plus **two** eval sets -- the OU set and the
rollout set -- and emits one scatter row per distribution. Reporting both is
not optional: the encoder trains on OU-set states and the planner only ever
sees physics rollouts, and the plan refuses to assert that gap away. It is a
logged quantity.

Usage::

    python scripts/identifiability/run_metrics.py \\
        checkpoint=lejepa/weights_epoch_100.pt \\
        ou_dataset=ogbench/cube_single_ou.lance \\
        rollout_dataset=ogbench/cube_single_predictor.lance
"""

import os
import sys
from pathlib import Path


if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = (
        'glfw' if sys.platform == 'darwin' or os.environ.get('DISPLAY')
        else 'osmesa'
    )

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger as logging  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import stable_pretraining as spt  # noqa: E402
from stable_pretraining import data as dt  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.identifiability import metrics as ident_metrics  # noqa: E402
from stable_worldmodel.identifiability.collect import load_manifest  # noqa: E402
from stable_worldmodel.identifiability import scatter as ident_scatter  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402


def encoder_preprocessor(model):
    """The **exact** preprocessing the encoder was trained under.

    Not optional and not cosmetic. `scripts/train/lejepa.py` feeds the encoder
    ImageNet-normalised tensors (range about [-2.1, 2.6]); a dataset loaded
    without a transform yields raw [0, 255]. The layouts match, so nothing
    raises -- the encoder is simply evaluated ~100x outside its input range and
    every column of the scatter becomes a measurement of that, not of
    identifiability.

    `image_size` is read off the checkpoint's own encoder config rather than a
    config field here, so a run cannot be scored at a resolution the encoder
    was never trained at.
    """
    encoder = model.encoder
    # Two conventions: HF backbones carry it on `.config`, the paper CNN on the
    # module. Resolved rather than assumed, because guessing here silently
    # rescales every frame the metrics are computed on.
    img_size = getattr(
        getattr(encoder, 'config', None), 'image_size', None
    ) or getattr(encoder, 'image_size', None)
    if img_size is None:
        raise AttributeError(
            f'cannot determine the input resolution of '
            f'{type(encoder).__name__}: expose `image_size` on the module or '
            '`config.image_size` as HF backbones do. Scoring at the wrong '
            'resolution silently invalidates every column of the scatter.'
        )
    img_size = int(img_size)
    return img_size, spt.data.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet, source='pixels', target='pixels'
        ),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


@torch.no_grad()
def embed_dataset(model, dataset, max_samples, device, batch_size=64):
    """Embed both views of up to ``max_samples`` pairs.

    Returns:
        tuple: ``(z, h, z_next, h_next)`` as numpy arrays.
    """
    model = model.to(device).eval()
    n = min(len(dataset), max_samples)

    z, h, z_next, h_next = [], [], [], []
    for start in range(0, n, batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, n))]
        pixels = torch.stack(
            [torch.as_tensor(np.asarray(r['pixels'])) for r in rows]
        ).to(device)
        latents = np.stack([np.asarray(r['latent/z']) for r in rows])

        emb = model.encode({'pixels': pixels.float()})['emb'].cpu().numpy()
        z.append(latents[:, 0])
        z_next.append(latents[:, 1])
        h.append(emb[:, 0])
        h_next.append(emb[:, 1])

    return (
        np.concatenate(z),
        np.concatenate(h),
        np.concatenate(z_next),
        np.concatenate(h_next),
    )


@hydra.main(
    version_base=None, config_path='./config', config_name='metrics'
)
def run(cfg: DictConfig):
    """Score one checkpoint on both distributions."""
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    model = load_pretrained(cfg.checkpoint)
    img_size, transform = encoder_preprocessor(model)
    logging.info(
        f'scoring {cfg.checkpoint} (m = {model.output_dim}, '
        f'img_size = {img_size})'
    )

    # The style probe is content-matched by construction, so it is embedded
    # once and reused for both distributions: style invariance is a property
    # of the encoder, not of the distribution being scored.
    h_style_a = h_style_b = None
    if cfg.get('style_dataset'):
        style_set = swm.data.load_dataset(
            cfg.style_dataset,
            num_steps=2,
            frameskip=1,
            keys_to_load=['pixels', 'latent/z'],
        )
        style_set.transform = transform
        z_s, h_style_a, z_s_next, h_style_b = embed_dataset(
            model, style_set, int(cfg.max_samples), device
        )
        # A real OU step at rho=0.9 puts max|dz| near 2.0; a probe at
        # rho=1-1e-8 puts it near 7e-4. Anything above 1e-2 is a step, not
        # rounding, and the metric would then be scoring the transition as if
        # it were style leakage.
        drift = float(np.abs(z_s - z_s_next).max())
        if drift > 1e-2:
            logging.warning(
                f'style probe {cfg.style_dataset} has content drift '
                f'{drift:.3e} between views -- it was not collected at '
                'rho ~ 1, so style_sensitivity also absorbs the OU step.'
            )
        logging.info(f'style probe: {len(h_style_a)} content-matched pairs')

    rows = []
    for distribution, dataset_name in (
        ('ou', cfg.ou_dataset),
        ('rollout', cfg.get('rollout_dataset')),
    ):
        if not dataset_name:
            logging.warning(
                f'no {distribution} dataset given -- the OU/rollout gap is a '
                'measurement the plan asks for, so this row is incomplete.'
            )
            continue

        dataset = swm.data.load_dataset(
            dataset_name,
            num_steps=2,
            frameskip=1,
            keys_to_load=['pixels', 'latent/z'],
        )
        dataset.transform = transform
        z, h, z_next, h_next = embed_dataset(
            model, dataset, int(cfg.max_samples), device
        )
        manifest = load_manifest(dataset_name, cfg.get('cache_dir'))
        rho = manifest.get('ou', {}).get('rho', cfg.program_constants.rho)

        scores = ident_metrics.compute_all(
            z, h, z_next, h_next, rho=rho, seed=int(cfg.seed),
            h_style_a=h_style_a, h_style_b=h_style_b,
        )
        logging.info(
            f'[{distribution}] procrustes/dim '
            f'{scores["procrustes_mse_per_dim"]:.4f}  orth '
            f'{scores["orth_err_normalized"]:.4f}  cond {scores["cond"]:.1f}  '
            f'probe {scores["probe_linear_r2"]:.4f}'
        )

        rows.append(
            ident_scatter.build_row(
                arm=cfg.arm,
                violation=manifest.get('violation', {}).get('key', 'none'),
                severity=manifest.get('violation', {}).get('severity', 0.0),
                seed=int(cfg.seed),
                distribution=distribution,
                metrics=scores,
                manifest=manifest,
                program_constants=OmegaConf.to_container(
                    cfg.program_constants, resolve=True
                ),
                success_rate=cfg.get('success_rate'),
                success_rate_oracle=cfg.get('success_rate_oracle'),
                success_rate_state=cfg.get('success_rate_state'),
                checkpoint=str(cfg.checkpoint),
                n_samples=int(len(z)),
            )
        )

    out = Path(cfg.scatter_path)
    total = ident_scatter.append_rows(out, rows)
    logging.success(f'appended {len(rows)} rows -> {out} ({total} total)')


if __name__ == '__main__':
    run()

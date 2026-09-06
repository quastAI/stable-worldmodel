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

import json
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

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.identifiability import metrics as ident_metrics  # noqa: E402
from stable_worldmodel.identifiability import scatter as ident_scatter  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402


def load_manifest(dataset_name, cache_dir):
    path = (
        Path(cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / f'{Path(dataset_name).stem}_manifest.json'
    )
    if not path.exists():
        raise FileNotFoundError(
            f'no manifest at {path}. Every scatter row must carry the dataset '
            'config it came from; a row without it cannot be told apart later '
            'from one configured differently.'
        )
    with open(path) as handle:
        return json.load(handle)


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
    logging.info(f'scoring {cfg.checkpoint} (m = {model.output_dim})')

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
        z, h, z_next, h_next = embed_dataset(
            model, dataset, int(cfg.max_samples), device
        )
        manifest = load_manifest(dataset_name, cfg.get('cache_dir'))
        rho = manifest.get('ou', {}).get('rho', cfg.program_constants.rho)

        scores = ident_metrics.compute_all(
            z, h, z_next, h_next, rho=rho, seed=int(cfg.seed)
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

"""Score an encoder checkpoint with the frozen metric suite.

Scores the **encoder alone** -- ``LeJEPA`` has no dynamics by design, so this
only ever calls ``model.encode(...)['emb']``.

Two things it does that the training-time diagnostics cannot, and they are the
reason a row from here is the authoritative one:

**It scores in eval mode, on the whole eval set.** The training-time keys are
per-batch and, for anything reading ``h[0] - h[1]``, are measured under
train-mode BatchNorm, where ``h`` is a function of the batch rather than of one
frame. ``delta`` measured 0.13 that way against 2.69 here on the same weights.

**It reports per latent.** A mean over ten latents whose observability through
the renderer spans two orders of magnitude is not a number about the encoder.
``probe_linear_per_latent`` and ``probe_mlp_per_latent`` are where a failure is
actually localised, and the linear/non-linear pair separates "the encoder never
saw this quantity" from "it saw it and did not linearise it".

By default it **refreshes BatchNorm running statistics** over the eval set
before scoring; see :func:`recalibrate_batchnorm` for why that is not a
cosmetic step.

Usage::

    python scripts/identifiability/run_metrics.py \\
        checkpoint=lejepa/weights_epoch_25.pt \\
        program_constants.lambda=5.0e-2
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

import stable_pretraining as spt  # noqa: E402
from stable_pretraining import data as dt  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.identifiability import metrics as ident_metrics  # noqa: E402
from stable_worldmodel.identifiability import results as ident_results  # noqa: E402
from stable_worldmodel.identifiability.collect import load_manifest  # noqa: E402
from stable_worldmodel.identifiability.latents import (  # noqa: E402
    CIRCULAR_ORDER,
)
from stable_worldmodel.wm.lejepa.module import state_dict_hash  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402


def encoder_preprocessor(model):
    """The **exact** preprocessing the encoder was trained under.

    Not optional and not cosmetic. `scripts/train/lejepa.py` feeds the encoder
    ImageNet-normalised tensors (range about [-2.1, 2.6]); a dataset loaded
    without a transform yields raw [0, 255]. The layouts match, so nothing
    raises -- the encoder is simply evaluated ~100x outside its input range and
    every column of the results row becomes a measurement of that, not of
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
            'resolution silently invalidates every column of the row.'
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


def resolve_rho(manifest, fallback):
    """The dataset's scalar rho, across two manifest schemas.

    Datasets collected before 2026-09-11 (the OU sampler's rewrite to
    scalar-only) carry the OLD `OUSampler.describe()` output, where `rho` is
    `self.rho.tolist()` -- a per-DIMENSION array left over from an anisotropic
    sampling capability this pipeline no longer has -- alongside a separate
    scalar `rho_mean`. The current sampler writes a plain float under `rho`
    and no `rho_mean` at all. `manifest.get('ou', {}).get('rho', ...)` reads
    the old array as-is, which is what raised
    ``TypeError: float() argument must be ... not 'list'`` here.

    This does not silently average over a real spread. The old sampler COULD
    assign different rho per dimension, and if it did here, the data does not
    have the single shared rho every rho-dependent metric below assumes
    (`alignment_floor`, `bound/*`, `residual_hermite_degree`) -- and the
    paper's own App. F proves isotropy is *necessary* for the simultaneous
    objective to be well-posed at all. So the per-dimension values are checked
    for uniformity before `rho_mean` is trusted, and a real spread raises
    rather than getting quietly averaged away.

    Returns:
        float: The resolved rho.

    Raises:
        ValueError: If an old-schema manifest's per-dimension rho is not
            uniform to within floating-point noise.
    """
    ou = manifest.get('ou', {})

    if 'rho_mean' in ou:
        per_dim = np.asarray(ou.get('rho', ou['rho_mean']), dtype=np.float64)
        spread = float(per_dim.max() - per_dim.min()) if per_dim.size else 0.0
        if spread > 1e-6:
            raise ValueError(
                f'manifest records a NON-uniform per-dimension rho (spread '
                f'{spread:.2e}, values {per_dim.tolist()}). This dataset was '
                'collected under the pre-refactor anisotropic-capable '
                'sampler and does not satisfy the isotropy every rho-'
                'dependent metric here assumes -- averaging it away would '
                'silently misreport `bound/*`. Re-collect with the current '
                'collect_cube_single_ou.py (isotropic-only) before scoring.'
            )
        logging.info(
            f'manifest is old-schema (pre-2026-09-11 OU sampler); rho is '
            f'uniform at {ou["rho_mean"]:.6f}, using it'
        )
        return float(ou['rho_mean'])

    if 'rho' in ou:
        return float(ou['rho'])

    return float(fallback)


def latent_names(manifest):
    """Flat per-coordinate names for the content latents, in ``z`` order.

    So a per-latent score is readable without cross-referencing the registry:
    ``cube.pos_xy`` of width 2 becomes ``cube.pos_xy[0]``, ``cube.pos_xy[1]``.
    """
    names = []
    for latent in manifest.get('latents', {}).get('latents', []):
        if latent.get('role') != 'content':
            continue
        width = int(latent.get('dim', 1))
        names += (
            [latent['name']]
            if width == 1
            else [f'{latent["name"]}[{i}]' for i in range(width)]
        )
    return names


def circular_targets(manifest, z):
    """The circular latents of ``z``, in radians, with their symmetry orders.

    An angle sampled on a fundamental domain has two defensible read-outs and
    the suite reports both -- see
    :func:`~stable_worldmodel.identifiability.metrics.probe_circular` for why a
    score of zero on the ordinary probe is ambiguous without this one.

    The conversion is the registry's own affine, read out of the manifest
    rather than by rebuilding the registry (which would need the environment):
    ``physical = center + (half_span / sigma_span) * z``, clipped to the
    declared bounds exactly as :meth:`LatentRegistry.to_physical` does.

    Returns:
        tuple: ``(angles, orders, names)`` with ``angles`` of shape ``(B, k)``
        in radians, or ``(None, None, None)`` when the profile has no circular
        content latent.
    """
    described = manifest.get('latents', {})
    sigma_span = float(described.get('sigma_span', 3.0))

    angles, orders, names = [], [], []
    offset = 0
    for latent in described.get('latents', []):
        width = int(latent.get('dim', 1))
        if latent.get('role') != 'content':
            continue
        order = CIRCULAR_ORDER.get(latent['name'])
        if order is not None:
            low = np.asarray(latent['low'], dtype=np.float64).reshape(-1)
            high = np.asarray(latent['high'], dtype=np.float64).reshape(-1)
            center = 0.5 * (low + high)
            half_span = 0.5 * (high - low)
            for i in range(width):
                raw = (
                    center[i] + (half_span[i] / sigma_span) * z[:, offset + i]
                )
                angles.append(np.clip(raw, low[i], high[i]))
                orders.append(order)
                names.append(
                    latent['name'] if width == 1 else f'{latent["name"]}[{i}]'
                )
        offset += width

    if not angles:
        return None, None, None
    return np.column_stack(angles), orders, names


@torch.no_grad()
def recalibrate_batchnorm(model, dataset, device, max_samples, batch_size=64):
    """Refresh BatchNorm running statistics, with the weights frozen.

    Not cosmetic. PyTorch's BatchNorm default ``momentum=0.1`` is an
    exponential window of roughly **ten batches**, so the running statistics a
    checkpoint carries are an estimate of the input marginal taken over the
    last ~2.5k frames of training. Under this profile's style randomisation --
    all lighting, both backgrounds, every material and colour, resampled per
    frame -- that marginal is extremely heterogeneous and ten batches is a
    noisy sample of it. Worse, under a **constant** learning-rate schedule the
    weights never stop moving, so the statistics are permanently stale relative
    to them and cannot converge by construction. A cosine anneal to zero hides
    this by letting the weights settle at the end; the exploration schedule
    does not.

    Everything downstream consumes the encoder in eval mode, so those stale
    statistics are what every reported number is computed through. This does
    one pass with the parameters untouched and ``momentum=None`` (a true
    cumulative average over the pass), which replaces the ten-batch window with
    an estimate over the whole sample.

    Only the BatchNorm modules are put in train mode -- not the model -- so
    nothing else changes behaviour.

    Returns:
        bool: Whether any BatchNorm module was found and refreshed.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    modules = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not modules:
        return False

    model.eval()
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    n = min(len(dataset), max_samples)
    for start in range(0, n, batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, n))]
        pixels = torch.stack(
            [torch.as_tensor(np.asarray(r['pixels'])) for r in rows]
        ).to(device)
        model.encode({'pixels': pixels.float()})

    model.eval()
    logging.info(
        f'refreshed running statistics of {len(modules)} BatchNorm modules '
        f'over {n} pairs'
    )
    return True


@hydra.main(version_base=None, config_path='./config', config_name='metrics')
def run(cfg: DictConfig):
    """Score one checkpoint and append one results row."""
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    model = load_pretrained(cfg.checkpoint)
    img_size, transform = encoder_preprocessor(model)
    logging.info(
        f'scoring {cfg.checkpoint} (m = {model.output_dim}, '
        f'img_size = {img_size})'
    )

    dataset = swm.data.load_dataset(
        cfg.ou_dataset,
        num_steps=2,
        frameskip=1,
        keys_to_load=['pixels', 'latent/z'],
    )
    dataset.transform = transform
    manifest = load_manifest(cfg.ou_dataset, cfg.get('cache_dir'))
    rho = resolve_rho(manifest, cfg.program_constants.rho)

    model = model.to(device)
    recalibrated = False
    if cfg.get('recalibrate_batchnorm', True):
        recalibrated = recalibrate_batchnorm(
            model, dataset, device, int(cfg.max_samples)
        )

    # The style probe is content-matched by construction: its two views share
    # their content exactly and differ only in style, which is what lets
    # `delta` split into nonlinearity and style leakage. Without it the bound
    # charges all of it to nonlinearity.
    h_style_a = h_style_b = None
    if cfg.get('style_dataset'):
        # Armed by default, so a missing probe must degrade rather than abort:
        # the rest of the suite is still worth recording, and
        # `has_style_probe` marks the row as one whose `delta` could not be
        # split. Only an *absent* dataset is tolerated -- a corrupt one raises.
        try:
            style_set = swm.data.load_dataset(
                cfg.style_dataset,
                num_steps=2,
                frameskip=1,
                keys_to_load=['pixels', 'latent/z'],
            )
        except (OSError, ValueError) as error:
            logging.warning(
                f'style probe {cfg.style_dataset} could not be loaded '
                f'({error}). Continuing without it: sigma_sq is unavailable, '
                'so `delta` cannot be split and the reported bound will '
                'attribute all of it to nonlinearity. Collect it with\n'
                '  python scripts/data/collect_cube_single_ou.py \\\n'
                f'      dataset_name={cfg.style_dataset} \\\n'
                '      program_constants.rho=0.99999999 num_pairs=20000'
            )
            style_set = None

        if style_set is not None:
            style_set.transform = transform
            z_s, h_style_a, z_s_next, h_style_b = embed_dataset(
                model, style_set, int(cfg.max_samples), device
            )
            # A real OU step at rho=0.9 puts max|dz| near 2.0; a probe at
            # rho=1-1e-8 puts it near 7e-4. Above 1e-2 it is a step, not
            # rounding, and the metric would score the transition as if it
            # were style leakage.
            drift = float(np.abs(z_s - z_s_next).max())
            if drift > 1e-2:
                logging.warning(
                    f'style probe has content drift {drift:.3e} between '
                    'views -- it was not collected at rho ~ 1, so '
                    'style_sensitivity also absorbs the OU step.'
                )
            logging.info(
                f'style probe: {len(h_style_a)} content-matched pairs'
            )

    z, h, _, h_next = embed_dataset(
        model, dataset, int(cfg.max_samples), device
    )

    circular_angles, circular_orders, circular_names = circular_targets(
        manifest, z
    )
    if circular_angles is not None:
        logging.info(
            f'harmonic probe armed for {circular_names} '
            f'(orders {circular_orders})'
        )

    scores = ident_metrics.compute_all(
        z,
        h,
        h_next=h_next,
        rho=rho,
        names=latent_names(manifest),
        seed=int(cfg.seed),
        h_style_a=h_style_a,
        h_style_b=h_style_b,
        circular_angles=circular_angles,
        circular_orders=circular_orders,
        circular_names=circular_names,
    )

    _report(scores)

    row = ident_results.build_row(
        checkpoint=str(cfg.checkpoint),
        dataset=str(cfg.ou_dataset),
        seed=int(cfg.seed),
        metrics=scores,
        manifest=manifest,
        program_constants=OmegaConf.to_container(
            cfg.program_constants, resolve=True
        ),
        epoch=_epoch_of(cfg.checkpoint),
        encoder_hash=state_dict_hash(model),
        bn_recalibrated=recalibrated,
        n_samples=int(len(z)),
    )
    out = Path(cfg.results_path)
    ident_results.append_row(out, row)
    logging.success(
        f'appended 1 row -> {out} '
        f'({len(ident_results.load_rows(out, latest_only=False))} total)'
    )


def _epoch_of(checkpoint):
    """Epoch number out of ``weights_epoch_N.pt``, or ``None``."""
    import re

    match = re.search(r'weights_epoch_(\d+)', str(checkpoint))
    return int(match.group(1)) if match else None


def _report(scores):
    """Log the row in reading order: localise, then aggregate, then the bound."""
    names = scores.get('probe_latent_names') or [
        f'z[{i}]' for i in range(len(scores['probe_linear_per_latent']))
    ]
    logging.info('per-latent read-out (linear | mlp):')
    order = np.argsort(scores['probe_linear_per_latent'])[::-1]
    for i in order:
        logging.info(
            f'  {names[i]:<20s} {scores["probe_linear_per_latent"][i]:+.4f} | '
            f'{scores["probe_mlp_per_latent"][i]:+.4f}'
        )
    logging.info(
        f'dead latents (linear R2 < {ident_metrics.DEAD_LATENT_R2}): '
        f'{[names[i] for i in scores["dead_latents"]]}'
    )

    if scores.get('has_circular_probe'):
        logging.info('circular latents (theta | harmonic read-out):')
        by_name = dict(zip(names, scores['probe_linear_per_latent']))
        for name, order, harmonic in zip(
            scores['probe_circular_names'],
            scores['probe_circular_orders'],
            scores['probe_circular_per_latent'],
        ):
            theta = by_name.get(name, float('nan'))
            if harmonic > 0.25 and harmonic > theta + 0.15:
                verdict = (
                    f'CIRCULAR CODING (m={order}), theta probe misreads it'
                )
            elif max(theta, harmonic) < 0.05:
                verdict = 'not represented in either coding'
            else:
                verdict = 'linearised'
            logging.info(
                f'  {name:<20s} {theta:+.4f} | {harmonic:+.4f}   {verdict}'
            )
    logging.info(
        f'recovered {scores["recovered_dimensions"]:.2f} of '
        f'{scores["trace_cov"]:.2f} embedding dimensions across '
        f'{scores["canonical_participation"]:.2f} effective directions'
    )
    logging.info(
        f'procrustes/dim {scores["procrustes_mse_per_dim"]:.4f} '
        f'(floor at this ceiling {scores["procrustes_floor"]:.4f}, '
        f'uninformative isotropic 2.0)  '
        f'orth {scores["orth_err_normalized"]:.4f}  '
        f'cond {scores["cond"]:.1f}'
    )

    if 'delta' not in scores:
        return
    verdict = (
        'OK'
        if scores['delta_admissible']
        else 'IMPOSSIBLE -- these embeddings cannot be eval-mode; do not '
        'quote the bound from this row'
    )
    logging.info(
        f'delta {scores["delta"]:.4f} (content {scores["delta_content"]:.4f}, '
        f'style {scores.get("delta_style", 0.0):.4f})  '
        f'floor {scores["delta_floor"]:.4f}  '
        f'residual degree {scores["residual_hermite_degree"]:.2f}  [{verdict}]'
    )
    logging.info(
        f'bound: D {scores["D"]:.2f}  predicted {scores["predicted_error"]:.2f}'
        f'  vacuous {scores["bound_vacuous"]}  '
        f'(needs mean probe R2 > {scores["mean_probe_r2_needed"]:.3f})'
    )


if __name__ == '__main__':
    run()

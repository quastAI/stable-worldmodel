"""Frozen-feature extraction: the expensive half of the probing experiment.

The encoder is run **once** per configuration and every feature variant,
every label and every window index is written to a single ``.npz``. Probes
are then fitted on that cache, so the linear probe, the MLP probe and all
targets see byte-identical features — which is the whole point of a clean
probing experiment.

Five feature variants come out of one forward pass over a training-shaped
window (``history_size`` context frames plus ``num_preds`` future frames,
``frameskip`` env steps apart):

============== ===================================================== ======
variant        what it is                                            label
============== ===================================================== ======
backbone_cls   ViT CLS token of the current frame, *before* the       t
               projector — the raw encoder representation.
emb            projector output for the current frame; this is what   t
               the LeWM loss and the predictor actually operate on.
emb_hist       ``emb`` concatenated over all context frames; what     t
               the planner's predictor is conditioned on.
pred_emb       ``predict(ctx)`` — the world model's *prediction* of   t+k
               the next latent, never having seen that frame.
emb_next_true  the true ``emb`` of that future frame. Upper bound     t+k
               for ``pred_emb``: the gap is prediction error, not
               representation quality.
pixels_lowres  16x16 average-pooled input frame. Control — a probe    t
               that only beats this baseline has learned nothing the
               raw pixels did not already expose.
============== ===================================================== ======

``t`` is the last context frame (index ``history_size - 1``) and ``t+k`` the
predicted frame (``history_size - 1 + num_preds``).

Two invariants keep the features comparable to training:

1. **The preprocessing is the training preprocessing.** ImageNet stats then
   resize for pixels, and the same z-score column normalizer for actions,
   both lifted from ``scripts/train/lewm.py``. A different action scale
   would silently change ``pred_emb``.
2. **Splits are by episode, never by frame.** Consecutive frames of a
   400-step episode are near-duplicates, so a frame-level split leaks the
   test set into the train set and every probe looks better than it is.
   (Training itself uses a clip-level ``random_split``; that is fine for
   fitting a world model and wrong for measuring one.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as logging
from tqdm import tqdm


SPLITS = ('train', 'val', 'test')

#: Which timestep of the window each variant's label must be read at.
#: ``'current'`` = last context frame, ``'next'`` = the predicted frame.
VARIANT_LABEL_STEP = {
    'backbone_cls': 'current',
    'emb': 'current',
    'emb_hist': 'current',
    'pixels_lowres': 'current',
    'pred_emb': 'next',
    'emb_next_true': 'next',
}

DEFAULT_VARIANTS = (
    'backbone_cls',
    'emb',
    'emb_hist',
    'pred_emb',
    'emb_next_true',
    'pixels_lowres',
)

LOWRES_SIZE = 16


@dataclass
class ExtractConfig:
    """Everything that defines a feature cache.

    Args:
        dataset_name: Passed to :func:`swm.data.load_dataset`.
        checkpoint: Checkpoint relative to ``<checkpoint_root>/checkpoints/``,
            e.g. ``lewm_q4_dr/weights_epoch_11.pt``.
        checkpoint_root: Overrides ``STABLEWM_HOME`` for the checkpoint
            lookup only (the dataset uses ``dataset_cache_dir``).
        dataset_cache_dir: Overrides ``STABLEWM_HOME`` for the dataset.
        random_init: Instantiate the architecture from ``config.json`` but
            skip the weights — the standard untrained-encoder control.
        history_size: Context frames per window (training ``wm.history_size``).
        num_preds: Prediction offset (training ``wm.num_preds``).
        frameskip: Env steps between window frames (training ``frameskip``).
        img_size: Encoder input resolution.
        episodes: How many episodes to draw per split.
        windows_per_episode: Windows sampled per episode.
        variants: Feature variants to store.
        seed: Seeds the episode split and the window sampling.
        batch_size: Windows per forward pass.
        num_workers: DataLoader workers.
        device: ``'cuda'``, ``'mps'``, ``'cpu'`` or ``None`` to autodetect.
        dtype: ``'float32'`` (default, most faithful) or ``'bfloat16'``.
    """

    dataset_name: str = 'ogbench/cube_quadruple_dr_expert.lance'
    checkpoint: str | None = None
    checkpoint_root: str | None = None
    dataset_cache_dir: str | None = None
    random_init: bool = False

    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 5
    img_size: int = 224

    episodes: dict = field(
        default_factory=lambda: {'train': 300, 'val': 60, 'test': 120}
    )
    windows_per_episode: int = 20
    variants: tuple[str, ...] = DEFAULT_VARIANTS

    seed: int = 0
    batch_size: int = 64
    num_workers: int = 4
    device: str | None = None
    dtype: str = 'float32'

    @property
    def num_steps(self) -> int:
        return self.history_size + self.num_preds

    @property
    def current_step(self) -> int:
        """Window index of the frame a ``'current'`` label refers to."""
        return self.history_size - 1

    @property
    def next_step(self) -> int:
        """Window index of the frame ``pred_emb`` predicts."""
        return self.history_size - 1 + self.num_preds

    def __post_init__(self):
        bad = [v for v in self.variants if v not in VARIANT_LABEL_STEP]
        if bad:
            raise ValueError(
                f'unknown feature variants {bad}; expected '
                f'{sorted(VARIANT_LABEL_STEP)}'
            )
        missing = [s for s in SPLITS if s not in self.episodes]
        if missing:
            raise ValueError(f'episodes is missing splits {missing}')
        if self.random_init and self.checkpoint is None:
            raise ValueError(
                'random_init still needs `checkpoint` — the architecture is '
                "read from that run's config.json"
            )


#############################
##      preprocessing      ##
#############################


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """ImageNet-normalize then resize — verbatim from ``train/lewm.py``.

    Duplicated rather than imported because the training entry point is a
    Hydra ``@main`` script. Keep the two in sync: a mismatch here shifts the
    encoder's input distribution away from what it was trained on.
    """
    from stable_pretraining import data as dt

    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def build_dataset(cfg: ExtractConfig, label_columns):
    """Open the Lance table with the training window layout and transform.

    Args:
        cfg: Extraction config.
        label_columns: Privileged/proprio columns the selected targets need.

    Returns:
        The reader, transform attached.
    """
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_worldmodel.data import column_normalizer

    keys_to_load = ['pixels', 'action', *label_columns]

    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.dataset_cache_dir,
        transform=None,
        num_steps=cfg.num_steps,
        frameskip=cfg.frameskip,
        keys_to_load=keys_to_load,
    )

    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        ),
        # Same z-score fit over the full action column as training. Labels
        # are deliberately left raw so metrics stay in physical units.
        # This caches the action column (~40 MB); label columns are read
        # per window instead, which is why their dims come from a sample
        # rather than from get_dim() -- that would cache all of them.
        column_normalizer(dataset, 'action', 'action'),
    ]
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


#############################
##      window sampling    ##
#############################


def episode_split(
    num_episodes: int, counts: dict, seed: int
) -> dict[str, np.ndarray]:
    """Partition episode indices into disjoint train/val/test sets.

    Args:
        num_episodes: Episodes available in the dataset.
        counts: ``{split: n_episodes}``.
        seed: RNG seed for the permutation.

    Returns:
        ``{split: sorted episode indices}``.
    """
    total = sum(counts[s] for s in SPLITS)
    if total > num_episodes:
        raise ValueError(
            f'asked for {total} episodes but the dataset has {num_episodes}'
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)

    out, cursor = {}, 0
    for split in SPLITS:
        n = int(counts[split])
        out[split] = np.sort(perm[cursor : cursor + n])
        cursor += n
    return out


def _clip_index_base(lengths: np.ndarray, span: int):
    """Offsets that turn an ``(episode, start)`` pair into a clip index.

    ``Dataset.clip_indices`` is built episode-major over the episodes long
    enough to hold a window, so the index of ``(ep, start)`` is
    ``base[position_of(ep)] + start``. Computed arithmetically because the
    full list is ~1.9M entries at the shipped config.
    """
    eligible = np.flatnonzero(lengths >= span)
    counts = lengths[eligible] - span + 1
    base = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    position = {int(ep): i for i, ep in enumerate(eligible)}
    return position, base, counts


def sample_windows(
    lengths: np.ndarray,
    episodes: np.ndarray,
    span: int,
    windows_per_episode: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw window start positions inside the given episodes.

    The final row of every episode is excluded: ``World.collect`` rotates
    the action column so the reset frame's placeholder action ends up there
    as ``NaN`` (Run.md section 6). Dropping those windows is cleaner than
    ``nan_to_num``-ing them, since a zeroed action would feed the predictor
    a step that never happened.

    Args:
        lengths: Episode lengths of the whole dataset.
        episodes: Episodes to sample from.
        span: ``num_steps * frameskip``.
        windows_per_episode: Windows to draw per episode (capped by what
            each episode can offer).
        seed: RNG seed.

    Returns:
        ``(clip_indices, episode_idx, start_step)``, sorted by clip index
        for read locality.
    """
    position, base, _ = _clip_index_base(lengths, span)
    rng = np.random.default_rng(seed)

    clips, eps, starts = [], [], []
    for ep in episodes:
        ep = int(ep)
        if ep not in position:
            continue
        # start + span <= length - 1  =>  the NaN action row is never read
        n_valid = int(lengths[ep]) - span
        if n_valid <= 0:
            continue
        take = min(windows_per_episode, n_valid)
        chosen = rng.choice(n_valid, size=take, replace=False)
        clips.append(base[position[ep]] + chosen)
        eps.append(np.full(take, ep, dtype=np.int64))
        starts.append(chosen.astype(np.int64))

    if not clips:
        raise ValueError('no valid windows in the requested episodes')

    clips = np.concatenate(clips)
    eps = np.concatenate(eps)
    starts = np.concatenate(starts)
    order = np.argsort(clips)
    return clips[order], eps[order], starts[order]


#############################
##      model loading      ##
#############################


def default_device() -> str:
    """First available of cuda / mps / cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_checkpoint(cfg: ExtractConfig) -> tuple[Path, dict]:
    """Locate a checkpoint file and read its model config.

    Mirrors :func:`swm.wm.utils.load_pretrained`'s local-path resolution:
    the name is relative to ``<checkpoint_root>/checkpoints/`` and
    ``config.json`` sits next to the ``.pt``.
    """
    import stable_worldmodel as swm

    root = swm.data.utils.get_cache_dir(
        cfg.checkpoint_root, sub_folder='checkpoints'
    )
    path = root / cfg.checkpoint
    if path.suffix != '.pt':
        candidates = sorted(path.glob('*.pt'))
        if len(candidates) != 1:
            raise ValueError(
                f'{path} holds {len(candidates)} .pt files; name one '
                'explicitly (e.g. run/weights_epoch_11.pt)'
            )
        path = candidates[0]
    if not path.exists():
        raise FileNotFoundError(f'checkpoint not found: {path}')

    config_path = path.parent / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            f'config.json not found next to {path.name} in {path.parent}'
        )
    with open(config_path) as f:
        return path, json.load(f)


def build_model(cfg: ExtractConfig):
    """Instantiate the world model, frozen and in eval mode.

    ``eval()`` is not cosmetic here: the LeWM projector ends in a
    ``BatchNorm1d``, so in train mode the features of a window would depend
    on the rest of its batch.

    Returns:
        ``(model, device, torch_dtype)``.
    """
    from hydra.utils import instantiate

    path, model_config = resolve_checkpoint(cfg)

    if cfg.random_init:
        # Fix the init so the control is reproducible across runs.
        torch.manual_seed(cfg.seed)
        logging.info(f'Instantiating a RANDOM-INIT control from {path.parent}')
    else:
        logging.info(f'Loading {path}')

    model = instantiate(model_config)
    if not cfg.random_init:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)

    device = cfg.device or default_device()
    dtype = getattr(torch, cfg.dtype)

    model = model.to(device=device, dtype=dtype)
    model = model.eval()
    model.requires_grad_(False)
    return model, device, dtype


#############################
##       extraction        ##
#############################


@torch.no_grad()
def _forward_window(model, batch, cfg: ExtractConfig, cls_holder):
    """Run one window batch and return every feature variant.

    Reproduces ``lejepa_forward``'s slicing so ``pred_emb`` is exactly the
    quantity the training loss regressed: context is the first
    ``history_size`` frames, and ``predict(...)[:, -1]`` is the prediction
    for window step ``history_size - 1 + num_preds``.
    """
    H = cfg.history_size
    info = {'pixels': batch['pixels'], 'action': batch['action']}

    cls_holder.clear()
    info = model.encode(info)  # hook fills cls_holder with the CLS tokens
    emb = info['emb']  # (B, T, D)
    act_emb = info['act_emb']

    pred = model.predict(emb[:, :H], act_emb[:, :H])  # (B, H, D)

    out = {}
    if 'emb' in cfg.variants:
        out['emb'] = emb[:, cfg.current_step]
    if 'emb_hist' in cfg.variants:
        out['emb_hist'] = emb[:, :H].reshape(emb.size(0), -1)
    if 'pred_emb' in cfg.variants:
        out['pred_emb'] = pred[:, -1]
    if 'emb_next_true' in cfg.variants:
        out['emb_next_true'] = emb[:, cfg.next_step]
    if 'backbone_cls' in cfg.variants:
        if not cls_holder:
            raise RuntimeError(
                'the projector hook captured nothing — model.encode() no '
                'longer routes the CLS token through model.projector'
            )
        cls = cls_holder[0].reshape(emb.size(0), cfg.num_steps, -1)
        out['backbone_cls'] = cls[:, cfg.current_step]
    if 'pixels_lowres' in cfg.variants:
        frame = batch['pixels'][:, cfg.current_step].float()
        pooled = F.adaptive_avg_pool2d(frame, LOWRES_SIZE)
        out['pixels_lowres'] = pooled.flatten(1)

    return out


def extract(
    cfg: ExtractConfig,
    label_columns,
    progress: bool = True,
) -> dict:
    """Encode sampled windows and return features, labels and metadata.

    Args:
        cfg: Extraction config.
        label_columns: Lance columns to keep as labels (see
            :func:`targets.required_columns`).
        progress: Show a per-split progress bar.

    Returns:
        ``{'features': {split: {variant: (N, D)}},
        'labels': {split: {column: (N, num_steps, dim)}},
        'meta': {...}}`` — all arrays float32/int64 NumPy on the host.
    """
    from torch.utils.data import DataLoader, Subset

    label_columns = list(label_columns)
    dataset = build_dataset(cfg, label_columns)
    model, device, dtype = build_model(cfg)

    # Capture the pre-projector CLS token without reimplementing encode():
    # LeWM.encode feeds exactly that tensor to model.projector.
    cls_holder: list[torch.Tensor] = []
    handle = None
    if 'backbone_cls' in cfg.variants:

        def _hook(_module, inputs, _output):
            cls_holder.append(inputs[0].detach())

        handle = model.projector.register_forward_hook(_hook)

    splits = episode_split(len(dataset.lengths), cfg.episodes, cfg.seed)
    span = cfg.num_steps * cfg.frameskip

    features: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, dict[str, np.ndarray]] = {}
    windows: dict[str, dict[str, np.ndarray]] = {}
    column_dims: dict[str, int] = {}

    try:
        for split in SPLITS:
            clips, eps, starts = sample_windows(
                dataset.lengths,
                splits[split],
                span,
                cfg.windows_per_episode,
                # Distinct stream per split; the episode sets are already
                # disjoint, this only decorrelates the start positions.
                seed=cfg.seed + 1 + SPLITS.index(split),
            )
            loader = DataLoader(
                Subset(dataset, clips.tolist()),
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=cfg.num_workers,
                pin_memory=device == 'cuda',
            )

            feat_chunks: dict[str, list[np.ndarray]] = {}
            label_chunks: dict[str, list[np.ndarray]] = {}
            it = tqdm(
                loader,
                desc=f'encode {split} ({len(clips)} windows)',
                disable=not progress,
            )
            for batch in it:
                gpu_batch = {
                    'pixels': batch['pixels'].to(device, dtype=dtype),
                    'action': batch['action'].to(device, dtype=dtype),
                }
                out = _forward_window(model, gpu_batch, cfg, cls_holder)
                for name, tensor in out.items():
                    feat_chunks.setdefault(name, []).append(
                        tensor.float().cpu().numpy()
                    )
                for col in label_columns:
                    arr = batch[col].float().numpy()
                    if arr.ndim != 3:
                        raise ValueError(
                            f"label column '{col}' collated to {arr.shape}; "
                            'expected (B, num_steps, dim)'
                        )
                    column_dims.setdefault(col, int(arr.shape[-1]))
                    label_chunks.setdefault(col, []).append(arr)

            features[split] = {
                k: np.concatenate(v) for k, v in feat_chunks.items()
            }
            labels[split] = {
                k: np.concatenate(v) for k, v in label_chunks.items()
            }
            windows[split] = {
                'episode_idx': eps,
                'start_step': starts,
                'clip_index': clips,
            }
            _check_finite(features[split], split)
    finally:
        if handle is not None:
            handle.remove()

    meta = {
        'config': asdict(cfg),
        'device': device,
        'label_columns': label_columns,
        'column_dims': column_dims,
        'episodes': {s: splits[s].tolist() for s in SPLITS},
        'num_windows': {s: int(len(windows[s]['clip_index'])) for s in SPLITS},
        'feature_dims': {
            k: int(v.shape[1]) for k, v in features['train'].items()
        },
        'variant_label_step': {
            v: (cfg.current_step if s == 'current' else cfg.next_step)
            for v, s in VARIANT_LABEL_STEP.items()
            if v in cfg.variants
        },
    }
    return {
        'features': features,
        'labels': labels,
        'windows': windows,
        'meta': meta,
    }


def _check_finite(feats: dict[str, np.ndarray], split: str) -> None:
    for name, arr in feats.items():
        n_bad = int((~np.isfinite(arr)).any(axis=1).sum())
        if n_bad:
            logging.warning(
                f'{split}/{name}: {n_bad} of {len(arr)} feature rows are '
                'not finite — they will drag every probe on this variant'
            )


#############################
##          cache          ##
#############################


def _key(kind: str, split: str, name: str) -> str:
    # '/' is a path separator inside the npz archive; keep keys flat.
    return f'{kind}|{split}|{name}'.replace('/', '~')


def save_features(path, payload: dict) -> Path:
    """Write a feature cache to ``path`` (``.npz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'meta': np.array(json.dumps(payload['meta']))}
    for kind in ('features', 'labels', 'windows'):
        for split, group in payload[kind].items():
            for name, arr in group.items():
                arrays[_key(kind, split, name)] = arr
    np.savez(path, **arrays)
    logging.info(f'Feature cache written to {path}')
    return path


def load_features(path) -> dict:
    """Read a feature cache written by :func:`save_features`."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(data['meta'].item())
        out = {
            'features': {s: {} for s in SPLITS},
            'labels': {s: {} for s in SPLITS},
            'windows': {s: {} for s in SPLITS},
            'meta': meta,
        }
        for kind in ('features', 'labels', 'windows'):
            for split in SPLITS:
                for name in _names_for(kind, meta):
                    out[kind][split][name] = data[_key(kind, split, name)]
    return out


def _names_for(kind: str, meta: dict):
    if kind == 'features':
        return list(meta['feature_dims'])
    if kind == 'labels':
        return list(meta['label_columns'])
    return ['episode_idx', 'start_step', 'clip_index']


__all__ = [
    'DEFAULT_VARIANTS',
    'SPLITS',
    'VARIANT_LABEL_STEP',
    'ExtractConfig',
    'build_dataset',
    'build_model',
    'episode_split',
    'extract',
    'get_img_preprocessor',
    'load_features',
    'resolve_checkpoint',
    'sample_windows',
    'save_features',
]

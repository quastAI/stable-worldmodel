"""Frozen-feature extraction: the expensive half of the probing experiment.

The encoder is run **once** and the resulting feature — ``emb``, the
projector's output for a single frame — is cached to one ``.npz`` alongside
every label and every frame's provenance. Probes are then fitted on that
cache, so every probe and every target see byte-identical features, which is
the whole point of a clean probing experiment.

``emb`` is not one representation among several: ``LeWM.encode`` reads the
ViT's CLS token (``last_hidden_state[:, 0]``) and passes it through
``model.projector`` to produce ``emb``, and ``model.predict`` consumes
exactly that vector — no other pooling, and no other tensor, sits between
the frame and the planner. Probing it is probing the single learned
representation the rest of the model actually uses. The ViT encodes every
frame independently (no temporal mixing happens before the predictor), so a
frame's ``emb`` does not depend on neighbouring frames — there is no need to
encode a multi-frame window to get it.

Two invariants keep the features comparable to training:

1. **The preprocessing is the training preprocessing.** ImageNet stats then
   resize, lifted from ``scripts/train/lewm.py``. A different resize would
   silently change ``emb``.
2. **Splits are by episode, never by frame.** Frames of a 400-step episode
   are near-duplicates, so a frame-level split leaks the test set into the
   train set and every probe looks better than it is. (Training itself uses
   a clip-level ``random_split``; that is fine for fitting a world model and
   wrong for measuring one.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm


SPLITS = ('train', 'val', 'test')


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
        img_size: Encoder input resolution.
        episodes: How many episodes to draw per split.
        frames_per_episode: Frames sampled per episode.
        seed: Seeds the episode split and the frame sampling.
        batch_size: Frames per forward pass.
        num_workers: DataLoader workers.
        device: ``'cuda'``, ``'mps'``, ``'cpu'`` or ``None`` to autodetect.
        dtype: ``'float32'`` (default, most faithful) or ``'bfloat16'``.
    """

    dataset_name: str = 'ogbench/cube_quadruple_dr_expert.lance'
    checkpoint: str | None = None
    checkpoint_root: str | None = None
    dataset_cache_dir: str | None = None

    img_size: int = 224

    episodes: dict = field(
        default_factory=lambda: {'train': 300, 'val': 60, 'test': 120}
    )
    frames_per_episode: int = 20

    seed: int = 0
    batch_size: int = 64
    num_workers: int = 4
    device: str | None = None
    dtype: str = 'float32'

    def __post_init__(self):
        missing = [s for s in SPLITS if s not in self.episodes]
        if missing:
            raise ValueError(f'episodes is missing splits {missing}')
        if self.checkpoint is None:
            raise ValueError('checkpoint is required')


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
    """Open the Lance table for single-frame reads and attach the transform.

    Args:
        cfg: Extraction config.
        label_columns: Privileged/proprio columns the selected targets need.

    Returns:
        The reader, transform attached. ``num_steps=1`` and ``frameskip=1``
        make every clip index address exactly one frame, so no action
        column or window layout is needed — the encoder is per-frame.
    """
    import stable_worldmodel as swm

    keys_to_load = ['pixels', *label_columns]

    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.dataset_cache_dir,
        transform=None,
        num_steps=1,
        frameskip=1,
        keys_to_load=keys_to_load,
    )
    dataset.transform = get_img_preprocessor(
        source='pixels', target='pixels', img_size=cfg.img_size
    )
    return dataset


#############################
##      frame sampling     ##
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


def _clip_index_base(lengths: np.ndarray):
    """Offsets that turn an ``(episode, frame)`` pair into a clip index.

    ``Dataset.clip_indices`` is built episode-major over every episode (a
    single-frame clip needs no minimum length), so the index of
    ``(ep, frame)`` is ``base[position_of(ep)] + frame``. Computed
    arithmetically because the full list is ~1.9M entries at the shipped
    config.
    """
    position = {int(ep): i for i, ep in enumerate(range(len(lengths)))}
    base = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    return position, base, lengths


def sample_frames(
    lengths: np.ndarray,
    episodes: np.ndarray,
    frames_per_episode: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw frame indices inside the given episodes.

    Args:
        lengths: Episode lengths of the whole dataset.
        episodes: Episodes to sample from.
        frames_per_episode: Frames to draw per episode (capped by episode
            length).
        seed: RNG seed.

    Returns:
        ``(clip_indices, episode_idx, frame_idx)``, sorted by clip index for
        read locality.
    """
    position, base, _ = _clip_index_base(lengths)
    rng = np.random.default_rng(seed)

    clips, eps, frames = [], [], []
    for ep in episodes:
        ep = int(ep)
        length = int(lengths[ep])
        if length <= 0:
            continue
        take = min(frames_per_episode, length)
        chosen = rng.choice(length, size=take, replace=False)
        clips.append(base[position[ep]] + chosen)
        eps.append(np.full(take, ep, dtype=np.int64))
        frames.append(chosen.astype(np.int64))

    if not clips:
        raise ValueError('no valid frames in the requested episodes')

    clips = np.concatenate(clips)
    eps = np.concatenate(eps)
    frames = np.concatenate(frames)
    order = np.argsort(clips)
    return clips[order], eps[order], frames[order]


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
    """Locate a checkpoint file and read its model config."""
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
    ``BatchNorm1d``, so in train mode the features of a batch would depend
    on the rest of that batch.

    Returns:
        ``(model, device, torch_dtype)``.
    """
    from hydra.utils import instantiate

    path, model_config = resolve_checkpoint(cfg)
    logging.info(f'Loading {path}')

    model = instantiate(model_config)
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
def _encode_frame(model, batch) -> np.ndarray:
    """Encode one batch of single-frame clips and return ``emb``.

    Reproduces the slice of ``LeWM.encode`` that produces ``info['emb']`` —
    the projector output for the frame, exactly what ``model.predict``
    conditions on.
    """
    info = {'pixels': batch['pixels']}
    info = model.encode(info)
    return info['emb'][:, 0].float().cpu().numpy()


def extract(
    cfg: ExtractConfig,
    label_columns,
    progress: bool = True,
) -> dict:
    """Encode sampled frames and return features, labels and metadata.

    Args:
        cfg: Extraction config.
        label_columns: Lance columns to keep as labels (see
            :func:`targets.required_columns`).
        progress: Show a per-split progress bar.

    Returns:
        ``{'features': {split: (N, D)}, 'labels': {split: {column: (N, 1,
        dim)}}, 'meta': {...}}`` — all arrays float32/int64 NumPy on the
        host. The label arrays keep a length-1 step axis so
        :func:`targets.build_labels` (written for a windowed cache) reads
        them unchanged at ``step=0``.
    """
    from torch.utils.data import DataLoader, Subset

    label_columns = list(label_columns)
    dataset = build_dataset(cfg, label_columns)
    model, device, dtype = build_model(cfg)

    splits = episode_split(len(dataset.lengths), cfg.episodes, cfg.seed)

    features: dict[str, np.ndarray] = {}
    labels: dict[str, dict[str, np.ndarray]] = {}
    windows: dict[str, dict[str, np.ndarray]] = {}
    column_dims: dict[str, int] = {}

    for split in SPLITS:
        clips, eps, frame_idx = sample_frames(
            dataset.lengths,
            splits[split],
            cfg.frames_per_episode,
            # Distinct stream per split; the episode sets are already
            # disjoint, this only decorrelates which frames get picked.
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

        feat_chunks: list[np.ndarray] = []
        label_chunks: dict[str, list[np.ndarray]] = {}
        it = tqdm(
            loader,
            desc=f'encode {split} ({len(clips)} frames)',
            disable=not progress,
        )
        for batch in it:
            gpu_batch = {'pixels': batch['pixels'].to(device, dtype=dtype)}
            feat_chunks.append(_encode_frame(model, gpu_batch))
            for col in label_columns:
                arr = batch[col].float().numpy()
                if arr.ndim != 3:
                    raise ValueError(
                        f"label column '{col}' collated to {arr.shape}; "
                        'expected (B, num_steps, dim)'
                    )
                column_dims.setdefault(col, int(arr.shape[-1]))
                label_chunks.setdefault(col, []).append(arr)

        features[split] = np.concatenate(feat_chunks)
        labels[split] = {
            k: np.concatenate(v) for k, v in label_chunks.items()
        }
        windows[split] = {
            'episode_idx': eps,
            'frame_idx': frame_idx,
            'clip_index': clips,
        }
        _check_finite(features[split], split)

    meta = {
        'config': asdict(cfg),
        'device': device,
        'label_columns': label_columns,
        'column_dims': column_dims,
        'episodes': {s: splits[s].tolist() for s in SPLITS},
        'num_frames': {s: int(len(windows[s]['clip_index'])) for s in SPLITS},
        'feature_dim': int(features['train'].shape[1]),
    }
    return {
        'features': features,
        'labels': labels,
        'windows': windows,
        'meta': meta,
    }


def _check_finite(feat: np.ndarray, split: str) -> None:
    n_bad = int((~np.isfinite(feat)).any(axis=1).sum())
    if n_bad:
        logging.warning(
            f'{split}/emb: {n_bad} of {len(feat)} feature rows are not '
            'finite — they will drag every probe fitted on this cache'
        )


#############################
##          cache          ##
#############################


def _key(kind: str, split: str, name: str | None = None) -> str:
    # '/' is a path separator inside the npz archive; keep keys flat.
    parts = [kind, split] if name is None else [kind, split, name]
    return '|'.join(parts).replace('/', '~')


def save_features(path, payload: dict) -> Path:
    """Write a feature cache to ``path`` (``.npz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'meta': np.array(json.dumps(payload['meta']))}
    for split, arr in payload['features'].items():
        arrays[_key('features', split)] = arr
    for kind in ('labels', 'windows'):
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
            'features': {},
            'labels': {s: {} for s in SPLITS},
            'windows': {s: {} for s in SPLITS},
            'meta': meta,
        }
        for split in SPLITS:
            out['features'][split] = data[_key('features', split)]
            for name in meta['label_columns']:
                out['labels'][split][name] = data[_key('labels', split, name)]
            for name in ('episode_idx', 'frame_idx', 'clip_index'):
                out['windows'][split][name] = data[_key('windows', split, name)]
    return out


__all__ = [
    'SPLITS',
    'ExtractConfig',
    'build_dataset',
    'build_model',
    'episode_split',
    'extract',
    'get_img_preprocessor',
    'load_features',
    'resolve_checkpoint',
    'sample_frames',
    'save_features',
]

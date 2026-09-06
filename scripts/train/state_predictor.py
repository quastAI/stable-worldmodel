"""Train arm P's predictor on ground-truth state.

Arm P is the "perfect representation" ceiling and the denominator of the
``SR / SR(P)`` axis, so it only means something if the *only* difference from
arm A is where the latent comes from. That is enforced here by sharing the
predictor family, capacity, optimiser, schedule and epoch budget with
``lejepa_predictor.py`` -- if this script ever needs to differ from that one in
any of those, arm P has stopped being a clean control.

The whitening statistics are computed from the training split and stored as
module buffers, so they travel with the checkpoint. Recomputing them at plan
time from whatever data happened to be around would silently change the
coordinates the planner works in.

Usage::

    python scripts/train/state_predictor.py
    python scripts/train/state_predictor.py data.dataset.name=...
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict

import stable_worldmodel as swm
from stable_worldmodel.wm.state import StateWM
from stable_worldmodel.wm.utils import save_pretrained


def state_forward(self, batch, stage, cfg):
    """Next-state prediction in whitened ground-truth coordinates.

    Deliberately the same objective as ``lejepa_predictor.py``'s, so that arm
    P and arm A differ in their representation and in nothing else.
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)

    emb, act_emb = output['emb'], output['act_emb']
    pred = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])

    output['loss'] = (pred - emb[:, n_preds:]).pow(2).mean()
    output['pred_loss'] = output['loss'].detach()

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return output


def whitening_stats(dataset, state_keys, max_samples=20000):
    """Mean and std of the ground-truth state over the training data.

    Also the natural place to catch the plan's SS5.E plumbing risk early: a
    column that is present but constant produces a zero standard deviation
    here, long before it becomes a flat cost surface at plan time.
    """
    rows = min(len(dataset), max_samples)
    stacked = []
    for i in range(rows):
        sample = dataset[i]
        parts = []
        for key in state_keys:
            value = np.asarray(sample[key], dtype=np.float64)
            parts.append(value.reshape(value.shape[0], -1)
                         if value.ndim > 1 else value.reshape(1, -1))
        stacked.append(np.concatenate(parts, axis=-1))

    states = np.concatenate(stacked, axis=0)
    mean = states.mean(axis=0)
    std = states.std(axis=0)

    constant = np.flatnonzero(std <= 1e-8)
    if constant.size:
        raise ValueError(
            f'state dimensions {constant.tolist()} are constant across the '
            'training set. Arm P would plan on a partly-degenerate latent and '
            'report a failure that looks like an identifiability result. '
            f'Check that {state_keys} are all recorded and varying.'
        )
    return mean, std


@hydra.main(
    version_base=None, config_path='./config', config_name='state_predictor'
)
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=os.environ.get('LOCAL_DATASET_DIR'),
        **dataset_cfg,
    )

    state_keys = list(cfg.state_keys)
    mean, std = whitening_stats(dataset, state_keys)
    logging.info(
        f'state dim {len(mean)} over {len(state_keys)} columns; '
        f'std range [{std.min():.4f}, {std.max():.4f}]'
    )

    with open_dict(cfg):
        cfg.model.predictor.input_dim = len(mean)
        cfg.model.predictor.output_dim = len(mean)
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        cfg.model.action_encoder.emb_dim = len(mean)

    model = StateWM(
        predictor=hydra.utils.instantiate(cfg.model.predictor),
        action_encoder=hydra.utils.instantiate(cfg.model.action_encoder),
        state_keys=state_keys,
        mean=mean,
        std=std,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **{**cfg.loader, 'shuffle': False, 'drop_last': False}
    )

    total_steps = cfg.trainer.max_epochs * len(train)
    module = spt.Module(
        model=model,
        forward=partial(state_forward, cfg=cfg),
        optim={
            'predictor_opt': {
                'modules': 'model',
                'optimizer': dict(cfg.optimizer),
                'scheduler': {
                    'type': 'LinearWarmupCosineAnnealingLR',
                    'warmup_steps': max(1, int(0.01 * total_steps)),
                    'max_steps': total_steps,
                },
                'interval': 'epoch',
            },
        },
    )

    trainer = pl.Trainer(
        **cfg.trainer, num_sanity_val_steps=1, enable_checkpointing=True
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        ckpt_path=None,
    )()

    save_pretrained(
        model,
        run_name=cfg.output_model_name,
        config=cfg.model,
        filename='weights_final.pt',
    )
    Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
        cfg.output_model_name,
    ).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    run()

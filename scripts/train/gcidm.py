import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def gcidm_forward(self, batch, stage, cfg):
    """encode observations, predict next states + the goal-conditioned plan.

    ``L = L_fwd + lambda_policy * L_policy + lambda_std * L_policy_std``, with
    the SIGReg anti-collapse term kept from LeWM. Every extra term is skipped
    entirely at weight 0, and at ``lambda_policy = 0`` this is LeWM exactly.

    Frame indexing (the repo's convention is that ``action[k]`` is the block
    *leaving* frame ``k`` -- see ``GCIDM.rollout``). With ``history_size = 3``,
    ``num_preds = 1`` and ``goal_horizon = 5`` the clip holds 8 frames and::

        cur      = ctx_len - 1     = 2    the "current" frame
        goal_idx = cur + H         = 7    +5 blocks = +25 env steps
        label    = action[2:7]            the 5 blocks joining them
        fwd tgt  = frames 1..3            unchanged from SMWM

    Frames 4-6 feed no loss, so only ``[0, 1, 2, 3, 7]`` are handed to the
    encoder: the clip is 2x SMWM's span but the ViT cost is only ~1.25x.
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    horizon = cfg.wm.goal_horizon
    lambd_sigreg = cfg.loss.sigreg.weight
    lambd_policy = cfg.loss.policy.weight
    lambd_std = cfg.loss.policy.std_weight

    cur = ctx_len - 1
    n_fwd = ctx_len + n_preds
    keep = list(range(n_fwd)) + [cur + horizon]

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode({**batch, 'pixels': batch['pixels'][:, keep]})

    emb = output['emb']  # (B, len(keep), D)
    act_emb = output['act_emb']  # (B, T, A_emb) -- full clip

    fwd_emb = emb[:, :n_fwd]
    goal_emb = emb[:, -1]

    ctx_emb = fwd_emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = fwd_emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # forward (LeWM) loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['loss'] = output['pred_loss']

    # Goal-conditioned plan loss. Not detached: shaping the encoder is the
    # whole point, and this is the only anti-collapse term at sigreg weight 0.
    if lambd_policy:
        mu, std = self.model.predict_plan(ctx_emb, goal_emb)
        act_label = batch['action'][:, cur : cur + horizon]
        output['policy_loss'] = (mu - act_label).pow(2).mean()
        output['loss'] = output['loss'] + lambd_policy * output['policy_loss']

        # Calibrate sigma against a DETACHED mean. Under a joint
        # heteroscedastic NLL the weight on the mean's error is 1/sigma^2,
        # which the head controls -- it could shrink sigma on easy samples and
        # inflate it on the hard, task-relevant ones, down-weighting exactly
        # the gradients we want reaching the encoder. Detaching confines this
        # term's gradient to the sigma head's own weights.
        if std is not None and lambd_std:
            resid = act_label - mu.detach()
            output['policy_std_loss'] = (
                0.5 * (resid / std).pow(2) + std.log()
            ).mean()
            output['loss'] = (
                output['loss'] + lambd_std * output['policy_std_loss']
            )

    if lambd_sigreg:
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
        output['loss'] = output['loss'] + lambd_sigreg * output['sigreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='gcidm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        effective_act_dim = cfg.data.dataset.frameskip * dataset.get_dim(
            'action'
        )
        cfg.model.action_encoder.input_dim = effective_act_dim
        # the head regresses the same flattened action blocks the encoder
        # embeds, which is also the space the planner searches over
        cfg.model.policy_head.action_dim = effective_act_dim

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(gcidm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[save_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()

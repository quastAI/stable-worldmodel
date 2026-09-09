"""Train the LeJEPA encoder on OU pairs -- passively, with no action path.

``scripts/train/lewm.py`` with three changes and nothing else:

1. :func:`lejepa_forward` replaces the LeWM forward. No predictor, no action
   encoder, no next-step target -- the objective is
   ``lambda * SIGReg(h) + (1 - lambda) * alignment(h)`` over the two views of
   an OU pair.
2. ``model.head.output_dim`` is set from the dataset's own ``latent/z`` width,
   so ``m = n`` is enforced by the data rather than by a config that could
   drift from it.
3. ``whitening_loss`` and :func:`alignment_diagnostics` are logged every step
   and never optimised. The latter reports ``L``, ``delta`` and ``epsilon`` in
   the paper's units, which the raw losses are not in; it is why
   ``program_constants.rho`` is read here at all.

Optimiser, schedule, checkpointing and the ``SaveCkptCallback`` are inherited
unchanged, so encoder capacity and training budget stay comparable to the LeWM
baseline.

Usage::

    python scripts/train/lejepa.py                                  # physical_content
    python scripts/train/lejepa.py profile=task_content              # + cube.color
    python scripts/train/lejepa.py model.head.output_dim=7           # V7a
    python scripts/train/lejepa.py trainer.max_epochs=20             # V8
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.wm.lejepa.losses import (
    alignment_diagnostics,
    alignment_loss,
    whitening_loss,
)
from stable_worldmodel.wm.lejepa.module import state_dict_hash
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Save a checkpoint after each epoch through ``save_pretrained``.

    Also records the encoder's parameter hash alongside the weights. The
    predictor trained in ``lejepa_predictor.py`` stores that hash and refuses
    to plan against a different encoder -- see
    :class:`~stable_worldmodel.wm.lejepa.module.FrozenEncoderWM`.
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            self._save(pl_module.model, trainer.current_epoch + 1)
        if (trainer.current_epoch + 1) == trainer.max_epochs:
            self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """The passive objective over the two views of an OU pair.

    ``batch['pixels']`` is ``(B, 2, C, H, W)`` -- the two frames the collector
    wrote as a two-step episode. There is no action, no context/target split
    and no next-step prediction: every departure from the LeWM forward is one
    of those three, and each is deliberate.
    """
    lambd = cfg.program_constants['lambda']

    output = self.model.encode(batch)
    emb = output['emb']  # (B, V, n)

    # SIGReg and both LeJEPA losses take (V, B, n).
    h = emb.transpose(0, 1)

    output['sigreg_loss'] = self.sigreg(h)
    output['align_loss'] = alignment_loss(h)
    output['loss'] = (
        lambd * output['sigreg_loss'] + (1.0 - lambd) * output['align_loss']
    )

    # Logged, never optimised: this is the metric `epsilon`, and V8 only means
    # something if it is independent of the objective being minimised.
    output['whitening_metric'] = whitening_loss(h)

    # The bound's own quantities, in the bound's own units -- `align_loss` and
    # `whitening_metric` are a per-element mean and a per-element mean-square,
    # neither of which is comparable to the paper's `L` or `epsilon`. `delta` is
    # the one App. H.8 finds binding, and nothing computed it before.
    diagnostics = alignment_diagnostics(
        h, float(cfg.program_constants['rho'])
    )
    output.update(
        {f'bound/{k}': v for k, v in diagnostics.items()}
    )

    logs = {
        f'{stage}/{k}': v.detach()
        for k, v in output.items()
        if 'loss' in k or k == 'whitening_metric' or k.startswith('bound/')
    }
    self.log_dict(logs, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lejepa')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(f'Loading dataset "{dataset_name}"')

    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transform = spt.data.transforms.Compose(
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    )
    dataset.transform = transform

    # `m = n` is a property of the *data*, not of this config: `n` is whatever
    # the latent registry declared when the dataset was generated. Reading it
    # off `latent/z` means the head can never silently disagree with the
    # ground truth the metrics will score against.
    n = dataset.get_dim('latent/z')
    with open_dict(cfg):
        if cfg.model.head.output_dim in (None, '???'):
            cfg.model.head.output_dim = n
    if cfg.model.head.output_dim != n:
        print(
            f'V7 dimension misspecification active: head outputs '
            f'{cfg.model.head.output_dim}, dataset declares n = {n}.'
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
    # drop_last stays True on val: SIGReg's statistic is multiplied by the
    # batch size, so a short final batch reports a differently-scaled number
    # (at 20k val samples and batch 256 the tail batch is 32, ~1/8 the
    # statistic) and drags the epoch mean. 32 dropped samples is the cheaper
    # trade than an uninterpretable validate/sigreg_loss.
    val_cfg = {**cfg.loader, 'shuffle': False}
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
            # 'step', NOT 'epoch'. `total_steps` is counted in optimizer steps
            # (max_epochs * len(train)), so on 'epoch' the scheduler advances
            # once per epoch and its counter only ever reaches `max_epochs` --
            # 100 against a 703-step warmup for a 200k-pair run. The whole
            # first epoch then runs at lr exactly 0, the peak reaches 14% of
            # the configured lr, and the cosine never anneals at all, so the
            # frozen checkpoint is taken at the run's highest lr. Verified:
            # 200 steps on 'epoch' moved align 0.0704 -> 0.0697 and sigreg
            # 70.7 -> 70.1, i.e. nothing; on 'step' the same 200 steps gave
            # sigreg 70.7 -> 1.7 and std(h) 0.39 -> 0.91.
            'interval': 'step',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as handle:
        OmegaConf.save(cfg, handle)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(
                run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1
            )
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer, module=module, data=data_module, ckpt_path=None
    )
    manager()

    # The predictor stage binds itself to this exact encoder, so the hash has
    # to be discoverable from the encoder run rather than recomputed by hand.
    encoder_hash = state_dict_hash(world_model)
    with open(run_dir / 'encoder_hash.txt', 'w') as handle:
        handle.write(encoder_hash)
    print(f'encoder hash: {encoder_hash}')


if __name__ == '__main__':
    run()

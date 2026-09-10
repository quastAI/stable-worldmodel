"""Train the LeJEPA encoder on OU pairs -- passively, with no action path.

``scripts/train/lewm.py`` with three changes and nothing else:

1. :func:`lejepa_forward` replaces the LeWM forward. No predictor, no action
   encoder, no next-step target -- the objective is
   ``lambda * SIGReg(h) + (1 - lambda) * alignment(h)`` over the two views of
   an OU pair.
2. ``model.head.output_dim`` is set from the dataset's own ``latent/z`` width,
   so ``m = n`` is enforced by the data rather than by a config that could
   drift from it.
3. ``whitening_loss``, :func:`alignment_diagnostics`,
   :func:`spectrum_diagnostics` and :func:`recovery_diagnostics` are logged
   every step and **never optimised**. ``alignment_diagnostics`` reports ``L``,
   ``delta``, ``epsilon`` and the bound they combine into, in the paper's
   units, which the raw losses are not in -- it is why
   ``program_constants.rho`` is read here at all. ``recovery_diagnostics``
   scores ``h`` against the recorded ``latent/z``, which the loader has always
   loaded and the forward never read; it is a label, so it stays out of the
   objective.

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
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.wm.lejepa.losses import (
    alignment_diagnostics,
    alignment_loss,
    recovery_diagnostics,
    spectrum_diagnostics,
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


class RecordCkptDirCallback(Callback):
    """Record where Lightning's checkpoints actually landed.

    ``spt.Manager`` runs in cache_dir mode unconditionally -- a ``cache_dir`` is
    mandatory, ``None`` raises -- and ``_configure_cache_dir_checkpointing``
    rewrites the ``dirpath`` of **every** ``ModelCheckpoint`` to
    ``$SPT_CACHE_DIR/runs/<date>/<time>/<run_id>/checkpoints/``. So the path
    cannot be set; it can only be discovered, and the ``<date>/<time>/<run_id>``
    part is not knowable before the run starts.

    This resolves it at ``on_train_start`` -- once Manager has done its rewrite
    but before the first checkpoint is written -- and drops a
    ``lightning_ckpt_dir.txt`` plus a ``lightning`` symlink next to the run's
    ``config.yaml``. Doing it at train start rather than after ``fit`` returns is
    the point: a run that dies at epoch 6 is exactly the run whose checkpoint
    directory you need to find, and that is the run whose ``fit`` never returns.
    """

    def __init__(self, run_dir):
        super().__init__()
        self.run_dir = Path(run_dir)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        directories = sorted(
            {
                str(cb.dirpath)
                for cb in trainer.callbacks
                if isinstance(cb, ModelCheckpoint) and cb.dirpath
            }
        )
        if not directories:
            print('no ModelCheckpoint is configured; no trainer state is saved.')
            return
        if len(directories) > 1:
            print(f'WARNING: checkpoints are split across {directories}')

        target = Path(directories[0])
        (self.run_dir / 'lightning_ckpt_dir.txt').write_text(f'{target}\n')

        link = self.run_dir / 'lightning'
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            # Not fatal: the .txt above is the authoritative record, and some
            # network volumes refuse symlinks.
            print(f'could not link {link} -> {target}: {error}')

        print(f'lightning checkpoints -> {target}')
        print(f'  recorded at {self.run_dir / "lightning_ckpt_dir.txt"}')


class VerifyScheduleCallback(Callback):
    """Raise at train start if the realised LR schedule is not the configured one.

    The schedule is the one thing in this script that has already failed
    silently: ``interval: 'epoch'`` against a step-counted ``max_steps`` held
    the whole first epoch at lr exactly 0, capped the peak at 14% of the
    configured value, and nothing logged it.

    ``schedule.shape: constant`` opens a second door to the same failure. It is
    expressed as ``eta_min == base_lr`` -- an ordinary constructor kwarg -- so a
    wrapper that quietly drops unknown scheduler kwargs would hand back a real
    cosine annealing to zero, and the run would look entirely healthy for 100
    epochs before its trend turned out to be an artefact of the LR. Nothing in
    the repo currently passes an extra scheduler kwarg, so the pass-through is
    unproven; this reads the constructed scheduler back and asserts instead.

    Args:
        expected: Scheduler attributes to check, by name.
    """

    def __init__(self, expected):
        super().__init__()
        self.expected = dict(expected)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if not trainer.is_global_zero:
            return

        configs = trainer.lr_scheduler_configs
        if not configs:
            raise RuntimeError(
                'no LR scheduler was configured; the schedule block in '
                'lejepa.yaml is not reaching the optimizer'
            )

        for config in configs:
            if config.interval != 'step':
                raise RuntimeError(
                    f'scheduler interval is {config.interval!r}, not '
                    "'step' -- max_steps is counted in optimizer steps, so on "
                    "'epoch' the counter only ever reaches max_epochs"
                )
            scheduler = config.scheduler
            for key, want in self.expected.items():
                got = getattr(scheduler, key, None)
                if got is None:
                    raise RuntimeError(
                        f'scheduler has no attribute {key!r} -- the kwarg did '
                        'not reach LinearWarmupCosineAnnealingLR'
                    )
                if abs(float(got) - float(want)) > 1e-12:
                    raise RuntimeError(
                        f'scheduler.{key} is {got!r}, expected {want!r} -- the '
                        'kwarg was dropped or overridden in transit'
                    )

        realised = {k: getattr(configs[0].scheduler, k) for k in self.expected}
        print(f'LR schedule verified (interval=step): {realised}')


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

    # Partial collapse is invisible in `epsilon`, which is an aggregate: a
    # single dead direction out of n moves it less than ordinary early-training
    # noise does. The smallest eigenvalue of Cov(h) and the participation ratio
    # do not average it away.
    output.update(
        {f'spectrum/{k}': v for k, v in spectrum_diagnostics(h).items()}
    )

    # The criterion metric itself, live. `latent/z` is already in every batch
    # -- the data config loads it and the forward has simply never read it --
    # so this costs a handful of n x n decompositions and answers "is it
    # recovering the latents", which no loss curve does. Never optimised: z is
    # a label, and a label in the objective would make this supervised
    # regression rather than an identifiability claim.
    latents = batch.get('latent/z')
    if latents is not None:
        output.update(
            {
                f'recovery/{k}': v
                for k, v in recovery_diagnostics(
                    h, latents.transpose(0, 1).to(h.dtype)
                ).items()
            }
        )

    # Which term is actually being minimised. `align_loss` flattening while
    # `sigreg_loss` still falls is the signature of lambda being too high, and
    # reading that off two separately-scaled curves is guesswork; the share is
    # one number in [0, 1].
    weighted_sigreg = lambd * output['sigreg_loss'].detach()
    output['balance/sigreg_share'] = weighted_sigreg / (
        weighted_sigreg + (1.0 - lambd) * output['align_loss'].detach()
    ).clamp_min(1e-12)

    logs = {
        f'{stage}/{k}': v.detach()
        for k, v in output.items()
        if 'loss' in k
        or k == 'whitening_metric'
        or k.startswith(('bound/', 'spectrum/', 'recovery/', 'balance/'))
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

    # Warmup is an ABSOLUTE step count, not `int(0.01 * total_steps)`. As a
    # fraction it rode on `trainer.max_epochs`, which is the V8 severity knob:
    # at 703 steps/epoch the 10-epoch s3072 run got 70 warmup steps where a
    # 100-epoch run gets 703, so V8 moved budget, ramp and anneal together.
    warmup_steps = max(1, min(int(cfg.schedule.warmup_steps), total_steps - 1))

    base_lr = float(cfg.optimizer.lr)
    if cfg.schedule.shape == 'constant':
        # LinearWarmupCosineAnnealingLR interpolates
        #   eta_min + (base_lr - eta_min) * (1 + cos(pi * progress)) / 2,
        # so eta_min == base_lr zeroes the cosine term and the lr is flat after
        # warmup. Cheaper than a second scheduler class, and verified at train
        # start rather than assumed -- see VerifyScheduleCallback.
        eta_min = base_lr
    elif cfg.schedule.shape == 'cosine':
        eta_min = 0.0
    else:
        raise ValueError(
            f'schedule.shape must be "constant" or "cosine", '
            f'got {cfg.schedule.shape!r}'
        )

    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': warmup_steps,
                'max_steps': total_steps,
                'eta_min': eta_min,
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

    callbacks = [
        SaveCkptCallback(
            run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1
        )
    ]
    if cfg.checkpoint.keep_every_epoch:
        # `filename` must not be 'last': Manager appends its own requeue saver
        # under that exact name, and two callbacks writing one filename in one
        # directory race. `save_top_k=-1` is what makes each epoch *stay*;
        # without it Lightning keeps one file and overwrites it.
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(run_dir / 'lightning'),  # redirected by Manager
                filename='epoch{epoch:03d}',
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=-1,
                save_last=False,
                save_on_train_epoch_end=True,
                enable_version_counter=False,
            )
        )
    callbacks.append(RecordCkptDirCallback(run_dir))
    callbacks.append(
        VerifyScheduleCallback(
            {
                'warmup_steps': warmup_steps,
                'max_steps': total_steps,
                'eta_min': eta_min,
            }
        )
    )

    if logger is not None:
        # The schedule is the one thing here that has already failed silently:
        # `interval: 'epoch'` against a step-counted `max_steps` held the whole
        # first epoch at lr exactly 0 and capped the peak at 14% of the
        # configured value, and nothing logged it. LearningRateMonitor requires
        # a logger, hence the guard.
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
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

"""Train the stage-D predictor on top of a frozen LeJEPA encoder (arms A and R).

Arms A and R differ **only in which checkpoint is loaded** -- arm A a trained
LeJEPA, arm R a randomly initialised one. Nothing else about this script
changes between them, which is what makes the random-encoder control a control
rather than a different experiment.

Only the predictor is optimised. That is enforced structurally rather than by
discipline: :class:`~stable_worldmodel.wm.lejepa.module.FrozenEncoderWM`
freezes the encoder on construction, keeps it in ``eval()`` through
``.train()``, and runs ``encode`` under ``no_grad``. The optimiser is
additionally scoped to the predictor and action encoder, so an encoder
parameter could not receive a gradient even if one reached it.

The encoder's parameter hash is recorded with the checkpoint. At plan time
``FrozenEncoderWM`` refuses an encoder that does not match it -- a
predictor/encoder mismatch degrades planning in a way indistinguishable from an
identifiability failure.

Usage::

    python scripts/train/lejepa_predictor.py encoder=lejepa/weights_epoch_100.pt
    python scripts/train/lejepa_predictor.py encoder=random    # arm R
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.identifiability.collect import load_manifest
from stable_worldmodel.wm.lejepa.module import FrozenEncoderWM, state_dict_hash
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained


def get_img_preprocessor(source, target, img_size=224):
    return dt.transforms.Compose(
        dt.transforms.ToImage(
            **dt.dataset_stats.ImageNet, source=source, target=target
        ),
        dt.transforms.Resize(img_size, source=source, target=target),
    )


def predictor_forward(self, batch, stage, cfg):
    """Next-embedding prediction in the frozen latent space.

    The LeWM forward with the SIGReg term removed: isotropy is the *encoder's*
    business and was settled before this stage began. Leaving it in would let
    the predictor stage reshape the embedding distribution the metrics were
    already computed on.
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)

    emb = output['emb']
    act_emb = output['act_emb']

    pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    tgt_emb = emb[:, n_preds:]

    output['loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['pred_loss'] = output['loss'].detach()

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(
    version_base=None, config_path='./config', config_name='lejepa_predictor'
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
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor('pixels', 'pixels', cfg.img_size)
    )

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    # ------------------------------------------------------------------
    # the frozen encoder
    # ------------------------------------------------------------------
    if cfg.encoder == 'random':
        # Arm R. A randomly initialised encoder of *identical architecture*,
        # so the arm isolates what training bought rather than what the
        # architecture bought.
        #
        # Its width cannot come from a checkpoint, so it comes from the encoder
        # dataset's manifest. Arm A reads `n` off the loaded `config.json`; if
        # arm R kept a hardcoded literal the two arms could differ in `m` and
        # the comparison would quietly stop being about training.
        encoder_dataset = cfg.get('encoder_dataset')
        if encoder_dataset:
            n = int(load_manifest(encoder_dataset)['latents']['n'])
            declared = cfg.random_encoder.head.output_dim
            if declared != n:
                print(
                    f'arm R: taking m = {n} from {encoder_dataset} '
                    f'(config said {declared})'
                )
            with open_dict(cfg):
                cfg.random_encoder.head.output_dim = n
        encoder = hydra.utils.instantiate(cfg.random_encoder)
    else:
        encoder = load_pretrained(cfg.encoder)

    encoder_hash = state_dict_hash(encoder)
    print(f'frozen encoder: {cfg.encoder} (hash {encoder_hash})')

    with open_dict(cfg):
        cfg.model.predictor.input_dim = encoder.output_dim
        cfg.model.predictor.hidden_dim = cfg.predictor_hidden_dim
        cfg.model.predictor.output_dim = encoder.output_dim
        cfg.model.action_encoder.emb_dim = encoder.output_dim
        cfg.model.encoder_hash = encoder_hash

    model = FrozenEncoderWM(
        encoder=encoder,
        predictor=hydra.utils.instantiate(cfg.model.predictor),
        action_encoder=hydra.utils.instantiate(cfg.model.action_encoder),
        encoder_hash=encoder_hash,
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
    optimizers = {
        'predictor_opt': {
            # Scoped to the predictor and action encoder. Belt and braces
            # against the encoder freeze, and it documents the intent where an
            # optimiser config is the natural place to look for it.
            'modules': ['model.predictor', 'model.action_encoder'],
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

    module = spt.Module(
        model=model,
        forward=partial(predictor_forward, cfg=cfg),
        optim=optimizers,
    )

    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
        cfg.get('subdir') or '',
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)

    from scripts.train.lejepa import SaveCkptCallback  # noqa: PLC0415

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(cfg.output_model_name, cfg.model, epoch_interval=1)
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
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
    with open(run_dir / 'encoder_hash.txt', 'w') as handle:
        handle.write(encoder_hash)


if __name__ == '__main__':
    run()

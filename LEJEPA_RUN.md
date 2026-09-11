# LeJEPA Identifiability on OGBench Cube-single

Run guide for the **LeJEPA encoder study** — the designed-OU dataset, the
passive encoder trained on it, and the metric suite that scores how much of the
true latent state the encoder recovered. This is a different pipeline from the
one in [`Run.md`](Run.md), which covers the LeWM / SMWM world-model ablations on
cube-**quadruple** with domain randomization. The two share the install, the env
vars and the on-disk layout; they share no datasets, configs or scripts.

Read `Run.md` §2–§3 first for prerequisites and install — everything there
applies unchanged. This document starts after `pip install -e '.[train,format]'`
succeeds.

> **Scope.** This pipeline trains **one encoder** and measures it. The predictor
> stage, the A/R/C arms, the planning success rates and the V1–V9 violation
> ladder were all removed: they answered questions about *comparisons between
> arms*, and the open question is now about a single encoder's recovery. Every
> knob they needed is gone from the configs rather than defaulted off, so there
> is nothing to accidentally half-enable. Re-adding one is a deliberate act —
> the sampler grows a field, `PROFILES` grows an entry.

---

## 1. What this pipeline is

One training stage and a metric suite:

| Stage | Script | What it learns |
|---|---|---|
| **encoder** | `scripts/train/lejepa.py` | A passive, action-free encoder on designed OU **positive pairs**. Objective is `λ·SIGReg + (1-λ)·alignment`. No predictor, no EMA target, no actions. |
| **metrics** | `scripts/identifiability/run_metrics.py` | Per-latent recovery of the true `z`, the canonical spectrum, and the theory's bound with an admissibility check on it. |

**The latent profile.** One shipped profile, `physical_content` (`n = 10` at one
cube): the 9 physical DOFs plus `cube.size`. `profile=` still selects it end to
end — collection, training and the metrics all interpolate it into the dataset
name, so a disagreement looks for a dataset nobody wrote rather than silently
scoring the wrong `z`.

> **`cube.size` is content and `camera.angle_delta` is excluded.** One rule:
> *a nuisance that shares a rendering channel with a content latent must not be
> style*, because style demands exact invariance along a direction the content
> itself depends on. Apparent cube size is the shared channel — `cube.size`,
> `cube.pos_z` and camera distance all move it.

## 2. Environment

Same variables as `Run.md`, plus one:

```bash
export STABLEWM_HOME=/path/with/space   # datasets/ and checkpoints/ both live here
export MUJOCO_GL=egl                    # Linux GPU node; see below
```

| Variable | Effect |
|---|---|
| `STABLEWM_HOME` | Root for `datasets/` **and** `checkpoints/`. Default `~/.stable_worldmodel` ([`data/utils.py:34`](stable_worldmodel/data/utils.py#L34)). |
| `MUJOCO_GL` | **Optional for the collectors** — all three auto-select (`glfw` on macOS or when `DISPLAY` is set, else `osmesa` + `PYOPENGL_PLATFORM`) and never override an existing export. Set `egl` by hand on a Linux GPU node: rendering is ~90% of collection runtime ([`collect_cube_single_ou.py:38-42`](scripts/data/collect_cube_single_ou.py#L38-L42)). |
| `LOCAL_DATASET_DIR` | Optional. If set, datasets resolve under `$LOCAL_DATASET_DIR/datasets/` instead of `$STABLEWM_HOME/datasets/` ([`lejepa.py:130`](scripts/train/lejepa.py#L130)). Does **not** affect checkpoints. |

> The three collectors run in Hydra **single-run** mode and drop
> `./outputs/<date>/<time>/.hydra/` in the cwd — none of these configs set a
> `hydra:` block. Append `hydra.run.dir=. hydra.output_subdir=null` to keep the
> repo clean. (`Run.md`'s `hydra.sweep.dir=/tmp/hydra` advice does not apply;
> that is for MULTIRUN configs.)

---

## 3. Order of operations

```
collect_cube_single_ou.py      ->  datasets/ogbench/cube_single_ou_<profile>.lance
   (twice: the pairs, and the                + cube_single_ou_<profile>_manifest.json
    style probe at rho ~ 1)
        |
        v
train/lejepa.py                ->  checkpoints/lejepa/weights_epoch_N.pt
                                   checkpoints/lejepa_s<seed>/encoder_hash.txt
        |
        v
run_metrics.py                 ->  outputs/lejepa_results.jsonl
   (the LAST few epochs only —
    see §5 on why not per-epoch)
```

Three things worth knowing about this shape:

- **Collect the style probe too.** It is the same collector with `rho` pushed
  to ~1, so the two views of each pair share their content and differ only in
  style. `run_metrics.py` arms it by default, and without it `delta` cannot be
  split into nonlinearity and style leakage — the reported bound then charges
  all of it to nonlinearity (§6).
- **Scoring is not per-epoch.** Each `run_metrics.py` invocation embeds 20k
  pairs twice over, plus the style probe, and trains the non-linear probe.
  Score the last two or three checkpoints, not every one.
- **Nothing downstream consumes the encoder.** There is no predictor and no
  planner in this pipeline, so `weights_epoch_N.pt` exists to be *measured*.

---

## 4. Stage A dataset — OU positive pairs

```bash
cd stable-worldmodel

python scripts/data/collect_cube_single_ou.py num_pairs=200000 seed=3072
```

Writes `$STABLEWM_HOME/datasets/ogbench/cube_single_ou_<profile>.lance` plus a sidecar
`cube_single_ou_<profile>_manifest.json`
([`collect.py:379-386`](stable_worldmodel/identifiability/collect.py#L379-L386)).

One positive pair per **two-step episode**: `pixels (2,H,W,3)`, `latent/z (2,n)`,
`latent/style`, `latent/view`, `ou/rho_per_dim`, plus every `privileged/*`,
`proprio/*`, `qpos` and `qvel` column the env emits — the recovery target is
rebuilt from those recorded rows without re-running the environment
([`collect.py:244-266`](stable_worldmodel/identifiability/collect.py#L244-L266)).

### Knobs

| Key | Default | Notes |
|---|---|---|
| `num_pairs` | 200000 | Pairs, i.e. 2× that many rendered frames. Split across shards, not per shard. |
| `seed` | 3072 | Sampler seed is `seed + shard*1_000_003`; the style stream uses `base_seed + 7`. |
| `shard` / `num_shards` | 0 / 1 | Disjoint slice + disjoint seed block. Validated `0 <= shard < num_shards`. |
| `dataset_name` | `ogbench/cube_single_ou_${latents.profile}.lance` | Interpolates the profile, so two profiles cannot overwrite each other. Sharded runs add `_shard{i}`. |
| `write_mode` | `overwrite` | Passed to `LanceWriter`. |
| `program_constants.rho` | 0.9 | The OU autocorrelation — the positive-pair strength. |
| `program_constants.lambda` | 3.0e-3 | Recorded into the manifest only; the **encoder** reads its own copy. |
| `latents.profile` | `physical_content` | The only shipped profile, `n = 10` at one cube — see §1. |
| `latents.yaw_half_arc` | 0.746… | `0.95 · π/4`, one fundamental domain of the cube's C4 yaw symmetry. Widening past `π/4` requires `env.marker.enabled=true`. |
| `latents.include_roll_pitch` | false | Switches the sampler to its tangent-space branch. |
| `latents.include_discrete` | false | Discrete latents are renderable but non-Gaussian — a structural V1. |
| `latents.pos_z_max` | 0.15 | Lifted / interpenetrating cubes allowed on purpose, so the induced V5 baseline is zero. |
| `ou.dist` | `gaussian` | `gaussian`, `laplace` or `gennorm`; `gennorm` needs `ou.alpha`. |
| `ou.noise_coupling` | 0.0 | Heteroscedastic noise (the V3 knob). |
| `ou.cross_corr` | 0.0 | Cross-dimension noise correlation (the V9 knob). |
| `env.image_size` | 224 | Sets both width and height. |
| `env.camera` | `front_pixels` | |
| `env.num_digits` | 0 | Floor-digit distractors; also the V6 knob. |

> `style.resample_within_pair` in the config is a **dead knob** — style is
> unconditionally resampled per view inside the pair loop
> ([`collect.py:231-241`](stable_worldmodel/identifiability/collect.py#L231-L241)).
> Do not rely on setting it to `false`.

### Cost and sharding

The script never steps physics — every frame is an independent OU draw pushed
through `render_content`, which is why this is affordable at all. Documented
figures ([`collect_cube_single_ou.py:13-18`](scripts/data/collect_cube_single_ou.py#L13-L18),
[`collect.py:9-26`](stable_worldmodel/identifiability/collect.py#L9-L26)):

| Path | Per frame | 200k pairs (400k frames) |
|---|---|---|
| `render_content`, all-content | 6.1 ms | **≈0.67 core-hours** |
| `reset()`, no recompile | 21.4 ms | — |
| `reset()`, all-content | 263.1 ms | ≈29.2 core-hours |

Single-process and single-threaded; there is **no sharded driver script** for
this collector (`collect_cube_quadruple_dr_sharded.py` is hard-wired to the
quadruple-DR script). Launch shards by hand and merge:

```bash
for i in 0 1 2 3 4 5 6 7; do
  python scripts/data/collect_cube_single_ou.py \
      num_pairs=200000 shard=$i num_shards=8 &
done; wait

swm merge $(for i in 0 1 2 3 4 5 6 7; do printf 'ogbench/cube_single_ou_<profile>_shard%d.lance ' $i; done) \
    --output ogbench/cube_single_ou_<profile> --overwrite
```

> **`swm merge` does not merge manifests.** Each shard writes its own
> `cube_single_ou_<profile>_shard{i}_manifest.json`, while everything
> downstream derives the sidecar name from the *dataset* name and so looks for
> `cube_single_ou_<profile>_manifest.json` (§6, §9). After merging, copy one
> shard's manifest to the merged name — the shards share every field that
> matters except the shard index.

> `swm merge` appends the format suffix itself — pass `--output name`, not
> `--output name.lance`. It needs at least two sources.

Writes go out in chunks of 2000 episodes (~600 MiB at 224×224). The chunked
writer is not an optimization: a MuJoCo GL context cannot be touched from
Lance's background thread, and doing so deadlocks at zero CPU
([`collect.py:294-314`](stable_worldmodel/identifiability/collect.py#L294-L314)).

---

## 5. Training — the encoder

```bash
python scripts/train/lejepa.py
```

`data=ogb_cube_single_ou` is the default data group; `profile=` picks which
profile's dataset inside it (§1). Useful overrides:

```bash
python scripts/train/lejepa.py \
    subdir=lejepa_s3072 trainer.max_epochs=25 \
    loader.batch_size=256 program_constants.lambda=5.0e-2

# ask for a narrower head than the data declares, on purpose
python scripts/train/lejepa.py model.head.output_dim=5
```

### Knobs

| Key | Default | Notes |
|---|---|---|
| `output_model_name` | `lejepa` | Checkpoint subfolder. |
| `subdir` | `${hydra:job.id}` | Where `config.yaml` / `encoder_hash.txt` go — see the warning below. |
| `program_constants.lambda` | 5.0e-2 | `loss = λ·sigreg + (1-λ)·align`, **in the paper's units** — `loss.sigreg.kwargs.pool_views: true` is what makes that so. Chosen from the paper's λ×ρ grid (Fig. 6 / App. H.6) read at **our** ρ. At ρ=0.9 the four candidates are indistinguishable on R²(h→z) — 0.981 / 0.989 / 0.993 / 0.992 for 1e-3 / 5e-3 / 1e-2 / 5e-2 — and separated only by orthogonality error, where 5e-2 is the best in the column: 0.461 / 0.210 / 0.142 / **0.062**. Orthogonality is what our recovery metrics score, so it is the row that decides. App. H.6's *prose* names {1e-3, 5e-3} instead, contradicting its own table at ρ=0.9; the table is what is cited here. 5e-2 degrades only from 1e-1 (0.847), and falls to 0.493 at ρ=0.99 — the collapse `rho: 0.9` stays clear of. Confirm with a 2-point shakedown before spending the seed budget. |
| `loss.sigreg.kwargs.pool_views` | `true` | Pools both views into one set of 2B samples and scales by 2B, as the paper's own implementation does, instead of a per-view statistic averaged over views. Away from the optimum the two differ by exactly 2× (measured 2.00×), so this is what makes λ comparable to the paper's figures instead of leaving a factor of 2 to be remembered. **LeWM keeps the per-view default** — its baselines were trained under it. |
| `program_constants.rho` | 0.9 | **Not read here** — it is the collector's knob, carried so the metric writer can copy it verbatim. |
| `trainer.max_epochs` | 100 | Pure budget knob now that `schedule.warmup_steps` is absolute. 100 × 703 = 70k steps, ≈1.8× the paper's own Reacher budget. |
| `schedule.constant_frac` | 0.5 | Fraction of the run held at **peak** lr before the cosine anneal to zero begins — the only shape knob. `0.5` is the paper's recipe (App. H.4: "constant learning rate for the first half of training, followed by cosine decay to zero"); `0.0` is a pure cosine; `1.0` is flat forever. The hold buys representation learning, the anneal makes the final number mean something. A constant lr does not converge, it orbits in a noise ball whose radius scales with the lr, and `align_loss` is a difference of the two views' embeddings — exactly what a jittering encoder inflates: the 25-epoch constant run plateaued at **2.29×** its alignment floor where the paper's annealed runs sit at **0.976×**, so that plateau was an artefact of the lr, not a ceiling. A pure cosine has the mirror problem — half the lr is gone by the halfway point, so a large budget buys mostly anneal. Price of any anneal: checkpoints inside it are not comparable to each other (each sits at its own lr), so read the trend for its shape, not the epoch it turns over — and **do not early-stop**, since at 0.5 the whole first half is at peak and killing in it forfeits the entire anneal. |
| `schedule.warmup_steps` | 700 | **Absolute**, not 1% of total steps — as a fraction it rode on `max_epochs`, so a 10-epoch run got 70 warmup steps where a 100-epoch run got 703 and no two budgets were comparable. |
| `trainer.precision` / `.accelerator` | `32` / `gpu` | **Not bf16**: SIGReg's inner term is a difference of two O(1) quantities whose true value is O(1/√B), then multiplied by B. For a CPU smoke run: `trainer.accelerator=cpu trainer.precision=32 trainer.devices=1 loader.num_workers=0 loader.persistent_workers=false loader.prefetch_factor=null`. |
| `loader.batch_size` | 256 | SIGReg is batch-size-scaled, so its absolute magnitude is not comparable across batch sizes. |
| `optimizer.lr` / `.weight_decay` | `${encoder_lr}` / `${encoder_weight_decay}` | From the encoder group — 3e-3/1e-4 for the CNN, 5e-5/1e-3 for the ViT. Schedule is the `schedule` block below (`wm/lejepa/schedule.py::WarmupHoldCosineLR`), stepped **per optimizer step**. It used to be `interval: 'epoch'` against a step-counted `max_steps`, which left the whole first epoch at lr exactly 0 and capped the peak at 14% of the configured lr with no annealing at all — see §8 defect 6. |
| `encoder` | `paper_cnn` | Config group; also sets `embed_dim`, the head shape and the optimizer (§1). |
| `img_size` | 224 | Shared by both encoders. `patch_size`, `encoder_scale` and `embed_dim` come from the encoder group. |
| `model.head.output_dim` | `null` | `null` ⇒ take `n` from the dataset's `latent/z` width, so `m = n` by construction. See below. |
| `model.head.hidden_dim` / `.norm` | from the encoder group | `null`/`false` for the CNN (its projection already ends in BatchNorm1d + GELU, so the head is a bare `Linear(256, n)` exactly as the reference); `2048`/`true` for the ViT. **Never a norm after the output** — that would force `diag(Cov(h))` to 1 and partly trivialise both ε and SIGReg. |
| `loss.sigreg.kwargs.knots` / `.num_proj` | 17 / 1024 | Epps–Pulley knots and random projections. |
| `train_split` | 0.9 | |
| `seed` | 3072 | **Only** seeds the train/val split and the DataLoader generators — not model init, not SIGReg's projections. Encoder hashes are therefore *not* reproducible from `seed` alone. |

**`model.head.output_dim: null` keeps `m = n` a property of the data.** An
explicit mismatching value is *allowed* and only prints `narrow/wide head
active: …`. A typo will not stop the run — check stdout for that line. Under
`m != n` there is no square `Q` to align with, so `recovery_diagnostics` returns
nothing and the per-latent probe in §6 is the metric that still applies.

### What to watch

Logged keys are `fit/…` and `validate/…` (not `train/…`). **Which prefix you
read matters, and not equally for all of them.**

The encoder carries BatchNorm after every conv stage plus a `BatchNorm1d`
before the head, and `LeJEPA.encode` folds the view axis into the batch — so in
**train mode** `h` is a function of the whole batch rather than of one frame.
Anything that reads only the marginal second moment of `h` agrees across modes;
anything that reads `h[0] − h[1]` does not. Measured on a 25-epoch run at epoch
25, same weights:

| key | `fit` | `validate` |
|---|---|---|
| `bound/trace_cov` | 9.5–9.9 | 9.87 |
| `spectrum/effective_rank` | 9.7–9.85 | 9.6 |
| `align_loss` | 0.05 | **0.11** |
| `bound/delta` | 0.02–0.3 | **2.69** |

The covariance is the same function in both modes. `delta` differs by an order
of magnitude, and the train-mode value sat *below its own theoretical floor*
(§6) — i.e. it was impossible. So the whole `bound/` group is logged on the
**validation stage only**, and `fit/align_loss` should be read against its floor
rather than against zero.

| Key | Stage | Healthy | Meaning |
|---|---|---|---|
| `fit/sigreg_loss` | both | **falls hard and fast** | Epps–Pulley isotropy statistic over 1024 projections. The only thing preventing collapse — there is no EMA target network. Should drop by ~two orders inside a few hundred steps. |
| `fit/align_loss` | both | falls **to its floor, not to zero** | `(h.mean(0) − h)²`. The floor is `(1 − ρ)·tr Cov(h) / (2n)` — 0.048 at ρ=0.9, n=10, `tr Cov(h)`=9.6. Read the `validate/` value; the `fit/` one is flattered. |
| `fit/balance/sigreg_share` | both | falls well below 1 | The share of the objective SIGReg accounts for. Pinned near 1 *while* `align_loss` is flat is λ too high — drop to 1e-2. Early dominance is expected. |
| `fit/whitening_metric` | both | drifts down | `‖Cov−I‖²_F/n²`. **Logged, never optimized** — computed under `no_grad` and excluded from the loss, which is the only thing that keeps ε an honest measurement of the bound's own term. A rise is a legitimate observation, not a bug. |
| `spectrum/trace_cov` | both | rises toward `n` | Heading to 0 is total collapse. The fastest collapse detector there is. |
| `spectrum/cov_eig_min` | both | rises toward 1 | Smallest eigenvalue of `Cov(h)`. Zero is a direction carrying nothing. |
| `spectrum/effective_rank` | both | rises toward `n` | `(Σλ)²/Σλ²`. Partial collapse is invisible in `whitening_metric`, which is an aggregate: one dead direction out of 10 contributes 1 to ε² and hides inside ordinary values. |
| `recovery/procrustes_mse_per_dim` | both | falls | The criterion metric — but **not scale-free**: `h = 0` scores 1.0 (§6). Treat it as a trend, and get the real number from `run_metrics.py`. |
| `recovery/r2_z_to_h`, `r2_h_to_z` | both | rise | Once `Cov(h) ≈ I` these two are *forced* to agree, so their agreement is not corroboration. |
| `recovery/cond` | both | falls | Condition number of the best linear map. Rising while `orth_err_normalized` falls means the recovered block is tidying up while some directions stay null. |
| `validate/bound/delta`, `bound/D`, `bound/predicted_error` | **validate only** | fall | The theory's own quantities. Absent from `fit/` on purpose. |
| `validate/bound/delta_floor` | **validate only** | — | The floor `delta` cannot go below. Compare the two *epoch means*: the per-batch flag is not logged because at 256 pairs it false-alarms on honest batches. |
| `lr-AdamW` | — | rises through warmup, then flat | From `LearningRateMonitor`, which needs a logger — so it is absent when W&B is off. |

**Kill rules.** There is no such thing as early-stopping a converging run here.
The anneal is where `align_loss` actually approaches its floor, there is no
resume (`ckpt_path=None`), and at `constant_frac: 0.5` the entire first half sits
at peak lr — so a run killed anywhere in it is not a shorter run, it is a
checkpoint frozen at peak lr with no anneal at all. The only saving is killing a
*broken* run in epoch 1 instead of at the end, which is what these rules are
for.

1. `lr-AdamW` exactly 0 for the whole first epoch → the schedule regressed to
   `interval: 'epoch'`. This has happened in this repo before (§8 defect 6) and
   cost a run that looked like it trained. Kill immediately.
2. `sigreg_loss` flat, or `trace_cov` → 0 → nothing is being regularised, or
   the representation is collapsing outright. Kill.
3. `effective_rank` below its step-0 value and still falling at the end of
   epoch 1 → SIGReg is losing. Kill; re-run at a higher λ.
4. `recovery/r2_z_to_h` still under ~0.1 at the end of epoch 2 with everything
   else moving → the losses are optimising and the latents are not being
   recovered. That is the one failure no loss curve shows.
5. `balance/sigreg_share` pinned at ~1 while `align_loss` is flat → λ too high.

`num_sanity_val_steps=1`, so one validation batch runs first — a shape or
`latent/z` problem surfaces before any training step.

### Outputs

```
$STABLEWM_HOME/checkpoints/<output_model_name>/
    weights_epoch_1.pt ... weights_epoch_100.pt   # one per epoch, never pruned
    config.json                                   # cfg.model only, overwritten each save
$STABLEWM_HOME/checkpoints/<subdir>/
    config.yaml                                   # the full resolved config
    encoder_hash.txt                              # 16-char digest, see §7
```

> **`subdir` now defaults to `${output_model_name}_s${seed}`**, not
> `${hydra:job.id}` — the latter resolves to an empty string on a plain single
> run, so `config.yaml` and `encoder_hash.txt` landed in the `checkpoints/`
> root and consecutive runs silently overwrote them. Still overridable
> (`subdir=armA_seed3072`).

> The encoder hash is printed as `encoder hash: <hash>` at the end of the run.
> **Record it with the run.** `seed` does not make it reproducible (see the knob
> table), so if you lose the `.pt` you cannot regenerate the pair.

---

## 6. Metrics

```bash
python scripts/identifiability/run_metrics.py \
    checkpoint=lejepa/weights_epoch_25.pt \
    program_constants.lambda=5.0e-2
```

`checkpoint` is the only mandatory override (`???` in the config), as a path
relative to `$STABLEWM_HOME/checkpoints/`. Pass `program_constants.lambda`
explicitly to match what the encoder actually trained at — it is recorded
verbatim in the row, and a stale value makes two rows silently incomparable.

**This scores the encoder alone.** `LeJEPA` has no dynamics by design; the
script only calls `model.encode(...)['emb']` and reads `model.output_dim`.

| Key | Default | Notes |
|---|---|---|
| `checkpoint` | `???` | Mandatory. |
| `ou_dataset` | `ogbench/cube_single_ou_${profile}.lance` | The pairs. Needs a sidecar `<stem>_manifest.json`; a missing one aborts. |
| `style_dataset` | `ogbench/cube_single_ou_style_${profile}.lance` | Same-content/different-style probe. **Armed by default** — see below. A missing one degrades with a warning rather than aborting. |
| `recalibrate_batchnorm` | `true` | Refresh BatchNorm running statistics before scoring. See below; this changes the numbers materially. |
| `max_samples` | 20000 | Embedding batch size is hardcoded at 64 — not a knob. |
| `device` | `cuda` | Falls back to `cpu` when CUDA is unavailable. |
| `results_path` | `outputs/lejepa_results.jsonl` | Relative to the **launch directory**, not `$STABLEWM_HOME`. |

The recovery target is the recorded `latent/z` column, *not* the `privileged/*`
readback columns. `latent/z` is what the sampler **asked for**, and the
collector commands a physical state rather than stepping physics, so the two
agree to the IK solver's residual — `audit_dataset` measures that gap directly
(§4) and it is ~3e-3 in z units.

### Why BatchNorm gets recalibrated, by default

PyTorch's BatchNorm default `momentum=0.1` is an exponential window of roughly
**ten batches**, so the running statistics a checkpoint carries are an estimate
of the input marginal over the last ~2.5k frames of training. Under this
profile's style randomisation — all lighting, both backgrounds, every material
and colour, resampled per frame — that marginal is extremely heterogeneous and
ten batches is a noisy sample of it. The weights are also still moving when
any mid-run checkpoint is taken, so its statistics are *stale* relative to them
as well. The anneal fixes that at the source for the **final** checkpoint, by
letting the weights settle; at `constant_frac: 1.0` they never settle at all and
the statistics cannot converge by construction.

Every number reported here is computed through those statistics, so
`run_metrics.py` does one no-grad pass with the parameters untouched and
`momentum=None` (a true cumulative average), replacing the ten-batch window
with an estimate over the whole sample. `bn_recalibrated` is recorded per row.
Turn it off only to measure the size of the effect, and expect the two rows to
differ.

### How to read a row

Three layers, in this order. Reading the aggregates first is how a localised
failure gets mistaken for a global one.

**1. Per latent.** `probe_linear_per_latent` and `probe_mlp_per_latent`, named
by `probe_latent_names`. This is where a failure is actually diagnosed: a mean
over ten latents whose observability through the renderer spans two orders of
magnitude is not a number about the encoder. The linear/non-linear pair
separates two very different findings — a latent the trained MLP recovers and
the linear probe does not is an encoder that *found* the quantity and failed to
linearise it; one neither recovers was never seen. Both probes are scored on a
held-out split, which is what makes them comparable.

> **A circular latent needs two columns, not one.** `cube.yaw` and
> `effector.yaw` live on a quotient — the cube is C4-symmetric about z, so
> θ and θ+π/2 are *the same pixels*, and the gripper's jaws repeat at π. Every
> function of the image is therefore periodic with that period, so the lowest
> coding available to the encoder is `(cos mθ, sin mθ)` with `m` the symmetry
> order (4 and 2; `latents.CIRCULAR_ORDER`), **not** `(cos θ, sin θ)` and not θ
> itself. `probe_circular_per_latent` scores that harmonic pair on the same
> held-out split as the ordinary probe, so the two are read side by side:
>
> | θ probe | harmonic probe | reading |
> |---|---|---|
> | ~0 | ~0 | the angle is not represented at all |
> | ~0 | high | represented in its natural circular coding — the θ probe was the wrong instrument, not the encoder wrong |
> | high | high | represented and linearised |
>
> Without the second column a yaw latent reading 0.000 is unattributable, which
> is exactly where the 25-epoch run left both of them.

**2. The spectrum.** `canonical_corr` (the σᵢ vector),
`recovered_dimensions` (`Σσ²`, reading as "how many latents' worth of
information") and `canonical_participation` (the effective *number* of
recovered directions, which separates "six directions at 0.85" from "four at
1.0"). `procrustes_mse_per_dim` and `r2_*` are summaries of this one spectrum
and cannot disagree with it — in particular `r2_z_to_h ≈ r2_h_to_z` is *forced*
once `Cov(h) ≈ I`, so their agreement checks whitening, not recovery.

> **`procrustes_mse_per_dim` is not scale-free.** It decodes exactly as
> `(tr Cov(h) + n − 2 Σσ_raw) / n`, so `h = 0` scores **1.0** and an isotropic
> embedding carrying no information at all scores **2.0**. A randomly
> initialised encoder measures near 1.0 because its embedding is *small*, not
> because it is halfway to correct. `procrustes_scale` is reported beside it:
> that is the part of the number which is about agreement.

**3. The bound, and whether it is allowed to say anything.**

- `delta`, split into `delta_content` and `delta_style` by the style probe.
- `residual_hermite_degree` — **read this before anything else in this group.**
  A style-invariant direction of `h` is a function of `z`, and a function of `z`
  decorrelates across one OU step at `ρ^k` for its degree-`k` Hermite content. A
  direction *uncorrelated* with `z` therefore has degree at least 2, which puts
  a hard floor under `delta`:
  `delta ≥ 2ρ(1−ρ)·(tr Cov(h) − recovered_dimensions)`, equivalently
  `D ≥ tr Cov(h) − recovered_dimensions`. Below 2 the measurement is
  **impossible** for any fixed encoder, and `delta_admissible` is false. That is
  not a bad run; it means the embeddings were not produced in eval mode, and no
  bound quoted from that row means anything.
- `predicted_error` and `bound_vacuous`. The bound says nothing once it exceeds
  `n`, which `h = 0` already achieves.
- `mean_probe_r2_needed` — the recovery level at which the bound *could* become
  non-vacuous, from the same floor. Below it, `predicted_error > n` is an
  arithmetic certainty and tracking it epoch to epoch measures nothing. At
  `n = 10` it is around **0.75**, so the bound is not a live target until the
  weakly-observed latents come in.

**4. The ceiling.** `dead_latents`, `n_dead_latents`, `r2_ceiling` and the two
Procrustes floors. `procrustes_floor` is a true lower bound (nothing can beat
it); `procrustes_floor_whitened` assumes whitening holds and is the realistic
target, which sampling noise can cross slightly. Report the aggregates against
these rather than against zero.

### The supervised ceiling — `run_oracle.py`

Every number in a metrics row is conditional on the encoder it was handed, so a
latent scoring ~0 has three explanations the suite cannot separate: the
objective did not pick it up, this backbone cannot represent it, or **the
renderer never put it in the pixels**. In the third case no `f` exists with
`f(g(z)) = Qz`, the theory's optimum is unreachable by construction, and the
zero is not a result about LeJEPA at all.

`observability_ceiling` looks like it settles this but cannot: it takes the
probe scores as an *input*, declares a latent dead because the probe read ~0,
then reports what the aggregates could reach assuming it stays there. Correct
as an aggregate correction, circular as an attribution.

```bash
python scripts/identifiability/run_oracle.py \
    checkpoint=lejepa/weights_epoch_100.pt
```

It loads the checkpoint for its **architecture**, re-initialises every
parameter (logged — a silent zero there would turn the ceiling into a fine-tune
of the encoder it is supposed to bound), and trains the same network on the
same pixels with the labels handed to it. `oracle_per_latent` is then an upper
bound for every row scored on that dataset:

| supervised | LeJEPA | reading |
|---|---|---|
| ~0 | ~0 | not in the pixels, or not representable by this backbone — **not a LeJEPA failure** |
| high | ~0 | the information is there and reachable; the objective did not select it |
| high | high | recovered |

Two things to hold onto. An **under-trained** oracle reports a ceiling that is
too *low*, which is the dangerous direction — it would excuse a genuine failure
as unobservability, so raise `epochs` if the per-latent scores are still
climbing. And a zero here is jointly about the renderer *and* the backbone; to
separate those, re-run against a checkpoint with a different encoder, since the
architecture comes from the checkpoint.

`init=pretrained` keeps the LeJEPA weights and fine-tunes instead. That is a
different question — how far the learned features get with supervision on top —
and is **not** a bound on them.

### The style probe

Collect it with the same collector, `rho` pushed to ~1:

```bash
python scripts/data/collect_cube_single_ou.py \
    dataset_name=ogbench/cube_single_ou_style_physical_content.lance \
    program_constants.rho=0.99999999 num_pairs=20000
```

The sampler refuses `rho = 1` outright (the theory's `rho` lies strictly inside
`(0, 1)`), and at `1 − 1e-8` the residual content drift is ~7e-4 in z units —
sub-pixel once rendered, against ~2.0 for a real step at `rho = 0.9`. So the two
views share their content for all practical purposes and differ only in style.
`run_metrics.py` warns if the drift exceeds 1e-2, which would mean the probe is
also absorbing an OU step.

It is not optional instrumentation. With style redrawn per view, `epsilon`
cannot tell "recovered the latent" from "filled the output direction with style
noise" — both give `epsilon ≈ 0`. `sigma_sq` is the only quantity that separates
them, and it also enters the alignment floor as `2ρσ²` and the gap as
`delta = delta_content + 2ρσ²`.

### Output — the results table

Append-only JSONL, **one row per invocation**. A re-score appends; the reader
(`results.load_rows`) takes the last row per `(checkpoint, dataset)`.

`SCHEMA_VERSION = '2.0.0'` and `METRIC_SUITE_VERSION = '2.0.0'` are on every
row. Five fields are required — `checkpoint`, `dataset`, `seed`,
`config_hash`, `metric_suite_version` — and a row missing any is refused
*before* coercion, so a `None` cannot arrive as the string `"None"`.

> **Metrics pass through whole.** The store this replaced enumerated the columns
> it kept, and so silently discarded everything the suite grew afterwards:
> `probe_linear_per_latent` and `procrustes_scale` were computed on every run
> and thrown away, and they are exactly the per-latent breakdown the study
> turned out to need. There is no whitelist now.

---

## 7. On-disk layout

```
$STABLEWM_HOME/                                  # default ~/.stable_worldmodel
├── datasets/
│   └── ogbench/
│       ├── cube_single_ou_<profile>.lance                # the pairs
│       ├── cube_single_ou_<profile>_manifest.json        # required by run_metrics.py
│       ├── cube_single_ou_<profile>_shard{i}.lance       # transient, see §4
│       ├── cube_single_ou_style_<profile>.lance          # the style probe
│       └── cube_single_ou_style_<profile>_manifest.json
├── checkpoints/
│   ├── lejepa/
│   │   ├── config.json                          # cfg.model only, overwritten each save
│   │   └── weights_epoch_{1..N}.pt              # one per epoch, never pruned
│   └── lejepa_s<seed>/
│       ├── config.yaml                          # the full resolved config
│       ├── encoder_hash.txt                     # 16-char digest
│       ├── lightning_ckpt_dir.txt               # where the trainer state really went
│       └── lightning -> $SPT_CACHE_DIR/runs/<date>/<time>/<run_id>/checkpoints/
└── outputs/
    └── lejepa_results.jsonl                     # append-only, one row per scoring
```

> **Two kinds of checkpoint, and only one has a settable path.**
> `weights_epoch_N.pt` is weights only, written by `SaveCkptCallback`, and is
> what the metric suite scores. `epochNNN.ckpt` is the full trainer state
> (optimizer, scheduler, loops, RNG) and is what a killed pod resumes from —
> Lightning counts from 0, so `epoch004.ckpt` produced `weights_epoch_5.pt`.
> Its directory **cannot be set**: `spt.Manager` always runs in cache_dir mode
> and rewrites every `ModelCheckpoint`'s `dirpath` into
> `$SPT_CACHE_DIR/runs/<date>/<time>/<run_id>/checkpoints/`. What you *can* set
> is `SPT_CACHE_DIR`. `RecordCkptDirCallback` resolves the path at **train
> start** — not after `fit` returns, because the run whose checkpoint directory
> you need to find is precisely the run whose `fit` never returns.

> **Resuming is not wired.** `lejepa.py` passes `ckpt_path=None`, and two traps
> sit behind changing that: `spt.Manager`'s `weights_only` defaults to **True**
> (optimizer and scheduler silently discarded — transfer-init, not resume), and
> Manager auto-resumes from its own `last.ckpt` only when
> `SLURM_RESTART_COUNT >= 1`, which never happens on RunPod.

---

## 8. Config defects — fixed

These stopped the shipped configs from running, or made them lie. All are fixed
in the repo; they are recorded here because a stale checkout will still have
them, and because several describe hazards that can recur. Rows about the
predictor stage, the arms and the V4 calibration were dropped along with that
code.

| # | Was | Now |
|---|---|---|
| 6 | **The blocker.** All three train scripts computed `total_steps = max_epochs * len(train)` and `warmup_steps = 1%` of it, then set `'interval': 'epoch'` — so the scheduler advanced once per *epoch* and its counter only ever reached `max_epochs` (100, against a 703-step warmup). The whole first epoch ran at lr **exactly 0**, the peak reached 14% of the configured lr, and the cosine never annealed, so the frozen checkpoint was taken at the run's highest lr. Measured: 200 steps moved `align` 0.0704→0.0697 and `sigreg` 70.7→70.1, i.e. nothing | `'interval': 'step'` in `lejepa.py`. The same 200 steps then gave `sigreg` 70.7→1.7 and `std(h)` 0.39→0.91. **`lewm.py` and `smwm.py` are deliberately left alone** — the existing DR baselines were trained with them, so changing them now would make future baseline runs incomparable to those checkpoints |
| 7 | `style.resample_within_pair` existed in `ogb_cube_single_ou.yaml` and was read **nowhere** — that yaml line was its only occurrence in the repo, so style was always resampled per view with no way to turn it off | Wired through `collect_pairs`, recorded in the manifest, and covered by `test_style_may_be_shared_across_a_pair`. Setting it false gives the theory's literal deterministic `x = g(z)` as a control arm |
| 8 | `excluded` latents were simply never written — `content_payload` handles content axes and `sample_style` handles style axes — so an excluded axis held whatever the opening `reset(options={'variation': ['all']})` drew for it. Since `base_seed` differs per shard, a sharded collection held a **different constant in every shard**: a nuisance perfectly correlated with shard identity, and recorded nowhere | `excluded_payload` pins every excluded variation axis to its axis's `init_value` on every frame, and the manifest records the values as `excluded_pinned` |
| 10 | The val loader set `drop_last: False`, but SIGReg's statistic is multiplied by the batch size, so the short tail batch (32 of 256 at 20k val samples) reported ~1/8 the statistic and dragged `validate/sigreg_loss_epoch` | `drop_last` stays True on val |
| 11 | `λ = 3e-3` was read off Fig. 6's 2-D grid, and the repo's per-view SIGReg made it **1.5e-3 in the paper's units** — ~30× below the grid's best setting at ρ=0.9, and only ~15× above the λ < 1e-4 band Fig. 6 shows fails at every ρ. Wrong side to err on, given `trace_cov` starts near 1.4 against n=10 | `pool_views: true` puts λ in the paper's units, and `λ = 5e-2` is Fig. 6's best orthogonality error at ρ=0.9 (0.062, against 0.142 / 0.210 / 0.461 for 1e-2 / 5e-3 / 1e-3) |
| 12 | `cube.color` and `background.floor_rgb` are both uniform on [0,1]³ and drawn independently, so the cube occasionally rendered the same colour as the floor and vanished — an undeclared occlusion, and a real injectivity failure, since two cube positions then give the same image. Measured: 4.9% of draws below an L∞ gap of 0.2 | `style.min_contrast: 0.2` redraws those, ~4.9% rejection, style mean essentially unchanged (0.541 → 0.563). Recorded in the manifest as `min_contrast` / `contrast_floor_applies`, and **automatically disabled, loudly, under any profile where `cube.color` is content** — rejecting on a content coordinate would truncate a marginal that is supposed to be Gaussian |
| — | The camera cannot be reframed: it is a pick-and-place frame and must see the whole workspace. So the cube stays at 20×20 px (0.78% of frame) and `cube.pos_xy`'s x component keeps only ~27 px of travel | **Not fixable — report it.** Expect per-dimension recovery on `cube.pos_xy[0]` to trail `[1]` (27 px vs 96 px of travel) and on `cube.pos_z` to trail both. That is a property of `g`, not of the encoder: Thm 1 quantifies over all measurable `h`, so framing affects reachability and δ, never the optimum. Score those dimensions with a stated ceiling rather than letting them drag the aggregate |

### The `physical_content` profile

One profile ([`latents.py`](stable_worldmodel/identifiability/latents.py)):
`n = 10` at one cube — the 9 physical DOFs plus `cube.size`.

| Latent | dims | Note |
|---|---|---|
| `cube.pos_xy` | 2 | The camera cannot be reframed, so `[0]` keeps ~27 px of travel against `[1]`'s 96 px. Expect it to trail. |
| `cube.pos_z` | 1 | Less travel than either. Lifted and interpenetrating cubes are allowed on purpose. |
| `cube.yaw` | 1 | Sampled on 95% of one C4 fundamental domain. **Pixel separability is not monotone in angle**: measured rms is 3.79 at 21°, 4.36 at 43° and 1.17 at 86°, because 90° *is* the identification. The arc is 85.5° wide, so its two endpoints are the least separable pair in it. |
| `effector.pos` | 3 | Reached by IK; round-trips to the solver's tolerance (~3e-3 in z units). |
| `effector.yaw` | 1 | One fundamental domain of the jaws' 2-fold symmetry. |
| `gripper.opening` | 1 | Bounds are the linkage's *measured* achievable range, not `[0, 1]`. |
| `cube.size` | 1 | A direct post-compilation `model.geom_size` write. **The one content latent with no `privileged/*` readback**, so its presence in the render cannot be confirmed from a recorded row — worth remembering if it scores near zero. |

The three profiles that used to sit beside this one — `task_content`,
`arm_c_content` and `all_content` — served comparisons between arms. Re-adding
one is a dict of roles plus an entry in `PROFILES`, which is why the mechanism
was kept.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `KeyError: unknown latent profile '…'` | The only registered profile is `physical_content`. |
| `FileNotFoundError: Cannot resolve '<name>'` | Dataset does not exist. The training script does not collect data. |
| `run_metrics.py` aborts on a missing manifest | Every dataset needs a `<stem>_manifest.json`; it carries `rho`, `config_hash` and `n`. Sharded runs write `_shard{i}_manifest.json` — see §4. |
| `FileNotFoundError: config.json not found in <dir>` | The `checkpoint=` path's directory has no `config.json` — it is written next to the weights by the training run. |
| `ValueError: Ambiguous checkpoint: multiple .pt files` | `checkpoint=` pointed at a directory. Name the `.pt`. |
| `narrow/wide head active: …` in stdout | Intentional if you set `model.head.output_dim`; otherwise a typo — the run will not stop. |
| `delta_admissible: false` in a results row | **Not a bad run.** `delta` is below a floor it cannot be below, so the embeddings were not produced in eval mode (§6). Do not quote the bound from that row. |
| `bound_vacuous: true` with `predicted_error` ≫ `n` | Expected below `mean_probe_r2_needed` (~0.75 at n=10). The bound is arithmetically unable to say anything yet (§6). |
| `style_sensitivity` absent from the row | No `style_dataset` was found (§6). Absent, deliberately, rather than a measured zero — `has_style_probe` says which. |
| `style probe ... has content drift` warning | The probe was not collected at ρ≈1, so it also absorbs the OU step (§6). |
| Results rows not where you expected | `results_path` is relative to the **launch directory**, not `$STABLEWM_HOME`. |
| A metric moved when you did not change the encoder | Check `bn_recalibrated` on both rows. Refreshing BatchNorm statistics changes the numbers materially (§6). |
| Collection uses one core out of N | Expected. The collector is single-threaded; `world.num_envs` is batching. Use `shard=`/`num_shards=`. |
| Collection deadlocks at 0% CPU | The GL-context / background-writer hazard. Do not replace `write_chunked` with `write_episodes` (§4). |
| `outputs/<date>/<time>/` dirs in the repo | Hydra single-run mode. Pass `hydra.run.dir=. hydra.output_subdir=null`. |
| wandb project is `stable-wm`, not `lejepa-identifiability` | `launcher/local.yaml` is merged after `_self_` and wins. Pass `wandb.config.project=` on the CLI. |
| `lr-AdamW` missing from W&B | `LearningRateMonitor` needs a logger, so it is absent whenever `wandb.enabled=false`. |

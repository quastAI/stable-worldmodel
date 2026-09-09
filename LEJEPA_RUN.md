# LeJEPA Identifiability on OGBench Cube-single

Run guide for the **LeJEPA identifiability program** — the designed-OU encoder,
the frozen-encoder predictor, and the metric suite that scores them. This is a
different pipeline from the one in [`Run.md`](Run.md), which covers the
LeWM / SMWM world-model ablations on cube-**quadruple** with domain
randomization. The two share the install, the env vars and the on-disk layout;
they share no datasets, configs or scripts.

Read `Run.md` §2–§3 first for prerequisites and install — everything there
applies unchanged. This document starts after `pip install -e '.[train,format]'`
succeeds.

> **Provenance.** Unlike `Run.md`, **none of the commands below have been run
> end-to-end.** They are derived from the scripts and configs by reading them,
> and every claim is cited to a file and line. The throughput figures are the
> ones the scripts themselves document, not measurements taken here. The one
> exception is the V4 calibration (§9), which was executed and reproduced its
> published boundary.

---

## 1. What this pipeline is

Two training stages, deliberately separated, plus a metric suite:

| Stage | Script | What it learns |
|---|---|---|
| **A** — encoder | `scripts/train/lejepa.py` | A passive, action-free encoder on designed OU **positive pairs**. Objective is `λ·SIGReg + (1-λ)·alignment`. No predictor, no EMA target, no actions. |
| **D** — predictor | `scripts/train/lejepa_predictor.py` | A one-step latent predictor **on top of a frozen encoder**. Objective is plain MSE in the frozen latent space. No SIGReg. |
| metrics | `scripts/identifiability/run_metrics.py` | Scores identifiability (recovery of the true latents) and, separately, planning. |

The separation is the point: isotropy is settled in stage A and cannot be
renegotiated in stage D, so the metrics computed on the encoder's embedding
distribution stay valid after the predictor is trained
([`lejepa_predictor.py:55-58`](scripts/train/lejepa_predictor.py#L55-L58)).

**The arms.** The program compares encoders while holding predictor capacity
fixed (`predictor_hidden_dim: 512`, identical across arms):

| Arm | Encoder | How |
|---|---|---|
| **A** | LeJEPA-trained | `encoder=lejepa/weights_epoch_100.pt` |
| **R** | Random init, frozen | `encoder=random` — architecturally identical, never trained |
| **C1 / C2** | Trajectory-derived pairs instead of OU pairs | `scripts/data/collect_cube_single_arm_c.py arm=c1\|c2` |

**The latent profiles.** Orthogonal to the arms, and the single knob
`profile=` selects one end to end — collection, training, the predictor and the
metrics all interpolate it, so no two ever share a filename:

| `profile=` | `n` | z contains | Use |
|---|---|---|---|
| `physical_content` **(default)** | 10 | the 9 physical DOFs + `cube.size` | The stage-A exit criterion, and the `n` the V4 calibration quotes. |
| `task_content` | 13 | + `cube.color` | A deliberate, separately-reported arm. `cube.color` is entangled with the lighting that stays style, so its 3 dims measure *the cost of a shared rendering channel* rather than clean recovery — see §12. |
| `arm_c_content` | 9 | the 9 physical DOFs only | For arm C, which rebuilds `z` from `compute_ob_info()` and so needs a `privileged/`/`proprio/` readback for every content latent. Also the round-trip tests. |

> **`cube.size` is content and `camera.angle_delta` is excluded, in every
> profile.** One rule: *a nuisance that shares a rendering channel with a
> content latent must not be style*, because style demands exact invariance
> along a direction the content itself depends on.
>
> `cube.size` shares the apparent-footprint cue with `cube.pos_z` and dominates
> it — sweeping size moves the cube's footprint 153 → 585 px, while sweeping
> `pos_z` over its *entire* ±3σ moves it only 272 → 428 px, so the content
> interval sits strictly inside the style interval and footprint carries no
> information about height. As content the requirement drops from "project size
> out" to "span the subspace size and height generate", and since
> identifiability is only up to a rotation *within* the content block, a mixture
> is an acceptable optimum that a linear probe separates afterwards.
>
> `camera.angle_delta` shares the projection channel with every positional
> latent: ±10° moves the cube's image centroid 32–36 px, while `cube.pos_xy`'s
> x component moves only 24–31 px across its whole range — the nuisance was
> larger than the signal. Excluded rather than promoted to content because eval
> holds the camera fixed; if that changes, promote it to content (`n += 2`)
> rather than returning it to style.
>
> Excluded axes are pinned to their axis's `init_value` on every frame and the
> value is recorded in the manifest as `excluded_pinned`. They used to be simply
> *unwritten*, which left them holding whatever the opening `reset` drew — a
> random constant, and a **different one per shard**, i.e. a nuisance correlated
> with shard identity.

```bash
# the exit criterion
python scripts/data/collect_cube_single_ou.py                    # profile default
python scripts/train/lejepa.py

# the shared-channel arm
python scripts/data/collect_cube_single_ou.py latents.profile=task_content
python scripts/train/lejepa.py profile=task_content
```

**The two encoders.** A second `encoder=` group, independent of the profile.
It carries the backbone, the head's shape *and* the optimizer settings
calibrated for it, so the architecture and its learning rate cannot drift
apart:

| `encoder=` | params | head | lr / wd | Why |
|---|---|---|---|---|
| `paper_cnn` **(default)** | 2.86M | bare `Linear(256, n)` | 3e-3 / 1e-4 | **The paper's own pixel encoder** (App. H.11), scaled 64→224px. Every frozen constant — λ, ρ, the lr — was calibrated on it, and its failures are interpretable. |
| `vit_small` | 22.4M | `LayerNorm → 2048 → n` | 5e-5 / 1e-3 | The LeWM baseline. A ViT↔CNN gap is itself a measurement (§8 of the paper names it open), but it is not the default — see below. |

```bash
python scripts/train/lejepa.py                      # paper_cnn
python scripts/train/lejepa.py encoder=vit_small    # labelled second arm
```

> **The paper uses no transformer anywhere.** Its three encoders are a 4-layer
> MLP (2D mixings), a matched inverse-RealNVP (the N=2…1024 sweep, chosen so
> "any failure of identifiability is due to the optimization landscape, not the
> encoder's function class"), and the CNN (pixels). The ViT came from LeWM, not
> from the paper.

> **Why the CNN is the default.** Thm 1 is architecture-agnostic — it asks only
> that `h` be measurable — so a ViT can in principle reach the same optimum.
> But the theorem is a statement about the *global optimum*, and §7 explicitly
> declines to address training dynamics. If a from-scratch ViT gives poor
> recovery you cannot separate "LeJEPA does not identify these latents" from
> "a 22M-param ViT with no augmentation on 400k frames did not converge" — and
> that ambiguity is fatal for an identifiability claim specifically. Note also
> that augmentation is *forbidden* here (any random transform would be an
> undeclared extra difference between the two views), which removes the main
> thing that makes from-scratch ViTs trainable at this data scale.

> **Scaling the CNN to 224px preserves the stage count, not the layer list.**
> The pool is global, so position is encoded in *which* channels fire, which
> only works while each cell of the final map sees most of the frame. The
> reference gets 4×4 at a 46px receptive field (~72% of a 64px frame); four
> stages at 224px would leave 14×14 at ~21% and average the spatial content
> away. Six stages restore it: 3×3 at ~85%. Pinned by
> `test_cnn_pool_stays_global_at_the_design_resolution`.

---

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
run_v4_calibration.py              ->  outputs/v4_calibration.json
   (gate: run once before spending the sweep budget; needs nothing)
        |
        v
collect_cube_single_ou.py          ->  datasets/ogbench/cube_single_ou_<profile>.lance
                                        + cube_single_ou_<profile>_manifest.json
        |
        +---------------------------+
        |                           |
        v                           |  (manifest gate: the predictor collector
train/lejepa.py                     |   reads the OU manifest and refuses to
   -> checkpoints/lejepa/           |   run if the env config disagrees)
        weights_epoch_N.pt          v
        config.json          collect_cube_single_predictor.py
        encoder_hash.txt        -> datasets/ogbench/cube_single_predictor.lance
        |                          + cube_single_predictor_split.json
        |                           |
        +------------+--------------+
        |            |
        v            v
run_metrics.py    train/lejepa_predictor.py
  -> outputs/        -> checkpoints/lejepa_predictor/weights_final.pt
     lejepa_scatter.jsonl
```

Two things worth knowing about this shape:

- **`run_metrics.py` branches off the encoder, not the predictor.** It scores a
  passive encoder and needs no predictor checkpoint at all (§9). The predictor
  is what the *planning* half of the program consumes.
- **Stage A and the predictor dataset are independent** except through the
  manifest gate, so both collections can run before you train anything. Only
  the predictor *training* needs a finished encoder checkpoint.

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
| `latents.profile` | `physical_content` | `physical_content` (n=9), `task_content` (n=12) or `all_content` (n=54) — see §1 and §12. |
| `latents.yaw_half_arc` | 0.746… | `0.95 · π/4`, one fundamental domain of the cube's C4 yaw symmetry. Widening past `π/4` requires `env.marker.enabled=true`. |
| `latents.include_roll_pitch` | false | Switches the sampler to its tangent-space branch. |
| `latents.include_discrete` | false | Discrete latents are renderable but non-Gaussian — a structural V1. |
| `latents.pos_z_max` | 0.15 | Lifted / interpenetrating cubes allowed on purpose, so the induced V5 baseline is zero. |
| `ou.dist` | `gaussian` | `gaussian`, `laplace` or `gennorm`; `gennorm` needs `ou.alpha`. |
| `ou.noise_coupling` | 0.0 | Heteroscedastic noise (the V3 knob). |
| `ou.cross_corr` | 0.0 | Cross-dimension noise correlation (the V9 knob). |
| `violation.name` / `.severity` | `none` / 0.0 | See §10. Severity must be in `[0, 1]`. |
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

## 5. Stage A training — the encoder

```bash
python scripts/train/lejepa.py
```

`data=ogb_cube_single_ou` is the default data group; `profile=` picks which
profile's dataset inside it (§1). Useful overrides:

```bash
python scripts/train/lejepa.py data=ogb_cube_single_ou \
    output_model_name=lejepa_armA subdir=armA_seed3072 \
    trainer.max_epochs=100 loader.batch_size=256 \
    program_constants.lambda=3.0e-3

python scripts/train/lejepa.py data=ogb_cube_single_ou model.head.output_dim=7   # V7a
python scripts/train/lejepa.py data=ogb_cube_single_ou trainer.max_epochs=20     # V8
```

### Knobs

| Key | Default | Notes |
|---|---|---|
| `output_model_name` | `lejepa` | Checkpoint subfolder. |
| `subdir` | `${hydra:job.id}` | Where `config.yaml` / `encoder_hash.txt` go — see the warning below. |
| `program_constants.lambda` | 5.0e-2 | `loss = λ·sigreg + (1-λ)·align`, **in the paper's units** — `loss.sigreg.kwargs.pool_views: true` is what makes that so. Chosen from App. H.11, the paper's own *pixel* sweep, where Fig. 14 puts 5e-2 best at ρ=0.9 (~0.93 vs ~0.75–0.82 for 1e-3/5e-3/1e-2); Fig. 6 agrees it is in the good region, degrading only from 1e-1. Confirm with a 2-point shakedown before spending the seed budget. |
| `loss.sigreg.kwargs.pool_views` | `true` | Pools both views into one set of 2B samples and scales by 2B, as the paper's own implementation does, instead of a per-view statistic averaged over views. Away from the optimum the two differ by exactly 2× (measured 2.00×), so this is what makes λ comparable to the paper's figures instead of leaving a factor of 2 to be remembered. **LeWM keeps the per-view default** — its baselines were trained under it. |
| `program_constants.rho` | 0.9 | **Not read here** — it is the collector's knob, carried so the metric writer can copy it verbatim. |
| `trainer.max_epochs` | 100 | V8 shortens this; severity 0 is the full schedule. |
| `trainer.precision` / `.accelerator` | `32` / `gpu` | **Not bf16**: SIGReg's inner term is a difference of two O(1) quantities whose true value is O(1/√B), then multiplied by B. For a CPU smoke run: `trainer.accelerator=cpu trainer.precision=32 trainer.devices=1 loader.num_workers=0 loader.persistent_workers=false loader.prefetch_factor=null`. |
| `loader.batch_size` | 256 | SIGReg is batch-size-scaled, so its absolute magnitude is not comparable across batch sizes. |
| `optimizer.lr` / `.weight_decay` | `${encoder_lr}` / `${encoder_weight_decay}` | From the encoder group — 3e-3/1e-4 for the CNN, 5e-5/1e-3 for the ViT. Schedule is hardcoded: linear warmup (1% of steps) + cosine, stepped **per optimizer step**. It used to be `interval: 'epoch'` against a step-counted `max_steps`, which left the whole first epoch at lr exactly 0 and capped the peak at 14% of the configured lr with no annealing at all — see §12 defect 6. |
| `encoder` | `paper_cnn` | Config group; also sets `embed_dim`, the head shape and the optimizer (§1). |
| `img_size` | 224 | Shared by both encoders. `patch_size`, `encoder_scale` and `embed_dim` come from the encoder group. |
| `model.head.output_dim` | `null` | `null` ⇒ take `n` from the dataset's `latent/z` width, so `m = n` by construction. See below. |
| `model.head.hidden_dim` / `.norm` | from the encoder group | `null`/`false` for the CNN (its projection already ends in BatchNorm1d + GELU, so the head is a bare `Linear(256, n)` exactly as the reference); `2048`/`true` for the ViT. **Never a norm after the output** — that would force `diag(Cov(h))` to 1 and partly trivialise both ε and SIGReg. |
| `loss.sigreg.kwargs.knots` / `.num_proj` | 17 / 1024 | Epps–Pulley knots and random projections. |
| `train_split` | 0.9 | |
| `seed` | 3072 | **Only** seeds the train/val split and the DataLoader generators — not model init, not SIGReg's projections. Encoder hashes are therefore *not* reproducible from `seed` alone. |

**`model.head.output_dim: null` is the V7 ladder in one field.** `null` takes
`n` from the dataset; an explicit mismatching value is *allowed* and only prints
`V7 dimension misspecification active: head outputs {m}, dataset declares n = {n}`
([`lejepa.py:147-155`](scripts/train/lejepa.py#L147-L155)). A typo will not stop
the run — check stdout for that line.

### What to watch

Logged keys are `fit/…` and `validate/…` (not `train/…`):

| Key | Healthy | Meaning |
|---|---|---|
| `fit/align_loss` | falls | `(h.mean(0) - h)²`. The two views share content up to one OU step and differ entirely in style, so the only way to lower this is to stop representing style. |
| `fit/sigreg_loss` | falls, stays bounded | Epps–Pulley isotropy statistic over 1024 projections. The only thing preventing collapse — there is no EMA target network. |
| `fit/loss` | falls | `λ·sigreg + (1-λ)·align` at λ=3e-3, so numerically dominated by alignment. |
| `fit/whitening_metric` | drifts down, bounded | **Logged, never optimized** — computed under `no_grad` and excluded from the loss. V8 asks how the theory's bound behaves under truncated optimization, which is only a question if ε is independent of what is minimized; optimizing it would make the bound partly self-fulfilling ([`losses.py:35-40`](stable_worldmodel/wm/lejepa/losses.py#L35-L40)). A rise is a legitimate observation, not a bug. |

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

## 6. Predictor dataset — rollouts

```bash
python scripts/data/collect_cube_single_predictor.py \
    num_traj=5000 policy_mixture=0.25 \
    env.width=224 env.height=224
```

Writes `$STABLEWM_HOME/datasets/ogbench/cube_single_predictor.lance` plus
`cube_single_predictor_split.json`, holding the held-out
`eval_rollout_episodes` indices — the last `eval_fraction` of episodes
([`collect_cube_single_predictor.py:208-209`](scripts/data/collect_cube_single_predictor.py#L208-L209)).

This dataset carries **no** `latents` / `violation` / `program_constants`
block, deliberately: its marginal is whatever the policy mixture produces and
is not supposed to be Gaussian, so V1–V4 and V9 do not apply here
([`ogb_cube_single_predictor.yaml:3-4`](scripts/data/config/ogb_cube_single_predictor.yaml#L3-L4)).

| Key | Default | Notes |
|---|---|---|
| `num_traj` | 5000 | Split across shards. |
| `shard` / `num_shards` | 0 / 1 | Seed spacing `seed + shard*(num_traj + world.num_envs)`. |
| `encoder_dataset` | `ogbench/cube_single_ou_${profile}.lance` | The gate — see below. Its *sidecar manifest* carries the declared env config. |
| `eval_fraction` | 0.1 | Tail of episodes reserved for eval. |
| `policy_type` | `plan_oracle` | Must be `plan_oracle` or `markov_oracle`. |
| `policy_mixture` | 0.25 | Fraction of random actions mixed into the expert. `0` uses the bare expert. |
| `p_stack` | 0.8 | |
| `world.num_envs` | 8 | **Batching, not parallelism** — `EnvPool` steps envs in a plain Python loop. Use processes to fill the CPU. |
| `world.max_episode_steps` | 400 | |

> **The manifest gate raises rather than warns.**
> `assert_style_range_matches` reads the sidecar manifest of
> `encoder_dataset` and raises `FileNotFoundError`
> if absent, or `ValueError` if `num_cubes`, `num_digits`, `marker_enabled` or
> `marker_face` disagree with the live env
> ([`collect_cube_single_predictor.py:94-129`](scripts/data/collect_cube_single_predictor.py#L94-L129)).
> **The OU dataset must therefore be collected first.**

> **Render resolution mismatch.** The `env` block sets no `width`/`height`, so
> MuJoCo renders at its 200×200 default and `World` upscales to
> `world.image_shape: [224,224]` — while the encoder dataset renders natively at
> 224. Pass `env.width=224 env.height=224` to match.

---

## 7. Stage D training — the predictor

```bash
python scripts/train/lejepa_predictor.py encoder=lejepa/weights_epoch_100.pt   # arm A
python scripts/train/lejepa_predictor.py encoder=random                        # arm R
```

`encoder=` is a path **relative to `$STABLEWM_HOME/checkpoints/`**, or the
literal `random`. Always name the `.pt` file, not the directory: the encoder run
leaves one checkpoint per epoch, and a directory with two or more `.pt` files
raises `ValueError: Ambiguous checkpoint`.

| Key | Default | Notes |
|---|---|---|
| `encoder` | `lejepa/weights_epoch_100.pt` | Checkpoint path, or `random` for arm R. |
| `predictor_hidden_dim` | 512 | **Held identical across arms A, R, P and C** — this is what makes the arms comparable. |
| `wm.history_size` | 3 | Context length. |
| `wm.num_preds` | 1 | Target offset. The data config derives `num_steps` from these two, so overriding them re-derives it — overriding `data.dataset.num_steps` by hand breaks the alignment with only a shape error. |
| `model.predictor.depth` / `.heads` / `.mlp_dim` | 6 / 16 / 2048 | |
| `loader.batch_size` | 128 | Half the encoder's 256. |
| `random_encoder.head.output_dim` | 9 | **Only used by arm R**; arm A takes its width from the loaded `config.json`. Read at runtime from `encoder_dataset`'s manifest (`latents.n`), so it follows the profile automatically — the literal is only a fallback for an unreachable manifest. |
| `encoder` (group) | `paper_cnn` | **Must match what the encoder was trained under.** Arm R builds its untrained encoder from this group, so a mismatch would compare arms of different architectures. Verified parameter-identical to arm A for both groups. |

There is **no `loss:` block** — SIGReg is deliberately absent from stage D.
The optimizer is scoped to `model.predictor` and `model.action_encoder` only,
belt-and-braces on top of the encoder freeze.

Logged: `fit/loss` and `fit/pred_loss` (a detached alias of the same value),
plus the `validate/` counterparts. `fit/loss` is one-step MSE in the frozen
latent space; it should fall with `validate/loss` tracking it. Absolute MSE is
**not** comparable across arms A and R without care — the two encoders produce
differently-scaled embeddings.

### The frozen-encoder hash

`state_dict_hash` is a 16-char SHA-256 digest over the sorted
`(name, float32 bytes)` of the encoder's `state_dict`
([`module.py:40-62`](stable_worldmodel/wm/lejepa/module.py#L40-L62)). It is
stamped into the predictor's saved config and re-checked when a
`FrozenEncoderWM` is reconstructed:

```
ValueError: frozen encoder hash mismatch: predictor was trained against
encoder <a>, but the encoder supplied hashes to <b>. Planning with a
mismatched pair degrades exactly like an identifiability failure, so this
is refused rather than reported.
```

Operationally: **a predictor checkpoint is usable only with the exact encoder
`.pt` it was trained against.** A different epoch of the same run, a re-run at
the same seed, or arm R vs arm A all hash differently and are refused at
construction. Never delete or overwrite an encoder `.pt` a predictor is bound
to.

> The predictor's saved `config.json` has **no `encoder` key**, so a bare
> `load_pretrained('lejepa_predictor/weights_final.pt')` cannot instantiate —
> the encoder must be injected via
> `load_pretrained(..., extra_args={'encoder': ...})`. Nothing in the repo
> currently loads one back.

---

## 8. Controls — arms C1 and C2

Trajectory-derived pairs instead of OU pairs: instead of drawing `(z, z')` from
the OU process, take two frames from a real rollout separated by a stride.

```bash
python scripts/data/collect_cube_single_arm_c.py arm=c1
python scripts/data/collect_cube_single_arm_c.py arm=c2
python scripts/data/collect_cube_single_arm_c.py arm=c2 frame_stride=7
```

Arm C uses the **`physical_content`** profile (`n = 9`), not `task_content` —
it rebuilds `z` from `env.compute_ob_info()` and so cannot carry
`cube.color`. See §12.

- `arm=c1` — stride 1 unconditionally.
- `arm=c2` with `frame_stride=null` (the default) — searches strides 1…40 and
  picks the one whose mean achieved autocorrelation is closest to
  `program_constants.rho`. The full search table goes into the manifest.
- `arm` interpolates into `dataset_name`, so `arm=c2` alone changes the output
  name to `ogbench/cube_single_arm_c2.lance`.

If `arm=c1` lands more than 0.05 away from the declared `rho`, the script warns
you to run C2 before attributing the A→C1 gap to anything.

> **No sharding.** There is no `shard`/`num_shards` key, and no documented
> timing figures.

> Actions are `env.action_space.sample()` — pure random. The `ExpertPolicy` is
> constructed and `reset()`, but never asked for an action, so `policy_type` and
> `p_stack` are functionally dead here *except* that constructing the policy can
> fail (§12).

> `freeze_appearance: true` and `env.camera` are **dead knobs** — appearance is
> re-randomized every episode regardless, and the camera is hardcoded to
> `front_pixels`.

> The arm-C dataset has a **narrower schema** than the OU dataset (no
> `latent/style`, no `ou/rho_per_dim`, no `privileged/*`/`proprio/*`), and
> `swm merge` rejects a column mismatch — arm-C and OU tables cannot be merged.

---

## 9. Metrics

```bash
python scripts/identifiability/run_metrics.py \
    checkpoint=lejepa/weights_epoch_100.pt \
    ou_dataset=ogbench/cube_single_ou_<profile>.lance \
    rollout_dataset=ogbench/cube_single_predictor.lance
```

`checkpoint` is the only mandatory override (`???` in the config). It is a path
relative to `$STABLEWM_HOME/checkpoints/`.

**This scores the encoder alone — no predictor, no planner.** `LeJEPA` has no
dynamics by design; the script only calls `model.encode(...)['emb']` and reads
`model.output_dim` ([`run_metrics.py:75,95-96`](scripts/identifiability/run_metrics.py#L75)).
A `lejepa_predictor` checkpoint is not an input here.

| Key | Default | Notes |
|---|---|---|
| `checkpoint` | `???` | Mandatory. |
| `arm` | `A` | Free-form text, unvalidated. Conventional values: `A`, `R`, `P`, `O`, `C1`, `C2`. |
| `ou_dataset` | — | Scored as `distribution='ou'`. |
| `rollout_dataset` | — | Scored as `distribution='rollout'`. Pass `null` to skip (see below). |
| `max_samples` | 20000 | Embedding batch size is hardcoded at 64 — not a knob. |
| `device` | `cuda` | Falls back to `cpu` when CUDA is unavailable. |
| `success_rate`, `success_rate_oracle`, `success_rate_state` | `null` | **Filled in by hand** — see below. |
| `style_dataset` | `null` | A same-content/different-style probe; enables the style-invariance metric. See below. |
| `scatter_path` | `outputs/lejepa_scatter.jsonl` | Relative to the **launch directory**, not `$STABLEWM_HOME`. |

**Both datasets are loaded with `keys_to_load=['pixels', 'latent/z']`** — the
recovery target is the recorded `latent/z` column, *not* the `privileged/*`
readback columns. Each dataset must also have a sidecar
`<stem>_manifest.json`, which supplies `rho`, the violation key and severity,
`config_hash`, `latents.n` and the isotropy margin. **A missing manifest aborts
the run.**

> **The default `rollout_dataset` cannot be scored as shipped.**
> `collect_cube_single_predictor.py` writes only `<stem>_split.json` — it never
> calls `write_manifest` — and the rollout dataset carries **no `latent/z`
> column** (that is written only by the OU and arm-C collectors). So the rollout
> half raises `FileNotFoundError` (manifest) or `KeyError` (`latent/z`).
>
> Worse, `append_rows` is called **once, after both distributions**, so a
> rollout-side failure **discards the already-computed OU row too**. Until this
> is fixed, pass `rollout_dataset=null`:
>
> ```bash
> python scripts/identifiability/run_metrics.py \
>     checkpoint=lejepa/weights_epoch_100.pt \
>     ou_dataset=ogbench/cube_single_ou_<profile>.lance \
>     rollout_dataset=null
> ```
>
> It logs `no rollout dataset given -- the OU/rollout gap is a measurement the
> plan asks for, so this row is incomplete.` — which is accurate: the row is
> usable but the OU/rollout gap is missing.

> **Success rates are not plumbed.** The three `success_rate*` keys default to
> `null` and are passed straight through. `scripts/plan/eval_wm.py` only prints
> and appends to a text file, and `World.evaluate` returns success as a
> **percent**. Copy them in by hand, consistently in one unit — the scatter only
> forms ratios (`sr_over_o`, `sr_over_p`).

### Output — the global scatter

Append-only, **one row per distribution** (so up to 2 rows per invocation),
logged as `appended N rows -> <path> (M total)`. Format is chosen by **file
suffix**: `.parquet` does a pandas read-concat-replace, anything else is plain
JSONL append. (The docstring says "Parquet when pyarrow is available" — the code
actually keys off the suffix.)

`SCATTER_SCHEMA_VERSION = '1.0.0'`, and `METRIC_SUITE_VERSION = '1.0.0'` is
recorded in every row. Seven columns are required
(`arm`, `violation`, `severity`, `seed`, `distribution`, `config_hash`,
`metric_suite_version`); a row missing any is refused.

### The suite

Frozen and versioned. `compute_all` runs, in order:

| Row fields | What it measures |
|---|---|
| `r2_z_to_h`, `r2_h_to_z` | Linear R² both directions. `R²(h→z)` high while `R²(z→h)` collapses is the **V4 second-Hermite signature**. |
| `procrustes_mse_per_dim` | **The criterion metric** — residual of the best orthogonal alignment `h ≈ Qz`. Per-dim is the `n`-comparable number, and it is what `measured_recovery_error` reports. |
| `orth_err_normalized`, `cond` | `‖AᵀA−I‖_F/√n` and the condition number of the best linear map. `cond` is the one that matters for planning. |
| `monotone_mse_per_dim` | Procrustes recovery after a per-coordinate isotonic (PAVA) warp — for latents with no Gaussian marginal. |
| `mcc_unaligned` | Greedy-matched mean correlation. **Logged, permutation-only, never a gate** — it penalizes correct rotations by construction. |
| `probe_linear_r2`, `probe_mlp_r2` | **The decoy metric.** Unaligned per-latent readout of `z` from `h`; the MLP rung is a deterministic random-feature ridge, no training loop. |
| `hermite2_excess` | How much better `He₂(z) = z²−1` explains `h` than the raw latent. Positive = the V4 substitution is actually happening. |
| `sigreg_z` | SIGReg statistic z-scored against a matched i.i.d.-Gaussian null recomputed at this sample size and width. |
| `epsilon` | `‖Cov(h) − I‖_F` — the bound's first term. |
| `delta` | Excess pair distance beyond what the OU step accounts for, clamped at 0. Only when a second view is present. |
| `D`, `predicted_error`, `spectral_gap`, `anisotropic` | The theory's prediction `D + (ε + D)²` with `D = δ / (2ρ(1−ρ))`. A ρ **vector** is reduced by its mean and sets `anisotropic`. |
| `probe_divergence` | `probe_linear_r2 − 1/(1 + orth_err_normalized)` — how far probe-ability has come apart from orthogonal recovery. |
| `recovery_in_gap_units` | Recovery error in spectral-gap units, for cross-environment comparability. |

### The style probe

`style_sensitivity` measures the thing the alignment loss is *supposed* to do:
discard style. It is the only number that distinguishes `task_content` from
Thm 1's setting, so under that profile it is not optional.

It cannot be computed from an ordinary OU pair — those two views differ by one
OU step **as well as** style, and scoring that would charge the transition as
style leakage. It needs a probe whose two views share content exactly, and the
OU collector already makes one: hold the content fixed by pushing ρ to 1.

```bash
# collect the probe -- same collector, content held fixed
python scripts/data/collect_cube_single_ou.py \
    dataset_name=ogbench/cube_single_ou_style_physical_content.lance \
    program_constants.rho=0.99999999 num_pairs=20000

# pass it to the metrics run
python scripts/identifiability/run_metrics.py \
    checkpoint=lejepa/weights_epoch_100.pt \
    style_dataset=ogbench/cube_single_ou_style_physical_content.lance \
    rollout_dataset=null
```

> **ρ=1 is rejected** — the sampler enforces the paper's ρ ∈ (0,1) strictly. At
> 1e-8 below it the residual content drift is ~7e-4 in z units, sub-pixel once
> rendered, against ~2.0 for a real step at ρ=0.9. `run_metrics` warns when a
> probe's drift exceeds 1e-2, so one collected at the wrong ρ cannot quietly be
> scored as style leakage.

Vacuous under `all_content` (no continuous style left to resample); the metric
sets `style_vacuous` when the two views embed identically. With no
`style_dataset` the field is **absent** from the row rather than recorded as a
measured zero.

### Arm-C rows

Arm-C datasets *are* scoreable (they have both `latent/z` and a manifest), but
their manifest has no `ou` or `violation` block, so `rho` falls back to
`program_constants.rho`, `violation` → `'none'`, `severity` → `0.0`, and
`rho_per_dim` → `[NaN]`.

### V4 calibration — run this before spending the sweep budget

```bash
python scripts/identifiability/run_v4_calibration.py
python scripts/identifiability/run_v4_calibration.py n=16 num_samples=40000
```

**No checkpoint, no dataset, no GPU** — it uses a *synthetic* analytically-optimal
encoder, because training would confound the boundary with optimization noise.
Pure numpy.

It proves the frozen metric suite can resolve the one published quantitative
prediction: recovery fails once `min ρ_α ≤ (max ρ_α)²`. At the frozen `ρ = 0.9`
with `n = 9` the crossing sits at **severity 0.32125**, and the ladder is 9 rungs
bracketing it — `(0, 0.1285, 0.19275, 0.257, 0.32125, 0.3855, 0.44975, 0.514, 1.0)`
— rather than a uniform sweep, which would put exactly one rung either side with
nothing nearby.

Writes `outputs/v4_calibration.json` (relative to the launch dir) plus a printed
table and three PASS/FAIL lines: `recovery_degrades_past_boundary`,
`r2_z_to_h_drops_past_boundary`, `hermite2_detectable_past_boundary`. **Any
failure exits non-zero:**

```
V4 calibration failed: the metric suite does not resolve the published
boundary. Fix this before spending the sweep budget -- every V4 row would
otherwise be uninterpretable.
```

The JSON is written *before* the exit, so a failed calibration still leaves its
evidence on disk.

> Nothing reads `v4_calibration.json` — the dependency is **procedural**, not
> wired. It is a gate you honour, not one the code enforces. And a calibration
> run at a different `rho` does not license the production sweep.

> The calibration is pinned to `n: 12` to match `task_content`. The boundary's
> location does **not** depend on `n` — the per-dim ρ fan is ±spread for any
> `n > 1`, so the crossing stays at ~0.32 — but the number should name the
> profile actually being swept.

---

## 10. The violation ladder

Selected on the OU collector:

```bash
python scripts/data/collect_cube_single_ou.py \
    violation.name=v4 violation.severity=0.32 \
    dataset_name=ogbench/cube_single_ou_v4_s032.lance
```

`violation.name` ∈ `none, v1, v2, v3, v4, v5, v6, v7, v8, v9`;
`violation.severity` ∈ `[0, 1]` (out of range raises).

| Name | Site | What it perturbs |
|---|---|---|
| `v1` | sampler | Non-Gaussian marginal — `dist=gennorm`, `alpha` from 2.0 toward 0.5. |
| `v2` | sampler | Non-stationarity — a drift with period 64. |
| `v3` | sampler | State-dependent noise — `noise_coupling = severity·1.5`. |
| `v4` | sampler | Anisotropic transitions — `rho` becomes a per-dim fan. |
| `v5` | sampler | Support truncation — rejection sampling at a radius in units of `√n`. |
| `v9` | sampler | Cross-dimension noise correlation. |
| `v6` | **env** | Occlusion. ⚠ Not applied by the collector. |
| `v7` | **training** | Dimension misspecification — set `model.head.output_dim` instead. |
| `v8` | **training** | Optimization gap — set `trainer.max_epochs` instead. |

> **`v6`, `v7` and `v8` produce an *unviolated* dataset plus a warning.** The
> collector only calls `sampler_kwargs()`; `env_kwargs()` and
> `training_overrides()` are never applied
> ([`collect_cube_single_ou.py:133-138`](scripts/data/collect_cube_single_ou.py#L133-L138)).
> Apply those by hand: `env.num_digits=` for V6, `model.head.output_dim=` for
> V7, `trainer.max_epochs=` for V8.

Per-violation sub-parameters (`v1.alpha_min`, `v4.max_spread_frac`, `v7.mode`, …)
are **not exposed on the command line** — `make_violation` is called with no
extra kwargs, so they stay at their dataclass defaults.

Severity ladders for sweep planning: `(0.0, 0.25, 0.5, 0.75, 1.0)` for the
quantitative set (`v1`, `v4`, `v8`) at 5 seeds; `(0.0, 0.33, 0.67, 1.0)` for
everything else at 3 seeds.

---

## 11. On-disk layout

```
$STABLEWM_HOME/                                  # default ~/.stable_worldmodel
├── datasets/
│   └── ogbench/
│       ├── cube_single_ou_<profile>.lance                 # stage A pairs
│       ├── cube_single_ou_<profile>_manifest.json         # the predictor collector's gate
│       ├── cube_single_ou_<profile>_shard{i}.lance         # transient, see §4
│       ├── cube_single_predictor.lance          # stage D rollouts
│       ├── cube_single_predictor_split.json     # held-out eval episodes
│       └── cube_single_arm_c{1,2}.lance         # controls
└── checkpoints/
    ├── config.yaml                              # ⚠ empty `subdir` puts these
    ├── encoder_hash.txt                          #    at the root, overwritten
    ├── lejepa/
    │   ├── config.json
    │   └── weights_epoch_{1..100}.pt
    └── lejepa_predictor/
        ├── config.json
        └── weights_epoch_N.pt, weights_final.pt
```

---

## 12. Config defects — fixed

These five stopped the shipped configs from running, or made them lie. All are
now fixed in the repo; they are recorded here because a stale checkout will
still have them.

| # | Was | Now |
|---|---|---|
| 1 | `ogb_cube_single_arm_c.yaml` and `ogb_cube_single_predictor.yaml` set `policy_type: plan`, which fails `ExpertPolicy`'s assert | `policy_type: plan_oracle` in both |
| 2 | `lejepa_predictor.yaml` hardcoded `random_encoder.head.output_dim: 9`, so arm R trained at the wrong width | `output_dim: 12`, matching `task_content` (9 physical + 3 `cube.color`) |
| 3 | Arm C used `task_content`, whose `cube.color` has no physical readback, so it collected **zero** trajectories — and two readback tests failed the same way | Arm C has its own `physical_content` profile, and both readback tests now use it — see below |
| 4 | `collect_cube_single_predictor.py` writes no manifest and no `latent/z`, so the rollout half of a metrics run cannot be scored | **Still open** — use `rollout_dataset=null` (§9) |
| 5 | `v4_calibration.yaml` pinned `n: 9` and called it "the stage-A exit-criterion profile (task_content)" | `n: 10`, naming `physical_content`, with a note that the boundary does not actually depend on `n` |
| 6 | **The blocker.** All three train scripts computed `total_steps = max_epochs * len(train)` and `warmup_steps = 1%` of it, then set `'interval': 'epoch'` — so the scheduler advanced once per *epoch* and its counter only ever reached `max_epochs` (100, against a 703-step warmup). The whole first epoch ran at lr **exactly 0**, the peak reached 14% of the configured lr, and the cosine never annealed, so the frozen checkpoint was taken at the run's highest lr. Measured: 200 steps moved `align` 0.0704→0.0697 and `sigreg` 70.7→70.1, i.e. nothing | `'interval': 'step'` in `lejepa.py` and `lejepa_predictor.py`. The same 200 steps then gave `sigreg` 70.7→1.7 and `std(h)` 0.39→0.91. **`lewm.py` and `smwm.py` are deliberately left alone** — the existing DR baselines were trained with them, so changing them now would make future baseline runs incomparable to those checkpoints |
| 7 | `style.resample_within_pair` existed in `ogb_cube_single_ou.yaml` and was read **nowhere** — that yaml line was its only occurrence in the repo, so style was always resampled per view with no way to turn it off | Wired through `collect_pairs`, recorded in the manifest, and covered by `test_style_may_be_shared_across_a_pair`. Setting it false gives the theory's literal deterministic `x = g(z)` as a control arm |
| 8 | `excluded` latents were simply never written — `content_payload` handles content axes and `sample_style` handles style axes — so an excluded axis held whatever the opening `reset(options={'variation': ['all']})` drew for it. Since `base_seed` differs per shard, a sharded collection held a **different constant in every shard**: a nuisance perfectly correlated with shard identity, and recorded nowhere | `excluded_payload` pins every excluded variation axis to its axis's `init_value` on every frame, and the manifest records the values as `excluded_pinned` |
| 9 | `lejepa_predictor.yaml`'s arm-R fallback width `output_dim: 9` tracked the old `physical_content` | `output_dim: 10`. Still only a fallback — the width is read from the encoder dataset's manifest at runtime |
| 10 | The val loader set `drop_last: False`, but SIGReg's statistic is multiplied by the batch size, so the short tail batch (32 of 256 at 20k val samples) reported ~1/8 the statistic and dragged `validate/sigreg_loss_epoch` | `drop_last` stays True on val |
| 11 | `λ = 3e-3` was read off Fig. 6's 2-D grid, and the repo's per-view SIGReg made it **1.5e-3 in the paper's units** — ~30× below App. H.11's best *pixel* setting at ρ=0.9, and only ~15× above the λ < 1e-4 band Fig. 6 shows fails at every ρ. Wrong side to err on, given `trace_cov` starts near 1.4 against n=10 | `pool_views: true` puts λ in the paper's units, and `λ = 5e-2` matches Fig. 14's best pixel setting at ρ=0.9 |
| 12 | `cube.color` and `background.floor_rgb` are both uniform on [0,1]³ and drawn independently, so the cube occasionally rendered the same colour as the floor and vanished — an undeclared occlusion, and a real injectivity failure, since two cube positions then give the same image. Measured: 4.9% of draws below an L∞ gap of 0.2 | `style.min_contrast: 0.2` redraws those, ~4.9% rejection, style mean essentially unchanged (0.541 → 0.563). Recorded in the manifest as `min_contrast` / `contrast_floor_applies`, and **automatically disabled under `task_content`**, where `cube.color` is content and rejecting on it would truncate a coordinate that is supposed to be Gaussian |
| — | The camera cannot be reframed: it is a pick-and-place frame and must see the whole workspace. So the cube stays at 20×20 px (0.78% of frame) and `cube.pos_xy`'s x component keeps only ~27 px of travel | **Not fixable — report it.** Expect per-dimension recovery on `cube.pos_xy[0]` to trail `[1]` (27 px vs 96 px of travel) and on `cube.pos_z` to trail both. That is a property of `g`, not of the encoder: Thm 1 quantifies over all measurable `h`, so framing affects reachability and δ, never the optimum. Score those dimensions with a stated ceiling rather than letting them drag the aggregate |

### The `physical_content` profile

There are now four profiles, not two
([`latents.py`](stable_worldmodel/identifiability/latents.py)):

| Profile | `n` (at 1 cube) | Content | For |
|---|---|---|---|
| `physical_content` | 10 | 9 physical DOFs + `cube.size` | **Stage A. The exit-criterion profile.** |
| `task_content` | 13 | + `cube.color` | The shared-rendering-channel arm. |
| `arm_c_content` | 9 | The 9 physical DOFs only | **Arm C**, and anything else that needs a readback for every content latent. |
| `all_content` | 52 | Every content-capable latent | Scaling datapoint only — see the warning below. |

> **`all_content` is not the way to get a robust content encoder**, even though
> it does make `x = g(z)` deterministic. Two things break instead. With no style
> resampled within a pair, appearance carries the same autocorrelation as
> physics, so the transition operator no longer ranks it below content and
> *nothing in the objective prefers cube position over light colour*. And the
> photometric block is jointly non-identifiable from a single frame (the image
> constrains albedo × illumination, not the factors), so `g` collapses
> directions in the 52-dim `z`: nominal `m = n`, but the effective `n` is
> smaller, which is the `m > n` regime the theory explicitly declines to cover.
> SIGReg then demands unit variance in directions carrying no image
> information, which the encoder can only supply by amplifying JPEG and
> aliasing noise or by bleeding content into them.

The split exists because the two collectors get `z` from different places. The
OU collector *writes* `z` from the sampler and the metric suite reads it back
from the recorded `latent/z` column, so `cube.color` is fine there. Arm C
instead **rebuilds** `z` from `env.compute_ob_info()`, which carries only
`privileged/*` and `proprio/*` keys — an appearance latent's `variation.*`
readback is simply absent, and the collector's "every content latent readable"
gate then rejects every frame without raising. That failure is silent, which is
why this is a separate profile rather than a per-run override, and why
`test_physical_content_is_nine_physical_dims` asserts every content latent has
a `privileged/`/`proprio/` readback.

The same constraint governs the two round-trip tests
(`test_readback_z_matches_requested_z`,
`test_readback_recovers_the_latents_that_were_written`), which both now run on
`physical_content`. That is not a workaround: what they assert is that
*simulator* ground truth tracks the requested `z` rather than drifting from it,
and that drift — a clipped value, an unconverged IK solve, a coupled joint that
did not track its driver — is a property of latents written into `qpos`.
`cube.color` is a direct `geom.rgba` write with nothing in between, so there is
no divergence for them to detect.

**What this means for scoring `cube.color`.** It is in `latent/z`, written from
the sampler, and [§9](#9-metrics)'s suite reads its target from that recorded
column — so it *is* scored, and stage A trains at `n = 12` correctly. What it
does not have is a simulator-verified ground truth, because nothing records the
rendered value back. For a directly-written axis that distinction is thin; for
anything that could be clipped or coupled it would matter, so do not promote a
*state*-written latent this way without also recording its readback.

> **Defect 4 remains.** Fixing it means either writing a manifest from the
> predictor collector and recording a `latent/z` column for rollout frames, or
> deciding that the rollout distribution is scored some other way. That is a
> design call, not a config typo.

## 13. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `KeyError: unknown latent profile '…'` | Registered profiles are `task_content`, `physical_content` and `all_content`. |
| `AssertionError: Invalid policy_type` | Stale checkout — `policy_type=plan_oracle` (§12). |
| `RuntimeError: no usable trajectories were collected.` | A content latent has no physical readback. Arm C must run on `physical_content` (§12). |
| `FileNotFoundError` on the encoder manifest | The OU dataset must be collected before the predictor dataset (§6). Sharded runs write `_shard{i}_manifest.json` — see §4. |
| `ValueError` naming `num_cubes` / `num_digits` / `marker_*` | Env config drifted between the OU and predictor collections (§6). |
| `ValueError: Ambiguous checkpoint: multiple .pt files` | `encoder=` pointed at a directory. Name the `.pt`. |
| `ValueError: frozen encoder hash mismatch` | The predictor is bound to one exact encoder `.pt` (§7). |
| `FileNotFoundError: config.json not found in <dir>` | `encoder=` path's directory has no `config.json` — it is written next to the weights by the encoder run. |
| `FileNotFoundError: Cannot resolve '<name>'` | Dataset does not exist. Neither training script collects data. |
| `V7 dimension misspecification active: …` in stdout | Intentional if you set `model.head.output_dim`; otherwise a typo — the run will not stop. |
| Collection uses one core out of N | Expected. Both collectors are single-threaded; `world.num_envs` is batching. Use `shard=`/`num_shards=`. |
| Collection deadlocks at 0% CPU | The GL-context / background-writer hazard. Do not replace `write_chunked` with `write_episodes` (§4). |
| `outputs/<date>/<time>/` dirs in the repo | Hydra single-run mode. Pass `hydra.run.dir=. hydra.output_subdir=null`. |
| `run_metrics.py` aborts on a missing manifest | Datasets need a `<stem>_manifest.json`. The rollout dataset never gets one — §12 defect 4. |
| `KeyError: latent/z` in `run_metrics.py` | The rollout dataset has no such column — §12 defect 4. Use `rollout_dataset=null`. |
| Metrics row logged as "incomplete" | Expected with `rollout_dataset=null`; the OU/rollout gap is simply not measured. |
| `V4 calibration failed: …` non-zero exit | The suite cannot resolve the published boundary. Do not start the sweep; the JSON evidence is still written (§9). |
| `style_sensitivity` absent from the row | No `style_dataset` was given (§9). Absent, deliberately, rather than a measured zero. |
| `style probe ... has content drift` warning | The probe was not collected at ρ≈1, so it also absorbs the OU step (§9). |
| Scatter rows not where you expected | `scatter_path` is relative to the **launch directory**, not `$STABLEWM_HOME`. |
| wandb project is `stable-wm`, not `lejepa-identifiability` | `launcher/local.yaml` is merged after `_self_` and wins. Pass `wandb.config.project=` on the CLI. |

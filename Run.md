# Short-Horizon Real-Time Planning on OGBench Cube

Setup and run guide for the LeWM / JEPA world-model ablations on **OGBench
cube-quadruple** with extended domain randomization, inside
**stable-worldmodel (SWM)**.

Everything below was run end-to-end on macOS 26 (arm64, Python 3.12, M1 Max) in
a clean virtualenv; the timings are measured there. The Ubuntu path — install
list, `osmesa` renderer, Python range — mirrors the project's CI, which runs
3.10/3.11/3.12 on `ubuntu-latest` with `MUJOCO_GL=osmesa`, so it is exercised
on every push; the container-specific notes and the CPU-node throughput
estimate in §5.5 are **not** verified, and §5.5 says how to check them in a
minute on the box itself.

---

## 1. Repository layout

| Directory | Role |
|---|---|
| `stable-worldmodel/` | **Primary.** The framework: envs, datasets, world models (LeWM/SMWM/DINO-WM/PLDM/TD-MPC2), planners, training and eval scripts. All project code lives here. |
| `ogbench/` | Upstream OGBench checkout (v1.2.1). Read-only reference — SWM installs `ogbench` from PyPI at the same version. Only install this editable if you intend to modify OGBench itself. |
| `le-wm/` | Original LeWM paper repo. **Parity reference only** — do not develop here. Its eval script targets an older SWM API (see §8). |

The project-specific pieces added to SWM:

- `swm/OGBCubeDR-v0` — `stable_worldmodel/envs/ogbench/dr_cube_env.py`
- oracle-subgoal evaluation — `stable_worldmodel/world/world.py`
- collection — `scripts/data/collect_cube_quadruple_dr.py` and the parallel
  driver `scripts/data/collect_cube_quadruple_dr_sharded.py`
- configs — `scripts/{data,plan,train}/config/*cube_quadruple_dr*`
- a seeding fix in `stable_worldmodel/envs/ogbench/cube_env.py` — see §5.4

---

## 2. Prerequisites

### Python

Use **3.10, 3.11 or 3.12**.

> **3.13 / 3.14 do not work.** `dm_control` depends on `labmaze`, which ships no
> wheel for those versions and falls back to a Bazel source build that is not
> available on a normal machine. The failure looks like
> `error: command 'bazel' failed: No such file or directory`.

### System packages — Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y \
  libgl1-mesa-dev libgl1 libglx-mesa0 libglfw3 \
  libosmesa6-dev libegl1 libopengl0 patchelf
```

(This is exactly the list CI installs, on `ubuntu-latest` with
`MUJOCO_GL=osmesa`.) Add `swig` only if you plan to install the full `[all]`
extra — see §3.

The package names above are the post-22.04 ones: `libgl1-mesa-glx` was split
into `libgl1` + `libglx-mesa0` and no longer exists on 24.04. `libosmesa6-dev`
is the one that matters on a headless box — it provides the software rasterizer
that `MUJOCO_GL=osmesa` needs. `libglfw3` and `libegl1` are only useful if you
have a display or a GPU respectively; installing all of them costs nothing and
keeps every renderer option open.

**In a container (RunPod, Docker) you are usually already root**, so drop
`sudo` — and images built `FROM nvidia/cuda` or `pytorch/pytorch` often set
`DEBIAN_FRONTEND` unset, which makes `apt-get install` hang on a tzdata prompt:

```bash
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libgl1-mesa-dev libgl1 libglx-mesa0 libglfw3 \
  libosmesa6-dev libegl1 libopengl0 patchelf
```

### System packages — macOS

None required. MuJoCo renders through the system GL, and video writing uses the
`ffmpeg` binary bundled by `imageio[ffmpeg]`, so no Homebrew `ffmpeg` is needed.

Install `swig` (`brew install swig`) only if you want the full `[all]` extra.

---

## 3. Install

```bash
cd stable-worldmodel
python3.12 -m venv .venv          # or: uv venv --python 3.12
source .venv/bin/activate

pip install -e '.[train,format]'
pip install ogbench pygame pymunk shapely opencv-python-headless \
            gymnasium-robotics pytest
```

That is the **verified** install: it brings up the CLI and passes the entire
test suite (1211 passed, 32 skipped) on a clean macOS venv.

> **On a Linux CPU node, install torch first — from the CPU index.** `torch` and
> `torchvision` are core dependencies of SWM, and on linux-x86_64 plain
> `pip install torch` resolves to the CUDA build: it drags in the `nvidia-*`
> wheels for several GB and is useless without a GPU. On a container with a
> 10–20 GB overlay that alone can fill the disk.
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -e '.[train,format]'      # now resolves against the CPU build
> ```
>
> Collection itself never touches torch beyond the import, so the CPU build is
> all you need to build the dataset.

Neither `ruff` nor `pre-commit` is in any extra — install them separately if
you intend to lint (§8).

<details>
<summary>Why not just <code>pip install -e '.[all]'</code> / <code>uv sync --all-extras</code>?</summary>

`[all]` pulls the `env` extra, which includes `gymnasium[all]` → `box2d-py`,
which needs `swig` at build time. Without it the install aborts:

```
error: command 'swig' failed: No such file or directory
ERROR: Failed building wheel for box2d-py
```

Box2D is irrelevant to this project (it backs gymnasium's classic Box2D envs).
Either install `swig` first, or use the scoped command above. CI uses
`uv sync --all-extras` on Ubuntu, where `swig` is present in the image.

</details>

Optional — use the local OGBench checkout instead of the PyPI package (only if
you are editing OGBench):

```bash
pip install -e ../ogbench
```

### Environment variables

```bash
# Where datasets and checkpoints live. Default: ~/.stable_worldmodel
export STABLEWM_HOME=/path/with/space          # ~30 GB at the shipped config, see §5

# Renderer backend — pick per platform:
export MUJOCO_GL=glfw                          # macOS, or Linux with a display
# export MUJOCO_GL=egl                         # headless Linux with a GPU
# export MUJOCO_GL=osmesa                      # headless Linux, CPU-only (CI, RunPod CPU pods)
# export PYOPENGL_PLATFORM=osmesa              # set alongside MUJOCO_GL=osmesa
```

| Where you are | `MUJOCO_GL` | Also set |
|---|---|---|
| macOS | `glfw` | — |
| Linux with a display | `glfw` | — |
| Headless Linux **with** a GPU | `egl` | — |
| Headless Linux, **CPU only** | `osmesa` | `PYOPENGL_PLATFORM=osmesa` |

> `MUJOCO_GL=egl` and `osmesa` are **rejected on macOS** (`RuntimeError: invalid
> value for environment variable MUJOCO_GL`). `glfw` needs a display, so it
> fails on a headless node. `egl` needs a GPU, so it fails on a CPU-only node —
> which matters because `scripts/plan/eval_wm.py` **defaults to `egl`**. Export
> the variable yourself on any remote box; the scripts only ever `setdefault`,
> so an explicit export always wins.

`collect_cube_quadruple_dr.py` picks its default from the platform — `glfw` on
macOS or when `DISPLAY` is set, `osmesa` (plus `PYOPENGL_PLATFORM`) otherwise —
so collection starts on a headless CPU node even if you forget. Nothing else in
the repo does this, so still export it.

Add `STABLEWM_HOME` and `MUJOCO_GL` to your shell profile; every command below
assumes they are set. On RunPod, `~` is on the container overlay and is wiped
when the pod is recreated — point `STABLEWM_HOME` at the persistent volume:

```bash
export STABLEWM_HOME=/workspace/.stable_worldmodel
```

### Smoke test

```bash
swm envs | grep OGBCubeDR          # -> swm/OGBCubeDR-v0   Continuous
swm fovs OGBCubeDR-v0              # table of all 22 randomization axes
pytest tests/envs/test_ogbench_cube_dr.py -q
```

---

## 4. The environment

`swm/OGBCubeDR-v0` extends `swm/OGBCube-v0` with the randomization axes the
stock env lacks. Construct it through `swm.World` or `gym.make`:

```python
import stable_worldmodel as swm

world = swm.World(
    'swm/OGBCubeDR-v0', num_envs=8, image_shape=(224, 224),
    env_type='quadruple', mode='data_collection', terminate_at_goal=False,
    num_digits=1, num_bg_materials=8, add_backdrop=True,
)
obs_info = world.reset(options={'variation': ['all']})
```

| Group | Axes | Cost |
|---|---|---|
| `light.*` | `position`, `direction`, `diffuse`, `ambient`, `specular`, `headlight_diffuse` | free |
| `background.*` | `floor_material`, `wall_material` | free |
| `digit.*` | `value`, `position`, `yaw` | free |
| `cube.*` | `color`, `start_position`, `start_yaw`, `goal_position`, `goal_yaw` | free |
| `agent.*`, `floor.color`, `camera.angle_delta`, `cube.size` | inherited | **recompiles the MJCF every reset** |

"Free" means applied to the already-compiled `MjModel`. `cube.size` genuinely
needs the recompile (mass and inertia depend on it); the others are inherited
from `CubeEnv`. Selecting only free axes keeps resets fast:

```python
world.reset(options={'variation': ['light.position', 'digit.value', ...]})
world.reset(options={'variation': ['all']})   # includes the recompiling axes
```

> Keep `num_digits` / `num_bg_materials` / `add_backdrop` **identical between
> collection and evaluation.** Datasets store pool *indices*, so a differently
> sized pool resolves them to different materials.

The digit decals exist to be probed: each step records
`privileged/digit_{i}_value`, `privileged/digit_{i}_pos`,
`privileged/digit_{i}_size`, `privileged/digit_{i}_visible` and
`privileged/digit_count`, plus the `variation_digit_*` columns — so you can fit
a linear probe on the frozen LeWM embedding and ask whether digit identity and
position survived encoding.

The decals are genuinely in the image, not just in the labels. Rendering each
scene twice — once as sampled, once with the decal parked under the floor —
changes a median of ~510 pixels of the 224×224 frame (range 170–820 over 12
sampled scenes, i.e. 0.3–1.6% of the frame). `digit_bounds` is calibrated so a
decal at any sampled size, yaw and `camera.angle_delta` stays fully inside the
recorded frame; see the argument's docstring before widening it.

> **A cube can still sit on top of a decal.** The separation predicates cover
> cube–cube and digit–digit overlap, but not cube–digit: the digit sampling box
> (`x∈[0.32, 0.53]`, `y∈[-0.21, 0.21]`) is nested inside the cube box
> (`x∈[0.3, 0.55]`, `y∈[-0.3, 0.3]`). A partly occluded decal still reports its
> value in full through `privileged/digit_0_value`, so expect a small amount of
> label noise on the probe. `privileged/digit_0_pos` lets you filter those
> steps out after the fact.

---

## 5. Dataset creation

### 5.1 Running it

```bash
cd stable-worldmodel

# Quick check first — 4 episodes, ~8 s
python scripts/data/collect_cube_quadruple_dr.py \
    num_traj=4 world.num_envs=2 world.max_episode_steps=25 \
    hydra.sweep.dir=/tmp/hydra

# Full run, parallel — 5000 episodes across 4 processes, then merged
python scripts/data/collect_cube_quadruple_dr_sharded.py --shards 4

# Full run, single process (~2.8x slower)
python scripts/data/collect_cube_quadruple_dr.py \
    num_traj=5000 hydra.sweep.dir=/tmp/hydra
```

Writes to `$STABLEWM_HOME/datasets/ogbench/cube_quadruple_dr_expert.lance`.

Knobs (`scripts/data/config/ogb_cube_quadruple_dr.yaml`):

| Key | Default | Notes |
|---|---|---|
| `num_traj` | 5000 | Episodes. Upstream uses 5000 for cube-quadruple. Split across shards, not per shard. |
| `world.num_envs` | 10 | Envs stepped per batch. Does **not** parallelize — see §5.2. Drives peak memory; lower it if RAM is tight. |
| `world.max_episode_steps` | 400 | Chained pick-and-places per episode; upstream's play recipe uses 1000. |
| `policy_type` | `plan_oracle` | `CubePlanOracle`, OGBench's `-play` oracle. |
| `p_stack` | `0.8` | Fixed P(stack), overriding the built-in per-`env_type` schedule (quadruple: `U(0.1, 0.5)`); a `[lo, hi]` pair samples uniformly per episode instead, `null` restores the schedule. Does **not** make a completed tower persist — see below. |
| `options.variation` | `[all]` | Which axes to resample per episode. |
| `seed` | 3072 | Base episode seed. The run is reproducible from it — see §5.4. |
| `shard` / `num_shards` | 0 / 1 | Slice of the budget this process collects. Set by the sharded driver. |
| `env.*` | — | Forwarded to the env; must match the eval config. |

> **`p_stack` biases toward stacking, it doesn't hold a finished tower.** Once
> every cube is stacked, no cube is left exposed to stack the next target
> onto, so `set_new_target` unconditionally sends it to a random floor spot —
> regardless of `p_stack`. Expect episodes to show towers forming and being
> torn down repeatedly, not a tower held until the episode ends.

> **The shipped config is deliberately not upstream's play recipe.** `p_stack:
> 0.8` over 400-step episodes trades recipe parity for stacking density. To
> reproduce `cube-quadruple-play-v0` instead — the stock `U(0.1, 0.5)` schedule
> over 1000-step episodes, ~11 chained pick-and-places, for general-purpose
> representation learning — run with
> `p_stack=null world.max_episode_steps=1000`. Note that 1000-step episodes
> also multiply the storage and time estimates in §5.2 by 2.5.

> Use `policy_type=plan_oracle`. `markov_oracle` hard-codes 0.02/0.04 alignment
> thresholds and 0.16–0.18 m lift heights, which break under `cube.size`
> randomization.

Hydra runs these configs in MULTIRUN mode and drops a `multirun/` directory in
the working directory; `hydra.sweep.dir=/tmp/hydra` keeps the repo clean. The
sharded driver passes this for you.

### 5.2 Cost, and why sharding

`EnvPool` steps its envs in a plain Python loop
(`stable_worldmodel/world/env_pool.py`), so collection is single-threaded:
`world.num_envs` controls batching and memory, not parallelism. One process
pins one core and leaves the rest of the machine idle. Use *processes* to fill
the CPU.

Measured on an M1 Max (10 cores, 32 GB, `MUJOCO_GL=glfw`) at the shipped
config — 400-step episodes, 224×224, `variation: [all]`:

| | Per episode | 5000 episodes |
|---|---|---|
| Single process | 3.47 s (~115 env-steps/s) | **≈4 h 50 min** |
| `--shards 4` | 1.22 s (~330 env-steps/s) | **≈1 h 40 min** + ~25 min merge |

Throughput is flat across episodes — no writer degradation over a long run.
Returns diminish past 4 shards (the render path contends) and each process
peaks near **6.7 GB** RSS at `world.num_envs=10`, so on a 16 GB machine drop to
`world.num_envs=5` rather than raising the shard count. On a base M1 (8 cores,
16 GB) budget ~2.5–3 h at 2 shards.

**Storage:** measured **14.7 KB per step**, i.e. 5.9 MB per 400-step episode
and **≈30 GB** for the full 5000-episode run. (An earlier 100 GB estimate here
assumed `num_digits=5` and 1000-step episodes.)

### 5.3 How sharding works

`collect_cube_quadruple_dr_sharded.py` launches N copies of the collect script
with `shard=i num_shards=N`. Each takes an even slice of `num_traj` and a
**disjoint block of episode seeds**, and writes its own
`cube_quadruple_dr_expert_shard{i}.lance`. When all shards exit 0 the driver
runs `swm merge`, which renumbers episodes into one contiguous range, then
deletes the shards (`--keep-shards` to keep them).

The disjoint seed block is the part that matters. `World` seeds episodes from a
contiguous integer run starting at the base seed, so two shards drawing from
overlapping blocks would silently record the *same* episodes twice. The stride
is the full `num_traj + num_envs` span, which stays disjoint whatever the split.

Shards never write to a shared table — the merge is a separate sequential pass —
so there is no concurrent-writer hazard. Overrides after `--` reach every shard:

```bash
python scripts/data/collect_cube_quadruple_dr_sharded.py --shards 4 \
    -- num_traj=2000 world.num_envs=5 world.max_episode_steps=500
```

If a shard fails, the driver skips the merge and leaves the completed shards on
disk, so you re-run only the failed indices:

```bash
python scripts/data/collect_cube_quadruple_dr.py shard=2 num_shards=4
swm merge ogbench/cube_quadruple_dr_expert_shard{0,1,2,3}.lance \
    -o ogbench/cube_quadruple_dr_expert --overwrite
```

> `swm merge` appends the format suffix itself, so pass `-o name`, **not**
> `-o name.lance` — the latter writes `name.lance.lance`.

The single-process path also appends rather than overwrites, so an interrupted
run keeps every finished episode; re-run with the remaining `num_traj` to top it
up (at a different `seed`, or you will re-record the same episodes).

### 5.4 Reproducibility

A run is reproducible from `seed` — same `seed`, `num_traj`, `num_shards` and
`world.num_envs` reproduce the dataset bit-for-bit, verified across two
independent 4-shard runs (identical `qpos`, `pixels` and every `variation_*`
column).

That took three fixes, because nothing about it held before:

- `CubeEnv.reset` hardcoded `seed=None` into `reset_variation_space` and never
  forwarded `seed` to `super().reset()` at all, so **every** domain
  randomization axis and the arm/target sampling redrew from OS entropy on
  every reset — `reset(seed=n)` twice gave two different scenes. Both now
  forward the caller's seed, matching what `PushTEnv.reset` already did. This
  touches shared code: `swm/OGBCube-v0` inherits the fix (1211 tests pass).
  `maze_env.py` and `scene_env.py` still have the original bug.
- `ExpertPolicy` was constructed without `seed=`, leaving its action-noise and
  `p_stack` draws on OS entropy.
- OGBench's own `CubePlanOracle` draws waypoint and timing jitter from the
  **global** numpy RNG with no seed hook, so the collect script now calls
  `np.random.seed(base_seed)`. This pins the process, not each episode
  independently — the global stream is consumed across episodes, which is why
  the shard layout has to match to reproduce a run.

### 5.5 On a Linux CPU node (RunPod)

Everything runs there — no GPU is needed for collection, and the sharded driver
scales better on a many-vCPU node than on a laptop. Two things change.

**Rendering moves to the CPU.** `osmesa` is a software rasterizer, and
rendering is not a rounding error in the step budget: measured over 2010 steps
on the Mac, `render_time` averages **5.5 ms of a 10.1 ms step — 55%**. Roughly
half the work therefore moves from the GPU to a core that is also running the
physics. Expect collection to be **~2–5× slower per step** than the numbers in
§5.2; the physics half additionally scales with single-core speed, where a
typical EPYC/Xeon vCPU is somewhat behind an M1 Max.

That range is a projection, not a measurement — I could not test on Ubuntu.
**Calibrate it on the node in about a minute** before committing to a full run:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
time python scripts/data/collect_cube_quadruple_dr.py \
    num_traj=10 dataset_name=ogbench/_bench.lance hydra.sweep.dir=/tmp/hydra
swm inspect ogbench/_bench.lance          # confirm 10 episodes x 401 steps
```

Subtract ~8 s of import overhead, divide by 10, and you have s/episode. Multiply
by `num_traj / shards` for the wall-clock of the real run.

**Sharding is how you get it back.** Set `--shards` to roughly the vCPU count,
but memory binds first: each process peaks near 6.7 GB at `world.num_envs=10`,
so on a 16-vCPU / 32 GB pod run

```bash
python scripts/data/collect_cube_quadruple_dr_sharded.py --shards 12 \
    -- world.num_envs=4
```

`world.num_envs` only sets batching, so lowering it costs nothing but a little
per-process overhead — and it is the knob that keeps N processes inside RAM.
Budget `shards × ~3 GB` at `num_envs=4` and leave headroom; an OOM-killed shard
takes its episodes with it (the driver then skips the merge and tells you which
indices to re-run).

Also worth knowing on a pod:

- **Disk.** The container filesystem is usually 10–20 GB, well under the ~30 GB
  dataset. Put `STABLEWM_HOME` on the mounted volume (`/workspace`), and check
  `df -h /workspace` first — shards transiently double the footprint until the
  merge finishes and they are deleted.
- **Python version.** 3.10–3.12 only (§2). Images built on 3.13 need a fresh
  venv; `python3.12 -m venv` after `apt-get install python3.12-venv`.
- **Detach long runs.** A 5000-episode collection outlives an SSH session — run
  it under `nohup`/`tmux`. The driver prints per-shard progress to stdout.
- **No `sudo`, no display.** Covered in §2 and §3 respectively.

### 5.6 Sampling sanity

Checked empirically, at the shipped config:

- **Rejection sampling has headroom.** `cube.start_position` is redrawn until
  no pair of cubes interpenetrates; acceptance is 0.81 at the smallest cube
  size and 0.45 at the largest, against a `max_tries` of 1000 — so the
  `RuntimeError: predicate not satisfied` path is unreachable in practice
  (P < 1e-260 at the worst size). 300 consecutive `variation: [all]` resets
  raised nothing.
- **Digit separation is inactive at `num_digits=1`** — `_pairwise_clear`
  returns True below two items — and `digit.count` is pinned to
  `Discrete(1, start=1)` so the only decal is never hidden by chance.
- **Every episode is distinct.** Across a 4-shard run, all sampled start
  layouts were unique — the seed blocks do not collide.

Inspect the result:

```bash
swm datasets                                              # list
swm inspect ogbench/cube_quadruple_dr_expert.lance        # columns, shapes, sizes
swm preview ogbench/cube_quadruple_dr_expert.lance        # sample frames
```

Sanity-check that these columns exist — they are what evaluation replays:
`qpos`, `qvel`, `privileged/block_{i}_pos`, `privileged/digit_{i}_value`, and
the `variation_*` columns (note: the Lance writer flattens `variation.a.b` to
`variation_a_b`).

---

## 6. Training

```bash
python scripts/train/lewm.py data=ogb_cube_quadruple_dr
```

Useful overrides:

```bash
python scripts/train/lewm.py data=ogb_cube_quadruple_dr \
    output_model_name=lewm_q4_dr \
    trainer.max_epochs=100 \
    loader.batch_size=128 \
    wm.history_size=3 wm.num_preds=1 \
    loss.sigreg.weight=0.09
```

Checkpoints land in `$STABLEWM_HOME/checkpoints/<output_model_name>/` as
`weights_epoch_N.pt` plus a `config.json` (the resolved model config, used to
re-instantiate at eval time).

Watch two numbers: `pred_loss` should fall, and `sigreg_loss` should stay
bounded — SIGReg is the only thing preventing representational collapse (there
is no EMA target network).

`swm checkpoints` lists what you have.

### Sensorimotor world model (SMWM)

`scripts/train/smwm.py` trains the sensorimotor variant: the same architecture as
LeWM plus an MLP **inverse dynamics model** (`stable_worldmodel.wm.smwm.module.InverseModel`,
`(z_t, z_{t+1}) -> a_t`) held *inside* the world model, and the objective

```
L = pred_loss + loss.inverse.weight * inv_loss + loss.sigreg.weight * sigreg_loss
```

`scripts/train/config/smwm.yaml` keeps the LeWM architecture verbatim (`encoder_scale=small`,
`embed_dim=384`, `wm.history_size=3`) and differs from `lewm.yaml` in exactly three places:

| Key | `lewm.yaml` | `smwm.yaml` | Why |
|---|---|---|---|
| `loss.inverse.weight` | — | `1.0` | the new IDM term |
| `loss.sigreg.weight` | `0.09` | `0.0` | **the IDM replaces SIGReg** as the anti-collapse term |
| `loader.batch_size` | `128` | `256` | matches the sensorimotor recipe |
| `optimizer.lr` | `5e-5` | `1e-4` | linearly scaled for the 2× batch |

```bash
python scripts/train/smwm.py data=ogb_cube_quadruple_dr \
    output_model_name=smwm_q4_dr \
    trainer.max_epochs=100 \
    loss.inverse.weight=1.0 \
    inverse.hidden_dim=256
```

New knobs: `loss.inverse.weight` (λ_inv) and `inverse.hidden_dim`. The IDM is supervised on
every consecutive latent pair in the window (3 pairs at
`num_steps = num_preds + history_size = 4`) and regresses the **normalized** flattened
action block, so its output width is `frameskip × action_dim = 25` — wired automatically in
`run()` from `dataset.get_dim('action')`.

**Watch `inv_loss` for collapse.** With `loss.sigreg.weight=0` the IDM is the *only* thing
keeping the latent from collapsing — `pred_loss` alone is minimised at zero by a constant
encoder, and there is no EMA target network. A run where `pred_loss` dives while `inv_loss`
plateaus high is collapsing. Two escape hatches: raise `loss.inverse.weight`, or put SIGReg
back with `loss.sigreg.weight=0.09` (that value was tuned at batch 128, and the SIGReg
statistic scales with batch size, so re-tune it if you also keep `batch_size=256`).

To reproduce a strict LeWM A/B instead, revert all four keys at once:

```bash
python scripts/train/smwm.py data=ogb_cube_quadruple_dr \
    loss.inverse.weight=0.0 loss.sigreg.weight=0.09 \
    loader.batch_size=128 optimizer.lr=5e-5
```

That is verified to reproduce `lewm.py`'s loss bit-for-bit.

> **`loader.batch_size=256` is not free.** With `history_size=3` and `num_preds=1` each batch
> pushes `256 × 4 = 1024` images at 224² through the ViT, double the LeWM baseline. If that
> OOMs, use `loader.batch_size=128 optimizer.lr=5e-5`, or keep the effective batch with
> `loader.batch_size=128 +trainer.accumulate_grad_batches=2` (leaving the LR at `1e-4`).
> The LR *schedule* needs no manual fix either way — `total_steps` and the 1% warmup both
> derive from `len(train)`, so they track the batch size automatically.

Evaluation needs no SMWM-specific flags: `load_pretrained` rebuilds the model — inverse
model included — from the checkpoint's `config.json`, so §7 applies verbatim with
`policy=smwm_q4_dr/weights_epoch_N.pt`.

**RunPod notebooks.** `scripts/notebooks/{train,plan,probe}_smwm_ogbcubedr.ipynb` are the
SMWM counterparts of the three `*_lewm_ogbcubedr.ipynb` notebooks, same flow and same
`/workspace` volume layout. Two of them take an optional pointer at the LeWM baseline so the
comparison lands in one table: `BASELINE_RESULTS_DIR` in the planning notebook, and
`BASELINE_MODEL_NAME` / `BASELINE_EPOCH` in the probing notebook (§11 there refits the same
probes on the LeWM checkpoint over identical features and episode splits). The probing code
in `scripts/probe/` is model-agnostic — it instantiates whatever `config.json` describes and
calls `model.encode` — so it needed no changes for SMWM.

> **The last action of every episode is `NaN`.** `World.collect` rotates the
> action column (`ep['action'].append(ep['action'].pop(0))`), which moves the
> reset frame's placeholder action to the end. The window sampler does not
> exclude it, so roughly one training window per episode — ~0.26% of samples at
> 400-step episodes, but enough to `NaN` a whole batch's gradient — contains it.
> This is generic SWM behaviour, not specific to this env. Guard the loss, or
> drop the affected windows, before a long training run.

> `data/ogb_cube_quadruple_dr.yaml` resolves `keys_to_merge: {proprio: proprio}`
> as a regex over the **loaded** keys, so the seven `proprio/*` source columns
> are listed explicitly in `keys_to_load`. Dropping them makes the merge
> concatenate an empty list and raise `need at least one array to concatenate`
> before the first step.

---

## 7. Evaluation

Evaluation replays a recorded expert episode: it restores the simulator to a
dataset step (including that episode's sampled randomization), then hands the
planner a **chain of oracle subgoals** spaced `goal_offset_steps` apart. Each
individual plan targets one fixed goal; success is measured against the
**final** subgoal, so it means the whole replayed segment was completed.

```bash
# Random baseline
python scripts/plan/eval_wm.py --config-name=cube_quadruple_dr policy=random

# Trained model
python scripts/plan/eval_wm.py --config-name=cube_quadruple_dr \
    policy=lewm_q4_dr/weights_epoch_100.pt

# Sensorimotor variant - identical command, only the checkpoint differs
python scripts/plan/eval_wm.py --config-name=cube_quadruple_dr \
    policy=smwm_q4_dr/weights_epoch_100.pt
```

Key settings (`scripts/plan/config/cube_quadruple_dr.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `eval.goal_offset_steps` | 25 | Env steps between consecutive subgoals. |
| `eval.num_subgoals` | 8 | Length of the chain. `1` = single fixed goal. |
| `eval.subgoal_budget` | 40 | Steps allowed per subgoal. |
| `eval.subgoal_advance` | `both` | `budget` / `reached` / `both`. |
| `eval.subgoal_tol` | 0.04 | Reach tolerance, metres. |
| `eval.eval_budget` | 320 | Total env steps. Usually `num_subgoals × subgoal_budget`. |
| `eval.num_eval` | 50 | Episodes **and** parallel envs — 50 MuJoCo envs at once. |
| `plan_config.horizon` | 5 | Planning horizon in action blocks. |
| `plan_config.receding_horizon` | 1 | `1` = replan every block; `== horizon` = open loop. |
| `plan_config.action_block` | 5 | Frameskip; must match the training `frameskip`. |
| `plan_config.history_len` | 3 | Context frames; must match `wm.history_size`. |

`eval_wm.py` asserts `horizon × action_block <= eval_budget`.

**Validate before trusting the chain.** `num_subgoals=1` must reproduce the
old single-goal numbers:

```bash
python scripts/plan/eval_wm.py --config-name=cube_quadruple_dr \
    policy=lewm_q4_dr/weights_epoch_100.pt eval.num_subgoals=1 eval.eval_budget=50
```

Then sweep subgoal density at a fixed total budget (hydra multirun):

```bash
python scripts/plan/eval_wm.py -m --config-name=cube_quadruple_dr \
    policy=lewm_q4_dr/weights_epoch_100.pt \
    eval.num_subgoals=1,2,4,8 eval.eval_budget=320
```

### Outputs

- **Metrics** — printed and appended to `<results_dir>/ogb_cube_quadruple_dr_results.txt`:
  `success_rate`, `episode_successes`, `subgoal_index` (how far each env got
  along the chain) and `subgoals_reached` (how many it actually reached rather
  than timed out on).
- **Videos** — one `env_{i}.mp4` per env, three labelled panels:
  `agent | dataset | goal`. The goal panel advances as the chain does.
- `results_dir` is the checkpoint's directory for a real policy; for
  `policy=random` it is `scripts/plan/` — pass `hydra.run.dir=` and clean up
  after, or you will leave `.mp4`s in the repo.

---

## 8. Testing

```bash
cd stable-worldmodel
pytest tests/ -q                              # full suite: 1211 passed, 32 skipped
pytest tests/envs/test_ogbench_cube_dr.py -q  # DR env unit tests
pytest tests/test_subgoal_eval.py -q          # subgoal scheduler
pytest tests/envs/test_ogbench_cube_dr_integration.py -q   # collect -> eval
```

Lint and format. The canonical path is pre-commit, which only touches staged
files:

```bash
pip install pre-commit
pre-commit run                 # staged files
pre-commit run --all-files     # whole repo
```

Or scope ruff to what you changed — do **not** run `ruff format` over the whole
tree, it picks up notebooks that pre-commit deliberately excludes and reformats
a handful of pre-existing files:

```bash
pip install ruff
ruff format <your files>
ruff check --select E4,E7,E9,F <your files>   # the rule set pre-commit enforces
```

> `tests/envs/test_fetch.py` fails when run **on its own**
> (`NamespaceNotFound: Namespace swm not found`) because it never imports
> `stable_worldmodel`, so nothing registers the `swm/*` ids. It passes as part
> of the full suite. Pre-existing, unrelated to this project.

### LeWM parity (optional)

`scripts/train/lewm.py::lejepa_forward` is equivalent to `le-wm/train.py`
(prediction MSE + SIGReg at weight 0.09, no EMA target). To confirm parity,
train both on `ogbench/cube_single_expert` and compare loss curves and CEM
success rate — but run the evaluation through **SWM's** `scripts/plan/eval_wm.py`.
`le-wm/eval.py` targets a removed API (`stable_worldmodel.solver.CEMSolver` with
a `model=` argument; current SWM is `stable_worldmodel.planning.solver.CEMSolver`
with `cost=` a `ShootingCostEvaluator`).

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error: command 'bazel' failed` while installing | Python 3.13/3.14. Use 3.10–3.12. |
| `ERROR: Failed building wheel for box2d-py` | `[all]` extra needs `swig`. Install it, or use the scoped install in §3. |
| `RuntimeError: invalid value for environment variable MUJOCO_GL: egl` | On macOS. Use `MUJOCO_GL=glfw`. |
| `MUJOCO_GL must be one of [...]: got 'cgl'` | dm_control rejects `cgl`. Use `glfw`. |
| Renderer fails / blank frames on a cluster | Headless node with `glfw`. Use `egl`, or `osmesa` (+ `PYOPENGL_PLATFORM=osmesa`). |
| `ModuleNotFoundError: ogbench` | The `env` extra was skipped. `pip install ogbench`. |
| `ModuleNotFoundError: cv2` / `pygame` / `shapely` | Same — install the packages listed in §3. |
| `Could not find a backend to open ... .mp4` | `pip install 'imageio[ffmpeg]'` (part of the `format` extra). |
| `NamespaceNotFound: Namespace swm` | Running `test_fetch.py` alone; run the whole suite. |
| Every reset is slow | A recompiling axis (`cube.size`, `floor.color`, `agent.color`, `camera.angle_delta`) is enabled. See §4. |
| Restored eval scenes look wrong | `num_digits` / `num_bg_materials` / `add_backdrop` differ between collection and eval. |
| Eval videos appear in `scripts/plan/` | `policy=random` writes next to the script. Pass `hydra.run.dir=`. |
| `multirun/` directories in the repo | Hydra MULTIRUN mode. Pass `hydra.sweep.dir=/tmp/hydra`. |
| `Could not override 'shard'` | Collect config predates the shard keys. They must exist in `ogb_cube_quadruple_dr.yaml` (hydra struct mode), or use `+shard=i`. |
| `need at least one array to concatenate` at train start | `keys_to_merge` source has no match in `keys_to_load`. See §6. |
| `NaN` loss after an epoch or two | The rotated reset-frame action. See §6. |
| Collection uses one core out of N | Expected — `EnvPool` is sequential. Use `collect_cube_quadruple_dr_sharded.py`. |
| Two collection runs with the same `seed` differ | Shard layout must match too (`num_shards`, `world.num_envs`). See §5.4. |
| `name.lance.lance` after a merge | `swm merge -o` appends the suffix. Pass `-o name`. |
| `invalid value for environment variable MUJOCO_GL: egl` on a CPU pod | `eval_wm.py` defaults to `egl`. Export `MUJOCO_GL=osmesa` + `PYOPENGL_PLATFORM=osmesa`. |
| `apt-get install` hangs in a container | tzdata prompt. Prefix `DEBIAN_FRONTEND=noninteractive`. See §2. |
| Unable to locate package `libgl1-mesa-glx` | Renamed after Ubuntu 22.04. Use the §2 list (`libgl1` + `libglx-mesa0`). |
| A shard dies with no traceback | OOM killer. Lower `world.num_envs` or `--shards`. See §5.5. |
| Dataset gone after a pod restart | `STABLEWM_HOME` was on the container overlay. Point it at `/workspace`. |

---

## 10. On-disk layout

```
$STABLEWM_HOME/                      # default ~/.stable_worldmodel
├── datasets/
│   └── ogbench/
│       ├── cube_quadruple_dr_expert.lance
│       └── cube_quadruple_dr_expert_shard{0..N}.lance   # transient, see §5.3
└── checkpoints/
    └── <output_model_name>/
        ├── config.json
        ├── weights_epoch_1.pt ...
        └── ogb_cube_quadruple_dr_results.txt
```

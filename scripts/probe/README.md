# Probing a trained world model on OGBCubeDR

Fits classical **linear** and **MLP** read-outs on ``emb`` — the frozen,
single-frame representation a trained LeWM checkpoint produces and the
*only* thing the predictor conditions on — and asks, target by target, what
survived encoding. `Run.md` §4 describes the digit decal as existing for
this purpose; this covers that and the rest of the task-relevant state.

`scripts/notebooks/probe_lewm_ogbcubedr.ipynb` is the intended interface —
it handles dataset download, checkpoint selection, the tables and the plots.
This directory is the machinery behind it, also runnable headless:

```bash
python scripts/probe/run_probing.py \
    --checkpoint lewm_q4_dr/weights_epoch_11.pt \
    --out $STABLEWM_HOME/probing/lewm_q4_dr
```

## Files

| File | Role |
|---|---|
| `targets.py` | The label registry: task-relevant targets in two groups (`state`, `nuisance`), each with the Lance columns it reads and a reducer. Read the module docstring — it explains why every cube target is permutation-invariant. |
| `features.py` | Frozen-feature extraction. Encodes each sampled frame once and caches `emb` plus every label to one `.npz`. |
| `fit.py` | The read-outs: closed-form ridge, logistic regression, MLP, plus metrics and the constant-predictor baseline. |
| `run_probing.py` | CLI that chains the two and writes `results.json` / `results.csv`. |

Read-out heads themselves live in the library, at
`stable_worldmodel/wm/probes.py` (`LinearProbe`, `MLPProbe`), next to the
existing `attach_probe` / `load_probe` helpers.

## What makes it a clean experiment

- **The encoder runs once.** Every probe and every target reads the same
  cached feature matrix, so differences between them cannot come from the
  encoder.
- **One feature, the one that matters.** `LeWM.encode` reads the ViT's CLS
  token and passes it through `model.projector` to produce `emb`; `predict`
  conditions on exactly that vector and nothing else. Probing anything other
  than `emb` (a pre-projector token, a pixel baseline, a different pooling)
  would be measuring a representation the model does not actually use.
- **Splits are by episode, never by frame.** Frames of a 400-step episode
  are near-duplicates; a frame-level split leaks the test set into the train
  set. (Training itself uses a clip-level `random_split` — fine for fitting
  a world model, wrong for measuring one.)
- **Three rungs per target** — constant-predictor `baseline` → `linear` →
  `mlp` — so a score is read as a difference, never in isolation. The
  `mlp − linear` gap is information present but not linearly decodable.
- **Linear regression probes are solved in closed form** (ridge, penalty
  chosen on validation). An SGD-fitted linear probe conflates "not linearly
  decodable" with "the optimizer did not converge"; a probing result must not
  depend on that.
- **Model selection reads validation only.** Ridge penalty, weight decay and
  early stopping all use the val split; test is touched once.

## What is probed

Only quantities the task actually needs, one target per physical quantity —
no redundant reductions and nothing the frame does not contain:

- **Arm**: `effector_pos`, `effector_yaw`, `gripper_opening`,
  `gripper_contact`. Joint angles are not probed — they are redundant with
  the end-effector pose for planning purposes.
- **Cubes**: `cube_pos_sorted` (all four positions) and `cube_z_sorted` (all
  four heights), both permutation-invariant since cube colour, and hence
  cube identity, is redrawn every episode. The centroid and max-height
  reductions are dropped as redundant with these.
- **Digit decal**: `digit_value` only — the decal's floor position/size are
  not planning-relevant and are not probed.
- **Nuisance** (kept, not dropped): `floor_material`, `wall_material`,
  `floor_rgb`, `light_pos` — domain-randomization axes that are visible but
  irrelevant to the task. Whether the representation keeps them is a
  genuine question, not a failure either way.

## One thing that will bite you

**Do not under-sample episodes.** Every episode re-draws lighting, camera
angle, cube colours and floor/wall materials, so episode-level appearance is
the dominant direction of variation in the features. At ~100 training
episodes a linear probe fits episode identity and *every* within-episode
target reads R² ≈ 0 on held-out episodes — measured, not hypothetical. The
notebook defaults to 1000 train episodes. Raise the episode count before
raising `--frames-per-episode`, which adds correlated samples rather than
independent ones.

**Five labels are constant within an episode** — every domain-randomization
axis (`digit_value`, `floor_material`, `wall_material`, `floor_rgb`,
`light_pos`). Their effective training size is the *episode* count, not the
frame count; the results carry this as `n_train_effective`. They are also
the targets for which an episode-level split is not merely good practice but
the only split that means anything.

To tell "not in the representation" apart from "in the representation but
entangled with episode appearance", refit on a deliberately leaky
frame-level split — same episodes on both sides. Section 11.4 of the
notebook does this; the leaky number is inflated by construction and must
never be quoted on its own.

## Outputs

Per run directory:

- `features_<tag>.npz` — the feature cache. Re-runs reuse it; delete to
  force a re-encode.
- `results.json` — rows, the resolved args, the target definitions and the
  cache manifests.
- `results.csv` — the same rows, flat.
- `probes/<tag>/` — the fitted read-outs, with `--save-probes`. They carry
  their own feature- and target-standardization buffers, so
  `probe.predict(raw_features)` returns physical units.

## `shortcut_ceiling.py` — the GCIDM pre-test

A second, **model-free** entry point in this directory. It answers a
different question from the probes above: not "what did this encoder keep?"
but "is the objective we are about to train even capable of forcing the
encoder to keep it?"

GCIDM regresses the action-block plan joining an observation to a goal
observation. Because the OGBench oracle's action is literally an effector
delta (`plan[i] - proprio/effector_pos`, see
`ogbench/manipspace/oracles/plan/plan_oracle.py`), the *sum* of that plan is
computable from gripper features alone — no cube state needed. Only the path
curvature (approach above the cube, descend, grasp, lift, clearance
waypoint) and the grasp timing genuinely require knowing where a cube is.

The script brackets that with two ceilings, fit from **ground-truth state**
instead of a learned encoder — so it needs no checkpoint and runs on CPU:

| Feature set | Columns | Meaning |
|---|---|---|
| `proprio` | every `proprio/*` (19 dims: joint pos/vel, effector pose, gripper opening/vel/contact) | **shortcut ceiling** — best R² using the robot only |
| `full_state` | `observation` (55 dims; = proprio + the four block poses) | **information ceiling** — best R² with perfect state |

The direction is inverted relative to the probes above: here the *features*
are state and the *target* is the action plan. Everything else is shared —
`fit.py`'s baseline / linear-ridge / mlp rungs, the episode-level split,
train-statistic whitening, one touch of the test split.

```bash
python scripts/probe/shortcut_ceiling.py --out runs/shortcut_ceiling_h5
```

**Reading it.** The gap `R²(full_state) − R²(proprio)` is the entire headroom
cube state can buy at this horizon.

- **gap < ~0.05** → NO-GO. No encoder can be pushed to represent cube state
  by this objective at this horizon, so training would be uninformative.
- **gap large** → GO. Record both numbers: labels are z-scored, so once
  GCIDM trains, `train/policy_loss` lands on the same axis via
  `R² ≈ 1 − policy_loss`, and its position between the two ceilings says how
  much of the available cube information the encoder actually used.
- **gap negative** → INCONCLUSIVE, reported as such. `observation` is a
  strict superset of the `proprio` columns, so a negative gap is impossible
  in the data and means the wider feature set is underfit. Raise
  `--train-episodes` / `--windows-per-episode`.

`--goal-horizon` also drives the fallback design. Sweeping it shows where the
gap is widest, which is the offset a long-horizon shaping head should use:

```bash
for H in 5 10 20; do
  python scripts/probe/shortcut_ceiling.py --goal-horizon $H \
         --out runs/shortcut_ceiling_h$H
done
```

`--frameskip`, `--context-frames` and `--goal-horizon` must match
`data.dataset.frameskip`, `wm.history_size` and `wm.goal_horizon` in
`scripts/train/config/gcidm.yaml` for the numbers to be comparable to a
training run.

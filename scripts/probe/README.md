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

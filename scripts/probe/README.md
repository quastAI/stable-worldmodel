# Probing a trained world model on OGBCubeDR

Fits classical **linear** and **MLP** read-outs on the frozen representations
of a LeWM checkpoint and asks, target by target, what survived encoding.
`Run.md` §4 describes the digit decals as existing for this purpose; this
covers those and 19 other labels.

`scripts/notebooks/probe_lewm_ogbcubedr.ipynb` is the intended interface —
it handles dataset download, checkpoint selection, the controls, the tables
and the plots. This directory is the machinery behind it, also runnable
headless:

```bash
python scripts/probe/run_probing.py \
    --checkpoint lewm_q4_dr/weights_epoch_11.pt \
    --with-random-init \
    --out $STABLEWM_HOME/probing/lewm_q4_dr
```

## Files

| File | Role |
|---|---|
| `targets.py` | The label registry: 20 targets in three groups (`state`, `nuisance`, `control`), each with the Lance columns it reads and a reducer. Read the module docstring — it explains why every cube target is permutation-invariant and why each control is a control. |
| `features.py` | Frozen-feature extraction. Encodes each sampled window once, caches six feature variants plus every label to one `.npz`. |
| `fit.py` | The read-outs: closed-form ridge, logistic regression, MLP, plus metrics and the constant-predictor baseline. |
| `run_probing.py` | CLI that chains the two and writes `results.json` / `results.csv`. |

Read-out heads themselves live in the library, at
`stable_worldmodel/wm/probes.py` (`LinearProbe`, `MLPProbe`), next to the
existing `attach_probe` / `load_probe` helpers.

## What makes it a clean experiment

- **The encoder runs once.** Every probe, target and capacity rung reads the
  same cached feature matrix, so differences between them cannot come from
  the encoder.
- **Splits are by episode, never by frame.** Frames 5 env-steps apart in a
  400-step episode are near-duplicates; a frame-level split leaks the test
  set into the train set. (Training itself uses a clip-level `random_split` —
  fine for fitting a world model, wrong for measuring one.)
- **Three rungs per target** — constant-predictor `baseline` → `linear` →
  `mlp` — so a score is read as a difference, never in isolation. The
  `mlp − linear` gap is information present but not linearly decodable.
- **Linear regression probes are solved in closed form** (ridge, penalty
  chosen on validation). An SGD-fitted linear probe conflates "not linearly
  decodable" with "the optimizer did not converge"; a probing result must not
  depend on that.
- **Model selection reads validation only.** Ridge penalty, weight decay and
  early stopping all use the val split; test is touched once.
- **Two positive controls** that bound what a high score can mean: an
  untrained encoder of the same architecture (`--with-random-init`), and a
  16×16 average-pooled input frame (`pixels_lowres`).
- **Four negative controls** the frame does not contain, each failing for a
  different documented reason, so *which* one lights up is the diagnosis.
  See `targets.py`'s module docstring.

## Feature variants

One forward pass over a training-shaped window (`history_size` context frames
plus `num_preds` future frames, `frameskip` env steps apart) yields:

| variant | what it is | label read at |
|---|---|---|
| `backbone_cls` | ViT CLS token, **before** the projector | current frame |
| `emb` | projector output — what the LeWM loss and the predictor see | current frame |
| `emb_hist` | `emb` over all context frames; the predictor's conditioning | current frame |
| `pred_emb` | the model's **prediction** of the next latent | predicted frame |
| `emb_next_true` | the true latent of that frame — the ceiling for `pred_emb` | predicted frame |
| `pixels_lowres` | 16×16 average-pooled input frame (pixel control) | current frame |

## Two things that will bite you

**Do not under-sample episodes.** Every episode re-draws lighting, camera
angle, cube colours, floor/wall materials and the backdrop, so episode-level
appearance is the dominant direction of variation in the features. At ~100
training episodes a linear probe fits episode identity and *every*
within-episode target reads R² ≈ 0 on held-out episodes — measured, not
hypothetical. The notebook defaults to 1000 train episodes. Raise the episode
count before raising `--windows-per-episode`, which adds correlated samples
rather than independent ones.

**Seven labels are constant within an episode** — every domain-randomization
axis (`digit_value`, `digit_pos`, `digit_size`, `floor_material`,
`wall_material`, `floor_rgb`, `light_pos`). Their effective training size is
the *episode* count, not the window count; the results carry this as
`n_train_effective`. They are also the targets for which an episode-level
split is not merely good practice but the only split that means anything.

To tell "not in the representation" apart from "in the representation but
entangled with episode appearance", refit on a deliberately leaky
window-level split — same episodes on both sides. Section 11.4 of the
notebook does this; the leaky number is inflated by construction and must
never be quoted on its own.

## Outputs

Per run directory:

- `features_<tag>.npz` — the feature cache (~400 MB per configuration at the
  notebook's default scale). Re-runs reuse it; delete to force a re-encode.
- `results.json` — rows, the resolved args, the target definitions, the cache
  manifests, and the `pred_emb` vs `emb_next_true` fidelity check.
- `results.csv` — the same rows, flat.
- `probes/<tag>/` — the fitted read-outs, with `--save-probes`. They carry
  their own feature- and target-standardization buffers, so
  `probe.predict(raw_features)` returns physical units.

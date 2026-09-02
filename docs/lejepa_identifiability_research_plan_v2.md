# Identifiability and Planning with LeJEPA

**A staged experimental program from OGBench Cube to photorealistic observation**

*Research plan · experimentation phase · v2*

---

## Thesis

Linear, orthogonal identifiability of the world's latent variables is achievable when the encoder is trained on data we design rather than data we can collect. Sim lets us design it. The program measures, per environment: (i) whether full latent identification is attainable, (ii) what each assumption violation costs, (iii) how far a realistically-collectable dataset falls short, and (iv) whether identifiability quality predicts planning success.

## Two claims under test

- **C1 — Attainability.** With OU-designed encoder data, orthogonal linear recovery reaches ceiling in each environment.
- **C2 — Consequence.** Encoder-attributable planning success is a tight, monotone function of identifiability quality, across environments and across violations.

C2 is the point of the program. C1 is the precondition that makes C2 measurable.

---

## 0. What the theory does and does not give us

Everything downstream depends on reading the four theorems precisely, so they are restated here in the form the program actually uses.

**Thm 1 (forward).** In the Gaussian world — independent latents, stationary marginals, additive noise, `z ∼ N(0, Iₙ)`, OU transition `z' = ρz + √(1−ρ²)η` — any measurable `h` with `h(z) ∼ N(0, Iₙ)` satisfies `L(h) ≥ 2(1−ρ)n`, with equality **iff** `h(z) = Qz` for some **orthogonal** `Q ∈ O(n)`.

The recovery class is **the orthogonal group, not the permutation group.** A perfectly identified encoder produces a generic rotation of the latents. This single fact determines the metric suite (§2) and is the most common way this kind of program goes wrong.

**Thm 2 (converse).** Within the same world class, the Gaussian is the *unique* latent distribution for which every minimiser is linear. The mechanism is Sturm–Liouville: the first non-constant eigenfunction of the transition operator is *always monotonic*, for any latent distribution; linear identifiability requires it to be *affine*, and only the Gaussian delivers that.

This gives us a graded prediction rather than a binary one. Under non-Gaussianity we should expect recovery **up to a monotone reparametrisation**, not total failure. Appendix F sharpens it: if all latents are i.i.d. from the same distribution with the same transition structure, identifiability holds up to an orthogonal rotation composed with a *shared elementwise monotone* map. Only in the Gaussian case is that map the identity.

**Thm 3 (approximate).** With `ε = ‖Cov(h(z)) − Iₙ‖_F` and `δ = L(h) − 2(1−ρ)tr(Cov(h(z)))`, define `D = δ / (2ρ(1−ρ))`. Then `min_{Q∈O(n)} E‖h(z) − Qz‖² ≤ D + (ε + D)²`.

Two consequences the program leans on. First, `2ρ(1−ρ)` is the **spectral gap**, and it is the exchange rate that converts an optimisation residual into a recovery error. It peaks at `ρ = 0.5` and vanishes at both `ρ → 0` and `ρ → 1`. The cost of *every* violation is therefore a function of where we sit on the ρ axis. Second, both `ε` and `δ` are computable **without ground truth**, which is what makes the Env 5 deliverable possible.

**Thm 4 (planning).** For `h(z) = Qz` with `Q` orthogonal, and for any finite-horizon control problem whose stage and terminal costs are **`O(n)`-invariant in the state**, the latent-space value function and optimal action sequence are *identical* to the true-space ones.

The invariance condition is a hard precondition, not a formality. It is why goal-conditioned latent L2 is the right cost family and why OGBench is the right task suite. It is also why any cost we introduce that is not `O(n)`-invariant silently exits the theorem.

**What the theory does not give.** App. D.2 is explicit: Thm 4 concerns the *encoder alone*. The action-conditioned transition `p̂(ẑ' | ẑ, a)` must still be learned and is *not* proven identifiable; the authors flag it as ongoing work requiring a persistent-excitation condition. What Thm 4 guarantees is only that the encoder does not corrupt the learning problem for the transition. **Our stage-D design must therefore separate encoder error from predictor error, or C2 is unmeasurable** (§1, stage D).

### 0.1 Which assumptions bind, and where

This deserves its own statement because it determines the scope of half the violation set, and getting it wrong would cause us to over-constrain data collection everywhere above stage A.

**The latent-process assumptions are assumptions about the encoder's *training distribution*, not about the encoder.** Gaussianity, stationarity, additive noise, and isotropy are the conditions under which the LeJEPA optimum *is* `Qz`. They shape what the objective's minimiser looks like. Once training has converged and the encoder is frozen, `h = f ∘ g` is a fixed deterministic function, applied **per frame**. If `h(z) ≈ Qz` on the region of latent space that matters, that approximation is a property of the map and holds at every frame it is evaluated on, regardless of how those frames were sequenced, correlated, or sampled.

So the predictor dataset — and the planning-time state distribution — are free of V1–V4 entirely. Rollouts may be non-Gaussian, non-stationary, heteroscedastic, and arbitrarily anisotropic in their per-dimension autocorrelation. **Anisotropy is the cleanest case: `ρ_α` only exists as a property of positive pairs, and after freezing there are no positive pairs.** The eigenvalue-interleaving failure mode of V4 is a statement about which functions minimise the alignment loss; it has no inference-time analogue.

What *does* transfer is everything about the observation map and about coverage:

| | Violation | Binds at encoder training | Binds at inference / predictor data | Mechanism at inference |
|---|---|---|---|---|
| V1 | Non-Gaussian marginal | yes | **no** | — |
| V2 | Non-stationarity | yes | **no** | — |
| V3 | State-dependent noise | yes | **no** | — |
| V4 | Anisotropic transitions | yes | **no** | — |
| V9 | Latent dependence | yes | **no** | — |
| V5 | Support truncation | yes (marginal shape) | **yes** (different mechanism) | rollouts leaving trained support |
| V6 | Occlusion | yes | **yes** | injectivity fails frame-by-frame |
| V7a/b | Dimension misspecification | yes | fixed at freeze | property of the frozen map |
| V8 | Optimisation gap | yes | fixed at freeze | property of the frozen map |

Two residual caveats keep this from being a blank cheque.

**Support and reweighting.** Thm 1 is a population-level statement over the training distribution, and Thm 3 bounds an *expected* recovery error under that distribution. A frozen encoder is a fixed function, but its *approximation quality* is not uniform over latent space. Where rollouts visit regions the OU marginal covered sparsely — or not at all — local error can be much larger than the reported aggregate. This is the real content of the program's rule that recovery is always reported on the rollout distribution as well as the OU distribution, and it is why Env 1 deliberately allows interpenetration and table-clipping so that evaluation support is a *subset* of training support.

**Style outside the randomisation range.** Invariance was trained, not proved. Appearance conditions in the predictor data that fall outside the domain-randomisation range are not covered by the alignment loss and will leak into the embedding. This is a per-frame property and it does transfer.

**Consequence for the program.** V1–V4 and V9 are stage-A/B concerns and arm-C concerns only. Predictor and planner data should be collected for *their own* requirements — coverage of the action space, excitation, task-relevant state visitation — which are different requirements, not weaker ones. App. D.2's note that transition identifiability would need a persistent-excitation condition is the relevant constraint there, and it is orthogonal to everything in V1–V4.

---

## Standing design decisions

- **Encoder training is passive.** LeJEPA as published; actions never enter the encoder.
- **Encoder data is generated by direct state-setting** from an OU process in latent space, then rendered. Physics is never stepped for encoder data. Contacts and regime switches are predictor-side concerns, not encoder-side.
- **The encoder is frozen** before any predictor or planner is trained.
- **Content vs style is declared in writing** before data generation. Content is recovered (scored by the recovery metrics); style is discarded by the alignment loss (scored by invariance). Domain randomisation is the mechanism that defines style. Style must be **resampled within each positive pair**, not per pair — otherwise the alignment loss has no gradient with which to discard it.
- **ρ and λ are frozen program constants**, declared before Env 1 alongside the seed and severity budget. They are not free hyperparameters per environment. Rationale in §2.1.
- **Costs used in planning are restricted to the `O(n)`-invariant family** (§4.3). This is a precondition of Thm 4, not a convenience.
- **Metric implementations and the seed/severity budget are frozen before Env 1** and never changed. Changing them mid-program silently destroys cross-environment comparability.

---

## 1. The cycle

Run identically in every environment. An environment is complete when stage D is measured — not when stage D is good. A failure to reach ceiling in stage A is a result about that environment, not a reason to stall.

### A. Identify

- Enumerate the true latents. Split into content and style. Record the split, and record **per-latent type, range, and characteristic timescale**.
- Design the OU sampler over content latents. Fix parameters, feasibility handling, and the render path. **Set per-dimension ρ isotropically by default** (§2.1) — anisotropy is a violation to be swept, not an accident to be inherited.
- Record the marginal actually sampled, not the one intended. Audit it with the z-scored SIGReg statistic against a matched-i.i.d.-Gaussian floor (the paper's floor is ≈ 1.2 at their sample size; recompute ours).
- Train the encoder passively on the rendered OU dataset.
- **Exit:** normalised Procrustes recovery error at floor with a near-orthogonal fitted map, on both the OU distribution and the physics-rollout distribution.

### B. Violate

- Run the fixed violation set V1–V9 (§3), each as a scalar sweep at several severities and seeds.
- Record each environment's **induced baseline** for every violation before sweeping on top of it. Occlusion (V6) and support truncation (V5) will have non-zero induced baselines in most environments. V1–V4 and V9 have *zero* induced baseline in stage A by construction, since the OU sampler sets the latent process directly; they acquire induced baselines only in arms C1/C2 (§0.1).
- **Output:** one cost curve per violation, for this environment, expressed in spectral-gap units (§2.1).

### C. Constructed real-world arm

Two variants, because the paper makes a specific and testable practical recommendation about data collection.

- **C1 — naive.** Rebuild the encoder dataset under constraints a real setup would impose: no state resets; trajectories only from a behaviour or scripted-teleop policy; whatever coverage that yields; fixed camera; natural (non-OU) marginal.
- **C2 — exploration-designed.** Same physical constraints, but the collection policy is designed to **approximate an isotropic random walk** in the content latents, which is the paper's explicit recommendation for keeping self-supervised pretraining data inside the theory's regime. Frame stride is tuned so that empirical per-dimension ρ lands near the program's declared ρ.
- Everything else — architecture, objective, compute, schedule — held identical to stage A.
- **Output:** position on the same identifiability axes for both variants, and an attribution of each gap to specific B-curves.

**Control note.** In arm A we choose ρ; in arm C it is whatever the policy and frame rate give us. Some of the naive A→C1 gap will be a ρ mismatch rather than a marginal-shape problem. C2's stride matching is the control that separates them. The paper's own Reacher result — identifiability peaks at *intermediate* stride, not monotonically in stride — means this cannot be skipped.

### D. Plan

Freeze the encoder. Train the predictor. Plan in latent space with an `O(n)`-invariant cost.

**Arms:**

| Arm | State | Dynamics | Role |
|---|---|---|---|
| **O** | ground-truth | ground-truth | Absolute oracle. Denominator for headline SR. |
| **P** | ground-truth | **learned predictor** | Isolates predictor quality. **Denominator for the C2 axis.** |
| **A** | OU-identified encoder | learned predictor | The designed-data arm. |
| **C1 / C2** | real-world-constrained encoder | learned predictor | The collectable-data arms. |
| **R** | random frozen encoder | learned predictor | Floor. |

Arm P is the addition that makes C2 well-posed. `SR / SR(O)` conflates encoder error with predictor error; in a contact-rich or long-horizon environment a low ratio would be unattributable. **`SR / SR(P)` is the encoder-attributable quantity and is the y-axis of the global scatter.** Report both.

The predictor architecture, capacity, and training budget are held fixed across all arms within an environment, and the same predictor family is used across environments wherever the action space permits.

Repeat for each violation severity from B. These runs populate the global scatter.

---

## 2. Metrics, fixed across the program

### 2.1 Program constants

| Constant | Value | Rationale |
|---|---|---|
| `ρ` (encoder OU, isotropic) | declared once, in `[0.8, 0.95]` | Grid search (Fig. 6) puts best identifiability at `λ ∈ [10⁻³, 10⁻²]`, `ρ ∈ [0.8, 0.95]`. Avoid `ρ = 0.99`: near-identical positive pairs make the invariance loss trivially small and let SIGReg dominate, collapsing the representation at moderate λ. |
| `λ` (SIGReg weight) | declared once, in `[10⁻³, 10⁻²]` | Same source. Below `10⁻⁴` Gaussianity regularisation is insufficient regardless of ρ; at `λ = 0.5` the representation collapses entirely. |
| Spectral gap `2ρ(1−ρ)` | derived, reported | The unit in which violation costs are expressed. |

Freezing ρ matters more than it looks. Thm 3's recovery error scales as `δ / (2ρ(1−ρ))`, so the *cost of every violation* depends on ρ. If Env 1 is calibrated at one ρ and Env 4 at another, the cost curves do not stack — and stacking them is the entire premise of §4.1. Where an environment forces a different ρ, report costs in spectral-gap units so the curves remain comparable.

### 2.2 The metric suite

| Quantity | Metric | Role |
|---|---|---|
| **Linear recovery (primary)** | `R²(h → z)` and `R²(z → h)`, bidirectional OLS on a held-out eval set | Direct comparability with the paper's Tables 1, 2, 5–7 |
| **Orthogonal recovery error (primary)** | `min_{Q∈O(n)} E‖h(z) − Qz‖²`, solved by SVD; report raw and `/n` | The exact quantity Thm 3 bounds |
| **Orthogonality gap** | `‖Q̂ᵀQ̂ − Iₙ‖_F / √n`, plus condition number of `Q̂` | Required for Thm 4 to bind; a well-fitting but ill-conditioned map still degrades planning |
| **Monotone recovery** | recovery error after fitting a monotone reparametrisation, then Procrustes | The Thm 2 / App. F prediction under non-Gaussianity |
| **Probe accessibility (decoy)** | per-latent linear *and* MLP probe `r` / `R²`, `h → z` only, no orthogonal alignment | Deliberately the wrong metric (§2.5). Logged to measure its divergence from actual recovery |
| **Style invariance** | sensitivity of embedding to style factors under known transformations | Did alignment discard the right things? |
| **Whitening error `ε`** | `‖Cov(h(z)) − Iₙ‖_F` | Thm 3 input; ground-truth-free |
| **Alignment gap `δ`** | `L(h) − 2(1−ρ)tr(Cov(h(z)))`, clamped `≥ 0` | Thm 3 input; ground-truth-free; the paper's strongest single predictor of recovery |
| **Predicted error** | `D + (ε + D)²`, `D = δ/(2ρ(1−ρ))` | Thm 3's own prediction; second x-axis of the global scatter |
| **Marginal Gaussianity** | z-scored SIGReg vs matched-i.i.d.-Gaussian floor | Audits the data, not the encoder. Essential for arms C1/C2 |
| **Planning (headline)** | `SR / SR(O)` | Absolute capability |
| **Planning (C2 axis)** | `SR / SR(P)` | Encoder-attributable capability. **The dependent variable of C2** |

Every identifiability metric is reported on **both** the OU training distribution and the physics-rollout evaluation distribution. The gap between them is itself a measurement.

### 2.3 On MCC

MCC is demoted to a secondary diagnostic and must never be a stage-A exit criterion.

MCC as conventionally implemented greedily matches encoder dimensions to latent dimensions, i.e. it scores recovery **up to permutation**. Thm 1 promises recovery **up to an arbitrary orthogonal rotation**. A perfectly correct encoder with a generic `Q` will therefore score poorly on MCC. Using it as the exit gate would cause us to reject exactly the runs that confirm the theorem.

It is still worth logging, in one specific role: MCC computed *without* prior alignment measures whether the encoder happened to land on axis-aligned coordinates, which is strictly stronger than the theorem guarantees and interesting when it occurs (App. F notes that sequential extraction à la xSFA achieves permutation identifiability, at the cost of fragility). Log it, label it as such, never gate on it.

### 2.4 Two definitional cautions

**`δ` under anisotropy.** The definition of `δ` requires a single scalar `ρ`. In arms C1/C2 and under V4, per-dimension `ρ_α` differ. Declare an aggregate `ρ̄` for computing `δ`, report the full per-dimension vector alongside it, and flag that Thm 3's `D` is not strictly applicable in the anisotropic case. Do not silently paper over this — the discrepancy is data.

**Monotone recovery under heterogeneous latents.** App. F's shared-monotone result assumes latents are i.i.d. from the same distribution with the same transition structure. Env 2 and above violate that (a drawer extension and a button are not i.i.d.). Fit **per-latent** monotone maps there, and report both the shared-map and per-latent-map residuals so the divergence between them is visible.

### 2.5 The probe-accessibility decoy metric

Linear probing is the field's default proxy for "the representation learned the right thing," and the paper's own introduction frames identifiability as the condition that makes probing trustworthy in the first place: probing tests whether a latent is *linearly readable*, identifiability requires that the whole representation *is* a linear image of the latents. The two can come apart, and evidence that they do so in practice, on exactly the kind of expert-trajectory data our arms C1/C2 collect, is available from a closely related system (LeWorldModel's Push-T probing results, trained on 20k expert episodes).

Three mechanisms let probe `r` stay high while orthogonal identifiability fails:

- **Direction.** A probe measures `h → z` only. Identifiability requires both directions; the paper's own Table 2 shows the trajectory encoder splitting exactly this way — `R²(h→z)` in the 0.5–0.9 range while `R²(z→h)` is at or below 0.5, sometimes negative.
- **Dimension.** Probing tolerates `m ≫ n`: a linear readout has enormous freedom to find *a* direction that carries a latent even when the ambient encoding is not a rotation of the latents. Thm 1 assumes `m = n`; the `m > n` regime is flagged in the paper as an open problem (§7), not something the guarantee covers.
- **Support.** A probe only has to be linear *on the training-data manifold*. Under the same restricted, non-Gaussian, non-stationary, anisotropic, correlated marginal that V1–V4 and V9 describe, a locally linear readout is easy even when the global map is far from `Qz` — the flattering direction of the support caveat in §0.1.

The decisive check on whether a probe number is measuring anything encoder-specific is a **no-training-signal control**: probe a frozen, off-the-shelf visual encoder that never saw the task's dynamics or any world-model objective (e.g. a pretrained DINO-family backbone) on the same data. If it matches or beats the purpose-trained encoder, the probe is measuring information presence and rough local linear accessibility, not recovery of a World Model. Run this control once per environment where probing is reported.

**Consequence for the program.** Report probe accessibility in every environment and every arm, always alongside — never in place of — the Procrustes recovery error and the orthogonality gap. The interesting quantity is not either number alone but their **divergence**: environments and violations where probe `r` stays high while orthogonal recovery collapses are exactly the cases in §5.2 where practitioners using the field-standard proxy would be misled about planning readiness. Feed this divergence into the global scatter as an annotation (§5.2) rather than a third axis, so the C2 result stays about the theorem's own quantities while still reporting how much probe accessibility a given amount of planning degradation can hide behind.

---

## 3. The fixed violation set

The same nine violations, with the same knob semantics, in every environment. This is what makes the curves stack across the ladder. Some are partly induced by an environment and cannot be switched off; record the induced baseline, then sweep.

### Group A — the theorem's assumptions on the latent process

| | Violation | Scalar knob | Targets |
|---|---|---|---|
| **V1** | Non-Gaussian marginal | Shape parameter of the generalized normal family (`α = 2` Gaussian, `α = 1` Laplace, `α → ∞` uniform) | Uniqueness condition (Thm 2) |
| **V2** | Non-stationarity | Drift rate of OU parameters within an episode | Stationarity |
| **V3** | State-dependent noise | Coupling between latent value and noise scale | Additive noise |
| **V4** | **Anisotropic transitions** | Spread of per-dimension `ρ_α` about the declared `ρ` | Isotropy — necessary for *simultaneous* extraction |
| **V9** | **Latent dependence** | Cross-correlation of the OU sampler's content latents (off-diagonal loading of the driving noise, or a shared factor mixed into two content dimensions) | Assumption (i), independence |

**V1** should use the paper's own sweep family so our curves are directly comparable to Fig. 4b and Figs. 7–8. Expect the peak at `α = 2` and a wider plateau for SIGReg than for pure whitening.

**V9 fills a real gap in v1 of this violation set.** Independence of the latents is Assumption (i) of the world model — it is what lets the transition operator separate and the multivariate Hermite basis factorise into per-dimension eigenproblems (App. A.2, eq. 24). It is not covered by V1–V4: a marginal can be perfectly Gaussian and stationary per dimension while the joint is not independent. Under a behaviour or expert policy this is close to the default state rather than an edge case — an agent that is continually adjacent to the object it manipulates has agent-position and object-position latents that move together. It is cheap to sweep for the same reason V4 is: encoder data comes from an OU process we design, so imposing a target cross-correlation on the driving noise is a single knob, and it is zero by construction in stage A.

**V4 is new relative to v1 of this plan and is the sharpest violation available.** App. F states that isotropic transitions are a *necessary* condition for the simultaneous (parallel) approach, with a closed-form criterion: `max_α K_α < 2 min_β K_β`, i.e. the fastest latent must not be more than twice as fast as the slowest. Translating to discrete-time autocorrelation via `K = −ln ρ / Δt`:

> **`min_α ρ_α > (max_α ρ_α)²`**

When it fails, the failure is *not* graceful. Eigenvalue interleaving causes the encoder to recover the **second Hermite polynomial of a slow latent in place of the first Hermite polynomial of a fast latent** — a discrete, structured substitution, not a smooth degradation. This makes V4 the best-instrumented violation in the whole theory: a closed-form threshold plus a specific predicted failure mode.

It is also nearly free to run, because encoder data comes from an OU process we design, so per-dimension ρ is a dial.

**V4 calibration target.** Apply the criterion to the paper's own Table 2 (Reacher trajectory condition):

| δ | ρ₀ | ρ₁ | `(max ρ)²` | Criterion holds? | Reported `R²(z→h)` |
|---|---|---|---|---|---|
| 1 | 1.000 | 0.999 | 1.000 | ✗ | −0.39 |
| 2 | 0.999 | 0.996 | 0.998 | ✗ | −0.47 |
| 4 | 0.997 | 0.991 | 0.994 | ✗ | −0.05 |
| 8 | 0.992 | 0.982 | 0.984 | borderline | 0.50 |
| 16 | 0.981 | 0.963 | 0.962 | ✓ | 0.44 |
| 32 | 0.959 | 0.928 | 0.920 | ✓ | 0.45 |
| 64 | 0.915 | 0.863 | 0.837 | ✓ | 0.44 |

The sign flip in `R²(z→h)` lands on the criterion boundary. Seven points is not proof — the trajectory condition violates several assumptions at once, and the paper attributes the pattern jointly to Gaussianity and spectral gap (Fig. 13) — but it is a ready-made target. **Reproducing this boundary with a clean, single-violation V4 sweep in Env 1 validates the sampler, the metric suite, and the anisotropy knob in one shot, before any real sweeping begins.** Also check for the predicted second-Hermite substitution directly, by regressing embeddings onto `He₂` of the slow latent.

### Group B — the observation map

| | Violation | Scalar knob | Targets |
|---|---|---|---|
| **V5** | Support truncation | Feasibility rejection strength | Marginal shape |
| **V6** | Occlusion | Camera pose, distractor density | Injectivity of the observation map |

V6 is qualitatively different from everything else here: it breaks injectivity, so exact recovery is impossible in principle rather than merely hard. Expect a genuine floor, and expect the floor to be environment-specific.

### Group C — the things never controlled in practice

| | Violation | Scalar knob | Targets |
|---|---|---|---|
| **V7a** | Under-specified dimension (`m < n`) | `n − m` | Which subspace is selected; superposition |
| **V7b** | Over-specified dimension (`m > n`) | `m − n` | Collapse vs redundancy in extra dimensions |
| **V8** | Optimisation gap | Residual at stopping | Approximate identifiability (Thm 3) |

**V7 is split deliberately.** §7 of the paper treats `m < n` and `m > n` as *different* open problems with different mechanisms: under-specification leaves the Gaussianity constraint unable to determine which subspace is selected or whether the system resorts to superposition; over-specification forces extra dimensions to collapse or encode redundancy. A single signed scalar sweep fitted with one curve would average two unrelated phenomena.

**V8 is the one violation with a quantitative theoretical prediction.** Thm 3 says recovery error `≤ D + (ε + D)²`. Sweeping the stopping residual therefore tests the bound directly, and it is the natural place to check whether the bound holds outside the population-optimum regime it was proved in. Note the paper's finding that the bound's binding term is `D` (alignment), not `ε` (whitening) — approximate whitening is essentially free in practice.

Group C violations are the ones never controlled in real setups, and therefore the ones that transfer.

### Prediction before measurement

For each environment, write down the predicted direction and rough magnitude of every violation cost **before** running it, derived from the theorem and from the Env 1 cost model. Predicted-vs-observed is the record that turns a set of sweeps into a cumulative result.

Three predictions are quantitative from the outset and should be recorded as such: the V4 threshold `min ρ_α > (max ρ_α)²`; the V1 peak at `α = 2`; and the V8 bound `D + (ε + D)²`.

---

## 4. The environment ladder

Environments are ordered by planning difficulty and observation realism. Each keeps a genuine planning purpose; the cycle is what changes nothing and the environment is what changes everything.

### Env 1 — OGBench Cube (single, then double)

**Planning purpose.** Goal-conditioned pick-and-place. The cycle is built and debugged here; expect stage A to take longer than everything above it combined.

**New for identification.** Baseline occlusion (V6) and baseline support truncation (V5) are induced and non-zero. This is where the violation cost model is constructed, and where the V4 calibration target is hit.

**Environment-specific decisions**

- **Marker.** Place an asymmetric marker on the cube. A textureless cube has 24-fold rotational symmetry, so distinct orientations render identically and orientation recovery is meaningless without it. The marker buys observability, not Gaussianity.
- **Orientation sampling.** No isotropic Gaussian exists on SO(3). Sample OU in the tangent space at identity and map via the exponential map, at small angular variance. Angular variance becomes an environment-specific extra knob: as it grows, curvature and eventually wraparound break the assumption.
- **Orientation scoring.** The declared content latent is the **tangent vector**, not the rotation, because that is what carries the Gaussian marginal. Scoring recovery against the rotation itself would misattribute exponential-map curvature to encoder failure. Report the tangent-space recovery as primary and the induced error on the manifold as a separate diagnostic.
- **Feasibility.** Allow interpenetration and table-clipping in encoder data rather than rejecting it. This keeps the marginal exactly Gaussian and makes evaluation support a subset of training support, which is benign. The V5 sweep then measures what rejection would have cost.
- **Single before double.** Going to two cubes adds object permutation symmetry. Run identical vs distinct markers to isolate symmetry cost from latent-dimension cost. Note that permutation symmetry interacts with V7 and with the orthogonal recovery class — with identical markers, the true latent map is only defined up to a permutation, so the recovery metric must quotient by it.

### Env 2 — OGBench Scene

**Planning purpose.** Multi-stage manipulation with articulated objects (drawer, window, button) and sequential dependencies. The predictor stops being trivial.

**New for identification.** Heterogeneous, bounded, partly near-discrete latents. A drawer extension is an interval; a button is near-binary. This is a structural V1 that cannot be switched off, because it is a property of the latent *marginals* we must sample from, not of the trajectories.

**It is also the first environment with a large induced V4 baseline — but only in arms C1/C2.** A button and a drawer have very different characteristic timescales under physics, so trajectory-collected encoder data will be strongly anisotropic. Stage A is unaffected: the OU sampler sets `ρ_α` directly and isotropy is free. Measure the induced per-dimension `ρ_α` on rollouts, check the criterion, and expect it to be a leading term in the A→C1 gap here. This is the first environment where the C2 stride-matching control earns its keep.

**Environment-specific decisions**

- Declare per-latent type, range, and timescale explicitly; recovery metrics must be computed type-aware.
- Report **both** shared-monotone and per-latent-monotone recovery (§2.4), since the i.i.d. premise of App. F's generalisation fails here.
- First real test of whether the Env 1 cost model predicts an induced violation it did not generate.
- Long-horizon planning makes the predictor a live variable; hold its architecture fixed across arms, and lean on arm P to attribute.

### Env 3 — OGBench Puzzle

**Planning purpose.** Combinatorial, near-discrete state, long horizon.

**New for identification.** The latent space is essentially discrete — the strongest available violation of a continuous Gaussian latent.

**Environment-specific decisions**

- Stage A is expected to degrade sharply or fail. That is the intended result: this environment locates the edge of the theorem's applicability.
- **Be precise about what is predicted.** The Sturm–Liouville machinery behind Thm 2 assumes constant diffusion and a stationary density with full support on `ℝ`. Genuinely discrete latents satisfy neither, so the monotone-recovery fallback has *no guarantee* here — unlike in Env 2, where interval-valued latents keep it plausible. Env 3 is outside the theory, not merely at its non-Gaussian edge. Record monotone recovery anyway; a surprise in either direction is informative.
- The decisive measurement is whether planning SR survives poor recovery. If it does, identifiability is sufficient but not necessary — a finding no theory result supplies. Arm P is what makes this interpretable: if `SR/SR(P)` stays high while recovery collapses, the claim is real; if only `SR/SR(O)` moves, the predictor is doing the work.
- May be reordered relative to Env 2 depending on Env 2 outcomes. It stays in the ladder either way.

### Env 4 — Isaac / ManiSkill-class manipulation

**Planning purpose.** The manipulation semantics of Env 1–2 under contact-rich dynamics, high physics fidelity and photorealistic rendering.

**New for identification.** Realistic observation statistics — lighting, materials, shadows, sensor noise — with ground truth still available. Last environment where recovery against true latents is directly computable.

**Environment-specific decisions**

- Content/style split becomes a first-class design object; appearance randomisation is now a real axis with its own invariance metric.
- **Style must be resampled within the positive pair.** Style held constant across a pair is indistinguishable from content by the alignment loss and will be encoded.
- **Flag the theoretical consequence:** within-pair style jitter is variation on the positive pair that is *not* the OU transition on content latents. It modifies the transition operator, so Env 4 is not merely a new axis — it steps partially outside the world Thm 1 describes. Record it as a declared, deliberate departure rather than discovering it later as an anomaly. One clean control worth running: an appearance-fixed condition, to bound how much of any Env 4 gap is attributable to this.
- Primary role: validate that the cost model built in Env 1–3 predicts behaviour under realistic observations.

### Env 5 — Cosmos-transferred Env 4

**Planning purpose.** Unchanged from Env 4 by construction. Only the observation channel is swapped, which is the entire reason to build it this way.

**New for identification.** Real-video observation statistics with ground truth preserved through the translation.

**Environment-specific decisions**

- **Verify latent preservation before trusting any number.** A generative model can silently move objects. Check with a detector or by re-estimating state from transferred frames; report the check. This is a gate, not a footnote.
- **Scientific payload:** this is the only environment with both ground-truth recovery *and* ground-truth-free proxies. Whichever proxies track recovery here are the ones to trust on a real robot, where ground truth does not exist. That is the deliverable that leaves simulation.
- **Candidate proxies, fixed in advance, ranked by the paper's own evidence:**
  1. **Alignment gap `δ`** — App. H.9 finds alignment loss the strongest single predictor of identifiability among converged runs. Primary candidate.
  2. **Thm 3's predicted error `D + (ε + D)²`** — the theory's own composite. Tests whether the bound is not just valid but *tight enough to rank*.
  3. **SIGReg residual** — correlated with whitening loss, weaker alone.
  4. **Whitening error `ε`** — expected to be the weakest, since App. H.8 shows large `ε` alone does not predict poor recovery while large `δ` does. Include it precisely so that this negative result is reproduced rather than assumed.
  5. **Orthogonality gap** of the fitted map.
  6. **Invariance under known transformations**, and probes against partial labels.
- Report the proxy ranking, not just which proxies "work" — the ordering is the transferable result.

---

## 5. Cumulative artifacts

Two objects accumulate across the ladder. They, not any single environment, are the output of the program.

### 5.1 The violation cost table

- Nine violations (ten curves, counting V7a/V7b separately) × five environments, each entry a degradation curve rather than a number.
- Costs expressed in **spectral-gap units** so they remain comparable if ρ is ever forced to differ.
- Constructed at Env 1, predicted forward, checked at every environment above.
- Where predictions hold, the table is a tool: it tells a practitioner in advance what their data will cost them.
- Where predictions break, the failure is an interaction effect between violations — more interesting than either alone, and invisible to single-assumption theory. The paper's own Reacher result is exactly such a case: Fig. 13 shows Gaussianity and spectral gap failing *together* at small stride and *separately* at large stride, which is why identifiability peaks in the middle.

### 5.2 The identifiability-to-planning scatter

- One point per (environment × violation × severity × seed).
- **y-axis: `SR / SR(P)`** — encoder-attributable planning success. Log `SR / SR(O)` alongside.
- **x-axis, two of them:**
  1. Measured orthogonal recovery error (and separately the orthogonality gap) — the ground-truth axis, available through Env 5.
  2. Thm 3's predicted error `D + (ε + D)²` — the **ground-truth-free** axis, available everywhere including on a real robot.
- Tight and monotone on axis (1) across five environments of increasing realism is the C2 result, and it fixes the exchange rate between representation quality and planning quality.
- Tight and monotone on axis (2) is the stronger and more portable result: it means a practitioner can predict planning degradation from training-side quantities alone. The paper establishes the ingredient (loss predicts identifiability); this program would establish the consequence (loss predicts planning).
- **Annotate each point with probe-accessibility divergence** (§2.5): probe `r` minus a rescaled orthogonal recovery score. Points with high divergence are the ones where the field-standard proxy would have reported the encoder as fine. Where those points also sit at low `SR/SR(P)`, that is the direct, quantitative version of the LeWM/DINOv2 observation that probe quality and planning-relevant recovery can decouple.
- **Log for this from the first Env 1 run. Retrofitting it is not possible.**

### 5.3 Planning capabilities unlocked by orthogonality

Orthogonal identifiability is what makes latent L2 a meaningful cost, and goal-conditioned latent L2 is `O(n)`-invariant, so OGBench is the right task family for the guarantee to bind. Prototype at Env 1, stress-test upward:

- **Goal specification directly in latent space.** App. D.1 makes the sharp point: a planner choosing a goal `ẑ*` and minimising `‖ẑ_T − ẑ*‖²` solves the true goal-reaching problem for `z* = Qᵀẑ*`. **The learned goal never needs to be mapped back to the true latent.** The plan is correct as a sequence of actions regardless of the coordinate system it was computed in.
- **Goals specified on a subspace** of latents rather than a full state.
- **Costs provably equivalent to the true-space cost** — the `O(n)`-invariant family: norms and radial functions, inner products between states, quadratic forms specified in learned coordinates, Gaussian belief updates, linear value functions in learned coordinates.
- **LQR in the learned latent.** App. D.1 shows the discrete algebraic Riccati equation is covariant under orthogonal reparametrisation, so the optimal feedback gain transforms as `K̂ = KQᵀ` and yields the *true* LQR action at every state. This is a concrete, checkable capability, not an analogy.
- **Well-conditioned latent-space optimisation.**
- **Attribution:** which latent dimensions a plan actually depends on.
- **Consistency requirement.** App. D.2: the residual orthogonal ambiguity requires encoder, transition model, and cost to be mutually consistent in the learned coordinates. State this as a design constraint on the stage-D pipeline, and verify it — an inconsistency here would look like an identifiability failure while being an integration bug.

> "Identifiability enables planning algorithms that are not available without it" is a stronger and more durable claim than "identifiability improves planning performance."

---

## 6. Known risks

| Risk | Mitigation |
|---|---|
| Stage-A exit criterion rejects correct encoders | Recovery is scored **up to orthogonal rotation**, never by unaligned MCC. Written into the frozen metric code and its tests |
| Env 1 stage A never reaches ceiling; cause ambiguous between pipeline and assumptions | Keep a free-space, no-occlusion, full-support, isotropic-ρ debug configuration available as a pipeline check whenever a stage-A failure is ambiguous. Additionally, the V4 Table-2 boundary reproduction serves as an end-to-end validation before real sweeping |
| Planning results unattributable between encoder and predictor | Arm P (learned predictor on ground-truth state) in every stage D. `SR/SR(P)` is the C2 axis |
| ρ or λ drift across environments, breaking cost-curve comparability | Frozen as program constants; costs reported in spectral-gap units; any forced deviation is logged as a first-class event |
| A→C gap silently confounded by a ρ mismatch | Arm C2 matches frame stride so empirical ρ lands near the declared ρ; per-dimension ρ reported for both arms |
| Anisotropy is inherited rather than swept, and its structured failure mode is read as generic degradation | V4 is an explicit knob with a closed-form threshold; induced baseline measured in arms C1/C2; test directly for second-Hermite substitution |
| Over-constraining predictor or planning data with assumptions that only bind at encoder training | §0.1 is the reference. V1–V4 do not apply after freezing; predictor data is collected for excitation and coverage instead |
| Encoder trains on OU-set states, predictor and planner see only physics rollouts | Treat as a distribution shift to be measured, not asserted away. Always report recovery on the rollout distribution |
| Metric or budget drift mid-program | Freeze implementations and seed/severity budget before Env 1; version the evaluation code separately from training code |
| Env 4 style jitter changes the transition operator without being noticed | Declared in advance as a departure from the theorem's world; appearance-fixed control run |
| Cosmos transfer silently perturbs latents | Latent-preservation check is a gate on Env 5, not a footnote |
| Non-`O(n)`-invariant cost sneaks into stage D, voiding Thm 4 | Cost family declared and enforced in code; any new cost requires an explicit invariance argument |
| Scope growth — full cycle on five environments is large | Pre-commit per environment to which claims get full seeds and which get one. Env 1–2 carry the full cycle; higher rungs may run a reduced cycle. Within a reduced cycle, V1, V4 and V8 are the non-negotiable violations, since they are the three with quantitative predictions |

---

## 7. Immediate next actions

1. **Fix the metric implementations, tests first:** bidirectional linear R²; Procrustes orthogonal recovery error (raw and `/n`); orthogonality gap and condition number; monotone recovery (shared and per-latent variants); `ε`, `δ`, and `D + (ε + D)²`; z-scored SIGReg with a matched-Gaussian floor; per-latent linear and MLP probe accessibility (§2.5). Include a test that a synthetic `h(z) = Qz` with random `Q ∈ O(n)` scores at floor on recovery error and **is not** required to score well on unaligned MCC; and a complementary test that a synthetic `h` with `m ≫ n` and a restricted-support `z` can score well on probe accessibility while failing Procrustes recovery, to confirm the decoy metric actually decouples the way §2.5 predicts before it is trusted on real runs.
2. **Pick and freeze the frozen-encoder probe control** for §2.5 (a specific pretrained visual backbone), so the DINOv2-style no-training-signal check is run identically across environments.
3. **Declare the program constants:** `ρ`, `λ`, seed budget, severity budget per violation. Record the rationale against Fig. 6's regimes.
4. **Declare the Env 1 content/style split**, the marker design, and the tangent-space scoring convention for orientation.
5. **Implement the tangent-space OU sampler** with per-dimension `ρ_α` exposed as a first-class parameter, a target cross-correlation knob for V9, and the direct state-setting render path.
6. **Run the V4 calibration check** against the Table 2 boundary before any production sweep: sweep anisotropy in the debug configuration, confirm the `min ρ_α > (max ρ_α)²` threshold, and test for second-Hermite substitution beyond it.
7. **Stand up the logging schema for the global scatter** — including arm P, both y-axes, both x-axes, and the probe-accessibility divergence annotation — before the first training run.

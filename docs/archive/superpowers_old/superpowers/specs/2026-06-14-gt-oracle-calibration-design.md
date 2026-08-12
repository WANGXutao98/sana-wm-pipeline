# Design Spec — GT-Oracle Calibration & Robustification of the SANA-WM Annotation Pipeline

> **Date:** 2026-06-14
> **Status:** Design approved, ready for implementation planning
> **Repo:** `/mnt/afs/davidwang/workspace/sana_wm_pipeline/`
> **Conda env:** `sana_wm`
> **Role in dissertation:** Paper #1 of a 3-paper arc — **D3 → D1 → D2**
> **Venue:** intentionally venue-agnostic (NeurIPS D&B *or* CVPR/ICCV; see §10)

---

## 0. One-paragraph summary

The SANA-WM annotation pipeline (VIPE ⊕ Pi3X ⊕ MoGe-2, three modes, plus the in-progress CADF
fusion kernels) produces metric-scale 6-DoF camera trajectories from RGB-only video, but its error
behaviour has only ever been measured on TUM desks plus a couple of OmniWorld/DL3DV scenes. No real
GT dataset can *independently vary* a single nuisance factor (e.g. amount of dynamic-object motion)
while holding the camera path fixed. **SimWorld can.** We use SimWorld as a controllable
ground-truth oracle to (C1) produce the first factorial error-characterization of a world-model
annotation pipeline, (C2) evaluate confidence-aware depth fusion (CADF) against *dense* GT depth +
GT pose across that factor grid — gated by a decisive validate-or-kill experiment — and (C3) train a
learned, transferable pseudo-label **quality filter** that predicts a clip's actual pose error from
pipeline-internal signals and replaces the heuristic Stage-04 thresholds, with downstream world-model
gains shown where real GT exists.

---

## 1. Thesis & why it is paper-worthy

The only labels SANA-WM trains on are **first frame + text caption + 6-DoF camera trajectory**
(conditioning the video itself). The geometric trajectory is the irreplaceable label that internet
data cannot otherwise supply, and the pipeline that produces it has never been characterized across
controlled conditions. SimWorld closes that gap because it emits perfect, dense GT (camera pose,
intrinsics, metric depth, instance segmentation, per-object 6-DoF) on infinite, controllable,
reproducible scenes.

Three contributions, none of which internet data can support:

- **C1 — Error characterization.** First factorial map of annotation error vs *independently varied*
  nuisances (scene type, camera-motion profile, scene dynamics, lighting/weather, depth range).
- **C2 — CADF, properly evaluated.** Confidence-aware fusion kernels scored against *dense* GT depth
  and GT pose across the grid — converting "small win on TUM" into "the demonstrated fix for a
  characterized failure mode." **Conditional on the §6 gating experiment.**
- **C3 — Learned transferable quality filter.** A light regressor that predicts a clip's true pose
  error from pipeline-internal features, trained where sim GT exists and deployed to auto-filter real
  internet clips — i.e. Stage 04 done right (learned + GT-calibrated), with a downstream WM check.

---

## 2. Scope decisions — what the pipeline actually needs (folded in)

Every SANA-WM module sorted by whether it *creates a label*, *selects clips*, or *adds volume*:

| Tier | Modules | Role | In scope? |
|---|---|---|---|
| **1 — Label producers (core)** | **VIPE + Pi3X + MoGe-2** | Metric-scale 6-DoF trajectory — the irreplaceable label | **Yes — already validated, the spine** |
| **2 — Label producer (substitutable)** | Qwen3.5-VL caption | Text label; *function* needed, *model* swappable | **Deferred to Phase 2** (Module D); swap for Qwen2.5-VL-7B |
| **3 — Curation gates** | UniMatch, DOVER, VLM/Table-6 | *Select* clips; do not create labels | **Demoted to FEATURES for Module Q**, not hard gates |
| **4 — Augmentation (blocked)** | FCGS + DiFix3D (Stage 03) | Additive DL3DV volume; not open-source | **Dropped entirely** |

### Folded-in decision A (scope)
- **Drop FCGS / DiFix3D** (Stage 03) from the entire D3 → D1 → D2 arc. Not open-source, DL3DV-only,
  irrelevant to label quality and to all three contributions. If DL3DV trajectory diversity is
  wanted later, use DL3DV's native multi-view captures directly.
- **UniMatch (optical-flow score)** and optionally **DOVER (quality score)** are bound as
  **feature extractors feeding Module Q**, not as standalone heuristic thresholds. UniMatch is
  open-source (~300 MB) and cheap to bind.
- **Qwen captioning** moves to **Phase 2** (downstream WM training only). It is off the critical path
  for the annotation-science contributions.

**Minimal stack for the whole arc:** `VIPE + Pi3X + MoGe-2 + CADF + (UniMatch as a feature)`.

---

## 3. Architecture — five isolated modules

| Module | Responsibility | Depends on | Independently testable via |
|---|---|---|---|
| **S** — SimWorld GT extractor | Script scenes × programmed camera paths × agent-action scripts × {weather, seed}; dump per-frame RGB / depth / object_mask / GT c2w + intrinsics / per-object 6-DoF into a **canonical GT-scene format** | SimWorld API only | schema test on a fixture capture |
| **O** — Oracle harness | Run the existing pipeline (default / gtdepth / gtpose × CADF kernels) on the *same* RGB; compute estimated-vs-GT errors (ATE RMSE, RTE rot/trans, depth AbsRel/δ1, metric-scale err, intrinsics err) as functions of the grid axes | canonical GT + pipeline outputs | synthetic known-error cases |
| **C** — CADF (extended) | The 4 kernels (`baseline_ema` / `conf_weighted` / `irls` / `robust_geomedian`) from the next-steps plan, scored vs **dense** GT | raw Pi3X/MoGe cache | existing kernel unit tests + GT-error |
| **Q** — Learned quality predictor | Feature extractor over pipeline-internal signals (Pi3X `conf` stats, MoGe/Pi3X residual histograms, VIPE BA residuals, UniMatch flow, scale-CV) → MLP/GBT → regress ATE/RTE; train on sim GT, validate on real GT subsets; deploy as the Stage-04 filter | (feature, GT-error) pairs | tiny-fixture train/eval |
| **D** — Downstream check | Finetune SANA-WM (or a small proxy) on Q-filtered vs unfiltered real shards; measure control-accuracy delta. Brings in Qwen captioning | filtered shards | staged / optional for v1 |

**Isolation contract.** Module S knows only SimWorld + the canonical format. Module O knows only the
canonical format + pipeline outputs. Module Q knows only feature vectors + scalar GT error. Each can
be developed and tested without the others present.

---

## 4. Data flow

```
SimWorld factor grid ─S─► canonical GT scenes ─┬─ RGB ──► pipeline ─► estimates ─┐
                                               └─ GT ────────────────────────────┤
                                                                                  ▼
                                            Module O ─► error tables/plots (C1) + CADF matrix (C2)
                                                       └─► (feature, GT-error) pairs ─► Q (C3)
                                                                                         │
                            real internet clips ──► pipeline internals ──► Q filter ─► filtered shards ─► D
```

---

## 5. Phasing — robust to the in-progress UE5 setup

UE5/SimWorld is being stood up on this H100 box **and** CMCC right now, so nothing blocks on it:

- **Phase 0 (start now, no UE5):** build **O + C + Q** on GT already in hand — TUM fr1/fr2, OmniWorld
  GT-depth, DL3DV gtpose. Also run the **§6 CADF gating experiment** here. Deliverable: cross-dataset
  CADF matrix + a first Q + the go/no-go on CADF.
- **Phase 1 (when UE5 renders):** add **Module S** → the controllable factor grid → C1's
  independently-varied error characterization, the part no real dataset can produce. Scientific core.
- **Phase 2:** **Module D** downstream validation → the "so what" for a methods-venue framing
  (brings in Qwen captioning).

---

## 6. Folded-in decision B — CADF validate-or-kill gating experiment

CADF only changes the per-frame **scale scalar** `s_t` (relative→metric conversion). The existing
estimator already has inverse-depth weighting (`w = 1/d_Pi3X`), EMA-0.99 smoothing, and an 80th-pct
Umeyama inlier filter — so Pi3X `conf` is *partly redundant* with machinery already present. CADF's
defensible theory of action is **suppressing metric-scale drift on long / dynamic / reflective /
textureless videos**, not improving rotation or local structure.

**Two cheap, decisive Phase-0 tests (on existing dense GT — OmniWorld, TUM) decide whether CADF stays
in scope:**

1. **Independence test.** Partial correlation of per-pixel `conf` with GT-depth error, *controlling
   for* `1/d_Pi3X`. If conf explains error variance **beyond** the existing weight → CADF has
   headroom. If not → drop CADF.
2. **Scale-vs-shape decomposition.** With GT depth, split final pose error into the part attributable
   to scale (the only part CADF can touch) vs the rest. If scale's share is small → CADF's ceiling is
   low regardless of kernel.

**Gate:**
- **Both positive →** CADF stays as contribution C2; run the full kernel × factor-grid matrix, with a
  mandatory `EMA {on/off} × window` ablation (EMA may mask CADF's per-frame benefit).
- **Either negative →** drop CADF; the paper stands on C1 (characterization) + C3 (learned filter),
  which are self-sufficient.

This gate is a hard prerequisite before any large CADF compute is spent.

---

## 7. Key engineering risks

1. **Coordinate conventions.** SimWorld/UE is left-handed, cm, X-forward/Z-up, pitch/yaw/roll degrees;
   the pipeline expects OpenCV-style c2w in metres. → explicit, **unit-tested** conversion in Module S.
2. **Label/RGB alignment fairness.** Module O must feed the pipeline the exact RGB the GT was rendered
   from; assert frame-index alignment (the project has been bitten by `frame_idx` before).
3. **VIPE OOM on long sequences.** Reuse existing chunking/truncation; log per-scene.
4. **Q sim→real transfer.** The central scientific risk for C3 — mitigate by including real GT subsets
   in validation and reporting sim-only vs sim+real-calibrated separately.

---

## 8. Testing & verification

TDD per project convention (pytest; currently 141 passing). New unit tests:
- canonical GT-scene schema,
- error-metric correctness on synthetic known-error inputs,
- Module S coordinate conversion (round-trip),
- Q train/eval on a tiny fixture.

**Regression guard:** reproduce the known TUM result (CADF `B < baseline A`) so refactors cannot
silently break the headline number.

---

## 9. Roadmap hooks for D1 / D2

- Module S scene/camera scripting → **D1 SimReal-4D** capture layer.
- Canonical GT-scene format → **D1** dataset schema (a superset of SANA-WM Stage 06).
- Module D training/eval harness → **D2 beyond-camera control** training loop.

---

## 10. Venue-agnostic framing

Same artifacts, two abstracts:
- **NeurIPS Datasets & Benchmarks:** lead with C1 (characterization) + released oracle + C3 (filter).
- **CVPR / ICCV:** lead with C2 (CADF) + C3 (learned method) + Module D downstream gains.

---

## 11. Out of scope (explicit)

- FCGS / DiFix3D / Stage-03 3DGS augmentation.
- Faithful Qwen3.5-VL captioning (Phase 2 only; substitute model acceptable).
- Per-frame-intrinsics BA C++/CUDA changes (known permanent gap; not required here).
- The D1 and D2 papers themselves (roadmap only).

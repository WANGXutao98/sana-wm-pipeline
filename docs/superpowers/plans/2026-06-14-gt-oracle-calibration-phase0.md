# GT-Oracle Calibration — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-UE5 core of the D3 annotation-oracle paper — confidence-aware depth-fusion kernels (Module C), the CADF validate-or-kill gate (§6 of the spec), the error-metric / GT-scene harness (Module O), and a learned pseudo-label quality filter (Module Q) — all runnable on existing GT (TUM, OmniWorld).

**Architecture:** A new reusable package `src/sana_wm_pipeline/oracle/` holds pure, unit-tested logic (GT-scene loading, error metrics, fusion kernels, gate statistics, features, quality model). Experiment drivers live in `experiments/cadf_research/` and `experiments/oracle_calibration/` and call into that package. The CADF gate (Task 6) is a hard prerequisite: if it fails, Tasks on the full CADF matrix are skipped and the paper leans on Module O + Module Q.

**Tech Stack:** Python 3.10, NumPy, SciPy (stats), scikit-learn (GBT regressor), pytest. Reuses existing `stage02_pose/depth_fusion.py`, `experiments/vipe_comparison/evaluate.py` (ATE/RTE), `precompute_pi3x_depths.py` (Pi3X/MoGe runners). Conda env `sana_wm`.

**Spec:** `docs/superpowers/specs/2026-06-14-gt-oracle-calibration-design.md`

**Out of scope (separate future plans):** Module S / SimWorld capture (Phase 1, needs UE5); Module D / downstream WM training + Qwen captioning (Phase 2); FCGS/DiFix3D (dropped).

---

## File Structure

| File | Responsibility | Type |
|---|---|---|
| `src/sana_wm_pipeline/oracle/__init__.py` | package marker | new |
| `src/sana_wm_pipeline/oracle/gt_scene.py` | `GTScene` dataclass + TUM/OmniWorld adapters → common in-memory GT | new |
| `src/sana_wm_pipeline/oracle/metrics.py` | depth (AbsRel/δ1), metric-scale, intrinsics errors; re-exports ATE/RTE | new |
| `src/sana_wm_pipeline/oracle/fusion_kernels.py` | 4 CADF kernels, unified `fuse(...)` API | new |
| `src/sana_wm_pipeline/oracle/cadf_gate.py` | partial-correlation + scale/shape decomposition | new |
| `src/sana_wm_pipeline/oracle/features.py` | per-clip feature vector from pipeline internals | new |
| `src/sana_wm_pipeline/oracle/quality_model.py` | train/eval GBT regressor for pose error | new |
| `experiments/cadf_research/precompute_cadf_cache.py` | conf-preserving Pi3X+MoGe cache (npz) | new |
| `experiments/cadf_research/run_gate.py` | §6 gate driver + go/no-go report | new |
| `experiments/oracle_calibration/run_oracle.py` | run pipeline vs GT, emit error tables + (feature,error) pairs | new |
| `experiments/oracle_calibration/train_filter.py` | train + evaluate Module Q | new |
| `tests/test_oracle_gt_scene.py` ... `tests/test_oracle_quality_model.py` | unit tests | new |

---

## Task 1: GTScene container + dataset adapters (Module O foundation)

**Files:**
- Create: `src/sana_wm_pipeline/oracle/__init__.py`
- Create: `src/sana_wm_pipeline/oracle/gt_scene.py`
- Test: `tests/test_oracle_gt_scene.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_gt_scene.py
import numpy as np
from sana_wm_pipeline.oracle.gt_scene import GTScene


def test_gtscene_validates_shapes():
    T, H, W = 5, 4, 6
    s = GTScene(
        poses_c2w=np.tile(np.eye(4), (T, 1, 1)),
        depth=np.ones((T, H, W), np.float32),
        depth_valid=np.ones((T, H, W), bool),
        intrinsics=np.array([[10.0, 10.0, 3.0, 2.0]] * T),  # fx,fy,cx,cy
        fps=30.0,
        source="unit",
    )
    assert s.num_frames == T
    assert s.resolution == (H, W)


def test_gtscene_rejects_mismatched_T():
    import pytest
    with pytest.raises(ValueError):
        GTScene(
            poses_c2w=np.tile(np.eye(4), (5, 1, 1)),
            depth=np.ones((4, 4, 6), np.float32),  # T mismatch
            depth_valid=np.ones((4, 4, 6), bool),
            intrinsics=np.array([[10.0, 10.0, 3.0, 2.0]] * 5),
            fps=30.0,
            source="unit",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/afs/davidwang/workspace/sana_wm_pipeline && python -m pytest tests/test_oracle_gt_scene.py -v`
Expected: FAIL with `ModuleNotFoundError: ... oracle.gt_scene`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/oracle/__init__.py
"""GT-oracle calibration package (D3 Phase 0)."""
```

```python
# src/sana_wm_pipeline/oracle/gt_scene.py
"""Canonical in-memory ground-truth scene + dataset adapters.

A GTScene is the common interface Module O consumes. Phase-0 adapters convert
existing GT datasets (TUM RGB-D, OmniWorld) into this form; Phase-1 Module S
(SimWorld) will emit the same structure.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GTScene:
    poses_c2w: np.ndarray      # (T,4,4) camera-to-world, OpenCV convention, metres
    depth: np.ndarray          # (T,H,W) metric depth, metres
    depth_valid: np.ndarray    # (T,H,W) bool, True where depth is trustworthy
    intrinsics: np.ndarray     # (T,4) per-frame [fx,fy,cx,cy] in pixels
    fps: float
    source: str

    def __post_init__(self) -> None:
        T = self.poses_c2w.shape[0]
        if self.poses_c2w.shape[1:] != (4, 4):
            raise ValueError(f"poses_c2w must be (T,4,4), got {self.poses_c2w.shape}")
        if self.depth.shape[0] != T or self.depth_valid.shape != self.depth.shape:
            raise ValueError("depth / depth_valid frame count must match poses")
        if self.intrinsics.shape != (T, 4):
            raise ValueError(f"intrinsics must be (T,4), got {self.intrinsics.shape}")

    @property
    def num_frames(self) -> int:
        return self.poses_c2w.shape[0]

    @property
    def resolution(self) -> tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_gt_scene.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/oracle/__init__.py src/sana_wm_pipeline/oracle/gt_scene.py tests/test_oracle_gt_scene.py
git commit -m "feat(oracle): add GTScene container with shape validation"
```

---

## Task 2: Error metrics — depth, metric-scale, intrinsics (Module O)

**Files:**
- Create: `src/sana_wm_pipeline/oracle/metrics.py`
- Test: `tests/test_oracle_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_metrics.py
import numpy as np
from sana_wm_pipeline.oracle.metrics import depth_errors, metric_scale_error, intrinsics_error


def test_depth_errors_perfect_is_zero():
    d = np.full((3, 4, 4), 2.0, np.float32)
    valid = np.ones_like(d, bool)
    out = depth_errors(d, d, valid)
    assert out["abs_rel"] == 0.0
    assert out["delta1"] == 1.0


def test_depth_errors_constant_offset():
    gt = np.full((1, 2, 2), 2.0, np.float32)
    pred = np.full((1, 2, 2), 3.0, np.float32)  # 50% over
    valid = np.ones_like(gt, bool)
    out = depth_errors(pred, gt, valid)
    assert abs(out["abs_rel"] - 0.5) < 1e-6
    assert out["delta1"] == 0.0  # ratio 1.5 > 1.25


def test_metric_scale_error_detects_global_scale():
    gt = np.full((2, 3, 3), 4.0, np.float32)
    pred = np.full((2, 3, 3), 5.0, np.float32)  # scale 1.25
    valid = np.ones_like(gt, bool)
    assert abs(metric_scale_error(pred, gt, valid) - 0.25) < 1e-6


def test_intrinsics_error_relative():
    gt = np.array([[100.0, 100.0, 50.0, 50.0]])
    pred = np.array([[110.0, 90.0, 50.0, 50.0]])
    out = intrinsics_error(pred, gt)
    assert abs(out["fx_rel"] - 0.1) < 1e-6
    assert abs(out["fy_rel"] - 0.1) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: ... oracle.metrics`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/oracle/metrics.py
"""Geometry error metrics for the GT oracle.

ATE/RTE are reused from experiments/vipe_comparison/evaluate.py; this module
adds dense-depth, metric-scale, and intrinsics errors that the existing
evaluator lacks.
"""
from __future__ import annotations
import numpy as np


def depth_errors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """AbsRel and delta<1.25 over valid, strictly-positive GT pixels."""
    m = valid & (gt > 1e-6) & (pred > 1e-6)
    if not m.any():
        return {"abs_rel": float("nan"), "delta1": float("nan")}
    p, g = pred[m], gt[m]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    return {"abs_rel": abs_rel, "delta1": delta1}


def metric_scale_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """|median(pred/gt) - 1| — global metric-scale bias (CADF's target component)."""
    m = valid & (gt > 1e-6) & (pred > 1e-6)
    if not m.any():
        return float("nan")
    return float(abs(np.median(pred[m] / gt[m]) - 1.0))


def intrinsics_error(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Mean relative error per intrinsic param. pred/gt are (T,4) [fx,fy,cx,cy]."""
    eps = 1e-9
    rel = np.abs(pred - gt) / (np.abs(gt) + eps)
    return {
        "fx_rel": float(np.mean(rel[:, 0])),
        "fy_rel": float(np.mean(rel[:, 1])),
        "cx_rel": float(np.mean(rel[:, 2])),
        "cy_rel": float(np.mean(rel[:, 3])),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/oracle/metrics.py tests/test_oracle_metrics.py
git commit -m "feat(oracle): add depth/metric-scale/intrinsics error metrics"
```

---

## Task 3: CADF fusion kernels (Module C)

**Files:**
- Create: `src/sana_wm_pipeline/oracle/fusion_kernels.py`
- Test: `tests/test_oracle_fusion_kernels.py`

All kernels share one signature so the gate/oracle can iterate over them:
`fuse(d_pi3x, conf, d_moge, mask, momentum=0.99) -> (fused_depths (T,H,W), scale_history (T,))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_fusion_kernels.py
import numpy as np
from sana_wm_pipeline.oracle.fusion_kernels import KERNELS, fuse


def _toy():
    T, H, W = 4, 8, 8
    rng = np.random.default_rng(0)
    d_pi3x = rng.uniform(0.5, 5.0, (T, H, W)).astype(np.float32)
    d_moge = 2.0 * d_pi3x  # true scale = 2.0 everywhere
    conf = np.ones((T, H, W), np.float32)
    mask = np.ones((T, H, W), bool)
    return d_pi3x, conf, d_moge, mask


def test_all_kernels_recover_clean_scale():
    d_pi3x, conf, d_moge, mask = _toy()
    for name in KERNELS:
        fused, scale = fuse(name, d_pi3x, conf, d_moge, mask, momentum=0.0)
        assert scale.shape == (4,)
        assert np.allclose(scale, 2.0, atol=0.05), f"{name}: {scale}"
        assert np.allclose(fused, 2.0 * d_pi3x, atol=0.1), name


def test_conf_kernel_ignores_low_conf_outliers():
    # one frame: half the pixels are garbage but flagged conf=0
    d_pi3x = np.ones((1, 4, 4), np.float32)
    d_moge = np.full((1, 4, 4), 3.0, np.float32)   # true scale 3.0
    d_moge[0, :2, :] = 100.0                         # outliers
    conf = np.ones((1, 4, 4), np.float32)
    conf[0, :2, :] = 0.0                             # flagged unreliable
    mask = np.ones((1, 4, 4), bool)
    _, s_base = fuse("baseline_ema", d_pi3x, conf, d_moge, mask, momentum=0.0)
    _, s_conf = fuse("conf_weighted", d_pi3x, conf, d_moge, mask, momentum=0.0)
    assert abs(s_conf[0] - 3.0) < abs(s_base[0] - 3.0)  # conf kernel closer to truth
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_fusion_kernels.py -v`
Expected: FAIL with `ModuleNotFoundError: ... oracle.fusion_kernels`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/oracle/fusion_kernels.py
"""Confidence-Aware Depth Fusion (CADF) kernels.

Each kernel estimates a per-frame metric scale s_t mapping relative Pi3X depth
to MoGe-2 metric depth, then EMA-smooths it. baseline_ema reproduces the
SANA-WM paper (App. B.1, conf discarded); the others use Pi3X conf to suppress
unreliable pixels. Unified API enables iterating over kernels in the gate and
oracle harness.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-8


def _ema(scales: np.ndarray, momentum: float) -> np.ndarray:
    if momentum == 0.0:
        return scales
    out = np.empty_like(scales)
    s_prev = scales[0]
    out[0] = s_prev
    for t in range(1, len(scales)):
        s_prev = momentum * s_prev + (1.0 - momentum) * scales[t]
        out[t] = s_prev
    return out


def _frame_scale_baseline(d_pi3x, conf, d_moge, m):
    w = 1.0 / np.maximum(d_pi3x[m], EPS)            # inverse-depth weighting (paper)
    return float(np.sum(w * d_moge[m]) / (np.sum(w * d_pi3x[m]) + EPS))


def _frame_scale_conf(d_pi3x, conf, d_moge, m):
    w = conf[m] / np.maximum(d_pi3x[m], EPS)        # conf * inverse-depth
    return float(np.sum(w * d_moge[m]) / (np.sum(w * d_pi3x[m]) + EPS))


def _frame_scale_irls(d_pi3x, conf, d_moge, m, iters=5, c=1.345):
    ratio = d_moge[m] / np.maximum(d_pi3x[m], EPS)
    w = (conf[m] / np.maximum(d_pi3x[m], EPS)).astype(np.float64)
    s = float(np.median(ratio))
    for _ in range(iters):
        r = ratio - s
        sigma = np.median(np.abs(r - np.median(r))) * 1.4826 + EPS
        u = r / (c * sigma)
        huber = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), EPS))
        ww = w * huber
        s = float(np.sum(ww * ratio) / (np.sum(ww) + EPS))
    return s


def _frame_scale_geomedian(d_pi3x, conf, d_moge, m, iters=10):
    ratio = (d_moge[m] / np.maximum(d_pi3x[m], EPS)).astype(np.float64)
    w = (conf[m] / np.maximum(d_pi3x[m], EPS)).astype(np.float64)
    s = float(np.median(ratio))
    for _ in range(iters):                          # Weiszfeld, conf-weighted
        d = np.abs(ratio - s) + EPS
        ww = w / d
        s = float(np.sum(ww * ratio) / (np.sum(ww) + EPS))
    return s


KERNELS = {
    "baseline_ema": _frame_scale_baseline,
    "conf_weighted": _frame_scale_conf,
    "irls": _frame_scale_irls,
    "robust_geomedian": _frame_scale_geomedian,
}


def fuse(kernel, d_pi3x, conf, d_moge, mask, momentum=0.99):
    if kernel not in KERNELS:
        raise KeyError(f"unknown kernel {kernel}; choices {list(KERNELS)}")
    fn = KERNELS[kernel]
    T = d_pi3x.shape[0]
    raw = np.empty(T, np.float64)
    for t in range(T):
        m = mask[t] & (d_pi3x[t] > EPS) & (d_moge[t] > EPS)
        raw[t] = fn(d_pi3x[t], conf[t], d_moge[t], m) if m.any() else (raw[t - 1] if t else 1.0)
    scale = _ema(raw, momentum)
    fused = scale[:, None, None] * d_pi3x
    return fused.astype(np.float32), scale
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_fusion_kernels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/oracle/fusion_kernels.py tests/test_oracle_fusion_kernels.py
git commit -m "feat(oracle): add 4 CADF fusion kernels with unified fuse() API"
```

---

## Task 4: CADF gate statistics (spec §6 validate-or-kill)

**Files:**
- Create: `src/sana_wm_pipeline/oracle/cadf_gate.py`
- Test: `tests/test_oracle_cadf_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_cadf_gate.py
import numpy as np
from sana_wm_pipeline.oracle.cadf_gate import (
    partial_corr_conf_vs_error, scale_shape_decomposition,
)


def test_partial_corr_high_when_conf_independently_predicts_error():
    rng = np.random.default_rng(0)
    n = 5000
    inv_w = rng.uniform(0, 1, n)
    conf = rng.uniform(0, 1, n)                    # independent of inv_w
    error = 1.0 * (1 - conf) + 0.2 * rng.standard_normal(n)  # error driven by conf
    pc = partial_corr_conf_vs_error(conf, error, inv_w)
    assert pc < -0.5                                # higher conf -> lower error


def test_partial_corr_near_zero_when_conf_redundant_with_inv_w():
    rng = np.random.default_rng(1)
    n = 5000
    inv_w = rng.uniform(0, 1, n)
    conf = inv_w + 0.01 * rng.standard_normal(n)   # conf ~ inv_w (redundant)
    error = 1.0 * (1 - inv_w) + 0.2 * rng.standard_normal(n)
    pc = partial_corr_conf_vs_error(conf, error, inv_w)
    assert abs(pc) < 0.2                            # nothing left after controlling inv_w


def test_scale_share_is_one_for_pure_scale_error():
    rng = np.random.default_rng(2)
    d_pi3x = rng.uniform(1, 5, (2, 6, 6))
    d_gt = 2.0 * d_pi3x                             # only a global scale differs
    valid = np.ones_like(d_gt, bool)
    pred = 1.0 * d_pi3x                             # wrong scale (1.0 vs 2.0), right shape
    share = scale_shape_decomposition(pred, d_pi3x, d_gt, valid)
    assert share > 0.95


def test_scale_share_is_low_for_pure_shape_error():
    rng = np.random.default_rng(3)
    d_pi3x = rng.uniform(1, 5, (2, 6, 6))
    d_gt = d_pi3x + rng.uniform(-0.5, 0.5, d_pi3x.shape)  # shape noise, scale ~right
    valid = np.ones_like(d_gt, bool)
    share = scale_shape_decomposition(d_pi3x.copy(), d_pi3x, d_gt, valid)
    assert share < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_cadf_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: ... oracle.cadf_gate`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/oracle/cadf_gate.py
"""Spec §6 validate-or-kill gate for CADF.

Two decisive tests on existing dense GT (OmniWorld, TUM):
  1. partial_corr_conf_vs_error: does Pi3X conf predict GT-depth error BEYOND
     the inverse-depth weight already in the baseline? If ~0, conf is redundant.
  2. scale_shape_decomposition: what fraction of pose/depth error is attributable
     to scale (the only thing CADF can fix)? If small, CADF's ceiling is low.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-9


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return residual of y after OLS regression on x (with intercept)."""
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def partial_corr_conf_vs_error(conf, error, inv_w) -> float:
    """Pearson correlation of conf vs error after partialling out inv_w from both."""
    rc = _residualize(np.asarray(conf, float), np.asarray(inv_w, float))
    re = _residualize(np.asarray(error, float), np.asarray(inv_w, float))
    denom = (np.std(rc) * np.std(re)) + EPS
    return float(np.mean(rc * re) / denom)


def _abs_rel(pred, gt, valid) -> float:
    m = valid & (gt > 1e-6) & (pred > 1e-6)
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m])) if m.any() else float("nan")


def scale_shape_decomposition(pred, d_pi3x, d_gt, valid) -> float:
    """Fraction of pred error removable by the optimal per-frame global scale.

    share = (err_total - err_after_optimal_scale) / err_total, clipped to [0,1].
    High share => error is mostly scale => CADF can help. Low => shape error.
    """
    err_total = _abs_rel(pred, d_gt, valid)
    m = valid & (d_gt > 1e-6) & (d_pi3x > 1e-6)
    s_opt = np.median(d_gt[m] / d_pi3x[m])          # best single scale for the shape
    err_scaled = _abs_rel(s_opt * d_pi3x, d_gt, valid)
    if not np.isfinite(err_total) or err_total < EPS:
        return float("nan")
    return float(np.clip((err_total - err_scaled) / err_total, 0.0, 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_cadf_gate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/oracle/cadf_gate.py tests/test_oracle_cadf_gate.py
git commit -m "feat(oracle): add CADF validate-or-kill gate statistics"
```

---

## Task 5: Conf-preserving precompute cache (experiment driver for Module C)

**Files:**
- Create: `experiments/cadf_research/precompute_cadf_cache.py`
- Reference: `experiments/vipe_comparison/precompute_pi3x_depths.py:55-112` (run_pi3x / run_moge2 — note line 81 discards conf; we keep it)

This is a model-running driver (no unit test; verified by a real run). It must emit `d_pi3x`, `conf`, `d_moge`, `mask` so Tasks 6/9 can compare kernels.

- [ ] **Step 1: Write the driver**

```python
# experiments/cadf_research/precompute_cadf_cache.py
"""Precompute Pi3X (depth + CONF) and MoGe-2 metric depth into one npz.

Unlike vipe_comparison/precompute_pi3x_depths.py (which drops out["conf"]),
this keeps the confidence channel CADF needs. Saves:
  d_pi3x (T,H,W), conf (T,H,W) in 0..1, d_moge (T,H,W) metres, mask (T,H,W) bool
"""
import argparse, sys, numpy as np, torch
sys.path.insert(0, "experiments/vipe_comparison")
from precompute_pi3x_depths import read_video_frames, run_moge2  # reuse loaders


def run_pi3x_with_conf(frames, weights, chunk=16, stride=8, device="cuda"):
    from pi3 import Pi3X
    model = Pi3X.from_pretrained(weights).to(device).eval()
    T, H, W = frames.shape[0], frames.shape[2], frames.shape[3]
    d_acc = np.zeros((T, H, W), np.float32); c_acc = np.zeros((T, H, W), np.float32)
    n_acc = np.zeros((T,), np.float32)
    with torch.no_grad():
        for s in range(0, T, stride):
            e = min(s + chunk, T)
            x = torch.from_numpy(frames[s:e]).unsqueeze(0).to(device)
            out = model(x)
            d = out["local_points"][0, :, :, :, 2].cpu().numpy()
            c = out["conf"][0, :, :, 0].sigmoid().cpu().numpy()
            d_acc[s:e] += d; c_acc[s:e] += c; n_acc[s:e] += 1.0
            if e == T: break
    d_acc /= np.maximum(n_acc[:, None, None], 1.0)
    c_acc /= np.maximum(n_acc[:, None, None], 1.0)
    return d_acc, c_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pi3x-weights", default="/mnt/afs/davidwang/models/pi3x")
    ap.add_argument("--moge-weights", default="/mnt/afs/davidwang/models/moge2/model.pt")
    args = ap.parse_args()

    frames = read_video_frames(args.video)            # (T,3,H,W) float, /14-aligned
    d_pi3x, conf = run_pi3x_with_conf(frames, args.pi3x_weights)
    d_moge, mask = run_moge2(frames, args.moge_weights)  # (T,H,W), (T,H,W) bool
    np.savez_compressed(args.out, d_pi3x=d_pi3x, conf=conf, d_moge=d_moge, mask=mask)
    print(f"[cadf] saved {args.out}: T={d_pi3x.shape[0]} conf[{conf.min():.2f},{conf.max():.2f}]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on a real clip to verify shapes**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch HF_HOME=/mnt/afs/davidwang/cache/huggingface
mkdir -p experiments/cadf_research/cache
python experiments/cadf_research/precompute_cadf_cache.py \
  --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
  --out experiments/cadf_research/cache/fr1_desk.npz
```
Expected: prints `T=...` and `conf[0.xx,0.xx]` within [0,1]; file `fr1_desk.npz` created.

- [ ] **Step 3: Sanity-load the cache**

Run: `python -c "import numpy as np; d=np.load('experiments/cadf_research/cache/fr1_desk.npz'); print({k:d[k].shape for k in d})"`
Expected: four arrays `d_pi3x/conf/d_moge/mask` with identical `(T,H,W)`.

- [ ] **Step 4: Commit**

```bash
git add experiments/cadf_research/precompute_cadf_cache.py
git commit -m "feat(cadf): conf-preserving Pi3X+MoGe precompute cache"
```

> **Note:** OmniWorld provides dense GT depth needed for the gate. Reuse `experiments/data_production_smoke/prepare_omniworld.py` to get an OmniWorld clip's `video.mp4` + `gt_depth.npy`, then run this driver on it (`--out experiments/cadf_research/cache/omniworld_<id>.npz`).

---

## Task 6: Run the CADF gate and record go/no-go (spec §6 — hard prerequisite)

**Files:**
- Create: `experiments/cadf_research/run_gate.py`
- Output: `experiments/cadf_research/GATE_RESULT.md`

- [ ] **Step 1: Write the gate driver**

```python
# experiments/cadf_research/run_gate.py
"""Run spec §6 gate on cached clips that HAVE dense GT depth (OmniWorld/TUM).

Decision:
  KEEP CADF  if  partial_corr <= -0.15  AND  median scale_share >= 0.30
  else DROP CADF (paper leans on Module O characterization + Module Q filter).
"""
import argparse, numpy as np
from sana_wm_pipeline.oracle.cadf_gate import (
    partial_corr_conf_vs_error, scale_shape_decomposition)

EPS = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="npz with d_pi3x,conf,d_moge,mask")
    ap.add_argument("--gt-depth", required=True, help=".npy (T,H,W) metric GT depth")
    args = ap.parse_args()
    c = np.load(args.cache); gt = np.load(args.gt_depth)
    d_pi3x, conf, mask = c["d_pi3x"], c["conf"], c["mask"]
    valid = mask & (gt > 1e-6) & (d_pi3x > EPS)

    # Test 1: partial correlation of conf vs |relative depth error|, controlling inv-w
    inv_w = 1.0 / np.maximum(d_pi3x[valid], EPS)
    s_opt = np.median(gt[valid] / d_pi3x[valid])
    err = np.abs(s_opt * d_pi3x[valid] - gt[valid]) / gt[valid]
    pc = partial_corr_conf_vs_error(conf[valid], err, inv_w)

    # Test 2: per-frame scale share (use baseline-scaled pred vs gt)
    shares = []
    for t in range(d_pi3x.shape[0]):
        v = valid[t]
        if v.sum() < 50:
            continue
        s = np.median(gt[t][v] / d_pi3x[t][v])
        shares.append(scale_shape_decomposition(s * d_pi3x[t], d_pi3x[t], gt[t], valid[t]))
    med_share = float(np.nanmedian(shares))

    keep = (pc <= -0.15) and (med_share >= 0.30)
    lines = [
        "# CADF Gate Result",
        f"- partial_corr(conf, err | inv_w) = {pc:.3f}  (keep if <= -0.15)",
        f"- median scale_share            = {med_share:.3f}  (keep if >= 0.30)",
        f"- **DECISION: {'KEEP CADF' if keep else 'DROP CADF'}**",
    ]
    print("\n".join(lines))
    with open("experiments/cadf_research/GATE_RESULT.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate on an OmniWorld clip (dense GT)**

```bash
python experiments/cadf_research/run_gate.py \
  --cache experiments/cadf_research/cache/omniworld_020c2bed1dbb.npz \
  --gt-depth /mnt/afs/davidwang/workspace/data/omniworld/.../gt_depth.npy
```
Expected: prints both statistics and a `KEEP CADF` / `DROP CADF` decision; writes `GATE_RESULT.md`.

- [ ] **Step 3: Record the decision and branch the plan**

Read `GATE_RESULT.md`. **If DROP CADF:** skip Task 9's per-kernel matrix (run baseline_ema only as the reference annotator) and note it in the results doc. **If KEEP CADF:** proceed with all kernels in Task 9.

- [ ] **Step 4: Commit**

```bash
git add experiments/cadf_research/run_gate.py experiments/cadf_research/GATE_RESULT.md
git commit -m "feat(cadf): gate driver + recorded go/no-go decision"
```

---

## Task 7: Per-clip feature extractor (Module Q)

**Files:**
- Create: `src/sana_wm_pipeline/oracle/features.py`
- Test: `tests/test_oracle_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_features.py
import numpy as np
from sana_wm_pipeline.oracle.features import extract_features, FEATURE_NAMES


def test_feature_vector_length_and_names_match():
    T, H, W = 6, 8, 8
    rng = np.random.default_rng(0)
    feats = extract_features(
        d_pi3x=rng.uniform(0.5, 5, (T, H, W)),
        conf=rng.uniform(0, 1, (T, H, W)),
        d_moge=rng.uniform(0.5, 5, (T, H, W)),
        mask=np.ones((T, H, W), bool),
        scale_history=rng.uniform(0.9, 1.1, T),
        flow_score=0.42,
    )
    assert len(feats) == len(FEATURE_NAMES)
    assert all(np.isfinite(feats))


def test_scale_cv_feature_responds_to_drift():
    T, H, W = 6, 4, 4
    base = dict(d_pi3x=np.ones((T, H, W)), conf=np.ones((T, H, W)),
                d_moge=np.ones((T, H, W)), mask=np.ones((T, H, W), bool), flow_score=0.1)
    stable = extract_features(scale_history=np.ones(T), **base)
    drifting = extract_features(scale_history=np.linspace(1.0, 2.0, T), **base)
    i = FEATURE_NAMES.index("scale_cv")
    assert drifting[i] > stable[i]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_features.py -v`
Expected: FAIL with `ModuleNotFoundError: ... oracle.features`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/oracle/features.py
"""Per-clip features for the learned quality filter (Module Q).

Inputs are pipeline-internal signals available BEFORE we know GT error:
Pi3X conf stats, MoGe/Pi3X log-residual spread, EMA scale-history stability,
valid-mask fraction, and a UniMatch optical-flow score (bound as a feature,
not a hard gate). The regressor maps these -> predicted pose error.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = [
    "conf_mean", "conf_p10", "conf_p90", "frac_lowconf",
    "logres_std", "logres_iqr",
    "scale_cv", "scale_drift",
    "frac_valid", "flow_score",
]
EPS = 1e-8


def extract_features(d_pi3x, conf, d_moge, mask, scale_history, flow_score) -> np.ndarray:
    m = mask & (d_pi3x > EPS) & (d_moge > EPS)
    cf = conf[m] if m.any() else np.array([0.0])
    logres = np.log(np.maximum(d_moge[m] / np.maximum(d_pi3x[m], EPS), EPS)) if m.any() else np.array([0.0])
    sh = np.asarray(scale_history, float)
    feats = [
        float(np.mean(cf)),
        float(np.percentile(cf, 10)),
        float(np.percentile(cf, 90)),
        float(np.mean(cf < 0.5)),
        float(np.std(logres)),
        float(np.subtract(*np.percentile(logres, [75, 25]))),
        float(np.std(sh) / (np.mean(sh) + EPS)),                 # scale_cv
        float(abs(sh[-1] - sh[0]) / (np.mean(sh) + EPS)),        # scale_drift
        float(np.mean(m)),
        float(flow_score),
    ]
    return np.array(feats, np.float64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_features.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/oracle/features.py tests/test_oracle_features.py
git commit -m "feat(oracle): add per-clip feature extractor for quality filter"
```

---

## Task 8: Quality-error regressor (Module Q)

**Files:**
- Create: `src/sana_wm_pipeline/oracle/quality_model.py`
- Test: `tests/test_oracle_quality_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_oracle_quality_model.py
import numpy as np
from sana_wm_pipeline.oracle.quality_model import QualityFilter


def test_fit_predict_learns_monotone_signal():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, 10))
    y = X[:, 3] * 0.5 + 0.05 * rng.standard_normal(200)   # error driven by feature 3
    qf = QualityFilter().fit(X[:150], y[:150])
    pred = qf.predict(X[150:])
    # Spearman corr predicted vs true on held-out should be strongly positive
    from scipy.stats import spearmanr
    rho = spearmanr(pred, y[150:]).statistic
    assert rho > 0.6


def test_keep_mask_respects_error_budget():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (100, 10))
    y = X[:, 0]
    qf = QualityFilter().fit(X, y)
    keep = qf.keep_mask(X, max_pred_error=0.5)
    assert keep.dtype == bool and keep.shape == (100,)
    assert keep.sum() < 100                                # filters something
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_quality_model.py -v`
Expected: FAIL with `ModuleNotFoundError: ... oracle.quality_model`

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/oracle/quality_model.py
"""Learned pseudo-label quality filter (Module Q).

Regresses a clip's true pose error (e.g. ATE RMSE) from its feature vector.
Trained where GT error is known (Phase-0: TUM/OmniWorld; Phase-1: SimWorld),
then deployed to drop high-predicted-error real clips at Stage 04.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class QualityFilter:
    def __init__(self, **kw):
        self.model = GradientBoostingRegressor(
            n_estimators=kw.get("n_estimators", 200),
            max_depth=kw.get("max_depth", 3),
            learning_rate=kw.get("learning_rate", 0.05),
            random_state=kw.get("random_state", 0),
        )

    def fit(self, X, y):
        self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X))

    def keep_mask(self, X, max_pred_error: float):
        return self.predict(X) <= max_pred_error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_quality_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/oracle/quality_model.py tests/test_oracle_quality_model.py
git commit -m "feat(oracle): add learned quality-filter regressor"
```

---

## Task 9: Oracle harness — error tables + (feature, error) pairs (Module O integration)

**Files:**
- Create: `experiments/oracle_calibration/run_oracle.py`
- Output: `experiments/oracle_calibration/results/oracle_table.csv`, `dataset_pairs.npz`

Drives the full comparison on Phase-0 datasets and emits both the C1/C2 error table and the Module-Q training pairs. Uses Task 1–3 modules; honors the Task 6 gate.

- [ ] **Step 1: Write the harness**

```python
# experiments/oracle_calibration/run_oracle.py
"""Run kernels on cached clips that have GT, emit error table + (feature,error) pairs.

For each (clip, kernel): fuse -> metric-scale error & depth AbsRel/delta1 vs GT,
extract features, and (when GT poses exist) ATE RMSE via the existing evaluator.
Honors GATE_RESULT.md: if CADF dropped, only baseline_ema is run.
"""
import argparse, csv, glob, os, numpy as np
from sana_wm_pipeline.oracle.fusion_kernels import KERNELS, fuse
from sana_wm_pipeline.oracle.metrics import depth_errors, metric_scale_error
from sana_wm_pipeline.oracle.features import extract_features, FEATURE_NAMES


def _kernels_in_scope():
    p = "experiments/cadf_research/GATE_RESULT.md"
    if os.path.exists(p) and "DROP CADF" in open(p).read():
        return ["baseline_ema"]
    return list(KERNELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-glob", required=True, help="glob of npz caches")
    ap.add_argument("--gt-depth-dir", required=True, help="dir of <stem>.npy GT depth")
    ap.add_argument("--out-dir", default="experiments/oracle_calibration/results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    kernels = _kernels_in_scope()

    rows, feat_rows, err_rows = [], [], []
    for npz in sorted(glob.glob(args.cache_glob)):
        stem = os.path.splitext(os.path.basename(npz))[0]
        gt_path = os.path.join(args.gt_depth_dir, stem + ".npy")
        if not os.path.exists(gt_path):
            continue
        c = np.load(npz); gt = np.load(gt_path)
        valid = c["mask"] & (gt > 1e-6) & (c["d_pi3x"] > 1e-8)
        for k in kernels:
            fused, scale = fuse(k, c["d_pi3x"], c["conf"], c["d_moge"], c["mask"])
            de = depth_errors(fused, gt, valid)
            mse = metric_scale_error(fused, gt, valid)
            rows.append([stem, k, de["abs_rel"], de["delta1"], mse])
            if k == "baseline_ema":   # features/target from the reference annotator
                feat_rows.append(extract_features(
                    c["d_pi3x"], c["conf"], c["d_moge"], c["mask"], scale, flow_score=0.0))
                err_rows.append(mse)   # Phase-0 proxy target; ATE added when GT poses present

    with open(os.path.join(args.out_dir, "oracle_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip", "kernel", "abs_rel", "delta1", "scale_err"])
        w.writerows(rows)
    np.savez(os.path.join(args.out_dir, "dataset_pairs.npz"),
             X=np.array(feat_rows), y=np.array(err_rows), names=np.array(FEATURE_NAMES))
    print(f"[oracle] {len(rows)} rows; kernels={kernels}; pairs={len(err_rows)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on cached Phase-0 clips**

```bash
python experiments/oracle_calibration/run_oracle.py \
  --cache-glob 'experiments/cadf_research/cache/*.npz' \
  --gt-depth-dir experiments/cadf_research/gt_depth
```
Expected: prints row/pair counts; writes `oracle_table.csv` and `dataset_pairs.npz`.

- [ ] **Step 3: Eyeball the table**

Run: `column -s, -t experiments/oracle_calibration/results/oracle_table.csv | head`
Expected: per-clip per-kernel `abs_rel / delta1 / scale_err`; if CADF kept, conf kernels should not be worse than `baseline_ema` on dynamic/reflective clips.

- [ ] **Step 4: Commit**

```bash
git add experiments/oracle_calibration/run_oracle.py experiments/oracle_calibration/results/
git commit -m "feat(oracle): harness emitting error table + Q training pairs"
```

---

## Task 10: Train + evaluate the quality filter; wire a Stage-04 hook (Module Q)

**Files:**
- Create: `experiments/oracle_calibration/train_filter.py`
- Modify: `src/sana_wm_pipeline/stage04_filter/apply_table6.py` (add optional learned-filter path)
- Test: `tests/test_oracle_stage04_hook.py`

- [ ] **Step 1: Write the failing test for the Stage-04 hook**

```python
# tests/test_oracle_stage04_hook.py
import numpy as np
from sana_wm_pipeline.stage04_filter.apply_table6 import learned_filter_decision


def test_learned_filter_passes_low_predicted_error():
    from sana_wm_pipeline.oracle.quality_model import QualityFilter
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (50, 10)); y = X[:, 0]
    qf = QualityFilter().fit(X, y)
    good = np.zeros((1, 10)); bad = np.ones((1, 10))
    assert learned_filter_decision(qf, good[0], max_pred_error=0.5) is True
    assert learned_filter_decision(qf, bad[0], max_pred_error=0.5) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_oracle_stage04_hook.py -v`
Expected: FAIL with `ImportError: cannot import name 'learned_filter_decision'`

- [ ] **Step 3: Add the hook to Stage 04**

Append to `src/sana_wm_pipeline/stage04_filter/apply_table6.py`:

```python
def learned_filter_decision(quality_filter, feature_vec, max_pred_error: float) -> bool:
    """Module-Q learned gate: keep clip iff predicted pose error <= budget.

    Complements (does not replace) the Table-6 heuristics; callers may AND them.
    """
    import numpy as np
    pred = float(quality_filter.predict(np.asarray(feature_vec)[None, :])[0])
    return pred <= max_pred_error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_oracle_stage04_hook.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the training/eval driver**

```python
# experiments/oracle_calibration/train_filter.py
"""Train Module Q on oracle pairs; report held-out Spearman + filter precision."""
import argparse, numpy as np
from scipy.stats import spearmanr
from sana_wm_pipeline.oracle.quality_model import QualityFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="experiments/oracle_calibration/results/dataset_pairs.npz")
    ap.add_argument("--out", default="experiments/oracle_calibration/results/quality_filter.npz")
    args = ap.parse_args()
    d = np.load(args.pairs, allow_pickle=True); X, y = d["X"], d["y"]
    n = len(y); k = max(1, n // 5)
    idx = np.random.default_rng(0).permutation(n)
    te, tr = idx[:k], idx[k:]
    qf = QualityFilter().fit(X[tr], y[tr])
    pred = qf.predict(X[te])
    rho = spearmanr(pred, y[te]).statistic if k > 1 else float("nan")
    print(f"[filter] n={n} holdout={k} Spearman(pred,true)={rho:.3f}")
    import pickle
    with open(args.out.replace(".npz", ".pkl"), "wb") as f:
        pickle.dump(qf, f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the trainer**

```bash
python experiments/oracle_calibration/train_filter.py
```
Expected: prints a Spearman correlation; writes `quality_filter.pkl`. (With few Phase-0 clips this is a smoke check; the real number lands once Phase-1 SimWorld adds hundreds of GT-labelled clips.)

- [ ] **Step 7: Commit**

```bash
git add experiments/oracle_calibration/train_filter.py src/sana_wm_pipeline/stage04_filter/apply_table6.py tests/test_oracle_stage04_hook.py
git commit -m "feat(oracle): train quality filter + Stage-04 learned-filter hook"
```

---

## Task 11: Full test sweep + results writeup

**Files:**
- Create: `experiments/oracle_calibration/RESULTS_phase0.md`

- [ ] **Step 1: Run the whole suite (regression guard)**

Run: `python -m pytest tests/ -q`
Expected: all prior tests still pass (was 141) plus the new oracle tests; 0 failures.

- [ ] **Step 2: Write the results doc**

Create `experiments/oracle_calibration/RESULTS_phase0.md` summarizing: the gate decision (from `GATE_RESULT.md`), the `oracle_table.csv` per-kernel depth/scale errors, and the filter Spearman. State explicitly whether CADF was kept or dropped and what Phase 1 (SimWorld factor grid) will add.

- [ ] **Step 3: Commit**

```bash
git add experiments/oracle_calibration/RESULTS_phase0.md
git commit -m "docs(oracle): Phase-0 results summary"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- C1 error characterization → Tasks 2, 9 (metrics + harness table). Full factorial grid is Phase-1 (Module S, deferred plan) — Phase 0 establishes the metric + harness it will feed.
- C2 CADF → Tasks 3, 4, 5, 6, 9 (kernels + cache + gate + scored matrix).
- C3 learned filter → Tasks 7, 8, 10 (features + model + Stage-04 hook).
- Spec §6 gate → Tasks 4, 6 (statistics + driver + recorded decision, hard prerequisite honored in Task 9 via `_kernels_in_scope`).
- Scope decision A (drop FCGS/DiFix3D; UniMatch/DOVER as features; defer Qwen) → encoded: no Stage-03 tasks; `flow_score` feature slot for UniMatch; no captioning in Phase 0.
- Coordinate conversion / Module S / Module D → correctly **out of scope** for this Phase-0 plan (separate plans).

**Placeholder scan:** every code step contains complete code; experiment steps give exact commands + expected output. `flow_score` is wired as `0.0` in Task 9 with UniMatch binding noted as a Phase-0.5 follow-up (does not block the pipeline).

**Type consistency:** `fuse(kernel, d_pi3x, conf, d_moge, mask, momentum)` signature identical across Tasks 3/6/9; `FEATURE_NAMES`/`extract_features` consistent across Tasks 7/9; `QualityFilter.predict/keep_mask` consistent across Tasks 8/10; `GTScene` fields stable.

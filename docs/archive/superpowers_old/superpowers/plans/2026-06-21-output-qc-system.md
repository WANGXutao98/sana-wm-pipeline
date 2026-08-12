# Output QC System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SANA-WM 管线在 CMCC 批量生产后输出的 ~20 万条样本，构建一套两阶段并行质检系统，自动产出 pass/reject/human_review 三份清单 + per-sample QC JSON + HTML 可视化报告，并支持 per-group 差异化阈值策略。

**Architecture:** Stage 1 对全量样本做 CPU-only numpy/json 快速扫描（~1.5 分钟，32 核），写 `stage1_results.jsonl`；Stage 2 仅对 Stage 1 标记为 `flag` 的样本 + 每 group 随机抽 5% 做视频帧数（PyAV）+ 深度轨迹分析，写 `stage2_results.jsonl`；Report 阶段合并两阶段结果，输出三份 manifest + HTML 报告。所有输入来自 tar shard（生产格式），per-group 阈值通过 `GroupConfig` 注册表配置，游戏类与真实世界类数据集差异化处理。

**Tech Stack:** Python 3.10+, numpy, av（PyAV，已在 deps），tarfile（标准库），multiprocessing（标准库），pandas（已在 deps），pytest, 既有的 `pose_quality.py`

## Global Constraints

- Python ≥ 3.10，所有代码加 `from __future__ import annotations`
- 不引入新的 pip 依赖（使用 numpy/av/pandas/tarfile/multiprocessing，均已在 pyproject.toml）
- 所有纯计算函数放在 `metrics.py`，无 IO 副作用
- 多进程用 `multiprocessing.Pool`（spawn 模式），worker 函数必须 top-level 可 pickle
- 生产默认：`strict_frames=False`（与 `run_worker.py:164` 一致，不强制 T=961）
- 日志用 `rich.console.Console`（已在 deps），JSON 输出用标准库 `json`
- 所有输出写到 `--output-dir`（默认 `./qc_output/`）
- pytest 测试路径在 `tests/`，`pythonpath = ["src"]`（pyproject.toml 已配置）
- ffprobe 路径通过 `conftest.py` 的 `.bin/` 注入，Stage 2 用 `av` 代替 ffprobe subprocess 读帧数

---

## File Map

```
src/sana_wm_pipeline/qc/
  __init__.py                   新建：package init，re-export 关键符号
  metrics.py                    新建：纯计算函数（SO3、轨迹、内参、caption），无 IO
  group_config.py               新建：GroupConfig dataclass + 分 group 阈值注册表
  stage1_fast.py                新建：Stage 1 全量扫描（读 tar shard，multiprocessing Pool）
  stage2_deep.py                新建：Stage 2 深度检测（av 帧数、冻结轨迹、caption 去重）
  report.py                     新建：合并结果 + 写三份 manifest + 生成 HTML 报告

scripts/run_qc.py               新建：CLI 入口（argparse），串行调 Stage1→Stage2→Report

tests/test_qc_metrics.py        新建：metrics.py 单元测试
tests/test_qc_group_config.py   新建：group_config.py 单元测试
tests/test_qc_stage1.py         新建：Stage 1 集成测试（用 testdata 合成 tar）
tests/test_qc_stage2.py         新建：Stage 2 集成测试（用 testdata 合成 tar）
tests/test_qc_report.py         新建：report 生成测试

已有文件（只读，不改动）：
  src/sana_wm_pipeline/stage02_pose/pose_quality.py   被 metrics.py import 复用
  src/sana_wm_pipeline/stage06_pack/schema.py          被 stage1_fast.py import 复用
  tests/conftest.py                                    已有 ffmpeg PATH 注入，不改动
```

---

## Task 1: `metrics.py` — 纯计算函数库

**Files:**
- Create: `src/sana_wm_pipeline/qc/metrics.py`
- Create: `src/sana_wm_pipeline/qc/__init__.py`
- Test: `tests/test_qc_metrics.py`

**Interfaces:**
- Produces:
  - `check_so3(poses_c2w) -> tuple[float, float, float]` → `(det_mean, det_std, orth_err_max)`
  - `check_first_frame(poses_c2w, atol) -> tuple[bool, float]` → `(ok, max_dev)`
  - `check_trajectory(poses_c2w, jump_threshold_m) -> tuple[float, float, float, int]` → `(traj_total_m, step_mean_m, step_max_m, n_jumps)`
  - `check_no_nan_inf(arrays) -> tuple[bool, list[str]]` → `(ok, reasons)`
  - `check_caption(caption, min_len) -> tuple[bool, int]` → `(ok, length)`
  - `compute_stage1_metrics(poses, intrinsics, scale, caption, meta_T, image_wh, jump_threshold_m, min_caption_len) -> dict`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_metrics.py
from __future__ import annotations
import numpy as np
import pytest
from sana_wm_pipeline.qc.metrics import (
    check_so3, check_first_frame, check_trajectory,
    check_no_nan_inf, check_caption, compute_stage1_metrics,
)


def _make_identity_poses(T: int) -> np.ndarray:
    """Return T identity 4x4 matrices as float32."""
    return np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))


def _make_intrinsics(T: int, fx: float = 700.0) -> np.ndarray:
    """Return (T,1,4) intrinsics with perfect principal point for 1280x720."""
    intr = np.zeros((T, 1, 4), dtype=np.float32)
    intr[:, 0, 0] = fx   # fx
    intr[:, 0, 1] = fx   # fy
    intr[:, 0, 2] = 640  # cx
    intr[:, 0, 3] = 360  # cy
    return intr


# --- check_so3 ---

def test_so3_identity_poses():
    poses = _make_identity_poses(10)
    det_mean, det_std, orth_err = check_so3(poses)
    assert abs(det_mean - 1.0) < 1e-6
    assert det_std < 1e-6
    assert orth_err < 1e-6


def test_so3_degenerate_rotation_detected():
    poses = _make_identity_poses(5)
    poses[:, :3, :3] = 0.0   # all-zero rotation → det=0
    det_mean, det_std, orth_err = check_so3(poses)
    assert abs(det_mean) < 1e-6


# --- check_first_frame ---

def test_first_frame_identity_ok():
    poses = _make_identity_poses(10)
    ok, dev = check_first_frame(poses, atol=1e-3)
    assert ok
    assert dev < 1e-6


def test_first_frame_shifted_fails():
    poses = _make_identity_poses(10)
    poses[0, 0, 3] = 1.0   # shift first frame
    ok, dev = check_first_frame(poses, atol=1e-3)
    assert not ok
    assert abs(dev - 1.0) < 1e-6


# --- check_trajectory ---

def test_trajectory_linear_motion():
    T = 100
    poses = _make_identity_poses(T)
    poses[:, 0, 3] = np.arange(T, dtype=np.float32) * 0.1   # 0.1m/frame
    traj_total, step_mean, step_max, n_jumps = check_trajectory(poses, jump_threshold_m=0.5)
    assert abs(traj_total - (T - 1) * 0.1) < 1e-4
    assert abs(step_mean - 0.1) < 1e-4
    assert abs(step_max - 0.1) < 1e-4
    assert n_jumps == 0


def test_trajectory_counts_jumps():
    T = 10
    poses = _make_identity_poses(T)
    poses[5, 0, 3] = 5.0   # one big jump at frame 5
    _, _, _, n_jumps = check_trajectory(poses, jump_threshold_m=0.5)
    assert n_jumps == 1


# --- check_no_nan_inf ---

def test_no_nan_inf_clean_arrays():
    a = np.ones((5, 4, 4), dtype=np.float32)
    b = np.ones((5, 1, 4), dtype=np.float32)
    ok, reasons = check_no_nan_inf({"poses": a, "intrinsics": b})
    assert ok
    assert reasons == []


def test_no_nan_inf_detects_nan():
    a = np.ones((5, 4, 4), dtype=np.float32)
    a[2, 1, 1] = float("nan")
    ok, reasons = check_no_nan_inf({"poses": a})
    assert not ok
    assert any("poses" in r for r in reasons)


def test_no_nan_inf_detects_inf():
    a = np.ones((5,), dtype=np.float32)
    a[0] = float("inf")
    ok, reasons = check_no_nan_inf({"scale": a})
    assert not ok


# --- check_caption ---

def test_caption_ok():
    ok, length = check_caption("A " + "x" * 60, min_len=50)
    assert ok
    assert length >= 50


def test_caption_too_short():
    ok, length = check_caption("hi", min_len=50)
    assert not ok


def test_caption_placeholder_rejected():
    for s in ["n/a", "N/A", "none", "no caption", ""]:
        ok, _ = check_caption(s, min_len=50)
        assert not ok, f"should reject placeholder: {repr(s)}"


# --- compute_stage1_metrics ---

def test_compute_stage1_metrics_pass():
    T = 100
    poses = _make_identity_poses(T)
    poses[:, 0, 3] = np.arange(T, dtype=np.float32) * 0.1
    intr = _make_intrinsics(T)
    scale = np.ones(T, dtype=np.float32)
    caption = "A " + "x" * 100
    result = compute_stage1_metrics(
        poses=poses, intrinsics=intr, scale=scale,
        caption=caption, meta_T=T,
        image_wh=(1280, 720),
        jump_threshold_m=0.5,
        min_caption_len=50,
    )
    assert result["t_aligned"] is True
    assert result["no_nan_inf"] is True
    assert result["so3_valid"] is True
    assert result["first_frame_ok"] is True
    assert result["caption_ok"] is True
    assert result["n_jumps"] == 0
    assert result["T"] == T


def test_compute_stage1_metrics_t_mismatch():
    T = 10
    poses = _make_identity_poses(T)
    intr = _make_intrinsics(T)
    scale = np.ones(T, dtype=np.float32)
    result = compute_stage1_metrics(
        poses=poses, intrinsics=intr, scale=scale,
        caption="A " * 30, meta_T=T + 1,  # mismatch
        image_wh=(1280, 720),
        jump_threshold_m=0.5,
        min_caption_len=50,
    )
    assert result["t_aligned"] is False
    assert "t_mismatch" in result["reasons"]
```

- [ ] **Step 2: 运行，确认测试失败**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_qc_metrics.py -v 2>&1 | head -20
```
预期：`ModuleNotFoundError: No module named 'sana_wm_pipeline.qc'`

- [ ] **Step 3: 实现 `__init__.py` 和 `metrics.py`**

**`src/sana_wm_pipeline/qc/__init__.py`:**
```python
from __future__ import annotations
```

**`src/sana_wm_pipeline/qc/metrics.py`:**
```python
"""Pure computation functions for Stage 1 QC metrics. No IO."""
from __future__ import annotations
import numpy as np
from sana_wm_pipeline.stage02_pose.pose_quality import (
    evaluate_pose_quality, horizontal_fov_deg, vertical_fov_deg,
)

_SO3_DET_ATOL = 1e-3      # |det(R) - 1| > this → SO3 fail
_SO3_ORTH_ATOL = 1e-3     # orth_err > this → SO3 fail
_FIRST_FRAME_ATOL = 1e-2  # first-frame deviation > this → fail


def check_so3(poses_c2w: np.ndarray) -> tuple[float, float, float]:
    """SO(3) validity check.

    Returns:
        (det_mean, det_std, orth_err_max)
    Caller decides pass/fail via thresholds.
    """
    R = poses_c2w[:, :3, :3].astype(np.float64)
    dets = np.linalg.det(R)
    orth_err = float(np.max(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3))))
    return float(dets.mean()), float(dets.std()), orth_err


def check_first_frame(poses_c2w: np.ndarray, atol: float = _FIRST_FRAME_ATOL) -> tuple[bool, float]:
    """Check poses_c2w[0] ≈ identity (paper App. D.3 first-frame anchor).

    Returns:
        (ok, max_deviation)
    """
    dev = float(np.max(np.abs(poses_c2w[0].astype(np.float64) - np.eye(4))))
    return dev <= atol, dev


def check_trajectory(
    poses_c2w: np.ndarray, jump_threshold_m: float
) -> tuple[float, float, float, int]:
    """Analyse per-frame translation steps.

    Returns:
        (traj_total_m, step_mean_m, step_max_m, n_jumps)
    n_jumps = count of consecutive steps > jump_threshold_m.
    """
    t = poses_c2w[:, :3, 3].astype(np.float64)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    if steps.size == 0:
        return 0.0, 0.0, 0.0, 0
    n_jumps = int((steps > jump_threshold_m).sum())
    return float(steps.sum()), float(steps.mean()), float(steps.max()), n_jumps


def check_no_nan_inf(arrays: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    """Check that no array contains NaN or Inf.

    Args:
        arrays: mapping of name → numpy array for error messages.

    Returns:
        (ok, reasons) — reasons is empty when ok=True.
    """
    reasons: list[str] = []
    for name, arr in arrays.items():
        a = np.asarray(arr)
        if not np.isfinite(a).all():
            n_bad = int((~np.isfinite(a)).sum())
            reasons.append(f"{name}: {n_bad} non-finite values")
    return len(reasons) == 0, reasons


def check_caption(caption: str, min_len: int = 50) -> tuple[bool, int]:
    """Check caption is non-empty, not a placeholder, and meets minimum length.

    Returns:
        (ok, length_after_strip)
    """
    stripped = caption.strip()
    _PLACEHOLDERS = {"", "n/a", "none", "no caption", "null", "tbd"}
    if stripped.lower() in _PLACEHOLDERS:
        return False, len(stripped)
    return len(stripped) >= min_len, len(stripped)


def compute_stage1_metrics(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    scale: np.ndarray,
    caption: str,
    meta_T: int,
    image_wh: tuple[int, int],
    jump_threshold_m: float,
    min_caption_len: int,
) -> dict:
    """Run all Stage 1 checks and return a flat metrics dict.

    All inputs assumed already loaded from numpy (caller handles IO).
    Returns dict suitable for JSON serialisation (all values are Python scalars or lists).

    Keys:
        T, t_aligned, no_nan_inf, so3_valid, det_R_mean, det_R_std, orth_err_max,
        first_frame_ok, first_frame_dev, traj_total_m, step_mean_m, step_max_m,
        n_jumps, jump_threshold_m, fov_x_min, fov_x_max, fov_y_min, fov_y_max,
        fov_ok, focal_div_max, focal_div_ok, fx_mean, cy_mean, cx_mean,
        caption_ok, caption_len, scale_all_ones, reasons
    """
    T = int(poses.shape[0])
    reasons: list[str] = []

    # T alignment across all arrays
    t_aligned = (poses.shape[0] == T and
                 intrinsics.shape[0] == T and
                 scale.shape[0] == T and
                 meta_T == T)
    if not t_aligned:
        reasons.append(
            f"t_mismatch: poses={poses.shape[0]} intr={intrinsics.shape[0]} "
            f"scale={scale.shape[0]} meta={meta_T}"
        )

    # NaN/Inf
    nan_ok, nan_reasons = check_no_nan_inf(
        {"poses": poses, "intrinsics": intrinsics, "scale": scale}
    )
    if not nan_ok:
        reasons.extend(nan_reasons)

    # SO(3)
    det_mean, det_std, orth_err = check_so3(poses)
    so3_valid = abs(det_mean - 1.0) <= _SO3_DET_ATOL and orth_err <= _SO3_ORTH_ATOL
    if not so3_valid:
        reasons.append(f"so3_invalid: det_mean={det_mean:.6f} orth_err={orth_err:.2e}")

    # First-frame anchor
    first_ok, first_dev = check_first_frame(poses)
    if not first_ok:
        reasons.append(f"first_frame_dev={first_dev:.4f} > {_FIRST_FRAME_ATOL}")

    # Trajectory
    traj_total, step_mean, step_max, n_jumps = check_trajectory(poses, jump_threshold_m)

    # Intrinsics (reuse pose_quality.py)
    W, H = image_wh
    pqr = evaluate_pose_quality(intrinsics, image_wh, scale)
    if not pqr.passed:
        reasons.extend(list(pqr.reasons))

    fx_vals = intrinsics[:, 0, 0].astype(np.float64)
    cx_mean = float(intrinsics[:, 0, 2].mean())
    cy_mean = float(intrinsics[:, 0, 3].mean())

    # Caption
    cap_ok, cap_len = check_caption(caption, min_len=min_caption_len)
    if not cap_ok:
        reasons.append(f"caption_len={cap_len} < {min_caption_len} or placeholder")

    # Scale (Default mode = all 1.0)
    scale_all_ones = bool(np.all(scale == 1.0))

    return {
        "T": T,
        "t_aligned": t_aligned,
        "no_nan_inf": nan_ok,
        "so3_valid": so3_valid,
        "det_R_mean": round(det_mean, 8),
        "det_R_std": round(det_std, 8),
        "orth_err_max": float(f"{orth_err:.3e}"),
        "first_frame_ok": first_ok,
        "first_frame_dev": round(first_dev, 6),
        "traj_total_m": round(traj_total, 3),
        "step_mean_m": round(step_mean, 4),
        "step_max_m": round(step_max, 4),
        "n_jumps": n_jumps,
        "jump_threshold_m": jump_threshold_m,
        "fov_x_min": round(pqr.fov_x_min, 2),
        "fov_x_max": round(pqr.fov_x_max, 2),
        "fov_y_min": round(pqr.fov_y_min, 2),
        "fov_y_max": round(pqr.fov_y_max, 2),
        "fov_ok": pqr.passed or not any("fov" in r for r in pqr.reasons),
        "focal_div_max": round(pqr.focal_divergence_max, 4),
        "focal_div_ok": pqr.focal_divergence_max <= 0.20,
        "fx_mean": round(float(fx_vals.mean()), 2),
        "cx_mean": round(cx_mean, 2),
        "cy_mean": round(cy_mean, 2),
        "caption_ok": cap_ok,
        "caption_len": cap_len,
        "scale_all_ones": scale_all_ones,
        "reasons": reasons,
    }
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_qc_metrics.py -v
```
预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/qc/__init__.py src/sana_wm_pipeline/qc/metrics.py tests/test_qc_metrics.py
git commit -m "feat(qc): add Stage 1 pure metrics library"
```

---

## Task 2: `group_config.py` — Per-Group 阈值注册表

**Files:**
- Create: `src/sana_wm_pipeline/qc/group_config.py`
- Test: `tests/test_qc_group_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `GroupConfig` dataclass（所有阈值字段）
  - `get_group_config(group_name: str) -> GroupConfig`（fallback 到 `_DEFAULT`）
  - `VERDICT_FAIL`, `VERDICT_FLAG`, `VERDICT_PASS` 常量字符串
  - `compute_verdict(metrics: dict, cfg: GroupConfig) -> tuple[str, list[str]]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_group_config.py
from __future__ import annotations
import pytest
from sana_wm_pipeline.qc.group_config import (
    GroupConfig, get_group_config, compute_verdict,
    VERDICT_FAIL, VERDICT_FLAG, VERDICT_PASS,
)


def _clean_metrics(T: int = 100) -> dict:
    """Minimal all-passing metrics dict."""
    return {
        "T": T,
        "t_aligned": True,
        "no_nan_inf": True,
        "so3_valid": True,
        "first_frame_ok": True,
        "fov_ok": True,
        "focal_div_ok": True,
        "caption_ok": True,
        "n_jumps": 0,
        "reasons": [],
    }


# --- GroupConfig basics ---

def test_default_config_exists():
    cfg = get_group_config("unknown-group-xyz")
    assert isinstance(cfg, GroupConfig)


def test_known_groups_have_configs():
    for g in ["wds-DL3DV-ALL-2K", "wds-sekai-real-walking-hq",
              "wds-OmniWorld-Game", "wds-SpatialVID-hq"]:
        cfg = get_group_config(g)
        assert isinstance(cfg, GroupConfig)


def test_game_group_more_lenient_than_real():
    game_cfg = get_group_config("wds-OmniWorld-Game")
    real_cfg = get_group_config("wds-DL3DV-ALL-2K")
    assert game_cfg.max_jumps_fail > real_cfg.max_jumps_fail
    assert game_cfg.jump_threshold_m >= real_cfg.jump_threshold_m


# --- compute_verdict ---

def test_verdict_pass_clean_sample():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean_metrics()
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_PASS
    assert reasons == []


def test_verdict_fail_on_structural_error():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean_metrics()
    m["so3_valid"] = False
    m["reasons"] = ["so3_invalid: det_mean=0.5"]
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_FAIL


def test_verdict_fail_on_nan():
    cfg = get_group_config("wds-sekai-real-walking-hq")
    m = _clean_metrics()
    m["no_nan_inf"] = False
    m["reasons"] = ["poses: 3 non-finite values"]
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_FAIL


def test_verdict_flag_on_excess_jumps():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean_metrics()
    # DL3DV max_jumps_flag=0 → even 1 jump → flag
    m["n_jumps"] = 1
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_FLAG
    assert any("jump" in r for r in reasons)


def test_verdict_fail_on_too_many_jumps():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean_metrics()
    m["n_jumps"] = cfg.max_jumps_fail + 1
    verdict, _ = compute_verdict(m, cfg)
    assert verdict == VERDICT_FAIL


def test_game_data_tolerates_moderate_jumps():
    cfg = get_group_config("wds-OmniWorld-Game")
    m = _clean_metrics()
    m["n_jumps"] = 10   # within game tolerance
    verdict, _ = compute_verdict(m, cfg)
    assert verdict == VERDICT_PASS


def test_verdict_flag_on_short_caption():
    cfg = get_group_config("wds-sekai-real-walking-hq")
    m = _clean_metrics()
    m["caption_ok"] = False
    m["reasons"] = ["caption_len=20 < 50 or placeholder"]
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_FLAG
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_qc_group_config.py -v 2>&1 | head -10
```
预期：`ImportError`

- [ ] **Step 3: 实现 `group_config.py`**

```python
# src/sana_wm_pipeline/qc/group_config.py
"""Per-group QC threshold configuration and verdict logic."""
from __future__ import annotations
from dataclasses import dataclass

VERDICT_PASS = "pass"
VERDICT_FLAG = "flag"    # borderline → human review queue
VERDICT_FAIL = "fail"

# Hard-fail structural checks: any True → FAIL regardless of group
_HARD_FAIL_KEYS = ("t_aligned", "no_nan_inf", "so3_valid", "first_frame_ok")
# Soft-fail checks: False → FLAG (not FAIL)
_SOFT_FLAG_KEYS = ("caption_ok",)


@dataclass(frozen=True)
class GroupConfig:
    jump_threshold_m: float    # step > this counted as a jump
    max_jumps_flag: int        # n_jumps > this → FLAG (if < max_jumps_fail)
    max_jumps_fail: int        # n_jumps > this → FAIL
    min_caption_len: int       # checked inside compute_stage1_metrics; used for reference


_REAL_STRICT = GroupConfig(
    jump_threshold_m=0.5,
    max_jumps_flag=0,
    max_jumps_fail=5,
    min_caption_len=50,
)
_SEKAI = GroupConfig(
    jump_threshold_m=0.5,
    max_jumps_flag=3,
    max_jumps_fail=15,
    min_caption_len=50,
)
_GAME_RELAXED = GroupConfig(
    jump_threshold_m=2.0,
    max_jumps_flag=20,
    max_jumps_fail=100,
    min_caption_len=50,
)
_DEFAULT = GroupConfig(
    jump_threshold_m=1.0,
    max_jumps_flag=10,
    max_jumps_fail=50,
    min_caption_len=50,
)

_REGISTRY: dict[str, GroupConfig] = {
    "wds-DL3DV-ALL-2K": _REAL_STRICT,
    "wds-sekai-real-walking-hq": _SEKAI,
    "wds-OmniWorld-Game": _GAME_RELAXED,
    "wds-SpatialVID-hq": _REAL_STRICT,
}


def get_group_config(group_name: str) -> GroupConfig:
    """Return per-group thresholds; falls back to _DEFAULT for unknown groups."""
    return _REGISTRY.get(group_name, _DEFAULT)


def compute_verdict(metrics: dict, cfg: GroupConfig) -> tuple[str, list[str]]:
    """Classify a sample as PASS / FLAG / FAIL using per-group thresholds.

    Args:
        metrics: output dict from compute_stage1_metrics().
        cfg: GroupConfig for this sample's group.

    Returns:
        (verdict, flag_reasons) — flag_reasons lists human-readable causes.
    """
    flag_reasons: list[str] = []

    # Hard structural failures → FAIL immediately
    for key in _HARD_FAIL_KEYS:
        if not metrics.get(key, True):
            return VERDICT_FAIL, list(metrics.get("reasons", []))

    if not metrics.get("fov_ok", True) or not metrics.get("focal_div_ok", True):
        return VERDICT_FAIL, list(metrics.get("reasons", []))

    # Jump count (per-group logic)
    n_jumps = metrics.get("n_jumps", 0)
    if n_jumps > cfg.max_jumps_fail:
        return VERDICT_FAIL, [f"n_jumps={n_jumps} > max_jumps_fail={cfg.max_jumps_fail}"]
    if n_jumps > cfg.max_jumps_flag:
        flag_reasons.append(f"n_jumps={n_jumps} > max_jumps_flag={cfg.max_jumps_flag}")

    # Soft flags
    for key in _SOFT_FLAG_KEYS:
        if not metrics.get(key, True):
            flag_reasons.extend(metrics.get("reasons", []))

    if flag_reasons:
        return VERDICT_FLAG, flag_reasons
    return VERDICT_PASS, []
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_qc_group_config.py -v
```
预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/qc/group_config.py tests/test_qc_group_config.py
git commit -m "feat(qc): add per-group threshold config and verdict logic"
```

---

## Task 3: `stage1_fast.py` — Stage 1 全量快速扫描

**Files:**
- Create: `src/sana_wm_pipeline/qc/stage1_fast.py`
- Test: `tests/test_qc_stage1.py`

**Interfaces:**
- Consumes: `metrics.compute_stage1_metrics`, `group_config.get_group_config`, `group_config.compute_verdict`
- Produces:
  - `scan_tar(tar_path, group_name) -> list[dict]`：扫描单个 tar shard，返回 per-sample 结果 list
  - `run_stage1(tar_paths, group_name, output_jsonl, n_workers) -> int`：并行扫描，写 jsonl，返回总样本数

**Stage 1 输出 JSONL 格式（每行一个样本）：**
```json
{
  "sample_id": "OmniWorld-Game_1f79eb96f021__splits_013-015",
  "group": "wds-OmniWorld-Game",
  "tar_path": "/path/to/shard-000000.tar",
  "verdict": "pass",
  "flag_reasons": [],
  "metrics": { ...compute_stage1_metrics 的返回值... }
}
```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_stage1.py
"""Stage 1 integration tests using synthetic tars built from testdata/."""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage1_fast import scan_tar, run_stage1

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory) -> Path:
    """Pack one testdata sample into a synthetic tar shard."""
    tar_path = tmp_path_factory.mktemp("qc_stage1") / "shard-000000.tar"
    exts = [".mp4", ".poses_c2w.npy", ".intrinsics.npy",
            ".scale.npy", ".caption.txt", ".meta.json"]
    with tarfile.open(tar_path, "w") as tf:
        for ext in exts:
            src = TESTDATA / (SAMPLE_ID + ext)
            tf.add(src, arcname=SAMPLE_ID + ext)
    return tar_path


@pytest.fixture(scope="session")
def corrupt_tar(tmp_path_factory) -> Path:
    """Tar with a sample missing the .scale.npy file."""
    tar_path = tmp_path_factory.mktemp("qc_stage1_corrupt") / "shard-000000.tar"
    exts = [".mp4", ".poses_c2w.npy", ".intrinsics.npy",
            ".caption.txt", ".meta.json"]  # intentionally missing scale
    with tarfile.open(tar_path, "w") as tf:
        for ext in exts:
            src = TESTDATA / (SAMPLE_ID + ext)
            tf.add(src, arcname=SAMPLE_ID + ext)
    return tar_path


@pytest.fixture(scope="session")
def nan_poses_tar(tmp_path_factory) -> Path:
    """Tar with NaN injected into poses_c2w."""
    tar_path = tmp_path_factory.mktemp("qc_stage1_nan") / "shard-000000.tar"
    buf = io.BytesIO()
    poses = np.load(TESTDATA / (SAMPLE_ID + ".poses_c2w.npy"))
    poses[0, 0, 0] = float("nan")
    np.save(buf, poses)
    with tarfile.open(tar_path, "w") as tf:
        for ext in [".mp4", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
        info = tarfile.TarInfo(name=SAMPLE_ID + ".poses_c2w.npy")
        data = buf.getvalue()
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return tar_path


# --- scan_tar tests ---

def test_scan_tar_returns_list(tiny_tar):
    results = scan_tar(tiny_tar, GROUP)
    assert isinstance(results, list)
    assert len(results) == 1


def test_scan_tar_has_required_keys(tiny_tar):
    result = scan_tar(tiny_tar, GROUP)[0]
    for key in ("sample_id", "group", "tar_path", "verdict", "flag_reasons", "metrics"):
        assert key in result, f"missing key: {key}"


def test_scan_tar_sample_id_correct(tiny_tar):
    result = scan_tar(tiny_tar, GROUP)[0]
    assert result["sample_id"] == SAMPLE_ID


def test_scan_tar_testdata_passes(tiny_tar):
    """testdata sample has valid SO3 and passes all hard checks."""
    result = scan_tar(tiny_tar, GROUP)[0]
    m = result["metrics"]
    assert m["so3_valid"] is True
    assert m["first_frame_ok"] is True
    assert m["no_nan_inf"] is True
    # OmniWorld-Game group: 5 jumps with threshold 2.0m should be fine
    assert result["verdict"] in ("pass", "flag")


def test_scan_tar_missing_file_fails(corrupt_tar):
    results = scan_tar(corrupt_tar, GROUP)
    assert len(results) == 1
    assert results[0]["verdict"] == "fail"
    assert any("missing" in r.lower() for r in results[0]["flag_reasons"])


def test_scan_tar_nan_fails(nan_poses_tar):
    results = scan_tar(nan_poses_tar, GROUP)
    assert results[0]["verdict"] == "fail"


# --- run_stage1 tests ---

def test_run_stage1_writes_jsonl(tiny_tar, tmp_path):
    out = tmp_path / "stage1.jsonl"
    count = run_stage1([tiny_tar], GROUP, out, n_workers=1)
    assert count == 1
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["sample_id"] == SAMPLE_ID


def test_run_stage1_appends_multiple_tars(tiny_tar, tmp_path):
    out = tmp_path / "stage1.jsonl"
    # Pass same tar twice to simulate two shards
    count = run_stage1([tiny_tar, tiny_tar], GROUP, out, n_workers=1)
    assert count == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_qc_stage1.py -v 2>&1 | head -15
```
预期：`ImportError: cannot import name 'scan_tar'`

- [ ] **Step 3: 实现 `stage1_fast.py`**

```python
# src/sana_wm_pipeline/qc/stage1_fast.py
"""Stage 1: fast full-coverage scan of tar shards (CPU-only, multiprocessing)."""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
from multiprocessing import Pool
from typing import Any
import numpy as np
from sana_wm_pipeline.qc.metrics import compute_stage1_metrics
from sana_wm_pipeline.qc.group_config import get_group_config, compute_verdict

_REQUIRED_EXTS = {".mp4", ".poses_c2w.npy", ".intrinsics.npy",
                  ".scale.npy", ".caption.txt", ".meta.json"}
IMAGE_WH = (1280, 720)   # all production samples are normalised to this


def _scan_one_sample(
    sample_id: str,
    file_bytes: dict[str, bytes],
    group_name: str,
    tar_path: str,
) -> dict[str, Any]:
    """Process a single sample from its raw file bytes. Top-level for pickle."""
    cfg = get_group_config(group_name)

    # Check all files present
    missing = [ext for ext in _REQUIRED_EXTS if ext not in file_bytes]
    if missing:
        return {
            "sample_id": sample_id,
            "group": group_name,
            "tar_path": tar_path,
            "verdict": "fail",
            "flag_reasons": [f"missing files: {missing}"],
            "metrics": {},
        }

    try:
        poses = np.load(io.BytesIO(file_bytes[".poses_c2w.npy"]))
        intrinsics = np.load(io.BytesIO(file_bytes[".intrinsics.npy"]))
        scale = np.load(io.BytesIO(file_bytes[".scale.npy"]))
        caption = file_bytes[".caption.txt"].decode("utf-8", errors="replace")
        meta = json.loads(file_bytes[".meta.json"].decode("utf-8"))
        meta_T = int(meta.get("T", -1))
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "group": group_name,
            "tar_path": tar_path,
            "verdict": "fail",
            "flag_reasons": [f"load_error: {exc}"],
            "metrics": {},
        }

    metrics = compute_stage1_metrics(
        poses=poses,
        intrinsics=intrinsics,
        scale=scale,
        caption=caption,
        meta_T=meta_T,
        image_wh=IMAGE_WH,
        jump_threshold_m=cfg.jump_threshold_m,
        min_caption_len=cfg.min_caption_len,
    )
    verdict, flag_reasons = compute_verdict(metrics, cfg)
    return {
        "sample_id": sample_id,
        "group": group_name,
        "tar_path": str(tar_path),
        "verdict": verdict,
        "flag_reasons": flag_reasons,
        "metrics": metrics,
    }


def _extract_samples_from_tar(tar_path: Path) -> dict[str, dict[str, bytes]]:
    """Read a tar shard and group file bytes by sample_id.

    Returns {sample_id: {".ext": bytes}}.
    """
    samples: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            name = member.name
            # Split on first dot after the last slash to get (sample_id, ext)
            stem, sep, suffix = name.partition(".")
            if not sep:
                continue
            ext = "." + suffix
            if ext not in _REQUIRED_EXTS and ext != ".mp4":
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            samples.setdefault(stem, {})[ext] = f.read()
    return samples


def scan_tar(tar_path: Path, group_name: str) -> list[dict]:
    """Scan all samples in one tar shard. Returns list of per-sample result dicts."""
    tar_path = Path(tar_path)
    samples = _extract_samples_from_tar(tar_path)
    results = []
    for sample_id, file_bytes in samples.items():
        result = _scan_one_sample(sample_id, file_bytes, group_name, str(tar_path))
        results.append(result)
    return results


def _worker_fn(args: tuple) -> list[dict]:
    tar_path, group_name = args
    return scan_tar(Path(tar_path), group_name)


def run_stage1(
    tar_paths: list[Path],
    group_name: str,
    output_jsonl: Path,
    n_workers: int = 32,
) -> int:
    """Run Stage 1 on all tar shards in parallel. Writes output_jsonl. Returns sample count."""
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    worker_args = [(str(p), group_name) for p in tar_paths]
    total = 0
    with open(output_jsonl, "w") as fout:
        with Pool(processes=min(n_workers, len(tar_paths))) as pool:
            for batch in pool.imap_unordered(_worker_fn, worker_args):
                for rec in batch:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
    return total
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_qc_stage1.py -v
```
预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/qc/stage1_fast.py tests/test_qc_stage1.py
git commit -m "feat(qc): add Stage 1 fast tar scanner with multiprocessing"
```

---

## Task 4: `stage2_deep.py` — Stage 2 深度检测

**Files:**
- Create: `src/sana_wm_pipeline/qc/stage2_deep.py`
- Test: `tests/test_qc_stage2.py`

**Interfaces:**
- Consumes: `stage1_results.jsonl`（前一阶段输出）
- Produces:
  - `count_video_frames_av(video_bytes) -> int`：用 PyAV 计算视频帧数
  - `check_trajectory_frozen(poses_c2w, frozen_threshold) -> tuple[bool, float]`：检测 SLAM 跟踪冻结
  - `deep_check_sample(sample_id, tar_path, group_name) -> dict`：完整 Stage 2 检测
  - `run_stage2(stage1_jsonl, output_jsonl, sample_frac, n_workers) -> int`：选样 + 并行运行

**Stage 2 触发规则：**
- Stage 1 `verdict == "flag"` → 100% 进 Stage 2
- Stage 1 `verdict == "pass"` → 按 group 随机抽 `sample_frac=0.05`（5%）
- Stage 1 `verdict == "fail"` → 不做 Stage 2（已确定拒绝）

**Stage 2 增量输出 JSON（追加到原 Stage 1 结果）：**
```json
{
  "sample_id": "...",
  "stage2": {
    "video_T": 782,
    "video_T_matches_npy": true,
    "traj_frozen": false,
    "frozen_ratio": 0.0,
    "reasons": []
  }
}
```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_stage2.py
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage2_deep import (
    count_video_frames_av, check_trajectory_frozen, deep_check_sample, run_stage2,
)

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"
EXPECTED_T = 782  # confirmed from data analysis


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory) -> Path:
    tar_path = tmp_path_factory.mktemp("qc_stage2") / "shard-000000.tar"
    exts = [".mp4", ".poses_c2w.npy", ".intrinsics.npy",
            ".scale.npy", ".caption.txt", ".meta.json"]
    with tarfile.open(tar_path, "w") as tf:
        for ext in exts:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return tar_path


@pytest.fixture(scope="session")
def sample_video_bytes() -> bytes:
    return (TESTDATA / (SAMPLE_ID + ".mp4")).read_bytes()


# --- count_video_frames_av ---

def test_count_video_frames_matches_npy(sample_video_bytes):
    T = count_video_frames_av(sample_video_bytes)
    assert T == EXPECTED_T


def test_count_video_frames_returns_positive(sample_video_bytes):
    T = count_video_frames_av(sample_video_bytes)
    assert T > 0


# --- check_trajectory_frozen ---

def test_trajectory_not_frozen_with_motion():
    T = 100
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    poses[:, 0, 3] = np.arange(T, dtype=np.float32) * 0.1
    frozen, ratio = check_trajectory_frozen(poses, frozen_threshold=1e-4)
    assert not frozen
    assert ratio < 0.01


def test_trajectory_frozen_detected():
    T = 100
    poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    # 80% of frames have identical translation (frozen)
    poses[10:90, 0, 3] = 5.0
    frozen, ratio = check_trajectory_frozen(poses, frozen_threshold=1e-4)
    assert frozen
    assert ratio > 0.7


# --- deep_check_sample ---

def test_deep_check_sample_structure(tiny_tar):
    result = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert "sample_id" in result
    assert "stage2" in result
    s2 = result["stage2"]
    for key in ("video_T", "video_T_matches_npy", "traj_frozen", "frozen_ratio", "reasons"):
        assert key in s2, f"missing stage2 key: {key}"


def test_deep_check_video_T_matches(tiny_tar):
    result = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert result["stage2"]["video_T"] == EXPECTED_T
    assert result["stage2"]["video_T_matches_npy"] is True


def test_deep_check_not_frozen(tiny_tar):
    result = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert result["stage2"]["traj_frozen"] is False


# --- run_stage2 ---

def test_run_stage2_processes_flagged(tiny_tar, tmp_path):
    # Build a synthetic stage1.jsonl where the sample is "flag"
    s1_jsonl = tmp_path / "stage1.jsonl"
    record = {
        "sample_id": SAMPLE_ID,
        "group": GROUP,
        "tar_path": str(tiny_tar),
        "verdict": "flag",
        "flag_reasons": ["n_jumps=5 > max_jumps_flag=3"],
        "metrics": {"T": EXPECTED_T},
    }
    s1_jsonl.write_text(json.dumps(record) + "\n")
    out = tmp_path / "stage2.jsonl"
    count = run_stage2(s1_jsonl, out, sample_frac=0.0, n_workers=1)
    assert count == 1
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "stage2" in rec


def test_run_stage2_skips_fail(tiny_tar, tmp_path):
    s1_jsonl = tmp_path / "stage1.jsonl"
    record = {
        "sample_id": SAMPLE_ID,
        "group": GROUP,
        "tar_path": str(tiny_tar),
        "verdict": "fail",
        "flag_reasons": ["so3_invalid"],
        "metrics": {},
    }
    s1_jsonl.write_text(json.dumps(record) + "\n")
    out = tmp_path / "stage2.jsonl"
    count = run_stage2(s1_jsonl, out, sample_frac=0.0, n_workers=1)
    assert count == 0
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_qc_stage2.py -v 2>&1 | head -10
```
预期：`ImportError`

- [ ] **Step 3: 实现 `stage2_deep.py`**

```python
# src/sana_wm_pipeline/qc/stage2_deep.py
"""Stage 2: deep targeted checks on flagged samples + random 5% per group."""
from __future__ import annotations
import io
import json
import random
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import numpy as np

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

_FROZEN_THRESHOLD = 1e-4   # consecutive step < this → frozen frame


def count_video_frames_av(video_bytes: bytes) -> int:
    """Count video frames using PyAV. No subprocess needed.

    Fast path uses container header; slow path decodes if header says 0.
    """
    if not _AV_AVAILABLE:
        raise RuntimeError("PyAV not installed; run: pip install av")
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        if stream.frames > 0:
            return int(stream.frames)
        # Slow path: decode and count
        return sum(1 for _ in container.decode(video=0))


def check_trajectory_frozen(
    poses_c2w: np.ndarray,
    frozen_threshold: float = _FROZEN_THRESHOLD,
) -> tuple[bool, float]:
    """Detect SLAM tracking loss (consecutive near-identical poses).

    Returns:
        (is_frozen, frozen_ratio) where frozen_ratio = fraction of consecutive
        pairs with step < frozen_threshold.
    Frozen is True if frozen_ratio > 0.5.
    """
    t = poses_c2w[:, :3, 3].astype(np.float64)
    if len(t) < 2:
        return False, 0.0
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    frozen_ratio = float((steps < frozen_threshold).mean())
    return frozen_ratio > 0.5, frozen_ratio


def _load_npy_from_tar(tf: tarfile.TarFile, name: str) -> np.ndarray:
    m = tf.getmember(name)
    f = tf.extractfile(m)
    return np.load(io.BytesIO(f.read()))


def deep_check_sample(
    sample_id: str,
    tar_path: Path,
    group_name: str,
) -> dict[str, Any]:
    """Run Stage 2 deep checks on a single sample.

    Reads from tar_path. Returns dict with 'sample_id' and 'stage2' keys.
    """
    tar_path = Path(tar_path)
    stage2: dict[str, Any] = {
        "video_T": -1,
        "video_T_matches_npy": None,
        "traj_frozen": None,
        "frozen_ratio": None,
        "reasons": [],
    }

    try:
        with tarfile.open(tar_path, "r") as tf:
            # Video frame count via PyAV
            try:
                mp4_member = tf.getmember(f"{sample_id}.mp4")
                video_bytes = tf.extractfile(mp4_member).read()
                video_T = count_video_frames_av(video_bytes)
                stage2["video_T"] = video_T
            except (KeyError, Exception) as e:
                stage2["reasons"].append(f"video_read_error: {e}")
                video_T = -1

            # Poses for frozen trajectory check
            try:
                poses = _load_npy_from_tar(tf, f"{sample_id}.poses_c2w.npy")
                npy_T = int(poses.shape[0])
                stage2["video_T_matches_npy"] = (video_T == npy_T)
                if video_T != npy_T and video_T > 0:
                    stage2["reasons"].append(
                        f"video_npy_T_mismatch: video={video_T} npy={npy_T}"
                    )
                frozen, frozen_ratio = check_trajectory_frozen(poses)
                stage2["traj_frozen"] = frozen
                stage2["frozen_ratio"] = round(frozen_ratio, 4)
                if frozen:
                    stage2["reasons"].append(
                        f"traj_frozen: {frozen_ratio:.1%} frames stationary"
                    )
            except (KeyError, Exception) as e:
                stage2["reasons"].append(f"poses_load_error: {e}")

    except Exception as e:
        stage2["reasons"].append(f"tar_open_error: {e}")

    return {"sample_id": sample_id, "stage2": stage2}


def _worker_fn(args: tuple) -> dict:
    sample_id, tar_path, group_name = args
    return deep_check_sample(sample_id, Path(tar_path), group_name)


def run_stage2(
    stage1_jsonl: Path,
    output_jsonl: Path,
    sample_frac: float = 0.05,
    n_workers: int = 16,
) -> int:
    """Select samples for Stage 2 and run deep checks.

    Selection rules:
    - verdict == "fail" → skip (already rejected)
    - verdict == "flag" → 100% include
    - verdict == "pass" → include with probability sample_frac (per-group random)

    Writes output_jsonl (one merged record per processed sample).
    Returns count of samples processed.
    """
    stage1_jsonl = Path(stage1_jsonl)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load stage1 results and select
    selected: list[tuple[str, str, str]] = []  # (sample_id, tar_path, group_name)
    all_records: dict[str, dict] = {}
    rng = random.Random(42)

    with open(stage1_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            all_records[sid] = rec
            verdict = rec.get("verdict", "pass")
            if verdict == "fail":
                continue
            if verdict == "flag" or rng.random() < sample_frac:
                selected.append((sid, rec["tar_path"], rec.get("group", "")))

    if not selected:
        output_jsonl.write_text("")
        return 0

    worker_args = [(sid, tp, gn) for sid, tp, gn in selected]
    with open(output_jsonl, "w") as fout:
        with Pool(processes=min(n_workers, len(worker_args))) as pool:
            for s2_rec in pool.imap_unordered(_worker_fn, worker_args):
                # Merge stage2 results into the original stage1 record
                sid = s2_rec["sample_id"]
                merged = dict(all_records[sid])
                merged["stage2"] = s2_rec["stage2"]
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

    return len(selected)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_qc_stage2.py -v
```
预期：全部 `PASSED`（`count_video_frames_av` 测试依赖 PyAV，已在 deps）

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/qc/stage2_deep.py tests/test_qc_stage2.py
git commit -m "feat(qc): add Stage 2 deep checker (PyAV frame count + frozen traj)"
```

---

## Task 5: `report.py` — 合并结果 + Manifest + HTML 报告

**Files:**
- Create: `src/sana_wm_pipeline/qc/report.py`
- Test: `tests/test_qc_report.py`

**Interfaces:**
- Consumes: `stage1_results.jsonl`, `stage2_results.jsonl`（可选）
- Produces:
  - `merge_results(stage1_jsonl, stage2_jsonl) -> list[dict]`：合并两阶段结果
  - `write_manifests(results, output_dir)`：写三份 txt manifest
  - `write_html_report(results, output_dir)`：写 `report.html`
  - `run_report(stage1_jsonl, stage2_jsonl, output_dir)`：一键调用上述三函数

**输出文件：**
```
{output_dir}/manifests/pass.txt          # 一行一个 sample_id
{output_dir}/manifests/reject.txt
{output_dir}/manifests/human_review.txt  # {sample_id}\t{pipe-joined reasons}
{output_dir}/report.html                 # 含 per-group 汇总表 + 关键指标分布
```

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_report.py
from __future__ import annotations
import json
from pathlib import Path
import pytest
from sana_wm_pipeline.qc.report import merge_results, write_manifests, write_html_report


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture
def sample_records() -> list[dict]:
    base = {
        "group": "wds-OmniWorld-Game",
        "tar_path": "/fake/shard.tar",
        "flag_reasons": [],
        "metrics": {"T": 100, "n_jumps": 0, "caption_len": 200, "det_R_mean": 1.0},
        "stage2": None,
    }
    return [
        {**base, "sample_id": "sample_pass_1", "verdict": "pass"},
        {**base, "sample_id": "sample_pass_2", "verdict": "pass"},
        {**base, "sample_id": "sample_flag_1", "verdict": "flag", "flag_reasons": ["n_jumps=5"]},
        {**base, "sample_id": "sample_fail_1", "verdict": "fail", "flag_reasons": ["so3_invalid"]},
    ]


# --- merge_results ---

def test_merge_results_stage2_none(tmp_path, sample_records):
    s1 = tmp_path / "stage1.jsonl"
    _write_jsonl(s1, sample_records)
    results = merge_results(s1, stage2_jsonl=None)
    assert len(results) == 4
    assert all("verdict" in r for r in results)


def test_merge_results_stage2_updates_record(tmp_path, sample_records):
    s1 = tmp_path / "stage1.jsonl"
    _write_jsonl(s1, sample_records)
    s2 = tmp_path / "stage2.jsonl"
    # Add stage2 entry for the flag sample
    flag_rec = dict(sample_records[2])
    flag_rec["stage2"] = {"video_T": 100, "video_T_matches_npy": True,
                          "traj_frozen": False, "frozen_ratio": 0.0, "reasons": []}
    _write_jsonl(s2, [flag_rec])
    results = merge_results(s1, s2)
    flag_result = next(r for r in results if r["sample_id"] == "sample_flag_1")
    assert flag_result["stage2"] is not None
    assert flag_result["stage2"]["video_T"] == 100


# --- write_manifests ---

def test_write_manifests_creates_files(tmp_path, sample_records):
    write_manifests(sample_records, tmp_path)
    assert (tmp_path / "manifests" / "pass.txt").exists()
    assert (tmp_path / "manifests" / "reject.txt").exists()
    assert (tmp_path / "manifests" / "human_review.txt").exists()


def test_write_manifests_correct_counts(tmp_path, sample_records):
    write_manifests(sample_records, tmp_path)
    pass_ids = (tmp_path / "manifests" / "pass.txt").read_text().strip().splitlines()
    reject_ids = (tmp_path / "manifests" / "reject.txt").read_text().strip().splitlines()
    review_lines = (tmp_path / "manifests" / "human_review.txt").read_text().strip().splitlines()
    assert len(pass_ids) == 2
    assert "sample_pass_1" in pass_ids
    assert len(reject_ids) == 1
    assert "sample_fail_1" in reject_ids
    assert len(review_lines) == 1
    assert "sample_flag_1" in review_lines[0]
    assert "n_jumps=5" in review_lines[0]


# --- write_html_report ---

def test_write_html_report_creates_file(tmp_path, sample_records):
    write_html_report(sample_records, tmp_path)
    report = tmp_path / "report.html"
    assert report.exists()
    html = report.read_text()
    assert "<html" in html.lower()


def test_write_html_report_contains_group_stats(tmp_path, sample_records):
    write_html_report(sample_records, tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "wds-OmniWorld-Game" in html
    assert "pass" in html.lower()
    assert "fail" in html.lower()
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_qc_report.py -v 2>&1 | head -10
```

- [ ] **Step 3: 实现 `report.py`**

```python
# src/sana_wm_pipeline/qc/report.py
"""Merge Stage 1 + Stage 2 results; write manifests and HTML report."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np


def merge_results(
    stage1_jsonl: Path,
    stage2_jsonl: Optional[Path],
) -> list[dict]:
    """Merge Stage 1 base records with Stage 2 deep-check additions.

    Stage 2 records (if any) update the matching Stage 1 record's 'stage2' key.
    Returns list of all records (order matches Stage 1 file).
    """
    records: dict[str, dict] = {}
    ordered: list[str] = []
    with open(stage1_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            records[sid] = rec
            ordered.append(sid)

    if stage2_jsonl and Path(stage2_jsonl).exists():
        with open(stage2_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec["sample_id"]
                if sid in records:
                    records[sid]["stage2"] = rec.get("stage2")

    return [records[sid] for sid in ordered]


def write_manifests(results: list[dict], output_dir: Path) -> None:
    """Write three manifest files to {output_dir}/manifests/."""
    output_dir = Path(output_dir)
    mdir = output_dir / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)

    pass_lines, reject_lines, review_lines = [], [], []
    for rec in results:
        sid = rec["sample_id"]
        verdict = rec.get("verdict", "pass")
        reasons = rec.get("flag_reasons", [])
        if verdict == "pass":
            pass_lines.append(sid)
        elif verdict == "fail":
            reject_lines.append(sid)
        elif verdict == "flag":
            review_lines.append(f"{sid}\t{'|'.join(reasons)}")

    (mdir / "pass.txt").write_text("\n".join(pass_lines) + "\n" if pass_lines else "")
    (mdir / "reject.txt").write_text("\n".join(reject_lines) + "\n" if reject_lines else "")
    (mdir / "human_review.txt").write_text(
        "\n".join(review_lines) + "\n" if review_lines else ""
    )


def _svg_histogram(values: list[float], width: int = 300, height: int = 80) -> str:
    """Generate a simple inline SVG bar-chart histogram."""
    if not values:
        return f'<svg width="{width}" height="{height}"></svg>'
    arr = np.array(values, dtype=float)
    counts, _ = np.histogram(arr, bins=min(20, len(set(values))))
    max_c = max(counts) if counts.max() > 0 else 1
    bar_w = width / len(counts)
    bars = []
    for i, c in enumerate(counts):
        bh = max(1, int(c / max_c * height))
        bars.append(
            f'<rect x="{i*bar_w:.1f}" y="{height - bh}" '
            f'width="{max(1, bar_w - 1):.1f}" height="{bh}" fill="#4a9"/>'
        )
    return (
        f'<svg width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>'
    )


def write_html_report(results: list[dict], output_dir: Path) -> None:
    """Write an HTML QC report to {output_dir}/report.html."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-group statistics
    group_stats: dict[str, dict] = defaultdict(lambda: {
        "pass": 0, "flag": 0, "fail": 0,
        "n_jumps": [], "caption_len": [], "step_max_m": [],
    })
    for rec in results:
        g = rec.get("group", "unknown")
        v = rec.get("verdict", "pass")
        group_stats[g][v] += 1
        m = rec.get("metrics", {})
        if "n_jumps" in m:
            group_stats[g]["n_jumps"].append(m["n_jumps"])
        if "caption_len" in m:
            group_stats[g]["caption_len"].append(m["caption_len"])
        if "step_max_m" in m:
            group_stats[g]["step_max_m"].append(m["step_max_m"])

    total = len(results)
    n_pass = sum(1 for r in results if r.get("verdict") == "pass")
    n_flag = sum(1 for r in results if r.get("verdict") == "flag")
    n_fail = sum(1 for r in results if r.get("verdict") == "fail")

    # Build group rows
    group_rows = []
    for g, s in sorted(group_stats.items()):
        g_total = s["pass"] + s["flag"] + s["fail"]
        pass_pct = 100 * s["pass"] / g_total if g_total else 0
        jmp_svg = _svg_histogram(s["n_jumps"])
        cap_svg = _svg_histogram(s["caption_len"])
        group_rows.append(
            f"<tr><td>{g}</td><td>{g_total}</td>"
            f"<td>{s['pass']} ({pass_pct:.1f}%)</td>"
            f"<td>{s['flag']}</td><td>{s['fail']}</td>"
            f"<td>{jmp_svg}</td><td>{cap_svg}</td></tr>"
        )

    # Human review table (first 200 only)
    flag_recs = [r for r in results if r.get("verdict") == "flag"][:200]
    review_rows = "".join(
        f"<tr><td>{r['sample_id']}</td><td>{r.get('group','')}</td>"
        f"<td>{'<br>'.join(r.get('flag_reasons', []))}</td></tr>"
        for r in flag_recs
    )

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SANA-WM Pipeline QC Report</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #eee; }}
  h1, h2 {{ color: #a8daff; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th, td {{ border: 1px solid #444; padding: 6px 10px; text-align: left; }}
  th {{ background: #2a2a4a; }}
  .stat {{ display: inline-block; margin: 10px 20px; font-size: 1.4em; }}
  .pass {{ color: #4a9; }} .flag {{ color: #fa3; }} .fail {{ color: #f44; }}
</style>
</head>
<body>
<h1>SANA-WM Pipeline QC Report</h1>
<div>
  <span class="stat">Total <b>{total}</b></span>
  <span class="stat pass">Pass <b>{n_pass}</b> ({100*n_pass/total:.1f}%)</span>
  <span class="stat flag">Flag <b>{n_flag}</b> ({100*n_flag/total:.1f}%)</span>
  <span class="stat fail">Fail <b>{n_fail}</b> ({100*n_fail/total:.1f}%)</span>
</div>
<h2>Per-Group Summary</h2>
<table>
  <tr><th>Group</th><th>Total</th><th>Pass</th><th>Flag</th><th>Fail</th>
      <th>Jump Dist</th><th>Caption Len Dist</th></tr>
  {"".join(group_rows)}
</table>
<h2>Human Review Queue (first 200 flagged samples)</h2>
<table>
  <tr><th>sample_id</th><th>group</th><th>flag_reasons</th></tr>
  {review_rows}
</table>
</body>
</html>"""

    (output_dir / "report.html").write_text(html, encoding="utf-8")


def run_report(
    stage1_jsonl: Path,
    stage2_jsonl: Optional[Path],
    output_dir: Path,
) -> None:
    """Full report pipeline: merge → manifests → HTML."""
    results = merge_results(stage1_jsonl, stage2_jsonl)
    write_manifests(results, output_dir)
    write_html_report(results, output_dir)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_qc_report.py -v
```
预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/sana_wm_pipeline/qc/report.py tests/test_qc_report.py
git commit -m "feat(qc): add report generator (manifests + HTML)"
```

---

## Task 6: `scripts/run_qc.py` — CLI 入口 + 集成测试

**Files:**
- Create: `scripts/run_qc.py`
- Test: 在 `tests/test_qc_stage1.py` 尾部追加集成测试

**CLI 用法：**
```bash
# 全流程（Stage1 + Stage2 + Report）
python scripts/run_qc.py \
  --tar-root /path/to/output/wds-OmniWorld-Game \
  --group wds-OmniWorld-Game \
  --output-dir ./qc_output \
  --n-workers 32

# 仅 Stage 1（调试时用）
python scripts/run_qc.py --tar-root ... --group ... --skip-stage2

# 仅 Report（Stage1/2 已跑完，重新生成报告）
python scripts/run_qc.py --report-only --output-dir ./qc_output
```

- [ ] **Step 1: 写集成测试（追加到 `tests/test_qc_stage1.py`）**

```python
# 追加到 tests/test_qc_stage1.py 末尾
import subprocess, sys

def test_cli_end_to_end(tiny_tar, tmp_path):
    """Run full CLI pipeline on a synthetic single-tar directory."""
    tar_dir = tmp_path / "tar_root"
    tar_dir.mkdir()
    import shutil
    shutil.copy(tiny_tar, tar_dir / "shard-000000.tar")

    result = subprocess.run(
        [sys.executable, "scripts/run_qc.py",
         "--tar-root", str(tar_dir),
         "--group", GROUP,
         "--output-dir", str(tmp_path / "qc_out"),
         "--n-workers", "1",
         "--skip-stage2"],
        capture_output=True, text=True,
        cwd="/mnt/afs/davidwang/workspace/sana_wm_pipeline",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "qc_out" / "stage1_results.jsonl").exists()
    assert (tmp_path / "qc_out" / "manifests" / "pass.txt").exists()
    assert (tmp_path / "qc_out" / "report.html").exists()
```

- [ ] **Step 2: 运行，确认失败**

```bash
python -m pytest tests/test_qc_stage1.py::test_cli_end_to_end -v 2>&1 | head -10
```

- [ ] **Step 3: 实现 `scripts/run_qc.py`**

```python
#!/usr/bin/env python3
# scripts/run_qc.py
"""CLI entry point for the SANA-WM Output QC System."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running from repo root without pip install -e
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.qc.stage1_fast import run_stage1
from sana_wm_pipeline.qc.stage2_deep import run_stage2
from sana_wm_pipeline.qc.report import run_report


def _find_tars(tar_root: Path) -> list[Path]:
    """Recursively find all .tar files under tar_root."""
    tars = sorted(tar_root.rglob("*.tar"))
    if not tars:
        print(f"[warn] No .tar files found under {tar_root}", flush=True)
    return tars


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SANA-WM Pipeline Output QC System (两阶段质检)"
    )
    parser.add_argument("--tar-root", type=Path,
                        help="Root directory containing tar shards (searched recursively)")
    parser.add_argument("--group", default="",
                        help="Dataset group name, e.g. wds-OmniWorld-Game")
    parser.add_argument("--output-dir", type=Path, default=Path("./qc_output"),
                        help="Output directory for all QC artefacts (default: ./qc_output)")
    parser.add_argument("--n-workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--sample-frac", type=float, default=0.05,
                        help="Fraction of passing samples to deep-check in Stage 2 (default: 0.05)")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Run Stage 1 only; skip Stage 2 deep checks")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip Stage 1+2, regenerate report from existing jsonl files")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s1_jsonl = out / "stage1_results.jsonl"
    s2_jsonl = out / "stage2_results.jsonl"

    if not args.report_only:
        if not args.tar_root:
            parser.error("--tar-root is required unless --report-only")
        tars = _find_tars(args.tar_root)
        print(f"[stage1] Found {len(tars)} tar shards under {args.tar_root}", flush=True)
        n = run_stage1(tars, args.group, s1_jsonl, n_workers=args.n_workers)
        print(f"[stage1] Scanned {n} samples → {s1_jsonl}", flush=True)

        if not args.skip_stage2:
            n2 = run_stage2(s1_jsonl, s2_jsonl,
                            sample_frac=args.sample_frac,
                            n_workers=args.n_workers)
            print(f"[stage2] Deep-checked {n2} samples → {s2_jsonl}", flush=True)
        else:
            print("[stage2] Skipped (--skip-stage2)", flush=True)

    s2_input = s2_jsonl if (not args.skip_stage2 and s2_jsonl.exists()) else None
    run_report(s1_jsonl, s2_input, out)
    print(f"[report] Written → {out}/report.html + {out}/manifests/", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行集成测试**

```bash
python -m pytest tests/test_qc_stage1.py -v -k "cli"
```
预期：`PASSED`

- [ ] **Step 5: 完整测试套件验证**

```bash
python -m pytest tests/test_qc_metrics.py tests/test_qc_group_config.py \
    tests/test_qc_stage1.py tests/test_qc_stage2.py tests/test_qc_report.py -v
```
预期：全部绿色

- [ ] **Step 6: Commit**

```bash
git add scripts/run_qc.py
git commit -m "feat(qc): add CLI run_qc.py + end-to-end integration test"
```

---

## 人力估算

| 任务 | 工作量 | 说明 |
|---|---|---|
| 自动化质检运行（Stage1+2） | ~0.5 人时 | 写 run 命令后机器自跑 |
| 报告解读 + 阈值调优 | 1-2 人时 | 看 HTML 报告，调 GroupConfig |
| **人工质检（flag 样本）** | **估算见下** | 主要人力投入 |

**人工质检人力估算（flag 样本）：**

基于观测到 OmniWorld-Game 4 个样本中跳变数 5-12，假设 flag 率 ~10-20%（各 group 平均）：

| 场景 | flag 率 | flag 样本数 | 人均速度 | 人力 |
|---|---|---|---|---|
| 乐观 | 5% | 1 万 | 200条/小时 | 50 人时 ≈ 6人×1天 |
| 基准 | 15% | 3 万 | 200条/小时 | 150 人时 ≈ 5人×3天 |
| 保守 | 25% | 5 万 | 200条/小时 | 250 人时 ≈ 5人×5天 |

人工质检任务：每条样本看 caption 是否准确描述场景 + 视频缩略图确认跳变是否为镜头切换（合理）还是 SLAM 漂移（需丢弃）。建议用 [Label Studio](https://labelstud.io/) 搭建批注工具，每条样本给出 `approve / reject / unsure` 三选一。

---

## 运行方法（全流程示例）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 激活环境（AFS）
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate abot-physworld

# 对 OmniWorld-Game group 运行完整质检
python scripts/run_qc.py \
  --tar-root /path/to/output/wds-OmniWorld-Game \
  --group wds-OmniWorld-Game \
  --output-dir ./qc_output/OmniWorld-Game \
  --n-workers 32

# 对 DL3DV group（严格阈值）
python scripts/run_qc.py \
  --tar-root /path/to/output/wds-DL3DV-ALL-2K \
  --group wds-DL3DV-ALL-2K \
  --output-dir ./qc_output/DL3DV \
  --n-workers 32

# 查看结果
ls qc_output/OmniWorld-Game/manifests/
# pass.txt  reject.txt  human_review.txt
open qc_output/OmniWorld-Game/report.html
```

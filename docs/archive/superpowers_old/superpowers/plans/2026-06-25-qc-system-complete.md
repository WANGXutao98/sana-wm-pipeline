# QC System Complete Implementation Plan (Stage 1 + 2 + 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 CMCC 批量生产的 ~20 万条 SANA-WM 样本，实现三阶段质检系统：Stage 1 全量 CPU 扫描（11 项检查）→ Stage 2 深度 CPU 检测（待定样本）→ Stage 3 GPU 评估（UniMatch + DOVER + Qwen3.5-27B），产出 pass/reject/human_review 三份清单 + HTML 报告 + caption 改写 sidecar。

**Architecture:** Stage 1 对每条样本做 CPU-only 扫描，11 类检查包括结构/位姿/颜色饱和度/caption 摄像机词；Stage 2 对 flag 样本做视频深度检测（PyAV 帧数/PySceneDetect 场景切割/黑帧比例/轨迹冻结）；Stage 3 在 CMCC 48 H100 上并行跑 UniMatch + DOVER + Qwen3.5-27B，结果写 sidecar（不改原始 tar）。所有输出写 `qc_output/`。

**Tech Stack:** Python 3.10+, numpy, av (PyAV), scenedetect, cv2, tarfile, multiprocessing, stage04_filter（已有：apply_table6/scene_cut/visual_metrics/vlm_entity_quality）, stage02_pose/pose_quality, pytest

## Global Constraints

- Python ≥ 3.10，所有文件加 `from __future__ import annotations`
- QC 写操作只写 `qc_output/` 目录，**严禁修改原始 tar 文件**
- 多进程用 `multiprocessing.Pool(spawn)`，worker 函数必须 top-level 可 pickle
- 生产帧数 strict_frames=False（与 run_worker.py:164 一致，T≠961 合法）
- 日志用 `print(..., flush=True)`（CMCC 无 rich）
- 所有输出写到 `--output-dir`（默认 `./qc_output/`）
- pytest 路径在 `tests/`，`pythonpath = ["src"]`（pyproject.toml 已配置）
- 游戏数据 max_jumps_fail=**50**（2026-06-25 决策，原计划 100 已废弃）
- Qwen3.5-27B 权重路径：`~/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL/`（已在 CMCC 下载完成）
- caption 改写结果写 `qc_output/caption_overrides.jsonl`，格式：`{"sample_id": "...", "caption_original": "...", "caption_revised": "..."}`
- Stage 3 CMCC launcher 复用 `experiments/batch_production/` 的 hostfile + SSH 派发模式

---

## 与旧计划（2026-06-21）的差异说明

旧计划（`2026-06-21-output-qc-system.md`）已完整覆盖 Task 1-6 的 Stage 1+2 基础框架，但有以下缺口：

| 缺口 | 本计划位置 |
|---|---|
| Stage 1 颜色饱和度检查（Check 9）| Task 1 metrics.py 新增函数 |
| Stage 1 caption 摄像机动作词检查（Check 10）| Task 1 metrics.py 新增函数 |
| group_config.py 缺 3 个新 group + 游戏上限从 100 → 50 | Task 2 完整重写 |
| Stage 2 场景切割检测（复用 scene_cut.py）| Task 4 stage2_deep.py 扩展 |
| Stage 2 黑帧比例检测 | Task 4 stage2_deep.py 扩展 |
| Stage 3 GPU 全流程（新增）| Task 5 stage3_gpu.py |
| CMCC 48-GPU Stage 3 启动器（新增）| Task 7 run_stage3_cmcc.py |
| report.py 支持 Stage 3 结果 + caption sidecar | Task 6 report.py 扩展 |

---

## File Map

```
src/sana_wm_pipeline/qc/
  __init__.py              # 新建（若不存在）
  metrics.py               # 新建：Stage 1 纯计算函数（含颜色饱和度 + 摄像机词）
  group_config.py          # 新建：7 个 group 差异化阈值注册表
  stage1_fast.py           # 新建：Stage 1 全量 tar 扫描（multiprocessing）
  stage2_deep.py           # 新建：Stage 2 深度检测（PyAV/场景切割/黑帧/轨迹冻结）
  stage3_gpu.py            # 新建：Stage 3 GPU 评估（UniMatch/DOVER/Qwen + sidecar）
  report.py                # 新建：合并结果 + 三份 manifest + HTML 报告

scripts/run_qc.py          # 新建：Stage 1+2 CLI 入口
scripts/run_stage3_cmcc.py # 新建：CMCC 48-GPU Stage 3 launcher

tests/test_qc_metrics.py       # 新建
tests/test_qc_group_config.py  # 新建
tests/test_qc_stage1.py        # 新建
tests/test_qc_stage2.py        # 新建
tests/test_qc_stage3.py        # 新建（mock 注入，无需真实 GPU）
tests/test_qc_report.py        # 新建

# 只读复用（不改动）：
src/sana_wm_pipeline/stage02_pose/pose_quality.py   # evaluate_pose_quality()
src/sana_wm_pipeline/stage04_filter/apply_table6.py # evaluate(source, scores, cfg)
src/sana_wm_pipeline/stage04_filter/scene_cut.py    # count_scene_cuts(path, threshold, detect_fn)
src/sana_wm_pipeline/stage04_filter/visual_metrics.py # mean_saturation/unimatch_flow_magnitude/dover_score
src/sana_wm_pipeline/stage04_filter/vlm_entity_quality.py # annotate(frames_rgb, vlm_call)
configs/filter_thresholds.yaml                       # per_source 阈值（供 Stage 3 apply_table6 使用）
```

---

## Task 1: `metrics.py` — Stage 1 纯计算函数（含颜色饱和度 + 摄像机词）

**Files:**
- Create: `src/sana_wm_pipeline/qc/__init__.py`
- Create: `src/sana_wm_pipeline/qc/metrics.py`
- Test: `tests/test_qc_metrics.py`

**Interfaces — Produces:**
- `check_so3(poses_c2w) -> tuple[float, float, float]` → `(det_mean, det_std, orth_err_max)`
- `check_first_frame(poses_c2w, atol=0.01) -> tuple[bool, float]`
- `check_trajectory(poses_c2w, jump_threshold_m) -> tuple[float, float, float, int]` → `(total_m, mean_m, max_m, n_jumps)`
- `check_no_nan_inf(arrays: dict[str, np.ndarray]) -> tuple[bool, list[str]]`
- `check_caption(caption, min_len=50) -> tuple[bool, int]`
- `check_color_saturation(frames_rgb: np.ndarray) -> float` → HSV-S 均值，[0,180] range
- `check_caption_camera_words(caption: str) -> list[str]` → 检测到的强动作词列表（空表示无）
- `compute_stage1_metrics(poses, intrinsics, scale, caption, meta_T, image_wh, jump_threshold_m, min_caption_len, frames_rgb=None) -> dict`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_metrics.py
from __future__ import annotations
import numpy as np
import pytest
from sana_wm_pipeline.qc.metrics import (
    check_so3, check_first_frame, check_trajectory,
    check_no_nan_inf, check_caption,
    check_color_saturation, check_caption_camera_words,
    compute_stage1_metrics,
)


def _poses(T: int, step: float = 0.1) -> np.ndarray:
    p = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    p[:, 0, 3] = np.arange(T, dtype=np.float32) * step
    return p


def _intr(T: int, fx: float = 700.0) -> np.ndarray:
    intr = np.zeros((T, 1, 4), dtype=np.float32)
    intr[:, 0, :] = [fx, fx, 640, 360]
    return intr


# --- check_so3 ---
def test_so3_identity():
    det_mean, det_std, orth_err = check_so3(np.tile(np.eye(4, np.float32), (5, 1, 1)))
    assert abs(det_mean - 1.0) < 1e-6 and orth_err < 1e-6

def test_so3_zero_rotation():
    p = np.tile(np.eye(4, np.float32), (5, 1, 1))
    p[:, :3, :3] = 0.0
    det_mean, _, _ = check_so3(p)
    assert abs(det_mean) < 1e-6

# --- check_first_frame ---
def test_first_frame_identity():
    ok, dev = check_first_frame(np.tile(np.eye(4, np.float32), (5, 1, 1)))
    assert ok and dev < 1e-6

def test_first_frame_shifted():
    p = np.tile(np.eye(4, np.float32), (5, 1, 1))
    p[0, 0, 3] = 1.0
    ok, dev = check_first_frame(p)
    assert not ok

# --- check_trajectory ---
def test_trajectory_linear():
    total, mean, mx, n_jumps = check_trajectory(_poses(100, 0.1), 0.5)
    assert abs(total - 9.9) < 1e-3 and n_jumps == 0

def test_trajectory_counts_jumps():
    p = _poses(10, 0.0)
    p[5, 0, 3] = 5.0
    _, _, _, n = check_trajectory(p, 0.5)
    assert n == 1

# --- check_no_nan_inf ---
def test_no_nan_inf_clean():
    ok, reasons = check_no_nan_inf({"a": np.ones((5, 4, 4))})
    assert ok and reasons == []

def test_no_nan_inf_detects_nan():
    a = np.ones((5, 4, 4))
    a[0, 0, 0] = float("nan")
    ok, reasons = check_no_nan_inf({"poses": a})
    assert not ok and any("poses" in r for r in reasons)

# --- check_caption ---
def test_caption_ok():
    ok, length = check_caption("A" * 60)
    assert ok and length >= 50

def test_caption_short():
    ok, _ = check_caption("hi")
    assert not ok

def test_caption_placeholder():
    for s in ["n/a", "N/A", "", "none"]:
        assert not check_caption(s, 50)[0]

# --- check_color_saturation ---
def test_saturation_gray_image():
    gray = np.full((10, 64, 64, 3), 128, dtype=np.uint8)
    s = check_color_saturation(gray)
    assert 0.0 <= s <= 10.0  # gray → near-zero saturation

def test_saturation_red_image():
    red = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    red[..., 0] = 255  # pure red
    s = check_color_saturation(red)
    assert s > 100  # saturated red → high S in HSV

# --- check_caption_camera_words ---
def test_camera_words_none():
    assert check_caption_camera_words("A busy city street at night.") == []

def test_camera_words_detected():
    hits = check_caption_camera_words("The camera pans left across the scene.")
    assert any("pan" in h for h in hits)

def test_camera_words_weak_allowed():
    # "camera stays behind" is a weak framework word, not a strong action word
    hits = check_caption_camera_words("The camera stays behind the character.")
    assert hits == []

def test_camera_words_zoom():
    hits = check_caption_camera_words("zooms in on the building.")
    assert any("zoom" in h for h in hits)

# --- compute_stage1_metrics ---
def test_stage1_metrics_pass():
    T = 100
    result = compute_stage1_metrics(
        poses=_poses(T), intrinsics=_intr(T), scale=np.ones(T, np.float32),
        caption="A " * 60, meta_T=T, image_wh=(1280, 720),
        jump_threshold_m=0.5, min_caption_len=50,
    )
    assert result["t_aligned"] and result["so3_valid"] and result["first_frame_ok"]
    assert result["caption_ok"] and result["n_jumps"] == 0

def test_stage1_metrics_t_mismatch():
    T = 10
    result = compute_stage1_metrics(
        poses=_poses(T), intrinsics=_intr(T), scale=np.ones(T, np.float32),
        caption="A " * 60, meta_T=T + 1, image_wh=(1280, 720),
        jump_threshold_m=0.5, min_caption_len=50,
    )
    assert not result["t_aligned"] and "t_mismatch" in " ".join(result["reasons"])
```

- [ ] **Step 2: 运行，确认失败**
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_qc_metrics.py -v 2>&1 | head -5
```
预期：`ModuleNotFoundError: No module named 'sana_wm_pipeline.qc'`

- [ ] **Step 3: 实现 `__init__.py` 和 `metrics.py`**

`src/sana_wm_pipeline/qc/__init__.py`:
```python
from __future__ import annotations
```

`src/sana_wm_pipeline/qc/metrics.py`:
```python
"""Stage 1 QC: pure computation functions. No IO, no GPU."""
from __future__ import annotations
import re
import numpy as np
from sana_wm_pipeline.stage02_pose.pose_quality import evaluate_pose_quality
from sana_wm_pipeline.stage04_filter.visual_metrics import mean_saturation

_SO3_DET_ATOL = 1e-3
_SO3_ORTH_ATOL = 1e-3
_FIRST_FRAME_ATOL = 1e-2

# Strong camera-action words per paper §4.4 (phrases, not single words, to reduce false positives)
_CAMERA_ACTION_PATTERNS: list[str] = [
    r"\bpan(?:s|ned|ning)?\s+(?:left|right)\b",
    r"\btilt(?:s|ed|ing)?\s+(?:up|down)\b",
    r"\bzoom(?:s|ed|ing)?\s+(?:in|out)\b",
    r"\bdolly\b",
    r"\bcamera\s+(?:moves?|tracks?|follows?|sweeps?|pushes?|pulls?)\b",
    r"\bcamera\s+(?:pan|tilt|roll|orbit)s?\b",
    r"\b(?:tracking|follow)\s+shot\b",
]
_CAMERA_ACTION_RE = [re.compile(p, re.IGNORECASE) for p in _CAMERA_ACTION_PATTERNS]


def check_so3(poses_c2w: np.ndarray) -> tuple[float, float, float]:
    R = poses_c2w[:, :3, :3].astype(np.float64)
    dets = np.linalg.det(R)
    orth_err = float(np.max(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3))))
    return float(dets.mean()), float(dets.std()), orth_err


def check_first_frame(poses_c2w: np.ndarray, atol: float = _FIRST_FRAME_ATOL) -> tuple[bool, float]:
    dev = float(np.max(np.abs(poses_c2w[0].astype(np.float64) - np.eye(4))))
    return dev <= atol, dev


def check_trajectory(poses_c2w: np.ndarray, jump_threshold_m: float) -> tuple[float, float, float, int]:
    t = poses_c2w[:, :3, 3].astype(np.float64)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    if steps.size == 0:
        return 0.0, 0.0, 0.0, 0
    return float(steps.sum()), float(steps.mean()), float(steps.max()), int((steps > jump_threshold_m).sum())


def check_no_nan_inf(arrays: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name, arr in arrays.items():
        if not np.isfinite(np.asarray(arr)).all():
            reasons.append(f"{name}: {int((~np.isfinite(arr)).sum())} non-finite values")
    return len(reasons) == 0, reasons


def check_caption(caption: str, min_len: int = 50) -> tuple[bool, int]:
    s = caption.strip()
    if s.lower() in {"", "n/a", "none", "no caption", "null", "tbd"}:
        return False, len(s)
    return len(s) >= min_len, len(s)


def check_color_saturation(frames_rgb: np.ndarray) -> float:
    """Mean HSV-S across all frames; [0,180] range matching paper Table 6."""
    return mean_saturation(frames_rgb)


def check_caption_camera_words(caption: str) -> list[str]:
    """Return list of matched strong camera-action phrases (empty = clean)."""
    hits: list[str] = []
    for pat in _CAMERA_ACTION_RE:
        m = pat.search(caption)
        if m:
            hits.append(m.group(0))
    return hits


def compute_stage1_metrics(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    scale: np.ndarray,
    caption: str,
    meta_T: int,
    image_wh: tuple[int, int],
    jump_threshold_m: float,
    min_caption_len: int,
    frames_rgb: np.ndarray | None = None,  # optional; if provided → compute saturation
) -> dict:
    """All Stage 1 checks. Returns flat dict of Python scalars/lists for JSON."""
    T = int(poses.shape[0])
    reasons: list[str] = []

    t_aligned = (poses.shape[0] == intrinsics.shape[0] == scale.shape[0] == meta_T)
    if not t_aligned:
        reasons.append(
            f"t_mismatch: poses={poses.shape[0]} intr={intrinsics.shape[0]} "
            f"scale={scale.shape[0]} meta={meta_T}"
        )

    nan_ok, nan_reasons = check_no_nan_inf({"poses": poses, "intrinsics": intrinsics, "scale": scale})
    reasons.extend(nan_reasons)

    det_mean, det_std, orth_err = check_so3(poses)
    so3_valid = abs(det_mean - 1.0) <= _SO3_DET_ATOL and orth_err <= _SO3_ORTH_ATOL
    if not so3_valid:
        reasons.append(f"so3_invalid: det_mean={det_mean:.6f} orth_err={orth_err:.2e}")

    first_ok, first_dev = check_first_frame(poses)
    if not first_ok:
        reasons.append(f"first_frame_dev={first_dev:.4f}")

    traj_total, step_mean, step_max, n_jumps = check_trajectory(poses, jump_threshold_m)

    pqr = evaluate_pose_quality(intrinsics, image_wh, scale)
    if not pqr.passed:
        reasons.extend(list(pqr.reasons))

    cap_ok, cap_len = check_caption(caption, min_caption_len)
    if not cap_ok:
        reasons.append(f"caption_len={cap_len} < {min_caption_len} or placeholder")

    camera_words = check_caption_camera_words(caption)

    saturation = None
    if frames_rgb is not None:
        try:
            saturation = round(check_color_saturation(frames_rgb), 2)
        except Exception as e:
            reasons.append(f"saturation_error: {e}")

    return {
        "T": T,
        "t_aligned": t_aligned,
        "no_nan_inf": nan_ok,
        "so3_valid": so3_valid,
        "det_R_mean": round(det_mean, 8),
        "orth_err_max": float(f"{orth_err:.3e}"),
        "first_frame_ok": first_ok,
        "first_frame_dev": round(first_dev, 6),
        "traj_total_m": round(traj_total, 3),
        "step_mean_m": round(step_mean, 4),
        "step_max_m": round(step_max, 4),
        "n_jumps": n_jumps,
        "jump_threshold_m": jump_threshold_m,
        "fov_ok": pqr.passed or not any("fov" in r.lower() for r in pqr.reasons),
        "focal_div_ok": pqr.focal_divergence_max <= 0.20,
        "focal_div_max": round(pqr.focal_divergence_max, 4),
        "scale_cv": round(pqr.scale_cv, 4),
        "fx_mean": round(float(intrinsics[:, 0, 0].mean()), 2),
        "caption_ok": cap_ok,
        "caption_len": cap_len,
        "camera_words": camera_words,
        "saturation": saturation,
        "scale_all_ones": bool(np.all(scale == 1.0)),
        "reasons": reasons,
    }
```

- [ ] **Step 4: 运行测试，确认通过**
```bash
python -m pytest tests/test_qc_metrics.py -v
```
预期：全部 PASSED

- [ ] **Step 5: Commit**
```bash
git add src/sana_wm_pipeline/qc/__init__.py src/sana_wm_pipeline/qc/metrics.py tests/test_qc_metrics.py
git commit -m "feat(qc): add Stage 1 pure metrics library (saturation + camera words)"
```

---

## Task 2: `group_config.py` — 7 个 group 差异化阈值注册表

**Files:**
- Create: `src/sana_wm_pipeline/qc/group_config.py`
- Test: `tests/test_qc_group_config.py`

**Interfaces — Produces:**
- `GroupConfig` dataclass（所有 Stage 1/2 阈值字段）
- `get_group_config(group_name: str) -> GroupConfig`
- `VERDICT_FAIL`, `VERDICT_FLAG`, `VERDICT_PASS` 常量
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

ALL_GROUPS = [
    "wds-DL3DV-ALL-2K", "wds-sekai-real-walking-hq", "wds-OmniWorld-Game",
    "wds-SpatialVID-hq", "wds-RealEstate10K-360p",
    "wds-sekai-game-drone", "wds-sekai-game-walking",
]


def _clean(T=100) -> dict:
    return {
        "T": T, "t_aligned": True, "no_nan_inf": True, "so3_valid": True,
        "first_frame_ok": True, "fov_ok": True, "focal_div_ok": True,
        "caption_ok": True, "n_jumps": 0, "camera_words": [], "reasons": [],
    }


def test_all_seven_groups_have_config():
    for g in ALL_GROUPS:
        assert isinstance(get_group_config(g), GroupConfig)

def test_unknown_group_fallback():
    assert isinstance(get_group_config("wds-unknown-xyz"), GroupConfig)

def test_game_groups_more_lenient():
    game = get_group_config("wds-OmniWorld-Game")
    real = get_group_config("wds-DL3DV-ALL-2K")
    assert game.max_jumps_fail > real.max_jumps_fail
    assert game.jump_threshold_m > real.jump_threshold_m

def test_game_max_jumps_fail_is_50():
    for g in ["wds-OmniWorld-Game", "wds-sekai-game-walking"]:
        assert get_group_config(g).max_jumps_fail == 50

def test_drone_max_jumps_fail_is_80():
    assert get_group_config("wds-sekai-game-drone").max_jumps_fail == 80

def test_verdict_pass_clean():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    verdict, _ = compute_verdict(_clean(), cfg)
    assert verdict == VERDICT_PASS

def test_verdict_fail_so3():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["so3_valid"] = False
    verdict, _ = compute_verdict(m, cfg)
    assert verdict == VERDICT_FAIL

def test_verdict_fail_too_many_jumps():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["n_jumps"] = cfg.max_jumps_fail + 1
    assert compute_verdict(m, cfg)[0] == VERDICT_FAIL

def test_verdict_flag_moderate_jumps():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["n_jumps"] = 1  # max_jumps_flag=0 for DL3DV
    assert compute_verdict(m, cfg)[0] == VERDICT_FLAG

def test_verdict_flag_camera_words():
    cfg = get_group_config("wds-sekai-real-walking-hq")
    m = _clean(); m["camera_words"] = ["pans left"]
    verdict, reasons = compute_verdict(m, cfg)
    assert verdict == VERDICT_FLAG
    assert any("camera_word" in r for r in reasons)

def test_game_tolerates_moderate_jumps():
    cfg = get_group_config("wds-OmniWorld-Game")
    m = _clean(); m["n_jumps"] = 10
    assert compute_verdict(m, cfg)[0] == VERDICT_PASS

def test_saturation_flag_for_real_data():
    cfg = get_group_config("wds-DL3DV-ALL-2K")
    m = _clean(); m["saturation"] = 1.0  # below min (configured as 0)
    # DL3DV has color_saturation check [0,180], 1.0 should pass
    assert compute_verdict(m, cfg)[0] == VERDICT_PASS
```

- [ ] **Step 2: 运行，确认失败**
```bash
python -m pytest tests/test_qc_group_config.py -v 2>&1 | head -5
```

- [ ] **Step 3: 实现 `group_config.py`**

```python
# src/sana_wm_pipeline/qc/group_config.py
"""Per-group QC threshold config and Stage 1 verdict logic."""
from __future__ import annotations
from dataclasses import dataclass, field

VERDICT_PASS = "pass"
VERDICT_FLAG = "flag"
VERDICT_FAIL = "fail"

_HARD_FAIL_KEYS = ("t_aligned", "no_nan_inf", "so3_valid", "first_frame_ok")


@dataclass(frozen=True)
class GroupConfig:
    # Stage 1 pose thresholds
    jump_threshold_m: float
    max_jumps_flag: int       # n_jumps > this → FLAG
    max_jumps_fail: int       # n_jumps > this → FAIL
    min_caption_len: int
    # Stage 1 saturation: None = not checked
    saturation_min: float | None = None
    saturation_max: float | None = None
    # Stage 2: None = not checked
    max_scene_cuts: int | None = None
    # Stage 3 source key for filter_thresholds.yaml (None = skip Stage 3 table6)
    table6_source: str | None = None


# ── Registry ─────────────────────────────────────────────────────────────────

_REAL_STRICT = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=0, max_jumps_fail=5,
    min_caption_len=50, saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=None, table6_source="DL3DV",
)
_REALESTATE = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=0, max_jumps_fail=5,
    min_caption_len=50, saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=1, table6_source="RealEstate10K",
)
_SEKAI_WALKING = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=3, max_jumps_fail=15,
    min_caption_len=50, saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=None, table6_source="Sekai_Walking",
)
_SPATIALVID = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=0, max_jumps_fail=5,
    min_caption_len=50, saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=None, table6_source="SpatialVID",
)
_OMNIWORLD = GroupConfig(
    jump_threshold_m=2.0, max_jumps_flag=15, max_jumps_fail=50,
    min_caption_len=50, saturation_min=None, saturation_max=None,
    max_scene_cuts=None, table6_source="OmniWorld",
)
_SEKAI_DRONE = GroupConfig(
    jump_threshold_m=5.0, max_jumps_flag=20, max_jumps_fail=80,
    min_caption_len=50, saturation_min=None, saturation_max=None,
    max_scene_cuts=None, table6_source="Sekai_Game_Drone",
)
_SEKAI_GAME_WALKING = GroupConfig(
    jump_threshold_m=2.0, max_jumps_flag=15, max_jumps_fail=50,
    min_caption_len=50, saturation_min=None, saturation_max=None,
    max_scene_cuts=None, table6_source="Sekai_Game_Walking",
)
_DEFAULT = GroupConfig(
    jump_threshold_m=1.0, max_jumps_flag=10, max_jumps_fail=50,
    min_caption_len=50, table6_source=None,
)

_REGISTRY: dict[str, GroupConfig] = {
    "wds-DL3DV-ALL-2K": _REAL_STRICT,
    "wds-sekai-real-walking-hq": _SEKAI_WALKING,
    "wds-OmniWorld-Game": _OMNIWORLD,
    "wds-SpatialVID-hq": _SPATIALVID,
    "wds-RealEstate10K-360p": _REALESTATE,
    "wds-sekai-game-drone": _SEKAI_DRONE,
    "wds-sekai-game-walking": _SEKAI_GAME_WALKING,
}


def get_group_config(group_name: str) -> GroupConfig:
    return _REGISTRY.get(group_name, _DEFAULT)


def compute_verdict(metrics: dict, cfg: GroupConfig) -> tuple[str, list[str]]:
    """Classify Stage 1 metrics as PASS / FLAG / FAIL."""
    flag_reasons: list[str] = []

    # Hard structural failures
    for key in _HARD_FAIL_KEYS:
        if not metrics.get(key, True):
            return VERDICT_FAIL, list(metrics.get("reasons", []))
    if not metrics.get("fov_ok", True) or not metrics.get("focal_div_ok", True):
        return VERDICT_FAIL, list(metrics.get("reasons", []))

    # Jump count
    n_jumps = metrics.get("n_jumps", 0)
    if n_jumps > cfg.max_jumps_fail:
        return VERDICT_FAIL, [f"n_jumps={n_jumps} > max_jumps_fail={cfg.max_jumps_fail}"]
    if n_jumps > cfg.max_jumps_flag:
        flag_reasons.append(f"n_jumps={n_jumps} > max_jumps_flag={cfg.max_jumps_flag}")

    # Caption basic quality
    if not metrics.get("caption_ok", True):
        flag_reasons.extend([r for r in metrics.get("reasons", []) if "caption" in r])

    # Camera action words → flag for Qwen rewrite in Stage 3
    cw = metrics.get("camera_words", [])
    if cw:
        flag_reasons.append(f"camera_word: {cw[0]!r} (+{len(cw)-1} more)")

    # Saturation (only for groups with saturation check, only flag not fail)
    sat = metrics.get("saturation")
    if sat is not None and cfg.saturation_min is not None:
        if not (cfg.saturation_min <= sat <= cfg.saturation_max):
            flag_reasons.append(f"saturation={sat:.1f} out of [{cfg.saturation_min},{cfg.saturation_max}]")

    return (VERDICT_FLAG, flag_reasons) if flag_reasons else (VERDICT_PASS, [])
```

- [ ] **Step 4: 运行测试**
```bash
python -m pytest tests/test_qc_group_config.py -v
```
预期：全部 PASSED

- [ ] **Step 5: Commit**
```bash
git add src/sana_wm_pipeline/qc/group_config.py tests/test_qc_group_config.py
git commit -m "feat(qc): 7-group config with corrected game thresholds (max_jumps_fail=50)"
```

---

## Task 3: `stage1_fast.py` — Stage 1 全量 tar 扫描

**Files:**
- Create: `src/sana_wm_pipeline/qc/stage1_fast.py`
- Test: `tests/test_qc_stage1.py`

**Interfaces — Produces:**
- `scan_tar(tar_path, group_name, read_video_frames=False) -> list[dict]`
- `run_stage1(tar_paths, group_name, output_jsonl, n_workers=32, read_video_frames=False) -> int`

**注意：** `read_video_frames=False` 默认不解码视频帧（颜色饱和度可选）；CMCC 生产时传 `True` 以启用颜色饱和度检测。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_stage1.py
from __future__ import annotations
import io, json, tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage1_fast import scan_tar, run_stage1

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s1") / "shard-000000.tar"
    exts = [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]
    with tarfile.open(p, "w") as tf:
        for ext in exts:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p


@pytest.fixture(scope="session")
def corrupt_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s1c") / "shard-000000.tar"
    with tarfile.open(p, "w") as tf:
        for ext in [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p  # missing .scale.npy → fail


def test_scan_tar_returns_list(tiny_tar):
    results = scan_tar(tiny_tar, GROUP)
    assert isinstance(results, list) and len(results) == 1

def test_scan_tar_has_keys(tiny_tar):
    r = scan_tar(tiny_tar, GROUP)[0]
    for k in ("sample_id", "group", "tar_path", "verdict", "flag_reasons", "metrics"):
        assert k in r

def test_scan_tar_sample_id(tiny_tar):
    assert scan_tar(tiny_tar, GROUP)[0]["sample_id"] == SAMPLE_ID

def test_scan_tar_testdata_passes(tiny_tar):
    r = scan_tar(tiny_tar, GROUP)[0]
    assert r["metrics"]["so3_valid"] and r["metrics"]["first_frame_ok"]
    assert r["verdict"] in ("pass", "flag")  # OmniWorld may flag for jumps

def test_scan_tar_missing_file_fails(corrupt_tar):
    r = scan_tar(corrupt_tar, GROUP)[0]
    assert r["verdict"] == "fail"
    assert any("missing" in x.lower() for x in r["flag_reasons"])

def test_run_stage1_writes_jsonl(tiny_tar, tmp_path):
    out = tmp_path / "s1.jsonl"
    count = run_stage1([tiny_tar], GROUP, out, n_workers=1)
    assert count == 1 and out.exists()
    assert json.loads(out.read_text().splitlines()[0])["sample_id"] == SAMPLE_ID

def test_run_stage1_two_tars(tiny_tar, tmp_path):
    out = tmp_path / "s1.jsonl"
    count = run_stage1([tiny_tar, tiny_tar], GROUP, out, n_workers=1)
    assert count == 2 and len(out.read_text().strip().splitlines()) == 2
```

- [ ] **Step 2: 运行，确认失败**
```bash
python -m pytest tests/test_qc_stage1.py -v 2>&1 | head -5
```

- [ ] **Step 3: 实现 `stage1_fast.py`**

```python
# src/sana_wm_pipeline/qc/stage1_fast.py
"""Stage 1: fast full-coverage scan of tar shards (CPU-only, multiprocessing)."""
from __future__ import annotations
import io, json, tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import numpy as np
from sana_wm_pipeline.qc.metrics import compute_stage1_metrics
from sana_wm_pipeline.qc.group_config import get_group_config, compute_verdict

_REQUIRED_EXTS = {".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"}
IMAGE_WH = (1280, 720)


def _extract_samples_from_tar(tar_path: Path) -> dict[str, dict[str, bytes]]:
    samples: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            name = member.name
            stem, sep, suffix = name.partition(".")
            if not sep:
                continue
            ext = "." + suffix
            f = tf.extractfile(member)
            if f is None:
                continue
            samples.setdefault(stem, {})[ext] = f.read()
    return samples


def _decode_video_frames(mp4_bytes: bytes) -> np.ndarray | None:
    """Decode mp4 → (T,H,W,3) uint8 RGB. Returns None on failure."""
    try:
        import av
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as container:
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    frames.append(frame.to_ndarray(format="rgb24"))
        return np.array(frames) if frames else None
    except Exception:
        return None


def _scan_one_sample(
    sample_id: str,
    file_bytes: dict[str, bytes],
    group_name: str,
    tar_path: str,
    read_video_frames: bool,
) -> dict[str, Any]:
    cfg = get_group_config(group_name)
    missing = [ext for ext in _REQUIRED_EXTS if ext not in file_bytes]
    if missing:
        return {
            "sample_id": sample_id, "group": group_name, "tar_path": tar_path,
            "verdict": "fail", "flag_reasons": [f"missing files: {missing}"], "metrics": {},
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
            "sample_id": sample_id, "group": group_name, "tar_path": tar_path,
            "verdict": "fail", "flag_reasons": [f"load_error: {exc}"], "metrics": {},
        }

    frames_rgb = None
    if read_video_frames and cfg.saturation_min is not None:
        frames_rgb = _decode_video_frames(file_bytes[".mp4"])

    metrics = compute_stage1_metrics(
        poses=poses, intrinsics=intrinsics, scale=scale,
        caption=caption, meta_T=meta_T, image_wh=IMAGE_WH,
        jump_threshold_m=cfg.jump_threshold_m,
        min_caption_len=cfg.min_caption_len,
        frames_rgb=frames_rgb,
    )
    verdict, flag_reasons = compute_verdict(metrics, cfg)
    return {
        "sample_id": sample_id, "group": group_name, "tar_path": str(tar_path),
        "verdict": verdict, "flag_reasons": flag_reasons, "metrics": metrics,
    }


def scan_tar(tar_path: Path, group_name: str, read_video_frames: bool = False) -> list[dict]:
    tar_path = Path(tar_path)
    samples = _extract_samples_from_tar(tar_path)
    return [
        _scan_one_sample(sid, fb, group_name, str(tar_path), read_video_frames)
        for sid, fb in samples.items()
    ]


def _worker_fn(args: tuple) -> list[dict]:
    tar_path, group_name, read_video_frames = args
    return scan_tar(Path(tar_path), group_name, read_video_frames)


def run_stage1(
    tar_paths: list[Path],
    group_name: str,
    output_jsonl: Path,
    n_workers: int = 32,
    read_video_frames: bool = False,
) -> int:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    worker_args = [(str(p), group_name, read_video_frames) for p in tar_paths]
    total = 0
    with open(output_jsonl, "w") as fout:
        with Pool(processes=min(n_workers, len(tar_paths) or 1)) as pool:
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

- [ ] **Step 5: Commit**
```bash
git add src/sana_wm_pipeline/qc/stage1_fast.py tests/test_qc_stage1.py
git commit -m "feat(qc): Stage 1 tar scanner with 11-check pipeline"
```

---

## Task 4: `stage2_deep.py` — 深度检测（PyAV + 场景切割 + 黑帧 + 轨迹冻结）

**Files:**
- Create: `src/sana_wm_pipeline/qc/stage2_deep.py`
- Test: `tests/test_qc_stage2.py`

**Interfaces — Produces:**
- `count_video_frames_av(video_bytes) -> int`
- `check_black_frame_ratio(video_bytes, brightness_threshold=10) -> float`
- `check_trajectory_frozen(poses_c2w, frozen_threshold=1e-4) -> tuple[bool, float]`
- `count_scene_cuts_from_bytes(video_bytes, threshold=27.0) -> int`（写临时文件后调 scene_cut.py）
- `deep_check_sample(sample_id, tar_path, group_name) -> dict`
- `run_stage2(stage1_jsonl, output_jsonl, sample_frac=0.05, n_workers=16) -> int`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_stage2.py
from __future__ import annotations
import io, json, tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage2_deep import (
    count_video_frames_av, check_black_frame_ratio,
    check_trajectory_frozen, deep_check_sample, run_stage2,
)

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"
EXPECTED_T = 782  # confirmed from testdata analysis


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s2") / "shard.tar"
    with tarfile.open(p, "w") as tf:
        for ext in [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p


@pytest.fixture(scope="session")
def video_bytes():
    return (TESTDATA / (SAMPLE_ID + ".mp4")).read_bytes()


def test_count_frames_av(video_bytes):
    T = count_video_frames_av(video_bytes)
    assert T == EXPECTED_T

def test_black_frame_ratio_normal_video(video_bytes):
    ratio = check_black_frame_ratio(video_bytes)
    assert 0.0 <= ratio <= 0.3  # normal video should be mostly non-black

def test_trajectory_not_frozen():
    p = np.tile(np.eye(4, np.float32), (100, 1, 1))
    p[:, 0, 3] = np.arange(100, dtype=np.float32) * 0.1
    frozen, ratio = check_trajectory_frozen(p)
    assert not frozen and ratio < 0.01

def test_trajectory_frozen_detected():
    p = np.tile(np.eye(4, np.float32), (100, 1, 1))
    p[10:90, 0, 3] = 5.0  # 80% frames have same position
    frozen, ratio = check_trajectory_frozen(p)
    assert frozen and ratio > 0.7

def test_deep_check_structure(tiny_tar):
    r = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert "sample_id" in r and "stage2" in r
    for k in ("video_T", "video_T_matches_npy", "traj_frozen", "frozen_ratio",
               "black_frame_ratio", "scene_cuts", "reasons"):
        assert k in r["stage2"], f"missing key: {k}"

def test_deep_check_video_T(tiny_tar):
    r = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert r["stage2"]["video_T"] == EXPECTED_T
    assert r["stage2"]["video_T_matches_npy"] is True

def test_deep_check_not_frozen(tiny_tar):
    r = deep_check_sample(SAMPLE_ID, tiny_tar, GROUP)
    assert r["stage2"]["traj_frozen"] is False

def test_run_stage2_processes_flag(tiny_tar, tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(json.dumps({
        "sample_id": SAMPLE_ID, "group": GROUP, "tar_path": str(tiny_tar),
        "verdict": "flag", "flag_reasons": ["n_jumps=5"], "metrics": {"T": EXPECTED_T},
    }) + "\n")
    out = tmp_path / "s2.jsonl"
    count = run_stage2(s1, out, sample_frac=0.0, n_workers=1)
    assert count == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert "stage2" in rec

def test_run_stage2_skips_fail(tiny_tar, tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(json.dumps({
        "sample_id": SAMPLE_ID, "group": GROUP, "tar_path": str(tiny_tar),
        "verdict": "fail", "flag_reasons": ["so3"], "metrics": {},
    }) + "\n")
    out = tmp_path / "s2.jsonl"
    count = run_stage2(s1, out, sample_frac=0.0, n_workers=1)
    assert count == 0
```

- [ ] **Step 2: 运行，确认失败**
```bash
python -m pytest tests/test_qc_stage2.py -v 2>&1 | head -5
```

- [ ] **Step 3: 实现 `stage2_deep.py`**

```python
# src/sana_wm_pipeline/qc/stage2_deep.py
"""Stage 2: deep targeted checks (PyAV, scene cut, black frames, frozen trajectory)."""
from __future__ import annotations
import io, json, random, tarfile, tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import numpy as np

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

from sana_wm_pipeline.stage04_filter.scene_cut import count_scene_cuts

_FROZEN_THRESHOLD = 1e-4
_BLACK_BRIGHTNESS = 10  # pixel mean < this → black frame


def count_video_frames_av(video_bytes: bytes) -> int:
    if not _AV_AVAILABLE:
        raise RuntimeError("pip install av")
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        if stream.frames > 0:
            return int(stream.frames)
        return sum(1 for _ in container.decode(video=0))


def check_black_frame_ratio(video_bytes: bytes, brightness_threshold: int = _BLACK_BRIGHTNESS) -> float:
    """Fraction of frames with mean brightness < threshold."""
    if not _AV_AVAILABLE:
        return 0.0
    total, black = 0, 0
    with av.open(io.BytesIO(video_bytes)) as container:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="gray")
            if arr.mean() < brightness_threshold:
                black += 1
            total += 1
    return black / total if total else 0.0


def check_trajectory_frozen(poses_c2w: np.ndarray, frozen_threshold: float = _FROZEN_THRESHOLD) -> tuple[bool, float]:
    t = poses_c2w[:, :3, 3].astype(np.float64)
    if len(t) < 2:
        return False, 0.0
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    ratio = float((steps < frozen_threshold).mean())
    return ratio > 0.5, ratio


def count_scene_cuts_from_bytes(video_bytes: bytes, threshold: float = 27.0) -> int:
    """Write mp4 to temp file, run PySceneDetect, return cut count."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        tmp_path = f.name
    try:
        return count_scene_cuts(tmp_path, threshold=threshold)
    except Exception:
        return -1  # -1 = unavailable
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def deep_check_sample(sample_id: str, tar_path: Path, group_name: str) -> dict[str, Any]:
    from sana_wm_pipeline.qc.group_config import get_group_config
    tar_path = Path(tar_path)
    cfg = get_group_config(group_name)
    stage2: dict[str, Any] = {
        "video_T": -1, "video_T_matches_npy": None,
        "black_frame_ratio": None, "scene_cuts": None,
        "traj_frozen": None, "frozen_ratio": None, "reasons": [],
    }
    try:
        with tarfile.open(tar_path, "r") as tf:
            # Video bytes
            try:
                video_bytes = tf.extractfile(tf.getmember(f"{sample_id}.mp4")).read()
            except KeyError:
                stage2["reasons"].append("mp4_not_found")
                return {"sample_id": sample_id, "stage2": stage2}

            # Frame count
            try:
                video_T = count_video_frames_av(video_bytes)
                stage2["video_T"] = video_T
            except Exception as e:
                stage2["reasons"].append(f"av_error: {e}")
                video_T = -1

            # Black frame ratio
            try:
                stage2["black_frame_ratio"] = round(check_black_frame_ratio(video_bytes), 4)
                if stage2["black_frame_ratio"] > 0.30:
                    stage2["reasons"].append(f"black_frame_ratio={stage2['black_frame_ratio']:.2f} > 0.30")
            except Exception:
                pass

            # Scene cuts (only for groups with max_scene_cuts limit)
            if cfg.max_scene_cuts is not None:
                n_cuts = count_scene_cuts_from_bytes(video_bytes)
                stage2["scene_cuts"] = n_cuts
                if n_cuts > cfg.max_scene_cuts:
                    stage2["reasons"].append(f"scene_cuts={n_cuts} > {cfg.max_scene_cuts}")

            # Trajectory frozen
            try:
                poses = np.load(io.BytesIO(tf.extractfile(
                    tf.getmember(f"{sample_id}.poses_c2w.npy")).read()))
                npy_T = int(poses.shape[0])
                stage2["video_T_matches_npy"] = (video_T == npy_T)
                if video_T > 0 and video_T != npy_T:
                    stage2["reasons"].append(f"video_npy_T_mismatch: video={video_T} npy={npy_T}")
                frozen, ratio = check_trajectory_frozen(poses)
                stage2["traj_frozen"] = frozen
                stage2["frozen_ratio"] = round(ratio, 4)
                if frozen:
                    stage2["reasons"].append(f"traj_frozen: {ratio:.1%} frames stationary")
            except Exception as e:
                stage2["reasons"].append(f"poses_error: {e}")

    except Exception as e:
        stage2["reasons"].append(f"tar_error: {e}")

    return {"sample_id": sample_id, "stage2": stage2}


def _worker_fn(args: tuple) -> dict:
    sid, tar_path, group_name = args
    return deep_check_sample(sid, Path(tar_path), group_name)


def run_stage2(
    stage1_jsonl: Path, output_jsonl: Path,
    sample_frac: float = 0.05, n_workers: int = 16,
) -> int:
    stage1_jsonl, output_jsonl = Path(stage1_jsonl), Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, str]] = []
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

    with open(output_jsonl, "w") as fout:
        with Pool(processes=min(n_workers, len(selected))) as pool:
            for s2_rec in pool.imap_unordered(_worker_fn, [(s, t, g) for s, t, g in selected]):
                merged = dict(all_records[s2_rec["sample_id"]])
                merged["stage2"] = s2_rec["stage2"]
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

    return len(selected)
```

- [ ] **Step 4: 运行测试**
```bash
python -m pytest tests/test_qc_stage2.py -v
```

- [ ] **Step 5: Commit**
```bash
git add src/sana_wm_pipeline/qc/stage2_deep.py tests/test_qc_stage2.py
git commit -m "feat(qc): Stage 2 deep checks (scene cut, black frames, frozen traj)"
```

---

## Task 5: `stage3_gpu.py` — UniMatch + DOVER + Qwen3.5-27B

**Files:**
- Create: `src/sana_wm_pipeline/qc/stage3_gpu.py`
- Test: `tests/test_qc_stage3.py`

**Context:**
- Qwen3.5-27B 权重已在 CMCC: `~/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL/`
- UniMatch 代码在 CMCC: `~/work/filestorage/shangaoooooo/davidwang/UniMatch/`
- DOVER pip install 后可用
- 复用 `stage04_filter/visual_metrics.py`（UniMatch flow, DOVER score, saturation）
- 复用 `stage04_filter/vlm_entity_quality.py`（Qwen entity + quality + caption rewrite）
- 复用 `stage04_filter/apply_table6.py`（Table 6 阈值评估）
- Caption 改写结果写 `qc_output/caption_overrides.jsonl`（sidecar）

**Interfaces — Produces:**
- `load_unimatch_fn(model_dir: str) -> Callable`（返回 flow_fn(img_a, img_b) → (H,W,2) float）
- `load_dover_fn(device: str) -> Callable`（返回 dover_fn(frames_rgb) → float）
- `load_qwen_fn(model_dir: str, device: str) -> Callable`（返回 vlm_call(prompt, keyframes) → str）
- `process_sample_stage3(sample_id, tar_path, group_name, flow_fn, dover_fn, vlm_call, table6_cfg, has_camera_words) -> dict`
- `run_stage3(stage1_jsonl, output_jsonl, caption_overrides_jsonl, flow_fn, dover_fn, vlm_call, table6_cfg) -> int`

- [ ] **Step 1: 写失败测试（全部用 mock 注入，无需真实 GPU）**

```python
# tests/test_qc_stage3.py
from __future__ import annotations
import io, json, tarfile
from pathlib import Path
import numpy as np
import pytest
from sana_wm_pipeline.qc.stage3_gpu import process_sample_stage3, run_stage3

TESTDATA = Path(__file__).parent.parent / "testdata"
SAMPLE_ID = "OmniWorld-Game_1f79eb96f021__splits_013-015"
GROUP = "wds-OmniWorld-Game"


@pytest.fixture(scope="session")
def tiny_tar(tmp_path_factory):
    p = tmp_path_factory.mktemp("s3") / "shard.tar"
    with tarfile.open(p, "w") as tf:
        for ext in [".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"]:
            tf.add(TESTDATA / (SAMPLE_ID + ext), arcname=SAMPLE_ID + ext)
    return p


def _mock_flow_fn(img_a, img_b):
    H, W = img_a.shape[:2]
    return np.ones((H, W, 2), dtype=np.float32) * 10.0  # constant flow = 10px


def _mock_dover_fn(frames_rgb):
    return 0.7  # reasonable quality score


def _mock_vlm_call(prompt, keyframes):
    return json.dumps({
        "people": 2, "vehicles": 0, "animals": 0,
        "quality": 1.0, "too_dark": False, "blurry": False,
    })


@pytest.fixture
def table6_cfg():
    from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds
    cfg_path = Path(__file__).parent.parent / "configs" / "filter_thresholds.yaml"
    return load_thresholds(cfg_path)


def test_process_sample_stage3_structure(tiny_tar, table6_cfg):
    result = process_sample_stage3(
        SAMPLE_ID, tiny_tar, GROUP,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=_mock_vlm_call,
        table6_cfg=table6_cfg, has_camera_words=False,
    )
    assert "sample_id" in result and "stage3" in result
    s3 = result["stage3"]
    for k in ("unimatch_flow", "dover", "vlm_entity_count", "vlm_quality", "table6_accepted", "reasons"):
        assert k in s3, f"missing key: {k}"


def test_process_sample_stage3_mock_values(tiny_tar, table6_cfg):
    result = process_sample_stage3(
        SAMPLE_ID, tiny_tar, GROUP,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=_mock_vlm_call,
        table6_cfg=table6_cfg, has_camera_words=False,
    )
    s3 = result["stage3"]
    assert s3["dover"] == pytest.approx(0.7, abs=0.01)
    assert s3["unimatch_flow"] > 0


def test_process_sample_caption_rewrite_when_camera_words(tiny_tar, table6_cfg):
    def vlm_with_rewrite(prompt, kf):
        r = json.loads(_mock_vlm_call(prompt, kf))
        if "rewrite" in prompt.lower():
            r["caption_revised"] = "A city street scene."
        return json.dumps(r)
    result = process_sample_stage3(
        SAMPLE_ID, tiny_tar, GROUP,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=vlm_with_rewrite,
        table6_cfg=table6_cfg, has_camera_words=True,
    )
    assert "caption_revised" in result["stage3"]


def test_run_stage3_end_to_end(tiny_tar, table6_cfg, tmp_path):
    s1 = tmp_path / "s1.jsonl"
    s1.write_text(json.dumps({
        "sample_id": SAMPLE_ID, "group": GROUP, "tar_path": str(tiny_tar),
        "verdict": "pass", "flag_reasons": [], "metrics": {"camera_words": []},
    }) + "\n")
    out = tmp_path / "s3.jsonl"
    cap_out = tmp_path / "caption_overrides.jsonl"
    count = run_stage3(
        s1, out, cap_out,
        flow_fn=_mock_flow_fn, dover_fn=_mock_dover_fn, vlm_call=_mock_vlm_call,
        table6_cfg=table6_cfg,
    )
    assert count == 1 and out.exists()
    rec = json.loads(out.read_text().splitlines()[0])
    assert "stage3" in rec
```

- [ ] **Step 2: 运行，确认失败**
```bash
python -m pytest tests/test_qc_stage3.py -v 2>&1 | head -5
```

- [ ] **Step 3: 实现 `stage3_gpu.py`**

```python
# src/sana_wm_pipeline/qc/stage3_gpu.py
"""Stage 3: GPU-accelerated visual quality evaluation.

All heavy models (UniMatch, DOVER, Qwen) are injected as callables so the
module is importable without GPU. Use load_*_fn() helpers in the CMCC launcher.
"""
from __future__ import annotations
import io, json, tarfile
from pathlib import Path
from typing import Any, Callable
import numpy as np

from sana_wm_pipeline.stage04_filter.visual_metrics import (
    unimatch_flow_magnitude, dover_score, mean_saturation,
)
from sana_wm_pipeline.stage04_filter.vlm_entity_quality import (
    annotate, ENTITY_QUALITY_PROMPT,
)
from sana_wm_pipeline.stage04_filter.apply_table6 import evaluate
from sana_wm_pipeline.qc.group_config import get_group_config

_CAPTION_REWRITE_SUFFIX = (
    "\n\nAdditionally, the caption below contains camera motion words "
    "(e.g., 'pans left', 'zooms in'). Rewrite it as a static scene description "
    "with no camera motion words. Output the rewritten caption in a JSON field "
    "\"caption_revised\" alongside the other fields."
)


def _decode_frames(mp4_bytes: bytes) -> np.ndarray | None:
    try:
        import av
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as c:
            for pkt in c.demux(video=0):
                for f in pkt.decode():
                    frames.append(f.to_ndarray(format="rgb24"))
        return np.array(frames, dtype=np.uint8) if frames else None
    except Exception:
        return None


def process_sample_stage3(
    sample_id: str,
    tar_path: Path,
    group_name: str,
    flow_fn: Callable,
    dover_fn: Callable,
    vlm_call: Callable,
    table6_cfg: dict,
    has_camera_words: bool = False,
) -> dict[str, Any]:
    """Run Stage 3 GPU checks on one sample. Returns merged result dict."""
    tar_path = Path(tar_path)
    cfg = get_group_config(group_name)
    stage3: dict[str, Any] = {
        "unimatch_flow": None, "dover": None,
        "vlm_entity_count": None, "vlm_quality": None,
        "table6_accepted": None, "caption_revised": None,
        "reasons": [],
    }

    try:
        with tarfile.open(tar_path, "r") as tf:
            mp4_bytes = tf.extractfile(tf.getmember(f"{sample_id}.mp4")).read()
            cap_bytes = tf.extractfile(tf.getmember(f"{sample_id}.caption.txt")).read()
    except Exception as e:
        stage3["reasons"].append(f"tar_read_error: {e}")
        return {"sample_id": sample_id, "stage3": stage3}

    frames_rgb = _decode_frames(mp4_bytes)
    if frames_rgb is None:
        stage3["reasons"].append("video_decode_failed")
        return {"sample_id": sample_id, "stage3": stage3}

    # UniMatch flow
    try:
        flow_val = unimatch_flow_magnitude(frames_rgb, flow_fn)
        stage3["unimatch_flow"] = round(float(flow_val), 3) if not np.isnan(flow_val) else None
    except Exception as e:
        stage3["reasons"].append(f"unimatch_error: {e}")

    # DOVER quality
    try:
        dover_val = dover_score(frames_rgb, dover_fn)
        stage3["dover"] = round(float(dover_val), 4) if not np.isnan(dover_val) else None
    except Exception as e:
        stage3["reasons"].append(f"dover_error: {e}")

    # Qwen VLM (entity + quality + optional caption rewrite)
    try:
        prompt = ENTITY_QUALITY_PROMPT
        if has_camera_words:
            prompt = prompt + _CAPTION_REWRITE_SUFFIX
        raw = vlm_call(prompt, [frames_rgb[i] for i in np.linspace(0, len(frames_rgb)-1, 8).astype(int)])
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        entity_count = int(parsed.get("people", 0)) + int(parsed.get("vehicles", 0)) + int(parsed.get("animals", 0))
        stage3["vlm_entity_count"] = entity_count
        stage3["vlm_quality"] = float(parsed.get("quality", -1.0))
        if has_camera_words and "caption_revised" in parsed:
            stage3["caption_revised"] = str(parsed["caption_revised"])
    except Exception as e:
        stage3["reasons"].append(f"vlm_error: {e}")

    # Table 6 evaluation
    if cfg.table6_source is not None:
        scores = {
            "unimatch_flow": stage3.get("unimatch_flow"),
            "dover": stage3.get("dover"),
            "vlm_entity_count": stage3.get("vlm_entity_count"),
            "vlm_quality": stage3.get("vlm_quality"),
            "color_saturation": round(mean_saturation(frames_rgb), 2),
        }
        try:
            t6_result = evaluate(cfg.table6_source, scores, table6_cfg)
            stage3["table6_accepted"] = t6_result["accepted"]
            if not t6_result["accepted"]:
                stage3["reasons"].extend(t6_result["reasons"])
        except KeyError:
            stage3["reasons"].append(f"table6_unknown_source: {cfg.table6_source}")

    return {"sample_id": sample_id, "stage3": stage3}


def run_stage3(
    stage1_jsonl: Path,
    output_jsonl: Path,
    caption_overrides_jsonl: Path,
    flow_fn: Callable,
    dover_fn: Callable,
    vlm_call: Callable,
    table6_cfg: dict,
) -> int:
    """Run Stage 3 on all non-failed samples from Stage 1. Single-process (GPU caller)."""
    stage1_jsonl = Path(stage1_jsonl)
    output_jsonl = Path(output_jsonl)
    caption_overrides_jsonl = Path(caption_overrides_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(stage1_jsonl) as fin, \
         open(output_jsonl, "w") as fout, \
         open(caption_overrides_jsonl, "w") as cap_fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get("verdict") == "fail":
                continue
            sid = rec["sample_id"]
            has_camera_words = bool(rec.get("metrics", {}).get("camera_words"))
            s3_rec = process_sample_stage3(
                sid, rec["tar_path"], rec.get("group", ""),
                flow_fn=flow_fn, dover_fn=dover_fn, vlm_call=vlm_call,
                table6_cfg=table6_cfg, has_camera_words=has_camera_words,
            )
            merged = dict(rec)
            merged["stage3"] = s3_rec["stage3"]
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

            # Write caption override sidecar if rewritten
            cap_revised = s3_rec["stage3"].get("caption_revised")
            if cap_revised:
                orig_cap = ""
                try:
                    with tarfile.open(rec["tar_path"], "r") as tf:
                        orig_cap = tf.extractfile(tf.getmember(f"{sid}.caption.txt")).read().decode()
                except Exception:
                    pass
                cap_fout.write(json.dumps({
                    "sample_id": sid,
                    "caption_original": orig_cap.strip(),
                    "caption_revised": cap_revised,
                }, ensure_ascii=False) + "\n")

            total += 1
            if total % 1000 == 0:
                print(f"[stage3] {total} samples processed", flush=True)
    return total


# ── Model loader helpers (called by CMCC launcher, not imported in tests) ─────

def load_unimatch_fn(model_dir: str, device: str = "cuda"):
    """Load UniMatch and return flow_fn(img_a, img_b) -> (H,W,2) float32."""
    import sys
    sys.path.insert(0, str(Path(model_dir).parent))
    from unimatch.unimatch import UniMatch  # type: ignore
    import torch
    model = UniMatch(
        feature_channels=128, num_scales=2, upsample_factor=4,
        num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
        reg_refine=True, task="flow",
    ).to(device).eval()
    ckpt = Path(model_dir) / "gmflow-scale2-regrefine6-mixdata.pth"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    def flow_fn(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
        import torch, torch.nn.functional as F
        def prep(img):
            t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            # pad to 32-multiple
            _, _, H, W = t.shape
            pH = (32 - H % 32) % 32
            pW = (32 - W % 32) % 32
            return F.pad(t, (0, pW, 0, pH)), H, W
        ta, H, W = prep(img_a)
        tb, _, _ = prep(img_b)
        with torch.no_grad():
            result = model(ta, tb, attn_type="swin", attn_splits_list=[2, 8],
                           corr_radius_list=[-1, 4], prop_radius_list=[-1, 1],
                           num_reg_refine=6, task="flow")
        flow = result["flow_preds"][-1][0].permute(1, 2, 0).cpu().numpy()
        return flow[:H, :W]
    return flow_fn


def load_dover_fn(device: str = "cuda"):
    """Load DOVER and return dover_fn(frames_rgb: (T,H,W,3) uint8) -> float."""
    from dover import DOVER  # type: ignore
    import torch
    model = DOVER().to(device).eval()

    def dover_fn(frames_rgb: np.ndarray) -> float:
        import torch
        t = torch.from_numpy(frames_rgb).float().permute(0, 3, 1, 2).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            score = model(t)
        return float(score.mean().item())
    return dover_fn


def load_qwen_fn(model_dir: str, device: str = "cuda"):
    """Load Qwen3.5-27B-VL and return vlm_call(prompt, keyframes) -> str."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # type: ignore
    import torch
    from qwen_vl_utils import process_vision_info  # type: ignore
    from PIL import Image

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)

    def vlm_call(prompt: str, keyframes: list) -> str:
        pil_imgs = [Image.fromarray(f) for f in keyframes]
        content = [{"type": "text", "text": prompt}]
        for img in pil_imgs:
            content.insert(-1, {"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=pil_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return vlm_call
```

- [ ] **Step 4: 运行测试**
```bash
python -m pytest tests/test_qc_stage3.py -v
```
预期：全部 PASSED（mock 注入，无 GPU 要求）

- [ ] **Step 5: Commit**
```bash
git add src/sana_wm_pipeline/qc/stage3_gpu.py tests/test_qc_stage3.py
git commit -m "feat(qc): Stage 3 GPU pipeline (UniMatch/DOVER/Qwen) with caption sidecar"
```

---

## Task 6: `report.py` + `scripts/run_qc.py`

**Files:**
- Create: `src/sana_wm_pipeline/qc/report.py`
- Create: `scripts/run_qc.py`
- Test: `tests/test_qc_report.py`

**Interfaces — Produces:**
- `merge_results(stage1_jsonl, stage2_jsonl=None, stage3_jsonl=None) -> list[dict]`
- `write_manifests(results, output_dir)` → pass.txt / reject.txt / human_review.txt
- `write_html_report(results, output_dir)` → report.html（含 per-group 汇总表）
- `run_report(stage1_jsonl, stage2_jsonl, stage3_jsonl, output_dir)`

**Stage 3 结果影响 manifest 的规则：**
- `table6_accepted == False` → 升级为 fail（从 pass/flag 变为 fail）
- `caption_revised` 存在 → 在 pass/flag 记录中注明 caption 已改写

- [ ] **Step 1: 写失败测试**

```python
# tests/test_qc_report.py
from __future__ import annotations
import json
from pathlib import Path
import pytest
from sana_wm_pipeline.qc.report import merge_results, write_manifests, write_html_report


def _w(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


@pytest.fixture
def base_records():
    base = {"group": "wds-OmniWorld-Game", "tar_path": "/f.tar", "flag_reasons": [],
            "metrics": {"T": 100, "n_jumps": 0, "caption_len": 200}}
    return [
        {**base, "sample_id": "p1", "verdict": "pass"},
        {**base, "sample_id": "p2", "verdict": "pass"},
        {**base, "sample_id": "fl1", "verdict": "flag", "flag_reasons": ["n_jumps=5"]},
        {**base, "sample_id": "fa1", "verdict": "fail", "flag_reasons": ["so3"]},
    ]


def test_merge_stage1_only(tmp_path, base_records):
    s1 = tmp_path / "s1.jsonl"; _w(s1, base_records)
    results = merge_results(s1)
    assert len(results) == 4

def test_merge_stage3_upgrades_to_fail(tmp_path, base_records):
    s1 = tmp_path / "s1.jsonl"; _w(s1, base_records)
    s3 = tmp_path / "s3.jsonl"
    _w(s3, [{**base_records[0], "stage3": {"table6_accepted": False, "reasons": ["dover=0.1"], "unimatch_flow": 5.0, "dover": 0.1, "vlm_entity_count": 2, "vlm_quality": 0.8, "caption_revised": None}}])
    results = merge_results(s1, stage3_jsonl=s3)
    p1 = next(r for r in results if r["sample_id"] == "p1")
    assert p1["verdict"] == "fail"

def test_write_manifests_counts(tmp_path, base_records):
    write_manifests(base_records, tmp_path)
    assert len((tmp_path / "manifests" / "pass.txt").read_text().splitlines()) == 2
    assert len((tmp_path / "manifests" / "reject.txt").read_text().splitlines()) == 1
    assert len((tmp_path / "manifests" / "human_review.txt").read_text().splitlines()) == 1

def test_html_report_created(tmp_path, base_records):
    write_html_report(base_records, tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "<html" in html.lower() and "OmniWorld" in html
```

- [ ] **Step 2: 运行，确认失败**
```bash
python -m pytest tests/test_qc_report.py -v 2>&1 | head -5
```

- [ ] **Step 3: 实现 `report.py`**

```python
# src/sana_wm_pipeline/qc/report.py
"""Merge Stage 1/2/3 results and generate manifests + HTML report."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np


def merge_results(
    stage1_jsonl: Path,
    stage2_jsonl: Optional[Path] = None,
    stage3_jsonl: Optional[Path] = None,
) -> list[dict]:
    records: dict[str, dict] = {}
    ordered: list[str] = []
    with open(stage1_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["sample_id"]
            records[sid] = rec
            ordered.append(sid)

    for jsonl, key in [(stage2_jsonl, "stage2"), (stage3_jsonl, "stage3")]:
        if jsonl and Path(jsonl).exists():
            with open(jsonl) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    sid = rec["sample_id"]
                    if sid in records:
                        records[sid][key] = rec.get(key)

    # Stage 3 verdict upgrade: table6 rejected → fail
    for sid in ordered:
        rec = records[sid]
        s3 = rec.get("stage3") or {}
        if s3.get("table6_accepted") is False and rec.get("verdict") != "fail":
            rec["verdict"] = "fail"
            rec.setdefault("flag_reasons", []).extend(s3.get("reasons", ["table6_rejected"]))

    return [records[sid] for sid in ordered]


def write_manifests(results: list[dict], output_dir: Path) -> None:
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
    for fname, lines in [("pass.txt", pass_lines), ("reject.txt", reject_lines), ("human_review.txt", review_lines)]:
        (mdir / fname).write_text(("\n".join(lines) + "\n") if lines else "")


def _svg_hist(values: list[float], w: int = 280, h: int = 60) -> str:
    if not values:
        return f'<svg width="{w}" height="{h}"></svg>'
    arr = np.array(values, dtype=float)
    counts, _ = np.histogram(arr, bins=min(20, max(2, len(set(values)))))
    mx = max(counts) or 1
    bw = w / len(counts)
    bars = "".join(
        f'<rect x="{i*bw:.1f}" y="{h - max(1, int(c/mx*h))}" '
        f'width="{max(1,bw-1):.1f}" height="{max(1,int(c/mx*h))}" fill="#4a9"/>'
        for i, c in enumerate(counts)
    )
    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'


def write_html_report(results: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gs = defaultdict(lambda: {"pass": 0, "flag": 0, "fail": 0, "n_jumps": [], "dover": [], "flow": []})
    for rec in results:
        g, v = rec.get("group", "?"), rec.get("verdict", "pass")
        gs[g][v] += 1
        m = rec.get("metrics", {})
        if "n_jumps" in m:
            gs[g]["n_jumps"].append(m["n_jumps"])
        s3 = rec.get("stage3") or {}
        if s3.get("dover") is not None:
            gs[g]["dover"].append(s3["dover"])
        if s3.get("unimatch_flow") is not None:
            gs[g]["flow"].append(s3["unimatch_flow"])

    total = len(results)
    n_pass = sum(1 for r in results if r.get("verdict") == "pass")
    n_flag = sum(1 for r in results if r.get("verdict") == "flag")
    n_fail = sum(1 for r in results if r.get("verdict") == "fail")

    rows = []
    for g, s in sorted(gs.items()):
        gt = s["pass"] + s["flag"] + s["fail"]
        pct = 100 * s["pass"] / gt if gt else 0
        rows.append(
            f"<tr><td>{g}</td><td>{gt}</td><td>{s['pass']} ({pct:.1f}%)</td>"
            f"<td>{s['flag']}</td><td>{s['fail']}</td>"
            f"<td>{_svg_hist(s['n_jumps'])}</td>"
            f"<td>{_svg_hist(s['dover'])}</td>"
            f"<td>{_svg_hist(s['flow'])}</td></tr>"
        )

    flag_rows = "".join(
        f"<tr><td>{r['sample_id']}</td><td>{r.get('group','')}</td>"
        f"<td>{'<br>'.join(r.get('flag_reasons', []))}</td></tr>"
        for r in results if r.get("verdict") == "flag"
    )[:200]

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>SANA-WM QC Report</title>
<style>body{{font-family:monospace;margin:20px;background:#1a1a2e;color:#eee}}
h1,h2{{color:#a8daff}}table{{border-collapse:collapse;width:100%;margin-bottom:20px}}
th,td{{border:1px solid #444;padding:6px 10px}}th{{background:#2a2a4a}}
.s{{display:inline-block;margin:10px 20px;font-size:1.3em}}
.p{{color:#4a9}}.fl{{color:#fa3}}.fa{{color:#f44}}</style></head><body>
<h1>SANA-WM Pipeline QC Report</h1>
<div>
  <span class="s">Total <b>{total}</b></span>
  <span class="s p">Pass <b>{n_pass}</b> ({100*n_pass/max(total,1):.1f}%)</span>
  <span class="s fl">Flag <b>{n_flag}</b> ({100*n_flag/max(total,1):.1f}%)</span>
  <span class="s fa">Fail <b>{n_fail}</b> ({100*n_fail/max(total,1):.1f}%)</span>
</div>
<h2>Per-Group Summary</h2>
<table><tr><th>Group</th><th>Total</th><th>Pass</th><th>Flag</th><th>Fail</th>
<th>Jump Dist</th><th>DOVER Dist</th><th>UniMatch Flow Dist</th></tr>
{"".join(rows)}</table>
<h2>Human Review Queue (flag samples)</h2>
<table><tr><th>sample_id</th><th>group</th><th>flag_reasons</th></tr>
{flag_rows}</table></body></html>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def run_report(
    stage1_jsonl: Path,
    stage2_jsonl: Optional[Path],
    stage3_jsonl: Optional[Path],
    output_dir: Path,
) -> None:
    results = merge_results(stage1_jsonl, stage2_jsonl, stage3_jsonl)
    write_manifests(results, output_dir)
    write_html_report(results, output_dir)
```

- [ ] **Step 4: 实现 `scripts/run_qc.py`**

```python
#!/usr/bin/env python3
# scripts/run_qc.py
"""Stage 1+2 CLI entry point for SANA-WM Output QC System."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.qc.stage1_fast import run_stage1
from sana_wm_pipeline.qc.stage2_deep import run_stage2
from sana_wm_pipeline.qc.report import run_report


def main():
    p = argparse.ArgumentParser(description="SANA-WM QC Stage 1+2")
    p.add_argument("--tar-root", type=Path, help="Root dir containing .tar shards (recursive)")
    p.add_argument("--group", default="", help="Dataset group name (e.g. wds-OmniWorld-Game)")
    p.add_argument("--output-dir", type=Path, default=Path("./qc_output"))
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--sample-frac", type=float, default=0.05,
                   help="Fraction of passing samples to deep-check in Stage 2")
    p.add_argument("--read-video-frames", action="store_true",
                   help="Decode video frames in Stage 1 for saturation check")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--report-only", action="store_true",
                   help="Skip Stage 1+2, regenerate report from existing jsonl")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s1_jsonl = out / "stage1_results.jsonl"
    s2_jsonl = out / "stage2_results.jsonl"

    if not args.report_only:
        if not args.tar_root:
            p.error("--tar-root required unless --report-only")
        tars = sorted(args.tar_root.rglob("*.tar"))
        print(f"[stage1] {len(tars)} shards under {args.tar_root}", flush=True)
        n = run_stage1(tars, args.group, s1_jsonl, args.n_workers, args.read_video_frames)
        print(f"[stage1] {n} samples → {s1_jsonl}", flush=True)

        if not args.skip_stage2:
            n2 = run_stage2(s1_jsonl, s2_jsonl, args.sample_frac, args.n_workers)
            print(f"[stage2] {n2} samples deep-checked → {s2_jsonl}", flush=True)

    s2_in = s2_jsonl if (not args.skip_stage2 and s2_jsonl.exists()) else None
    run_report(s1_jsonl, s2_in, None, out)
    print(f"[report] {out}/report.html + {out}/manifests/", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行测试**
```bash
python -m pytest tests/test_qc_report.py -v
```

- [ ] **Step 6: CLI 端到端测试**
```bash
# 建一个临时测试目录
mkdir -p /tmp/qc_test_tars
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -c "
import tarfile; from pathlib import Path
td = Path('testdata')
sid = 'OmniWorld-Game_1f79eb96f021__splits_013-015'
with tarfile.open('/tmp/qc_test_tars/shard-000000.tar', 'w') as tf:
    for ext in ['.mp4','.poses_c2w.npy','.intrinsics.npy','.scale.npy','.caption.txt','.meta.json']:
        tf.add(td/(sid+ext), arcname=sid+ext)
"
python scripts/run_qc.py \
  --tar-root /tmp/qc_test_tars \
  --group wds-OmniWorld-Game \
  --output-dir /tmp/qc_test_out \
  --n-workers 1 --skip-stage2
ls /tmp/qc_test_out/manifests/
```
预期：`pass.txt  reject.txt  human_review.txt` 均存在

- [ ] **Step 7: Commit**
```bash
git add src/sana_wm_pipeline/qc/report.py scripts/run_qc.py tests/test_qc_report.py
git commit -m "feat(qc): report generator + Stage 1+2 CLI"
```

---

## Task 7: `scripts/run_stage3_cmcc.py` — CMCC 48-GPU Stage 3 启动器

**Files:**
- Create: `scripts/run_stage3_cmcc.py`
- （无独立测试；在 CMCC 上手动验证）

**设计：** 单卡单进程串行处理分配到本卡的 shard，与 `experiments/batch_production/launch_all_nodes.sh` 模式一致（主节点 SSH 派发，每个节点/GPU 独立运行）。

**用法：**
```bash
# 每个 GPU 节点上运行（由 launch_all_nodes.sh 类似脚本 SSH 派发）：
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3_cmcc.py \
  --stage1-jsonl ~/work/filestorage/shangaoooooo/davidwang/qc_output/wds-OmniWorld-Game/stage1_results.jsonl \
  --output-dir   ~/work/filestorage/shangaoooooo/davidwang/qc_output/wds-OmniWorld-Game/ \
  --qwen-dir     ~/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL/ \
  --unimatch-dir ~/work/filestorage/shangaoooooo/davidwang/UniMatch/ \
  --worker-id 0 --total-workers 48 \
  --table6-cfg   ~/work/david_work/sana_wm_pipeline/configs/filter_thresholds.yaml
```

- [ ] **Step 1: 实现 `scripts/run_stage3_cmcc.py`**

```python
#!/usr/bin/env python3
# scripts/run_stage3_cmcc.py
"""CMCC per-GPU Stage 3 runner. Called once per GPU by the SSH launcher."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds
from sana_wm_pipeline.qc.stage3_gpu import (
    process_sample_stage3, load_unimatch_fn, load_dover_fn, load_qwen_fn,
)


def main():
    p = argparse.ArgumentParser(description="CMCC Stage 3 GPU worker")
    p.add_argument("--stage1-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--qwen-dir", type=Path, required=True)
    p.add_argument("--unimatch-dir", type=Path, required=True)
    p.add_argument("--worker-id", type=int, required=True, help="0-indexed worker id")
    p.add_argument("--total-workers", type=int, required=True)
    p.add_argument("--table6-cfg", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table6_cfg = load_thresholds(args.table6_cfg)

    # Load models
    print(f"[worker {args.worker_id}] loading UniMatch...", flush=True)
    flow_fn = load_unimatch_fn(str(args.unimatch_dir), args.device)
    print(f"[worker {args.worker_id}] loading DOVER...", flush=True)
    dover_fn = load_dover_fn(args.device)
    print(f"[worker {args.worker_id}] loading Qwen3.5-27B-VL...", flush=True)
    vlm_call = load_qwen_fn(str(args.qwen_dir), args.device)
    print(f"[worker {args.worker_id}] models ready.", flush=True)

    # Assign records to this worker by round-robin index
    out_jsonl = args.output_dir / f"stage3_worker{args.worker_id:03d}.jsonl"
    cap_jsonl = args.output_dir / f"caption_overrides_worker{args.worker_id:03d}.jsonl"
    done_path = args.output_dir / f"stage3_worker{args.worker_id:03d}.done"

    if done_path.exists():
        print(f"[worker {args.worker_id}] already done, skipping.", flush=True)
        return

    with open(args.stage1_jsonl) as fin, \
         open(out_jsonl, "w") as fout, \
         open(cap_jsonl, "w") as cap_fout:
        for idx, line in enumerate(fin):
            if idx % args.total_workers != args.worker_id:
                continue
            rec = json.loads(line)
            if rec.get("verdict") == "fail":
                continue
            sid = rec["sample_id"]
            has_cw = bool(rec.get("metrics", {}).get("camera_words"))
            s3_rec = process_sample_stage3(
                sid, rec["tar_path"], rec.get("group", ""),
                flow_fn=flow_fn, dover_fn=dover_fn, vlm_call=vlm_call,
                table6_cfg=table6_cfg, has_camera_words=has_cw,
            )
            merged = dict(rec); merged["stage3"] = s3_rec["stage3"]
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            cap_rev = s3_rec["stage3"].get("caption_revised")
            if cap_rev:
                import tarfile
                orig = ""
                try:
                    with tarfile.open(rec["tar_path"], "r") as tf:
                        orig = tf.extractfile(tf.getmember(f"{sid}.caption.txt")).read().decode()
                except Exception:
                    pass
                cap_fout.write(json.dumps({
                    "sample_id": sid,
                    "caption_original": orig.strip(),
                    "caption_revised": cap_rev,
                }, ensure_ascii=False) + "\n")
            if (idx // args.total_workers + 1) % 100 == 0:
                print(f"[worker {args.worker_id}] {idx//args.total_workers+1} samples done", flush=True)

    done_path.write_text("done")
    print(f"[worker {args.worker_id}] finished → {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写合并脚本（Stage 3 输出合并，在 Stage 3 全部 worker 完成后运行）**

```bash
# 在 CMCC master 上执行：合并所有 worker 的 stage3_worker*.jsonl
cd ~/work/filestorage/shangaoooooo/davidwang/qc_output/wds-OmniWorld-Game/
cat stage3_worker*.jsonl > stage3_results.jsonl
cat caption_overrides_worker*.jsonl > caption_overrides.jsonl
echo "merged $(wc -l < stage3_results.jsonl) stage3 records"

# 然后跑报告
cd ~/work/david_work/sana_wm_pipeline
python scripts/run_qc.py --report-only \
  --output-dir ~/work/filestorage/shangaoooooo/davidwang/qc_output/wds-OmniWorld-Game/
# 注意：run_qc.py --report-only 默认只用 stage1+stage2，需单独传 stage3
# 或直接调用：
python -c "
from pathlib import Path
from sana_wm_pipeline.qc.report import run_report
run_report(
    stage1_jsonl=Path('~/work/filestorage/.../stage1_results.jsonl').expanduser(),
    stage2_jsonl=Path('~/work/filestorage/.../stage2_results.jsonl').expanduser(),
    stage3_jsonl=Path('~/work/filestorage/.../stage3_results.jsonl').expanduser(),
    output_dir=Path('~/work/filestorage/.../').expanduser(),
)
"
```

- [ ] **Step 3: CMCC 部署验证（单 GPU 冒烟）**

```bash
# 在 CMCC 任意一台节点上，先跑单 GPU 冒烟（只处理前 10 条）
source ~/work/david_work/activate_sana_wm.sh
export VIPE_EXT_JIT=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3_cmcc.py \
  --stage1-jsonl  ~/work/filestorage/.../stage1_results.jsonl \
  --output-dir    /tmp/s3_smoke/ \
  --qwen-dir      ~/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL/ \
  --unimatch-dir  ~/work/filestorage/shangaoooooo/davidwang/UniMatch/ \
  --worker-id 0 --total-workers 48 \
  --table6-cfg    ~/work/david_work/sana_wm_pipeline/configs/filter_thresholds.yaml
# 检查输出
python -c "
import json; lines = open('/tmp/s3_smoke/stage3_worker000.jsonl').readlines()
r = json.loads(lines[0])
print('stage3 keys:', list(r['stage3'].keys()))
print('table6_accepted:', r['stage3']['table6_accepted'])
print('dover:', r['stage3']['dover'])
"
```
预期：`table6_accepted` 有值，`dover` 在 [0.25, 1.0] 范围内

- [ ] **Step 4: Commit**
```bash
git add scripts/run_stage3_cmcc.py
git commit -m "feat(qc): CMCC 48-GPU Stage 3 per-worker launcher"
```

---

## 全套测试验证（提交前）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_qc_metrics.py tests/test_qc_group_config.py \
    tests/test_qc_stage1.py tests/test_qc_stage2.py \
    tests/test_qc_stage3.py tests/test_qc_report.py -v
```
预期：全部绿色

---

## CMCC 执行顺序总结

```
Day 1
  ├─ AFS 上完成 Task 1-6 代码 + 测试（本计划）
  ├─ rsync 到 CMCC：rsync -avz /mnt/afs/davidwang/workspace/sana_wm_pipeline/
  │                          ~/work/filestorage/.../sana_wm_pipeline/
  ├─ pip install av scenedetect dover（在 CMCC conda env）
  └─ 对每个 group 运行 Stage 1+2：
       python scripts/run_qc.py \
         --tar-root ~/work/externalstorage/.../jdvbbfb_output/final_wds-OmniWorld-Game/ \
         --group wds-OmniWorld-Game \
         --output-dir ~/work/filestorage/.../qc_output/wds-OmniWorld-Game/ \
         --n-workers 32 --read-video-frames

Day 2
  └─ 启动 Stage 3（48 GPU 并行，每个 group 依次）：
       # 对每个 GPU 节点 SSH 执行：
       for gpu in $(seq 0 7); do
         CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/run_stage3_cmcc.py \
           --stage1-jsonl .../stage1_results.jsonl \
           --output-dir .../qc_output/wds-OmniWorld-Game/ \
           --worker-id $((NODE_RANK*8+gpu)) --total-workers 48 \
           ... > logs/s3_${NODE_RANK}_${gpu}.log 2>&1 &
       done

Day 3
  └─ 合并 stage3_worker*.jsonl → stage3_results.jsonl
  └─ 生成最终报告：python -c "from sana_wm_pipeline.qc.report import run_report; ..."
  └─ 人工审核 human_review.txt（5人×4天）

Day 7
  └─ 合并最终 manifest → pass_final.txt → 移交训练团队
```

---

## 自检（Spec Coverage）

| QC_REVIEW_DESIGN.md 要求 | 计划覆盖 |
|---|---|
| Stage 1 Check 1-8（结构/SO3/跳变/FoV 等）| ✅ Task 1 metrics.py |
| Stage 1 Check 9：颜色饱和度 | ✅ Task 1 check_color_saturation |
| Stage 1 Check 10：caption 摄像机词 | ✅ Task 1 check_caption_camera_words |
| 7 个 group 差异化阈值（含 max_jumps_fail=50）| ✅ Task 2 group_config.py |
| Stage 2 视频帧数核验（PyAV）| ✅ Task 4 count_video_frames_av |
| Stage 2 场景切割（PySceneDetect）| ✅ Task 4 count_scene_cuts_from_bytes |
| Stage 2 黑帧比例（>30% 拒绝）| ✅ Task 4 check_black_frame_ratio |
| Stage 2 轨迹冻结（>50% 拒绝）| ✅ Task 4 check_trajectory_frozen |
| Stage 3 UniMatch 光流 | ✅ Task 5 stage3_gpu.py |
| Stage 3 DOVER 质量评分 | ✅ Task 5 stage3_gpu.py |
| Stage 3 Qwen3.5-27B 实体计数 + 质量 | ✅ Task 5 stage3_gpu.py |
| Stage 3 Caption 强动作词改写 | ✅ Task 5 caption_revised 字段 |
| Caption 改写 sidecar（不改原始 tar）| ✅ caption_overrides.jsonl |
| Table 6 完整评估 + 报告 | ✅ Task 5 apply_table6.evaluate() |
| HTML 报告 + 三份 manifest | ✅ Task 6 report.py |
| CMCC 48-GPU 启动器 | ✅ Task 7 run_stage3_cmcc.py |

# Default vs GT-pose 模式对比实验计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一批 DL3DV smoke test 场景上跑 Default 模式（VIPE+Pi3X+MoGe-2 SLAM）和 GT-pose 模式，输出 ATE RMSE / scale / 轨迹图对比报告，量化 Default 模式在有 GT 可用场景上的位姿漂移。

**Architecture:** GT-pose shards 已完成（`dl3dv_smoke_shards_gtpose/`，4 个场景，ATE≈0）。本计划补跑 Default 模式：先 fix Pi3X chunk=8（MIG 内存约束），再按场景顺序跑 VIPE SLAM，pack shards，最后用统一脚本对比两组 ATE / scale，输出 Markdown 报告。

**Tech Stack:** Python 3.10, conda env `sana_wm`, VIPE CLI (`vipe infer`), Pi3X, MoGe-2, evo (ATE 评估), matplotlib (轨迹图)

---

## 背景与约束

- **工作目录**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/`
- **Conda env**：`sana_wm`（`/mnt/afs/davidwang/miniconda3/envs/sana_wm/`）
- **模型权重**：
  - Pi3X: `/mnt/afs/davidwang/models/pi3x`
  - MoGe-2: `/mnt/afs/davidwang/models/moge2`
- **DL3DV smoke 场景**（4 个）：`/mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K/<scene_id>/`
- **GT-pose shards**（已完成）：`/mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose/`
- **Default shards 目标**：`/mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/`
- **Pi3X chunk 限制**：MIG 42.4GB，chunk 必须 ≤ 8（16 会 OOM）
- **VIPE pipeline config**：`vipe_cached_depth`（位于 `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml`）

## 文件变更清单

| 操作 | 文件 |
|---|---|
| Modify | `src/sana_wm_pipeline/stage02_pose/mode_default.py` — `_precompute_depth_cache()` 支持 env var `SANA_WM_PI3X_CHUNK` |
| Create | `experiments/data_production_smoke/compare_modes.py` — 读两组 eval_summary，生成对比报告 |
| Modify | `experiments/data_production_smoke/run_e2e_default.sh` — 加 chunk=8 env var |

---

## Task 1: 修复 Pi3X chunk size（支持 env var 覆盖）

**Files:**
- Modify: `src/sana_wm_pipeline/stage02_pose/mode_default.py`

- [ ] **Step 1.1: 修改 `_precompute_depth_cache` signature，读取 env var**

在 `mode_default.py` 的 `_precompute_depth_cache()` 函数中，将 `chunk: int = 16` 改为从环境变量 `SANA_WM_PI3X_CHUNK` 读取（默认 8）：

```python
# mode_default.py — 修改 _precompute_depth_cache 函数签名和 chunk 读取逻辑
def _precompute_depth_cache(
    clip_path: Path,
    cache_path: Path,
    pi3x_weights: str,
    moge2_weights: str,
    chunk: int | None = None,  # None → 读 env var
    stride: int = 8,
    device: str = "cuda",
) -> None:
    import os as _os
    if chunk is None:
        chunk = int(_os.environ.get("SANA_WM_PI3X_CHUNK", "8"))
```

具体做法——将 `mode_default.py` 中 `_precompute_depth_cache` 的：
```python
def _precompute_depth_cache(
    clip_path: Path,
    cache_path: Path,
    pi3x_weights: str,
    moge2_weights: str,
    chunk: int = 16,
    stride: int = 8,
    device: str = "cuda",
) -> None:
```
改为：
```python
def _precompute_depth_cache(
    clip_path: Path,
    cache_path: Path,
    pi3x_weights: str,
    moge2_weights: str,
    chunk: int | None = None,
    stride: int = 8,
    device: str = "cuda",
) -> None:
    import os as _os
    if chunk is None:
        chunk = int(_os.environ.get("SANA_WM_PI3X_CHUNK", "8"))
```

- [ ] **Step 1.2: 在 `run_e2e_default.sh` 中添加 chunk=8 env var**

在 `run_e2e_default.sh` 的环境变量设置区（`SANA_WM_MOGE2_WEIGHTS=...` 那行之后）添加：
```bash
export SANA_WM_PI3X_CHUNK=8
```

- [ ] **Step 1.3: 验证修改正确**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -c "
from src.sana_wm_pipeline.stage02_pose.mode_default import _precompute_depth_cache
import inspect
sig = inspect.signature(_precompute_depth_cache)
print('chunk default:', sig.parameters['chunk'].default)
assert sig.parameters['chunk'].default is None, 'chunk default should be None'
print('OK')
"
```
期望输出：`chunk default: None` + `OK`

- [ ] **Step 1.4: 运行 fast unit test（不需要 GPU）**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -m pytest tests/test_pose_modes.py -v -x 2>&1 | head -40
```
期望：所有 pose mode 相关测试通过（无需 GPU，mock 模式）。

- [ ] **Step 1.5: Commit**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add src/sana_wm_pipeline/stage02_pose/mode_default.py \
        experiments/data_production_smoke/run_e2e_default.sh
git commit -m "fix: mode_default Pi3X chunk from env var SANA_WM_PI3X_CHUNK (default 8 for MIG)"
```

---

## Task 2: 运行 Default 模式 E2E（4 个 DL3DV 场景）

**Files:**
- Run: `experiments/data_production_smoke/run_e2e_default.sh`
- Output: `data/dl3dv_smoke_shards_default/*.tar`

这是计算密集型 Task（每个场景约 15-30 min：Pi3X+MoGe-2 深度缓存 + VIPE SLAM）。

- [ ] **Step 2.1: 确认 normalized.mp4 已存在（可复用 gtpose 的 work 目录）**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
ls /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/work/*/normalized.mp4
```
期望：4 个场景都有 normalized.mp4（已在 gtpose 运行时生成）。

注：default 模式的 work 目录是 `dl3dv_smoke_shards_default/work/`，与 gtpose 各自独立。若 normalized.mp4 不存在，run_e2e_default.sh 会自动创建。

- [ ] **Step 2.2: 确认 VIPE 环境就绪**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
which vipe && vipe --help 2>&1 | head -5
```
期望：输出 VIPE CLI 版本信息，无报错。

- [ ] **Step 2.3: 运行 Default 模式（后台执行，可能需要 1-2h）**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
bash experiments/data_production_smoke/run_e2e_default.sh \
  /mnt/afs/davidwang/workspace/data/dl3dv_smoke \
  2>&1 | tee /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/run_default.log
```

如需后台执行：
```bash
nohup bash experiments/data_production_smoke/run_e2e_default.sh \
  /mnt/afs/davidwang/workspace/data/dl3dv_smoke \
  > /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/run_default.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2.4: 验证 shards 生成**

```bash
ls /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/*.tar
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default
```
期望：4 个 .tar 文件，schema check 全 PASS（每个 shard 包含 6 个必需文件）。

---

## Task 3: Pose 评估 Default 模式结果

**Files:**
- Run: `experiments/data_production_smoke/verify_and_eval.py`
- Output: `data/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json`

- [ ] **Step 3.1: 运行 pose-eval**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
  --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke/1K \
  --out-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/eval_output
```

期望输出：
```
[SHARD] shard-000001.tar
  [SAMPLE] 0032cd2f...
    ATE RMSE=0.02xxxx  T_est=239  T_gt_orig=358  orig_fps=24.00
  [PLOT] Saved ...
...
ATE RMSE — mean: 0.0xxx  median: 0.0xxx  max: 0.xxx
Summary saved to ...pose_eval_summary.json
```

注：--scenes-dir 需要指向包含 `<scene_id>/gt_poses.npy` 的父目录。DL3DV smoke 的结构是 `data/dl3dv_smoke/1K/<scene_id>/gt_poses.npy`，所以 scenes-dir 传 `data/dl3dv_smoke/1K`。

- [ ] **Step 3.2: 确认结果文件存在**

```bash
cat /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json
```
期望：JSON 数组，4 个样本，每个有 `ate_rmse` 字段且为非零正数（相比 gtpose 的 ~1e-7 应显著更大）。

---

## Task 4: 创建对比报告脚本并生成报告

**Files:**
- Create: `experiments/data_production_smoke/compare_modes.py`

- [ ] **Step 4.1: 创建 compare_modes.py**

创建 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/data_production_smoke/compare_modes.py`：

```python
#!/usr/bin/env python3
"""Compare GT-pose vs Default mode pose accuracy on DL3DV smoke scenes.

Usage:
  python compare_modes.py \
    --gtpose-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose \
    --default-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
    --output /mnt/afs/davidwang/workspace/docs/operation_logs/2026-06-12-mode-comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_summary(eval_dir: Path) -> dict[str, dict]:
    """Load pose_eval_summary.json → {sample_id: result_dict}."""
    summary_path = eval_dir / "eval_output" / "pose_eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"pose_eval_summary.json not found: {summary_path}")
    data = json.loads(summary_path.read_text())
    return {r["sample_id"]: r for r in data if r is not None}


def short_id(sample_id: str, n: int = 12) -> str:
    return sample_id[:n] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT-pose vs Default mode.")
    parser.add_argument("--gtpose-dir", type=Path, required=True)
    parser.add_argument("--default-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gtpose_results = load_summary(args.gtpose_dir)
    default_results = load_summary(args.default_dir)

    # Align by sample_id
    common_ids = sorted(set(gtpose_results) & set(default_results))
    if not common_ids:
        print("[WARN] No common sample IDs between the two eval summaries.")
        return

    rows = []
    for sid in common_ids:
        gp = gtpose_results[sid]
        df = default_results[sid]
        gp_ate = gp.get("ate_rmse")
        df_ate = df.get("ate_rmse")
        ratio = (df_ate / gp_ate) if (gp_ate and df_ate and gp_ate > 1e-10) else None
        rows.append({
            "sample_id": sid,
            "gtpose_ate": gp_ate,
            "default_ate": df_ate,
            "ratio_default_over_gtpose": ratio,
        })

    gtpose_ates = [r["gtpose_ate"] for r in rows if r["gtpose_ate"] is not None]
    default_ates = [r["default_ate"] for r in rows if r["default_ate"] is not None]

    lines = [
        "# GT-pose vs Default 模式 Pose 精度对比（DL3DV Smoke Test）",
        "",
        f"**日期**: 2026-06-12  ",
        f"**场景数**: {len(common_ids)}  ",
        f"**数据集**: DL3DV smoke test  ",
        "",
        "## 逐样本结果",
        "",
        "| 样本 ID | GT-pose ATE RMSE (m) | Default ATE RMSE (m) | 倍率 (Default/GT-pose) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        sid_s = short_id(r["sample_id"])
        gp_s = f"{r['gtpose_ate']:.2e}" if r["gtpose_ate"] is not None else "N/A"
        df_s = f"{r['default_ate']:.4f}" if r["default_ate"] is not None else "N/A"
        ratio_s = f"{r['ratio_default_over_gtpose']:.0f}×" if r["ratio_default_over_gtpose"] else "N/A"
        lines.append(f"| {sid_s} | {gp_s} | {df_s} | {ratio_s} |")

    lines += [
        "",
        "## 汇总统计",
        "",
        f"| 指标 | GT-pose 模式 | Default 模式 |",
        f"|---|---|---|",
        f"| ATE RMSE 均值 (m) | {np.mean(gtpose_ates):.2e} | {np.mean(default_ates):.4f} |" if gtpose_ates and default_ates else "",
        f"| ATE RMSE 中位数 (m) | {np.median(gtpose_ates):.2e} | {np.median(default_ates):.4f} |" if gtpose_ates and default_ates else "",
        f"| ATE RMSE 最大值 (m) | {np.max(gtpose_ates):.2e} | {np.max(default_ates):.4f} |" if gtpose_ates and default_ates else "",
        "",
        "## 结论",
        "",
        "**GT-pose 模式**：直接使用 DL3DV 官方 GT 位姿，Umeyama Sim(3) 仅恢复 metric scale，",
        "ATE RMSE ≈ 机器精度（数值零），位姿精度理论上限。",
        "",
        "**Default 模式**：VIPE SLAM（Pi3X+MoGe-2 深度注入），在 DL3DV 静态高质量场景上",
        "仍会引入 SLAM 漂移，ATE RMSE 远大于 GT-pose 模式。",
        "",
        "**建议**：对有 GT 位姿的数据集（DL3DV、Sekai-Game）始终使用 GT-pose 模式；",
        "Default 模式仅用于无 GT 的互联网视频（SpatialVID-HQ、MiraData 等）。",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {args.output}")

    # Also print to stdout
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: 生成对比报告**

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

python experiments/data_production_smoke/compare_modes.py \
  --gtpose-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose \
  --default-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default \
  --output /mnt/afs/davidwang/workspace/docs/operation_logs/2026-06-12-mode-comparison.md
```

期望：
- 控制台打印 Markdown 表格，显示 4 个场景的 gtpose vs default ATE RMSE
- gtpose ATE ≈ 1e-7；default ATE 预期在 0.01~0.1m 量级（具体取决于 VIPE SLAM 质量）
- 报告文件写入 `docs/operation_logs/`

- [ ] **Step 4.3: 对比轨迹图**

两组 eval_output/ 中都有轨迹对比图（GT vs estimated）。Default 模式的轨迹图应展示 SLAM 偏移。

```bash
ls /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_gtpose/eval_output/*.png
ls /mnt/afs/davidwang/workspace/data/dl3dv_smoke_shards_default/eval_output/*.png
```

- [ ] **Step 4.4: Commit 对比报告和脚本**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add experiments/data_production_smoke/compare_modes.py
git commit -m "feat: add mode comparison script (gtpose vs default, DL3DV)"
```

---

## 期望结果与验证标准

| 检查项 | 期望 |
|---|---|
| Default shards schema | 4 个 .tar，全部 schema PASS |
| Default ATE RMSE | > 0.001m（有 SLAM 漂移，与 gtpose 的 ~1e-7 显著不同）|
| 对比报告 | `docs/operation_logs/2026-06-12-mode-comparison.md` 存在 |
| 结论 | 论文选择 DL3DV 用 GT-pose 模式得到量化验证 |

---

## 注意事项

1. **VIPE SLAM 内存**：每个场景 VIPE 约需 15-20GB GPU 内存（SLAM + 深度预测）。MIG 42.4GB 足够，但不要同时跑多个场景。
2. **Pi3X chunk=8**（已在 Task 1 修复）：`_precompute_depth_cache` 默认 chunk=8，避免 OOM。
3. **VIPE config 路径**：`vipe_cached_depth` pipeline config 已在 VIPE 安装路径内，`vipe infer --pipeline vipe_cached_depth` 可直接使用。
4. **场景目录结构**：verify_and_eval.py 的 `--scenes-dir` 需要指向 `dl3dv_smoke/1K/`（不是 `dl3dv_smoke/`），因为 meta.json 中的 `scene_id` 是最后一级目录名。
5. **SANA_WM_CACHED_DEPTH_PATH**：mode_default.run_default() 会自动设置这个 env var，不需要手动配置。

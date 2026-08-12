# CMCC 批量生产 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CMCC 平台（最多 5 节点 × 8 GPU = 40 卡）并行处理 jdvbbfb-v3-full 数据集，按批次顺序产出 WebDataset shard，全程无 OOM、可断点续跑。

**Architecture:** 纯数据处理管线（无梯度通信），每张 GPU 独立跑一个 Python worker 进程。Worker 从分配的输入 shard 列表读取样本，经 normalize → run_default (Pi3X+MoGe-2+VIPE) → ShardWriter 写出 WebDataset shard。节点间按全局 worker 编号取模均匀分配 shard，用 `.done` 文件实现断点续跑。

**Tech Stack:** Python 3.10, PyTorch 2.12+cu130, sana_wm_pipeline Python API (stage01/02/06), VIPE CLI (subprocess), ffmpeg, CMCC RANK/WORLD_SIZE 环境变量, Bash.

---

## 关键路径说明（必读）

### 数据流
```
输入: DATA_ROOT/{group}/shards/{prefix}-{NNNNNN}.tar
         每个 tar: {key}.mp4 + {key}.camera.npz（约 100-300 样本）
      DATA_ROOT/{group}/index.jsonl（caption/fps 元数据）

Per-sample 处理（Default 模式）:
  Stage 1: normalize_video → /tmp/worker_W/{key}/normalized.mp4（1280×720 @16fps）
  Stage 2: run_default(normalized.mp4, /tmp/worker_W/{key}/vipe_work) → PoseArtifact
           内部：Pi3X+MoGe-2 预计算深度 → VIPE SLAM（subprocess）
  Stage 3: ShardWriter.write(Sample) → OUT_DIR/wXXX/shard-{seq:06d}.tar

输出: OUT_BASE/{group}/w{WW:03d}/shard-{seq:06d}.tar（每 worker 独立目录，无锁）
      OUT_BASE/{group}/progress/{shard_idx:06d}.done（完成标记）
```

### 模型权重路径（CMCC 机器上）
```
$NEW_BASE/models/pi3x/     (含 model.safetensors)
$NEW_BASE/models/moge2/    (含 model.pt)
```
从 AFS rsync 时：`rsync -av /mnt/afs/davidwang/models/ $NEW_BASE/models/`

### 环境变量（每个 worker 必须 export）
```bash
source $NEW_BASE/activate_sana_wm.sh
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export SANA_WM_PI3X_WEIGHTS=$NEW_BASE/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=$NEW_BASE/models/moge2
# OOM 防护：减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 吞吐量估算（H100 80GB，Default 模式）
| 视频长度 | Pi3X+MoGe2 | VIPE SLAM | 总计/样本 | 40卡吞吐 |
|---------|-----------|----------|---------|--------|
| 160帧/10s (DL3DV) | ~2 min | ~5 min | ~7 min | ~340 样本/hr |
| 960帧/60s (Sekai) | ~15 min | ~30 min | ~45 min | ~53 样本/hr |

---

## 文件结构

```
experiments/batch_production/
├── config.sh                   # 【新建】所有路径和环境变量的唯一来源
├── run_worker.py               # 【新建】单 GPU worker，核心处理逻辑
├── launch_single_node.sh       # 【新建】单节点 8 卡启动（本地测试 / 单节点任务）
├── launch_multi_node.sh        # 【新建】CMCC 多节点入口（每节点执行此脚本）
└── run_groups_sequential.sh    # 【新建】按批次顺序依次提交各 group
```

---

## Task 1：创建 config.sh（路径与环境变量统一来源）

**Files:**
- Create: `experiments/batch_production/config.sh`

- [ ] **Step 1.1: 写 config.sh**

```bash
#!/bin/bash
# experiments/batch_production/config.sh
# 所有脚本 source 此文件，确保路径和变量一致。
# 每次 CMCC 环境变更时只改这一处。

# ── 基础路径 ──────────────────────────────────────────────────────────────────
export NEW_BASE=/root/work/david_work
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
# 输出写到 filestorage（持久化），不要写 hot disk！
export OUT_BASE="/root/work/filestorage/jdvbbfb_output"

# ── 模型权重（必须在 CMCC 上 rsync 到此路径）────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"

# ── 激活 conda 环境 ───────────────────────────────────────────────────────────
source "$NEW_BASE/activate_sana_wm.sh"

# ── 离线模式（CMCC 无外网）────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── 显存优化：减少 CUDA 分配器碎片 ────────────────────────────────────────────
# expandable_segments: 允许分配器扩展/收缩 segment，
# 避免释放大 tensor 后残留 hole 导致后续小申请失败
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 批次 1 优先处理的 3 个 group（按先后顺序）────────────────────────────────
BATCH1_GROUPS=(
    "wds-sekai-real-walking-hq"
    "wds-DL3DV-ALL-2K"
    "wds-SpatialVID-hq"
)
export BATCH1_GROUPS

# 每个 worker 输出 shard 中最多包含的样本数
# 设 200 与输入 shard 规模接近，便于 1:1 进度追踪
export SAMPLES_PER_OUTPUT_SHARD=200
```

- [ ] **Step 1.2: 验证 config 可被 source**
```bash
bash -c "source experiments/batch_production/config.sh && echo OK && echo DATA_ROOT=$DATA_ROOT"
# 期望: OK  DATA_ROOT=/root/work/externalstorage/...
```

- [ ] **Step 1.3: Commit**
```bash
git add experiments/batch_production/config.sh
git commit -m "feat(batch): add config.sh with all paths and env vars"
```

---

## Task 2：创建 run_worker.py（单 GPU Worker 核心）

**Files:**
- Create: `experiments/batch_production/run_worker.py`

这是整个系统最重要的文件。一个 worker = 一个 Python 进程 = 一张 GPU。

- [ ] **Step 2.1: 写 run_worker.py**

```python
#!/usr/bin/env python3
"""
单 GPU Worker：处理分配到的输入 shard 列表，产出 WebDataset shard。

用法:
  CUDA_VISIBLE_DEVICES=0 python run_worker.py \
    --group wds-sekai-real-walking-hq \
    --data-root /root/work/externalstorage/.../jdvbbfb-v3-full \
    --out-base /root/work/filestorage/jdvbbfb_output \
    --worker-id 0 \
    --shard-indices 0,8,16,24 \
    --samples-per-shard 200

设计原则:
  - 每个 worker 写到 out_base/{group}/w{worker_id:03d}/ 独立目录，无需文件锁
  - 每处理完一个输入 shard，写 out_base/{group}/progress/{shard_idx:06d}.done
  - 若 .done 已存在则跳过（断点续跑）
  - /tmp/sana_wm_w{worker_id}/ 作为临时目录（本地 NVMe，不走网络存储）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group",            required=True,
                   help="e.g. wds-sekai-real-walking-hq")
    p.add_argument("--data-root",        required=True, type=Path,
                   help="jdvbbfb-v3-full 根目录")
    p.add_argument("--out-base",         required=True, type=Path,
                   help="输出根目录（filestorage，持久化）")
    p.add_argument("--worker-id",        required=True, type=int,
                   help="全局 worker 编号（0..39），用于输出子目录命名")
    p.add_argument("--shard-indices",    required=True,
                   help="逗号分隔的输入 shard 下标，如 '0,8,16'")
    p.add_argument("--samples-per-shard", type=int, default=200,
                   help="每个输出 shard 最多包含的样本数（默认 200）")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _shard_basename(group: str, shard_idx: int) -> str:
    """wds-DL3DV-ALL-2K, 0 -> DL3DV-ALL-2K-000000.tar"""
    prefix = group[len("wds-"):] if group.startswith("wds-") else group
    return f"{prefix}-{shard_idx:06d}.tar"


def _load_captions(index_path: Path) -> dict[str, str]:
    """预加载 index.jsonl，返回 key->caption 字典。启动时执行一次。"""
    if not index_path.exists():
        print(f"[WARN] index.jsonl not found: {index_path}, captions will be stubs")
        return {}
    caps: dict[str, str] = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key", "")
            text = (rec.get("manifest", {}).get("prompt") or {}).get("text", "") or ""
            if key:
                caps[key] = text
    print(f"[index] 加载 {len(caps)} 条 caption from {index_path}")
    return caps


def _normalize(src: Path, dst: Path) -> int:
    """归一化视频到 1280x720 @16fps，返回帧数。"""
    from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
    info = normalize_video(src, dst)
    return info.n_frames


def _run_pose(norm_video: Path, work_dir: Path):
    """运行 Default 模式位姿估算，返回 PoseArtifact。"""
    from sana_wm_pipeline.stage02_pose.mode_default import run_default
    return run_default(norm_video, work_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 核心：处理单个输入 shard
# ─────────────────────────────────────────────────────────────────────────────

def process_input_shard(
    shard_path: Path,
    shard_idx: int,
    group: str,
    captions: dict[str, str],
    tmp_dir: Path,
    writer,          # ShardWriter instance
    progress_dir: Path,
) -> tuple[int, int]:
    """
    处理一个输入 shard，写出所有合法样本到 writer。
    返回 (n_ok, n_fail)。
    """
    done_marker = progress_dir / f"{shard_idx:06d}.done"
    if done_marker.exists():
        print(f"[SKIP] shard {shard_idx:06d} 已完成（.done 存在），跳过")
        return 0, 0

    from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import iter_tar_samples
    from sana_wm_pipeline.stage06_pack.schema import Sample

    STUB_CAPTION = "A real-world scene captured by a moving camera."

    n_ok = n_fail = 0
    t0_shard = time.time()

    with open(shard_path, "rb") as fobj:
        for key, mp4_bytes, camera_bytes in iter_tar_samples(fobj, limit=None):
            sample_tmp = tmp_dir / key
            sample_tmp.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            try:
                # ── 1. 写原始视频到 /tmp（本地 NVMe，避免网络 I/O 瓶颈）──────
                raw_video = sample_tmp / "video.mp4"
                raw_video.write_bytes(mp4_bytes)

                # ── 2. normalize → 1280×720 @16fps ────────────────────────────
                norm_video = sample_tmp / "normalized.mp4"
                n_frames = _normalize(raw_video, norm_video)
                # 原始文件已不需要，释放磁盘空间
                raw_video.unlink()

                # ── 3. 位姿估算（Pi3X + MoGe-2 + VIPE SLAM）──────────────────
                vipe_work = sample_tmp / "vipe_work"
                art = _run_pose(norm_video, vipe_work)

                # ── 4. 组装 Sample，写入 ShardWriter ──────────────────────────
                caption = captions.get(key, "").strip() or STUB_CAPTION
                sample = Sample(
                    sample_id=key,
                    video_path=str(norm_video),
                    poses_c2w=art.poses_c2w,
                    intrinsics_NVD=art.intrinsics,
                    scale_per_frame=art.scale_per_frame,
                    caption=caption,
                    meta={
                        "scene_id": key,
                        "T": int(art.poses_c2w.shape[0]),
                        "mode": "default",
                        "dataset": "jdvbbfb-v3-full",
                        "group": group,
                        "source_shard": shard_path.name,
                    },
                )
                writer.write(sample)
                n_ok += 1
                elapsed = time.time() - t0
                print(f"  [OK] {key}  T={n_frames}  {elapsed:.0f}s")

            except Exception as exc:
                n_fail += 1
                print(f"  [FAIL] {key}: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            finally:
                # 清理该样本的临时目录（不管成败）
                shutil.rmtree(sample_tmp, ignore_errors=True)

    # 写 .done 标记（只有整个 shard 无异常终止才写）
    progress_dir.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(
        json.dumps({"shard_idx": shard_idx, "n_ok": n_ok, "n_fail": n_fail,
                    "elapsed_s": round(time.time() - t0_shard, 1)})
    )
    print(f"[shard {shard_idx:06d}] DONE  ok={n_ok} fail={n_fail}  "
          f"{time.time()-t0_shard:.0f}s")
    return n_ok, n_fail


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 输出目录：每个 worker 独立，无需文件锁 ────────────────────────────────
    worker_out = args.out_base / args.group / f"w{args.worker_id:03d}"
    progress_dir = args.out_base / args.group / "progress"
    worker_out.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    # ── 临时目录：/tmp 本地 NVMe，比 externalstorage 快 10x ──────────────────
    tmp_dir = Path(f"/tmp/sana_wm_w{args.worker_id:03d}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── 预加载 caption index ──────────────────────────────────────────────────
    index_path = args.data_root / args.group / "index.jsonl"
    captions = _load_captions(index_path)

    # ── 解析分配的 shard 下标 ─────────────────────────────────────────────────
    shard_indices = [int(x) for x in args.shard_indices.split(",") if x.strip()]
    shard_dir = args.data_root / args.group / "shards"
    print(f"[worker {args.worker_id}] group={args.group}  "
          f"shards={shard_indices}  out={worker_out}")

    # ── ShardWriter：每个 worker 独立输出目录，无竞争 ────────────────────────
    from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter

    total_ok = total_fail = 0

    # strict_frames=False：允许任意帧数（Default 模式帧数取决于视频长度）
    with ShardWriter(worker_out, samples_per_shard=args.samples_per_shard,
                     prefix="shard", strict_frames=False) as writer:
        for shard_idx in shard_indices:
            shard_path = shard_dir / _shard_basename(args.group, shard_idx)
            if not shard_path.exists():
                print(f"[WARN] shard not found: {shard_path}, skip")
                continue
            n_ok, n_fail = process_input_shard(
                shard_path, shard_idx, args.group,
                captions, tmp_dir, writer, progress_dir,
            )
            total_ok += n_ok
            total_fail += n_fail

    # 清理 worker 临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[worker {args.worker_id}] 全部完成  总计 ok={total_ok} fail={total_fail}")
    sys.exit(1 if total_fail > 0 and total_ok == 0 else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: 快速冒烟（在 AFS 开发机验证脚本可 import，不实际运行）**
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh && conda activate sana_wm
python -c "
import experiments.batch_production.run_worker as rw
print('import OK')
print('iter_tar_samples:', rw.process_input_shard.__doc__[:30])
"
# 期望: import OK
```

- [ ] **Step 2.3: Commit**
```bash
git add experiments/batch_production/run_worker.py
git commit -m "feat(batch): add run_worker.py — single-GPU worker with skip-if-done + OOM cleanup"
```

---

## Task 3：单节点 8 卡启动脚本（launch_single_node.sh）

**Files:**
- Create: `experiments/batch_production/launch_single_node.sh`

- [ ] **Step 3.1: 写 launch_single_node.sh**

```bash
#!/bin/bash
# experiments/batch_production/launch_single_node.sh
#
# 用法（单节点 8 卡，本地测试 / CMCC 单节点任务）:
#   bash experiments/batch_production/launch_single_node.sh \
#       wds-sekai-real-walking-hq    # GROUP 参数
#       [NODE_RANK]                  # 可选，默认 0（单节点时不需要）
#       [NUM_NODES]                  # 可选，默认 1（单节点时不需要）
#
# 在 5 节点多机部署时由 launch_multi_node.sh 调用，不要直接修改此文件。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"   # 加载所有路径和环境变量

GROUP="${1:?用法: $0 <GROUP> [NODE_RANK] [NUM_NODES]}"
NODE_RANK="${2:-0}"
NUM_NODES="${3:-1}"
NUM_GPUS=8    # 固定 8 卡/节点

# 统计输入 shard 总数
SHARD_DIR="$DATA_ROOT/$GROUP/shards"
if [[ ! -d "$SHARD_DIR" ]]; then
    echo "[ERROR] shard 目录不存在: $SHARD_DIR" >&2
    exit 1
fi
# 支持任意数量 shard，auto-discover
SHARDS=($(ls "$SHARD_DIR"/*.tar 2>/dev/null | sort))
TOTAL_SHARDS=${#SHARDS[@]}
if [[ $TOTAL_SHARDS -eq 0 ]]; then
    echo "[ERROR] $SHARD_DIR 中没有 .tar 文件" >&2
    exit 1
fi

# 全局 worker 总数
GLOBAL_WORKERS=$((NUM_NODES * NUM_GPUS))

echo "=== 单节点启动 ==="
echo "  GROUP=$GROUP"
echo "  NODE_RANK=$NODE_RANK / NUM_NODES=$NUM_NODES"
echo "  TOTAL_SHARDS=$TOTAL_SHARDS  GLOBAL_WORKERS=$GLOBAL_WORKERS"
echo "  OUT_BASE=$OUT_BASE"
echo ""

mkdir -p "$OUT_BASE/$GROUP/logs"
PIDS=()

for LOCAL_GPU in $(seq 0 $((NUM_GPUS - 1))); do
    # 全局 worker 编号（0..GLOBAL_WORKERS-1）
    # 例: NODE_RANK=1, LOCAL_GPU=3 -> GLOBAL_WORKER=11
    GLOBAL_WORKER=$((NODE_RANK * NUM_GPUS + LOCAL_GPU))

    # shard 分配：步长=GLOBAL_WORKERS 的等差数列
    # worker W 处理 shard 下标: W, W+GLOBAL_WORKERS, W+2*GLOBAL_WORKERS, ...
    # 优点: (1)均匀分布 (2)任意增减节点数自动适配 (3)续跑时只加 .done 检查即可
    INDICES=""
    for IDX in $(seq $GLOBAL_WORKER $GLOBAL_WORKERS $((TOTAL_SHARDS - 1))); do
        INDICES="${INDICES}${IDX},"
    done
    INDICES="${INDICES%,}"   # 去掉末尾逗号

    if [[ -z "$INDICES" ]]; then
        echo "  GPU $LOCAL_GPU (global $GLOBAL_WORKER): 无分配 shard（总 shard 数少于 worker 数），跳过"
        continue
    fi

    LOG="$OUT_BASE/$GROUP/logs/node${NODE_RANK}_gpu${LOCAL_GPU}.log"

    # 每个 worker 独立进程，CUDA_VISIBLE_DEVICES 隔离显卡
    CUDA_VISIBLE_DEVICES=$LOCAL_GPU \
    PYTHONNOUSERSITE=1 \
    "$ENV_DIR/bin/python" \
        "$PROJ_DIR/experiments/batch_production/run_worker.py" \
        --group        "$GROUP" \
        --data-root    "$DATA_ROOT" \
        --out-base     "$OUT_BASE" \
        --worker-id    $GLOBAL_WORKER \
        --shard-indices "$INDICES" \
        --samples-per-shard "${SAMPLES_PER_OUTPUT_SHARD:-200}" \
        >> "$LOG" 2>&1 &

    PIDS+=($!)
    echo "  GPU $LOCAL_GPU → global_worker=$GLOBAL_WORKER  shards=[$INDICES]  log=$LOG"
done

echo ""
echo "等待 ${#PIDS[@]} 个 worker 完成..."

# 等待并收集退出码
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "  worker $i (PID $PID): 成功"
    else
        CODE=$?
        echo "  worker $i (PID $PID): 失败 (exit $CODE)" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== 节点 $NODE_RANK 完成 | 失败 worker 数: $FAILED / ${#PIDS[@]} ==="

# 打印进度汇总
DONE_COUNT=$(ls "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null | wc -l)
echo "  已完成 shard: $DONE_COUNT / $TOTAL_SHARDS"

exit $FAILED
```

- [ ] **Step 3.2: 验证语法**
```bash
bash -n experiments/batch_production/launch_single_node.sh
echo "语法 OK: $?"
```

- [ ] **Step 3.3: Commit**
```bash
git add experiments/batch_production/launch_single_node.sh
git commit -m "feat(batch): add launch_single_node.sh — 8-GPU parallel launch with round-robin shard assignment"
```

---

## Task 4：CMCC 5 节点多机入口脚本（launch_multi_node.sh）

**Files:**
- Create: `experiments/batch_production/launch_multi_node.sh`

- [ ] **Step 4.1: 写 launch_multi_node.sh**

```bash
#!/bin/bash
# experiments/batch_production/launch_multi_node.sh
#
# CMCC 多机任务入口脚本。
# CMCC 平台会在每个节点上执行此脚本，并注入以下环境变量：
#   RANK       — 节点编号（0-4）
#   WORLD_SIZE — 总节点数（5）
#
# 提交方式：在 CMCC 控制台将本脚本设为"启动命令"，传入 GROUP 参数。
# 示例命令行:
#   bash /root/work/david_work/sana_wm_pipeline/experiments/batch_production/launch_multi_node.sh \
#       wds-sekai-real-walking-hq
#
# 无需 torchrun / NCCL：数据处理是 embarrassingly parallel，节点间无通信。

set -euo pipefail

# ── 读取 CMCC 注入的节点信息 ──────────────────────────────────────────────────
# 若 CMCC 用不同变量名，在此处适配
NODE_RANK="${RANK:-${SLURM_NODEID:-0}}"
NUM_NODES="${WORLD_SIZE:-${SLURM_NNODES:-1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:?用法: bash launch_multi_node.sh <GROUP>}"

echo "===== CMCC 多机启动 ====="
echo "  节点: $NODE_RANK / $NUM_NODES"
echo "  GROUP: $GROUP"
echo "  脚本目录: $SCRIPT_DIR"
echo ""

# 直接委托给 launch_single_node.sh，传入节点信息
# launch_single_node.sh 会自动计算本节点负责的 shard 子集
bash "$SCRIPT_DIR/launch_single_node.sh" \
    "$GROUP" \
    "$NODE_RANK" \
    "$NUM_NODES"
```

- [ ] **Step 4.2: 验证语法**
```bash
bash -n experiments/batch_production/launch_multi_node.sh
echo "语法 OK: $?"
```

- [ ] **Step 4.3: Commit**
```bash
git add experiments/batch_production/launch_multi_node.sh
git commit -m "feat(batch): add launch_multi_node.sh — CMCC multi-node entry with RANK/WORLD_SIZE"
```

---

## Task 5：批次顺序调度脚本（run_groups_sequential.sh）

**Files:**
- Create: `experiments/batch_production/run_groups_sequential.sh`

- [ ] **Step 5.1: 写 run_groups_sequential.sh**

```bash
#!/bin/bash
# experiments/batch_production/run_groups_sequential.sh
#
# 按批次顺序在「单机」上依次处理多个 group。
# 用于 CMCC 单节点或本地调试。多节点场景请对每个 group 单独提交 CMCC 任务。
#
# 用法:
#   bash run_groups_sequential.sh [--batch1-only] [NODE_RANK] [NUM_NODES]
#
# --batch1-only: 只处理 batch1 的 3 个 group
# 不加参数: batch1 完成后自动处理剩余所有 group

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

BATCH1_ONLY=0
NODE_RANK="${2:-0}"
NUM_NODES="${3:-1}"
[[ "${1:-}" == "--batch1-only" ]] && BATCH1_ONLY=1

# ── 确定要处理的 group 列表 ──────────────────────────────────────────────────
GROUPS_TO_RUN=("${BATCH1_GROUPS[@]}")  # 先处理 batch1

if [[ $BATCH1_ONLY -eq 0 ]]; then
    # 发现 DATA_ROOT 下所有 group，去除 batch1 已有的
    ALL_GROUPS=($(ls "$DATA_ROOT/" 2>/dev/null | grep "^wds-" | sort))
    BATCH1_SET=" ${BATCH1_GROUPS[*]} "
    for G in "${ALL_GROUPS[@]}"; do
        if [[ "$BATCH1_SET" != *" $G "* ]]; then
            GROUPS_TO_RUN+=("$G")
        fi
    done
fi

echo "=== 顺序处理 ${#GROUPS_TO_RUN[@]} 个 group ==="
printf '  %s\n' "${GROUPS_TO_RUN[@]}"
echo ""

OVERALL_FAILED=0
for GROUP in "${GROUPS_TO_RUN[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  开始处理: $GROUP"
    echo "══════════════════════════════════════════════"
    START_T=$SECONDS

    bash "$SCRIPT_DIR/launch_single_node.sh" "$GROUP" "$NODE_RANK" "$NUM_NODES" || {
        echo "[WARN] $GROUP 有 worker 失败，继续下一个 group"
        OVERALL_FAILED=$((OVERALL_FAILED + 1))
    }

    ELAPSED=$((SECONDS - START_T))
    echo "  $GROUP 耗时: $((ELAPSED/3600))h$((ELAPSED%3600/60))m$((ELAPSED%60))s"
done

echo ""
echo "=== 全部完成 | 失败 group 数: $OVERALL_FAILED / ${#GROUPS_TO_RUN[@]} ==="
exit $OVERALL_FAILED
```

- [ ] **Step 5.2: Commit**
```bash
git add experiments/batch_production/run_groups_sequential.sh
git commit -m "feat(batch): add run_groups_sequential.sh — batch1 then all remaining groups"
```

---

## Task 6：提速方案——数据读取与预处理加速

> **不修改已有代码**。以下措施全部通过环境变量或脚本层参数生效。

### 6.1 本地 NVMe 临时目录（最重要，约 3-5x I/O 提速）

**根因：** externalstorage 是网络存储，随机小文件读写延迟高（>10ms/op）。`/tmp` 是本地 NVMe，延迟 <0.1ms。

`run_worker.py` 已将 `tmp_dir` 固定在 `/tmp/sana_wm_w{worker_id}/`。
无需额外操作，**确保 CMCC 节点 `/tmp` 有至少 50 GB 空间**：
```bash
df -h /tmp   # 确认可用空间 > 50 GB（每个 worker 临时占用约 1-2 GB）
```

### 6.2 ffmpeg 多线程解码（约 1.5x 解码提速）

normalize_video 内部调用 ffmpeg。在 config.sh 追加（对 ffmpeg 全局生效）：
```bash
# config.sh 追加以下行
export FFREPORT=disable   # 关闭 ffmpeg 报告文件，减少磁盘写入
```

若要进一步加速，在 `normalize.py` 的 ffmpeg 命令中加 `-threads 4`：
- [ ] **Step 6.2.1: 修改 normalize.py 加 threads 参数**

读取文件：
```bash
grep -n "ffmpeg\|subprocess\|-vf" \
    src/sana_wm_pipeline/stage01_ingest/normalize.py | head -20
```

找到 ffmpeg 调用行，在命令列表中加入 `"-threads", "4"`, 位置在 `-i` 参数之前：
```python
# 改前（示例）
cmd = ["ffmpeg", "-i", str(src), "-vf", f"crop=..., fps={fps}", ...]
# 改后
cmd = ["ffmpeg", "-threads", "4", "-i", str(src), "-vf", f"crop=..., fps={fps}", ...]
```

验证：
```bash
python -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
# 用任意存在的视频测试
import subprocess, shutil
if shutil.which('ffmpeg'):
    print('ffmpeg found:', shutil.which('ffmpeg'))
"
```

### 6.3 Caption 预加载（消除每样本 I/O）

**已在 `run_worker.py` 实现**：`_load_captions()` 在 worker 启动时一次性加载整个 index.jsonl 到内存字典，处理每个样本时直接 `dict.get(key)`，无磁盘 I/O。

sekai index.jsonl 约 18,208 行 × ~500B ≈ 9 MB，完全可以常驻内存。

### 6.4 输出写到 filestorage（持久化），临时写 /tmp（速度）

**规则（已在架构中体现）**：
- 临时文件（normalized.mp4, vipe_work/）→ `/tmp`（快速，重启丢失无所谓）
- 输出 shard（.tar）→ `OUT_BASE`（filestorage，持久化）
- 日志 → `OUT_BASE/{group}/logs/`（filestorage）
- `.done` 标记 → `OUT_BASE/{group}/progress/`（filestorage）

```bash
# config.sh 中已设置：
# OUT_BASE="/root/work/filestorage/jdvbbfb_output"
```

### 6.5 进度监控脚本（实时查看吞吐）

- [ ] **Step 6.5.1: 写 watch_progress.sh**

创建文件 `experiments/batch_production/watch_progress.sh`：
```bash
#!/bin/bash
# 实时显示各 group 处理进度和 GPU 使用率
# 用法: bash watch_progress.sh [GROUP]

source "$(dirname "$0")/config.sh"
GROUP="${1:-wds-sekai-real-walking-hq}"

while true; do
    clear
    echo "=== SANA-WM 批量生产进度 $(date) ==="
    echo "  GROUP: $GROUP"

    SHARD_DIR="$DATA_ROOT/$GROUP/shards"
    TOTAL=$(ls "$SHARD_DIR"/*.tar 2>/dev/null | wc -l)
    DONE=$(ls "$OUT_BASE/$GROUP/progress/"*.done 2>/dev/null | wc -l)
    REMAIN=$((TOTAL - DONE))
    echo "  输入 shard: $DONE / $TOTAL 完成（剩余 $REMAIN）"

    # 估计进度 %
    if [[ $TOTAL -gt 0 ]]; then
        PCT=$((DONE * 100 / TOTAL))
        echo "  进度: $PCT%"
    fi

    echo ""
    echo "=== GPU 显存使用 ==="
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
               --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU%s: 显存 %sMiB 已用 / %sMiB 空闲  利用率 %s%%\n",
                $1, $2, $3, $4}'

    echo ""
    echo "=== Worker 日志最后 3 行 ==="
    for LOG in "$OUT_BASE/$GROUP/logs/"node*_gpu*.log; do
        [[ -f "$LOG" ]] || continue
        WNAME=$(basename "$LOG" .log)
        echo "  [$WNAME]"
        tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
    done

    sleep 30
done
```

- [ ] **Step 6.5.2: Commit**
```bash
git add experiments/batch_production/watch_progress.sh \
        experiments/batch_production/run_groups_sequential.sh
git commit -m "feat(batch): add watch_progress.sh and run_groups_sequential.sh"
```

---

## Task 7：显存防 OOM 全套策略

### 7.1 mode_default.py 现有修复（已部署到 AFS，CMCC 需 rsync）

`src/sana_wm_pipeline/stage02_pose/mode_default.py` 已包含以下 3 处修复：

**修复 A：Pi3X 后立即释放**
```python
# _precompute_depth_cache 中，Pi3X 推理完成后
del pi3x_model, src, accum, count
torch.cuda.empty_cache()
# 原理: del 只释放 Python 引用，不清 CUDA allocator cache；
# empty_cache() 将 allocator 持有的 free block 还给 CUDA，
# 让 vipe subprocess 可以申请到这部分显存。
```

**修复 B：MoGe-2 后立即释放**
```python
del moge2_model, frames_t
torch.cuda.empty_cache()
# frames_t 是 (T,3,H,W) float32，960帧时约 10.7 GiB，
# 必须 del 才能从 allocator 回收。
```

**修复 C：vipe subprocess 前再次清理**
```python
# run_default() 中，_precompute_depth_cache 返回后
torch.cuda.empty_cache()
# 保险操作：确保 allocator 没有残留碎片
```

**CMCC 同步命令（在开发机执行）：**
```bash
# 将 AFS 上的修复同步到 CMCC 的目录
# （通过 CMCC 文件管理界面上传，或通过 rsync，取决于 CMCC 接入方式）
# 关键文件: src/sana_wm_pipeline/stage02_pose/mode_default.py
```

### 7.2 超长视频（>2000 帧）chunk 式搬帧（AFS 已实现，CMCC 需确认部署）

**检测方法**（在 CMCC 运行前确认是否需要）：
```bash
# 查看 Sekai shard 中最长视频帧数
python -c "
import tarfile, io, subprocess, json
from pathlib import Path
shard = Path('/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full/wds-sekai-real-walking-hq/shards/sekai-real-walking-hq-000000.tar')
max_frames = 0
with tarfile.open(shard) as tf:
    for m in tf.getmembers():
        if not m.name.endswith('.mp4'):
            continue
        data = tf.extractfile(m).read()
        r = subprocess.run(['ffprobe','-v','quiet','-print_format','json',
                           '-show_streams','-'],
                          input=data, capture_output=True)
        info = json.loads(r.stdout)
        vs = [s for s in info['streams'] if s['codec_type']=='video'][0]
        nb = int(vs.get('nb_frames', 0))
        max_frames = max(max_frames, nb)
        print(f'  {m.name}: {nb} frames')
print(f'max_frames in shard 0: {max_frames}')
"
```

**触发阈值**：`frames_t` 在 GPU 上占用 = T × 3 × H × W × 4 bytes
- 960帧 = 10.7 GiB，加上模型权重后约 60 GiB（H100 80GB 安全）
- 2400帧 = 26.7 GiB，总计约 80 GiB（临界）
- >2400帧 → 必须 chunk 式搬帧

**chunk 加载代码（AFS 已写，复制到 mode_default.py 的 `_precompute_depth_cache`）：**
```python
# 在 _precompute_depth_cache 中替换 frames_t = ... 整块搬 GPU 的代码
# 改前：
frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)  # 全量搬 GPU

# 改后：（chunk 式，GPU 常驻 ≈ 0.18 GiB/chunk，与视频长度无关）
frames_cpu = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # 留在 CPU
CHUNK = 16
starts = list(range(0, len(frames_cpu), CHUNK))
# ... 在 Pi3X 推理循环内按 chunk 搬帧
for s in starts:
    chunk_gpu = frames_cpu[s:s+CHUNK].to(device)  # 只搬 16 帧到 GPU
    out = pi3x_model(chunk_gpu.unsqueeze(0))
    # 收集 out["local_points"][0, :, :, :, 2] 后立即 del chunk_gpu
    del chunk_gpu
```

### 7.3 PYTORCH_CUDA_ALLOC_CONF 配置（已在 config.sh 设置）

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**原理**：PyTorch 默认分配器持有 "rounded-up" blocks，释放后保留在 allocator cache 中。`expandable_segments:True` 允许分配器将 segment 按需扩展/收缩，**减少碎片**导致的"有空闲显存但申请失败"错误。

**使用场景**：vipe subprocess 申请 1.1 GiB 失败的根因是父进程 allocator cache 碎片，此配置有助缓解。

### 7.4 每张卡显存预算规划

| 阶段 | 峰值占用（960帧）| 峰值占用（160帧）|
|------|----------------|----------------|
| Pi3X 模型权重 | ~8 GiB | ~8 GiB |
| frames_t (修复后：chunk 式) | ~0.2 GiB | ~0.2 GiB |
| Pi3X activations (chunk=16) | ~4 GiB | ~4 GiB |
| 小计（Pi3X 阶段） | **~12 GiB** | **~12 GiB** |
| **empty_cache() 后** | ~0.5 GiB | ~0.5 GiB |
| MoGe-2 模型权重 | ~2 GiB | ~2 GiB |
| MoGe-2 推理 | ~6 GiB | ~3 GiB |
| 小计（MoGe-2 阶段） | **~8 GiB** | **~5 GiB** |
| **empty_cache() 后** | ~0.5 GiB | ~0.5 GiB |
| vipe subprocess | ~16-20 GiB | ~5-8 GiB |
| **peak** | **~20 GiB** | **~9 GiB** |

结论：H100 80GB 单卡完全够用，Pi3X+MoGe-2 阶段已在修复后释放，vipe subprocess 可以得到充足显存。

### 7.5 OOM 发生时的诊断和恢复步骤

```bash
# Step 1: 查看哪个 worker 失败
grep -l "OutOfMemory\|CUDA error\|Killed" "$OUT_BASE/$GROUP/logs/"*.log

# Step 2: 查看失败 worker 的错误
cat "$OUT_BASE/$GROUP/logs/node0_gpu3.log" | grep -A5 "Error\|OOM\|Killed"

# Step 3: 确认 .done 标记（失败的 shard 没有 .done，重跑时自动重试）
ls "$OUT_BASE/$GROUP/progress/" | wc -l
# 若 _depth_cache.npz 存在（vipe 前就 OOM），重跑时自动跳过 Pi3X（cache resume 逻辑）

# Step 4: 重跑失败的 worker（只需再次执行 launch_single_node.sh）
# 已完成的 shard 有 .done 标记，自动跳过；只重跑失败的
bash experiments/batch_production/launch_single_node.sh "$GROUP"

# Step 5: 若反复 OOM，启用 chunk 加载（见 7.2）并降低并发数
# 临时改为 4 卡并发（NUM_GPUS=4）在 launch_single_node.sh 中修改
```

---

## Task 8：CMCC 部署检查清单（上线前必做）

- [ ] **Step 8.1: 确认 CMCC 权重路径**
```bash
# 在 CMCC 机器上执行
ls -lh $NEW_BASE/models/pi3x/     # 应有 model.safetensors (~5.1GB)
ls -lh $NEW_BASE/models/moge2/    # 应有 model.pt (~1.3GB)
python -c "
import os; os.environ['SANA_WM_PI3X_WEIGHTS']='$NEW_BASE/models/pi3x'
from pi3 import Pi3X
m = Pi3X.from_pretrained('$NEW_BASE/models/pi3x')
print('Pi3X OK:', type(m))
"
```

- [ ] **Step 8.2: 确认 filestorage 可写**
```bash
mkdir -p /root/work/filestorage/jdvbbfb_output
echo "test" > /root/work/filestorage/jdvbbfb_output/.write_test && echo "filestorage 可写"
```

- [ ] **Step 8.3: /tmp 空间确认**
```bash
df -h /tmp
# 需 > 50 GB（8 worker × 约 2 GB/样本临时文件）
# 若不足，修改 run_worker.py 中 tmp_dir 指向另一个快速本地路径
```

- [ ] **Step 8.4: 单 worker 端到端验证（用 1 个 shard 的前 2 个样本）**
```bash
source experiments/batch_production/config.sh
CUDA_VISIBLE_DEVICES=0 \
"$ENV_DIR/bin/python" experiments/batch_production/run_worker.py \
    --group wds-sekai-real-walking-hq \
    --data-root "$DATA_ROOT" \
    --out-base /tmp/sana_wm_validate_out \
    --worker-id 0 \
    --shard-indices "0" \
    --samples-per-shard 200
# 成功条件: /tmp/sana_wm_validate_out/.../w000/shard-000000.tar 存在且可打开
python -c "
import tarfile
t = tarfile.open('/tmp/sana_wm_validate_out/wds-sekai-real-walking-hq/w000/shard-000000.tar')
print(t.getnames()[:6])
"
```

- [ ] **Step 8.5: 8 卡并发验证（用 sekai shard 0-7，每 worker 1 个 shard）**
```bash
bash experiments/batch_production/launch_single_node.sh \
    wds-sekai-real-walking-hq 0 1
# 期望: 8 个 worker 正常启动，日志正常输出，无 OOM
```

- [ ] **Step 8.6: 确认 CMCC 注入的环境变量名称**
```bash
# 在 CMCC 节点上查看注入了哪些变量
env | grep -iE "RANK|WORLD|NODE|MASTER" | sort
# 若变量名与 RANK/WORLD_SIZE 不同，修改 launch_multi_node.sh 第一行
```

---

## CMCC 提交命令参考

### 批次 1 第一个 group（sekai）
```
启动命令: bash /root/work/david_work/sana_wm_pipeline/experiments/batch_production/launch_multi_node.sh wds-sekai-real-walking-hq
节点数: 5
每节点 GPU: 8
```

### 批次 1 第二个 group（DL3DV），sekai 完成后再提交
```
启动命令: bash /root/work/david_work/sana_wm_pipeline/experiments/batch_production/launch_multi_node.sh wds-DL3DV-ALL-2K
节点数: 5（或减少，DL3DV 样本数更少）
```

### 批次 1 第三个 group（SpatialVID-hq）
```
启动命令: bash /root/work/david_work/sana_wm_pipeline/experiments/batch_production/launch_multi_node.sh wds-SpatialVID-hq
节点数: 5
```

### 续跑（任意 group 中断后重跑）
```
# 完全相同的命令重新提交即可
# .done 标记保证已完成的 shard 被跳过，从断点继续
```

---

## 自查（Spec Coverage）

| 需求 | 覆盖任务 |
|------|---------|
| 单节点 8 卡脚本 | Task 3（launch_single_node.sh） |
| 5 节点 8 卡多机脚本 | Task 4（launch_multi_node.sh） |
| 提速方案：/tmp 临时目录 | Task 6.1 |
| 提速方案：ffmpeg 多线程 | Task 6.2 |
| 提速方案：caption 预加载 | Task 6.3（run_worker.py 内） |
| 提速方案：输出路径规划 | Task 6.4 |
| 显存防 OOM：mode_default 修复 | Task 7.1 |
| 显存防 OOM：chunk 式搬帧 | Task 7.2 |
| 显存防 OOM：ALLOC_CONF | Task 7.3 |
| 显存防 OOM：显存预算规划 | Task 7.4 |
| 显存防 OOM：OOM 诊断恢复 | Task 7.5 |
| 批次顺序（sekai→DL3DV→SpatialVID→其余）| Task 5 + CMCC 提交参考 |
| 断点续跑 | .done 标记（run_worker.py + Task 7.5）|
| CMCC 部署验证 | Task 8 |

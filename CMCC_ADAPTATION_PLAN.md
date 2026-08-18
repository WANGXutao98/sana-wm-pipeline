# CMCC 适配开发方案

**分支**: `refactor/sana-wm-align-cmcc`  
**生成日期**: 2026-08-16  
**目标**: 完成 CMCC 离线环境适配，确保冒烟测试通过

---

## 一、环境相关问题

### 1.1 CMCC 平台适配开发需要完成的工作

根据 `CMCC_MIGRATION_COMPLETE_GUIDE.md`，需要完成：

#### 阶段 1：环境配置（CMCC 侧）

1. **解压代码包**
   ```bash
   cd /root/work/david_work/sana_wm_optimized
   tar -xzf sana_wm_pipeline_clean_20260816.tar.gz
   ```

2. **修改硬编码路径**（3 个核心文件）
   - `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py`
   - `scripts/stage3_batch_minimal.py`
   - `scripts/stage3_test_5s.py`
   
   **修改内容**:
   ```bash
   # 路径映射
   /mnt/afs/davidwang/models/                      → /root/work/david_work/models/
   /mnt/afs/davidwang/workspace/sana_wm_pipeline/  → /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/
   /mnt/afs/davidwang/workspace/data/*/tmp/        → /root/work/david_work/tmp/
   ```

3. **设置环境变量**
   ```bash
   # 添加到 ~/.bashrc
   export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
   export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2
   export TORCH_HOME=/root/work/david_work/cache/torch
   export HF_HOME=/root/work/david_work/cache/huggingface
   export PYTHONPATH=/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:$PYTHONPATH
   ```

4. **创建必需目录**
   ```bash
   mkdir -p /root/work/david_work/models/{pi3x,moge2}
   mkdir -p /root/work/david_work/tmp
   mkdir -p /root/work/david_work/cache/{torch,huggingface}
   mkdir -p /root/work/david_work/smoke_pass_videos
   mkdir -p /root/work/david_work/smoke_pass_results
   ```

5. **激活 Conda 环境**（已提前配置）
   ```bash
   conda activate sana_wm_qc_env
   ```

#### 阶段 2：代码适配（本地开发）

在 `refactor/sana-wm-align-cmcc` 分支完成：

1. **新增 CMCC 专用冒烟测试脚本**（见第三部分）
2. **更新 batch_production 脚本**（见第二部分）
3. **验证路径配置的一致性**

---

### 1.2 环境自检命令

激活 `sana_wm_qc_env` 后，执行以下自检：

```bash
# ========== 自检脚本 ==========
#!/bin/bash
# 文件名: cmcc_env_check.sh

echo "=== CMCC 环境自检 ==="
echo ""

# 1. Conda 环境
echo "1. Conda 环境检查"
conda info | grep "active environment"
python --version
echo ""

# 2. 关键 Python 包
echo "2. Python 包导入测试"
python -c "
import sys
print(f'Python: {sys.version}')

# 核心依赖
import torch
print(f'✓ torch {torch.__version__} (cuda: {torch.cuda.is_available()})')

import numpy as np
print(f'✓ numpy {np.__version__}')

import cv2
print(f'✓ opencv {cv2.__version__}')

# 项目模块（需要先设置 PYTHONPATH）
try:
    import sana_wm_pipeline
    print(f'✓ sana_wm_pipeline')
except ImportError as e:
    print(f'✗ sana_wm_pipeline: {e}')

try:
    from pi3 import Pi3X
    from moge.model.v2 import MoGeModel
    print(f'✓ pi3 (Pi3X)')
    print(f'✓ moge (MoGeModel)')
except ImportError as e:
    print(f'✗ pi3/moge: {e}')

# vipe 模块（third_party/vipe 需要在 PYTHONPATH 中）
try:
    import vipe
    print(f'✓ vipe (third_party/vipe)')
except ImportError as e:
    print(f'✗ vipe: {e}')
"
echo ""

# 3. 环境变量
echo "3. 环境变量检查"
echo "SANA_WM_PI3X_WEIGHTS  = ${SANA_WM_PI3X_WEIGHTS:-未设置}"
echo "SANA_WM_MOGE2_WEIGHTS = ${SANA_WM_MOGE2_WEIGHTS:-未设置}"
echo "TORCH_HOME            = ${TORCH_HOME:-未设置}"
echo "PYTHONPATH            = ${PYTHONPATH:-未设置}"
echo ""

# 4. 关键目录
echo "4. 关键目录检查"
check_dir() {
    if [ -d "$1" ]; then
        echo "✓ $1"
    else
        echo "✗ $1 (不存在)"
    fi
}

check_dir "/root/work/david_work/sana_wm_optimized/sana_wm_pipeline"
check_dir "/root/work/david_work/models"
check_dir "/root/work/david_work/tmp"
check_dir "/root/work/david_work/smoke_pass_videos"
check_dir "/root/work/david_work/smoke_pass_results"
echo ""

# 5. 模型权重
echo "5. 模型权重检查"
check_file() {
    if [ -f "$1" ]; then
        echo "✓ $1 ($(du -sh "$1" | cut -f1))"
    else
        echo "✗ $1 (不存在)"
    fi
}

check_file "/root/work/david_work/models/DOVER/pretrained_weights/DOVER.pth"
check_file "/root/work/david_work/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth"
echo ""

# 6. GPU 检查
echo "6. GPU 状态"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 7. 测试数据
echo "7. 冒烟测试数据"
VIDEO_COUNT=$(find /root/work/david_work/smoke_pass_videos -name "*.mp4" 2>/dev/null | wc -l)
echo "视频文件数量: $VIDEO_COUNT"
if [ $VIDEO_COUNT -gt 0 ]; then
    echo "样本示例:"
    find /root/work/david_work/smoke_pass_videos -name "*.mp4" | head -3
fi
echo ""

echo "=== 自检完成 ==="
```

**预期输出**:
- ✅ 所有 Python 包可导入
- ✅ 环境变量已设置
- ✅ 关键目录存在
- ✅ 模型权重已部署
- ✅ GPU 可用
- ✅ 测试数据存在

---

## 二、旧脚本改动评估

### 2.1 目录内容分析

`experiments/batch_production/` 包含：

| 文件 | 用途 | 是否需要修改 |
|------|------|-------------|
| `config.sh` | 环境配置、路径定义 | ⚠️ **需要修改** |
| `run_worker.py` | 单 GPU Worker 主程序 | ✅ **无需修改** |
| `launch_all_nodes.sh` | 多节点启动脚本 | ⚠️ **需要修改** |
| `launch_single_node.sh` | 单节点启动脚本 | ⚠️ **需要修改** |
| `run_groups_sequential.sh` | 顺序处理多组 | ⚠️ **需要修改** |
| `sync_to_nodes.sh` | 代码同步脚本 | ⚠️ **需要修改** |
| `stop_all_nodes.sh` | 停止所有节点 | ✅ **无需修改** |
| `watch_progress.sh` | 进度监控 | ✅ **无需修改** |
| `shard_io.py` | Shard IO 工具 | ✅ **无需修改** |
| `README.md` | 文档 | ❌ **可选更新** |

---

### 2.2 详细改动说明

#### 2.2.1 `config.sh` - 必须修改

**当前状态**:
```bash
export NEW_BASE="${NEW_BASE:-/root/work/david_work}"
export ENV_DIR="$NEW_BASE/sana_wm_env"  # ← 环境名称错误
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"

export DATA_ROOT="${DATA_ROOT:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full}"
export OUT_BASE="${OUT_BASE:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output}"

source "$NEW_BASE/activate_sana_wm.sh"  # ← 脚本可能不存在
```

**问题**:
1. **环境名称错误**: `sana_wm_env` → 应为 `sana_wm_qc_env`
2. **激活脚本不存在**: `activate_sana_wm.sh` 需要创建或使用 conda 标准方式
3. **路径映射**: 与当前最新分支的路径不完全一致

**修改方案**:
```bash
#!/bin/bash
# ── 基础路径 ──────────────────────────────────────────────────────────────────
export NEW_BASE="${NEW_BASE:-/root/work/david_work}"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"

# DATA_ROOT / OUT_BASE 可被启动脚本外部 export 覆盖
export DATA_ROOT="${DATA_ROOT:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full}"
export OUT_BASE="${OUT_BASE:-/root/work/filestorage/davidwang/jdvbbfb_output}"

# ── 模型权重路径 ──────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# ── conda 环境激活（修正环境名称）─────────────────────────────────────────────
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
        source /root/miniconda3/etc/profile.d/conda.sh
    conda activate sana_wm_qc_env
fi

# ── 离线模式（CMCC 无外网）────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── 显存优化：减少 CUDA allocator 碎片 ────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 添加项目到 PYTHONPATH ─────────────────────────────────────────────────────
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party:$PYTHONPATH"

# ── 批次 1 优先 group（仅在不传 --groups 时作默认队列）────────────────────────
BATCH1_GROUPS=(
    "wds-sekai-real-walking-hq"
    "wds-DL3DV-ALL-2K"
    "wds-SpatialVID-hq"
)
export BATCH1_GROUPS

# 每个输出 shard 最多样本数
export SAMPLES_PER_OUTPUT_SHARD=200
```

**修改理由**:
- 修正环境名称为实际存在的 `sana_wm_qc_env`
- 移除对不存在的 `activate_sana_wm.sh` 的依赖
- 使用 conda 标准激活方式
- 添加 `PYTHONPATH` 设置
- 调整输出路径为文档建议的标准路径

---

#### 2.2.2 `launch_all_nodes.sh` - 需要修改

**当前状态**:
```bash
HOSTFILE="${HOSTFILE:-/root/work/david_work/sana-wm-48-hostfiles/hostfile}"
```

**问题**:
- Hostfile 路径可能在新环境中不存在

**修改方案**:
```bash
# 添加 hostfile 检查
HOSTFILE="${HOSTFILE:-/root/work/david_work/hostfiles/hostfile}"
if [ ! -f "$HOSTFILE" ]; then
    echo "错误: Hostfile 不存在: $HOSTFILE"
    echo "请创建 hostfile 或设置 HOSTFILE 环境变量"
    exit 1
fi
```

**修改理由**: 增加容错性，避免 hostfile 不存在时静默失败

---

#### 2.2.3 `launch_single_node.sh` - 需要修改

**问题**: 依赖 `config.sh`，后者已修改，需验证兼容性

**修改方案**: 无需修改代码，但需测试确认 `config.sh` 修改后的兼容性

---

#### 2.2.4 `run_groups_sequential.sh` - 需要修改

**问题**: 同上，依赖 `config.sh`

**修改方案**: 无需修改代码，测试验证即可

---

#### 2.2.5 `sync_to_nodes.sh` - 需要修改

**当前状态**:
```bash
HOSTFILE="${HOSTFILE:-/root/work/david_work/sana-wm-48-hostfiles/hostfile}"
LOCAL_DIR="${LOCAL_DIR:-/root/work/david_work/sana_wm_optimized/sana_wm_pipeline}"
```

**问题**: Hostfile 路径可能不存在

**修改方案**: 同 `launch_all_nodes.sh`，添加 hostfile 检查

---

#### 2.2.6 `run_worker.py` - 无需修改

**理由**:
- 代码已支持多模式（`default`/`gt_depth`/`gt_pose`）
- 路径通过命令行参数传入，无硬编码
- 与最新分支功能一致

---

### 2.3 修改优先级

| 优先级 | 文件 | 修改内容 | 影响范围 |
|-------|------|---------|---------|
| **P0** | `config.sh` | 修正环境名称、激活方式、路径 | 所有脚本依赖 |
| **P1** | `launch_all_nodes.sh` | 添加 hostfile 检查 | 多节点启动 |
| **P1** | `sync_to_nodes.sh` | 添加 hostfile 检查 | 代码同步 |
| **P2** | `README.md` | 更新文档说明 | 可选 |

---

## 三、生成 CMCC 冒烟测试启动脚本

### 3.1 脚本设计

**文件名**: `experiments/data_production_smoke/smoke_cmcc_pass.sh`

**设计要点**:
1. 基于 `smoke_sekai.sh`（CMCC 历史可运行）的结构
2. 适配 `smoke_spatialvid.sh` 的批量处理逻辑
3. 使用 `sana_wm_qc_env` 环境
4. 支持多个视频的批量测试
5. 输出详细的验证报告

---

### 3.2 脚本内容

```bash
#!/bin/bash
# CMCC 冒烟测试脚本（Pass Videos）
# 用途：验证 CMCC 环境部署成功，确保核心功能可用
# 用法：bash experiments/data_production_smoke/smoke_cmcc_pass.sh
set -euo pipefail

# ── 路径配置（CMCC）──────────────────────────────────────────────────────────
export NEW_BASE="/root/work/david_work"
export PROJ_DIR="$NEW_BASE/sana_wm_optimized/sana_wm_pipeline"
export VIDEO_DIR="$NEW_BASE/smoke_pass_videos"
export OUT_BASE="$NEW_BASE/smoke_pass_results"

# ── 模型权重 ──────────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# ── 离线模式 + 优化 ───────────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 激活环境 ──────────────────────────────────────────────────────────────────
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
        source /root/miniconda3/etc/profile.d/conda.sh
    conda activate sana_wm_qc_env
fi

# 添加项目到 PYTHONPATH
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party:$PYTHONPATH"

echo "========================================"
echo "CMCC 冒烟测试 - Pass Videos"
echo "========================================"
echo "项目目录: $PROJ_DIR"
echo "视频目录: $VIDEO_DIR"
echo "输出目录: $OUT_BASE"
echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo ""

# ── 预检：环境验证 ────────────────────────────────────────────────────────────
echo "=== [1/6] 环境预检 ==="
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')

import sana_wm_pipeline
print('✓ sana_wm_pipeline')

import vipe
from pi3 import Pi3X
from moge.model.v2 import MoGeModel
print('✓ vipe, pi3, moge')

import numpy as np, cv2
print(f'✓ numpy {np.__version__}, opencv {cv2.__version__}')
"

if [ $? -ne 0 ]; then
    echo "✗ 环境预检失败，请检查依赖"
    exit 1
fi

echo ""

# ── 检查视频目录 ──────────────────────────────────────────────────────────────
if [ ! -d "$VIDEO_DIR" ]; then
    echo "✗ 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

VIDEO_FILES=($(find "$VIDEO_DIR" -name "*.mp4" | sort))
VIDEO_COUNT=${#VIDEO_FILES[@]}

if [ $VIDEO_COUNT -eq 0 ]; then
    echo "✗ 视频目录为空: $VIDEO_DIR"
    exit 1
fi

echo "=== [2/6] 发现视频文件 ==="
echo "视频数量: $VIDEO_COUNT"
for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO" .mp4)
    SIZE=$(du -sh "$VIDEO" | cut -f1)
    echo "  [$((i+1))/$VIDEO_COUNT] $BASENAME ($SIZE)"
done
echo ""

# ── 创建输出目录 ──────────────────────────────────────────────────────────────
mkdir -p "$OUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_BASE/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "=== [3/6] 输出目录 ==="
echo "本次运行: $RUN_DIR"
echo ""

# ── 批量处理视频 ──────────────────────────────────────────────────────────────
cd "$PROJ_DIR"

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_VIDEOS=()

for i in "${!VIDEO_FILES[@]}"; do
    VIDEO="${VIDEO_FILES[$i]}"
    BASENAME=$(basename "$VIDEO" .mp4)
    SCENE_DIR="$RUN_DIR/$BASENAME"
    mkdir -p "$SCENE_DIR"
    
    echo "=== [4/6] 处理样本 [$((i+1))/$VIDEO_COUNT]: $BASENAME ==="
    
    # 复制原始视频到工作目录
    cp "$VIDEO" "$SCENE_DIR/video.mp4"
    
    # Stage 1: 归一化
    echo "  [Stage 1] 视频归一化..."
    NORM_VIDEO="$SCENE_DIR/normalized.mp4"
    python -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
try:
    info = normalize_video(Path('$SCENE_DIR/video.mp4'), Path('$NORM_VIDEO'))
    print(f'  ✓ 归一化完成: {info.n_frames} 帧 @ {info.fps}fps ({info.width}x{info.height})')
except Exception as e:
    print(f'  ✗ 归一化失败: {e}')
    exit(1)
" 2>&1 | tee "$SCENE_DIR/stage1.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 1 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 1)")
        continue
    fi
    
    # Stage 2: VIPE SLAM
    echo "  [Stage 2] VIPE SLAM (Pi3X + MoGe-2)..."
    VIPE_WORK="$SCENE_DIR/vipe_work_default"
    ARTIFACT_JSON="$VIPE_WORK/pose_artifact_default.json"
    mkdir -p "$VIPE_WORK"
    
    python -c "
import json
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default

try:
    art = run_default(Path('$NORM_VIDEO'), Path('$VIPE_WORK'))
    print(f'  ✓ SLAM 完成: poses {art.poses_c2w.shape}, intrinsics {art.intrinsics.shape}')
    
    # 保存结果
    Path('$ARTIFACT_JSON').write_text(json.dumps({
        'poses_c2w': art.poses_c2w.tolist(),
        'intrinsics': art.intrinsics.tolist(),
        'scale_per_frame': art.scale_per_frame.tolist(),
    }))
except Exception as e:
    print(f'  ✗ SLAM 失败: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1 | tee "$SCENE_DIR/stage2.log"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 2 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 2)")
        continue
    fi
    
    # Stage 6: 打包 WebDataset
    echo "  [Stage 6] 打包 WebDataset shard..."
    SHARDS_DIR="$SCENE_DIR/shards"
    mkdir -p "$SHARDS_DIR"
    SHARD="$SHARDS_DIR/$BASENAME.tar"
    
    python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path

try:
    scene_id = "$BASENAME"
    art = json.loads(Path("$ARTIFACT_JSON").read_text())
    poses = np.array(art["poses_c2w"], np.float32)
    intr = np.array(art["intrinsics"], np.float32)
    scale = np.array(art["scale_per_frame"], np.float32)
    
    def add_npy(tf, key, arr):
        b = io.BytesIO()
        np.save(b, arr)
        raw = b.getvalue()
        ti = tarfile.TarInfo(f"{scene_id}.{key}")
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))
    
    with tarfile.open("$SHARD", "w") as tf:
        # 视频
        vb = Path("$NORM_VIDEO").read_bytes()
        ti = tarfile.TarInfo(f"{scene_id}.mp4")
        ti.size = len(vb)
        tf.addfile(ti, io.BytesIO(vb))
        
        # Pose 数据
        add_npy(tf, "poses_c2w.npy", poses)
        add_npy(tf, "intrinsics.npy", intr)
        add_npy(tf, "scale.npy", scale)
        
        # Caption（占位）
        cap = "CMCC smoke test video"
        cb = cap.encode()
        ti = tarfile.TarInfo(f"{scene_id}.caption.txt")
        ti.size = len(cb)
        tf.addfile(ti, io.BytesIO(cb))
        
        # Metadata
        meta = json.dumps({
            "scene_id": scene_id,
            "T": len(poses),
            "mode": "default",
            "dataset": "cmcc_smoke_pass",
            "group": "smoke-test"
        }).encode()
        ti = tarfile.TarInfo(f"{scene_id}.meta.json")
        ti.size = len(meta)
        tf.addfile(ti, io.BytesIO(meta))
    
    print(f'  ✓ Shard 打包完成: {len(poses)} 帧')
except Exception as e:
    print(f'  ✗ 打包失败: {e}')
    exit(1)
PYEOF
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  ✗ Stage 6 失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VIDEOS+=("$BASENAME (Stage 6)")
        continue
    fi
    
    echo "  ✓ 样本处理完成: $BASENAME"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo ""
done

# ── 生成测试报告 ──────────────────────────────────────────────────────────────
echo "=== [5/6] 生成测试报告 ==="

REPORT_FILE="$RUN_DIR/smoke_test_report.txt"
cat > "$REPORT_FILE" <<EOF
CMCC 冒烟测试报告
==================

运行时间: $(date)
输出目录: $RUN_DIR

测试结果
--------
总样本数: $VIDEO_COUNT
成功: $SUCCESS_COUNT
失败: $FAIL_COUNT

EOF

if [ $FAIL_COUNT -gt 0 ]; then
    echo "失败样本:" >> "$REPORT_FILE"
    for failed in "${FAILED_VIDEOS[@]}"; do
        echo "  - $failed" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" <<EOF
详细日志
--------
每个样本的详细日志保存在对应目录下:
  - stage1.log: 视频归一化日志
  - stage2.log: VIPE SLAM 日志

EOF

cat "$REPORT_FILE"

# ── 最终结果 ──────────────────────────────────────────────────────────────────
echo ""
echo "=== [6/6] 测试完成 ==="
echo "报告文件: $REPORT_FILE"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓✓✓ 冒烟测试全部通过 ($SUCCESS_COUNT/$VIDEO_COUNT) ✓✓✓"
    exit 0
else
    echo "✗✗✗ 冒烟测试部分失败 (成功: $SUCCESS_COUNT, 失败: $FAIL_COUNT) ✗✗✗"
    exit 1
fi
```

---

### 3.3 脚本特点

1. **环境自适应**
   - 自动检测并激活 `sana_wm_qc_env`
   - 兼容不同 conda 安装路径

2. **完整的错误处理**
   - 每个阶段独立捕获错误
   - 失败样本不影响后续样本
   - 生成详细的失败报告

3. **批量处理**
   - 支持目录下所有 `.mp4` 文件
   - 自动统计成功/失败数量

4. **详细日志**
   - 每个样本生成独立的 stage1/stage2 日志
   - 最终生成汇总报告

5. **符合 CMCC 环境**
   - 使用 CMCC 标准路径
   - 离线模式配置
   - 显存优化参数

---

### 3.4 使用方法

**CMCC 侧执行**:

```bash
# 1. 准备测试视频
mkdir -p /root/work/david_work/smoke_pass_videos
# 将测试视频复制到该目录

# 2. 运行冒烟测试
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_pass.sh

# 3. 查看结果
ls -la /root/work/david_work/smoke_pass_results/run_*/
cat /root/work/david_work/smoke_pass_results/run_*/smoke_test_report.txt
```

---

## 四、实施计划

### 4.1 本地开发（在 `refactor/sana-wm-align-cmcc` 分支）

1. **修改 `experiments/batch_production/config.sh`**
   - 修正环境名称
   - 更新激活方式
   - 调整路径映射

2. **创建 `experiments/data_production_smoke/smoke_cmcc_pass.sh`**
   - 使用上面提供的脚本内容

3. **创建 `scripts/cmcc_env_check.sh`**
   - 使用上面提供的自检脚本

4. **更新 `experiments/batch_production/launch_all_nodes.sh`**
   - 添加 hostfile 检查

5. **更新 `experiments/batch_production/sync_to_nodes.sh`**
   - 添加 hostfile 检查

6. **创建 `CMCC_ADAPTATION_PLAN.md`**（本文档）

---

### 4.2 提交变更

```bash
# 添加新文件
git add experiments/data_production_smoke/smoke_cmcc_pass.sh
git add scripts/cmcc_env_check.sh
git add CMCC_ADAPTATION_PLAN.md

# 提交修改
git add experiments/batch_production/config.sh
git add experiments/batch_production/launch_all_nodes.sh
git add experiments/batch_production/sync_to_nodes.sh

# 提交
git commit -m "feat: CMCC adaptation for offline deployment

- Update config.sh to use sana_wm_qc_env
- Add smoke_cmcc_pass.sh for CMCC smoke test
- Add cmcc_env_check.sh for environment validation
- Add hostfile checks to multi-node scripts
- Add CMCC_ADAPTATION_PLAN.md documentation"

# 推送到远程
git push -u origin refactor/sana-wm-align-cmcc
```

---

### 4.3 CMCC 侧验证流程

1. **传输代码包到 CMCC**
2. **解压并修改路径**（参考文档第一部分）
3. **运行环境自检**
   ```bash
   bash scripts/cmcc_env_check.sh
   ```
4. **运行冒烟测试**
   ```bash
   bash experiments/data_production_smoke/smoke_cmcc_pass.sh
   ```
5. **验证批量生产脚本**（可选）
   ```bash
   bash experiments/batch_production/launch_single_node.sh
   ```

---

## 五、风险点与注意事项

### 5.1 环境依赖

- ⚠️ `sana_wm_qc_env` 必须提前配置好所有依赖
- ⚠️ Pi3x/Moge2 模型权重必须提前部署到指定路径
- ⚠️ DOVER/UniMatch 权重必须存在

### 5.2 路径一致性

- ⚠️ 所有脚本的路径必须与 `CMCC_PATH_CONFIGURATION_CHECKLIST.md` 保持一致
- ⚠️ 环境变量必须在 `~/.bashrc` 中持久化

### 5.3 测试数据

- ⚠️ 冒烟测试需要准备至少 1-3 个短视频（< 5 秒）
- ⚠️ 视频格式必须为 `.mp4`

### 5.4 回滚方案

如果修改后脚本无法运行，可回退到旧版本：

```bash
git checkout HEAD~1 experiments/batch_production/config.sh
```

---

## 六、文档版本

**版本**: v1.0  
**生成日期**: 2026-08-16  
**维护者**: Claude (Ponytail Mode)  
**下一步**: 人工审核本方案，确认无误后执行实施计划

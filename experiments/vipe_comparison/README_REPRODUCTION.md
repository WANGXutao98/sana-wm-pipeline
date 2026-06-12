# TUM RGB-D 实验复现指南 — VIPE + Pi3X + MoGe-2

## 概览

本指南提供从零开始完全复现 VIPE+MoGe-2+Pi3X 在 TUM RGB-D 数据集上的实验步骤。

**实验内容：** fr1/desk (28s) 和 fr2/desk (99s) 两个序列的位姿估计精度对比
- **Method A：** VIPE + metric3d-small（论文基线）
- **Method B：** VIPE + Pi3X+MoGe-2（SANA-WM 增强版）

**预期结果：**
- fr1/desk: Method B 全 9 项指标领先，ATE RMSE ↓36%
- fr2/desk: Method B 后半段平移漂移 ↓70%，尺度偏差 18.5%→3.3%

---

## 前置条件

### 硬件和环境
- **GPU：** NVIDIA H100 或同等级 (Pi3X + MoGe-2 推理)
- **RAM：** ≥ 64 GB (深度预计算缓存最多 2.1 GB)
- **Storage：** ≥ 50 GB (TUM 下载 + 缓存 + 结果)
- **Conda 环境：** `sana_wm`（来自 `setup.sh`）

### 预下载权重

```bash
# 验证权重存在
test -f /mnt/afs/davidwang/models/pi3x/model.safetensors && echo "✓ Pi3X" || echo "✗ Pi3X"
test -f /mnt/afs/davidwang/models/moge2/model.pt && echo "✓ MoGe-2" || echo "✗ MoGe-2"
```

---

## 快速开始（一键运行）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 设置环境变量
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface

# 一键运行 fr1 + fr2
bash experiments/vipe_comparison/run_corrected.sh

# 或仅运行 fr1
bash experiments/vipe_comparison/run_corrected.sh fr1
```

**总耗时：**
- fr1 only: ~30 min
- fr1 + fr2: ~100 min
- 主要瓶颈：深度预计算 (fr2: 35 min)

---

## 详细步骤

### Step 1: 数据准备 (5 min)

```bash
python experiments/vipe_comparison/prepare_tum.py \
    --out experiments/vipe_comparison/data
```

**输出验证：**
```bash
ls -lh experiments/vipe_comparison/data/rgbd_dataset_freiburg{1,2}_desk/video.mp4
ls experiments/vipe_comparison/data/rgbd_dataset_freiburg{1,2}_desk/gt_aligned.txt
```

期望：两个序列的 video.mp4 和 gt_aligned.txt 都存在

### Step 2: Pi3X + MoGe-2 深度预计算 (8-35 min，取决于序列长度)

关键：这一步离线运行，生成 NPZ 缓存，供 SLAM 使用。

```bash
# fr1/desk
python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --out   experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz

# fr2/desk（耗时更长）
python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4 \
    --out   experiments/vipe_comparison/results/cache_pi3x_moge2_fr2_desk.npz
```

**输出验证：**
```bash
python << 'EOF'
import numpy as np
d = np.load('experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz')
print(f"depths shape: {d['depths'].shape}, dtype: {d['depths'].dtype}")
print(f"scale_history shape: {d['scale_history'].shape}")
print(f"scale range: [{d['scale_history'].min():.3f}, {d['scale_history'].max():.3f}]")
EOF
```

期望：深度形状 (T, H, W)，scale_history 为 EMA 融合历史

### Step 3: Method A — VIPE + metric3d-small (12 min，含模型下载)

```bash
vipe infer experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --pipeline vipe_metric3d_small \
    --output experiments/vipe_comparison/results/method_A_m3d
```

**输出验证：**
```bash
ls -la experiments/vipe_comparison/results/method_A_m3d/pose/video.npz
```

### Step 4: Method B — VIPE + Pi3X+MoGe-2 (8 min)

关键：环境变量 `SANA_WM_CACHED_DEPTH_PATH` 必须设置。

```bash
export SANA_WM_CACHED_DEPTH_PATH=experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz
vipe infer experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --pipeline vipe_cached_depth \
    --output experiments/vipe_comparison/results/method_B_cached
```

**输出验证：**
```bash
ls -la experiments/vipe_comparison/results/method_B_cached/pose/video.npz
```

### Step 5: 评测 (< 1 min)

```bash
python experiments/vipe_comparison/evaluate.py \
    --seq experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk \
    --results experiments/vipe_comparison/results
```

**输出示例：**
```
GT poses loaded: 570 frames
============================================================
Method: A: VIPE + metric3d-small (论文 baseline)
  ATE RMSE:     0.0355 m  (Sim3 对齐后)
  ...

Method: B: VIPE + Pi3X+MoGe-2 (cached, SANA-WM)
  ATE RMSE:     0.0227 m
  ...
```

**图表输出：**
```bash
ls -la experiments/vipe_comparison/results/plots/comparison.png
```

---

## 故障排除

### 错误 1: `vipe command not found`

```bash
# 重新激活环境
conda activate sana_wm
which vipe  # 应显示 /mnt/afs/.../bin/vipe

# 若仍未找到，安装 VIPE
pip install vipe
```

### 错误 2: `ModuleNotFoundError: No module named 'pi3'`

```bash
conda activate sana_wm
pip install pi3  # 或 pi3-0.1
```

### 错误 3: `GeoCalib 权重下载失败`

```bash
# 从 AFS 备份恢复
mkdir -p ~/.cache/torch/hub/geocalib
cp /mnt/afs/davidwang/cache/torch/hub/geocalib/pinhole.tar \
   ~/.cache/torch/hub/geocalib/
```

### 错误 4: `SANA_WM_CACHED_DEPTH_PATH not set`

```bash
# 确保在运行 Method B 前设置环境变量
export SANA_WM_CACHED_DEPTH_PATH=$(pwd)/experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz
echo $SANA_WM_CACHED_DEPTH_PATH  # 验证
```

---

## 实验设计说明

### 为什么是两个方法对比？

- **Method A (metric3d-small)：** VIPE 原始论文使用的深度后端，已在众多 SLAM 任务验证
- **Method B (Pi3X+MoGe-2)：** SANA-WM 论文 App.B.1 提出的增强方案，结合：
  - **Pi3X：** 视频级一致性深度估计（跨帧连贯性）
  - **MoGe-2：** 米制尺度锚定（绝对尺度准确性）
  - **EMA 融合：** 时序平滑（减少闪烁和尺度漂移）

### 为什么是 fr1 和 fr2？

| 序列 | 长度 | 帧数 | 用途 |
|---|---|---|---|
| fr1/desk | 28s | 613 | 短序列基准，验证核心方法有效性 |
| fr2/desk | 99s | 2257 | 长序列基准，验证长视频漂移稳定性 |

TUM fr2 特别重要：Method B 相对 A 的优势从 fr1 的 36% 上升到 fr2 的 70%，验证了长视频稳定性的主张。

### 预期结果解读

```
fr1/desk (短序列):
├─ ATE RMSE 降幅：↓36%（绝对值小，相对比较更有意义）
├─ 尺度偏差：7.9% → 1.1%（MoGe-2 米制锚定效果显著）
└─ RTE 后半漂移：↓34-36%（长视频稳定性开始显现）

fr2/desk (长序列):
├─ ATE RMSE 降幅：↓10%（绝对 ATE 本身更小，百分比降幅意义弱）
├─ 尺度偏差：18.5% → 3.3%（偏差 ↓82%，是核心指标）
└─ RTE 后半漂移：↓70%（长视频稳定性的强力证明）
```

✅ **实验成功标志：** Method B 在 fr2/desk 的尺度偏差和后半段平移漂移上全面领先。

---

## 已知限制

1. **Per-frame intrinsics BA 未实现**（论文 App.B.1 末尾）
   - 当前 VIPE BA intrinsics 是全局固定的
   - 论文理想状态是 (fx, fy, cx, cy) 每帧优化
   - 影响：RTE 旋转指标上 Method B 优势不明显（fr2 甚至略劣于 A）
   - 修复复杂度：需改 VIPE C++/CUDA BA 核心，超出实验范围

2. **帧率差异**
   - TUM 原始采集：30 fps
   - SANA-WM 训练目标：16 fps
   - MP4 生成：30 fps（VIPE 内部可能有帧率处理）

3. **短序列局限**
   - fr1 仅 28s，长视频漂移优势不够突出
   - fr2 (99s) 是更强的验证

---

## 代码清单

### 实验脚本
- `experiments/vipe_comparison/prepare_tum.py` — TUM 数据下载 + MP4 生成
- `experiments/vipe_comparison/precompute_pi3x_depths.py` — Pi3X+MoGe-2 融合
- `experiments/vipe_comparison/run_corrected.sh` — 一键执行脚本
- `experiments/vipe_comparison/evaluate.py` — ATE/RTE 评测

### VIPE 源码修改（5 处）
- `third_party/vipe/vipe/priors/depth/base.py` — 新增 `frame_idx` 字段
- `third_party/vipe/vipe/slam/components/buffer.py` — 传 `frame_idx`
- `third_party/vipe/vipe/priors/depth/__init__.py` — 新增 `cached` 分支
- `third_party/vipe/vipe/priors/depth/cached.py` — 新建 `CachedDepthModel`
- `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml` — Method B 配置

### 配置文件
- `third_party/vipe/configs/pipeline/vipe_metric3d_small.yaml` — Method A
- `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml` — Method B

---

## 下一步

- 在其他数据集（如 DL3DV）上验证方法有效性
- 实现 per-frame intrinsics BA 优化（如有需要）
- 长序列数据集上的进一步评估（如 ScanNet, 7-Scenes）

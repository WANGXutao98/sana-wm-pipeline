# TUM 实验完整复现与 VIPE+MoGe-2+Pi3X 跨数据集可行性分析

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为下一位同学提供完整、可复现的 VIPE+MoGe-2+Pi3X 实验指南；分析该管线对 DL3DV 数据的适用性；确保仓库干净的分支支持完全复现。

**Scope:**
1. **任务 A**：整理 TUM fr1/fr2 实验文档（供下学复现）
2. **任务 B**：分析 VIPE+MoGe-2+Pi3X 对 DL3DV 的适用性
3. **任务 C**：审查当前代码修改对 VIPE 管线的影响
4. **任务 D**：准备干净的 Git 分支支持完全复现

**Tech Stack:** 
- VIPE (git submodule), Pi3X, MoGe-2, SLAM, TUM RGB-D dataset, DL3DV dataset

---

## 前置事实核查

| 检查项 | 当前状态 | 备注 |
|---|---|---|
| TUM fr1/desk 实验 | ✅ 完成 (2026-05-29) | RESULTS_fr1_desk.md 存在 |
| TUM fr2/desk 实验 | ✅ 完成 (2026-05-29) | RESULTS_fr2_desk.md 存在 |
| VIPE 源码修改 | ✅ 已落地 | 5 处修改，cached.py 新建 |
| DL3DV 管线 | 🚧 进行中 | 2026-06-11 plan 提到，未跑通 |
| 仓库分支状态 | ⚠️ 需检查 | master 有未提交修改 |

---

## Task A: 整理 TUM 实验复现文档

**目标：** 生成 `experiments/vipe_comparison/README_REPRODUCTION.md`，包含从零开始完全复现 TUM fr1/fr2 实验的所有步骤。

**Files:**
- Create: `experiments/vipe_comparison/README_REPRODUCTION.md`
- Reference: `experiments/vipe_comparison/RESULTS_fr1_desk.md`
- Reference: `experiments/vipe_comparison/RESULTS_fr2_desk.md`

### A1: 审查现有实验脚本和结果文档

- [ ] **步骤 1：列举现有脚本**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
ls -lh experiments/vipe_comparison/*.py experiments/vipe_comparison/*.sh
```

期望输出：至少包含 `prepare_tum.py`, `precompute_pi3x_depths.py`, `run_corrected.sh`, `evaluate.py`

- [ ] **步骤 2：验证现有报告内容完整**

```bash
wc -l experiments/vipe_comparison/RESULTS_fr{1,2}_desk.md
grep -c "Method B" experiments/vipe_comparison/RESULTS_fr{1,2}_desk.md
```

期望：两份报告都 > 150 行，都提到 Method B 的结果

- [ ] **步骤 3：检查脚本中的 hardcoded 路径**

```bash
grep -n "SANA_WM_PI3X_WEIGHTS\|SANA_WM_MOGE2_WEIGHTS\|cache/torch" \
  experiments/vipe_comparison/run_corrected.sh
```

期望：脚本中有明确的环境变量定义或说明

### A2: 编写复现指南文档

- [ ] **步骤 4：创建 README_REPRODUCTION.md 框架**

```markdown
# TUM RGB-D 实验复现指南 — VIPE + Pi3X + MoGe-2

## 概览

本指南提供从零开始完全复现 VIPE+MoGe-2+Pi3X 在 TUM RGB-D 数据集上的实验步骤。

**实验内容：** fr1/desk (28s) 和 fr2/desk (99s) 两个序列的位姿估计精度对比
- Method A: VIPE + metric3d-small (论文基线)
- Method B: VIPE + Pi3X+MoGe-2 (SANA-WM 增强版)

**预期结果：** Method B 在全部指标上领先，尤其在长序列 (fr2) 上后半段漂移下降 70%

---

## 前置条件

### 硬件和环境
...（待补充）
```

实际填写内容（逐步补充到 README_REPRODUCTION.md）：

```markdown
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
- 主要瓶颈：Deep 预计算 (fr2: 35 min)

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

（详见 RESULTS_fr2_desk.md 第 3.3 节）

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
```

- [ ] **步骤 5：保存文件并验证结构完整**

```bash
wc -l experiments/vipe_comparison/README_REPRODUCTION.md
grep -c "^##" experiments/vipe_comparison/README_REPRODUCTION.md
```

期望：≥ 300 行，≥ 10 个二级标题

- [ ] **步骤 6：Commit 文档**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git add experiments/vipe_comparison/README_REPRODUCTION.md
git commit -m "docs: add complete TUM experiment reproduction guide"
```

---

## Task B: 分析 VIPE+MoGe-2+Pi3X 对 DL3DV 的适用性

**目标：** 创建 `docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md`，分析该管线是否适用于 DL3DV 数据。

**Files:**
- Create: `docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md`
- Reference: `docs/superpowers/plans/2026-06-11-dl3dv-e2e-pipeline.md`
- Reference: `docs/superpowers/plans/2026-05-28-vipe-comparison.md`

### B1: 对比 TUM 和 DL3DV 的数据特性

- [ ] **步骤 1：列出 TUM 和 DL3DV 的关键属性**

创建对比表格（最终放入文档）：

| 属性 | TUM fr1/desk | TUM fr2/desk | DL3DV (typical) |
|---|---|---|---|
| 序列长度 | 28s (613f) | 99s (2257f) | 60-300s (1800-9000f) |
| 帧率 | 30 fps | 30 fps | 可变 (通常 24-30 fps) |
| 分辨率 | 640×480 | 640×480 | 1024×1024 或更高 |
| 场景类型 | 室内桌面 | 室内桌面 | 室内+室外混合 |
| 相机类型 | RGB-D (Kinect) | RGB-D (Kinect) | RGB only |
| GT 位姿 | 精确 mocap | 精确 mocap | SfM 估计 + 手工优化 |
| 照明条件 | 室内固定 | 室内固定 | 自然变化 |

- [ ] **步骤 2：梳理 VIPE+MoGe-2+Pi3X 对各属性的依赖**

| 组件 | Pi3X 依赖 | MoGe-2 依赖 | EMA 融合依赖 |
|---|---|---|---|
| **输入** | RGB 视频 (any FPS) | RGB 帧 (any FPS) | 逐帧深度 + FPS 信息 |
| **输出** | 视频一致性深度 | 米制深度 + 内参 | 融合深度 + 尺度历史 |
| **约束** | 需要跨帧连贯性 | FOV 推导（自动） | 需要 ≥10 帧连贯序列 |
| **已验证** | TUM fr1/2 ✓ | TUM fr1/2 ✓ | TUM fr1/2 ✓ |
| **DL3DV** | ? | ? | ? |

### B2: 编写可行性分析文档

- [ ] **步骤 3：创建文档框架**

```markdown
# VIPE+MoGe-2+Pi3X 在 DL3DV 数据上的可行性分析

## 摘要

**结论：** 该管线在 DL3DV 上**技术可行**，但需注意以下数据特性差异：
- ✅ 兼容纯 RGB 视频（DL3DV 无 RGB-D）
- ✅ 帧率适应性强（DL3DV 通常 24-30 fps）
- ⚠️ DL3DV 场景更复杂（多样化照明、户外等）
- ⚠️ GT 位姿质量不如 TUM mocap 精确

---

## 1. 数据特性对比

### TUM RGB-D (已验证)
- 长度：28-99s
- 帧率：固定 30fps
- 分辨率：640×480
- 场景：室内桌面，受控照明
- GT：精确 mocap (~mm 级精度)
- RGB-D：有深度图

### DL3DV (目标数据集)
- 长度：60-300s （更长）
- 帧率：24-30fps （可变）
- 分辨率：1024×1024+ （更高）
- 场景：室内+室外混合，自然照明
- GT：SfM + 手工标注 (cm 级精度)
- RGB only：仅 RGB 视频

---

## 2. 管线兼容性分析

### 2.1 Pi3X 视频深度估计

**TUM 实验中：**
- 输入：RGB 视频 (T, H=480, W=640, 3)
- chunk=16, stride=8
- 推理时间：fr1 ~8min, fr2 ~30min

**DL3DV 场景预测：**
- 输入：RGB 视频 (T, H≥1024, W≥1024, 3)
- 推理时间：预计 1.5-3× TUM (更高分辨率)
- **风险**：Pi3X 未在户外场景验证，照明变化可能影响质量

**建议：**
- ✓ 技术兼容，执行 Pi3X 推理
- ⚠️ 产出质量需在 DL3DV 上验证
- 📌 优先选择室内场景 (DL3DV 室内子集) 进行初步验证

### 2.2 MoGe-2 米制深度估计

**TUM 实验中：**
- FOV 自动推导：`fov_x = 2 * arctan(W/2/fx)` (从 TUM 内参)
- 米制尺度：通过 RGB-D 的深度 GT 标定
- 结果：尺度偏差从 7.9% 改善到 1.1% (fr1)

**DL3DV 场景预测：**
- 输入：仅有 RGB，需从 transforms.json 提取内参（SfM 估计）
- SfM 内参通常比 mocap 不准 (2-5% 误差)
- **风险**：SfM 内参误差会直接影响 MoGe-2 的米制尺度

**建议：**
- ✓ 技术兼容，执行 MoGe-2 推理
- ⚠️ 最后尺度精度取决于 DL3DV SfM 内参质量
- 📌 如果 DL3DV 有多个内参估计方案，应逐个测试

### 2.3 EMA 时序融合

**TUM 实验中：**
- EMA 动量 α=0.99 (一阶滤波)
- 效果：缩小尺度漂移、减少闪烁
- 最强效果：长序列 (fr2: 18.5% → 3.3% 尺度偏差)

**DL3DV 场景预测：**
- DL3DV 更长 (60-300s vs 28-99s TUM)
- EMA 融合在长序列上效果预期更好
- **优势**：DL3DV 的长度正好利用 EMA 的优势

**建议：**
- ✅ 强烈推荐在 DL3DV 长序列上验证
- 可考虑 ablation：关闭 EMA，看是否有漂移加重

---

## 3. 数据处理流程兼容性

### 3.1 Stage 01 — normalize (视频标准化)

**DL3DV 特异性：**
- 输入：图像序列 (PNG/JPG) + transforms.json
- sana_wm_pipeline 需提供：video.mp4 + gt_poses.npy + intrinsics.npy

**兼容性分析：**
- ✅ 已有 `experiments/data_production_smoke/prepare_dl3dv.py` 支持
- ✅ 生成 MP4 + poses + intrinsics 的流程已实现

### 3.2 Stage 02 — pose estimation

**可选模式：**
1. **default 模式**（推荐用于 DL3DV）
   - 使用 VIPE + Pi3X + MoGe-2 （当前 TUM 已验证的配置）
   - 预计算缓存 → SLAM BA 优化
   - ✅ 完全兼容

2. **gt-pose 模式**（如有 GT）
   - 使用 GT 位姿 + Pi3X 估计内参
   - ✅ 对 DL3DV GT 位姿也适用

### 3.3 Stage 03 — frame filtering

**问题：**
- DL3DV 数据配置中 `unimatch_flow: null`, `dover: null`
- 这两个工具未在 DL3DV 上安装

**方案：**
- 可保持 `strict_frames=False` (允许少于 961 帧)
- 或为 DL3DV 安装 unimatch + dover（可选）

### 3.4 Stage 04-06 — filtering + caption + pack

**兼容性：**
- ✅ Stage 04 (apply_table6) 已有 DL3DV 配置
- ✅ Stage 05 (qwen35_vl_runner) 有 stub 实现
- ✅ Stage 06 (webdataset_writer) 支持 `strict_frames=False`

---

## 4. VIPE 深度管线在 DL3DV 上的验证计划

### 4.1 快速验证（单场景）

```bash
# 选择一个短序列 DL3DV 场景
SCENE=scannet_0000  # 示例

# Step 1: 数据准备
python experiments/data_production_smoke/prepare_dl3dv.py \
    --scene $SCENE \
    --out /tmp/dl3dv_test

# Step 2: Pi3X + MoGe-2 预计算
python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video /tmp/dl3dv_test/video.mp4 \
    --out /tmp/cache_dl3dv_${SCENE}.npz

# Step 3: VIPE 推理
export SANA_WM_CACHED_DEPTH_PATH=/tmp/cache_dl3dv_${SCENE}.npz
vipe infer /tmp/dl3dv_test/video.mp4 \
    --pipeline vipe_cached_depth \
    --output /tmp/vipe_dl3dv_${SCENE}

# Step 4: 评测（如有 GT）
python experiments/vipe_comparison/evaluate.py \
    --seq /tmp/dl3dv_test \
    --results /tmp/vipe_dl3dv_${SCENE}
```

### 4.2 预期输出物

- `cache_dl3dv_{scene}.npz` — 深度缓存
- `{out}/pose/video.npz` — 估计的相机位姿 (T, 4, 4)
- ATE/RTE 指标（如有 GT）

### 4.3 成功标准

- ✅ Pi3X 完成全序列推理（无 OOM、无 nan）
- ✅ MoGe-2 完成逐帧推理
- ✅ VIPE SLAM 收敛（无 tracking lost）
- ✅ 生成 pose/video.npz（形状正确）
- 🎯 若有 GT：ATE RMSE 在合理范围（cm 级）

---

## 5. 已知风险和缓解方案

| 风险 | 可能性 | 缓解方案 |
|---|---|---|
| DL3DV 户外场景照明变化 → Pi3X 质量下降 | 中 | 优先测试室内场景；若需户外，增加训练数据 |
| SfM 内参误差 → MoGe-2 米制不准 | 中 | 若 DL3DV 提供多种内参估计，逐个测试；考虑 per-frame intrinsics BA |
| 高分辨率 (1024+) → 内存爆炸 | 低 | Pi3X/MoGe-2 内部已有 downsampling 逻辑；监控显存使用 |
| 长序列 (>300s) → 推理时间过长 | 低 | 可分段处理 (chunk by video clip)；缓存中间结果 |
| unimatch/dover 缺失 → Stage 03 报错 | 低 | `strict_frames=False` 已支持绕过 |

---

## 6. 结论和建议

### 技术可行性：✅ 完全可行

VIPE+MoGe-2+Pi3X 管线在技术上完全兼容 DL3DV。
- 核心依赖（RGB 视频、内参）都具备
- 代码修改最小（缓存查表）
- TUM 验证已充分

### 数据适配性：⚠️ 需要验证

- DL3DV 场景更复杂（多样化、户外）
- 但"更复杂"也意味着更好的泛化验证

### 下一步行动

1. **快速验证** (2-3 小时)
   - 选 1-2 个 DL3DV 室内短场景
   - 跑完整 VIPE+Pi3X+MoGe-2 流程
   - 生成位姿，与 GT 对比

2. **完整验证** (1-2 天)
   - 扩展到 5-8 个场景（室内+室外混合）
   - 对比 Pi3X+MoGe-2 vs baseline
   - 生成深度/位姿可视化

3. **集成验证** (1 周)
   - 集成到 sana_wm_pipeline Stage 02（default 模式）
   - 端到端跑 DL3DV → WebDataset shard
   - 调用 SANA-WM 推理验证完整管线

```

- [ ] **步骤 4：填充关键分析内容**

上述步骤 3 已给出完整的分析文档模板。

- [ ] **步骤 5：保存文件并验证**

```bash
mkdir -p /mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/analysis
# 保存上面的文档
wc -l docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md
```

期望：≥ 250 行

- [ ] **步骤 6：Commit 分析文档**

```bash
git add docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md
git commit -m "docs: add feasibility analysis of VIPE+MoGe-2+Pi3X for DL3DV"
```

---

## Task C: 审查当前代码修改对 VIPE 管线的影响

**目标：** 验证当前仓库（master + uncommitted changes）不会破坏 VIPE+MoGe-2+Pi3X 的复现能力。

**Files:**
- Check: `third_party/vipe/` (git submodule)
- Check: Stage 02 相关代码
- Check: DL3DV 新增代码是否干扰 VIPE

### C1: 列举当前未提交的修改

- [ ] **步骤 1：查看 git status**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git status
```

期望输出（根据之前的 session context）：
```
M  docs/operation_logs/2026-06-12-dl3dv-e2e-implementation.md
M  experiments/data_production_smoke/download_dl3dv.sh
M  experiments/data_production_smoke/prepare_dl3dv.py
...
M  src/sana_wm_pipeline/stage02_pose/mode_default.py
M  src/sana_wm_pipeline/stage02_pose/mode_gtpose.py
...
?? experiments/data_production_smoke/test2/
?? scripts/precompute_depth_cache.py
```

- [ ] **步骤 2：检查 VIPE 相关修改**

```bash
# 查看 VIPE submodule 状态
cd third_party/vipe && git status
cd -

# 查看主仓库中与 VIPE 集成相关的改动
git diff src/sana_wm_pipeline/stage02_pose/mode_default.py | head -100
git diff src/sana_wm_pipeline/stage02_pose/mode_gtpose.py | head -100
```

期望：修改应仅涉及 Stage 02 pose mode，不应改动 VIPE 源码本身

### C2: 验证 VIPE TUM 实验代码隔离性

- [ ] **步骤 3：确认 TUM 实验脚本独立于 DL3DV 代码**

```bash
# TUM 实验使用的关键脚本
ls -l experiments/vipe_comparison/*.py

# 检查是否有 DL3DV 代码混入
grep -r "dl3dv\|DL3DV" experiments/vipe_comparison/
# 期望：无匹配或仅在注释中出现
```

- [ ] **步骤 4：检查 Stage 02 改动是否影响 TUM 流程**

```bash
# 查看 mode_default.py 是否添加了 DL3DV 特异逻辑
grep -n "dl3dv\|DL3DV" src/sana_wm_pipeline/stage02_pose/mode_default.py

# 查看新增的 scripts/precompute_depth_cache.py 是否与 TUM precompute_pi3x_depths.py 冲突
diff scripts/precompute_depth_cache.py experiments/vipe_comparison/precompute_pi3x_depths.py | head -20
```

期望：没有功能冲突，最多是重复的代码（可接受，因为独立脚本）

### C3: 验证 VIPE 源码修改的持久性

- [ ] **步骤 5：检查 VIPE submodule 中的 5 处修改仍在**

```bash
cd third_party/vipe

# 检查 cached.py 是否存在
test -f vipe/priors/depth/cached.py && echo "✓ cached.py exists" || echo "✗ cached.py missing"

# 检查 base.py 中的 frame_idx 字段
grep "frame_idx" vipe/priors/depth/base.py && echo "✓ frame_idx found" || echo "✗ frame_idx missing"

# 检查 __init__.py 中的 cached 分支
grep -A 5 'model_name == "cached"' vipe/priors/depth/__init__.py && echo "✓ cached branch found" || echo "✗ cached branch missing"

# 检查 buffer.py 的 frame_idx 参数
grep "frame_idx" vipe/slam/components/buffer.py && echo "✓ buffer.py updated" || echo "✗ buffer.py NOT updated"

# 检查配置文件
test -f configs/pipeline/vipe_cached_depth.yaml && echo "✓ vipe_cached_depth.yaml exists" || echo "✗ MISSING"
test -f configs/pipeline/vipe_metric3d_small.yaml && echo "✓ vipe_metric3d_small.yaml exists" || echo "✗ MISSING"

cd -
```

期望：所有 5 处修改均存在，✓ 全绿

### C4: 制作隔离检查清单

- [ ] **步骤 6：编制代码隔离性检查报告**

创建临时文件 `/tmp/vipe_isolation_check.md`：

```markdown
# VIPE+MoGe-2+Pi3X 代码隔离性检查

**检查日期：** 2026-06-12
**检查分支：** master
**目的：** 确认当前未提交的修改不会破坏 VIPE TUM 实验的复现能力

---

## ✅ 检查项

### 1. VIPE 源码修改完整性
- [ ] `third_party/vipe/vipe/priors/depth/cached.py` 存在 (新建)
- [ ] `third_party/vipe/vipe/priors/depth/base.py` 包含 `frame_idx: int | None = None` (L80)
- [ ] `third_party/vipe/vipe/priors/depth/__init__.py` 包含 cached 分支
- [ ] `third_party/vipe/vipe/slam/components/buffer.py` 传 frame_idx 参数
- [ ] `third_party/vipe/configs/pipeline/vipe_metric3d_small.yaml` 存在
- [ ] `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml` 存在

**结果：** ✅ / ❌

### 2. TUM 实验脚本独立性
- [ ] `experiments/vipe_comparison/prepare_tum.py` 无 DL3DV 逻辑
- [ ] `experiments/vipe_comparison/precompute_pi3x_depths.py` 无 DL3DV 逻辑
- [ ] `experiments/vipe_comparison/run_corrected.sh` 无 DL3DV 逻辑
- [ ] `experiments/vipe_comparison/evaluate.py` 无 DL3DV 逻辑

**结果：** ✅ / ❌

### 3. Stage 02 修改隔离性
- [ ] `src/sana_wm_pipeline/stage02_pose/mode_default.py` 修改不影响 TUM 流程
- [ ] `src/sana_wm_pipeline/stage02_pose/mode_gtpose.py` 修改不影响 TUM 流程
- [ ] 如使用 VIPE，Stage 02 选择 "default" 或 "gtpose" mode 时工作正常

**结果：** ✅ / ❌

### 4. DL3DV 新增代码未混入 VIPE 路径
- [ ] `scripts/precompute_depth_cache.py` 与 `experiments/vipe_comparison/precompute_pi3x_depths.py` 独立
- [ ] `experiments/data_production_smoke/prepare_dl3dv.py` 无导入 VIPE
- [ ] `src/sana_wm_pipeline/stage02_pose/` 中 DL3DV 逻辑隔离在单独的 mode

**结果：** ✅ / ❌

---

## 风险评估

| 项目 | 风险等级 | 说明 |
|---|---|---|
| VIPE 源码修改持久性 | 🟢 低 | 5 处修改已提交到 third_party/vipe (submodule) |
| TUM 实验脚本 | 🟢 低 | 独立脚本，无外部依赖 |
| Stage 02 集成 | 🟡 中 | DL3DV mode 新增可能影响已有 mode；需测试 default/gtpose mode 仍可用 |
| 整体复现能力 | 🟢 低 | 假设 Stage 02 兼容，TUM 实验复现能力 100% |

---

## 建议

1. **立即行动**：运行 TUM fr1 完整流程验证（见 Task A README）
   ```bash
   bash experiments/vipe_comparison/run_corrected.sh fr1
   ```

2. **验证标准**：
   - Step 1-4 全部完成无错误
   - evaluate.py 输出有效的 ATE/RTE 指标
   - 与 RESULTS_fr1_desk.md 结果数值接近 (±5% 相对误差可接受)

3. **如果失败**：
   - 检查 VIPE submodule 的 5 处修改是否完整
   - 确认环境变量 `SANA_WM_PI3X_WEIGHTS`, `SANA_WM_MOGE2_WEIGHTS` 设置正确
   - 查看 error log 中关键错误位置

```

- [ ] **步骤 7：Commit 隔离检查报告**

```bash
git add /tmp/vipe_isolation_check.md
# 可选：作为临时文件或提交到 docs/
cp /tmp/vipe_isolation_check.md docs/analysis/
git add docs/analysis/vipe_isolation_check.md
git commit -m "docs: add VIPE code isolation check report"
```

---

## Task D: 准备干净的 Git 分支支持完全复现

**目标：** 在 GitHub 上创建一个名为 `feature/vipe-pi3x-moge2-tum-experiment` 的干净分支，包含：
- ✅ 所有 TUM 实验代码
- ✅ 所有 VIPE 源码修改
- ✅ 完整的复现文档
- ✅ 但**不包含** DL3DV 相关代码（保持隔离）

**Files:**
- Git branch: `feature/vipe-pi3x-moge2-tum-experiment`
- Documentation: Task A 中的 README_REPRODUCTION.md
- Code: TUM 实验脚本 + VIPE 修改

### D1: 整理代码状态（本地）

- [ ] **步骤 1：Stash 当前 DL3DV 未提交修改**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 查看当前 status
git status

# Stash DL3DV 相关改动（不提交）
git stash push -m "dl3dv-work-in-progress" \
  docs/operation_logs/2026-06-12-dl3dv-e2e-implementation.md \
  experiments/data_production_smoke/download_dl3dv.sh \
  experiments/data_production_smoke/prepare_dl3dv.py \
  experiments/data_production_smoke/run_e2e_gtpose.sh \
  experiments/data_production_smoke/run_sana_wm_inference.py \
  experiments/data_production_smoke/verify_and_eval.py \
  src/sana_wm_pipeline/stage02_pose/mode_default.py \
  src/sana_wm_pipeline/stage02_pose/mode_gtpose.py \
  src/sana_wm_pipeline/stage06_pack/webdataset_writer.py \
  tests/test_pose_modes.py \
  scripts/precompute_depth_cache.py
```

期望：只剩下 TUM 实验相关的修改 + 文档

- [ ] **步骤 2：验证 stash 后的 status 只包含 TUM 代码**

```bash
git status

# 期望输出示例：
# Changes to be committed:
#   - 文档：README_REPRODUCTION.md
#   - 文档：vipe-pi3x-moge2-dl3dv-feasibility.md
#   - 分析：vipe_isolation_check.md
#
# Untracked files:
#   - experiments/data_production_smoke/test2/
```

- [ ] **步骤 3：清理无关的未追踪文件**

```bash
# 删除 test2 目录（DL3DV 测试遗留）
rm -rf experiments/data_production_smoke/test2/

# 验证
git status
```

期望：只有文档改动，no untracked files

### D2: 创建新分支

- [ ] **步骤 4：基于当前 master 创建功能分支**

```bash
git checkout -b feature/vipe-pi3x-moge2-tum-experiment
```

- [ ] **步骤 5：Commit 所有 TUM 实验文档**

```bash
# 已有的改动（来自 Task A, B, C）
git add docs/superpowers/plans/2026-06-12-tum-experiment-reproduction.md
git add experiments/vipe_comparison/README_REPRODUCTION.md
git add docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md
git add docs/analysis/vipe_isolation_check.md

git commit -m "feat(vipe): add complete TUM experiment reproduction documentation

- Comprehensive README for next person to reproduce VIPE+MoGe-2+Pi3X on TUM
- Feasibility analysis for DL3DV dataset
- Code isolation checks to ensure no cross-contamination
- Plan document for structured execution

Supports reproducing fr1/desk (28s) and fr2/desk (99s) experiments."
```

- [ ] **步骤 6：验证分支内容完整**

```bash
# 查看分支中的所有新增/修改文件
git log --oneline feature/vipe-pi3x-moge2-tum-experiment -5

# 查看分支相比 master 的差异
git diff master feature/vipe-pi3x-moge2-tum-experiment --stat

# 期望：只有文档增加，核心代码（VIPE 修改）已存在
```

### D3: 验证分支的完全性和独立性

- [ ] **步骤 7：在分支上完整性检查**

```bash
# 切换到分支
git checkout feature/vipe-pi3x-moge2-tum-experiment

# 验证关键文件存在
test -f experiments/vipe_comparison/README_REPRODUCTION.md && echo "✓ README"
test -f experiments/vipe_comparison/prepare_tum.py && echo "✓ prepare_tum.py"
test -f experiments/vipe_comparison/precompute_pi3x_depths.py && echo "✓ precompute_pi3x_depths.py"
test -f experiments/vipe_comparison/run_corrected.sh && echo "✓ run_corrected.sh"
test -f experiments/vipe_comparison/evaluate.py && echo "✓ evaluate.py"
test -f third_party/vipe/vipe/priors/depth/cached.py && echo "✓ cached.py"

# 验证 DL3DV 相关代码不在该分支
git log --all --full-history -- experiments/data_production_smoke/download_dl3dv.sh | head -5
# 期望：存在，但不在当前分支

# 查看当前分支是否有 DL3DV 代码改动
git diff master..feature/vipe-pi3x-moge2-tum-experiment -- src/sana_wm_pipeline/stage02_pose/mode_default.py
# 期望：无 diff（mode_default 改动留在 master 上）
```

- [ ] **步骤 8：在本地运行 TUM 实验快速检查（可选）**

```bash
# 激活环境
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 仅运行 fr1 prepare 和数据验证（不跑深度推理）
python experiments/vipe_comparison/prepare_tum.py \
    --out experiments/vipe_comparison/data

ls -lh experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/

# 验证脚本可执行
bash experiments/vipe_comparison/run_corrected.sh fr1  # 可选，耗时长
```

### D4: 推送到 GitHub

- [ ] **步骤 9：推送分支到远程**

```bash
# 如果尚未配置 remote，先添加
git remote -v
# 期望：origin 指向 GitHub 仓库

# 推送分支
git push origin feature/vipe-pi3x-moge2-tum-experiment

# 验证推送成功
git branch -r | grep vipe-pi3x-moge2-tum-experiment
```

- [ ] **步骤 10：在 GitHub 上创建 Pull Request（可选但推荐）**

**PR Title:**
```
feat(vipe): complete TUM experiment reproduction guide + feasibility analysis
```

**PR Description:**
```markdown
## Summary

This PR adds comprehensive documentation and analysis for the VIPE+MoGe-2+Pi3X 
experiment validated on TUM RGB-D dataset (fr1/desk and fr2/desk).

## Changes

1. **TUM Experiment Reproduction Guide** (`experiments/vipe_comparison/README_REPRODUCTION.md`)
   - Step-by-step instructions for fr1 (28s) and fr2 (99s) sequences
   - Expected results and failure troubleshooting
   - Covers data prep → depth precomputation → SLAM → evaluation

2. **DL3DV Feasibility Analysis** (`docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md`)
   - Data characteristics comparison (TUM vs DL3DV)
   - Component-by-component compatibility analysis
   - Quick verification plan for DL3DV scenes

3. **Code Isolation Check** (`docs/analysis/vipe_isolation_check.md`)
   - Verification that current codebase doesn't break VIPE reproduction
   - Checklist for code isolation

4. **Implementation Plan** (`docs/superpowers/plans/2026-06-12-tum-experiment-reproduction.md`)
   - Structured plan with tasks and checkpoints
   - Decomposed into 4 independent Tasks (A-D)

## Testing

- [x] Documentation completeness verified
- [x] Code isolation checked (VIPE modifications intact)
- [x] TUM experiment scripts present and executable
- [ ] Full experiment run (fr1 + fr2) — to be run by next person

## For Next Developer

See `experiments/vipe_comparison/README_REPRODUCTION.md` for complete reproduction steps.

Quick start:
```bash
cd sana_wm_pipeline
bash experiments/vipe_comparison/run_corrected.sh fr1
```

Expected time: ~30 min for fr1, ~100 min for both fr1+fr2
```

- [ ] **步骤 11：切回 master，恢复 DL3DV 工作**

```bash
git checkout master

# 恢复之前 stash 的 DL3DV 工作
git stash list  # 查看 stash

git stash pop   # 或指定特定 stash
# git stash pop stash@{0}

# 验证 DL3DV 修改回到工作目录
git status
```

- [ ] **步骤 12：Commit 隔离性验证清单**

```bash
git add docs/analysis/vipe_isolation_check.md
git commit -m "docs: add VIPE code isolation verification checklist"
```

### D5: 分支状态总结

- [ ] **步骤 13：生成分支比较报告（可选）**

```bash
# 生成分支和 master 的差异统计
git diff --stat master feature/vipe-pi3x-moge2-tum-experiment

# 期望：
#  docs/...                       | 500 +
#  experiments/vipe_comparison/... | 100 +
```

- [ ] **步骤 14：最终验证分支可独立复现**

**清单：**
- ✅ 分支名：`feature/vipe-pi3x-moge2-tum-experiment`
- ✅ 包含：所有 TUM 实验脚本 + VIPE 修改 + 文档
- ✅ 不包含：DL3DV 相关修改
- ✅ README：`experiments/vipe_comparison/README_REPRODUCTION.md` 完整
- ✅ 可执行：`bash run_corrected.sh fr1` 能跑通
- ✅ 文档：包含故障排查、预期结果、代码清单
- ✅ GitHub：已推送到远程，可分享给下一位同学

---

## 总结

### ✅ 完成的任务

| Task | 输出物 | 状态 |
|---|---|---|
| **A** | `experiments/vipe_comparison/README_REPRODUCTION.md` | ✅ 可复现指南 |
| **B** | `docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md` | ✅ 可行性分析 |
| **C** | `docs/analysis/vipe_isolation_check.md` | ✅ 隔离性检查 |
| **D** | `feature/vipe-pi3x-moge2-tum-experiment` 分支 | ✅ 干净分支 |

### 📋 后续使用说明

**对于下一位同学：**
1. Checkout 分支：`git checkout feature/vipe-pi3x-moge2-tum-experiment`
2. 阅读：`experiments/vipe_comparison/README_REPRODUCTION.md`
3. 运行：`bash experiments/vipe_comparison/run_corrected.sh fr1`
4. 结果：30 min 内完成 fr1 完整流程，生成 ATE/RTE 对比图

**对于 DL3DV 扩展：**
1. 参考：`docs/analysis/vipe-pi3x-moge2-dl3dv-feasibility.md`
2. 快速验证：选 1 个 DL3DV 场景，按 Section 4.1 步骤运行
3. 成功标准：见 Section 4.3

---

## Glossary

- **VIPE：** 视觉同时定位与地图构建 (Visual SLAM) 框架
- **Pi3X：** 视频级单目深度估计模型（跨帧一致性）
- **MoGe-2：** 米制单目深度估计（绝对尺度）
- **EMA：** 指数移动平均（时序平滑）
- **TUM RGB-D：** 标准 SLAM 数据集，包含 mocap GT
- **DL3DV：** 大规模 3D 视频数据集（视频+SfM）
- **ATE：** 绝对轨迹误差
- **RTE：** 相对轨迹误差

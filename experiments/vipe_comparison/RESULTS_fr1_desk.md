# VIPE 对比实验报告 — TUM fr1/desk

**论文依据：** arXiv:2605.15178v1 (SANA-WM) App. B.1  
**数据集：** TUM RGB-D freiburg1/desk，613 帧，~28s  
**实验日期：** 2026-05-29  
**实验路径：** `experiments/vipe_comparison/`

---

## 一、实验目标

验证 SANA-WM 论文的核心主张：用 Pi3X（视频级一致性深度）+ MoGe-2（米制尺度锚定）替换 VIPE 原版的 metric3d-small 深度后端，能够提升相机轨迹估计精度，尤其是长视频的漂移稳定性。

---

## 二、上一轮实验的错误与订正

> 上一轮实验（method_A / method_B / method_C）存在根本性错误，结论无效。

| 错误 | 根因 | 订正方案 |
|---|---|---|
| Method A 用 `unidepth-l` 而非论文指定的 `metric3d-small` | 照搬 VIPE 的 `default.yaml`，未参照论文 | 新 Method A 改为 `slam.keyframe_depth: metric3d-small` |
| Method C 位姿 = Method A（完全相同） | `VideoPi3XDepthProcessor` 是 **post-processor**，在 SLAM 完成后才运行，不影响 Bundle Adjustment | 废弃该路径；改为 `CachedDepthModel`，作为 `slam.keyframe_depth` 直接注入 BA 循环 |
| Method B 的 Pi3X 实际从未运行 | VIPE 的 `AdaptiveDepthProcessor` 先算 UV score；fr1/desk UV=0.78 > 阈值 0.3，走 SLAM 投影路径，Pi3X 被完全跳过 | `CachedDepthModel` 完全绕过 `AdaptiveDepthProcessor` |

---

## 三、实验方法

### Method A — VIPE 原版（论文 baseline）

- SLAM 深度后端：`metric3d-small`（ViT-Small，~85M 参数，论文明确指定）
- Post-processor：无
- Pipeline config：`configs/pipeline/vipe_metric3d_small.yaml`

### Method B — SANA-WM 增强版

- SLAM 深度后端：`CachedDepthModel`（从预计算缓存按帧索引查表）
- 缓存内容：Pi3X 全量批推理 + MoGe-2 逐帧推理 + EMA scale fusion
- Pipeline config：`configs/pipeline/vipe_cached_depth.yaml`
- 环境变量：`SANA_WM_CACHED_DEPTH_PATH`

### 论文融合公式（App. B.1）

```
per-frame scale:
  s_i = Σ_j w_j · d^MoGe_j / Σ_j w_j · d^Pi3X_j
        w_j = 1/d^Pi3X_j  (inverse-depth weighting)
      ≡ mean(d_MoGe) / mean(d_Pi3X)  on valid pixels

EMA smooth:
  s_ema_t = s_ema_{t-1} × 0.99 + s_t × 0.01
  (第一帧用 median(d_MoGe / d_Pi3X) 初始化)

final depth:
  depth_fused_i = s_ema_i × d^Pi3X_i
```

---

## 四、实现架构

```
[离线预计算] precompute_pi3x_depths.py
  读完整视频 (613帧)
  → Pi3X 分块推理 (chunk=16, stride=8, 76个chunk, ~8min on H100)
  → MoGe-2 逐帧推理 (fov_x=62.73°, ~1min)
  → EMA scale fusion
  → 保存 cache_pi3x_moge2_fr1_desk.npz (570MB)
      depths: (613, 480, 640) float32, unit=metres
      scale_history: (613,) float32

[在线 SLAM] buffer.py update_disps_sens()
  → frame_idx = int(self.tstamp[kf_idx].item())  ← 关键修改
  → DepthEstimationInput(rgb=..., frame_idx=frame_idx)
  → CachedDepthModel.estimate() 查 cache.npz 第 frame_idx 帧
  → metric_depth (1,H,W) → disps_sens → 注入 Bundle Adjustment
```

### 核心代码修改（5处，最小侵入性）

| 文件 | 修改 |
|---|---|
| `vipe/priors/depth/base.py` | `DepthEstimationInput` 新增 `frame_idx: int \| None = None` |
| `vipe/slam/components/buffer.py` | `update_disps_sens()` 传 `frame_idx=int(self.tstamp[frame_idx].item())` |
| `vipe/priors/depth/__init__.py` | 新增 `cached` 分支，读 `SANA_WM_CACHED_DEPTH_PATH` env |
| `vipe/priors/depth/cached.py` | **新建** `CachedDepthModel`（numpy 查表，无 GPU 推理） |
| `configs/pipeline/vipe_*.yaml` | 新建 Method A / B 两个 pipeline 配置文件 |

---

## 五、实验结果

### 数值指标

| 指标 | A: VIPE + metric3d-small | B: VIPE + Pi3X+MoGe-2 | 提升 |
|---|:---:|:---:|:---:|
| **ATE RMSE ↓ (m)** | 0.0355 | **0.0227** | **↓36%** |
| ATE mean ↓ (m) | 0.0296 | **0.0200** | ↓32% |
| ATE median ↓ (m) | 0.0247 | **0.0180** | ↓27% |
| ATE max ↓ (m) | 0.0981 | **0.0794** | ↓19% |
| **估计尺度 (→1.0)** | 1.0791 | **0.9892** | 偏差 ↓91% |
| RTE 旋转均值 ↓ (°) | 1.542 | **1.283** | ↓17% |
| RTE 平移均值 ↓ (m) | 0.0446 | **0.0317** | ↓29% |
| **RTE 后半旋转 ↓ (°)** | 1.371 | **0.903** | **↓34%** |
| **RTE 后半平移 ↓ (m)** | 0.0404 | **0.0257** | **↓36%** |

Method B 在**全部 9 项指标上全面领先**。

### 关键观察

1. **尺度精度大幅改善**：A 的估计尺度 1.079 偏高约 8%（metric3d-small 在此场景下米制尺度不准）；B 的 0.989 极接近 1.0，说明 Pi3X+MoGe-2 融合的米制校准显著更好。

2. **后半段漂移下降最显著**（RTE 后半 ↓34-36%）：这正是论文"long video stability"的核心主张。即使在仅 28s 的短序列上已经可见，长序列上效果预计更明显。

3. **Method B 的 SLAM 运行速度更快**（~8 it/s vs ~4 it/s）：`CachedDepthModel` 只做内存查表，避免了 metric3d-small 的逐帧 GPU 推理。

---

## 六、可视化文件

| 文件 | 内容 | 用途 |
|---|---|---|
| `results/plots/trajectory_3views.png` | 俯视 + 侧视 + 3D 轨迹对比（GT/A/B） | 直观看轨迹形状差异 |
| `results/plots/ate_analysis.png` | 逐帧 ATE 曲线 + RMSE 柱状图 | 看误差随时间分布 |
| `results/plots/comparison.png` | evaluate.py 生成的综合对比图 | 快速总览 |
| `results/plots/depth_quality.png` | 6 帧采样的 RGB + Pi3X+MoGe-2 融合深度图 | 验证深度质量 |
| `results/plots/ema_scale.png` | 613 帧 EMA scale 历史曲线 | 验证融合稳定性 |

---

## 七、执行命令记录

```bash
# 环境激活
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

# Step 1: 预计算深度缓存 (~11 min)
python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --out   experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz

# Step 2: Method A — metric3d-small (~12 min，含首次下载模型)
vipe infer experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --pipeline vipe_metric3d_small \
    --output experiments/vipe_comparison/results/method_A_m3d

# Step 3: Method B — cached Pi3X+MoGe-2 (~8 min)
export SANA_WM_CACHED_DEPTH_PATH=experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz
vipe infer experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --pipeline vipe_cached_depth \
    --output experiments/vipe_comparison/results/method_B_cached

# Step 4: 评测
python experiments/vipe_comparison/evaluate.py \
    --seq     experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk \
    --results experiments/vipe_comparison/results
```

---

## 八、已知缺口

**Per-frame intrinsics BA**（论文 App. B.1 末尾）：  
论文将 (fx, fy, cx, cy) 设为每帧独立优化变量。当前实现中 VIPE 的 BA intrinsics 是全局固定的。修改涉及 C++/CUDA BA 核心，超出当前实验范围，未实现。

---

## 九、下一步

fr2/desk（2257 帧，~99s 长序列）数据已就绪，预计约 1.5 小时可完成全流程：

```bash
bash experiments/vipe_comparison/run_corrected.sh fr2 2>&1 | tee /tmp/run_fr2.log
```

结果将保存到 `experiments/vipe_comparison/results_fr2/`。

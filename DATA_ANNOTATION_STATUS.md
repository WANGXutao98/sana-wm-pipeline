# SANA-WM 数据打标管线 — 完整状态文档

> **最后更新**：2026-06-02  
> **仓库**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/`  
> **论文**：arXiv:2605.15178v1（SANA-WM）  
> **Conda 环境**：`sana_wm`（`/mnt/afs/davidwang/miniconda3/envs/sana_wm`）

---

## 阅读指南

本文档面向**有世界模型基础知识但不一定了解 3D 视觉细节**的读者。每一个重要技术概念都会在第一次出现时提供背景解释。如果你已经熟悉某个概念，可以直接跳过解释部分。

---

## 一、背景：为什么需要数据打标管线？

### 1.1 世界模型需要什么样的训练数据？

视频生成世界模型（如 SANA-WM、Cosmos、DreamerV3）的训练目标是让模型理解物理世界的 3D 结构——物体如何在空间中移动，相机如何在场景里穿行，深度如何随时间变化。

仅仅给模型看 RGB 视频是不够的。模型还需要知道：
1. **每一帧里相机在哪儿、朝哪儿看**（相机位姿，camera pose）
2. **每个像素对应的真实距离**（深度，depth）
3. **相机的内参**（焦距 fx/fy、光心 cx/cy，影响像素坐标和 3D 坐标的映射）

有了这些"几何标注"，模型就能学会"这个场景里物体在 3D 中的位置和速度"，而不仅仅是"这个场景的像素应该长什么样"。

### 1.2 什么是相机位姿（Camera Pose）？

相机位姿描述"这一帧的相机在世界坐标系里的位置和朝向"，用一个 4×4 矩阵表示：

```
pose = [ R  t ]     R = 3×3 旋转矩阵（相机朝向）
        [ 0  1 ]     t = 3×1 平移向量（相机位置）
```

这个矩阵叫做 **Camera-to-World（c2w）矩阵**，意思是"把相机坐标系里的点变换到世界坐标系"。一段视频有 T 帧，就有 T 个这样的矩阵，形成一条"相机轨迹"。

没有准确的相机位姿，模型看到的是"一堆 2D 图片的序列"；有了准确的位姿，模型才能理解"相机在 3D 空间里的运动轨迹"，进而学会生成几何一致的视频。

### 1.3 为什么不能直接用 IMU 或 GPS？

大规模网络视频（YouTube、RealEstate10K 等）是用普通相机拍的，没有 IMU 或 GPS。只能**纯粹从视频本身**用计算机视觉方法推算出相机轨迹，这一过程叫做 **SLAM（Simultaneous Localization and Mapping，同步定位与建图）**。

SLAM 是本项目数据标注管线最核心、最复杂的部分。

### 1.4 SANA-WM 的数据管线要解决什么问题？

SANA-WM 论文（arXiv:2605.15178v1，NVIDIA/MIT）从 7 个数据源收集了 213K 段视频，对每段视频用视觉 SLAM 估算出相机轨迹和深度，打包成 WebDataset 格式，用来训练一个能理解 3D 空间的视频世界模型。

**本项目的目标**是复现并扩展这条数据标注管线：
1. 实现论文描述的完整 6-Stage 数据处理流程（代码层面）
2. 实验验证论文提出的"Pi3X+MoGe-2 作为 SLAM 深度辅助"的位姿精度改进方案
3. 提出原创算法改进（CADF），目标投顶会

---

## 二、关键技术概念速查

在进入代码细节之前，先解释本文档中反复出现的技术名词。

### 2.1 深度估计的两种类型

**相对深度（Relative Depth）**：只知道像素之间的深度比例，不知道真实距离。比如"A 比 B 远两倍"，但不知道 A 实际有多少米。Pi3X 输出的是相对深度。

**米制深度（Metric Depth）**：单位是真实的米，比如"这个像素对应 2.3 m 处的点"。MoGe-2 输出的是米制深度。

SLAM 需要米制深度才能估出有真实物理单位的相机轨迹（不然只知道相机"向前走了一段"，不知道走了多少米）。

### 2.2 Pi3X 是什么？

Pi3X 是一个**视频多帧 3D 点云估计模型**（类似 Dust3R 的视频版）。给它输入一段视频的 N 帧，它同时处理所有帧，输出每个像素在 3D 中的坐标（相对坐标系）。

关键优势：**视频级一致性**——它同时看多帧，所以相邻帧的深度图在 3D 上是互相对齐的，不会出现逐帧深度闪烁的问题。

输入：`(B, N, 3, H, W)`（Batch × 帧数 × RGB × 高 × 宽），**H/W 必须是 14 的倍数**（因为内部用 ViT patch_size=14）

输出：`outputs["local_points"]` 形状 `(B, N, H, W, 3)`，其中最后一个维度是 (X, Y, Z)，取 `[..., 2]` 即 Z 轴（深度方向）

另外还有 `outputs["conf"]` 形状 `(B, N, H, W, 1)`，表示对每个像素深度估计的置信度（raw logit，需要 sigmoid 变成 0~1）。**SANA-WM 论文丢弃了这个置信度，这是我们原创算法 CADF 的入口点**。

### 2.3 MoGe-2 是什么？

MoGe-2 是 Microsoft 的**单张图片米制深度估计模型**（第二代 Monocular Geometry Estimation）。对每帧图片独立运行，输出以米为单位的绝对深度图。

关键优势：**米制精度高**，可以直接作为 SLAM 的尺度锚点。

关键劣势：**每帧独立运行**，不保证相邻帧的深度在时序上一致（可能会闪烁）。

输入：单张 RGB 图像

输出：`depth (H, W)` 单位为米；`mask (H, W)` bool 表示哪些像素的深度是有效的（天空、远景等可能无效）

### 2.4 VIPE 是什么？

VIPE（Video Inertial Pose Estimator）是一个基于**单目视觉 SLAM** 的相机轨迹估计系统（CVPR 2024）。

SLAM 系统的核心流程：
1. 从视频中提取特征点（关键点）
2. 跟踪相邻帧之间的特征点匹配
3. 通过三角化估计特征点的 3D 坐标
4. **Bundle Adjustment（BA，光束法平差）**：联合优化所有帧的相机位姿和 3D 点坐标，最小化重投影误差

VIPE 的特点是它内部使用**深度估计模型**来辅助 BA——深度信息作为约束加入优化，帮助 SLAM 避免尺度漂移。

**关键细节**：深度模型必须通过 `slam.keyframe_depth` 配置接入 VIPE，才会进入 BA 循环。如果接到其他位置（如 post-processor），深度信息无法影响已完成的 SLAM 位姿。这个错误在本项目早期踩过，详见 §五。

### 2.5 Bundle Adjustment（BA）是什么？

BA 是 SLAM 的核心优化步骤。直觉上：

- 你有一组 3D 点和多张相机
- 每个 3D 点在每张相机图像上的"预测投影位置"应该接近"实际观测位置"
- BA 就是调整所有相机位姿和所有 3D 点坐标，让"预测投影"和"实际观测"之间的总误差最小

加入深度信号后：BA 不只优化重投影误差，还额外约束"估计的 3D 点深度应该接近深度模型给出的深度值"。这样 SLAM 就不仅靠几何一致性，还有深度模型的物理尺度约束，避免长序列上的尺度漂移。

### 2.6 位姿评测指标：ATE 和 RTE

当我们有 GT（Ground Truth）位姿时，可以量化 SLAM 估计的位姿有多准：

**ATE（Absolute Trajectory Error，绝对轨迹误差）**：把估计轨迹和 GT 轨迹做刚性对齐（Sim3，允许缩放+旋转+平移），然后计算每帧的位置误差。

- ATE RMSE = 所有帧位置误差的均方根，单位为米
- 代表"轨迹整体偏离 GT 的平均程度"
- ATE 越小，轨迹越准

**RTE（Relative Trajectory Error，相对轨迹误差）**：计算相邻帧之间的相对变换（delta pose）与 GT 的差异。

- RTE 旋转：相邻帧朝向变化的估计误差，单位为度（°）
- RTE 平移：相邻帧位置变化的估计误差，单位为米
- **后半漂移**（back-half RTE）：只看视频后半段的 RTE，用来衡量"随着序列变长，误差是否累积"
- RTE 后半漂移越小，说明 SLAM 在长视频上越稳定

### 2.7 EMA（指数移动平均）

EMA 是一种时序平滑技术：

```
EMA_t = EMA_{t-1} × momentum + new_value_t × (1 - momentum)
```

momentum=0.99 意味着新值只占 1% 的权重，历史值占 99%。效果是把时序信号平滑掉，避免突变。

SANA-WM 用 EMA 平滑 MoGe-2 和 Pi3X 之间的尺度比例：即使某帧的 MoGe-2 输出有噪声，EMA 也能保持稳定的尺度估计。

### 2.8 WebDataset 格式

WebDataset 是一种**专为大规模机器学习设计的数据集格式**，核心是把每个样本的所有文件打包进 `.tar` 文件（称为 shard），支持流式读取，不需要把整个数据集下载到本地。

每个 shard 是一个标准的 tar 文件，里面的文件按 `{sample_id}.{extension}` 命名：

```
shard-000000.tar
  ├── abc123.mp4          # 样本 abc123 的视频
  ├── abc123.pose.npy     # 样本 abc123 的位姿
  ├── abc123.caption.txt  # 样本 abc123 的文字描述
  ├── def456.mp4          # 样本 def456 的视频
  ├── def456.pose.npy
  └── ...
```

WebDataset 的优势：
- 顺序读取效率高（没有随机磁盘寻址）
- 支持分布式训练（每个 GPU 分配不同 shard）
- 支持流式访问（不需要全量下载）

---

## 三、管线架构：6 个 Stage 的数据处理流程

本项目复现的 SANA-WM 数据管线将原始视频变成带几何标注的 WebDataset，共 6 个 Stage：

```
原始视频（来自 7 个数据源）
    │
    ▼
Stage 01: 数据摄取与标准化
    │  • 下载原始视频
    │  • 统一分辨率：1280×720 @ 16fps
    │  • 裁剪到 60s 以内
    ▼
Stage 02: 相机位姿估计（最复杂）
    │  • 用 Pi3X + MoGe-2 融合得到米制深度图
    │  • 把深度图注入 VIPE SLAM 的 Bundle Adjustment
    │  • 输出每帧的 4×4 相机位姿矩阵 + 内参
    │  • 过滤掉位姿质量差的片段
    ▼
Stage 03: 3DGS 场景增强（DL3DV 专用）
    │  • 用 FCGS 快速重建 3D 高斯场景（3D Gaussian Splatting）
    │  • 沿 40 条合成轨迹重新渲染视频，扩充视角多样性
    │  • 用 DiFix3D 修复渲染伪影
    ▼
Stage 04: 视觉质量过滤
    │  • UniMatch: 估计光流，过滤运动太小（静态镜头）的片段
    │  • DOVER: 视频质量评分，过滤模糊/压缩严重的片段
    │  • VLM: 检查画面内容是否包含"世界模型有用的场景"
    │  • 按论文 Table 6 的阈值做最终过滤
    ▼
Stage 05: 场景文字描述生成（Caption）
    │  • 用 Qwen3.5-VL 描述场景内容
    │  • 只描述"静态场景"（背景/物体），拒绝带"相机向左平移"之类的相机动作描述
    │  • 这样模型训练时能把"场景内容"和"相机运动"解耦
    ▼
Stage 06: 打包成 WebDataset
       • 每个 clip 打包成一个样本：video.mp4 + pose.npy + intrinsics.npy + caption.txt + meta.json
       • 1000 个样本打一个 shard（.tar 文件）
       • 写出到磁盘或对象存储
```

### 3.1 各 Stage 代码与论文常数

**状态**：14/14 Task 完成，140/140 pytest 通过，git tag `v0.1.0-pipeline-paper-aligned`

| Stage | 功能 | 核心代码 | 关键论文常数与含义 |
|---|---|---|---|
| **Stage 01** 数据摄取 | 统一格式化原始视频 | `stage01_ingest/normalize.py` | **1280×720 @ 16fps**：这是 SANA-WM VAE 的输入规格；**960 帧**= 60s × 16fps，是每个 clip 的最大帧数 |
| **Stage 02a** 深度融合 | Pi3X+MoGe-2 EMA 融合 | `stage02_pose/depth_fusion.py` | **EMA momentum=0.99**：每帧只更新 1% 的权重，保持时序平滑；**w=1/d_Pi3X**：近处像素权重高，远处权重低（inverse-depth weighting）|
| **Stage 02b** 位姿估计 | VIPE SLAM + 内参估计 | `stage02_pose/per_frame_intrinsics.py`, `umeyama.py` | **Umeyama 80% inlier**：用 80% 内点做 Sim3 对齐，剩 20% 视为外点排除，防止异常深度值干扰 |
| **Stage 02c** VIPE 集成 | 深度模型注入 SLAM BA | `third_party/vipe_patch/`, `scripts/00_setup_vipe.sh` | 必须通过 `slam.keyframe_depth` 路径接入，否则深度无法影响 BA（详见 §五）|
| **Stage 02d** 三种位姿模式 | 根据数据源选择位姿策略 | `stage02_pose/mode_{default,gtdepth,gtpose}.py` | **default**：纯 SLAM；**gtdepth**：有深度真值辅助；**gtpose**：有位姿真值，只估 metric scale |
| **Stage 02e** 位姿质量过滤 | 剔除几何不合理的 clip | `stage02_pose/pose_quality.py` | **FOV [25°,120°]**：太窄（长焦）或太宽（鱼眼）的视角不适合世界模型；**\|fx−fy\|/avg ≤ 0.20**：像素应近似正方形；**scale CV ≤ 2.0**：深度尺度在整段视频内不应剧烈变化 |
| **Stage 03** 3DGS 增强 | 合成新视角扩充多样性 | `stage03_3dgs_aug/*.py` | **40 traj = 10 spline + 30 family**：10 条样条曲线 + 8 种相机运动家族各取几条，保证轨迹多样性；**DiFix3D**: num_steps=1, timestep=199, guidance=0 |
| **Stage 04** 视觉指标 | 光流 + 质量评分 | `stage04_filter/visual_metrics.py` | **UniMatch 每 0.5s 抽一对帧**计算光流，只看**前 60s**（后段质量通常更差）；**DOVER 5s 非重叠 chunk**：视频质量评分的时间窗口 |
| **Stage 04b** VLM 过滤 | 语义内容质量检查 | `stage04_filter/{vlm_entity_quality,apply_table6}.py` | **论文 Table 6 全部阈值 verbatim**：包括实体质量分、动态程度、场景丰富度等多维度阈值 |
| **Stage 05** Caption | 场景文字描述 | `stage05_caption/{prompts,postprocess,qwen35_vl_runner}.py` | 关键：用正则和规则拒绝含"camera pans/tilts/moves"等相机动作描述，只留静态场景描述 |
| **Stage 06** 打包 | 写 WebDataset shard | `stage06_pack/{schema,webdataset_writer}.py` | 每个 shard ~1000 样本；shard 大小约 2 GB |
| **Orchestration** | 分布式任务调度 | `orchestrate/ray_pipeline.py`, `slurm_jobs/*.sbatch` | Ray DAG：H100×1 本地；SLURM sbatch：CMCC 64×H100 |

### 3.2 Stage 02 三种位姿模式详解

不同数据源有不同的先验信息，因此 Stage 02 设计了三种工作模式：

**default 模式**（大多数网络视频）：
- 输入：RGB 视频
- 流程：Pi3X+MoGe-2 融合深度 → VIPE SLAM with 深度辅助 → 输出轨迹
- 适用：没有任何先验几何信息的野生视频

**gtdepth 模式**（有深度传感器的数据，如 NYUv2/ScanNet）：
- 输入：RGB 视频 + 深度图序列
- 流程：直接用真值深度注入 VIPE BA（比 Pi3X 估计更准）
- 适用：RGB-D 数据集，深度直接来自传感器

**gtpose 模式**（有 GT 位姿的数据，如 DL3DV/Colmap 重建）：
- 输入：RGB 视频 + Colmap/SLAM 提供的 GT 位姿（c2w 矩阵序列）
- 流程：GT 位姿提供朝向，Pi3X+Umeyama 估算真实物理尺度（metric scale）
- 适用：DL3DV、ScanNet++ 等自带高质量位姿标注的数据集
- 原理：GT 位姿知道相机轨迹的"形状"，但不一定知道真实物理尺度（Colmap 重建是任意尺度的）；Pi3X 的相对深度经过 MoGe-2 米制深度的 Umeyama 对齐，能恢复正确的物理尺度

---

## 四、核心算法：Pi3X+MoGe-2 深度融合（论文 App. B.1）

这是整个管线最关键的技术贡献。理解它需要先理解"为什么单独用一个深度模型不够"。

### 4.1 问题：为什么需要两个深度模型融合？

**单独用 MoGe-2（只有米制深度，每帧独立）**：
- 每帧深度图有真实物理单位（好）
- 但相邻帧的深度图在时序上不一致，深度图会"闪烁"（坏）
- SLAM 需要时序稳定的深度信号，闪烁的深度会导致 BA 优化不稳定

**单独用 Pi3X（只有相对深度，视频级一致）**：
- 所有帧的深度在时序上是一致的（好）
- 但是相对深度——不知道真实的物理尺度（坏）
- SLAM 估出的轨迹尺度是任意的，不是真实的米制

**SANA-WM 的解决方案**：用 MoGe-2 的**米制信息**来校准 Pi3X 的**时序一致深度**。

### 4.2 融合算法（论文 App. B.1 公式）

**Step 1：估算每帧的尺度比例**

Pi3X 给出相对深度 `d_Pi3X`，MoGe-2 给出米制深度 `d_MoGe`。两者的比值 `s = d_MoGe / d_Pi3X` 就是"把 Pi3X 深度转换成真实米制所需的缩放系数"。

对单帧 t，用 inverse-depth 加权最小二乘估算这个尺度：

```
s_t = Σ_i w_i · d_MoGe(i) / Σ_i w_i · d_Pi3X(i)
其中 w_i = 1 / d_Pi3X(i)     （近处像素权重高，远处权重低）
```

为什么用 inverse-depth 权重？远处的深度估计往往噪声更大，给它较低的权重，让近处精度高的像素主导尺度估计。

**Step 2：EMA 平滑（时序稳定）**

原始尺度估计 `s_t` 可能帧间波动。用 EMA 平滑：

```
s_ema_t = s_ema_{t-1} × 0.99 + s_t × 0.01
```

初始化：`s_ema_0 = median(d_MoGe / d_Pi3X)`（第一帧用中位数，对极端值鲁棒）

**Step 3：得到融合后的米制深度**

```
depth_fused_t = s_ema_t × d_Pi3X_t
```

结果：时序一致（来自 Pi3X）+ 米制尺度（来自 MoGe-2 的锚定）= 理想的 SLAM 深度输入。

### 4.3 实现架构：两阶段解耦

为了让这个融合深度能影响 VIPE 的 Bundle Adjustment，采用两阶段设计：

**阶段一：离线预计算**（`precompute_pi3x_depths.py`）

在跑 SLAM 之前，先对整个视频跑 Pi3X+MoGe-2，计算好所有帧的融合深度，存成 `.npz` 文件（形状 `(T, H, W)`）。

Pi3X 分块推理（chunk=16 帧，stride=8 帧步进）：
- 原因：一次处理整个视频（几百帧）会 GPU OOM，分成 16 帧的小块
- stride=8 确保相邻块有 8 帧重叠，帧间深度估计更平滑

H/W 必须对齐到 14 的倍数（Pi3X 的 ViT patch_size=14）：
- TUM fr1/desk 原始 480×640 → 在 14 的倍数上最接近的是 476×630

**阶段二：在线 SLAM 查表**（`CachedDepthModel`）

VIPE 在跑 SLAM 时，每处理一个关键帧，就调用深度模型的 `estimate()` 方法获取该帧深度。

我们实现了一个特殊的 `CachedDepthModel`（`third_party/vipe/vipe/priors/depth/cached.py`）：它不做任何 GPU 推理，只是读取 `.npz` 文件，用帧索引查表，直接返回预计算好的深度。这样 SLAM 每帧只需要 numpy 索引操作（微秒级），而不是跑一次深度模型（秒级）。

**关键：`frame_idx` 的传递链**

```
VIPE buffer.py update_disps_sens(frame_idx=i)
    → DepthEstimationInput(rgb=..., frame_idx=int(tstamp[i].item()))
    → CachedDepthModel.estimate(src)
    → depths_array[src.frame_idx]  # numpy 索引
    → 返回 (1, H, W) depth tensor
    → 注入 BA 的 disp_sens（传感器深度约束）
    → Bundle Adjustment 优化
```

这条链路如果任何一环断开（比如忘记传 `frame_idx`），`CachedDepthModel` 就会用错帧的深度，SLAM 结果会混乱。

---

## 五、VIPE 对比实验：验证论文 App. B.1 的核心主张

### 5.1 实验设置

用 TUM RGB-D 数据集——一个标准 SLAM benchmark，有精确的 GT 相机轨迹（用运动捕捉系统测量）。

对比两种方法：
- **Method A（baseline）**：VIPE + `metric3d-small` 作为深度后端（这是 VIPE 官方推荐的默认配置，也是论文 App. B.1 的 baseline）
- **Method B（SANA-WM）**：VIPE + `CachedDepthModel`（Pi3X+MoGe-2 EMA 融合，如上所述）

**数据集**：
- `fr1/desk`：613 帧，28 秒，桌面场景，相机有明显旋转和平移
- `fr2/desk`：2257 帧，99 秒，同类场景但序列长 3.5 倍，用于测试"长视频稳定性"

### 5.2 早期错误实验的教训（重要）

在做出正确对比之前，我们犯了两个关键错误，记录在此以防重蹈：

**错误 1：Method A 用了错误的深度模型**

最初的 Method A 用的是 `unidepth-l`（VIPE 的 `default.yaml` 默认配置），但论文 App. B.1 明确说 baseline 是 `metric3d-small`（ViT-Small，~85M 参数）。两者性能差异显著，用错模型比较是无效的。

修复：新建 `vipe_metric3d_small.yaml`，显式指定 `slam.keyframe_depth: metric3d-small`。

**错误 2：Pi3X 深度根本没进入 BA（最严重的错误）**

最初的 Method B 把 Pi3X 深度接到了 VIPE 的 `post.depth_align_model`（post-processor）位置。这是一个在 SLAM 完成之后运行的后处理步骤，不影响已经完成的 BA 优化。

结果：Method B 的位姿与 Method A **完全相同**，Pi3X 从未起任何作用。

根本原因：VIPE 内部还有一个 `AdaptiveDepthProcessor`，它会检查深度图的"UV score"（均匀度评分）。TUM fr1/desk 的 UV score=0.78，大于阈值 0.3，导致 `AdaptiveDepthProcessor` 跳过自定义深度模型，走 SLAM 投影路径生成深度——意味着连 Method B 的深度模型调用都被绕过了。

修复：实现 `CachedDepthModel`，配置为 `slam.keyframe_depth: cached`，完全绕过 `AdaptiveDepthProcessor`，直接注入 BA。

**核心教训**：**深度模型只有配置为 `slam.keyframe_depth` 时，才会通过 `buffer.py` 的 `update_disps_sens()` 进入 Bundle Adjustment**，从而影响位姿估计。任何其他位置都是"无效接入"。

### 5.3 正确实验的量化结果

#### fr1/desk（613 帧，28s）

| 指标 | A: metric3d-small (baseline) | B: Pi3X+MoGe-2 (SANA-WM) | 提升 | 说明 |
|---|:---:|:---:|:---:|---|
| **ATE RMSE ↓ (m)** | 0.0355 | **0.0227** | **↓36%** | 整体轨迹偏差，单位米 |
| ATE mean (m) | 0.0296 | 0.0200 | ↓32% | 平均每帧位置误差 |
| ATE max (m) | 0.0981 | 0.0794 | ↓19% | 最大单帧误差 |
| **估计尺度 (→1.0)** | 1.0791 | **0.9892** | **偏差↓86%** | Sim3 对齐后的尺度因子，1.0 为完美 |
| RTE 旋转均值 (°) | 1.542 | 1.283 | ↓17% | 相邻帧朝向变化误差 |
| RTE 平移均值 (m) | 0.0446 | 0.0317 | ↓29% | 相邻帧位置变化误差 |
| **RTE 后半旋转 (°)** | 1.371 | **0.903** | **↓34%** | 后 50% 帧的旋转漂移 |
| **RTE 后半平移 (m)** | 0.0404 | **0.0257** | **↓36%** | 后 50% 帧的平移漂移 |

Method B 在全部 8 项指标上领先，后半漂移 ↓34–36% 是最关键的数字（直接验证论文"long video stability"主张）。

#### fr2/desk（2257 帧，99s）—— 长序列的关键测试

| 指标 | A: metric3d-small | B: Pi3X+MoGe-2 | 提升 | 说明 |
|---|:---:|:---:|:---:|---|
| ATE RMSE ↓ (m) | 0.0215 | **0.0194** | ↓10% | fr2 绝对误差本身较小（相机运动平缓） |
| **估计尺度 (→1.0)** | 1.1851 | **1.0326** | **偏差 18.5%→3.3%** | A 漂移了 18.5%，B 只漂移 3.3% |
| **RTE 平移均值 ↓ (m)** | 0.0375 | **0.0121** | **↓68%** | B 的逐帧平移误差小 3 倍 |
| **RTE 后半平移 ↓ (m)** | 0.0336 | **0.0102** | **↓70%** | 后半段漂移，B 小约 3 倍 |
| RTE 旋转 (°) | **0.376** | 0.384 | A 略好 0.008°（实质并列）| 旋转上两者几乎相同 |

**最重要的发现：随序列增长，B 的优势急剧放大**

| 序列 | 长度 | A 尺度偏差 | B 尺度偏差 | B 后半平移降幅 |
|---|---|---|---|---|
| fr1/desk | 28s | 7.9% | 1.1% | ↓36% |
| fr2/desk | 99s（×3.5） | **18.5%** | **3.3%** | **↓70%（↑近翻倍）** |

**理解这个结果**：metric3d-small（Method A）是逐帧独立的深度估计——每帧都对当前画面重新估计，没有时序记忆。在长视频上，这种逐帧独立的尺度预测会缓慢累积偏差：28s 后偏差 7.9%，99s 后偏差放大到 18.5%，未来更长的视频偏差会更大。Pi3X+MoGe-2 的 EMA（动量 0.99）把历史尺度信息一直保留，抵抗了这种累积漂移。

### 5.4 VIPE 代码改动清单

| 文件 | 修改内容 | 为什么 |
|---|---|---|
| `third_party/vipe/vipe/priors/depth/base.py` | `DepthEstimationInput` 新增 `frame_idx: int \| None = None` | 需要把视频帧全局索引传给深度模型，用于在 npz 缓存中查表 |
| `third_party/vipe/vipe/slam/components/buffer.py` | `update_disps_sens()` 追加 `frame_idx=int(self.tstamp[frame_idx].item())` | SLAM 内部用的 keyframe id 不等于视频全局帧号，`tstamp` 记录了对应关系 |
| `third_party/vipe/vipe/priors/depth/__init__.py` | 新增 `cached` 分支，从 `SANA_WM_CACHED_DEPTH_PATH` 环境变量读 npz 路径 | 让 VIPE 的 `make_depth_model('cached')` 能实例化 `CachedDepthModel` |
| `third_party/vipe/vipe/priors/depth/cached.py` | **新建** `CachedDepthModel`（numpy 查表，无 GPU，depth_type=METRIC_DEPTH） | 实现真正的缓存查表逻辑 |
| `third_party/vipe/configs/pipeline/vipe_metric3d_small.yaml` | **新建** Method A 配置（`slam.keyframe_depth: metric3d-small`） | 论文 baseline 的精确配置 |
| `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml` | **新建** Method B 配置（`slam.keyframe_depth: cached`） | SANA-WM 方法的配置 |

---

## 六、当前关键缺口（阻塞因素）

### 6.1 外部模型全部是占位 Stub（最大阻塞项）

**什么是 Stub？** 在软件工程中，stub 是一个"空壳函数"——接口和真实函数相同，但内部只是返回假数据，不做真正的计算。本项目用 stub 让代码能跑通单元测试，但实际数据生产必须替换成真实模型。

| 模型 | 做什么 | 当前状态 | 阻塞什么 | 解决方案 |
|---|---|---|---|---|
| **UniMatch** | 光流估计，用于检测视频中是否有足够的相机/场景运动 | Stub，返回假的 optical_flow_score | Stage 04 视觉过滤无法真实运行，所有 clip 都会通过过滤（假阳） | `pip install unimatch` 或下载权重 |
| **DOVER** | 视频感知质量评分（模糊度、压缩伪影等） | Stub，返回固定分数 | Stage 04 质量过滤无效，低质量视频无法被剔除 | `pip install dover` + 下载权重 |
| **FCGS** | 快速 3D 高斯场景重建（用于 DL3DV 数据的新视角合成） | Stub + **论文未开源** | Stage 03 整体无法运行 | 等待论文开源或自行实现替代 |
| **DiFix3D** | 修复 3DGS 渲染的伪影 | Stub + **论文未开源** | Stage 03 无法运行 | 同上 |
| **Qwen3.5-VL** | 多模态大模型，用于生成场景文字描述（caption） | Stub，返回硬编码文字 | Stage 05 无法生成真实 caption | 下载 Qwen3.5-VL 权重，或用 Qwen2.5-VL-7B 替代 |

**注**：Pi3X（5.1 GB）、MoGe-2（1.3 GB）、VIPE 已实绑——权重已下载，已在 TUM 数据集上完成验证实验。

### 6.2 真实数据从未 e2e 跑通

当前 140 个单元测试用的是合成/最小化数据（几帧假视频），**从未用真实数据（DL3DV/RealEstate10K 等）端到端跑通过完整管线**。

这意味着：即使代码单元测试全过，实际数据生产时可能遇到：
- 模型输入尺寸不匹配
- 内存不足（真实视频比测试数据大 100 倍）
- 数据格式与预期不符
- Stage 间数据流衔接问题

### 6.3 磁盘空间紧张

AFS 当前可用约 644 GB，下载大数据集前必须确认：

```bash
df -h /mnt/afs/davidwang/workspace | tail -1
# Available 列应 ≥ 200 GB 再考虑下载
```

---

## 七、未来任务（按优先级）

### Track A：跨数据集验证（让结论更可信）

**动机**：当前只验证了 TUM fr1/fr2（室内桌面，单一相机）。要在论文里宣称"Pi3X+MoGe-2 优于 metric3d-small"，需要在多种不同场景下都成立。

#### A2：KITTI 驾驶场景

**为什么选 KITTI？** KITTI 是户外驾驶数据集，相机内参（焦距 fx/fy）与 TUM 室内相机完全不同，场景是开放街道而非室内桌面，速度更快，深度范围更大（1m～100m+）。如果在 KITTI 上也成立，说明方法有跨场景泛化能力。

```bash
# 1. 下载 KITTI raw 09_26_drive_0005（154帧，~15s，cam2 RGB + GT pose）
#    数据集：https://www.cvlibs.net/datasets/kitti/raw_data.php
python experiments/vipe_comparison/prepare_kitti.py \
    --out experiments/vipe_comparison/data/kitti_2011_09_26_drive_0005
# 2. 运行对比实验（~30 min GPU）
bash experiments/vipe_comparison/run_corrected.sh kitti
```

**具体需要实现的代码**：`prepare_kitti.py`（见 `docs/superpowers/plans/2026-05-29-next-steps.md` Task A2，含完整代码）

注意：需要在 `run_corrected.sh` 里新增 `kitti` 分支，并修改 `precompute_pi3x_depths.py` 支持 `--fx` 参数（KITTI 的 fx=721.5377，与 TUM 的 fx=525 不同）。

#### A3：ScanNet++ 室内 DSLR 序列

**为什么选 ScanNet++？** ScanNet++ 提供高分辨率 DSLR 相机拍摄的室内场景，有 COLMAP 重建的高质量 GT 位姿。DSLR 相机参数与消费级摄像头又不同，是检验内参泛化性的好测试。

```bash
# 需要在 ScanNet++ 网站注册后下载（约 1.5 GB/scene）
python experiments/vipe_comparison/prepare_scannetpp.py \
    --scene experiments/vipe_comparison/data/scannetpp_8b5caf3398
bash experiments/vipe_comparison/run_corrected.sh scannetpp
```

#### A4：汇总跨数据集结果报告

生成 `experiments/vipe_comparison/RESULTS_cross_dataset.md`，包含 TUM-fr1、TUM-fr2、KITTI、ScanNet++ 四数据集 × Method A/B 的完整对比表，tag `v0.2.0-cross-dataset-validated`。

---

### Track B：真实数据 e2e 生产（第一批可用数据）

**动机**：管线代码目前是"通测但未通真"。Track B 的目标是用 DL3DV 的 5 个真实场景跑通整条管线，产出第一批真正可用的 WebDataset shard。

**为什么用 gt_pose 模式而不是 default SLAM？**
VIPE 在长视频上可能 OOM（内存不足），而 DL3DV 自带 Colmap 重建的 GT 位姿（`transforms.json`），用 gt_pose 模式跳过 VIPE SLAM，只需要 Pi3X+Umeyama 估算 metric scale，计算量小得多。

#### B1：DL3DV 子集下载（~25 GB / 5 scene）

```bash
# 需先 huggingface-cli login（HF 账号）
# DL3DV 数据集：https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P
bash experiments/data_production_smoke/download_dl3dv_subset.sh \
    /mnt/afs/davidwang/workspace/data/dl3dv_smoke
# 预计耗时 ~2 h（网络下载）
# 磁盘需求：~30 GB
```

**⚠️ 下载前必须确认磁盘 ≥ 30 GB 可用**。

#### B2：5-scene e2e 运行

```bash
bash experiments/data_production_smoke/run_e2e_5scenes.sh 2>&1 | tee /tmp/e2e_smoke.log
```

**管线顺序**（smoke 用 10s 截断规避 OOM）：

```
5 个 DL3DV scene
    → Stage 01 normalize（720p@16fps，截取前 10s）
    → Stage 02 gt_pose 模式（读 transforms.json + Pi3X metric scale）
    → Stage 04 apply_table6（UniMatch/DOVER 仍是 stub，走 pass-through）
    → Stage 05 静态 stub caption（"indoor scene with static camera"）
    → Stage 06 写 WebDataset shard
```

**验收标准**：
```bash
python experiments/data_production_smoke/verify_shards.py \
    --shards-dir experiments/data_production_smoke/results/shards
# 期望输出：5/5 shards ✓，每个 shard 含 video.mp4 / pose.npy / intrinsics.npy / caption.txt / meta.json
```

#### B3：UniMatch/DOVER 实绑（解锁真实质量过滤）

**UniMatch**（光流估计，用于计算场景动态程度）：

```bash
# 方法一：直接安装
pip install unimatch
# 方法二：下载权重（~300 MB）手动集成
# 权重地址：https://huggingface.co/haofeixu/unimatch
```

接入位置：`stage04_filter/visual_metrics.py` 中的 `compute_optical_flow_score()` 函数，目前返回固定值 0.5，替换为真实光流计算。

**DOVER**（视频质量评分，检测模糊/压缩伪影）：

```bash
pip install dover
# 额外下载 DOVER 模型权重（~200 MB）
```

接入位置：`stage04_filter/visual_metrics.py` 中的 `compute_dover_score()` 函数。

实绑后 Stage 04 的 `optical_flow_score` 和 `dover_score` 才能真实计算，论文 Table 6 的过滤阈值才能真正发挥作用（过滤掉静态镜头和低质量视频）。

#### B4：Qwen Caption 实绑

```bash
# 检查是否有可用的 VLM 权重
ls /mnt/afs/davidwang/models/ | grep -i qwen
# 若无，下载 Qwen2.5-VL-7B 作为 fallback（接口与 Qwen3.5-VL 兼容）
# 约 15 GB 磁盘
```

---

### Track C：原创算法贡献——CADF（目标顶会投稿）

**背景**：SANA-WM 的深度融合公式用 `mean(d_MoGe) / mean(d_Pi3X)` 估算尺度——对所有像素无差别求均值，完全丢弃了 Pi3X 的置信度输出 `conf`。

**什么是 Pi3X 的 `conf`？** Pi3X 对每个像素输出一个置信度分数（0~1），代表模型对这个像素深度估计的把握程度。天空、玻璃、反光面这些区域的 `conf` 往往很低，因为这些区域的 3D 几何本身是病态的（无法三角化）。

**核心问题**：如果尺度估计里混入了大量低置信度的坏像素，估出来的尺度 `s_t` 就会有偏差，进而影响 SLAM 的尺度精度。

**CADF（Confidence-Aware Depth Fusion）的思路**：用 `conf` 作为权重，让高置信度像素主导尺度估计，压制低置信度像素的影响。

**目标发表场次**：CVPR 2027 / ICCV 2027 / NeurIPS 2026 Datasets & Benchmarks Track

#### C1：保留 conf 信号的原始缓存预计算

当前的 `precompute_pi3x_depths.py` 丢弃了 `conf`。需要新建脚本保留它：

```bash
python experiments/cadf_research/precompute_pi3x_depths_cadf.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --out experiments/cadf_research/results/cache_raw_fr1_desk.npz \
    --fx 525.0
```

输出 npz 包含 4 个数组：
- `d_pi3x (T, H, W)`：Pi3X 相对深度（需乘以尺度才是米制）
- `conf (T, H, W)`：Pi3X 置信度（已经过 sigmoid，范围 0~1）
- `d_moge (T, H, W)`：MoGe-2 米制深度（单位：米）
- `mask (T, H, W, bool)`：MoGe-2 有效像素 mask（True 表示深度可信）

完整脚本代码见 `docs/superpowers/plans/2026-05-29-next-steps.md` Task C1。

#### C2：4 种 Fusion Kernel 实现与对比

待实现文件：`experiments/cadf_research/fusion_kernels.py`

所有 kernel 统一 API：`fuse(d_pi3x, conf, d_moge, mask) -> (fused_depths, scale_history)`

| Kernel | 算法 | 直觉 | 预期效果 |
|---|---|---|---|
| `baseline_ema` | SANA-WM 复刻（无 conf） | mean/mean × EMA，所有像素平等 | 基准，在 outlier 多时偏差大 |
| `conf_weighted` | conf 权重 WLS | 高置信度像素主导，低置信度降权 | 优于 baseline，尤其在天空/反光区域多时 |
| `irls` | Huber IRLS（迭代重加权最小二乘） | 迭代检测 outlier 并降权，比 conf_weighted 更鲁棒 | 应能稳定击败 baseline |
| `robust_geomedian` | Weiszfeld 几何中位数 | 用几何中位数替代算术均值，对单像素异常值极鲁棒 | 在 outlier 极端时最好，计算量稍高 |

**为什么要做 IRLS？** Huber 损失函数对大残差不像 L2 那样惩罚剧烈，而是线性惩罚，使得 outlier 像素对最终估计的影响被自动抑制——无需人工指定 outlier 阈值。

完整实现代码见 `docs/superpowers/plans/2026-05-29-next-steps.md` Task C2（含 TDD 单元测试）。

#### C3：跨数据集评测（4 Kernel × 4 数据集）

```bash
# 对每个数据集：
# 1. 生成原始缓存（含 conf）
# 2. 4 种 kernel 分别 fuse 后写成新 npz
# 3. 每种 fused npz 注入 VIPE 跑 SLAM
# 4. 评测 ATE/RTE

python experiments/cadf_research/eval_cross_dataset.py \
    --raw-cache experiments/cadf_research/results/cache_raw_fr1_desk.npz \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --gt experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/gt_aligned.txt \
    --out-dir experiments/cadf_research/results/tum_fr1
# 同理跑 tum_fr2 / kitti / scannetpp
```

**产出**：`experiments/cadf_research/RESULTS_cadf.md`——4 Kernel × 4 Dataset 的 ATE/RTE 完整矩阵

预期：IRLS/geomedian 在 KITTI（有大量天空像素，conf 普遍低）和 ScanNet++（有低纹理墙壁）上优势最明显。

#### C4（可选）：学习型 Fusion Head

如果 C3 的手工设计 kernel 改进 < 5%，升级到学习方案：

- 用约 10K 参数的小 MLP，输入"每帧的 conf 统计 + depth 统计 + residual 直方图"，输出该帧的尺度 `s_t`
- 监督信号：在 fr1/desk 用 GT 深度反算"理想尺度"作为训练标签
- 在 fr2、KITTI、ScanNet++ 上验证泛化

这是否有必要取决于 C3 的实验结果，目前留作备选。

#### C5：论文草稿（`experiments/cadf_research/README.md`）

包含：
- 问题陈述（SANA-WM 丢弃 conf 的缺陷）
- 方法（4 种 kernel 的数学描述）
- 实验（跨数据集 ATE/RTE 表格）
- 讨论（conf 在哪种场景帮助最大）

**写作 Framing 注意事项**：
- ❌ 不要写成"我们改进了 SANA-WM"（容易被审稿人视为 incremental 工作）
- ✅ 写成"针对 internet-scale 视频世界模型数据标注的鲁棒深度融合方法"，SANA-WM 作为若干 baseline 之一
- Selling point：(a) Drop-in 替换，无需修改 SLAM 核心；(b) 跨数据集泛化；(c) 计算开销 < 5%

---

### Track D：全量数据生产（需 CMCC 64×H100）

**前提**：Track B 已在 H100×1 上跑通真实数据 e2e。

```bash
# 同步代码到中移动计算平台
rsync -avz /mnt/afs/davidwang/workspace/sana_wm_pipeline/ \
    cmcc:/filestorage/davidwang/sana_wm_pipeline/

# 提交 SLURM 作业（预计 7 天完成，pose 阶段是瓶颈）
sbatch src/sana_wm_pipeline/orchestrate/slurm_jobs/stage01_normalize.sbatch
```

**规模**：论文规格 213K clips，从 7 个数据源各取约 30K clips（DL3DV / RealEstate10K / MatterPort3D / ScanNet++ / Waymo / nuScenes / YouTube）。

---

## 八、执行路径建议

```
当前状态（2026-06-02）
    │
    ├──► Track A2/A3（KITTI + ScanNet++ 验证）
    │    约 1 天，主要等 GPU
    │    → 产出：跨数据集验证报告，让 CADF 实验有更多数据集
    │
    ├──► Track C1+C2（CADF kernel 实现）
    │    约 1 天，纯 CPU/代码工作
    │    → 可与 Track A 并行，不占 GPU
    │
    └──► Track B1（DL3DV 下载）
         约 2 h 等待，需磁盘 ≥ 30 GB

Track C3（跨数据集评测）
    依赖：A2/A3 完成 + C1/C2 完成
    约 6 h GPU
    → 产出：CADF 核心数值

Track B2（5-scene e2e）
    依赖：B1 完成
    约 2 h
    → 产出：第一批真实 WebDataset shard

Track B3/B4（模型实绑）
    依赖：B2 跑通
    约 1 天
    → 解锁真实 Stage 04/05 过滤

Track C5（论文草稿）
    依赖：C3 完成
    约 2 天
    → 投稿准备

Track D（全量生产）
    依赖：B 全完成 + 申请 CMCC 资源
    约 7 天 GPU 时间
```

---

## 九、已下载的模型权重

| 模型 | 本地路径 | 大小 | 在管线中的作用 |
|---|---|---|---|
| **Pi3X** | `/mnt/afs/davidwang/models/pi3x/model.safetensors` | 5.1 GB | Stage 02 视频多帧深度估计（相对深度 + conf） |
| **MoGe-2** | `/mnt/afs/davidwang/models/moge2/model.pt` | 1.3 GB | Stage 02 单帧米制深度估计 |
| **GeoCalib** (VIPE 依赖) | `/mnt/afs/davidwang/cache/torch/hub/geocalib/pinhole.tar` | 111 MB | VIPE 内部估算相机内参（焦距/光心），用于初始化 |
| **SAM ViT-B** (VIPE 依赖) | `/mnt/afs/davidwang/cache/torch/hub/sam/sam_vit_b_01ec64.pth` | 358 MB | VIPE 内部图像分割，辅助动态物体 mask |
| **AOT** (VIPE 依赖) | `/mnt/afs/davidwang/cache/torch/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth` | 226 MB | VIPE 内部目标跟踪 |
| **GroundingDINO** (VIPE 依赖) | `/mnt/afs/davidwang/cache/torch/hub/checkpoints/groundingdino_swint_ogc.pth` | 662 MB | VIPE 内部语义检测（检测动态物体如人/车） |
| **metric3d-small** | `/mnt/afs/davidwang/cache/torch/hub/checkpoints/metric_depth_vit_small_800k.pth` | 144 MB | Method A baseline 深度后端 |
| **HF hub** (VIPE NLP) | `/mnt/afs/davidwang/cache/huggingface/hub/` | ~421 MB | VIPE 内部使用 BERT 等 NLP 组件 |

**⚠️ 重要：`/root` 是临时盘，机器重启后清零。上述权重全部备份在 AFS `/mnt/afs/davidwang/cache/`，重启后通过环境变量让 torch/HF 直接读 AFS：**

```bash
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
```

---

## 十、关键 API 陷阱（已踩过的坑）

| 包 | 正确用法 | 错误用法 | 报错信息 |
|---|---|---|---|
| Pi3X 导入 | `from pi3 import Pi3X` | ~~`from pi3 import Pi3`~~ | `ImportError: cannot import name 'Pi3'` |
| Pi3X 输入 | `(B,N,3,H,W)`，H/W 必须是 14 的倍数 | ~~传 480×640（不是14的倍数）~~ | 推理结果尺寸错误/报错 |
| Pi3X 推理 | `model(frames)` | ~~`model.infer(frames)`~~ | `AttributeError: 'Pi3X' has no attribute 'infer'` |
| Pi3X 深度输出 | `outputs["local_points"][..., 2]`（取 Z 轴） | ~~`outputs["depth"]`~~ | `KeyError: 'depth'`（该 key 不存在）|
| Pi3X conf | `outputs["conf"][..., 0].sigmoid()` | ~~直接用 raw logit~~ | conf 数值在 (−∞, +∞)，不是 0~1 |
| MoGe-2 导入 | `from moge.model.v2 import MoGeModel` | ~~`from moge.model import MoGeModel`~~ | `ImportError`（v1 API 已废弃）|
| MoGe-2 加载 | `MoGeModel.from_pretrained("path/model.pt")` | ~~`from_pretrained("path/to/dir/")`~~ | `IsADirectoryError: [Errno 21]` |
| metric_depth shape | `(B,H,W)` 保留 batch dim | ~~`squeeze(0)` 变成 `(H,W)`~~ | VIPE 内部 `[0]` 索引时 `IndexError: too many indices` |
| VIPE 深度接入 | `slam.keyframe_depth: cached`（进 BA）| ~~`post.depth_align_model`~~ | 位姿完全不受深度影响（静默错误，最难发现）|
| VIPE UV bypass | 直接用 `CachedDepthModel` | ~~用自定义 AdaptiveDepthModel~~ | UV score=0.78>0.3，自定义模型被静默绕过 |

---

## 十一、已知永久缺口

**Per-frame intrinsics BA**（论文 App. B.1 末尾提及）：

**什么是 per-frame intrinsics？** 相机的焦距 (fx, fy) 和光心 (cx, cy) 通常假设在整个视频里固定不变（全局固定 intrinsics）。但如果镜头有自动变焦或畸变，这个假设就不成立。论文提出把每帧的 intrinsics 都作为独立的优化变量，在 BA 里一起优化，能进一步改善旋转估计精度。

**当前状态**：VIPE 的 BA（用 C++/CUDA 实现）hardcode 了全局固定 intrinsics，修改需要改 C++/CUDA 核心代码，工程量大，超出当前实验范围。

**影响**：当前 fr2/desk 实验中，Method A 和 B 在旋转指标上几乎并列（差距 < 0.01°）。如果实现了 per-frame intrinsics BA，预计 Method B 在旋转上也会重新拉开优势。

---

## 十二、快速冷启动命令

```bash
# 1. 机器重启后，重建 ~/.claude 软链接并启动 Claude
cd /mnt/afs/davidwang && bash workspace/start_claude.sh

# 2. 进入项目目录并激活环境
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm

# 3. 修复 git safe.directory（每个新 shell 都需要）
git config --global --add safe.directory $(pwd)
git config --global --add safe.directory $(pwd)/third_party/vipe

# 4. 设置模型缓存路径（避免重新下载 ~1.9 GB 权重）
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

# 5. 验证环境健康
python -m pytest -q   # 期望：140 passed, 0 failed

# 6. 如果 GeoCalib 权重丢失（VIPE 启动时报 ConnectionResetError）：
mkdir -p ~/.cache/torch/hub/geocalib
cp /mnt/afs/davidwang/cache/torch/hub/geocalib/pinhole.tar \
    ~/.cache/torch/hub/geocalib/
```

---

## 十三、实验结论文档索引

| 文档 | 内容 |
|---|---|
| `experiments/vipe_comparison/RESULTS_fr1_desk.md` | fr1/desk 28s 完整实验报告（数值+可视化+分析） |
| `experiments/vipe_comparison/RESULTS_fr2_desk.md` | fr2/desk 99s 完整实验报告（含 GeoCalib 错误记录和恢复方案） |
| `docs/superpowers/plans/2026-05-29-next-steps.md` | 三条 Track A/B/C 的完整实施计划（含所有代码片段） |
| `PROGRESS.md` | 项目技术进度（代码细节、API 陷阱，随时更新） |
| `DATA_ANNOTATION_STATUS.md` | 本文件（面向入门读者的宏观状态总览） |

---

## 十四、已发布标注数据集分析：jdvbbfb-v3-full

> **调研日期**：2026-06-02  
> **HF 地址**：`https://huggingface.co/datasets/junchaoh-cs/jdvbbfb-v3-full`

### 14.1 这是什么数据集？

这是一个**已经用 SANA-WM 风格数据管线完成标注并发布的 WebDataset 数据集**。

可以这样理解它的意义：我们正在构建的数据管线，已经有人（用户名 `junchaoh-cs`，来自 `/mnt/cephfs/data/processing/junchao.huang/` 路径，推测是合作团队成员）跑过一遍，并将结果上传到了 HuggingFace。数据打包时间是 2026-05-24，比本项目的 VIPE 实验完成时间（2026-05-29）早了 5 天。

**这个数据集可以直接作为 SANA-WM 或类似视频世界模型的训练数据，无需再跑我们的管线**。

### 14.2 数据集规模

| 指标 | 数值 |
|---|---|
| 总大小 | **2.29 TB** |
| 总 clip 数 | **469,757 个**（论文 213K 的 2.2 倍，是 v3 扩展版）|
| 总分片数 | 1,355 个 TAR 文件 |
| 月下载量 | ~8,800 次 |
| 视频规格 | 1280×720，**24 fps** |
| 相机参数格式 | per-frame NPZ，pose 用 `opencv_c2w` 4×4 矩阵，intrinsics 用 `[fx, fy, cx, cy]`（原始分辨率像素单位）|

**注意**：帧率是 24fps，而我们管线目标是 16fps（论文 §3.1 指定）。直接使用需确认世界模型训练是否接受 24fps 输入。

### 14.3 数据源构成（7+1 个来源）

| 数据源 | Clip 数 | 分片数 | 原始大小 | 场景类型 | 与 SANA-WM 论文关系 |
|---|---:|---:|---:|---|---|
| **SpatialVID-hq** | 365,362 | 714 | 1.11 TB | 室内/室外多场景，高质量 | 论文可能使用 |
| **RealEstate10K-360p** | 73,165 | 143 | 135.5 GB | 室内房产视频（房源展示） | 论文明确使用 |
| **sekai-real-walking-hq** | 18,208 | 287 | 610.8 GB | 真实世界行走第一人称视角 | v3 扩展 |
| **DL3DV-ALL-2K** | 9,993 | 50 | 183.7 GB | 高质量室内/室外 3D 场景（COLMAP GT）| 论文明确使用 |
| **sekai-game-walking** | 1,618 | 43 | 90.6 GB | 游戏世界行走视角 | v3 扩展 |
| **sekai-game-drone** | 932 | 5 | 8.96 GB | 游戏世界无人机视角 | v3 扩展 |
| **OmniWorld-Game** | 479 | 76 | 130.8 GB | 游戏世界（含 NPC 动作，角色扮演场景）| 论文未提及 |
| **Context-as-Memory** | 未统计 | — | — | 未知（2026-05-25 后独立验证）| 论文未提及 |
| **合计** | **≥469,757** | **≥1,355** | **~2.27 TB** | — | — |

**SpatialVID-hq 为什么这么大？** 365K clips 占总数的 78%，原始大小 1.11 TB。推测这是一个专门为视频世界模型数据采集的大规模数据源，可能来自某个内部数据采集平台。

### 14.4 每个样本（clip）包含什么？

每个 clip 在 WebDataset TAR 文件里的结构：

```
{sample_id}/
├── video.mp4           # 视频片段（1280×720，24fps，约 5~10s）
│
├── camera.npz          # per-frame 相机参数，包含：
│   ├── poses           # numpy array, shape (T, 4, 4)
│   │                   # T = 帧数（与 video 帧数相同）
│   │                   # 每个 4×4 矩阵是 opencv_c2w（相机坐标到世界坐标的变换）
│   └── intrinsics      # numpy array, shape (T, 4)
│                       # 每行是 [fx, fy, cx, cy]（像素单位，基于原始分辨率）
│
├── depth/              # 深度图序列（每帧一个）
│   # source_depth_tars 中，具体格式待确认
│
├── flow/               # 光流序列（每帧到下一帧的光流）
│   # source_flow_tars 中，具体格式待确认
│
└── {metadata}          # 元数据，包含：
    # sample_id, dataset, split (train/val/test)
    # uid (场景 ID), group_index, scene_hash
    # source_split_indices, source_relpath, annotation_relpath
    # camera_source_intrinsics (原始内参)
```

**什么是 opencv_c2w？** 相机坐标系到世界坐标系的变换矩阵。具体地：

```
P_world = pose_c2w × P_camera
```

OpenCV 坐标约定：X 轴向右，Y 轴向下，Z 轴向前（相机朝向）。这与 OpenGL（Y 轴向上）不同，使用时需注意约定转换。

### 14.5 与本项目 Stage 06 输出格式的对比

| 字段 | 本项目 Stage 06 schema | jdvbbfb-v3-full | 兼容性与差异说明 |
|---|---|---|---|
| 视频 | `video.mp4` | `video.mp4` | ✅ 完全兼容 |
| 位姿 | `pose.npy`（T,4,4） | `camera.npz` 内的 `poses`（T,4,4）| ⚠️ 字段名不同，读取代码需适配 |
| 内参 | `intrinsics.npy`（T,4）独立文件 | `camera.npz` 内的 `intrinsics`（T,4）| ⚠️ 合并在同一 npz，需改读取逻辑 |
| 描述 | `caption.txt` | `prompt`（dict 结构，含详细场景描述）| ⚠️ 格式稍有差异 |
| 元数据 | `meta.json` | `metadata`（dict，含 scene_hash 等）| ✅ 内容类似 |
| 深度 | ——（当前未输出）| 有 depth tar | — 我们可以考虑后续补充 |
| 光流 | ——（当前未输出）| 有 flow tar | — 我们可以考虑后续补充 |

**结论**：两种格式的核心信息（视频+位姿）是一致的，但字段命名和打包方式有差异。如果要把 jdvbbfb-v3-full 的数据读成我们 Stage 06 格式，需要写一个简单的转换脚本（约 30 行代码）。

### 14.6 与 SANA-WM 论文数据的关系

SANA-WM 论文提到使用 7 个数据源、213K clips。jdvbbfb-v3-full 有 469K+ clips，是论文规模的 2.2 倍，推测是后续的 v3 扩展版本（可能在论文发布后继续扩充数据）：

| 论文中的数据源 | 在 jdvbbfb-v3-full 中 | 规模对比 |
|---|---|---|
| DL3DV | ✅ DL3DV-ALL-2K | 9,993 clips（论文未披露具体数字）|
| RealEstate10K | ✅ RealEstate10K-360p | 73,165 clips |
| ScanNet++ | ❓（未明确对应） | 可能合并在 SpatialVID-hq 中 |
| MatterPort3D | ❓ | 可能合并在 SpatialVID-hq 中 |
| Waymo | ❓ | 未明确 |
| nuScenes | ❓ | 未明确 |
| YouTube 爬取 | ❓ | 可能是 SpatialVID-hq 的一部分 |
| **v3 新增** | SpatialVID-hq（最大）、Sekai 系列、OmniWorld-Game、Context-as-Memory | 论文未提及 |

### 14.7 本地使用策略

**核心问题**：2.29 TB 超出 AFS 可用空间（~644 GB），不能全量下载。

**方案 A：流式读取（推荐，不占本地磁盘）**

```python
import webdataset as wds

# 流式访问 DL3DV-ALL-2K 子集的前 5 个 shard
# 注意：需要 HF 账号 token，因为 HF 原始文件通过重定向访问
dataset = (
    wds.WebDataset(
        "https://huggingface.co/datasets/junchaoh-cs/jdvbbfb-v3-full/"
        "resolve/main/wds-DL3DV-ALL-2K/shards/DL3DV-ALL-2K-{000000..000004}.tar",
        handler=wds.warn_and_continue,
    )
    .decode()  # 自动解码 mp4/npz/txt
    .to_tuple("mp4", "camera.npz")  # 取视频和相机参数
)

for video_bytes, camera_npz in dataset:
    import io, numpy as np
    camera = np.load(io.BytesIO(camera_npz))
    poses = camera["poses"]       # (T, 4, 4)
    intrinsics = camera["intrinsics"]  # (T, 4): [fx, fy, cx, cy]
    print(f"Clip: {poses.shape[0]} frames, fx={intrinsics[0,0]:.1f}")
```

**方案 B：按数据源分批下载（建议先下小的）**

```bash
# 先下载最小的 sekai-game-drone（仅 8.96 GB，5 个 shard）
df -h /mnt/afs/davidwang/workspace | tail -1  # 确认有足够空间

huggingface-cli download junchaoh-cs/jdvbbfb-v3-full \
    --include "wds-sekai-game-drone/*" \
    --repo-type dataset \
    --local-dir /mnt/afs/davidwang/workspace/data/jdvbbfb_game_drone \
    --local-dir-use-symlinks False
# 约 9 GB，下载约 15 min

# 如果空间足够，再下 DL3DV-ALL-2K（183 GB，50 个 shard）
# 这是与我们 Track B 最相关的数据源（DL3DV 有 Colmap GT pose）
huggingface-cli download junchaoh-cs/jdvbbfb-v3-full \
    --include "wds-DL3DV-ALL-2K/*" \
    --repo-type dataset \
    --local-dir /mnt/afs/davidwang/workspace/data/jdvbbfb_dl3dv \
    --local-dir-use-symlinks False
# 约 184 GB，下载约 2~3 h
```

**⚠️ 下载前必须确认磁盘空间**：
```bash
df -h /mnt/afs/davidwang/workspace | tail -1
# "Available" 列必须 ≥ 下载大小 × 1.2（留 20% buffer）
```

### 14.8 对本项目的价值与使用建议

| 用途 | 具体说明 | 建议优先级 |
|---|---|---|
| **直接作为训练数据** | schema 与论文高度对齐，可直接送入 SANA-WM 或类似世界模型的训练，完全跳过我们的数据生产管线 | ⭐⭐⭐ 最高 |
| **验证 Stage 06 格式兼容性** | 下载 1 个 shard，对比 camera.npz 的具体数值和格式，确认我们输出的 `pose.npy`/`intrinsics.npy` 与它兼容，写对齐转换脚本 | ⭐⭐⭐ 高 |
| **Track B e2e 的 baseline 对比** | 用 DL3DV-ALL-2K 中已标注的 clip 作为"参考答案"，对比我们的管线对同一场景产出的位姿精度是否接近 | ⭐⭐ 中 |
| **Track C CADF 算法验证** | 从 camera.npz 提取 GT 位姿，对比我们 4 种 fusion kernel 的 SLAM 位姿与 GT 的差异，扩大评测集 | ⭐⭐ 中 |
| **Context-as-Memory 调研** | 这个数据源在论文未提及，可能是视频记忆/长视频理解类任务，值得研究其具体格式和用途 | ⭐ 低 |

### 14.9 已知问题与注意事项

1. **HuggingFace Dataset Viewer 损坏**：页面显示 `DatasetGenerationError`（TypeCasting 失败），但这只影响 HF 的在线预览，不影响 TAR 文件本身。验证日志（`validation_summary_20260524.json`）显示所有数据 100% 完整，可以放心下载使用。

2. **帧率 24fps vs 论文 16fps**：jdvbbfb-v3-full 的视频全部是 24fps。SANA-WM 论文 §3.1 指定训练数据为 16fps（960 帧/60s）。直接使用 24fps 数据训练，等效于每段 clip 的时间覆盖变短（960 帧 × 1/24s = 40s，而非 60s）。这是否影响模型性能需要实验验证，或者在加载数据时做一次帧率转换（24fps → 16fps，即每 3 帧取 2 帧）。

3. **License 未声明**：HuggingFace 页面未标注具体许可证。在用于任何对外发布的训练之前，需联系 `junchaoh-cs`（邮件或 HF Discussion 区）确认使用条款。各子数据源（RealEstate10K、DL3DV 等）本身有各自的 license，需分别遵守。

4. **Context-as-Memory 数量未知**：该数据源在主验证日志（2026-05-24）中未出现，在 2026-05-25 的独立验证文件（`verify_context_as_memory_20260525.json`）中被单独处理。具体样本数和 schema 有待进一步调研（可通过 `huggingface-cli` 查看 `wds-Context-as-Memory/` 目录内容）。

5. **来源路径**：验证日志显示原始数据路径为 `/mnt/cephfs/data/processing/junchao.huang/...`，说明这是由合作团队在内部 CephFS 存储上产出后上传到 HF 的。不是我们独立复现的结果，是一份已有的标注产出。

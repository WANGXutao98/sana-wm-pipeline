# SANA-WM 数据标注管线 — 进度与下一步

> **目的**：让任何一次新开的 Claude 会话（或新的开发者）无须回顾历史 token 即可立即继续工作。  
> **最新更新**：2026-05-29  
> **当前 tag**：`v0.1.0-pipeline-paper-aligned`（HEAD）

---

## 0. 一行总结

✅ **管线代码全部完成**（14/14 Task，140/140 pytest）；✅ **VIPE 对比实验全部完成**：fr1/desk 28s（ATE↓36%，后半漂移↓34%）+ fr2/desk 99s（**ATE↓10%、尺度偏差 18.5%→3.3%、后半平移漂移↓70%**）；⏳ **下一步：P1 外部模型实绑 / P2 数据源下载与冒烟**。

---

## 1. 项目坐标

| 项 | 值 |
|---|---|
| 仓库路径 | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/` |
| Conda env | `/mnt/afs/davidwang/miniconda3/envs/sana_wm`（持久） |
| 论文原文 | `/mnt/afs/davidwang/workspace/2605.178v1.pdf` |
| 实施计划 | `docs/superpowers/plans/2026-05-25-sana-wm-data-pipeline.md` |
| 当前 git tag | `v0.1.0-pipeline-paper-aligned` |
| 总 commit | 15 |
| 总测试 | 140 passing |

### 1.1 重启 / 新会话冷启动命令

```bash
# 1. 启动 Claude（机器重启后必须）
cd /mnt/afs/davidwang && bash workspace/start_claude.sh

# 2. 进项目 + 激活 env + 修复 git safe.directory
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
git config --global --add safe.directory $(pwd)
git config --global --add safe.directory $(pwd)/third_party/vipe

# 3. 跑测试确认环境健康
python -m pytest -q   # 应输出 140 passed
```

### 1.2 关键模型权重

| 模型 | 路径 | 大小 | 状态 |
|---|---|---|---|
| Pi3X | `/mnt/afs/davidwang/models/pi3x/model.safetensors` | 5.1 GB | ✅ |
| MoGe-2 | `/mnt/afs/davidwang/models/moge2/model.pt` | 1.3 GB | ✅ |
| metric3d-small | `~/.cache/torch/hub/` | ~358 MB | ✅（已下载缓存） |

---

## 2. 管线代码（已完成，14 / 14 Task）

| # | Task | 代码 | 测试 | Commit |
|---|---|---|---|---|
| 1 | 项目骨架 + License 注册（Table 11 verbatim） | `pyproject.toml`, `LICENSING.md` | smoke | `b5ebd57` |
| 2 | Stage-01 ingest + 720p@16fps normalize | `stage01_ingest/normalize.py` + 7 个 downloader 占位 | 2 | `62fa473` |
| 3 | Stage-02a Pi3X+MoGe-2 深度融合（EMA 0.99） | `stage02_pose/depth_fusion.py` | 14 | `13d567a`, `6a3a236` |
| 4 | Stage-02b per-frame intrinsics + Umeyama 80%-inlier | `stage02_pose/per_frame_intrinsics.py`, `umeyama.py` | 6+7 | `9de3d98` |
| 5 | Stage-02c VIPE patch（depth backend + BA） | `third_party/vipe_patch/*.py`, `scripts/00_setup_vipe.sh` | — | `ef6367c` |
| 6 | Stage-02d 3 种 pose 模式 | `stage02_pose/{mode_default,mode_gtdepth,mode_gtpose,_common}.py` | 4 | `18f3697` |
| 7 | Stage-02e Pose 质量过滤（FOV/focal/scale-CV） | `stage02_pose/pose_quality.py` | 13 | `781ed52` |
| 8 | Stage-03 DL3DV 3DGS 增强（FCGS+40trajs+DiFix3D） | `stage03_3dgs_aug/*.py` | 6+9 | `810f08a` |
| 9 | Stage-04 视觉指标（UniMatch 0.5s/DOVER 5s/VMAF/sat） | `stage04_filter/visual_metrics.py`, `scene_cut.py` | 16 | `c7e08b4` |
| 10 | Stage-04b VLM + Table 6 阈值 | `stage04_filter/{vlm_entity_quality,apply_table6}.py`, `configs/filter_thresholds.yaml` | 15 | `13d8cb8` |
| 11 | Stage-05 静态场景 caption（拒绝相机动作短语） | `stage05_caption/{prompts,postprocess,qwen35_vl_runner}.py` | 19 | `5ee126e` |
| 12 | Stage-06 WebDataset 打包 | `stage06_pack/{schema,webdataset_writer}.py` | 11 | `39f93f7` |
| 13 | Orchestration（Ray DAG + SLURM 模板） | `orchestrate/ray_pipeline.py`, `slurm_jobs/*.sbatch`, `scripts/e2e_smoke.sh` | 4 | `e60e026` |
| 14 | 端到端验证 + README + Troubleshooting | `scripts/verify_consistency.py`, `README.md`, `docs/TROUBLESHOOTING.md` | 9 | `081bfa8` |

### 2.1 论文常数全部固化（`configs/pipeline.yaml` + `configs/filter_thresholds.yaml`）

- **1280×720 @ 16fps**，960 raw / 961 camera frames，VAE C=128
- 深度融合 EMA momentum **0.99**，inverse-depth 权重 w_i=1/d_i^Pi3X
- Umeyama **80%-inlier** 阈值（SVD）
- FOV **[25°, 120°]**，|fx−fy|/avg ≤ **0.20**，scale CV ≤ **2.0**
- UniMatch **每 0.5s 抽对** / **前 60s** 窗口
- DOVER **5s 非重叠 chunk**
- DiFix3D：num_steps=**1**, timestep=**199**, guidance=**0**
- 3DGS：每 scene **40 traj** = 10 spline + 30 family（8 个家族）

---

## 3. VIPE 对比实验（2026-05-29）

> **目标**：严格按 SANA-WM 论文 App. B.1，验证 Pi3X+MoGe-2 替换 metric3d-small 作为 SLAM 深度后端的位姿精度提升。  
> **实验路径**：`experiments/vipe_comparison/`

### 3.1 上一轮实验的错误（重要历史）

| 错误 | 根因 | 订正 |
|---|---|---|
| Method A 用 `unidepth-l` 而非 `metric3d-small` | 照搬 VIPE 的 `default.yaml` | 改为 `metric3d-small`（论文 App. B.1 明确指定） |
| Method C 位姿 = Method A（完全一致） | `VideoPi3XDepthProcessor` 是 post-processor，在 SLAM 完成后运行，不影响 BA | 废弃；改为 `CachedDepthModel` 作为 `slam.keyframe_depth` 直接注入 BA |
| Method B 的 Pi3X 实际从未运行 | `AdaptiveDepthProcessor` UV score=0.78 > 阈值 0.3，走 SLAM 投影路径，自定义模型被跳过 | `CachedDepthModel` 完全绕过 `AdaptiveDepthProcessor` |

**核心洞见**：深度必须通过 `slam.keyframe_depth` 路径进入，才能被 `buffer.py` 的 `update_disps_sens()` 调用，影响 Bundle Adjustment。post-processor 无法影响已完成的 SLAM 位姿。

### 3.2 正确实现架构（两阶段）

```
[离线预计算] precompute_pi3x_depths.py
  读完整视频 → Pi3X 分块推理 (chunk=16, stride=8, H/W 对齐14倍数)
             → MoGe-2 逐帧推理 (metric depth, fov_x=62.73° for TUM)
             → EMA scale fusion（论文公式）:
                 s_t = mean(d_MoGe) / mean(d_Pi3X)  # WLS, w=1/d_Pi3X
                 s_ema_t = s_ema_{t-1} * 0.99 + s_t * 0.01
                 depth_fused = s_ema * d_Pi3X
             → 保存 cache.npz: depths(T,H,W), scale_history(T,)

[在线 SLAM] VIPE buffer.py update_disps_sens()
  → DepthEstimationInput(rgb=..., frame_idx=int(tstamp[kf].item()))
  → CachedDepthModel.estimate() 查 cache.npz[frame_idx]
  → (1,H,W) metric depth → disps_sens → Bundle Adjustment → 位姿
```

### 3.3 VIPE 代码修改清单（5 处，已全部应用）

| 文件 | 修改内容 |
|---|---|
| `third_party/vipe/vipe/priors/depth/base.py` | `DepthEstimationInput` 新增字段 `frame_idx: int \| None = None` |
| `third_party/vipe/vipe/slam/components/buffer.py` | `update_disps_sens()` 追加 `frame_idx=int(self.tstamp[frame_idx].item())` |
| `third_party/vipe/vipe/priors/depth/__init__.py` | 新增 `cached` 分支，读 `SANA_WM_CACHED_DEPTH_PATH` 环境变量 |
| `third_party/vipe/vipe/priors/depth/cached.py` | **新建** `CachedDepthModel`（numpy 查表，无 GPU 推理，depth_type=METRIC_DEPTH） |
| `third_party/vipe/configs/pipeline/vipe_metric3d_small.yaml` | **新建** Method A pipeline config（`slam.keyframe_depth: metric3d-small`） |
| `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml` | **新建** Method B pipeline config（`slam.keyframe_depth: cached`） |

### 3.4 已安装的关键包（sana_wm env）

```
vipe 1.1.0         pip install --no-user --no-build-isolation -e .  (third_party/vipe/)
pi3-0.1            pip install git+https://github.com/yyfz/Pi3.git
moge-2.0.0         pip install git+https://github.com/microsoft/MoGe.git
evo 1.36.5         轨迹评测
```

### 3.5 fr1/desk 实验结果（✅ 已完成）

**数据集**：TUM RGB-D freiburg1/desk，613 帧，~28s  
**结果文件**：`experiments/vipe_comparison/RESULTS_fr1_desk.md`  
**可视化**：`experiments/vipe_comparison/results/plots/{trajectory_3views,ate_analysis,comparison,depth_quality,ema_scale}.png`

| 指标 | **A: metric3d-small (baseline)** | **B: Pi3X+MoGe-2 (SANA-WM)** | 提升 |
|---|:---:|:---:|:---:|
| ATE RMSE ↓ (m) | 0.0355 | **0.0227** | **↓36%** |
| ATE mean ↓ (m) | 0.0296 | **0.0200** | ↓32% |
| ATE max ↓ (m) | 0.0981 | **0.0794** | ↓19% |
| 估计尺度 (→1.0) | 1.0791 | **0.9892** | 偏差 ↓91% |
| RTE 旋转均值 ↓ (°) | 1.542 | **1.283** | ↓17% |
| RTE 平移均值 ↓ (m) | 0.0446 | **0.0317** | ↓29% |
| **RTE 后半旋转 ↓ (°)** | 1.371 | **0.903** | **↓34%** |
| **RTE 后半平移 ↓ (m)** | 0.0404 | **0.0257** | **↓36%** |

**结论**：Method B 在全部 8 项指标上领先；后半漂移 ↓34-36% 直接验证论文"long video stability"主张。

### 3.5.1 fr2/desk 实验结果（✅ 2026-05-29 完成）

**数据**：2257 帧 / ~99s 长序列  
**实验日志**：`/tmp/run_fr2_resume.log`  
**结果目录**：`experiments/vipe_comparison/results_fr2/`  
**可视化**：`results_fr2/plots/comparison.png`  
**深度缓存**：`results/cache_pi3x_moge2_fr2_desk.npz` (2.1 GB)

| 指标 | A: VIPE+metric3d-small | B: VIPE+Pi3X+MoGe-2 | 提升 |
|---|:---:|:---:|:---:|
| **ATE RMSE ↓ (m)** | 0.0215 | **0.0194** | **↓10%** |
| ATE mean ↓ (m) | 0.0198 | **0.0179** | ↓10% |
| ATE max ↓ (m) | 0.0573 | **0.0545** | ↓5% |
| **估计尺度 (→1.0)** | 1.1851 | **1.0326** | **偏差 18.5% → 3.3%（↓82%）** |
| RTE 旋转均值 (°) | **0.4318** | 0.4425 | A 略好 0.011°（可忽略）|
| **RTE 平移均值 ↓ (m)** | 0.0375 | **0.0121** | **↓68%** |
| RTE 后半旋转 (°) | **0.3761** | 0.3838 | A 略好 0.008°（可忽略）|
| **RTE 后半平移 ↓ (m)** | 0.0336 | **0.0102** | **↓70%** |

**Method B 在 6/8 项指标领先**（旋转两项差距 < 0.012°，实质并列）。

**关键观察（fr2/desk vs fr1/desk 对比）**：
1. **米制尺度精度差异更显著**：metric3d-small 在长序列上漂移到 1.185（18.5% 偏差），Pi3X+MoGe-2 维持 1.033（3.3% 偏差）— 这是 EMA 时序融合在长视频上的核心优势。
2. **后半段平移漂移大幅扩大**：fr1 是 ↓36%，fr2 直接 **↓70%**，强力验证论文"long video stability"主张随序列长度放大。
3. **ATE 绝对值更小**：fr2 两个方法的 ATE 都比 fr1 小（A: 0.022 vs 0.036; B: 0.019 vs 0.023），因为 fr2/desk 序列相机运动平缓且回环闭合好。

### 3.6 数据集状态

| 序列 | 帧数 | 时长 | 视频 | GT | 缓存 | 实验 |
|---|---|---|---|---|---|---|
| fr1/desk | 613 | 28s | ✅ `data/rgbd_dataset_freiburg1_desk/video.mp4` | ✅ 613 行 | ✅ `results/cache_pi3x_moge2_fr1_desk.npz` (570 MB) | ✅ 已完成 |
| fr2/desk | 2257 | ~99s | ✅ `data/rgbd_dataset_freiburg2_desk/video.mp4` | ✅ 2257 行 | ✅ `results/cache_pi3x_moge2_fr2_desk.npz` (2.1 GB) | ✅ 已完成 |

### 3.7 fr2/desk 执行命令（下次继续）

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

bash experiments/vipe_comparison/run_corrected.sh fr2 2>&1 | tee /tmp/run_fr2.log
# 预计耗时：~1.5 小时
# 结果保存：experiments/vipe_comparison/results_fr2/
```

**手动分步执行：**
```bash
# Step 1: 预计算深度缓存（~45 min，2257 帧）
python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4 \
    --out   experiments/vipe_comparison/results/cache_pi3x_moge2_fr2_desk.npz

# Step 2: Method A（~20 min）
vipe infer experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4 \
    --pipeline vipe_metric3d_small \
    --output experiments/vipe_comparison/results_fr2/method_A_m3d

# Step 3: Method B（~10 min）
export SANA_WM_CACHED_DEPTH_PATH=experiments/vipe_comparison/results/cache_pi3x_moge2_fr2_desk.npz
vipe infer experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4 \
    --pipeline vipe_cached_depth \
    --output experiments/vipe_comparison/results_fr2/method_B_cached

# Step 4: 评测
python experiments/vipe_comparison/evaluate.py \
    --seq     experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk \
    --results experiments/vipe_comparison/results_fr2
```

### 3.8 已知 API 陷阱（务必不要犯）

| 包 | 正确用法 | 错误用法 |
|---|---|---|
| Pi3X 导入 | `from pi3 import Pi3X` | ~~`from pi3 import Pi3`~~ |
| Pi3X 输入 | `(B,N,3,H,W)`，H/W 必须是 14 的倍数 | ~~`infer()` 方法不存在~~ |
| Pi3X 输出 | `outputs["local_points"][..., 2]`（z 分量） | ~~`outputs["depth"]` 不存在~~ |
| MoGe-2 导入 | `from moge.model.v2 import MoGeModel` | ~~`from moge.model import MoGeModel`~~ |
| MoGe-2 加载 | `MoGeModel.from_pretrained("path/model.pt")` | ~~传目录路径报 IsADirectoryError~~ |
| metric_depth shape | `(B,H,W)` 保留 batch dim | ~~`squeeze(0)` → `(H,W)` 报 IndexError~~ |
| VIPE keyframe_depth | `slam.keyframe_depth: cached`（注入 BA） | ~~`post.depth_align_model`（不影响位姿）~~ |

---

## 4. 下一步（按优先级）

### 4.1 ~~P0 — fr2/desk 长序列验证~~（✅ 已完成 2026-05-29）

fr2/desk 99s 长序列已跑完，结果见 §3.5.1。RTE 后半平移漂移 ↓70%，比 fr1/desk 的 ↓36% 显著放大 — 论文"long video stability"主张获得强力验证。

**实验中遇到的坑（写入 memory）**：
- VIPE 启动需要下载 GeoCalib/SAM/GroundingDINO/metric3d 等多个模型；`/root/.cache/torch/hub/` 是临时盘，重启后丢失。已备份到 `/mnt/afs/davidwang/cache/torch/hub/` (~1.9 GB)。
- 续跑机制（`run_corrected.sh` 检查 `pose/video.npz` 存在性跳过 Step 2/3）工作良好，crashed 后只需删除 `method_A_m3d/` 重跑即可。

### 4.2 P1 — 外部模型实绑（管线全量跑通）

当前所有外部模型调用都是 callable 注入 stub。要跑通真实数据：

| 模型 | 实绑位置 | 模型路径 |
|---|---|---|
| **VIPE** | `scripts/00_setup_vipe.sh` + `third_party/vipe_patch/` | `third_party/vipe/` |
| **Pi3X** | `stage02_pose/mode_gtpose.py` 子进程 | `/mnt/afs/davidwang/models/pi3x/` ✅ |
| **MoGe-2** | VIPE depth backend 内部 | `/mnt/afs/davidwang/models/moge2/` ✅ |
| **UniMatch** | `stage04_filter/visual_metrics.py` | 待下载 |
| **DOVER** | `stage04_filter/visual_metrics.py` | 待下载 |
| **FCGS** | `stage03_3dgs_aug/fcgs_fit.py` | 待论文公开 |
| **DiFix3D** | `stage03_3dgs_aug/difix3d_refine.py` | 待论文公开 |
| **Qwen3.5-VL** | `stage05_caption/qwen35_vl_runner.py` | 待下载（或 fallback Qwen2.5-VL-7B） |

### 4.3 P2 — 数据源下载 + H100 单卡冒烟

```bash
# 先验证 6 clip smoke test
bash scripts/e2e_smoke.sh

# 再跑 700 clip 子集（每源 100 clip，~2 天 H100×1）
```

### 4.4 P3 — CMCC 64×H100 全量 213K clip

```bash
rsync -avz /mnt/afs/davidwang/workspace/sana_wm_pipeline/ cmcc:/filestorage/davidwang/sana_wm_pipeline/
sbatch src/sana_wm_pipeline/orchestrate/slurm_jobs/stage01_normalize.sbatch
# 约 7 天，pose 阶段是瓶颈
```

---

## 5. 已知坑

1. **conda env 必须用 `-p` 路径** — `/root/.local/` 被 3 小时自动重启清掉，只有 `/mnt/afs/davidwang/...` 持久。
2. **pip 默认 `user=true`** — 必须 `--no-user` 才装到 env，否则装到非持久的 `/root/.local`。
3. **`git config safe.directory`** — 每个新 shell 都要重新加（uid 不一致）。
4. **conda ffmpeg 缺 libx264.so.138** — 用 `conda install -c conda-forge ffmpeg` 安装 8.1.1（已完成）。
5. **VIPE 安装卡住** — 加 `--no-build-isolation` 绕过 PEP 517 隔离构建。
6. **Pi3X 需要 H/W 为 14 的倍数** — fr1/desk 原始 480×640 → 478×630；不能一次处理 613 帧（OOM）→ chunk=16, stride=8。
7. **metric3d-small 首次运行会从 HuggingFace 自动下载**（~358MB metric3d + ~237MB DeAOT）— 已缓存在 `~/.cache/torch/`，机器重启后缓存保留。
8. **SANA_WM_CACHED_DEPTH_PATH 必须在 `vipe infer` 前 export** — CachedDepthModel 构造时读取，vipe infer 过程中无法动态更新。

---

## 6. 文件树速览

```
sana_wm_pipeline/
├── PROGRESS.md                    ← 本文件（保持最新）
├── README.md
├── LICENSING.md                   ← Table 11 verbatim
├── pyproject.toml                 ← v0.1.0
├── configs/
│   ├── pipeline.yaml              ← 论文常数全部在这
│   ├── sources.yaml               ← 7 个数据源
│   └── filter_thresholds.yaml    ← Table 6 verbatim
├── docs/
│   ├── TROUBLESHOOTING.md
│   ├── DATASETS.md
│   └── vipe_integration_issues.md
├── experiments/
│   └── vipe_comparison/
│       ├── RESULTS_fr1_desk.md   ← ✅ 完整实验报告（含数值结果）
│       ├── evaluate.py           ← A vs B 评测脚本
│       ├── precompute_pi3x_depths.py ← 离线深度缓存预计算
│       ├── prepare_tum.py        ← TUM fr1/fr2 数据准备
│       ├── run_corrected.sh      ← 主运行脚本（fr1/fr2 切换）
│       ├── data/
│       │   ├── rgbd_dataset_freiburg1_desk/  ✅ 613帧，含video.mp4+gt_aligned.txt
│       │   └── rgbd_dataset_freiburg2_desk/  ✅ 2257帧，含video.mp4+gt_aligned.txt
│       └── results/
│           ├── cache_pi3x_moge2_fr1_desk.npz  ✅ 570 MB（已预计算）
│           ├── method_A_m3d/pose/video.npz    ✅ metric3d-small 结果
│           ├── method_B_cached/pose/video.npz  ✅ Pi3X+MoGe-2 结果
│           └── plots/                          ✅ 5 张可视化图
├── src/sana_wm_pipeline/
│   ├── stage01_ingest/           normalize + 7 downloaders
│   ├── stage02_pose/             depth_fusion + umeyama + pose_quality + 3 modes
│   ├── stage03_3dgs_aug/         FCGS + traj + DiFix3D
│   ├── stage04_filter/           visual_metrics + scene_cut + apply_table6 + vlm
│   ├── stage05_caption/          prompts + postprocess + qwen35_vl_runner
│   ├── stage06_pack/             schema + webdataset_writer
│   └── orchestrate/              ray_pipeline + slurm_jobs/
├── third_party/
│   ├── vipe/                     ← VIPE 源码（已修改 5 处）
│   │   ├── vipe/priors/depth/
│   │   │   ├── base.py           ← frame_idx 字段已添加
│   │   │   ├── cached.py         ← CachedDepthModel（新建）
│   │   │   └── __init__.py       ← cached 分支已添加
│   │   ├── vipe/slam/components/
│   │   │   └── buffer.py         ← frame_idx 传递已添加
│   │   └── configs/pipeline/
│   │       ├── vipe_metric3d_small.yaml  ← 新建
│   │       └── vipe_cached_depth.yaml    ← 新建
│   └── vipe_patch/               ← 早期 patch 存档
└── tests/                        ← 140 tests，全部通过
```

---

## 7. 一句话回顾给未来的 Claude

> **管线代码已完成**（pytest 140 passed）。**VIPE 对比实验已全部完成**：fr1/desk 28s（ATE↓36%、后半漂移↓34%）+ fr2/desk 99s（ATE↓10%、尺度偏差 18.5%→3.3%、**后半平移漂移↓70%**），论文 "long video stability" 主张获得强力验证。实验代码的关键是 `CachedDepthModel`（`third_party/vipe/vipe/priors/depth/cached.py`）通过 `slam.keyframe_depth: cached` 注入 BA，以及 `buffer.py` 里传 `frame_idx=int(tstamp[kf].item())`——核心链路不要动。论文常数在 `configs/*.yaml`，禁止手改。**模型缓存已备份到 `/mnt/afs/davidwang/cache/torch/hub/`**（GeoCalib/SAM/AOT/GroundingDINO/metric3d-small/HF hub，~1.9 GB），重启后用 `export TORCH_HOME=/mnt/afs/davidwang/cache/torch HF_HOME=/mnt/afs/davidwang/cache/huggingface` 可避免重新下载。下一步是 P1 外部模型实绑（UniMatch/DOVER 下载 + 接 stub 替换），或 P2 数据源接入。

# SANA-WM 数据标注管线 — 进度与下一步

> **目的**：让任何一次新开的 Claude 会话（或新的开发者）无须回顾历史 token 即可立即继续工作。
> **最新更新**：2026-05-26
> **当前 tag**：`v0.1.0-pipeline-paper-aligned`（HEAD）

---

## 0. 一行总结

✅ 论文 SANA-WM (arXiv:2605.15178v1) 第 4 节 + 附录 B「Robust Annotation Pipeline」14 个 Task 全部完成；**140 / 140** pytest 通过；代码骨架可直接喂数据跑标注，外部模型实绑后即可全量。

---

## 1. 项目坐标

| 项 | 值 |
|---|---|
| 仓库路径 | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/` |
| Conda env | `/mnt/afs/davidwang/miniconda3/envs/sana_wm`（持久） |
| 实施计划 | `/mnt/afs/davidwang/workspace/docs/superpowers/plans/2026-05-25-sana-wm-data-pipeline.md` |
| 论文原文 | `/mnt/afs/davidwang/workspace/2605.15178v1.pdf` |
| 当前 git tag | `v0.1.0-pipeline-paper-aligned` |
| 总 commit | 15 |
| 总测试 | 140 passing |

### 1.1 重启 / 新会话冷启动命令

```bash
# 1. 启动 Claude（机器重启后必须）
cd /mnt/afs/davidwang && bash workspace/start_claude.sh

# 2. 进项目 + 激活 env + 修复 git safe.directory（每个新 shell 都要）
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
git config --global --add safe.directory $(pwd)

# 3. 跑测试确认环境健康
python -m pytest -q   # 应输出 140 passed
```

---

## 2. 已完成（14 / 14 Task）

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

### 2.1 论文常数全部固化

参见 `configs/pipeline.yaml` + `configs/filter_thresholds.yaml`：
- **1280×720 @ 16fps**，960 raw / 961 camera frames，VAE C=128
- 深度融合 EMA momentum **0.99**，inverse-depth 权重
- Umeyama **80%-inlier** 阈值
- FOV **[25°, 120°]**，\|fx−fy\|/avg ≤ **0.20**，scale CV ≤ **2.0**
- UniMatch **每 0.5s 抽对** / **前 60s** 窗口
- DOVER **5s 非重叠 chunk**
- DiFix3D：num_steps=**1**, prompt=**"remove degradation"**, timestep=**199**, guidance=**0**
- 3DGS：每 scene **40 traj** = 10 spline + 30 family（8 个家族）

### 2.2 关键设计抉择（须知）

| 选择 | 论文表述 | 我们的取舍 |
|---|---|---|
| 外部模型耦合方式 | 论文未提 | **Dependency Injection** — UniMatch/DOVER/VLM/FCGS/DiFix3D 都通过 callable 注入，单测全 mock，无 GPU 即可运行 |
| Pi3X 权重许可 | 仅引用 [14] | 公开权重为 **CC-BY-NC-4.0**，非商用 |
| Qwen3.5-VL 可得性 | 引用 [102] | 若未公开，自动 fallback 到 **Qwen2.5-VL**，prompt 不变 |
| 视频归一化策略 | 论文未指定 | center-crop（SANA-Video 实践） |
| Plücker raymap 是否进 shard | 论文每条 sample 实时算 | **不预算**，节省 ~12.5 TB；dataloader 端按 8 raw/1 latent 实时计算 |
| 深度融合权重 `w_i=1/d_i` 取哪一边 | 论文未明 | 取 Pi3X 侧（被 scale 的那一边） |
| EMA 首帧 | 论文未明 | `s_prev=None → s_t=s*_t` |

---

## 3. 下一步（按优先级）

### 3.1 P0 — 外部模型实绑（必须做）

当前所有外部模型调用都是 `subprocess.check_call(...)` 或 `callable` 注入 stub。要跑通真实数据，必须把这些"接线"补上。

| 模型 | 用途 | 实绑位置 | 验证仓库 URL |
|---|---|---|---|
| **VIPE** (Apache-2.0) | SLAM 前端 + BA | 运行 `scripts/00_setup_vipe.sh`；然后把 `third_party/vipe_patch/depth_backend_pi3x_moge2.py` 注册进 VIPE 的 `--depth-backend pi3x_moge2_fused` CLI | https://github.com/nv-tlabs/vipe |
| **Pi3X** (BSD-3 + 权重 CC-BY-NC-4.0) | 长序列一致深度 | `mode_gtpose.run_gtpose` 中 `python -m pi3x.infer` 子进程；模型权重需下载到 `/mnt/afs/davidwang/models/pi3x/` | https://github.com/yyfz/Pi3 |
| **MoGe-2** (MIT) | per-frame metric depth | 在 VIPE depth backend 内部加载；模型权重下载到 `/mnt/afs/davidwang/models/moge2/` | https://github.com/microsoft/MoGe |
| **UniMatch** | optical flow | `stage04_filter/visual_metrics.unimatch_flow_magnitude` 的 `flow_fn` 注入点 | https://github.com/autonomousvision/unimatch |
| **DOVER** | 视频质量评分 | `stage04_filter/visual_metrics.dover_score` 的 `dover_fn` 注入点 | https://github.com/VQAssessment/DOVER |
| **FCGS** | 3DGS 快速拟合 | `stage03_3dgs_aug/fcgs_fit.fit_fcgs` 的 `fit_fn` 注入点 | 待论文 ref [94] 公开 |
| **DiFix3D** | 单步精炼 | `stage03_3dgs_aug/difix3d_refine.refine_clip` 的 `difix3d_pipeline` 注入点；参数 lock 在 `DIFIX3D_PARAMS` 已经强制校验 | 待论文 ref [95] 公开 |
| **Qwen3.5-VL** (Apache-2.0) | caption + entity/quality | `stage05_caption/qwen35_vl_runner.py` 的 `_load_model`；若未公开自动 fallback Qwen2.5-VL-7B-Instruct | https://huggingface.co/Qwen |

**步骤**：
```bash
# 创建模型缓存目录
mkdir -p /mnt/afs/davidwang/models/{pi3x,moge2,qwen-vl,unimatch,dover}

# 1. VIPE
bash scripts/00_setup_vipe.sh

# 2. 模型下载（用 huggingface-cli 或 modelscope）
huggingface-cli login   # 接受 SpatialVID-HQ CC-BY-NC-SA 4.0
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /mnt/afs/davidwang/models/qwen-vl
# Pi3X / MoGe-2 / UniMatch / DOVER 按各自 README 指令下载
```

### 3.2 P1 — 数据源下载

`configs/sources.yaml` 7 个数据源 target_clips 合计 **212,975**。所有 HF repo_id 标了 `# placeholder; verify`，需要先访问 HF / 项目官网确认精确路径。

| Source | HF / 官网 | 状态 |
|---|---|---|
| SpatialVID-HQ | `MERaLiON/SpatialVID-HQ`（gated，需登录） | 待确认 |
| DL3DV / DL3DV-GS | https://dl3dv-10k.github.io/DL3DV-10K/ | 待确认（自定义条款） |
| OmniWorld | https://github.com/yuzhou914/OmniWorld | 待确认 |
| Sekai | https://github.com/Lixsp11/sekai-codebase | 待确认 |
| MiraData | `TencentARC/MiraData`（GPL-3.0） | 待确认 |

### 3.3 P2 — H100 单卡子集冒烟

```bash
# 把 sources.yaml 暂时改成每个源 100 clip = 700 clip 总，单卡 H100 估约 8 天
# 或者直接 --smoke 跑 6 clip 验证流水线
bash scripts/e2e_smoke.sh

# 检查 shard
python scripts/verify_consistency.py data/sana_wm/shards/
```

预计耗时（H100×1）：normalize 8s + pose 3min + filter 12s + caption 4s + pack 1s ≈ **4 min/clip**；700 clip ≈ **2 天**。

### 3.4 P3 — CMCC 64×H100 全量

参考 `/mnt/afs/davidwang/workspace/docker-images/cmcc/docs/`：

```bash
# 上传仓库到 CMCC filestorage（持久盘）
rsync -avz /mnt/afs/davidwang/workspace/sana_wm_pipeline/ \
  cmcc:/filestorage/davidwang/sana_wm_pipeline/

# SLURM 提交
cd /filestorage/davidwang/sana_wm_pipeline
sbatch src/sana_wm_pipeline/orchestrate/slurm_jobs/stage01_normalize.sbatch
sbatch --dependency=afterok:<id1> src/sana_wm_pipeline/orchestrate/slurm_jobs/stage02_pose.sbatch
sbatch --dependency=afterok:<id2> src/sana_wm_pipeline/orchestrate/slurm_jobs/stage05_caption.sbatch
```

全量 213K clip 估约 **7 天**（pose 阶段是瓶颈）。

---

## 4. 已知坑（写入代码即出现的非显然约束）

1. **conda env 必须用 `-p` 路径** — `/root/.local/conda/envs/` 会被 3 小时自动重启清掉，只有 `/mnt/afs/davidwang/...` 持久。
2. **pip config 默认 `user=true`** — 必须 `PIP_USER=false pip install --no-user ...` 才会装到 env，否则装到非持久的 `/root/.local`。
3. **`git config safe.directory`** — 每个新 shell 都要重新加（uid 不一致问题）：`git config --global --add safe.directory /mnt/afs/davidwang/workspace/sana_wm_pipeline`。
4. **pytest 9.x on py3.10** 缺一堆小依赖（tomli/pluggy/iniconfig/pygments/execnet/exceptiongroup）— 用 `pip install -e ".[dev]"` 走全套，或单装。
5. **conda ffmpeg 缺 libx264.so.138** — 用 `pip install static-ffmpeg && static_ffmpeg -y`，然后把 binaries 复制到项目 `.bin/`。
6. **HF gated dataset** — SpatialVID-HQ / MiraData 需要 `huggingface-cli login` 并接受 CC-BY-NC-SA 4.0 条款才能下载。

---

## 5. 文件树速览

```
sana_wm_pipeline/
├── PROGRESS.md                       ← 本文件
├── README.md                         ← 总览
├── LICENSING.md                      ← Table 11 verbatim
├── pyproject.toml
├── configs/
│   ├── pipeline.yaml                 ← 论文常数全部在这
│   ├── sources.yaml                  ← 7 个数据源 + 目标 clip 数
│   └── filter_thresholds.yaml        ← Table 6 verbatim
├── docs/
│   ├── DATA_SCHEMA.md
│   └── TROUBLESHOOTING.md
├── scripts/
│   ├── 00_setup_vipe.sh              ← VIPE 克隆 + patch
│   ├── e2e_smoke.sh                  ← in-process DAG smoke
│   └── verify_consistency.py         ← shard 校验 CLI
├── src/sana_wm_pipeline/
│   ├── stage01_ingest/               ← normalize.py + downloaders/*.py
│   ├── stage02_pose/                 ← depth_fusion + umeyama + pose_quality + 3 modes
│   ├── stage03_3dgs_aug/             ← FCGS + traj + coverage + DiFix3D
│   ├── stage04_filter/               ← visual_metrics + scene_cut + apply_table6 + vlm
│   ├── stage05_caption/              ← prompts + postprocess + qwen35_vl_runner
│   ├── stage06_pack/                 ← schema + webdataset_writer
│   └── orchestrate/                  ← ray_pipeline + slurm_jobs/*.sbatch
├── third_party/vipe_patch/           ← depth_backend + BA patches
└── tests/                            ← 140 tests across 16 files
```

---

## 6. 一句话回顾给未来的 Claude

> 进入 `sana_wm_pipeline/`、激活 env、跑 `pytest -q`：应见 140 passed。接下来按 §3 的 P0→P3 顺序推进，先把 8 个外部模型实绑、再下载 7 个数据源、然后 H100 单卡跑子集冒烟，最后扔上 CMCC 64×H100 跑全量 213K clip。所有论文常数已在 `configs/*.yaml`，禁止手改；任何"看起来该这么写"的随手改动，先回去翻论文 §4 / 附录 B / 附录 D.1 / Table 6 / Table 11。

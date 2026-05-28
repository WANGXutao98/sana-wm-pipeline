# SANA-WM 全部数据资产清单与下载指南

> 严格按 SANA-WM 论文 (arXiv:2605.15178v1) §4 Table 1 + Appendix Table 11 整理。
> 所有 URL 已 2026-05-26 通过 Web 搜索验证（见每条目下面的 "Source verified" 行）。

---

## 摘要：要下哪些 + 总盘

### A. 训练数据源（7 个，论文 Table 1，目标 212,975 clips）

| Source | 论文计数 | Pose 模式 | License | HF/官方 仓库 | 估盘 |
|---|---|---|---|---|---|
| **SpatialVID-HQ** | 158,369 | default | CC-BY-NC-SA 4.0（gated） | `SpatialVID/SpatialVID-HQ` | **~3.53 TB** |
| **DL3DV (real)** | 5,691 | gt-pose | DL3DV custom terms | `DL3DV/DL3DV-ALL-960P` | ~1.6 TB（960p 全量） |
| **DL3DV-GS** | 14,881 | gt-pose | 同 DL3DV（合成产物） | — Stage-03 派生 | — |
| **OmniWorld** | 1,720 | gt-depth | CC-BY-NC-SA 4.0 | `InternRobotics/OmniWorld` | ~多 TB（多模态） |
| **Sekai-Game** | 3,560 | gt-pose | 项目条款 | `Lixsp11/Sekai-Project` | ~90 GB |
| **Sekai-Walking-HQ** | 9,767 | default | 项目条款 | `Lixsp11/Sekai-Project` | ~ 数百 GB |
| **MiraData** | 18,987 | default | GPL-3.0（数据 host 单独条款） | `TencentARC/MiraData` | **330K 版本 ~ 数 TB**；可选 9K/42K 小版本 |

**全量原始数据估计 ~ 8 TB+**。

### B. 外部模型 / 工具（8 个，论文 Table 11）

| Asset | 用途 | License | 仓库 |
|---|---|---|---|
| **VIPE** | SLAM + BA 引擎（patch 本仓 stage02） | Apache-2.0 | `nv-tlabs/vipe` |
| **Pi3X** | 长序列一致深度 | BSD-3 + 权重 CC-BY-NC-4.0 | `yyfz/Pi3` + `yyfz233/Pi3X` (HF) |
| **MoGe-2** | per-frame metric depth | MIT | `Ruicheng/moge-2-vitl-normal` |
| **UniMatch** | optical flow（视觉过滤） | MIT-ish | `autonomousvision/unimatch` |
| **DOVER** | 视频质量评分 | Apache-2.0 | `VQAssessment/DOVER` |
| **FCGS** | 3DGS 快速拟合（DL3DV-GS） | 研究代码 | `YihangChen-ee/FCGS` |
| **DiFix3D** | 单步精炼 | NVIDIA License | `nv-tlabs/Difix3D` + 权重 `nvidia/difix` (HF) |
| **Qwen3.5-VL** (fallback Qwen2.5-VL) | caption + 实体计数 + 质量 flag | Apache-2.0 | `Qwen/Qwen3.5-VL-7B-Instruct` / `Qwen/Qwen2.5-VL-7B-Instruct` |

---

## C. 推荐落地策略（回答你的问题）

### ✅ 我的建议：完全同意你的方案。本机验证用 SpatialVID-HQ 小子集即可。

**理由：**

1. **覆盖最多**：SpatialVID-HQ 用 `default` pose 模式（modified VIPE + Pi3X + MoGe-2 + per-frame BA），是论文 6 个标注路径里**最复杂**的；其他源（OmniWorld 的 `gt_depth`、DL3DV 的 `gt_pose`）都是这条路径的简化版。
2. **占比最大**：论文 Table 1 里 SpatialVID 占 **158,369 / 212,975 = 74%**，把它跑通就把主路径打通了。
3. **盘小好下**：HF 支持 `--include "group_0001/*"` 部分下载，单个 group ~ 15.5 GB；裁出 **10 个 clip 用于 smoke ≈ 150 MB**，完全在本机 H100 可承受。
4. **缺什么补什么**：唯一不能覆盖的是：
   - Stage-03 3DGS 增强（DL3DV-only）
   - `gt_pose` / `gt_depth` 两条 pose 模式分支

   这些可以后续各下 **2-3 个 clip** 单独验证（DL3DV、OmniWorld、Sekai-Game 各下小子集即可），不影响主路径冒烟。

5. **CMCC 上同样适用**：先用 SpatialVID 一个 group 在 CMCC 跑通整套，再开 64×H100 全量。

### 推荐时间线

```
[本机 H100×1，今天可做]
SpatialVID-HQ group_0001 (10 clip) → e2e_smoke → verify_consistency  ≈ 2-4 h

[本机 H100×1，明天补]
+ OmniWorld 2 clip 验证 gt_depth 路径
+ DL3DV 2 clip 验证 gt_pose 路径
+ Sekai-Game 2 clip 复查 gt_pose                                      ≈ 3-5 h

[CMCC 64×H100]
按 §5 下全量；预计 pose 阶段 7 天                                       ≈ 1 周
```

---

# 详细下载指令

## D.1 SpatialVID-HQ ⭐（首要 + 本机 smoke 用）

- **HF dataset**: https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ
- **官方 GitHub**: https://github.com/NJU-3DV/SpatialVID
- **Source verified**: 2026-05-26（Web 搜索；NJU-3DV / CVPR 2026）
- **License**: CC-BY-NC-SA 4.0 — **gated**，先 `huggingface-cli login` 并去 HF 页面点 "Agree and access repository"
- **总盘**: ~3.53 TB（74 个 group_xxxx 子目录，每组 ~ 14 GB 视频 + 1.5 GB 标注）

### 本机 smoke 下载（仅 group_0001）

```bash
# 1. 登录 HF（一次性）
pip install --no-user "huggingface_hub[cli]"
huggingface-cli login          # 浏览器贴 HF token（去 https://huggingface.co/settings/tokens）

# 2. 去仓库页面 https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ 点 "Agree"
#    (gated 数据集必须先在网页同意条款)

# 3. 只下 group_0001（≈ 15 GB，本机够用）
huggingface-cli download SpatialVID/SpatialVID-HQ \
  --repo-type dataset \
  --include "group_0001/*" \
  --local-dir /mnt/afs/davidwang/data/spatialvid_hq
```

### CMCC 全量下载

```bash
# 全量 3.53 TB；推荐 nohup 后台
nohup huggingface-cli download SpatialVID/SpatialVID-HQ \
  --repo-type dataset \
  --local-dir /filestorage/davidwang/data/spatialvid_hq \
  > /filestorage/davidwang/logs/spatialvid_hq_download.log 2>&1 &
```

---

## D.2 DL3DV-10K（real 5,691 clip + 派生 GS 14,881）

- **HF dataset (960p)**: https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P
- **官方 GitHub**: https://github.com/DL3DV-10K/Dataset
- **Source verified**: 2026-05-26
- **License**: DL3DV custom terms（需要先在 GitHub 页面 apply 同意条款）

### 推荐：用官方 download.py 拉子集

```bash
# 1. 官方下载脚本（支持灵活子集）
wget https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/scripts/download.py

# 2. 本机 smoke：拉 1K 子集的视频 + pose
python download.py \
  --odir /mnt/afs/davidwang/data/dl3dv \
  --subset 1K \
  --resolution 960P \
  --file_type video \
  --clean_cache

# 3. 同时拉 pose（GT pose 是 gt_pose 模式的核心输入）
python download.py \
  --odir /mnt/afs/davidwang/data/dl3dv \
  --subset 1K \
  --resolution 960P \
  --file_type images+poses \
  --clean_cache
```

### CMCC 全量

```bash
# 5,691 个视频 + pose（论文用的 real 子集）
python download.py \
  --odir /filestorage/davidwang/data/dl3dv \
  --subset 10K --resolution 960P --file_type video --clean_cache
python download.py \
  --odir /filestorage/davidwang/data/dl3dv \
  --subset 10K --resolution 960P --file_type images+poses --clean_cache
```

> DL3DV-GS（14,881 synthetic）= 本仓 Stage-03 在 DL3DV real 之上自动产出，**不需要单独下载**。

---

## D.3 OmniWorld（synthetic 1,720 clip，gt_depth 模式）

- **HF dataset**: https://huggingface.co/datasets/InternRobotics/OmniWorld
- **官方 GitHub**: https://github.com/yangzhou24/OmniWorld（ICLR 2026）
- **Source verified**: 2026-05-26
- **License**: CC-BY-NC-SA 4.0
- **重要子集**: 论文用的是 **OmniWorld-Game**（合成 720p，96K clip / 214h / 18M frames）

```bash
# 推荐先下 OmniWorld-Game 子集
huggingface-cli download InternRobotics/OmniWorld \
  --repo-type dataset \
  --include "OmniWorld-Game/*" \
  --local-dir /mnt/afs/davidwang/data/omniworld

# 或全量
huggingface-cli download InternRobotics/OmniWorld \
  --repo-type dataset \
  --local-dir /filestorage/davidwang/data/omniworld
```

---

## D.4 Sekai（Game + Walking-HQ）

- **HF dataset**: https://huggingface.co/datasets/Lixsp11/Sekai-Project
- **官方 GitHub**: https://github.com/Lixsp11/sekai-codebase（NeurIPS 2025）
- **Source verified**: 2026-05-26
- **License**: 项目自有条款（无标准 OSS license，按项目页面）

### 子集文件名（已知）

| 文件 | 内容 | 大小 |
|---|---|---|
| `sekai-game-walking.zip.part_aa` | 游戏行走数据 part 1 | 48.3 GB |
| `sekai-game-walking.zip.part_ab` | 游戏行走数据 part 2 | 41.9 GB |
| `sekai-real-walking-hq.zip.part_*` | 真实行走 HQ（论文 9,767 clip 来源） | 需查 |

```bash
# 整 dataset
huggingface-cli download Lixsp11/Sekai-Project \
  --repo-type dataset \
  --local-dir /filestorage/davidwang/data/sekai

# 或只下 game-walking
huggingface-cli download Lixsp11/Sekai-Project \
  --repo-type dataset \
  --include "sekai-game-walking.zip.part_*" \
  --local-dir /filestorage/davidwang/data/sekai

# 解压后合并 part 文件
cat sekai-game-walking.zip.part_* > sekai-game-walking.zip
unzip sekai-game-walking.zip
```

---

## D.5 MiraData（real 18,987 clip）

- **HF dataset**: https://huggingface.co/datasets/TencentARC/MiraData
- **官方 GitHub**: https://github.com/mira-space/MiraData
- **Source verified**: 2026-05-26
- **License**: GPL-3.0（仓库），原视频按各 host 条款（YouTube/Pexels/Mixkit/etc.）
- **特殊**：MiraData 只发 **meta CSV**，视频要按 URL 自行下载

### 步骤

```bash
# 1. 下 meta CSV
huggingface-cli download TencentARC/MiraData \
  --repo-type dataset \
  --local-dir /filestorage/davidwang/data/miradata

# 2. 用官方脚本按 URL 拉视频（注意：YouTube 大量限流，建议分批 + 代理）
git clone https://github.com/mira-space/MiraData.git
cd MiraData

python download_data.py \
  --meta_csv /filestorage/davidwang/data/miradata/miradata_v1_9k.csv \
  --download_start_id 0 \
  --download_end_id 100 \
  --raw_video_save_dir /filestorage/davidwang/data/miradata/raw \
  --clip_video_save_dir /filestorage/davidwang/data/miradata/clips
```

> 论文 18,987 clip 大约对应 MiraData v1 的 42K 版本子集（去重 + 过滤后）。
> 也可以先用最小的 **9K 版本**做冒烟验证。

---

# E. 模型权重下载（CMCC 跑前必装）

```bash
mkdir -p /filestorage/davidwang/models/{vipe,pi3x,moge2,qwen-vl,unimatch,dover,fcgs,difix3d}

# E.1 VIPE — 通过本仓 setup 脚本自动拉
bash scripts/00_setup_vipe.sh

# E.2 Pi3X 权重（CC-BY-NC-4.0 非商用）
huggingface-cli download yyfz233/Pi3X --local-dir /filestorage/davidwang/models/pi3x

# E.3 MoGe-2（verified 2026-05-28：正确 repo 为 Ruicheng/moge-2-vitl-normal）
git clone https://github.com/microsoft/MoGe /filestorage/davidwang/code/MoGe
hf download Ruicheng/moge-2-vitl-normal --local-dir /filestorage/davidwang/models/moge2

# E.4 Qwen3.5-VL（若已发布）或 fallback Qwen2.5-VL
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /filestorage/davidwang/models/qwen-vl

# E.5 UniMatch
git clone https://github.com/autonomousvision/unimatch /filestorage/davidwang/code/unimatch
# 模型权重见 unimatch README

# E.6 DOVER
git clone https://github.com/VQAssessment/DOVER /filestorage/davidwang/code/DOVER
# checkpoint 在 release 页面下载

# E.7 FCGS
git clone https://github.com/YihangChen-ee/FCGS /filestorage/davidwang/code/FCGS

# E.8 DiFix3D
git clone https://github.com/nv-tlabs/Difix3D /filestorage/davidwang/code/Difix3D
huggingface-cli download nvidia/difix --local-dir /filestorage/davidwang/models/difix3d
```

> **盘空间估算**：所有模型权重合计 ~80 GB（Qwen-VL 7B 占大头 ~15 GB；DiFix3D ~5 GB；Pi3X ~2 GB；MoGe-2 ~1 GB；其他更小）。

---

# F. 本机 smoke 验证执行序列（按你建议）

```bash
# 0. 假设已激活 env + 项目目录
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm

# 1. 下 SpatialVID-HQ group_0001（~15 GB；先在 HF 网页同意 gated 条款）
huggingface-cli login
huggingface-cli download SpatialVID/SpatialVID-HQ \
  --repo-type dataset \
  --include "group_0001/*" \
  --local-dir /mnt/afs/davidwang/data/spatialvid_hq

# 2. 配置数据路径
#    编辑 configs/pipeline.yaml 把 paths.raw_root 指到上面下载目录
#    然后从 group_0001 里挑 10 个 clip 复制到 raw_root/spatialvid_hq/

# 3. 装外部模型权重到 /mnt/afs/davidwang/models/（同 E 节）
#    Pi3X + MoGe-2 + Qwen2.5-VL 是冒烟必需

# 4. 应用 VIPE patch
bash scripts/00_setup_vipe.sh

# 5. 跑 smoke
bash scripts/e2e_smoke.sh

# 6. 验证产出
python scripts/verify_consistency.py /mnt/afs/davidwang/workspace/data/sana_wm/shards/
```

预期输出：
- 至少 1 个 `shard-000000.tar` 在 `out_root` 下
- `verify_consistency.py` 全 OK，0 fail

---

# G. 一致性检查清单（论文/配置 vs 我们）

| 项 | 论文/官方说法 | 本仓 `configs/sources.yaml` | 状态 |
|---|---|---|---|
| SpatialVID-HQ repo | `SpatialVID/SpatialVID-HQ` (HF) | `NJU-PCALab/SpatialVID-HQ` (旧) | ⚠️ 需更新 |
| OmniWorld repo | `InternRobotics/OmniWorld` (HF) | `yyfz233/OmniWorld` (旧) | ⚠️ 需更新 |
| Sekai repo | `Lixsp11/Sekai-Project` (HF) | `Lixsp11/sekai` (旧) | ⚠️ 需更新 |
| MiraData repo | `TencentARC/MiraData` (HF) | `TencentARC/MiraData` | ✅ |
| DL3DV download | 官方 download.py + apply | — | ✅（不强求 HF） |

> 这些差异会被本次更新一并修掉。

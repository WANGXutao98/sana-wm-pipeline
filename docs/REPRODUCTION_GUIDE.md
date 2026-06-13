# SANA-WM 数据标注管线复现指南

> **For agentic workers:** This guide documents the end-to-end pipeline for
> the SANA-WM paper (arXiv:2605.15178v1). Three pose-annotation modes are
> validated: GT-pose (DL3DV), GT-depth (OmniWorld), and Default (both).
> Run `pytest tests/ -v` first to verify unit tests, then follow the
> dataset-specific sections below.

**当前验证结果（2026-06-13，H100 单卡）：**

| 模式 | 数据集 | ATE RMSE | SANA-WM PSNR | SANA-WM SSIM |
|------|--------|----------|-------------|-------------|
| GT-pose | DL3DV (4 scenes) | ~1.8e-7 m | — | — |
| GT-depth | OmniWorld `020c2bed1dbb` | 9.07 mm | 18.50 dB | 0.7302 |
| Default | OmniWorld `020c2bed1dbb` | 9.08 mm | 18.84 dB | 0.7575 |
| Default | DL3DV `0032cd2f` | 127.7 mm | — | — |

---

## 前置要求

| 项目 | 最低要求 |
|---|---|
| GPU | NVIDIA H100 80GB（或 A100 80GB） |
| 磁盘 | ≥ 200 GB（模型 ~107 GB + 数据 ~25 GB） |
| 网络 | 可访问 GitHub、HuggingFace、ModelScope |
| conda | Miniconda3，路径持久化（AFS 或本地固态）|

> ⚠️ 本指南所有 `<YOUR_BASE>` 均需替换为持久化工作目录，例如
> `/home/yourname` 或 `/data/yourname`。**不要**使用 `/tmp`。

---

## Task 1：克隆仓库

- [ ] **Step 1.1：克隆主仓库及子模块**

```bash
git clone --recurse-submodules \
  https://github.com/WANGXutao98/sana-wm-pipeline.git
cd sana-wm-pipeline
```

- [ ] **Step 1.2：验证子模块**

```bash
ls third_party/vipe/vipe/priors/depth/cached.py
# 预期：文件存在（非空），说明子模块克隆成功
```

- [ ] **Step 1.3：克隆 Sana 推理仓库**

```bash
cd <YOUR_BASE>
git clone https://github.com/NVlabs/Sana.git
cd Sana && git checkout 40151c8   # 已验证的 commit
```

---

## Task 2：创建 Conda 环境

- [ ] **Step 2.1：创建环境（持久化路径）**

```bash
conda create -p <CONDA_BASE>/envs/sana_wm python=3.10 -c conda-forge -y
```

> ⚠️ 必须用 `-p <绝对路径>`，不要用 `-n sana_wm`（默认路径可能在临时目录）。

- [ ] **Step 2.2：激活并验证**

```bash
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate <CONDA_BASE>/envs/sana_wm
python --version   # 预期：Python 3.10.x
```

---

## Task 3：安装依赖

**顺序严格**，部分包有版本冲突。

- [ ] **Step 3.1：安装主管线包**

```bash
cd sana-wm-pipeline
pip install -e ".[dev]"
```

- [ ] **Step 3.2：安装 VIPE**

```bash
pip install -e third_party/vipe
vipe --help   # 预期：输出 "NVIDIA Video Pose Engine (ViPE) CLI"
```

- [ ] **Step 3.3：安装 Pi3X**

```bash
pip install git+https://github.com/yyfz/Pi3.git
python -c "from pi3 import Pi3X; print('Pi3X OK')"
```

- [ ] **Step 3.4：安装 MoGe-2**

```bash
pip install git+https://github.com/microsoft/MoGe.git
python -c "from moge.model.v2 import MoGeModel; print('MoGe-2 OK')"
```

- [ ] **Step 3.5：安装 SANA-WM 专用依赖**

```bash
pip install pyrallis flash-linear-attention einops ftfy came-pytorch
```

- [ ] **Step 3.6：安装 mmcv（顺序敏感）**

```bash
pip install "setuptools<80"
pip install --no-build-isolation mmcv==1.7.2
python -c "from mmcv import Registry; print('mmcv OK')"
```

- [ ] **Step 3.7：安装其余依赖**

```bash
pip install \
  termcolor omegaconf sentencepiece qwen-vl-utils \
  diffusers accelerate "timm>=0.9.0" patch-conv \
  scikit-image static-ffmpeg evo scipy matplotlib
```

> ⚠️ `timm` 必须 ≥0.9.0：同时提供 `timm.layers`（VIPE 需要）和
> `timm.models.layers` 兼容 shim（SANA-WM 需要）。

- [ ] **Step 3.8：全量验证**

```bash
python -c "
import pyrallis, fla, einops, ftfy, termcolor, omegaconf
import sentencepiece, diffusers, accelerate, timm, skimage, evo, scipy
from mmcv import Registry
from pi3 import Pi3X
from moge.model.v2 import MoGeModel
import vipe
print('ALL IMPORTS OK')
print('timm:', timm.__version__)   # 应 >=0.9.0
"
```

- [ ] **Step 3.9：运行单元测试**

```bash
cd sana-wm-pipeline
pytest tests/ -v --tb=short
# 预期：全部通过（all passed，数量随版本演进可能变化）
```

---

## Task 4：下载模型权重（共 ~107 GB）

- [ ] **Step 4.1：Pi3X 权重（5.1 GB）**

```bash
mkdir -p <YOUR_BASE>/models/pi3x
huggingface-cli download yyfz/Pi3X model.safetensors \
  --local-dir <YOUR_BASE>/models/pi3x
```

- [ ] **Step 4.2：MoGe-2 权重（1.3 GB）**

```bash
mkdir -p <YOUR_BASE>/models/moge2
huggingface-cli download microsoft/MoGe-Vitl14-RoPE model.pt \
  --local-dir <YOUR_BASE>/models/moge2
```

- [ ] **Step 4.3：Gemma-2-2b-it（4.9 GB）**

```bash
huggingface-cli download google/gemma-2-2b-it \
  --local-dir <YOUR_BASE>/models/gemma-2-2b-it
```

- [ ] **Step 4.4：SANA-WM 权重（~96 GB）**

```bash
huggingface-cli download Efficient-Large-Model/SANA-WM_bidirectional \
  --local-dir <YOUR_BASE>/models/SANA-WM_bidirectional \
  --repo-type model
ls <YOUR_BASE>/models/SANA-WM_bidirectional/
# 预期包含：config.yaml  dit/  refiner/  vae/
```

- [ ] **Step 4.5：修复 Sana 的 Gemma 路径（必做）**

```bash
SANA_DIR=<YOUR_BASE>/Sana
grep -n "gemma-2-2b-it" $SANA_DIR/diffusion/model/builder.py
# 找到如下行（约 76 行）：
#   "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
# 将其改为指向本地路径：
sed -i 's|"Efficient-Large-Model/gemma-2-2b-it"|"<YOUR_BASE>/models/gemma-2-2b-it"|' \
  $SANA_DIR/diffusion/model/builder.py
grep "gemma-2-2b-it" $SANA_DIR/diffusion/model/builder.py
# 预期：显示本地路径
```

> **原因**：原配置映射到 `refiner/text_encoder`（Gemma3，hidden_size=3840），
> 而 Stage1 DiT 期望 2304 维，会导致矩阵乘法报错。

- [ ] **Step 4.6：设置环境变量（每次实验前执行）**

```bash
export SANA_WM_PI3X_WEIGHTS=<YOUR_BASE>/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=<YOUR_BASE>/models/moge2
export TORCH_HOME=<YOUR_BASE>/cache/torch
export HF_HOME=<YOUR_BASE>/cache/huggingface
export SANA_DIR=<YOUR_BASE>/Sana
mkdir -p $TORCH_HOME $HF_HOME
```

> 建议写入 `~/.bashrc` 或项目根目录 `env.sh`，每次 `source env.sh`。

---

## Task 5：DL3DV 数据准备

适用模式：**GT-pose**（精确零误差位姿）、**Default**（SLAM 估计位姿）

- [ ] **Step 5.1：下载 4 个 smoke test 场景**

```bash
DATA_DIR=<YOUR_BASE>/data/dl3dv_smoke
for SCENE_ID in \
  "0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac" \
  "00534f5868a6f72e77befbdb06e35ee9dc34e175dddf0e64e8b1922e494c8e24" \
  "00713c8c22cf3b2ef6495b1da5484da9972921442d85a0a3c8be57f7aa9bbbb5" \
  "008c201a7eff27ce0413f7931a48e92cf05ded9d9b7cf16cc2276ff3b80c7b22"
do
  huggingface-cli download DL3DV/DL3DV-ALL-2K \
    --include "1K/${SCENE_ID}/*" \
    --repo-type dataset \
    --local-dir $DATA_DIR \
    --local-dir-use-symlinks False
done
```

- [ ] **Step 5.2：预处理（images → video.mp4 + GT 位姿）**

```bash
cd sana-wm-pipeline
for SCENE_ID in \
  "0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac" \
  "00534f5868a6f72e77befbdb06e35ee9dc34e175dddf0e64e8b1922e494c8e24" \
  "00713c8c22cf3b2ef6495b1da5484da9972921442d85a0a3c8be57f7aa9bbbb5" \
  "008c201a7eff27ce0413f7931a48e92cf05ded9d9b7cf16cc2276ff3b80c7b22"
do
  python experiments/data_production_smoke/prepare_dl3dv.py \
    $DATA_DIR/1K/$SCENE_ID
done
ls $DATA_DIR/1K/0032cd2f*/
# 预期含：video.mp4  gt_poses.npy  gt_intrinsics.npy  orig_fps.txt
```

---

## Task 6：OmniWorld 数据准备

适用模式：**GT-depth**（GT 深度图锚定位姿）、**Default**（仅用 RGB 视频）

OmniWorld 数据存于 ModelScope，格式为 `.tar.gz`（annotations + videos 分开）。

- [ ] **Step 6.1：安装 modelscope**

```bash
pip install modelscope
```

- [ ] **Step 6.2：下载 OmniWorld-Game 单场景**

```bash
SCENE_ID="020c2bed1dbb"
OMNI_CACHE=<YOUR_BASE>/data/omniworld
ANNOT_OUT=$OMNI_CACHE/annotations/OmniWorld-Game
VIDEO_OUT=$OMNI_CACHE/videos/OmniWorld-Game
mkdir -p $ANNOT_OUT $VIDEO_OUT

# 用 modelscope CLI 下载（需先安装：pip install modelscope）
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'AI-ModelScope/OmniWorld',
    cache_dir='$OMNI_CACHE',
    allow_patterns=['*$SCENE_ID*'],
)
print('Download complete')
"
# 若 modelscope 下载失败（网络受限），可用 git lfs：
# git clone https://www.modelscope.cn/datasets/AI-ModelScope/OmniWorld.git \
#   $OMNI_CACHE/AI-ModelScope-OmniWorld
```

> **注意**：下载完成后，annotations（含 `*_depth_*.tar.gz`）和 videos（含 `*_rgb_*.tar.gz`）
> 会分别位于对应子目录。`prepare_omniworld.py`（Step 6.3）会自动解压 tar.gz，
> 无需手动操作。

- [ ] **Step 6.3：用 prepare_omniworld.py 提取场景**

```bash
cd sana-wm-pipeline
python experiments/data_production_smoke/prepare_omniworld.py \
  --annot-dir $ANNOT_OUT/$SCENE_ID \
  --video-dir $VIDEO_OUT/$SCENE_ID \
  --out-dir   <YOUR_BASE>/data/omniworld_smoke/$SCENE_ID \
  --split-idx 0 \
  --max-frames 80
ls <YOUR_BASE>/data/omniworld_smoke/$SCENE_ID/
# 预期：video.mp4  gt_depth.npy  gt_poses.npy  gt_intrinsics.npy
# (orig_fps.txt 可能也存在)
```

---

## Task 7：GT-pose 模式（DL3DV，4 场景）

GT-pose 使用 `gt_poses.npy`（来自 DL3DV `transforms.json`）直接作为位姿，不运行 SLAM。位姿精度近似数值零（ATE ≈ 1.8e-7 m）。

- [ ] **Step 7.1：修改 run_e2e_gtpose.sh 路径**

打开 `experiments/data_production_smoke/run_e2e_gtpose.sh`，将以下硬编码路径替换为你的实际路径：

```bash
# 第 10-15 行附近，将：
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
# 替换为：
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate <CONDA_BASE>/envs/sana_wm
export TORCH_HOME=<YOUR_BASE>/cache/torch
export HF_HOME=<YOUR_BASE>/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=<YOUR_BASE>/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=<YOUR_BASE>/models/moge2
```

同样修改 `OUT_DIR`（约第 20 行）：
```bash
OUT_DIR="<YOUR_BASE>/data/dl3dv_smoke_shards_gtpose"
```

- [ ] **Step 7.2：运行 GT-pose 端到端（全部 4 场景）**

```bash
cd sana-wm-pipeline
bash experiments/data_production_smoke/run_e2e_gtpose.sh \
  <YOUR_BASE>/data/dl3dv_smoke
# 全部 4 场景约 20-25 分钟（H100）
# 各阶段耗时：normalize ~8s, Pi3X ~5min（首次 Triton 编译慢）
```

- [ ] **Step 7.3：验证 shard schema**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose
# 预期：5/5 shards valid（含 1 个空占位 shard）
```

- [ ] **Step 7.4：运行 ATE 评估**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose \
  --scenes-dir <YOUR_BASE>/data/dl3dv_smoke/1K \
  --out-dir    <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose/eval_output
cat <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose/eval_output/pose_eval_summary.json
# 预期（ATE RMSE 应在 1e-7 量级）：
# [{"sample_id": "0032cd2f...", "ate_rmse": 1.79e-07}, ...]
```

---

## Task 8：GT-depth 模式（OmniWorld，GT 深度图锚定）

GT-depth 使用 OmniWorld 精确深度图（uint16 PNG → metres）+ MoGe-2 metric anchor + VIPE SLAM。**仅适用于有 GT 深度的数据集（OmniWorld、Hypersim 等）。**

ATE RMSE ≈ 9 mm（VIPE SLAM 小量漂移，GT depth 约束位移尺度）。

> 以下步骤将 `<YOUR_BASE>/data/omniworld_smoke` 记为 `WORK_BASE`，在终端中执行前先设置：
> ```bash
> WORK_BASE=<YOUR_BASE>/data/omniworld_smoke
> ```

- [ ] **Step 8.1：运行 GT-depth 端到端（OmniWorld 单场景）**

```bash
cd sana-wm-pipeline
ANNOT_DIR=<YOUR_BASE>/data/omniworld/annotations/OmniWorld-Game/020c2bed1dbb
VIDEO_DIR=<YOUR_BASE>/data/omniworld/videos/OmniWorld-Game/020c2bed1dbb

# 若 annot 和 video 在同一目录（已用 prepare_omniworld.py 提取）：
# bash experiments/data_production_smoke/run_e2e_gtdepth.sh \
#   <YOUR_BASE>/data/omniworld_smoke/020c2bed1dbb

# 若分开存储（ModelScope 原始结构）：
bash experiments/data_production_smoke/run_e2e_gtdepth.sh \
  $ANNOT_DIR \
  $VIDEO_DIR
```

脚本含 Stage 0（提取/跳过）、Stage 1（normalize）、Stage 1b（depth 重采样）、
Stage 2（VIPE SLAM）、Stage 5（stub caption）、Stage 6（pack shard）。

耗时约（H100）：Stage 0 ~30s，Stage 2 ~15 min（首次 VIPE JIT ~2 min）。

- [ ] **Step 8.2：运行 ATE 评估**

```bash
WORK_BASE=<YOUR_BASE>/data/omniworld_smoke
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir ${WORK_BASE}/shards_gtdepth \
  --scenes-dir ${WORK_BASE} \
  --out-dir    ${WORK_BASE}/shards_gtdepth/eval_output
cat ${WORK_BASE}/shards_gtdepth/eval_output/pose_eval_summary.json
# 预期：[{"sample_id": "020c2bed1dbb", "ate_rmse": 0.009072}]
```

---

## Task 9：Default 模式（OmniWorld + DL3DV，Pi3X+MoGe-2+VIPE）

Default 模式是通用管线：Pi3X 预测相对深度 → MoGe-2 提供 metric anchor（EMA 融合）→ VIPE SLAM（`vipe_cached_depth` 管线）。适用于任何视频。

### 9a：Default 模式 on OmniWorld

- [ ] **Step 9.1：确认 normalized.mp4 已存在（GT-depth Task 8 产生）**

```bash
ls <YOUR_BASE>/data/omniworld_smoke/020c2bed1dbb/normalized.mp4
# 若不存在，先运行 run_e2e_gtdepth.sh（它的 Stage 0-1 生成 normalized.mp4）
```

- [ ] **Step 9.2：运行 Default 端到端（OmniWorld）**

```bash
cd sana-wm-pipeline
ANNOT_DIR=<YOUR_BASE>/data/omniworld/annotations/OmniWorld-Game/020c2bed1dbb
bash experiments/data_production_smoke/run_e2e_default_omniworld.sh \
  $ANNOT_DIR
# 约 20-25 分钟（H100）：Pi3X ~10 min, VIPE SLAM ~10 min
```

脚本检查 `normalized.mp4` 是否存在（不重新提取），然后运行 Pi3X+MoGe-2 深度
缓存 → VIPE SLAM → Stage 6 打包 → schema check → ATE 评估。

- [ ] **Step 9.3：验证结果**

```bash
cat <YOUR_BASE>/data/omniworld_smoke/shards_default/eval_output/pose_eval_summary.json
# 预期：[{"sample_id": "020c2bed1dbb", "ate_rmse": 0.009078}]
```

### 9b：Default 模式 on DL3DV

- [ ] **Step 9.4：修改 run_e2e_default.sh 路径（与 Task 7.1 类似）**

打开 `experiments/data_production_smoke/run_e2e_default.sh`，将 `/mnt/afs/davidwang/...` 路径替换为 `<YOUR_BASE>/...`。

- [ ] **Step 9.5：运行 Default 端到端（DL3DV 单场景）**

```bash
cd sana-wm-pipeline
bash experiments/data_production_smoke/run_e2e_default.sh \
  <YOUR_BASE>/data/dl3dv_smoke/1K/0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac
# 约 25 分钟（H100）
```

- [ ] **Step 9.6：验证 ATE 评估结果**

```bash
cat <YOUR_BASE>/data/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json
# 预期：[{"sample_id": "0032cd2f...", "ate_rmse": 0.127655}]
# SLAM 漂移 ~12.8 cm（DL3DV 无 GT 深度约束，漂移大于 OmniWorld）
```

---

## Task 10：SANA-WM 推理生成视频

对 GT-depth 和 Default 模式的 OmniWorld shard 各运行一次推理，生成可对比的视频。

- [ ] **Step 10.1：修改 run_sana_wm_inference.py 中的模型路径**

打开 `experiments/data_production_smoke/run_sana_wm_inference.py`，找到并修改：

```python
# 约第 315 行 default_model：
default_model = Path("<YOUR_BASE>/models/SANA-WM_bidirectional/dit/sana_wm_1600m_720p.safetensors")
# 约第 127 行 local_config：
local_config = Path("<YOUR_BASE>/models/SANA-WM_bidirectional/config.yaml")
# 约第 145-146 行 refiner 路径：
"--refiner_root",       "<YOUR_BASE>/models/SANA-WM_bidirectional/refiner",
"--refiner_gemma_root", "<YOUR_BASE>/models/SANA-WM_bidirectional/refiner/text_encoder",
```

- [ ] **Step 10.2：GT-depth shard 推理（OmniWorld）**

```bash
WORK_BASE=<YOUR_BASE>/data/omniworld_smoke
python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir ${WORK_BASE}/shards_gtdepth \
  --sana-dir   <YOUR_BASE>/Sana \
  --output-dir <YOUR_BASE>/data/sana_wm_results_gtdepth \
  --sample-limit 1
# 约 3 分钟/样本（H100）：DiT 60步 ~100s, LTX-2 refiner ~6s, VAE ~15s
```

- [ ] **Step 10.3：Default shard 推理（OmniWorld）**

```bash
python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir ${WORK_BASE}/shards_default \
  --sana-dir   <YOUR_BASE>/Sana \
  --output-dir <YOUR_BASE>/data/sana_wm_results_default \
  --sample-limit 1
```

- [ ] **Step 10.4：验证输出文件**

```bash
ls <YOUR_BASE>/data/sana_wm_results_gtdepth/020c2bed1dbb/
ls <YOUR_BASE>/data/sana_wm_results_default/020c2bed1dbb/
# 预期各含：first_frame.png  *_generated.mp4  *_sbs.mp4
```

推理输出指标（PSNR/SSIM vs GT 原始视频）由脚本自动打印到日志。

---

## Task 11：三模式对比

- [ ] **Step 11.1：生成 3-panel 对比视频和 Markdown 报告**

```bash
cd sana-wm-pipeline
python experiments/data_production_smoke/compare_omniworld_modes.py \
  --sample-id 020c2bed1dbb \
  --out-dir <YOUR_BASE>/data/omniworld_smoke/comparison
```

输出：
- `comparison/020c2bed1dbb_comparison_3panel.mp4`（左：GT 原始 | 中：GT-depth 生成 | 右：Default 生成）
- `comparison/comparison_report.md`（含位姿精度 + 视频质量数值表格）

- [ ] **Step 11.2：验证预期结论**

```bash
cat <YOUR_BASE>/data/omniworld_smoke/comparison/comparison_report.md
```

预期数值：

| 指标 | GT-depth | Default |
|------|----------|---------|
| ATE RMSE | 9.07 mm | 9.08 mm |
| PSNR vs GT | 18.50 dB | **18.84 dB** (+0.34) |
| SSIM vs GT | 0.7302 | **0.7575** (+0.027) |

**解读：** ATE 几乎相同（两者均使用 VIPE SLAM）；Default PSNR 略优，原因是
Pi3X 预测深度的平滑性比 OmniWorld GT 深度更符合 SANA-WM 的训练分布。

---

## 常见问题

### `ModuleNotFoundError: No module named 'timm.layers'`
```bash
pip install "timm>=0.9.0"
```

### `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`
transformers 5.x 问题。**本仓库 `third_party/vipe` 已修复**（`bertwarper.py`
已重写 `BertModelWarper`）。若遇到此错误，请确认子模块已更新到最新 commit：
```bash
git submodule update --remote third_party/vipe
```

### `ModuleNotFoundError: No module named 'fla'`（重启后）
```bash
pip install flash-linear-attention
```

### `ModuleNotFoundError: No module named 'mmcv'`（重启后）
```bash
pip install "setuptools<80" && pip install --no-build-isolation mmcv==1.7.2
```

### ffmpeg 找不到 libx264
```bash
pip install static-ffmpeg
```

### VIPE OOM
在 `mode_default.py` 中将 `chunk: int = 16` 改为 `chunk: int = 8`。

### Git 安全目录警告
```bash
git config --global --add safe.directory $(pwd)
```

---

## 文件结构速查

```
sana-wm-pipeline/
├── experiments/data_production_smoke/
│   ├── prepare_dl3dv.py              # DL3DV: images → video.mp4 + gt_poses.npy
│   ├── prepare_omniworld.py          # OmniWorld: tar.gz → video.mp4 + gt_depth.npy
│   ├── run_e2e_gtpose.sh             # GT-pose 模式（DL3DV，4 场景）
│   ├── run_e2e_gtdepth.sh            # GT-depth 模式（OmniWorld）
│   ├── run_e2e_default.sh            # Default 模式（DL3DV）
│   ├── run_e2e_default_omniworld.sh  # Default 模式（OmniWorld）★ 新增
│   ├── verify_and_eval.py            # schema check + ATE pose 评估
│   ├── run_sana_wm_inference.py      # SANA-WM 推理包装器
│   ├── compare_modes.py              # DL3DV 两模式对比报告
│   └── compare_omniworld_modes.py    # OmniWorld 三模式对比（★ 新增）
├── src/sana_wm_pipeline/
│   ├── stage01_ingest/normalize.py   # 视频归一化（1280×720 @ 16fps）
│   ├── stage02_pose/
│   │   ├── mode_gtpose.py            # GT-pose: gt_poses.npy + Umeyama Sim(3)
│   │   ├── mode_gtdepth.py           # GT-depth: GT depth + MoGe-2 + VIPE SLAM
│   │   └── mode_default.py           # Default: Pi3X+MoGe-2 缓存 + VIPE SLAM
│   ├── stage05_caption/              # 字幕生成（smoke 中用 stub fallback）
│   └── stage06_pack/                 # 打包为 WebDataset .tar shard
└── third_party/vipe/                 # VIPE SLAM（子模块，含 transformers 5.x 修复）
    └── vipe/priors/track_anything/groundingdino/models/main/bertwarper.py
        # BertModelWarper 自实现 get_head_mask/get_extended_attention_mask/
        # invert_attention_mask，兼容 transformers 5.12.0+
```

---

## 三模式能力矩阵

| 特性 | GT-pose | GT-depth | Default |
|------|---------|---------|---------|
| 输入 | RGB 视频 + GT 位姿 | RGB 视频 + GT 深度图 | RGB 视频（仅需此项）|
| 数据来源 | DL3DV（有 `transforms.json`）| OmniWorld / Hypersim | 任意视频 |
| SLAM 运行 | ✗（直接用 GT）| ✓（VIPE，GT depth 约束）| ✓（VIPE，Pi3X 约束）|
| ATE RMSE（已验证）| ~1.8e-7 m | ~9 mm | ~9 mm (OmniWorld) / ~128 mm (DL3DV) |
| SANA-WM PSNR | — | 18.50 dB | **18.84 dB** |
| 适用规模 | Smoke test | OmniWorld 数据集 | 生产数据（SpatialVID、MiraData 等）|

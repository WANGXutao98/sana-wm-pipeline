# DL3DV GT-pose vs Default Mode 实验复现指南

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零开始在 H100 单卡上完整复现 SANA-WM DL3DV smoke test：GT-pose 和 Default 两种位姿标注模式，并对生成视频进行定量对比。

**Architecture:** 主仓库（sana-wm-pipeline）+ VIPE 子模块（WANGXutao98/vipe）+ 独立克隆的 Sana 推理仓库；所有中间产物通过 WebDataset `.tar` shard 传递。

**Tech Stack:** Python 3.10, PyTorch 2.12, VIPE SLAM, Pi3X, MoGe-2, SANA-WM DiT + LTX-2 Refiner, evo, WebDataset

---

## 前置要求

| 项目 | 最低要求 |
|---|---|
| GPU | NVIDIA H100 80GB（或 A100 80GB） |
| 磁盘 | ≥ 200 GB 可用空间（模型约 107 GB + 数据约 25 GB） |
| 网络 | 可访问 GitHub、HuggingFace |
| Python | 通过 conda 安装 3.10（不要用系统 Python）|
| conda | Miniconda3 或 Anaconda，路径需持久化（AFS 或本地固态）|

> ⚠️ **重要**：本指南中所有 `<YOUR_BASE>` 均需替换为你自己的持久化工作目录，例如 `/home/yourname` 或 `/data/yourname`。**不要**使用 `/tmp` 或容器热盘（重启后数据丢失）。

---

## Task 1：克隆仓库

- [ ] **Step 1.1：克隆主仓库及子模块**

```bash
git clone --recurse-submodules https://github.com/WANGXutao98/sana-wm-pipeline.git
cd sana-wm-pipeline
```

`--recurse-submodules` 会自动克隆 `third_party/vipe`（指向 `WANGXutao98/vipe`）。

- [ ] **Step 1.2：验证子模块已克隆**

```bash
ls third_party/vipe/vipe/priors/depth/cached.py
```

预期输出：文件存在（非空），说明子模块克隆成功。

- [ ] **Step 1.3：克隆 Sana 推理仓库**

```bash
cd <YOUR_BASE>
git clone https://github.com/NVlabs/Sana.git
cd Sana && git checkout 40151c8   # 经过验证的 commit
```

记录路径：后续称为 `SANA_DIR=<YOUR_BASE>/Sana`。

- [ ] **Step 1.4：修复 Sana 的 Gemma 路径（必做）**

```bash
# 找到 builder.py 中 gemma-2-2b-it 的映射行
grep -n "gemma-2-2b-it" $SANA_DIR/diffusion/model/builder.py
```

找到如下行（行号可能为 76 附近）：
```python
"gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
```

将其改为指向本地权重路径（在 Task 4 下载模型后填入）：
```python
"gemma-2-2b-it": "<YOUR_BASE>/models/gemma-2-2b-it",
```

> **原因**：原配置映射到 `refiner/text_encoder`（Gemma3，hidden_size=3840），而 Stage1 DiT 期望 2304 维，会导致矩阵乘法报错。

---

## Task 2：创建 Conda 环境

- [ ] **Step 2.1：创建环境**

```bash
# 将 <CONDA_BASE> 替换为你的 miniconda 安装路径，例如 /home/yourname/miniconda3
conda create -p <CONDA_BASE>/envs/sana_wm python=3.10 -c conda-forge -y
```

> ⚠️ 必须用 `-p <绝对路径>` 指定持久化路径，不要用 `-n sana_wm`（默认路径可能在容器临时目录）。

- [ ] **Step 2.2：激活环境**

```bash
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate <CONDA_BASE>/envs/sana_wm
```

验证：
```bash
python --version
# 预期：Python 3.10.x
```

---

## Task 3：安装依赖

**顺序严格**，部分包有版本冲突，必须按下列顺序安装。

- [ ] **Step 3.1：安装主管线包（editable）**

```bash
cd sana-wm-pipeline
pip install -e ".[dev]"
```

预期：安装 `sana-wm-pipeline 0.1.0`，以及 torch、numpy 等基础依赖。

- [ ] **Step 3.2：安装 VIPE（editable，子模块）**

```bash
pip install -e third_party/vipe
```

验证：
```bash
vipe --help
# 预期：输出 "NVIDIA Video Pose Engine (ViPE) CLI"
```

- [ ] **Step 3.3：安装 Pi3X**

```bash
pip install git+https://github.com/yyfz/Pi3.git
```

验证：
```bash
python -c "from pi3 import Pi3X; print('Pi3X OK')"
```

- [ ] **Step 3.4：安装 MoGe-2**

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

验证：
```bash
python -c "from moge.model.v2 import MoGeModel; print('MoGe-2 OK')"
```

- [ ] **Step 3.5：安装 SANA-WM 专用依赖**

```bash
pip install pyrallis flash-linear-attention einops ftfy came-pytorch
```

- [ ] **Step 3.6：安装 mmcv（顺序敏感！）**

```bash
# 必须先降 setuptools，再 --no-build-isolation
pip install "setuptools<80"
pip install --no-build-isolation mmcv==1.7.2
```

验证：
```bash
python -c "from mmcv import Registry; print('mmcv OK')"
```

- [ ] **Step 3.7：安装其余依赖**

```bash
pip install \
  termcolor omegaconf sentencepiece qwen-vl-utils \
  diffusers accelerate "timm>=0.9.0" patch-conv \
  scikit-image static-ffmpeg evo scipy matplotlib
```

> ⚠️ `timm` 必须安装 **≥0.9.0**（不是 0.6.13）：0.9+ 同时提供 `timm.layers`（VIPE 需要）和 `timm.models.layers` 兼容 shim（SANA-WM 需要）。

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

---

## Task 4：下载模型权重

共需 4 组权重，总计约 107 GB。

- [ ] **Step 4.1：下载 Pi3X 权重（5.1 GB）**

```bash
mkdir -p <YOUR_BASE>/models/pi3x
# 从 HuggingFace 下载（需要 huggingface-cli 登录）
huggingface-cli download yyfz/Pi3X model.safetensors \
  --local-dir <YOUR_BASE>/models/pi3x
```

> 若无 HuggingFace 账号，可用 `hfd.sh`：
> `bash hfd.sh yyfz/Pi3X --local-dir <YOUR_BASE>/models/pi3x`

- [ ] **Step 4.2：下载 MoGe-2 权重（1.3 GB）**

```bash
mkdir -p <YOUR_BASE>/models/moge2
huggingface-cli download microsoft/MoGe-Vitl14-RoPE model.pt \
  --local-dir <YOUR_BASE>/models/moge2
```

- [ ] **Step 4.3：下载 Gemma-2-2b-it（4.9 GB）**

```bash
huggingface-cli download google/gemma-2-2b-it \
  --local-dir <YOUR_BASE>/models/gemma-2-2b-it
```

- [ ] **Step 4.4：下载 SANA-WM 权重（~96 GB）**

```bash
huggingface-cli download Efficient-Large-Model/SANA-WM_bidirectional \
  --local-dir <YOUR_BASE>/models/SANA-WM_bidirectional \
  --repo-type model
```

下载后验证目录结构：
```bash
ls <YOUR_BASE>/models/SANA-WM_bidirectional/
# 预期包含：config.yaml  dit/  refiner/  vae/
ls <YOUR_BASE>/models/SANA-WM_bidirectional/dit/
# 预期：sana_wm_1600m_720p.safetensors
ls <YOUR_BASE>/models/SANA-WM_bidirectional/refiner/text_encoder/ | head -3
# 预期：多个 model-xxxxx-of-00011.safetensors
```

- [ ] **Step 4.5：回到 Task 1.4，填入 Gemma 本地路径**

```bash
sed -i 's|"Efficient-Large-Model/gemma-2-2b-it"|"<YOUR_BASE>/models/gemma-2-2b-it"|' \
  $SANA_DIR/diffusion/model/builder.py
# 验证修改
grep "gemma-2-2b-it" $SANA_DIR/diffusion/model/builder.py
```

- [ ] **Step 4.6：设置环境变量（每次实验前执行）**

```bash
export SANA_WM_PI3X_WEIGHTS=<YOUR_BASE>/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=<YOUR_BASE>/models/moge2
export TORCH_HOME=<YOUR_BASE>/cache/torch
export HF_HOME=<YOUR_BASE>/cache/huggingface
export SANA_DIR=<YOUR_BASE>/Sana

mkdir -p $TORCH_HOME $HF_HOME
```

> 建议将这 6 行写入 `~/.bashrc` 或项目根目录的 `env.sh`，每次 `source env.sh`。

---

## Task 5：下载 DL3DV 数据

- [ ] **Step 5.1：下载 4 个 smoke test 场景**

```bash
export HF_HOME=<YOUR_BASE>/cache/huggingface
DATA_DIR=<YOUR_BASE>/data/dl3dv_smoke

# 使用内置脚本（注意：脚本中的 SCENES 列表是示例，需替换为实际 scene hash）
# 直接使用已验证的 4 个场景 hash：
for SCENE_ID in \
  "0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac" \
  "00534f5868a6f72e77befbdb06e35ee9dc34e175dddf0e64e8b1922e494c8e24" \
  "00713c8c22cf3b2ef6495b1da5484da9972921442d85a0a3c8be57f7aa9bbbb5" \
  "008c201a7eff27ce0413f7931a48e92cf05ded9d9b7cf16cc2276ff3b80c7b22"
do
  echo "==> Downloading $SCENE_ID"
  huggingface-cli download DL3DV/DL3DV-ALL-2K \
    --include "1K/${SCENE_ID}/*" \
    --repo-type dataset \
    --local-dir $DATA_DIR \
    --local-dir-use-symlinks False
done
```

预期结构：
```
<YOUR_BASE>/data/dl3dv_smoke/1K/
  0032cd2f.../images/*.png  transforms.json
  00534f58.../images/*.png  transforms.json
  00713c8c.../images/*.png  transforms.json
  008c201a.../images/*.png  transforms.json
```

- [ ] **Step 5.2：预处理（图片 → video.mp4 + GT 位姿）**

```bash
cd sana-wm-pipeline
python experiments/data_production_smoke/prepare_dl3dv.py $DATA_DIR/1K/0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac
python experiments/data_production_smoke/prepare_dl3dv.py $DATA_DIR/1K/00534f5868a6f72e77befbdb06e35ee9dc34e175dddf0e64e8b1922e494c8e24
python experiments/data_production_smoke/prepare_dl3dv.py $DATA_DIR/1K/00713c8c22cf3b2ef6495b1da5484da9972921442d85a0a3c8be57f7aa9bbbb5
python experiments/data_production_smoke/prepare_dl3dv.py $DATA_DIR/1K/008c201a7eff27ce0413f7931a48e92cf05ded9d9b7cf16cc2276ff3b80c7b22
```

验证（每个场景应有）：
```bash
ls $DATA_DIR/1K/0032cd2f*/
# 预期含：video.mp4  gt_poses.npy  gt_intrinsics.npy  orig_fps.txt
```

---

## Task 6：GT-pose 模式端到端

- [ ] **Step 6.1：修改 run_e2e_gtpose.sh 中的路径**

打开 `experiments/data_production_smoke/run_e2e_gtpose.sh`，修改以下 4 行：

```bash
# 原（硬编码路径）：
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

# 改为（替换 <CONDA_BASE> 和 <YOUR_BASE>）：
source <CONDA_BASE>/etc/profile.d/conda.sh
conda activate <CONDA_BASE>/envs/sana_wm
export TORCH_HOME=<YOUR_BASE>/cache/torch
export HF_HOME=<YOUR_BASE>/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=<YOUR_BASE>/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=<YOUR_BASE>/models/moge2
```

同样修改 `OUT_DIR`：
```bash
OUT_DIR="<YOUR_BASE>/data/dl3dv_smoke_shards_gtpose"
```

- [ ] **Step 6.2：运行 GT-pose 端到端脚本**

```bash
cd sana-wm-pipeline
bash experiments/data_production_smoke/run_e2e_gtpose.sh <YOUR_BASE>/data/dl3dv_smoke
```

各场景耗时约（H100）：
- Step 0 prepare: ~30s（已完成则跳过）
- Step 1 normalize: ~8s
- Step 2 Pi3X 推理: ~5min（首次编译 Triton kernel 较慢）
- Step 4 caption: <1s（stub fallback）
- Step 5 pack: ~1s

全部 4 场景约 **20-25 分钟**。

- [ ] **Step 6.3：验证 shard schema**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose
```

预期输出（共 5 个 tar，含 1 个空占位）：
```
  [OK]   shard-000001.tar  (1 samples all valid)
  ...
Result: 5/5 shards valid
```

---

## Task 7：GT-pose Pose 评估

- [ ] **Step 7.1：运行 ATE 评估**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose \
  --scenes-dir <YOUR_BASE>/data/dl3dv_smoke/1K \
  --out-dir    <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose/eval_output
```

> ⚠️ `--scenes-dir` 必须指向 `dl3dv_smoke/1K/`（含场景子目录层级），不是 `dl3dv_smoke/`。

- [ ] **Step 7.2：验证结果**

```bash
cat <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose/eval_output/pose_eval_summary.json
```

预期值（ATE RMSE 应在 1e-7 量级，近似数值零）：
```json
[
  {"sample_id": "0032cd2f...", "ate_rmse": 1.79e-07},
  {"sample_id": "00534f58...", "ate_rmse": 1.58e-07},
  {"sample_id": "00713c8c...", "ate_rmse": 2.10e-07},
  {"sample_id": "008c201a...", "ate_rmse": 2.16e-07}
]
```

---

## Task 8：Default 模式端到端（单场景）

- [ ] **Step 8.1：创建工作目录，准备输入视频**

```bash
SAMPLE_ID="0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac"
WORK_DIR=<YOUR_BASE>/data/dl3dv_smoke_shards_default/work/$SAMPLE_ID
mkdir -p $WORK_DIR
```

Default 模式需要 16fps 归一化视频（与 GT-pose 共用，若已跑过 GT-pose 可复用）：

```bash
# 若 GT-pose 步骤已运行，直接复制 normalized.mp4
cp <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose/work/$SAMPLE_ID/normalized.mp4 $WORK_DIR/

# 若未运行 GT-pose，手动归一化：
python - <<'EOF'
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from pathlib import Path
import os
sample_id = "0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac"
src = Path(os.environ["DATA_DIR"]) / "1K" / sample_id / "video.mp4"
dst = Path(os.environ["WORK_DIR"]) / "normalized.mp4"
info = normalize_video(src, dst)
print(f"Normalized: {info.n_frames} frames @ {info.fps} fps")
EOF
```

- [ ] **Step 8.2：创建 Default 模式运行脚本**

将下列内容写入 `<YOUR_BASE>/data/dl3dv_smoke_shards_default/run_default_0032.py`，**替换所有路径**：

```python
#!/usr/bin/env python3
import os, sys, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 路径配置（替换为你的实际路径）──
BASE = "<YOUR_BASE>"
os.environ["SANA_WM_PI3X_WEIGHTS"]  = f"{BASE}/models/pi3x"
os.environ["SANA_WM_MOGE2_WEIGHTS"] = f"{BASE}/models/moge2"
os.environ["TORCH_HOME"]             = f"{BASE}/cache/torch"
os.environ["HF_HOME"]                = f"{BASE}/cache/huggingface"
sys.path.insert(0, f"{BASE}/sana-wm-pipeline/src")

SAMPLE_ID  = "0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac"
WORK_DIR   = Path(f"{BASE}/data/dl3dv_smoke_shards_default/work/{SAMPLE_ID}")
NORM_VIDEO = WORK_DIR / "normalized.mp4"
POSE_DIR   = WORK_DIR / "pose_work"
ARTIFACT   = WORK_DIR / "pose_artifact.npz"
OUT_DIR    = Path(f"{BASE}/data/dl3dv_smoke_shards_default")

assert NORM_VIDEO.exists(), f"normalized.mp4 not found: {NORM_VIDEO}"

# ── Phase A+B: Pi3X+MoGe-2 深度缓存 → VIPE SLAM ──
if not ARTIFACT.exists():
    log.info("Phase A: Pi3X+MoGe-2 depth cache precompute (~10 min)...")
    log.info("Phase B: VIPE SLAM (vipe_cached_depth pipeline, ~10 min)...")
    from sana_wm_pipeline.stage02_pose.mode_default import run_default
    art = run_default(NORM_VIDEO, POSE_DIR)
    np.savez_compressed(str(ARTIFACT),
        poses_c2w=art.poses_c2w,
        intrinsics=art.intrinsics,
        scale_per_frame=art.scale_per_frame)
    log.info("pose_artifact.npz saved: T=%d frames", art.poses_c2w.shape[0])
else:
    log.info("pose_artifact.npz exists, skipping pose estimation.")

# ── Phase C: 打包 WebDataset shard ──
data = np.load(str(ARTIFACT))
from sana_wm_pipeline.stage06_pack.schema import Sample
from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter
from sana_wm_pipeline.stage05_caption.qwen35_vl_runner import CAPTION_FALLBACK

sample = Sample(
    sample_id=SAMPLE_ID,
    video_path=str(NORM_VIDEO),
    poses_c2w=data["poses_c2w"].astype(np.float32),
    intrinsics_NVD=data["intrinsics"].astype(np.float32),
    scale_per_frame=data["scale_per_frame"].astype(np.float32),
    caption=CAPTION_FALLBACK,
    meta={"source": "DL3DV", "pose_mode": "default", "scene_id": SAMPLE_ID},
)
with ShardWriter(str(OUT_DIR), samples_per_shard=100, strict_frames=False) as w:
    w.write(sample)
log.info("Done. Shard at %s", OUT_DIR)
```

- [ ] **Step 8.3：后台运行 Default 模式（约 25 分钟）**

```bash
conda activate <CONDA_BASE>/envs/sana_wm
nohup python <YOUR_BASE>/data/dl3dv_smoke_shards_default/run_default_0032.py \
  > <YOUR_BASE>/data/dl3dv_smoke_shards_default/run_default_0032.log 2>&1 &
echo "PID: $!"

# 监控进度
tail -f <YOUR_BASE>/data/dl3dv_smoke_shards_default/run_default_0032.log
```

各阶段日志关键词：
| 阶段 | 日志关键词 | 耗时 |
|---|---|---|
| Pi3X 分块推理 | `Phase A: Pi3X+MoGe-2` | ~10 min |
| VIPE SLAM | `vipe infer` 启动 | ~10 min |
| 打包 | `Packing WebDataset shard` | ~1 min |
| 完成 | `Done. Shard at` | — |

- [ ] **Step 8.4：验证 shard**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_default
# 预期：1 样本 valid
```

---

## Task 9：Default 模式 Pose 评估

- [ ] **Step 9.1：运行 ATE 评估**

```bash
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_default \
  --scenes-dir <YOUR_BASE>/data/dl3dv_smoke/1K \
  --out-dir    <YOUR_BASE>/data/dl3dv_smoke_shards_default/eval_output
```

- [ ] **Step 9.2：验证结果**

```bash
cat <YOUR_BASE>/data/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json
```

预期（SLAM 漂移 ~13cm）：
```json
[{"sample_id": "0032cd2f...", "ate_rmse": 0.127655}]
```

---

## Task 10：SANA-WM 推理生成

对两种模式的 shard 各运行一次推理，生成可对比的视频。

- [ ] **Step 10.1：修改 run_sana_wm_inference.py 中的模型路径**

打开 `experiments/data_production_smoke/run_sana_wm_inference.py`，找到并修改默认模型路径（约第 315 行）：

```python
# 原：
default_model = Path("/mnt/afs/davidwang/models/SANA-WM_bidirectional/dit/sana_wm_1600m_720p.safetensors")
# 改为：
default_model = Path("<YOUR_BASE>/models/SANA-WM_bidirectional/dit/sana_wm_1600m_720p.safetensors")
```

同样修改 `run_inference()` 函数中的 `local_config`（约第 127 行）：

```python
# 原：
local_config = Path("/mnt/afs/davidwang/models/SANA-WM_bidirectional/config.yaml")
# 改为：
local_config = Path("<YOUR_BASE>/models/SANA-WM_bidirectional/config.yaml")
```

以及 `--refiner_root` 和 `--refiner_gemma_root`（约第 145-146 行）：
```python
"--refiner_root",       "<YOUR_BASE>/models/SANA-WM_bidirectional/refiner",
"--refiner_gemma_root", "<YOUR_BASE>/models/SANA-WM_bidirectional/refiner/text_encoder",
```

- [ ] **Step 10.2：GT-pose 推理（~3 min）**

```bash
python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose \
  --sana-dir   <YOUR_BASE>/Sana \
  --output-dir <YOUR_BASE>/data/sana_wm_results_gtpose \
  --sample-limit 1
```

- [ ] **Step 10.3：Default 推理（~3 min）**

```bash
python experiments/data_production_smoke/run_sana_wm_inference.py \
  --shards-dir <YOUR_BASE>/data/dl3dv_smoke_shards_default \
  --sana-dir   <YOUR_BASE>/Sana \
  --output-dir <YOUR_BASE>/data/sana_wm_results_default \
  --sample-limit 1
```

推理各阶段耗时（H100）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| DiT 60步 DDIM | ~100s | ~1.66s/step（Triton 首次编译较慢） |
| LTX-2 Refiner 3步 | ~6s | |
| VAE 解码 | ~15s | |
| 视频编码 + 指标 | ~60s | |
| **合计** | **~3 min/样本** | |

> 帧数约束：LTX-2 VAE 要求 `num_frames = 8k+1`。GT 239 帧 → 有效 233 帧 → **输出 232 帧**，属正常。

- [ ] **Step 10.4：验证输出文件**

```bash
ls <YOUR_BASE>/data/sana_wm_results_gtpose/0032cd2f*/
# 预期：first_frame.png  *_generated.mp4  *_sbs.mp4

ls <YOUR_BASE>/data/sana_wm_results_default/0032cd2f*/
# 预期：first_frame.png  *_generated.mp4  *_sbs.mp4
```

---

## Task 11：生成对比报告

- [ ] **Step 11.1：生成 Markdown 报告**

```bash
python experiments/data_production_smoke/compare_modes.py \
  --gtpose-eval  <YOUR_BASE>/data/dl3dv_smoke_shards_gtpose/eval_output/pose_eval_summary.json \
  --default-eval <YOUR_BASE>/data/dl3dv_smoke_shards_default/eval_output/pose_eval_summary.json \
  --out          <YOUR_BASE>/results/mode-comparison.md

cat <YOUR_BASE>/results/mode-comparison.md
```

- [ ] **Step 11.2：验证预期结论**

| 指标 | GT-pose | Default | 预期差距 |
|---|---|---|---|
| ATE RMSE (m) | ~1.8e-7 | ~0.128 | ~700,000× |
| PSNR (dB) | ~11.2 | ~11.3 | 相近（生成式模型，非重建）|
| SSIM | ~0.151 | ~0.154 | 相近 |
| 视频语义 | 场景内容忠实 | **可能出现幻觉**（凭空出现人物） | 视觉差异显著 |

> **Default 模式视频中出现人物属正常现象**，原因：SLAM 漂移（ATE=12.8cm）使位姿条件信号与第一帧不一致，扩散模型退回训练先验，而训练数据中人物场景占比高。

---

## 常见问题

### `ModuleNotFoundError: No module named 'timm.layers'`
timm 版本过低。执行：
```bash
pip install "timm>=0.9.0"
```

### `ModuleNotFoundError: No module named 'fla'`（重启后）
```bash
pip install flash-linear-attention
```

### `ModuleNotFoundError: No module named 'mmcv'`（重启后）
```bash
pip install "setuptools<80" && pip install --no-build-isolation mmcv==1.7.2
```

### `TypeError: ape() got an unexpected keyword argument 'verbose'`
`evo` 版本较新，已移除 `verbose` 参数。本仓库已修复（`verify_and_eval.py`），若自行编写评估脚本请移除该参数。

### ffmpeg 找不到 libx264
```bash
pip install static-ffmpeg
```
`prepare_dl3dv.py` 已自动调用 static-ffmpeg，无需额外配置。

### VIPE OOM（显存不足）
在 `mode_default.py` 中将 `chunk: int = 16` 改为 `chunk: int = 8`，以减少 Pi3X 单次推理帧数。

### Git 安全目录警告
```bash
git config --global --add safe.directory $(pwd)
```

---

## 文件结构速查

```
sana-wm-pipeline/
├── experiments/data_production_smoke/
│   ├── prepare_dl3dv.py          # DL3DV 预处理（images → video.mp4 + gt_poses.npy）
│   ├── run_e2e_gtpose.sh         # GT-pose 模式端到端（4 场景）
│   ├── run_e2e_default.sh        # Default 模式端到端（多场景）
│   ├── verify_and_eval.py        # schema check + ATE pose 评估
│   ├── run_sana_wm_inference.py  # SANA-WM 推理包装器
│   └── compare_modes.py          # 生成对比报告 Markdown
├── src/sana_wm_pipeline/
│   ├── stage01_ingest/normalize.py   # 视频归一化（1280×720 @ 16fps）
│   ├── stage02_pose/
│   │   ├── mode_gtpose.py            # GT-pose：Pi3X + Umeyama Sim(3)
│   │   └── mode_default.py           # Default：Pi3X+MoGe-2 缓存 → VIPE SLAM
│   ├── stage05_caption/              # 字幕生成（smoke 中用 stub fallback）
│   └── stage06_pack/                 # 打包为 WebDataset .tar shard
└── third_party/vipe/                 # VIPE SLAM（子模块，含 CachedDepthModel 补丁）
    └── configs/pipeline/
        ├── vipe_cached_depth.yaml    # Default 模式用的 VIPE 配置
        └── vipe_metric3d_small.yaml  # 原生 VIPE baseline 配置
```

# SANA-WM 模型推理冒烟测试（单卡 H100）

## Context

**修正先前误解**：用户的"冒烟推理"指的是用 NVlabs 官方仓 https://github.com/NVlabs/Sana 里的 `inference_video_scripts/inference_sana_wm.py` 跑 **SANA-WM 2.6B 模型本身做视频生成**，而不是数据标注管线的 e2e smoke。

**目标**：在 H100×1 上，从一张静态图 + 一条 prompt，让 SANA-WM 生成一段 ~5–10s 的 720p 视频，作为模型可用性 / 显存 / 速度的基准冒烟。

**已知事实（来自 GitHub WebFetch 调研）**：

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/NVlabs/Sana`（License: Apache-2.0） |
| 推理脚本 | `inference_video_scripts/inference_sana_wm.py`（约 1200 行） |
| 模型 HF 仓 | `Efficient-Large-Model/SANA-WM_bidirectional`（DiT 1.6B + refiner + Gemma 文本编码器 + config.yaml） |
| 环境安装 | `./environment_setup.sh sana` → conda env `sana`，Python 3.11 + CUDA 12.8 + torch 2.9.1 + xformers 0.0.33 + flash-attn ≥ 2.7 + Pi3（git） |
| 输入 | `--image RGB.jpg` + `--prompt prompt.txt` + （`--camera c2w.npy (F,4,4)` **或** `--action "w-80,jw-40"` DSL）+ 可选 `--intrinsics`（缺失时用 Pi3X 估） |
| 输出 | `{output_dir}/{name}_generated.mp4`，704×1280，H264，默认 161 帧 @ 16fps（≈10s） |
| 关键约束 | LTX-2 VAE 强制 `num_frames = 8k + 1`，脚本自动 snap |
| 显存优化 | `--no_refiner`（不下 LTX-2 17B refiner，省 ~34 GB）/ `--offload_vae` / `--offload_refiner` / `--step 20` |
| Pi3X 权重 | **已有** `/mnt/afs/davidwang/models/pi3x/`（5.1 GB） — 可直接用脚本内嵌的 `estimate_intrinsics_with_pi3x` 路径 |
| AFS 可用空间 | ~644 GB；SANA-WM DiT bf16 ≈ 3.2 GB + Gemma ≈ 2–4 GB + 新 conda env ≈ 5 GB ≈ **15 GB** 即可启动（跳过 refiner） |

**关键风险与对策**：
1. `/root` 重启清零 → 必须 `HF_HOME=/mnt/afs/davidwang/cache/huggingface`，否则 SANA-WM 权重重启丢失。
2. 现有 `sana_wm` env（torch 旧版本，用于数据管线）与官方要求的 torch 2.9.1 / cuda 12.8 不兼容 → **必须新建 conda env**，不要复用。
3. flash-attn 编译耗时 30–60 min → 第一次 setup 必须算入工时。
4. 默认 step=60 + refiner 启用时 1 段 10s 视频可能 ~10 min；smoke 用 `--no_refiner --step 20 --num_frames 33` 控制在 ~2 min。

---

## 关键文件清单

### 新建（全部在 AFS 持久路径下）

- **`/mnt/afs/davidwang/workspace/Sana/`** — git clone 仓库根目录
- **`/mnt/afs/davidwang/miniconda3/envs/sana_inference/`** — 新 conda env（不复用 `sana_wm`）
- **`/mnt/afs/davidwang/workspace/Sana_smoke/`** — smoke 测试工作目录，含：
  - `inputs/init_frame.jpg` — 1 张 720×1280 RGB 初始帧（首选复用 `/mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/` 中任一帧；或从 fr1/desk video.mp4 抽第 0 帧）
  - `inputs/prompt.txt` — 1 段静态场景描述（按论文 §4 captioning 原则：禁说"pan left / move forward"，只描述物体与布局）
  - `outputs/` — 生成视频落地
  - `smoke_run.sh` — 一键 smoke 脚本
- **`/mnt/afs/davidwang/cache/huggingface/hub/models--Efficient-Large-Model--SANA-WM_bidirectional/`** — 模型权重落盘点
- **`/mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh`** — env 守护（`HF_HOME`、`TORCH_HOME`、`SANA_WM_PI3X_WEIGHTS`、`CUDA_VISIBLE_DEVICES`）

### 复用现有

- `/mnt/afs/davidwang/models/pi3x/model.safetensors` — 内参估计用
- `/mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4` — 抽首帧用
- `/mnt/afs/davidwang/cache/huggingface/hub/` 已有缓存（GeoCalib / SAM / AOT / BERT 等不会被这里用到，但 HF cache 目录结构已就位）

---

## 实施步骤

### Step 1：clone 官方仓 + 创建独立 conda env（~45 min，含 flash-attn 编译）

```bash
# 1.1 clone
cd /mnt/afs/davidwang/workspace
git clone https://github.com/NVlabs/Sana.git Sana
cd Sana
git log -1 --oneline   # 记录 commit hash 留档

# 1.2 用官方脚本创建新 env，名字定为 sana_inference 避开既有 sana_wm
# environment_setup.sh 内部会:
#   conda create -n $1 python=3.11 cuda-toolkit=12.8 -c nvidia
#   pip install torch==2.9.1 ... xformers==0.0.33.post2 (cu128 index)
#   pip install mmcv==1.7.2 --no-build-isolation
#   pip install flash-attn --no-build-isolation
#   pip install git+https://github.com/yyfz/Pi3.git --no-deps
#   pip install -e .
bash environment_setup.sh sana_inference 2>&1 | tee /tmp/sana_env.log
```

**预期产出**：`/mnt/afs/davidwang/miniconda3/envs/sana_inference/` 存在；末尾无报错。`pip list | grep -E "torch|xformers|flash|pi3"` 全部命中。

**已知坑**：flash-attn 编译约 30–60 min；mmcv 编译约 5 min；网络抖动可能导致 git+ 安装失败 → 失败时重跑该 pip 行即可。

### Step 2：写 `env_guard.sh`（让 HF 缓存落 AFS）

```bash
mkdir -p /mnt/afs/davidwang/workspace/Sana_smoke
cat > /mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh <<'EOF'
#!/usr/bin/env bash
# Source before any SANA-WM inference run.  Idempotent.
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_inference
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export CUDA_VISIBLE_DEVICES=0
echo "[env_guard] sana_inference env + HF_HOME=$HF_HOME ready"
EOF
chmod +x /mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh
```

### Step 3：预下载 SANA-WM 模型权重（**显式落 AFS**，~10–15 min）

```bash
source /mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh

# 只下 DiT 1.6B + config（先不下 LTX-2 refiner 与 Gemma — smoke 用 --no_refiner）
# 注：DiT bf16 ≈ 3.2 GB，全量 fp32 ≈ 6.4 GB
huggingface-cli download Efficient-Large-Model/SANA-WM_bidirectional \
  --include "config.yaml" "dit/sana_wm_1600m_720p.safetensors" \
  --local-dir-use-symlinks False
# HF_HOME 已设，权重会落 /mnt/afs/davidwang/cache/huggingface/hub/...

# 验证文件存在
find /mnt/afs/davidwang/cache/huggingface/hub -name "sana_wm_1600m_720p.safetensors" -exec ls -lh {} \;
```

**预期**：safetensors 文件大小约 3.2 GB（bf16），不能为空。

### Step 4：准备 smoke 输入（image + prompt + 用 action DSL 避开 camera npy，~2 min）

```bash
source /mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh
mkdir -p /mnt/afs/davidwang/workspace/Sana_smoke/{inputs,outputs}

# 4.1 从 fr1/desk video.mp4 抽第一帧（已知存在的真实素材）
ffmpeg -y -i /mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
  -vf "select=eq(n\,0)" -vframes 1 \
  /mnt/afs/davidwang/workspace/Sana_smoke/inputs/init_frame.jpg

# 4.2 写一段静态场景 caption（论文 §4 风格：禁 camera motion 词汇）
cat > /mnt/afs/davidwang/workspace/Sana_smoke/inputs/prompt.txt <<'EOF'
A cluttered office desk with several books, a computer monitor, a desk lamp,
and various stationery items arranged on a wooden surface, under warm indoor lighting.
EOF
```

**关于轨迹**：脚本支持 `--action "w-160"` DSL（前进 160 步），免做 camera npy。这是 smoke 最简方案。

### Step 5：写 `smoke_run.sh` 一键脚本

```bash
cat > /mnt/afs/davidwang/workspace/Sana_smoke/smoke_run.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh
cd /mnt/afs/davidwang/workspace/Sana
WORK=/mnt/afs/davidwang/workspace/Sana_smoke

# Smoke 配置（关闭 refiner + 低步数 + 短帧数）：
#   - 33 帧 = 8×4+1，约 2s @ 16fps（满足 LTX-2 VAE 8k+1 约束）
#   - step=20（默认 60），不影响产出有效性，仅画质差
#   - no_refiner：跳过 LTX-2 17B
#   - offload_vae：encode/decode 之间 CPU offload，省 ~6 GB 显存
#   - action "w-32"：前进 32 步（与 num_frames 量级匹配）
python inference_video_scripts/inference_sana_wm.py \
    --image "$WORK/inputs/init_frame.jpg" \
    --prompt "$WORK/inputs/prompt.txt" \
    --output_dir "$WORK/outputs" \
    --name smoke_v1 \
    --action "w-32" \
    --num_frames 33 \
    --fps 16 \
    --step 20 \
    --cfg_scale 5.0 \
    --no_refiner \
    --offload_vae \
    --seed 42 \
    2>&1 | tee "$WORK/outputs/smoke_v1.log"

# PASS 判定：mp4 存在且 > 100 KB
OUT="$WORK/outputs/smoke_v1_generated.mp4"
if [[ ! -f "$OUT" ]]; then
  echo "[smoke] FAIL: $OUT missing"; exit 1
fi
SIZE=$(stat -c %s "$OUT")
if [[ "$SIZE" -lt 102400 ]]; then
  echo "[smoke] FAIL: $OUT only $SIZE bytes"; exit 1
fi
# 用 ffprobe 校验帧数与分辨率
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames,width,height \
  -of default=nokey=0 "$OUT"
echo "[smoke] PASS: $OUT ($((SIZE/1024)) KB)"
EOF
chmod +x /mnt/afs/davidwang/workspace/Sana_smoke/smoke_run.sh
```

**预期产出**：
- `outputs/smoke_v1_generated.mp4`，>100 KB
- ffprobe 输出 `nb_read_frames=33`、`width=1280`、`height=704`
- 总耗时 ~3–5 min（含 DiT/Gemma 首次加载 ~1 min + 推理 ~2 min）
- 峰值显存：< 35 GB（offload_vae 状态）

### Step 6：跑 smoke

```bash
bash /mnt/afs/davidwang/workspace/Sana_smoke/smoke_run.sh
```

### Step 7：观察日志关键信号

`outputs/smoke_v1.log` 应当出现：
- `LTX-2 VAE requires num_frames = 8k+1` 警告（若 33 正好满足则无此告警）
- Pi3X 加载耗时（~10s）— 因为我们没传 intrinsics，会触发 Pi3X 估计
- DiT 模型加载耗时（~30s）
- 20 步 flow_euler_ltx 采样进度条
- VAE decode 耗时（offload_vae 启用时会显示 CPU↔GPU 拷贝）
- 末尾 `[smoke] PASS: ... KB`

### Step 8：把整套提交到 sana_wm_pipeline 仓（可选，便于复现）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
mkdir -p experiments/sana_wm_inference_smoke
cp /mnt/afs/davidwang/workspace/Sana_smoke/env_guard.sh experiments/sana_wm_inference_smoke/
cp /mnt/afs/davidwang/workspace/Sana_smoke/smoke_run.sh experiments/sana_wm_inference_smoke/
cp /mnt/afs/davidwang/workspace/Sana_smoke/inputs/prompt.txt experiments/sana_wm_inference_smoke/inputs_prompt.txt
git add experiments/sana_wm_inference_smoke/
git commit -m "exp: SANA-WM 1.6B model inference smoke on H100 (33 frames, no refiner)"
```

---

## 验证

```bash
# 端到端一条龙（机器重启后亦可）
bash /mnt/afs/davidwang/workspace/Sana_smoke/smoke_run.sh

# 视频可视化校验（如果可投屏 / scp 到本地）
ls -lh /mnt/afs/davidwang/workspace/Sana_smoke/outputs/smoke_v1_generated.mp4
ffprobe -v error -show_format /mnt/afs/davidwang/workspace/Sana_smoke/outputs/smoke_v1_generated.mp4

# 显存峰值（在另一终端运行）
nvidia-smi -lms 500 --query-gpu=memory.used --format=csv,noheader -i 0
```

**整体 PASS 标准**：
1. `smoke_v1_generated.mp4` 存在，size > 100 KB
2. ffprobe 报告 33 frames × 1280×704
3. nvidia-smi 显示峰值显存 < 40 GB（H100 80 GB 之内有大量余量）
4. 日志末尾 `[smoke] PASS`

---

## 升级路径（不在本计划，留给后续）

跑通 step=20 / 33 帧 / no_refiner 的 minimal smoke 之后，下一档：

| 升级 | 命令变更 | 期望耗时 | 期望产出 |
|---|---|---|---|
| 标准画质 | `--step 60`（不变其他） | ~10 min | 同分辨率，质量提升 |
| 标准时长 | `--num_frames 161` (~10s) | ~15 min | 论文同长视频 |
| 启用 refiner | 去掉 `--no_refiner`，加 `--offload_refiner` | +20 min + 34 GB 下载 | 论文 + refiner 的最佳质量 |
| 真实 camera 轨迹 | 把 `--action` 换成 `--camera trajectory.npy` | 同上 | 可控相机运动 |
| 60s 长视频 | `--num_frames 961` | ~1.5 h | 论文主张的 1-min 生成 |

---

## 不在本次计划内（明确排除）

- 不下载 LTX-2 17B refiner（smoke 用 `--no_refiner`）
- 不准备 camera npy（用 action DSL `"w-32"` 替代）
- 不修改 NVlabs/Sana 源码（任何 bug 由后续 issue 跟进）
- 不与数据管线 e2e_smoke.sh 混合（那是另一回事）
- 不复用 `sana_wm` conda env（torch 版本冲突）

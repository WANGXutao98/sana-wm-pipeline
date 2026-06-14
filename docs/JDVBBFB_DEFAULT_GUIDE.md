# jdvbbfb-v3-full Default 模式运行指南

数据集：`junchaoh-cs/jdvbbfb-v3-full`（Hugging Face，gated: manual，已授权）
摄取层代码：`src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py`

---

## 数据集结构

8 个 WebDataset 子集，合计 475,954 样本 / 1,353 shards：

| 子集 | samples | shards | 分辨率 | fps |
|------|--------:|-------:|--------|-----|
| wds-DL3DV-ALL-2K | 9,993 | 87 | 1920×1080 | 30 |
| wds-OmniWorld-Game | 6,576 | 65 | 1280×720 | 24 |
| wds-SpatialVID-hq | 365,362 | 714 | 1280×720 | ~59.94 |
| wds-RealEstate10K-360p | 73,165 | 143 | 640×360 | 30 |
| wds-sekai-real-walking-hq | 18,208 | 287 | 1280×720 | 30 |
| wds-sekai-game-walking | 1,618 | 43 | 1920×1080 | 30 |
| wds-sekai-game-drone | 932 | 5 | 1920×1080 | 30 |
| wds-Context-as-Memory | 100 | 9 | 640×360 | 30 |

每个样本 = `{key}.mp4` + `{key}.camera.npz`（`json_members_in_shards=false`，caption 在 `index.jsonl`）。

### camera.npz schema（per_frame_camera_npz_v1，实测）

| 字段 | shape | 说明 |
|------|-------|------|
| `c2w` | (T,4,4) float32 | GT camera-to-world，opencv 约定 |
| `K_px` | (T,4) float32 | [fx,fy,cx,cy]，原始分辨率像素 |
| `fps` | scalar | 原始帧率 |
| `width/height` | scalar | 原始分辨率 |
| `vipe_c2w` | (T,4,4) float32 | 已跑好的 VIPE 参考位姿 |

---

## H100 单机运行（开发 / 验证）

### 前置条件

```bash
# HF token（gated 数据集，需手动申请授权）
export HF_HOME=/mnt/afs/davidwang/cache/huggingface   # token 在 AFS，重启不丢

# 权重
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export DISABLE_XFORMERS=1
export VIPE_EXT_JIT=1
```

### Step 1：准备 scene 目录（HF 流式，取 1 个样本）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm

python experiments/data_production_smoke/prepare_jdvbbfb.py \
  --repo junchaoh-cs/jdvbbfb-v3-full \
  --group wds-DL3DV-ALL-2K --shard-idx 0 --sample-limit 1 \
  --out-base /mnt/afs/davidwang/workspace/data/jdvbbfb_smoke
```

产物（scene 目录 6 个文件）：
- `video.mp4` — 原始 RGB 视频
- `gt_poses.npy` — GT c2w (T,4,4) float32，ATE 评估用
- `gt_intrinsics.npy` — GT K_px (T,4) float32
- `orig_fps.txt` — 原始帧率（verify_and_eval 下采样到 16fps 用）
- `caption.txt` — 文本描述
- `vipe_ref_poses.npy` — 数据集预计算的 VIPE 参考位姿

### Step 2：完整 default E2E（normalize → Pi3X+MoGe-2+VIPE → pack → eval）

```bash
bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh wds-DL3DV-ALL-2K 0
# 约 20-30 分钟（首次 VIPE JIT ~2min + Pi3X ~10min + SLAM ~10min）
```

CMCC 本地模式（数据已在 externalstorage，无需 HF token）：

```bash
export JDVBBFB_LOCAL_ROOT=/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full
bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh wds-DL3DV-ALL-2K 0 /root/work/<hotdisk>/jdvbbfb_out
```

---

## CMCC 大规模处理

> 数据集已在 CMCC externalstorage 就位：`/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full/`（持久，重启不丢）。

### 打包流程（H100 → CMCC）

打包内容（在 H100 执行）：
- **env**：`conda-pack -p sana_wm-cmcc` → `sana_wm-cmcc.tar.gz`
- **代码**：`tar --dereference` 打包 `sana_wm_pipeline/`（`--exclude '*/data'`）
- **权重**：`pi3x/` + `moge2/` + `cache/torch/hub/`（VIPE/SLAM 六件套）+ `bert-base-uncased`
- **传输**：`transfer_via_modelscope.sh upload davidxwang/conda-cmcc <tarball>`

详细步骤见计划文件：`docs/superpowers/plans/2026-06-14-jdvbbfb-default-mode-adaptation.md`（CMCC 执行步骤 A/B/C 节）。

### CMCC 单样本验证

```bash
source $NEW_BASE/start_env.sh
export JDVBBFB_LOCAL_ROOT=/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full
export TORCH_HOME=$NEW_BASE/cache/torch
export HF_HOME=$NEW_BASE/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=$NEW_BASE/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=$NEW_BASE/models/moge2
export DISABLE_XFORMERS=1 VIPE_EXT_JIT=1

cd $NEW_BASE/workspace/sana_wm_pipeline
bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh \
  wds-DL3DV-ALL-2K 0 $NEW_BASE/work/jdvbbfb_out
```

### CMCC 批量处理（多卡并行）

```bash
# 1. 批量 prepare（本地读全量 shard）
GROUP=wds-DL3DV-ALL-2K
N_SHARDS=$(ls $JDVBBFB_LOCAL_ROOT/$GROUP/shards/*.tar | wc -l)
for i in $(seq 0 $((N_SHARDS-1))); do
  python experiments/data_production_smoke/prepare_jdvbbfb.py \
    --local-root "$JDVBBFB_LOCAL_ROOT" --group "$GROUP" \
    --shard-idx "$i" --sample-limit 0 --out-base "$OUT"
done

# 2. 多卡并行 default 标注（8 卡示例，按 GPU 轮转分配 scene）
mapfile -t SCENES < <(find $OUT -mindepth 1 -maxdepth 1 -type d | sort)
for gpu in $(seq 0 7); do
  ( for idx in "${!SCENES[@]}"; do
      [ $((idx % 8)) -ne $gpu ] && continue
      CUDA_VISIBLE_DEVICES=$gpu python -c "
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from sana_wm_pipeline.stage02_pose.mode_default import run_default
import json
sc=Path('${SCENES[$idx]}')
nv=sc/'normalized.mp4'
if not nv.exists(): normalize_video(sc/'video.mp4', nv)
wd=sc/'vipe_work_default'; wd.mkdir(exist_ok=True)
art=run_default(nv,wd)
(wd/'pose_artifact_default.json').write_text(json.dumps({
    'poses_c2w':art.poses_c2w.tolist(),
    'intrinsics':art.intrinsics.tolist(),
    'scale_per_frame':art.scale_per_frame.tolist()}))
"
    done ) &
done
wait

# 3. 定期 rsync 产出到持久盘
rsync -a $OUT/shards_default/ \
  /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_default_out/$GROUP/
```

---

## 关键注意事项

- **HF token**：默认写 `/root/.cache/huggingface/token`，`/root` 重启丢失。始终用 `export HF_HOME=/mnt/afs/davidwang/cache/huggingface` 或手动 cp 到 AFS。
- **热盘 vs 持久盘**：工作目录放热盘（速度），shard 产出定期 rsync 到 `externalstorage`（持久）。
- **CMCC env**：不能用 `conda activate`，必须 `source bin/activate`（conda-pack 解压后）。
- **ATE 期望值**：DL3DV default 模式约 ~127.7mm（无 GT depth 约束），量级合理即通过。

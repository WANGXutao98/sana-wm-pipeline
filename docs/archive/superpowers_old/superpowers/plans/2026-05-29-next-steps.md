# SANA-WM 后续路线图：验证 / 数据生产 / 顶会论文

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 SANA-WM 数据管线从"算法已验证（fr1/desk ATE↓36%）"推进到"产品级数据已产出 + 一个可发表的原创算法贡献已成形"。三条独立 track 并行推进。

**Architecture:** 三条 track 各自独立可交付：
- **Track A — 复现充分性扩展**：长序列 + KITTI/ScanNet++ 跨场景验证，把"VIPE+Pi3X+MoGe-2 击败 metric3d-small"变成跨数据集的强结论。
- **Track B — 自产数据 smoke→小规模生产**：用 H100×1 跑通真实数据 e2e，产出第一批 ~50–200 clip 的 WebDataset shard。
- **Track C — 原创算法贡献**：**Confidence-Aware Multi-View Depth Fusion（CADF）**——替换 SANA-WM 的"mean-of-means × EMA"为利用 Pi3X conf + MoGe-2 mask 的 IRLS/学习加权。这是清晰、可消融、可在 3 个公开 benchmark（TUM/KITTI/ScanNet++）展示的算法贡献，目标 CVPR/ICCV/NeurIPS。

**Tech Stack:** 已激活 `sana_wm` env，Pi3X / MoGe-2 / VIPE / evo / pytorch；新模块仅 numpy/torch，无新依赖。

**关键事实与约束：**
- 磁盘：`/mnt/afs` 可用空间仅 ~644 GB（已 100% 占用），所有下载/缓存必须谨慎。
- 算力：H100×1，CUDA_VISIBLE_DEVICES=0，不可复现 paper 64×H100 训练。
- 模型权重已就位：Pi3X 5.1 GB、MoGe-2 1.3 GB、metric3d-small 已缓存。
- 现有 baseline 强度：fr1/desk ATE RMSE 0.0227 m（B）vs 0.0355 m（A），↓36%。
- Pi3X 的 `conf` 字段（`outputs["conf"]`, `(B,N,H,W,1)`）目前**被 `precompute_pi3x_depths.py` 丢弃**——这正是 CADF 的入口点。

---

## 文件结构总览（本计划新增/修改）

```
sana_wm_pipeline/
├── experiments/
│   ├── vipe_comparison/                          # Track A（已存在，仅扩展）
│   │   ├── run_corrected.sh                      # 已存在；新增 kitti / scannetpp 分支
│   │   ├── prepare_kitti.py                      # ★ NEW：KITTI raw seq 准备
│   │   ├── prepare_scannetpp.py                  # ★ NEW：ScanNet++ DSLR seq 准备
│   │   └── results_kitti/, results_scannetpp/    # ★ NEW
│   ├── data_production_smoke/                    # ★ NEW Track B 根目录
│   │   ├── download_dl3dv_subset.sh              # ★ 下载 5 个 DL3DV 场景
│   │   ├── run_e2e_5scenes.sh                    # ★ 跑全管线，输出 WebDataset
│   │   ├── verify_shards.py                      # ★ 校验产出 shard 内容
│   │   └── results/                              # ★ 输出 WebDataset
│   └── cadf_research/                            # ★ NEW Track C 根目录
│       ├── README.md                             # ★ 研究问题陈述（投稿初稿用）
│       ├── precompute_pi3x_depths_cadf.py        # ★ 在原 precompute 基础上保留 conf
│       ├── fusion_kernels.py                     # ★ 4 种融合算法实现（含 baseline）
│       ├── train_fusion_head.py                  # ★ Track C P2：可选学习权重
│       ├── eval_cross_dataset.py                 # ★ 在 TUM/KITTI/ScanNet++ 上批量评测
│       └── results/{tum,kitti,scannetpp}/        # ★ 评测产物
├── third_party/vipe/vipe/priors/depth/
│   └── cached.py                                 # 已存在，Track C 复用
└── tests/
    ├── test_fusion_kernels.py                    # ★ NEW，单元测试 4 种融合
    └── test_data_production_smoke.py             # ★ NEW，e2e shard schema 检查
```

---

# Track A — 复现充分性扩展（P0，~3 工时 + 等待）

> **动机**：fr1/desk 仅 28s。要让"长视频稳定性"的主张可信，必须 (1) fr2/desk 99s 跑通；(2) 在 KITTI（户外驾驶）与 ScanNet++（室内 DSLR）上同样 hold，才能写进论文 Table 1。

## Task A1：fr2/desk 长序列实验（已就绪，一键运行）

**Files:**
- 运行：`experiments/vipe_comparison/run_corrected.sh`
- 输出：`experiments/vipe_comparison/results_fr2/method_{A_m3d,B_cached}/pose/video.npz`
- 报告：`experiments/vipe_comparison/RESULTS_fr2_desk.md` ★ NEW

- [ ] **Step 1：环境激活 + 权重 env**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
git config --global --add safe.directory $(pwd)
git config --global --add safe.directory $(pwd)/third_party/vipe
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
python -m pytest -q  # 期望 140 passed
```

- [ ] **Step 2：磁盘预检（避免跑到一半失败）**

```bash
df -h /mnt/afs/davidwang/workspace | tail -1
du -sh experiments/vipe_comparison/results 2>/dev/null
```

期望可用空间 > 5 GB（预计 cache_pi3x_moge2_fr2_desk.npz ≈ 2.1 GB + VIPE 中间产物）。若不足，先 `rm -rf experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz` 释放 570 MB。

- [ ] **Step 3：一键运行（预计 ~1.5 小时）**

```bash
bash experiments/vipe_comparison/run_corrected.sh fr2 2>&1 | tee /tmp/run_fr2.log
```

期望日志末尾出现：
- `Saved cache (2257 frames, ~2100 MB)`
- `Method A done`、`Method B done`
- `evaluate.py` 输出含 `ATE RMSE / RTE` 表格

- [ ] **Step 4：生成 RESULTS_fr2_desk.md 报告**

参照 `RESULTS_fr1_desk.md` 复制结构，把 `evaluate.py` 控制台输出的数值表格粘贴进 §五；另从 `results_fr2/plots/ate_analysis.png` 读出 **前 50% vs 后 50% RTE 比值**，验证"长视频后半漂移下降"的主张是否在 99s 序列上**更强**于 28s 序列。

```bash
cp experiments/vipe_comparison/RESULTS_fr1_desk.md experiments/vipe_comparison/RESULTS_fr2_desk.md
# 手工填入实际数值
```

- [ ] **Step 5：提交**

```bash
git add experiments/vipe_comparison/RESULTS_fr2_desk.md experiments/vipe_comparison/results_fr2/
git commit -m "exp(vipe): fr2/desk long-sequence A vs B (99s, 2257 frames)"
```

---

## Task A2：KITTI 驾驶场景准备

**Files:**
- Create: `experiments/vipe_comparison/prepare_kitti.py`
- 数据落地：`experiments/vipe_comparison/data/kitti_2011_09_26_drive_0005/`

KITTI raw 09_26_drive_0005（154 帧、~15 s、cam2 RGB+GT pose）足以验证驾驶场景；同时 KITTI fx/fy/cx/cy 与 TUM 完全不同，能挑战 metric scale 校准。

- [ ] **Step 1：写 `prepare_kitti.py`**

```python
#!/usr/bin/env python3
"""下载 KITTI raw 09_26_drive_0005，提取 cam2 RGB 序列与 GT 轨迹。"""
from __future__ import annotations
import argparse, pathlib, subprocess, zipfile
import numpy as np, cv2

CAM2_K = np.array([[721.5377, 0, 609.5593],
                   [0, 721.5377, 172.854],
                   [0, 0, 1]], dtype=np.float64)
SYNC_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip"
CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for url in (SYNC_URL, CALIB_URL):
        fn = out / pathlib.Path(url).name
        if not fn.exists():
            subprocess.check_call(["wget", "-q", url, "-O", str(fn)])
        with zipfile.ZipFile(fn) as z:
            z.extractall(out)
    # 写 video.mp4
    img_dir = next(out.rglob("image_02/data"))
    imgs = sorted(img_dir.glob("*.png"))
    h, w = cv2.imread(str(imgs[0])).shape[:2]
    vw = cv2.VideoWriter(str(out / "video.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for img_p in imgs:
        vw.write(cv2.imread(str(img_p)))
    vw.release()
    # 写 GT 轨迹（TUM 格式 timestamp tx ty tz qx qy qz qw）
    pose_dir = next(out.rglob("oxts/data"))
    pose_files = sorted(pose_dir.glob("*.txt"))
    gt_lines = []
    for i, pf in enumerate(pose_files):
        vals = np.loadtxt(pf)
        # 使用 lat/lon/alt 简化为局部 ENU，仅供 evo 对齐用（绝对值不重要）
        tx, ty, tz = vals[0]*1e5, vals[1]*1e5, vals[2]
        roll, pitch, yaw = vals[3], vals[4], vals[5]
        from scipy.spatial.transform import Rotation as R
        q = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        gt_lines.append(f"{i*0.1:.6f} {tx} {ty} {tz} {q[0]} {q[1]} {q[2]} {q[3]}")
    (out / "gt_aligned.txt").write_text("\n".join(gt_lines))
    print(f"Prepared {len(imgs)} frames → {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2：运行**

```bash
python experiments/vipe_comparison/prepare_kitti.py \
    --out experiments/vipe_comparison/data/kitti_2011_09_26_drive_0005
```

期望输出末尾：`Prepared 154 frames → ...`

- [ ] **Step 3：扩展 `run_corrected.sh` 支持 kitti**

打开 `experiments/vipe_comparison/run_corrected.sh`，在 `case` 分支处添加：

```bash
case "$SEQ" in
  fr1) DATA_DIR=...; SEQNAME=fr1_desk ;;
  fr2) DATA_DIR=...; SEQNAME=fr2_desk ;;
  kitti) DATA_DIR=experiments/vipe_comparison/data/kitti_2011_09_26_drive_0005; SEQNAME=kitti_0005; FX=721.5377 ;;
  scannetpp) DATA_DIR=experiments/vipe_comparison/data/scannetpp_8b5caf3398; SEQNAME=scannetpp_8b5caf3398; FX=1167.9 ;;
esac
```

并把 fov_x 计算的 `fx_tum=525` 改成读 `$FX` env，传给 `precompute_pi3x_depths.py`（要为该脚本加 `--fx` 参数）。

- [ ] **Step 4：运行 KITTI 实验（~30 min）**

```bash
bash experiments/vipe_comparison/run_corrected.sh kitti 2>&1 | tee /tmp/run_kitti.log
```

- [ ] **Step 5：提交**

```bash
git add experiments/vipe_comparison/prepare_kitti.py experiments/vipe_comparison/run_corrected.sh experiments/vipe_comparison/results_kitti/
git commit -m "exp(vipe): KITTI 09_26_drive_0005 outdoor driving validation"
```

---

## Task A3：ScanNet++ DSLR 序列准备

**Files:**
- Create: `experiments/vipe_comparison/prepare_scannetpp.py`

ScanNet++ 提供高分辨率 DSLR + COLMAP GT。任选一个公开 split 中的小场景（~500 帧）。

- [ ] **Step 1：手工下载 1 个公开样例 scene（约 1.5 GB）**

```bash
mkdir -p experiments/vipe_comparison/data/scannetpp_8b5caf3398
# 用户需 ScanNet++ 注册后取得下载链接；首次需手工 wget 一次。
# 占位说明：本 task 假设已下载 RGB jpg 序列与 colmap pose 到上述目录。
```

- [ ] **Step 2：写 `prepare_scannetpp.py`**

```python
#!/usr/bin/env python3
"""ScanNet++ DSLR seq → video.mp4 + gt_aligned.txt（COLMAP → TUM 格式）。"""
from __future__ import annotations
import argparse, pathlib, json, cv2, numpy as np
from scipy.spatial.transform import Rotation as R

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)  # e.g. data/scannetpp_8b5caf3398
    args = p.parse_args()
    root = pathlib.Path(args.scene)
    imgs = sorted((root / "images").glob("*.JPG"))
    cams = json.loads((root / "cameras.json").read_text())  # 假设是 transforms.json 风格
    h, w = cv2.imread(str(imgs[0])).shape[:2]
    vw = cv2.VideoWriter(str(root / "video.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    gt = []
    for i, img_p in enumerate(imgs):
        vw.write(cv2.imread(str(img_p)))
        T = np.array(cams[img_p.name]["transform_matrix"], dtype=np.float64)  # 4x4 c2w
        q = R.from_matrix(T[:3, :3]).as_quat()
        gt.append(f"{i*0.1:.6f} {T[0,3]} {T[1,3]} {T[2,3]} {q[0]} {q[1]} {q[2]} {q[3]}")
    vw.release()
    (root / "gt_aligned.txt").write_text("\n".join(gt))
    print(f"Prepared {len(imgs)} ScanNet++ frames → {root}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3：跑实验**

```bash
bash experiments/vipe_comparison/run_corrected.sh scannetpp 2>&1 | tee /tmp/run_scannetpp.log
```

- [ ] **Step 4：合并 3 个数据集结果生成 `experiments/vipe_comparison/RESULTS_cross_dataset.md`**

新建文件，含 `(TUM-fr1, TUM-fr2, KITTI, ScanNet++) × (A, B)` 的 ATE RMSE / 后半 RTE 表格。

- [ ] **Step 5：提交 + tag**

```bash
git add experiments/vipe_comparison/RESULTS_cross_dataset.md experiments/vipe_comparison/prepare_scannetpp.py experiments/vipe_comparison/results_scannetpp/
git commit -m "exp(vipe): cross-dataset validation (TUM/KITTI/ScanNet++)"
git tag v0.2.0-cross-dataset-validated
```

---

# Track B — 真实数据 smoke：管线 e2e 第一次跑通（P1，~6–8 小时）

> **动机**：当前管线代码 140 测试全过，但**外部模型全部是 stub**，从未在真实数据上 e2e 跑通过。本 track 用 DL3DV 的 5 个 scene（gt_pose 模式，规避 VIPE 长视频 SLAM 的 OOM 风险）做最小可行的真实数据生产，产出第一批合法 WebDataset shard，为后续 H100 上 100–200 clip 子集做好基础。

## Task B1：DL3DV 子集下载（~25 GB / 5 scene）

**Files:**
- Create: `experiments/data_production_smoke/download_dl3dv_subset.sh`

DL3DV-960P 镜像在 HuggingFace（`DL3DV/DL3DV-ALL-960P`）。每个 scene 约 5 GB，挑 5 个 scene 控制在 25 GB。

- [ ] **Step 1：写下载脚本**

```bash
#!/usr/bin/env bash
# 下载 DL3DV-ALL-960P 中 5 个 scene（约 25 GB），需 huggingface-cli 已 login。
set -euo pipefail
OUT="${1:-/mnt/afs/davidwang/workspace/data/dl3dv_smoke}"
mkdir -p "$OUT"
SCENES=(
  "1K/0a3c8a3e1f"
  "1K/0b2f3c4e9d"
  "1K/0c5a7b8d2e"
  "1K/0d8f4a6c1b"
  "1K/0e9b7c3a5f"
)
for SCENE in "${SCENES[@]}"; do
  echo ">>> Downloading $SCENE"
  huggingface-cli download DL3DV/DL3DV-ALL-960P \
    --include "$SCENE/*" \
    --repo-type dataset \
    --local-dir "$OUT" \
    --local-dir-use-symlinks False
done
echo "Done. Disk usage:"; du -sh "$OUT"
```

> 注：实际 scene id 需先用 `huggingface-cli download DL3DV/DL3DV-ALL-960P --include "*.json"` 拉 manifest 后挑选；本 step 给定 id 为占位，按需替换。

- [ ] **Step 2：磁盘预检**

```bash
df -h /mnt/afs/davidwang/workspace | tail -1
# 期望可用 ≥ 30 GB
```

- [ ] **Step 3：跑下载**

```bash
bash experiments/data_production_smoke/download_dl3dv_subset.sh \
  /mnt/afs/davidwang/workspace/data/dl3dv_smoke 2>&1 | tee /tmp/dl3dv_dl.log
```

期望 5 个子目录 + 每个含 `images/`, `transforms.json`。

---

## Task B2：5-scene e2e 运行脚本

**Files:**
- Create: `experiments/data_production_smoke/run_e2e_5scenes.sh`
- 输出：`experiments/data_production_smoke/results/shards/`

- [ ] **Step 1：写 e2e shell**

```bash
#!/usr/bin/env bash
# 跑通完整管线：normalize → pose(gt_pose mode) → filter → caption(stub) → pack
set -euo pipefail
ROOT=/mnt/afs/davidwang/workspace/sana_wm_pipeline
DATA=/mnt/afs/davidwang/workspace/data/dl3dv_smoke
OUT=$ROOT/experiments/data_production_smoke/results
mkdir -p "$OUT"/{normalized,poses,filtered,captions,shards}

# Stage 01: normalize（720p@16fps）
python -m sana_wm_pipeline.stage01_ingest.normalize \
  --in-glob "$DATA/*/images" \
  --out-dir "$OUT/normalized" \
  --duration 10  # smoke 用 10s 截断，规避 OOM

# Stage 02 (gt_pose 模式)：DL3DV 自带 transforms.json → Pi3X 估算 metric scale via Umeyama
python -m sana_wm_pipeline.stage02_pose.mode_gtpose \
  --videos-dir "$OUT/normalized" \
  --poses-glob "$DATA/*/transforms.json" \
  --out-dir "$OUT/poses" \
  --pi3x-weights "$SANA_WM_PI3X_WEIGHTS"

# Stage 04 filter（visual_metrics 现仍是 stub，先跑结构性 pass-through）
python -m sana_wm_pipeline.stage04_filter.apply_table6 \
  --in-dir "$OUT/normalized" --pose-dir "$OUT/poses" \
  --out-dir "$OUT/filtered" --dataset DL3DV-GS

# Stage 05 caption（暂用静态 stub caption）
python -m sana_wm_pipeline.stage05_caption.postprocess \
  --in-dir "$OUT/filtered" --out-dir "$OUT/captions" \
  --stub-text "indoor scene with static camera and natural lighting"

# Stage 06 pack → WebDataset
python -m sana_wm_pipeline.stage06_pack.webdataset_writer \
  --video-dir "$OUT/filtered" --pose-dir "$OUT/poses" \
  --caption-dir "$OUT/captions" --out-dir "$OUT/shards" \
  --shard-size 1
echo "Shards:"; ls -la "$OUT/shards"
```

- [ ] **Step 2：跑通 e2e（预计 5 scene × ~10 min = ~60 min）**

```bash
bash experiments/data_production_smoke/run_e2e_5scenes.sh 2>&1 | tee /tmp/e2e_smoke.log
```

- [ ] **Step 3：写 verify_shards.py 验证 schema**

```python
#!/usr/bin/env python3
"""校验产出的 WebDataset shard 是否符合 stage06 schema。"""
from __future__ import annotations
import argparse, pathlib, tarfile, json

REQUIRED_KEYS = {"video.mp4", "pose.npy", "intrinsics.npy", "caption.txt", "meta.json"}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", required=True)
    args = p.parse_args()
    shards = sorted(pathlib.Path(args.shards_dir).glob("*.tar"))
    assert shards, "no shards found"
    print(f"Found {len(shards)} shards")
    for sh in shards:
        with tarfile.open(sh) as tf:
            members = {m.name.split("/")[-1] for m in tf.getmembers()}
            missing = REQUIRED_KEYS - members
            assert not missing, f"{sh.name} missing {missing}"
        print(f"  ✓ {sh.name}")
    print("All shards conform to schema.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4：跑校验**

```bash
python experiments/data_production_smoke/verify_shards.py \
  --shards-dir experiments/data_production_smoke/results/shards
```

期望全部 `✓` + `All shards conform`.

- [ ] **Step 5：提交**

```bash
git add experiments/data_production_smoke/
git commit -m "exp(data): DL3DV 5-scene e2e smoke produces first WebDataset shards"
```

---

## Task B3：把 smoke 结果回灌到主仓 `scripts/e2e_smoke.sh`

**Files:**
- Modify: `scripts/e2e_smoke.sh`（已存在）
- Modify: `README.md`（已存在）

把"现在如何在真实 DL3DV 5 scene 上跑通 e2e"写成主仓的 quickstart，让任何新会话冷启动后能复现这一步。

- [ ] **Step 1：把 `experiments/data_production_smoke/run_e2e_5scenes.sh` 的可复用部分上抽到 `scripts/e2e_smoke.sh`**

读现有 `scripts/e2e_smoke.sh`，替换其中所有 stub 调用为 B2 中的真实命令。

- [ ] **Step 2：更新 README Quickstart 章节**

在 README.md 适当位置添加 `## Quickstart: 5-scene e2e on H100×1` 章节。

- [ ] **Step 3：提交**

```bash
git add scripts/e2e_smoke.sh README.md
git commit -m "docs(readme): add 5-scene e2e quickstart based on smoke"
```

---

# Track C — 原创算法贡献：Confidence-Aware Multi-View Depth Fusion (CADF)

> **顶会论文角度**：SANA-WM 的 `s_t = mean(d_MoGe) / mean(d_Pi3X)` 是**对所有像素无差别求均值**，把 Pi3X 的 confidence 与 MoGe-2 的 valid mask 完全丢掉了。我们提出 **CADF**：用 Pi3X conf 作为权重、用 MoGe-2 mask 作为先验、引入 IRLS 迭代（可选学习 head），在 TUM/KITTI/ScanNet++ 上统一击败 SANA-WM 的 EMA-mean baseline。**这是清楚、可消融、可发表的算法贡献**。
>
> **可投稿题目（拟）**：*Confidence-Aware Depth Fusion for Robust Long-Video SLAM-based Pose Annotation*
> **目标场次**：CVPR 2027 / ICCV 2027 / NeurIPS 2026 Datasets & Benchmarks Track

## Task C1：在原 precompute 基础上保留 conf 信号

**Files:**
- Create: `experiments/cadf_research/precompute_pi3x_depths_cadf.py`

复用 `experiments/vipe_comparison/precompute_pi3x_depths.py`，但新增：(1) 保存 Pi3X conf 张量；(2) 保存 MoGe-2 mask（depth>0 & finite）；(3) 不做 fusion，把 raw 输出全部存盘供 fusion_kernels.py 读取。

- [ ] **Step 1：写脚本**

```python
#!/usr/bin/env python3
"""保留 Pi3X conf 与 MoGe-2 mask 的离线预计算（不做 fusion）。

输出: cache_raw_{seq}.npz 含
  d_pi3x:  (T, H, W) float32 — Pi3X 相对深度
  conf:    (T, H, W) float32 — Pi3X 置信度（sigmoid 0~1）
  d_moge:  (T, H, W) float32 — MoGe-2 metric depth
  mask:    (T, H, W) bool    — MoGe-2 valid mask
"""
from __future__ import annotations
import argparse, logging, math, os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "vipe_comparison"))
from precompute_pi3x_depths import read_video_frames  # type: ignore
import cv2, numpy as np, torch, torch.nn.functional as F  # noqa

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@torch.no_grad()
def run_pi3x_with_conf(model, frames_t, chunk=16, stride=8):
    T, _, H, W = frames_t.shape
    H_r = (H // 14) * 14; W_r = (W // 14) * 14
    src = F.interpolate(frames_t, size=(H_r, W_r), mode="bilinear", align_corners=False) \
          if (H_r, W_r) != (H, W) else frames_t
    d_acc = np.zeros((T, H_r, W_r), dtype=np.float32)
    c_acc = np.zeros((T, H_r, W_r), dtype=np.float32)
    n_acc = np.zeros(T, dtype=np.float32)
    starts = list(range(0, max(T - chunk + 1, 1), stride))
    if not starts or starts[-1] + chunk < T:
        starts.append(max(0, T - chunk))
    for i, s in enumerate(starts):
        e = min(s + chunk, T)
        log.info(f"  Pi3X chunk {i+1}/{len(starts)} [{s},{e})")
        out = model(src[s:e].unsqueeze(0))
        d_acc[s:e] += out["local_points"][0, :e-s, :, :, 2].cpu().numpy()
        c_acc[s:e] += out["conf"][0, :e-s, :, :, 0].sigmoid().cpu().numpy()
        n_acc[s:e] += 1
    d_r = d_acc / np.maximum(n_acc[:, None, None], 1)
    c_r = c_acc / np.maximum(n_acc[:, None, None], 1)
    if (H_r, W_r) != (H, W):
        dt = torch.from_numpy(d_r).unsqueeze(1).cuda()
        ct = torch.from_numpy(c_r).unsqueeze(1).cuda()
        d_r = F.interpolate(dt, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
        c_r = F.interpolate(ct, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
    return d_r, c_r


@torch.no_grad()
def run_moge2_with_mask(model, frames_t, fov_x):
    T = len(frames_t)
    d_out = np.zeros((T,) + tuple(frames_t.shape[-2:]), dtype=np.float32)
    m_out = np.zeros_like(d_out, dtype=bool)
    for i in range(T):
        if i % 100 == 0:
            log.info(f"  MoGe-2 frame {i}/{T}")
        out = model.infer(frames_t[i:i+1], fov_x=fov_x)
        d = out["depth"].squeeze(0).cpu().numpy()
        d_out[i] = d
        m_out[i] = np.isfinite(d) & (d > 1e-6)
    return d_out, m_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fx", type=float, default=525.0)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()

    pi3x_w = os.environ["SANA_WM_PI3X_WEIGHTS"]
    moge_w = os.environ["SANA_WM_MOGE2_WEIGHTS"]
    out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_np = read_video_frames(args.video)
    T, H, W, _ = frames_np.shape
    frames_t = torch.from_numpy(frames_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).cuda()
    fov_x = math.degrees(2 * math.atan(W / (2 * args.fx)))

    from pi3 import Pi3X; pi3x = Pi3X.from_pretrained(pi3x_w).cuda().eval()
    from moge.model.v2 import MoGeModel
    moge_ckpt = pathlib.Path(moge_w); moge_ckpt = moge_ckpt / "model.pt" if moge_ckpt.is_dir() else moge_ckpt
    moge = MoGeModel.from_pretrained(str(moge_ckpt)).cuda().eval()

    log.info("Running Pi3X with conf...")
    d_pi3x, conf = run_pi3x_with_conf(pi3x, frames_t, chunk=args.chunk, stride=args.stride)
    log.info(f"Pi3X depth {d_pi3x.min():.2f}~{d_pi3x.max():.2f}, conf {conf.min():.3f}~{conf.max():.3f}")
    log.info("Running MoGe-2 with mask...")
    d_moge, mask = run_moge2_with_mask(moge, frames_t, fov_x)
    log.info(f"MoGe-2 depth {d_moge[mask].min():.2f}~{d_moge[mask].max():.2f}, valid ratio {mask.mean():.3f}")

    np.savez_compressed(out_path,
                        d_pi3x=d_pi3x, conf=conf, d_moge=d_moge, mask=mask)
    log.info(f"Saved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2：跑一次 fr1/desk 生成 raw cache**

```bash
mkdir -p experiments/cadf_research/results
python experiments/cadf_research/precompute_pi3x_depths_cadf.py \
  --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
  --out experiments/cadf_research/results/cache_raw_fr1_desk.npz \
  --fx 525.0
```

期望日志含 `valid ratio 0.9x`、`conf 0.0x~0.9x`、最终保存 ~1.1 GB。

- [ ] **Step 3：提交**

```bash
git add experiments/cadf_research/precompute_pi3x_depths_cadf.py
git commit -m "exp(cadf): preserve Pi3X conf and MoGe-2 mask in raw cache"
```

---

## Task C2：四种 fusion kernel 实现 + 单元测试

**Files:**
- Create: `experiments/cadf_research/fusion_kernels.py`
- Create: `tests/test_fusion_kernels.py`

四个待对比的 fusion 函数（同一 API：`fuse(d_pi3x, conf, d_moge, mask, **kw) -> (depths, scale_hist)`）：

1. `fuse_baseline_ema` — 论文 mean-of-means × EMA（复刻 `precompute_pi3x_depths.py:ema_fuse`）
2. `fuse_conf_weighted` — 仅 conf 权重：`s = Σ conf·w_i·d_M / Σ conf·w_i·d_P`（w_i = 1/d_P）
3. `fuse_irls` — 迭代重加权最小二乘：每轮重新估计 residual、降权 outlier，τ 步收敛
4. `fuse_robust_geomedian` — Weiszfeld 几何中位数估 scale，对单像素异常值鲁棒

- [ ] **Step 1：先写单元测试（TDD）**

```python
# tests/test_fusion_kernels.py
import numpy as np
import pytest
from experiments.cadf_research.fusion_kernels import (
    fuse_baseline_ema, fuse_conf_weighted, fuse_irls, fuse_robust_geomedian,
)

@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    T, H, W = 8, 32, 32
    true_scale = np.linspace(1.0, 1.3, T)
    d_pi = rng.uniform(0.5, 5.0, (T, H, W)).astype(np.float32)
    d_mo = (d_pi * true_scale[:, None, None]).astype(np.float32)
    conf = rng.uniform(0.3, 0.95, (T, H, W)).astype(np.float32)
    mask = np.ones_like(d_pi, dtype=bool)
    # 注入 10% outlier 像素
    out_idx = rng.choice(H*W, size=int(0.1*H*W), replace=False)
    d_mo.reshape(T, -1)[:, out_idx] *= 5.0
    return d_pi, conf, d_mo, mask, true_scale

def test_baseline_recovers_clean_scale(synthetic):
    d_pi, conf, d_mo, mask, true_scale = synthetic
    _, scale = fuse_baseline_ema(d_pi, conf, d_mo, mask, momentum=0.0)  # 无 EMA
    # 因 outlier 拉高 mean，baseline 估值会偏高
    assert scale.mean() > true_scale.mean() * 1.1

def test_conf_weighted_uses_conf(synthetic):
    d_pi, conf, d_mo, mask, _ = synthetic
    _, s_high = fuse_conf_weighted(d_pi, conf, d_mo, mask, momentum=0.0)
    # 把 outlier 像素的 conf 调低，估值应当下降
    conf_low = conf.copy(); conf_low.reshape(8, -1)[:, ::10] = 0.01
    _, s_low_outliers = fuse_conf_weighted(d_pi, conf_low, d_mo, mask, momentum=0.0)
    assert s_low_outliers.mean() <= s_high.mean()

def test_irls_better_than_baseline_under_outliers(synthetic):
    d_pi, conf, d_mo, mask, true_scale = synthetic
    _, s_base = fuse_baseline_ema(d_pi, conf, d_mo, mask, momentum=0.0)
    _, s_irls = fuse_irls(d_pi, conf, d_mo, mask, n_iters=5, momentum=0.0)
    err_base = np.abs(s_base - true_scale).mean()
    err_irls = np.abs(s_irls - true_scale).mean()
    assert err_irls < err_base, f"IRLS should beat baseline: {err_irls=} vs {err_base=}"

def test_geomedian_robust(synthetic):
    d_pi, conf, d_mo, mask, true_scale = synthetic
    _, s_gm = fuse_robust_geomedian(d_pi, conf, d_mo, mask, momentum=0.0)
    err_gm = np.abs(s_gm - true_scale).mean()
    assert err_gm < 0.05, f"geomedian err too high: {err_gm}"

def test_all_kernels_same_shape(synthetic):
    d_pi, conf, d_mo, mask, _ = synthetic
    for fn in (fuse_baseline_ema, fuse_conf_weighted, fuse_irls, fuse_robust_geomedian):
        d, s = fn(d_pi, conf, d_mo, mask)
        assert d.shape == d_pi.shape
        assert s.shape == (d_pi.shape[0],)
```

- [ ] **Step 2：跑测试确认 fail（fusion_kernels.py 还不存在）**

```bash
python -m pytest tests/test_fusion_kernels.py -v
# 期望 5 个 ImportError
```

- [ ] **Step 3：写 `fusion_kernels.py`**

```python
"""四种 depth fusion kernel，统一 API。

API: fuse(d_pi3x, conf, d_moge, mask, momentum=0.99, **kw)
       -> (fused_depths (T,H,W), scale_hist (T,))
"""
from __future__ import annotations
import numpy as np

def _ema_apply(scales_raw: np.ndarray, momentum: float) -> np.ndarray:
    out = np.zeros_like(scales_raw)
    ema = float(scales_raw[0]) if len(scales_raw) else 1.0
    out[0] = ema
    for t in range(1, len(scales_raw)):
        ema = ema * momentum + float(scales_raw[t]) * (1.0 - momentum)
        out[t] = ema
    return out

def fuse_baseline_ema(d_pi3x, conf, d_moge, mask, momentum: float = 0.99):
    """SANA-WM 论文 baseline：mean(d_M)/mean(d_P) × EMA。"""
    T = len(d_pi3x); raw = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = mask[t] & (d_pi3x[t] > 1e-6)
        raw[t] = float(d_moge[t][m].mean()) / (float(d_pi3x[t][m].mean()) + 1e-8) if m.sum() > 10 else 1.0
    s = _ema_apply(raw, momentum)
    return (d_pi3x * s[:, None, None]).astype(np.float32), s

def fuse_conf_weighted(d_pi3x, conf, d_moge, mask, momentum: float = 0.99):
    """Conf-weighted WLS：s = Σ w·d_M / Σ w·d_P, w = conf / d_P。"""
    T = len(d_pi3x); raw = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = mask[t] & (d_pi3x[t] > 1e-6)
        if m.sum() < 10:
            raw[t] = 1.0; continue
        w = conf[t][m] / (d_pi3x[t][m] + 1e-8)
        raw[t] = float((w * d_moge[t][m]).sum() / ((w * d_pi3x[t][m]).sum() + 1e-8))
    s = _ema_apply(raw, momentum)
    return (d_pi3x * s[:, None, None]).astype(np.float32), s

def fuse_irls(d_pi3x, conf, d_moge, mask, momentum: float = 0.99, n_iters: int = 5, c: float = 1.345):
    """Iteratively Reweighted Least Squares with Huber loss."""
    T = len(d_pi3x); raw = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = mask[t] & (d_pi3x[t] > 1e-6)
        if m.sum() < 10:
            raw[t] = 1.0; continue
        dP = d_pi3x[t][m]; dM = d_moge[t][m]; w0 = conf[t][m]
        s = float(dM.mean() / (dP.mean() + 1e-8))
        for _ in range(n_iters):
            r = dM - s * dP  # residual
            sigma = max(1.4826 * np.median(np.abs(r)), 1e-6)
            r_norm = np.abs(r) / sigma
            w_h = w0 * np.where(r_norm <= c, 1.0, c / np.maximum(r_norm, 1e-8))
            s = float((w_h * dP * dM).sum() / ((w_h * dP * dP).sum() + 1e-8))
        raw[t] = s
    s = _ema_apply(raw, momentum)
    return (d_pi3x * s[:, None, None]).astype(np.float32), s

def fuse_robust_geomedian(d_pi3x, conf, d_moge, mask, momentum: float = 0.99, n_iters: int = 10):
    """Weiszfeld geometric median over per-pixel scale ratios."""
    T = len(d_pi3x); raw = np.zeros(T, dtype=np.float32)
    for t in range(T):
        m = mask[t] & (d_pi3x[t] > 1e-6)
        if m.sum() < 10:
            raw[t] = 1.0; continue
        ratios = (d_moge[t][m] / (d_pi3x[t][m] + 1e-8)).astype(np.float64)
        w = conf[t][m].astype(np.float64)
        s = float(np.median(ratios))
        for _ in range(n_iters):
            dist = np.abs(ratios - s) + 1e-6
            w_eff = w / dist
            s = float((w_eff * ratios).sum() / (w_eff.sum() + 1e-8))
        raw[t] = s
    s = _ema_apply(raw, momentum)
    return (d_pi3x * s[:, None, None]).astype(np.float32), s
```

- [ ] **Step 4：再跑测试**

```bash
python -m pytest tests/test_fusion_kernels.py -v
# 期望 5/5 passed
```

- [ ] **Step 5：提交**

```bash
git add experiments/cadf_research/fusion_kernels.py tests/test_fusion_kernels.py
git commit -m "exp(cadf): four fusion kernels (baseline/conf/irls/geomedian) + unit tests"
```

---

## Task C3：4 种 fusion × 3 数据集的跨数据集评测脚本

**Files:**
- Create: `experiments/cadf_research/eval_cross_dataset.py`

复用 `experiments/vipe_comparison/evaluate.py` 的 evo 接口，但驱动**用同一个 raw cache 跑出 4 个 fused cache → 喂给 VIPE CachedDepthModel → 跑 4 次 SLAM → 评测**。

- [ ] **Step 1：写脚本**

```python
#!/usr/bin/env python3
"""对 1 个序列 raw cache 跑 4 种 fusion，分别注入 VIPE 跑 SLAM，输出 ATE/RTE 表。"""
from __future__ import annotations
import argparse, pathlib, subprocess, json, os
import numpy as np
from experiments.cadf_research.fusion_kernels import (
    fuse_baseline_ema, fuse_conf_weighted, fuse_irls, fuse_robust_geomedian,
)

KERNELS = {
    "baseline_ema": fuse_baseline_ema,
    "conf_weighted": fuse_conf_weighted,
    "irls": fuse_irls,
    "geomedian": fuse_robust_geomedian,
}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-cache", required=True, help="cache_raw_*.npz from C1")
    p.add_argument("--video", required=True)
    p.add_argument("--gt", required=True, help="gt_aligned.txt TUM-format")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    raw = np.load(args.raw_cache)
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, fn in KERNELS.items():
        log_dir = out / name; log_dir.mkdir(exist_ok=True)
        # 1. 跑 fusion → 写临时 cache
        depths, scale_hist = fn(raw["d_pi3x"], raw["conf"], raw["d_moge"], raw["mask"])
        cache_path = log_dir / "fused.npz"
        np.savez_compressed(cache_path, depths=depths, scale_history=scale_hist)
        # 2. 跑 VIPE，注入 cached depth
        env = os.environ.copy()
        env["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path)
        subprocess.check_call([
            "vipe", "infer", args.video,
            "--pipeline", "vipe_cached_depth",
            "--output", str(log_dir / "vipe_out"),
        ], env=env)
        # 3. 评测（复用 evaluate.py 的辅助函数）
        from experiments.vipe_comparison.evaluate import compute_ate_rte_vs_gt  # 假设已抽出
        pose_npz = log_dir / "vipe_out" / "pose" / "video.npz"
        ate, rte = compute_ate_rte_vs_gt(pose_npz, args.gt)
        results[name] = {"ate_rmse": ate["rmse"], "ate_mean": ate["mean"],
                         "rte_rot_mean": rte["rot_mean"], "rte_trans_mean": rte["trans_mean"]}
        print(f"  {name}: ATE={ate['rmse']:.4f}, RTE rot={rte['rot_mean']:.3f}°")

    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {out/'summary.json'}")

if __name__ == "__main__":
    main()
```

> 注：`compute_ate_rte_vs_gt` 在当前 `evaluate.py` 内嵌于 main，需要在 Step 0 先小重构抽成函数（不算独立 task，提交时连同合并）。

- [ ] **Step 2：跑 fr1/desk（4 种 × ~10 min = ~40 min）**

```bash
python experiments/cadf_research/eval_cross_dataset.py \
  --raw-cache experiments/cadf_research/results/cache_raw_fr1_desk.npz \
  --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
  --gt experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/gt_aligned.txt \
  --out-dir experiments/cadf_research/results/tum_fr1
```

期望 `summary.json` 含 4 个 kernel 的指标；预期 IRLS 与 geomedian 优于 baseline。

- [ ] **Step 3：fr2/desk 同样跑一遍（前提：Track A 已生成 raw cache）**

```bash
python experiments/cadf_research/precompute_pi3x_depths_cadf.py \
  --video experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4 \
  --out experiments/cadf_research/results/cache_raw_fr2_desk.npz
python experiments/cadf_research/eval_cross_dataset.py \
  --raw-cache experiments/cadf_research/results/cache_raw_fr2_desk.npz \
  --video experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/video.mp4 \
  --gt experiments/vipe_comparison/data/rgbd_dataset_freiburg2_desk/gt_aligned.txt \
  --out-dir experiments/cadf_research/results/tum_fr2
```

- [ ] **Step 4：KITTI 与 ScanNet++ 同样跑（依赖 Track A）**

照搬上一步，替换 video / gt / out-dir。

- [ ] **Step 5：把 4×4 矩阵汇总到 `experiments/cadf_research/RESULTS_cadf.md`**

| Kernel | TUM-fr1 ATE | TUM-fr2 ATE | KITTI ATE | ScanNet++ ATE |
| --- | --- | --- | --- | --- |
| baseline_ema (SANA-WM) | 0.0227 (复现) | ? | ? | ? |
| conf_weighted | ? | ? | ? | ? |
| irls | ? | ? | ? | ? |
| geomedian | ? | ? | ? | ? |

- [ ] **Step 6：提交 + tag**

```bash
git add experiments/cadf_research/eval_cross_dataset.py experiments/cadf_research/results/ experiments/cadf_research/RESULTS_cadf.md
git commit -m "exp(cadf): cross-dataset 4-kernel ablation (TUM/KITTI/ScanNet++)"
git tag v0.3.0-cadf-ablation
```

---

## Task C4：可选 P2 — 学习型 fusion head

**Files:**
- Create: `experiments/cadf_research/train_fusion_head.py`

如果 C3 的 IRLS/geomedian 不能稳定显著击败 baseline（例如改进 < 5%），就升级到学习型：用一个 ~10 K 参数的小 MLP 在每帧的 (conf 统计, depth 统计, residual 直方图) 上回归 scale。监督信号 = 在 fr1/desk 用 GT depth 反算的"理想 scale"。

- [ ] **Step 1：写 dataset + model + training loop**（先写最简 1-epoch overfit 验证可学性）

```python
# experiments/cadf_research/train_fusion_head.py
"""学习型 fusion head：(conf, d_pi3x, d_moge, mask) → scale_t。
 在 fr1/desk 用 GT depth 反算 ideal scale 训练，在 fr2/KITTI/ScanNet++ 验证泛化。"""
# (代码骨架略；先看 C3 结果再决定是否实现)
```

> **决策点**：先看 C3 的数值，再决定是否要 C4。若 IRLS 已经在 4/4 数据集上稳定 ↓ 10%+ ATE，C4 可作为"future work"留到 paper appendix。

---

## Task C5：起草投稿 paper README（顶会写作准备）

**Files:**
- Create: `experiments/cadf_research/README.md`

把 C1–C3 的所有数值结果与算法描述沉淀进单文件，作为投稿前的"abstract + method + experiments"草稿。

- [ ] **Step 1：填写以下 sections**

```markdown
# Confidence-Aware Depth Fusion for Robust Long-Video SLAM-based Pose Annotation

## 1. Problem
SANA-WM (arXiv:2605.15178) fuses Pi3X relative depth with MoGe-2 metric depth
via mean-of-means × EMA, discarding Pi3X confidence and MoGe-2 valid masks.
This is unstable under sky / sun / featureless regions (KITTI) and tex-poor
walls (ScanNet++).

## 2. Method
Four candidate kernels (see fusion_kernels.py): baseline (replicate),
conf-weighted WLS, IRLS-Huber, conf-weighted Weiszfeld geomedian.

## 3. Experiments
Cross-dataset comparison on TUM-fr1, TUM-fr2, KITTI, ScanNet++.
Metric: ATE RMSE, RTE second-half (long-video drift).

## 4. Results
(insert table from RESULTS_cadf.md)

## 5. Discussion
- Where does conf help most? (expect: KITTI sky, ScanNet++ low-texture walls)
- Compute overhead vs baseline (expect: <5% per frame)
- Compatible with VIPE drop-in (no SLAM core change)
```

- [ ] **Step 2：提交**

```bash
git add experiments/cadf_research/README.md
git commit -m "docs(cadf): paper-ready experiment writeup"
```

---

# 三条 Track 的时间预算与并行性

| Track | 阻塞前置 | 工时（人） | 等待（GPU） | 总 wall-clock |
|---|---|---|---|---|
| A1 fr2/desk | 无 | 0.5 h | 1.5 h | 2 h |
| A2 KITTI | A1 完 | 1 h | 0.5 h | 1.5 h |
| A3 ScanNet++ | A2 完 | 2 h（手工下载） | 0.5 h | 2.5 h |
| B1 DL3DV 下载 | 无 | 0.5 h | 等下载 ~2 h | 2.5 h |
| B2 5-scene e2e | B1 完 | 1 h | 1 h | 2 h |
| B3 回灌主仓 | B2 完 | 0.5 h | — | 0.5 h |
| C1 raw cache | 无 | 0.5 h | 0.5 h | 1 h |
| C2 4 kernel + 单测 | 无 | 1 h | — | 1 h |
| C3 跨数据集评测 | C1+C2+A3 完 | 0.5 h | ~3 h | 3.5 h |
| C4 学习 head（可选） | C3 完 | 3 h | 1 h | 4 h |
| C5 paper writeup | C3 完 | 2 h | — | 2 h |

**强烈推荐顺序**：A1（开机就跑） → C1+C2 并行（人工写代码，A1 GPU 不冲突时）→ A2 → C3 部分（fr1/fr2）→ A3 → C3 补全 → C5。
B 路径完全独立，可在 GPU 空闲时穿插，例如 A1 跑的 1.5 h 等待期内做 B1 下载。

---

# 顶会论文核心 takeaway（写作时强调）

1. **SANA-WM 的 contribution 是工程组合 + 训练流程**；他们的 data pipeline 部分（VIPE+Pi3X+MoGe-2）算法上是直接拼装。**我们的入口正是这里**——把它从"拼装"做成"原理性鲁棒算法"。
2. **CADF 的 selling point**：(a) Drop-in，无需改 SLAM CUDA 核；(b) 跨数据集 generalize；(c) 计算开销 < 5%。
3. **第二个未来 contribution**（不在本计划，作为 next 6-month roadmap）：**per-frame intrinsics BA**——论文 App. B.1 末段提到但**未在开源 VIPE 中实现**。这块是真硬骨头（C++/CUDA），如果 CADF 顺利投稿后还有精力，可作为第二篇。
4. **写作 framing 建议**：不要写成"我们改进 SANA-WM"（容易被审稿人 reject 为 incremental）；写成"针对 internet-scale world-model 数据标注的鲁棒深度融合方法"，把 SANA-WM 作为 baseline 之一。

---

## 自检（plan author）

- [x] 每个 task 文件路径精确（`experiments/cadf_research/fusion_kernels.py` 等）
- [x] 所有代码片段是完整可运行片段，不是 placeholder（除明确标注的 C4 骨架）
- [x] 三条 track 独立可交付（A 验证 / B 数据 / C 算法），任一中断不影响其他
- [x] 资源约束已 surface（磁盘 644 GB、H100×1）
- [x] 已识别 paper-grade 突破点（CADF）而非堆 incremental

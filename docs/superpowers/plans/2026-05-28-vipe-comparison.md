# VIPE 原版 vs SANA-WM 增强版位姿估计精度对比实验

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在带 GT 相机位姿的公开数据集（TUM RGB-D）上，对比 VIPE 原版（unidepth-l 后端）与 SANA-WM 增强版（Pi3X + MoGe-2 后端）的相机轨迹估计精度。

**Architecture:** VIPE 以相同的 MP4 视频为输入，分别用两种深度后端运行一次；输出的 `pose/<stem>.npz`（T,4,4）cam-to-world 矩阵与 TUM GT 轨迹做 Sim(3) 对齐后计算 ATE/RTE。

**Tech Stack:** Python 3.10, VIPE CLI (`vipe infer`), pi3-0.1 (`pi3.Pi3X`), moge-2.0.0 (`moge.model.v2.MoGeModel`), numpy, scipy, matplotlib, evo

---

## 前置事实（已核实）

| 项目 | 路径 / 值 |
|-----|----------|
| Pi3X 权重 | `/mnt/afs/davidwang/models/pi3x/model.safetensors` (5.1 GB ✅) |
| MoGe-2 权重 | `/mnt/afs/davidwang/models/moge2/model.pt` (1.3 GB ✅) |
| Pi3X 类 | `from pi3 import Pi3X` — `Pi3X.from_pretrained(path)` |
| Pi3X forward 输入 | `(B, N, 3, H, W)` float32 [0,1] |
| Pi3X forward 输出 | dict: `camera_poses (B,N,4,4)`, `local_points (B,N,H,W,3)`, `conf (B,N,H,W,1)` |
| MoGe-2 类 | `from moge.model.v2 import MoGeModel` — `MoGeModel.from_pretrained(path)` |
| MoGe-2 infer 输入 | `(B, 3, H, W)` 或 `(3, H, W)` float32 [0,1] |
| MoGe-2 infer 输出 | dict: `depth (B,H,W)`, `points (B,H,W,3)`, `intrinsics (B,3,3)` |
| VIPE 输出格式 | `<out>/pose/<stem>.npz` → keys: `data (T,4,4)`, `inds (T,)` |
| conda 环境 | `sana_wm` |
| 实验目录 | `experiments/vipe_comparison/` |

**注意：** Pi3X `forward` 在 `(B, N, 3, H, W)` 的 N 维度上处理视频帧序列，`camera_poses` 直接输出 cam-to-world（坐标系：OpenCV Right-Down-Forward），**第一帧始终为 identity**。深度从 `local_points[..., 2]` 取 z 分量得到。

---

## 文件结构

```
experiments/vipe_comparison/
├── prepare_tum.py          # 下载 TUM fr1/desk、提取帧、生成 MP4 和 assoc.txt
├── run_method_A.sh         # VIPE + unidepth-l（原版）
├── run_method_B.sh         # VIPE + pi3x_moge2（SANA-WM 增强）
├── evaluate.py             # 加载两组位姿 → ATE/RTE → 绘图
└── configs/
    └── sana_wm_pi3x_moge2.yaml  # 新 pipeline 配置（pi3x_moge2 后端）
```

---

## Task 1: 安装 evo 并下载 TUM fr1/desk

**Files:**
- Create: `experiments/vipe_comparison/prepare_tum.py`

- [ ] **Step 1: 安装 evo**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm
pip install --no-user evo --upgrade --no-binary evo 2>&1 | tail -5
```

期望输出（含 `Successfully installed evo-...`）。

- [ ] **Step 2: 创建实验目录**

```bash
mkdir -p experiments/vipe_comparison/configs
mkdir -p experiments/vipe_comparison/data
mkdir -p experiments/vipe_comparison/results/method_A
mkdir -p experiments/vipe_comparison/results/method_B
```

- [ ] **Step 3: 编写 prepare_tum.py**

```python
#!/usr/bin/env python3
"""
下载 TUM RGB-D fr1/desk，生成 MP4 和 GT 对齐文件。

运行:
  python experiments/vipe_comparison/prepare_tum.py \
      --out experiments/vipe_comparison/data
"""
import argparse, os, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

URL = "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
ASSOC_URL = "https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py"

def download(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[download] {url}")
    subprocess.check_call(["wget", "-q", "-O", str(dest), url])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/vipe_comparison/data")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 下载 TGZ
    tgz = out / "rgbd_dataset_freiburg1_desk.tgz"
    download(URL, tgz)

    # 2. 解压
    seq_dir = out / "rgbd_dataset_freiburg1_desk"
    if not seq_dir.exists():
        print("[extract] decompressing...")
        subprocess.check_call(["tar", "xzf", str(tgz), "-C", str(out)])
    else:
        print("[skip] already extracted")

    # 3. 下载 associate.py
    assoc_py = out / "associate.py"
    download(ASSOC_URL, assoc_py)

    # 4. 生成 associations.txt（RGB 时间戳 ↔ GT 时间戳，容忍 0.02s）
    assoc_file = seq_dir / "associations.txt"
    if not assoc_file.exists():
        print("[assoc] generating associations.txt...")
        result = subprocess.run(
            [sys.executable, str(assoc_py),
             str(seq_dir / "rgb.txt"),
             str(seq_dir / "groundtruth.txt")],
            capture_output=True, text=True
        )
        assoc_file.write_text(result.stdout)
        print(f"[assoc] {assoc_file}: {len(result.stdout.strip().splitlines())} matched pairs")
    else:
        print("[skip] associations.txt exists")

    # 5. 生成 MP4（按 associations.txt 中 RGB 帧顺序）
    mp4_path = seq_dir / "video.mp4"
    if not mp4_path.exists():
        print("[mp4] generating video.mp4...")
        lines = [l for l in assoc_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        frame_paths = []
        gt_poses_lines = []
        for line in lines:
            parts = line.split()
            # associate.py 输出: ts_rgb rgb_path ts_gt tx ty tz qx qy qz qw
            rgb_file = seq_dir / parts[1]
            frame_paths.append(str(rgb_file))
            gt_poses_lines.append(" ".join(parts[2:]))  # ts tx ty tz qx qy qz qw

        # 保存 GT 对齐序列（按帧序号排列）
        gt_aligned = seq_dir / "gt_aligned.txt"
        gt_aligned.write_text("# timestamp tx ty tz qx qy qz qw\n" + "\n".join(gt_poses_lines))
        print(f"[gt] gt_aligned.txt: {len(gt_poses_lines)} poses")

        # 读取第一帧获取分辨率
        sample = cv2.imread(frame_paths[0])
        h, w = sample.shape[:2]
        # VIPE 要求 1280×720 @ 16fps，但对评测来说用原始分辨率更合理
        # 此处保持 640×480（TUM 原始），fps=30（与 TUM 采集频率一致）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, 30.0, (w, h))
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                print(f"[warn] cannot read {fp}")
                continue
            writer.write(frame)
        writer.release()
        print(f"[mp4] {mp4_path}: {len(frame_paths)} frames @ {w}×{h} 30fps")
    else:
        print("[skip] video.mp4 exists")

    print("\n[done] Data ready:")
    print(f"  MP4:  {mp4_path}")
    print(f"  GT:   {seq_dir}/gt_aligned.txt")
    print(f"  Stem: {mp4_path.stem}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行并验证数据准备**

```bash
conda activate sana_wm
python experiments/vipe_comparison/prepare_tum.py \
    --out experiments/vipe_comparison/data
```

期望输出：
```
[mp4] .../video.mp4: 570 frames @ 640×480 30fps
[gt]  gt_aligned.txt: 570 poses
```
（fr1/desk 有 613 RGB 帧，associate.py 匹配后约 570 对，数量可能略有差异）

- [ ] **Step 5: 提交**

```bash
git add experiments/
git commit -m "experiment: add TUM data preparation script for VIPE comparison"
```

---

## Task 2: 创建 pi3x_moge2 VIPE pipeline 配置

**Files:**
- Create: `experiments/vipe_comparison/configs/sana_wm_pi3x_moge2.yaml`

- [ ] **Step 1: 确认 VIPE 已安装**

```bash
conda activate sana_wm
vipe --version 2>&1
```

期望输出：`vipe X.Y.Z`。若显示 "not found"，先运行 `bash scripts/00_setup_vipe.sh`。

- [ ] **Step 2: 确认 sana_wm_pose_only.yaml 已复制到 VIPE**

```bash
ls third_party/vipe/configs/pipeline/sana_wm_pose_only.yaml
```

若不存在，运行 `bash scripts/00_setup_vipe.sh`。

- [ ] **Step 3: 创建 pi3x_moge2 pipeline yaml**

```yaml
# experiments/vipe_comparison/configs/sana_wm_pi3x_moge2.yaml
# SANA-WM 增强版 pipeline（pi3x_moge2 后端）
# 与 sana_wm_pose_only.yaml 的唯一区别：两处深度后端改为 pi3x_moge2
defaults:
  - default

init:
  instance: null

slam:
  keyframe_depth: pi3x_moge2    # ← 从 unidepth-l 改为 pi3x_moge2

post:
  depth_align_model: pi3x_moge2 # ← 从 adaptive_unidepth-l 改为 pi3x_moge2

output:
  save_viz: false
  save_artifacts: true
  viz_downsample: 2
  viz_attributes: [['rgb']]
```

保存路径：`experiments/vipe_comparison/configs/sana_wm_pi3x_moge2.yaml`

- [ ] **Step 4: 安装到 VIPE 的配置目录**

```bash
cp experiments/vipe_comparison/configs/sana_wm_pi3x_moge2.yaml \
   third_party/vipe/configs/pipeline/sana_wm_pi3x_moge2.yaml
echo "[ok] sana_wm_pi3x_moge2 installed"
```

- [ ] **Step 5: 提交**

```bash
git add experiments/vipe_comparison/configs/sana_wm_pi3x_moge2.yaml
git commit -m "experiment: add pi3x_moge2 pipeline config for VIPE comparison"
```

---

## Task 3: 编写并运行 Method A（VIPE + unidepth-l 原版）

**Files:**
- Create: `experiments/vipe_comparison/run_method_A.sh`

- [ ] **Step 1: 编写 run_method_A.sh**

```bash
#!/usr/bin/env bash
# Method A: VIPE with unidepth-l depth backend (vanilla)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ="${REPO_ROOT}/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk"
OUT="${REPO_ROOT}/experiments/vipe_comparison/results/method_A"
VIDEO="${SEQ}/video.mp4"

if [ ! -f "${VIDEO}" ]; then
  echo "[error] video.mp4 not found. Run prepare_tum.py first."
  exit 1
fi

mkdir -p "${OUT}"

echo "[method_A] Running VIPE with unidepth-l backend..."
echo "[method_A] Input:  ${VIDEO}"
echo "[method_A] Output: ${OUT}"

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

vipe infer "${VIDEO}" \
    --output "${OUT}" \
    --pipeline sana_wm_pose_only

echo "[method_A] Done."
echo "[method_A] Pose artifact: ${OUT}/pose/video.npz"

# 快速验证
python - <<'PYEOF'
import numpy as np, sys
from pathlib import Path

npz = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/vipe_comparison/results/method_A/pose/video.npz")
if not npz.exists():
    print(f"[error] {npz} not found"); sys.exit(1)
d = np.load(npz)
print(f"[check] poses shape={d['data'].shape}, inds shape={d['inds'].shape}")
print(f"[check] first pose (should be ~identity):\n{d['data'][0]}")
PYEOF
```

```bash
chmod +x experiments/vipe_comparison/run_method_A.sh
```

- [ ] **Step 2: 运行 Method A**

```bash
bash experiments/vipe_comparison/run_method_A.sh 2>&1 | tee experiments/vipe_comparison/results/method_A.log
```

预计耗时：2~10 分钟（H100 加速）。

期望输出末尾：
```
[check] poses shape=(570, 4, 4), inds shape=(570,)
[check] first pose (should be ~identity):
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 ...
```

- [ ] **Step 3: 提交**

```bash
git add experiments/vipe_comparison/run_method_A.sh
git commit -m "experiment: add Method A (VIPE unidepth-l) run script"
```

---

## Task 4: 编写并运行 Method B（VIPE + Pi3X + MoGe-2 增强版）

**Files:**
- Create: `experiments/vipe_comparison/run_method_B.sh`

- [ ] **Step 1: 编写 run_method_B.sh**

```bash
#!/usr/bin/env bash
# Method B: VIPE with Pi3X + MoGe-2 depth backend (SANA-WM enhanced)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEQ="${REPO_ROOT}/experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk"
OUT="${REPO_ROOT}/experiments/vipe_comparison/results/method_B"
VIDEO="${SEQ}/video.mp4"

if [ ! -f "${VIDEO}" ]; then
  echo "[error] video.mp4 not found. Run prepare_tum.py first."
  exit 1
fi

mkdir -p "${OUT}"

# 模型权重路径（已下载）
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2

echo "[method_B] Running VIPE with Pi3X + MoGe-2 backend (SANA-WM)..."
echo "[method_B] PI3X weights: ${SANA_WM_PI3X_WEIGHTS}"
echo "[method_B] MoGe2 weights: ${SANA_WM_MOGE2_WEIGHTS}"
echo "[method_B] Input:  ${VIDEO}"
echo "[method_B] Output: ${OUT}"

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

vipe infer "${VIDEO}" \
    --output "${OUT}" \
    --pipeline sana_wm_pi3x_moge2

echo "[method_B] Done."
echo "[method_B] Pose artifact: ${OUT}/pose/video.npz"

python - <<'PYEOF'
import numpy as np, sys
from pathlib import Path

npz = Path("experiments/vipe_comparison/results/method_B/pose/video.npz")
if not npz.exists():
    print(f"[error] {npz} not found"); sys.exit(1)
d = np.load(npz)
print(f"[check] poses shape={d['data'].shape}, inds shape={d['inds'].shape}")
print(f"[check] first pose (should be ~identity):\n{d['data'][0]}")
PYEOF
```

```bash
chmod +x experiments/vipe_comparison/run_method_B.sh
```

- [ ] **Step 2: 运行 Method B（先验证权重可用）**

```bash
# 先验证模型可以加载（不需要完整推理）
conda activate sana_wm
python - <<'PYEOF'
import torch
from pi3 import Pi3X
from moge.model.v2 import MoGeModel

print("Loading Pi3X...")
m = Pi3X.from_pretrained("/mnt/afs/davidwang/models/pi3x")
print(f"Pi3X loaded: {sum(p.numel() for p in m.parameters())/1e6:.0f}M params")

print("Loading MoGe-2...")
m2 = MoGeModel.from_pretrained("/mnt/afs/davidwang/models/moge2")
print(f"MoGe-2 loaded: {sum(p.numel() for p in m2.parameters())/1e6:.0f}M params")
print("Both models OK.")
PYEOF
```

期望输出：
```
Pi3X loaded: ~600M params
MoGe-2 loaded: ~300M params
Both models OK.
```

- [ ] **Step 3: 运行 Method B**

```bash
bash experiments/vipe_comparison/run_method_B.sh 2>&1 | tee experiments/vipe_comparison/results/method_B.log
```

预计耗时：5~20 分钟（Pi3X 首次加载 + 推理比 unidepth 慢）。

- [ ] **Step 4: 提交**

```bash
git add experiments/vipe_comparison/run_method_B.sh
git commit -m "experiment: add Method B (VIPE pi3x_moge2) run script"
```

---

## Task 5: 编写评测脚本 evaluate.py

**Files:**
- Create: `experiments/vipe_comparison/evaluate.py`

- [ ] **Step 1: 编写 evaluate.py**

```python
#!/usr/bin/env python3
"""
评测 VIPE 原版 vs SANA-WM 增强版位姿估计精度。

用法:
  python experiments/vipe_comparison/evaluate.py
  python experiments/vipe_comparison/evaluate.py --seq data/rgbd_dataset_freiburg1_desk
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


# ─── 参数 ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--seq", default="experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk")
parser.add_argument("--results", default="experiments/vipe_comparison/results")
args = parser.parse_args()

SEQ     = Path(args.seq)
RESULTS = Path(args.results)
PLOT_DIR = RESULTS / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ─── GT 加载（gt_aligned.txt：帧序号已与 MP4 对齐）─────────────────────────────

def load_gt(path: Path) -> np.ndarray:
    """
    读取 gt_aligned.txt，格式：# timestamp tx ty tz qx qy qz qw
    返回 (N, 4, 4) cam-to-world（world = TUM mocap frame）
    """
    poses = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        vals = [float(x) for x in line.split()]
        # timestamp tx ty tz qx qy qz qw
        tx, ty, tz = vals[1], vals[2], vals[3]
        qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        poses.append(T)
    return np.stack(poses)  # (N, 4, 4)


def load_vipe_poses(npz_path: Path) -> np.ndarray:
    """
    读取 VIPE 输出的 pose/<stem>.npz。
    keys: data (T, 4, 4) cam-to-world, inds (T,)
    第一帧已被 mode_default._interp_poses 归一化为 identity。
    """
    d = np.load(npz_path)
    poses = d["data"].astype(np.float32)   # (T, 4, 4)
    inds  = d["inds"]                       # (T,)
    # 若 VIPE 只写了关键帧，按 inds 插值到完整帧（与 mode_default 逻辑一致）
    T_full = int(inds.max()) + 1
    if len(poses) == T_full:
        return poses
    # 线性插值平移，旋转用最近邻（简单近似）
    full = np.zeros((T_full, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            full[:, i, j] = np.interp(np.arange(T_full), inds, poses[:, i, j])
    return full


# ─── Umeyama Sim(3) 对齐 ─────────────────────────────────────────────────────

def umeyama_align(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    最小二乘 Sim(3) 对齐：dst ≈ scale * R @ src + t
    src, dst: (N, 3) 轨迹点
    返回 (scale, R, t)
    """
    n     = len(src)
    mu_s  = src.mean(0)
    mu_d  = dst.mean(0)
    s_c   = src - mu_s
    d_c   = dst - mu_d
    var_s = (s_c ** 2).sum() / n
    cov   = (d_c.T @ s_c) / n
    U, D, Vt = np.linalg.svd(cov)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    S_mat = np.diag([1.0, 1.0, det_sign])
    R     = U @ S_mat @ Vt
    scale = float((D * np.diag(S_mat)).sum() / var_s)
    t     = mu_d - scale * R @ mu_s
    return scale, R, t


def align_to_gt(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, float]:
    """
    用 Sim(3) 对齐 pred 轨迹到 gt，返回对齐后的 pred 和估计尺度。
    pred, gt: (N, 4, 4)
    """
    n = min(len(pred), len(gt))
    pred_t = pred[:n, :3, 3]
    gt_t   = gt[:n,  :3, 3]
    scale, R, t = umeyama_align(pred_t, gt_t)
    aligned = pred[:n].copy()
    aligned[:, :3, 3] = (scale * (R @ pred_t.T)).T + t
    aligned[:, :3, :3] = R @ pred[:n, :3, :3]
    return aligned, scale


# ─── 指标计算 ──────────────────────────────────────────────────────────────────

def compute_ate(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Absolute Trajectory Error（ATE）with Sim(3) alignment。
    返回 rmse, mean, median, max, per_frame_errors, scale
    """
    n = min(len(pred), len(gt))
    aligned, scale = align_to_gt(pred[:n], gt[:n])
    errs = np.linalg.norm(aligned[:, :3, 3] - gt[:n, :3, 3], axis=1)
    return {
        "rmse":   float(np.sqrt((errs ** 2).mean())),
        "mean":   float(errs.mean()),
        "median": float(np.median(errs)),
        "max":    float(errs.max()),
        "scale":  scale,
        "per_frame": errs,
    }


def compute_rte(pred: np.ndarray, gt: np.ndarray, delta: int = 30) -> dict:
    """
    Relative Trajectory Error（RTE）：相隔 delta 帧的相对运动误差。
    delta=30 ≈ 1 秒（30fps）。
    分前半段/后半段报告，后半段高误差 = 长视频漂移。
    pred, gt: (N, 4, 4)
    """
    n = min(len(pred), len(gt))
    rot_errs, trans_errs = [], []
    for i in range(0, n - delta, delta):
        dT_gt   = np.linalg.inv(gt[i])   @ gt[i + delta]
        dT_pred = np.linalg.inv(pred[i]) @ pred[i + delta]
        dT_err  = np.linalg.inv(dT_gt) @ dT_pred
        R_err   = dT_err[:3, :3]
        cos_val = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        rot_errs.append(float(np.degrees(np.arccos(cos_val))))
        trans_errs.append(float(np.linalg.norm(dT_err[:3, 3])))
    half = len(rot_errs) // 2
    return {
        "rot_mean":       float(np.mean(rot_errs)),
        "trans_mean":     float(np.mean(trans_errs)),
        "rot_2nd_half":   float(np.mean(rot_errs[half:])),
        "trans_2nd_half": float(np.mean(trans_errs[half:])),
    }


# ─── 主评测流程 ────────────────────────────────────────────────────────────────

gt_poses = load_gt(SEQ / "gt_aligned.txt")
print(f"GT poses loaded: {len(gt_poses)} frames")

methods = {
    "A: VIPE + unidepth-l (原版)": RESULTS / "method_A" / "pose" / "video.npz",
    "B: VIPE + Pi3X+MoGe-2 (SANA-WM)": RESULTS / "method_B" / "pose" / "video.npz",
}

results: dict[str, dict] = {}
for name, npz_path in methods.items():
    if not npz_path.exists():
        print(f"[skip] {name}: {npz_path} not found")
        continue
    pred = load_vipe_poses(npz_path)
    ate  = compute_ate(pred, gt_poses)
    rte  = compute_rte(pred, gt_poses, delta=30)
    results[name] = {"ate": ate, "rte": rte}
    print(f"\n{'='*60}")
    print(f"Method: {name}")
    print(f"  帧数:         pred={len(pred)}, gt={len(gt_poses)}, eval={min(len(pred),len(gt_poses))}")
    print(f"  ATE RMSE:     {ate['rmse']:.4f} m  (Sim3 对齐后)")
    print(f"  ATE mean:     {ate['mean']:.4f} m")
    print(f"  ATE median:   {ate['median']:.4f} m")
    print(f"  ATE max:      {ate['max']:.4f} m")
    print(f"  估计尺度:     {ate['scale']:.4f}  (理想值 ≈ 1.0 当深度为米制)")
    print(f"  RTE 旋转均值: {rte['rot_mean']:.3f}°")
    print(f"  RTE 平移均值: {rte['trans_mean']:.4f} m")
    print(f"  RTE 后半漂移旋转: {rte['rot_2nd_half']:.3f}°  ← 长视频稳定性指标")
    print(f"  RTE 后半漂移平移: {rte['trans_2nd_half']:.4f} m")

if len(results) < 2:
    print("\n[warn] 少于两组结果，跳过对比图")
else:
    # ─── 可视化 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {
        "A: VIPE + unidepth-l (原版)": "tab:red",
        "B: VIPE + Pi3X+MoGe-2 (SANA-WM)": "tab:blue",
    }

    # 1) 俯视轨迹图
    ax = axes[0]
    ax.set_title("轨迹对比（俯视 X-Z 平面）", fontsize=10)
    gt_t = gt_poses[:, :3, 3]
    ax.plot(gt_t[:, 0], gt_t[:, 2], "k-", lw=2, label="GT", zorder=10)
    for name, data in results.items():
        npz = methods[name]
        pred = load_vipe_poses(npz)
        n = min(len(pred), len(gt_poses))
        aligned, _ = align_to_gt(pred[:n], gt_poses[:n])
        ax.plot(aligned[:, 0, 3], aligned[:, 2, 3],
                color=colors[name], lw=1.2, label=name, alpha=0.8)
    ax.legend(fontsize=7); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")

    # 2) 逐帧 ATE 误差曲线
    ax = axes[1]
    ax.set_title("逐帧 ATE（Sim3 对齐）", fontsize=10)
    for name, data in results.items():
        errs = data["ate"]["per_frame"]
        ax.plot(errs, color=colors[name], lw=0.8, label=name, alpha=0.9)
    half_n = len(list(results.values())[0]["ate"]["per_frame"]) // 2
    ax.axvline(x=half_n, color="gray", ls="--", lw=0.8, label="中点")
    ax.legend(fontsize=7); ax.set_xlabel("帧序号"); ax.set_ylabel("误差 (m)")

    # 3) 柱状图：ATE RMSE 对比
    ax = axes[2]
    ax.set_title("ATE RMSE（越低越好）", fontsize=10)
    names = list(results.keys())
    vals  = [results[n]["ate"]["rmse"] for n in names]
    x     = np.arange(len(names))
    bars  = ax.bar(x, vals, color=[colors[n] for n in names], alpha=0.8)
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=8)
    ax.set_ylabel("RMSE (m)")

    plt.tight_layout()
    out_fig = PLOT_DIR / "comparison.png"
    plt.savefig(out_fig, dpi=150)
    print(f"\n[plot] 图表保存到 {out_fig}")

    # ─── 汇总对比表 ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'指标':<30} {'A: VIPE原版':>15} {'B: SANA-WM':>15}")
    print("="*65)
    rows = [
        ("ATE RMSE (Sim3) ↓ (m)",    "ate", "rmse"),
        ("ATE mean ↓ (m)",            "ate", "mean"),
        ("RTE 旋转均值 ↓ (°)",         "rte", "rot_mean"),
        ("RTE 平移均值 ↓ (m)",         "rte", "trans_mean"),
        ("RTE 后半段旋转 ↓ (°)",       "rte", "rot_2nd_half"),
        ("RTE 后半段平移 ↓ (m)",       "rte", "trans_2nd_half"),
    ]
    for label, key, sub in rows:
        vals_str = []
        for n in ["A: VIPE + unidepth-l (原版)", "B: VIPE + Pi3X+MoGe-2 (SANA-WM)"]:
            vals_str.append(f"{results[n][key][sub]:>15.4f}")
        print(f"{label:<30} {'  '.join(vals_str)}")
    print("="*65)
```

- [ ] **Step 2: 确认写入正确路径**

```bash
ls experiments/vipe_comparison/evaluate.py
```

- [ ] **Step 3: 运行评测（需要两组结果都存在）**

```bash
conda activate sana_wm
python experiments/vipe_comparison/evaluate.py
```

期望终端输出类似：
```
GT poses loaded: 570 frames

============================================================
Method: A: VIPE + unidepth-l (原版)
  ATE RMSE:  0.0XXX m
  ...

Method: B: VIPE + Pi3X+MoGe-2 (SANA-WM)
  ATE RMSE:  0.0XXX m
  ...

=================================================================
指标                            A: VIPE原版      B: SANA-WM
=================================================================
ATE RMSE (Sim3) ↓ (m)            0.XXXX          0.XXXX
...
```

- [ ] **Step 4: 提交**

```bash
git add experiments/vipe_comparison/evaluate.py
git commit -m "experiment: add VIPE comparison evaluation script"
```

---

## 一键运行脚本（所有步骤串行执行）

```bash
#!/usr/bin/env bash
# experiments/vipe_comparison/run_all.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

echo "=== Step 0: 准备数据 ==="
python "${REPO}/experiments/vipe_comparison/prepare_tum.py"

echo "=== Step 1: Method A (VIPE unidepth-l) ==="
bash "${REPO}/experiments/vipe_comparison/run_method_A.sh"

echo "=== Step 2: Method B (VIPE pi3x_moge2) ==="
bash "${REPO}/experiments/vipe_comparison/run_method_B.sh"

echo "=== Step 3: 评测 ==="
python "${REPO}/experiments/vipe_comparison/evaluate.py"
```

---

## 预期结果解读

### 核心指标含义

| 指标 | 含义 | SANA-WM 预期改善 |
|-----|------|-----------------|
| ATE RMSE (Sim3 对齐) | 全局轨迹精度，尺度已纠正 | 轻微改善（Pi3X 长序列一致性） |
| RTE 后半段平移误差 | 长视频漂移量 | 明显改善（Pi3X 减少累积误差） |
| 估计尺度（scale）| Sim3 对齐的尺度因子，越接近 1.0 说明深度越准确米制 | 改善（MoGe-2 提供米制锚点） |

### fr1/desk 序列的合理数量级（基于 TUM 已发布基准）

TUM fr1/desk 是一个~28s 的桌面场景。优秀 VO 系统在此序列上 ATE RMSE 约 0.01~0.03 m。VIPE 作为视觉里程计，预计：
- Method A（unidepth-l）: ATE RMSE ≈ 0.02~0.08 m
- Method B（Pi3X + MoGe-2）: ATE RMSE ≈ 0.01~0.06 m

**注意：** 具体数值取决于 VIPE 的 SLAM 设置，上述范围仅为量级参考，不构成保证。

### 如何判断实验成功

1. Method B 的 **ATE RMSE 低于 Method A**，则验证 SANA-WM 深度增强有效
2. Method B 的 **RTE 后半段误差低于 Method A**，则验证 Pi3X 减少了长视频漂移
3. Method B 的**估计尺度更接近 1.0**，则验证 MoGe-2 提供了更准确的米制深度

---

## 已知限制

1. **VIPE 内部调用 Pi3X 的方式**：Pi3X 在 VIPE 中作为深度后端（`DepthEstimationModel`），VIPE 的 SLAM 前端决定最终位姿；Pi3X 提供深度先验，不直接决定位姿。
2. **TUM 短序列的局限性**：fr1/desk 仅 28s（570 帧），长视频漂移的优势可能不显著。如果 A/B 差距很小，可用 fr2/desk（99s, 2965 帧）重复实验。
3. **帧率差异**：TUM 采集 30fps，SANA-WM 训练目标 16fps。MP4 以 30fps 输出，VIPE 内部如有帧率处理逻辑，结果可能与 16fps 输入有细微差别。


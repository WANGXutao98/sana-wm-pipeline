# VIPE 原版 vs SANA-WM 增强版 —— 位姿估计精度对比实验方案

> 目标：用带 GT 位姿的公开数据集，量化验证 SANA-WM 的深度后端升级
> （Metric3D-Small → Pi3X + MoGe-2）对相机轨迹精度的提升。

---

## 一、实验全景

### 1.1 核心假设与可验证命题

SANA-WM 论文第 4 节声称：
> *"we found its original depth estimation unstable on long videos. We therefore replace the depth backend with Pi3X and MoGe-2"*

这个断言在论文里是**文字描述、无定量支撑**。本实验将其转化为可测量命题：

| 可测命题 | 对应指标 |
|---|---|
| Pi3X 提供更长时序一致的深度 | 轨迹后半段 RTE（相对轨迹误差）下降 |
| MoGe-2 提供更准的米制尺度锚点 | Scale Error（绝对尺度误差）下降 |
| 两者组合使整体位姿估计更准 | ATE（绝对轨迹误差）整体下降 |

### 1.2 三路对比方案

| 方案 | 深度后端 | 米制尺度 | 角色 |
|---|---|---|---|
| **A: 原始 VIPE** | 内置 VideoDepthAnything | 内部估计（近米制）| 论文 Baseline |
| **B: Pi3X only** | Pi3X 长序列一致深度 | **无**（相对尺度）| 控制变量（只换深度一致性）|
| **C: Pi3X + MoGe-2** | Pi3X | MoGe-2 米制锚点 | SANA-WM 增强版 |

方案 B vs A → 分离"深度时序一致性"贡献  
方案 C vs B → 分离"MoGe-2 米制尺度锚点"贡献  
方案 C vs A → SANA-WM 增强的总体效果

### 1.3 数据集选择

**主数据集：TUM RGB-D**（VIPE 论文自己的 benchmark）
- GT 来源：运动捕捉系统，精度 < 1mm
- 下载免费，无需申请
- VIPE 报告了 TUM 上的基线数字，可直接对标

| 序列 | 时长 | 帧数 | 特点 | 用途 |
|---|---|---|---|---|
| `fr1/desk` | 28s | 613帧 | 短序列，桌面扫描 | 快速验证 |
| `fr2/desk` | 99s | 2965帧 | 长序列，接近60s目标 | 主实验 |
| `fr3/long_office` | 91s | 2728帧 | 长序列+大范围运动 | 长视频稳定性 |

**推荐从 `fr1/desk` 开始，跑通后再用 `fr2/desk` 验证长视频效果。**

---

## 二、环境配置

```bash
# ── 创建独立环境 ─────────────────────────────────────────────
conda create -n vipe_compare python=3.10 -y
conda activate vipe_compare

# ── 安装 VIPE（原始版） ───────────────────────────────────────
git clone https://github.com/nv-tlabs/vipe.git
cd vipe
conda env update -f envs/base.yml           # 更新 CUDA/PyTorch 依赖
pip install -r envs/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation -e .
cd ..

# ── 安装 Pi3X ────────────────────────────────────────────────
git clone https://github.com/yyfz/Pi3.git
cd Pi3
pip install -e .
cd ..

# ── 安装 MoGe-2（Microsoft）────────────────────────────────
# 确认最新版：原始 MoGe 仓库，MoGe-2 在同一 repo 的新模型卡里
pip install git+https://github.com/microsoft/MoGe.git

# ── 通用依赖 ─────────────────────────────────────────────────
pip install numpy scipy matplotlib tqdm pillow opencv-python evo
# evo 是专用轨迹评测工具，支持 TUM/KITTI 格式，自带 ATE/RPE
```

---

## 三、数据集下载与准备

```bash
# ── 下载 TUM RGB-D ────────────────────────────────────────────
mkdir -p data/tum && cd data/tum

# 快速验证用（28s）
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar xzf rgbd_dataset_freiburg1_desk.tgz

# 主实验用（99s，长视频）
wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
tar xzf rgbd_dataset_freiburg2_desk.tgz
cd ../..

# ── TUM 数据结构说明 ──────────────────────────────────────────
# rgbd_dataset_freiburg1_desk/
#   rgb/             ← RGB 帧，文件名 = 时间戳.png
#   rgb.txt          ← 时间戳 → 文件名映射
#   groundtruth.txt  ← GT 位姿: timestamp tx ty tz qx qy qz qw
#   depth/           ← 深度图（可选，本实验不用）

# ── 生成关联文件（RGB帧 ↔ GT位姿时间戳对齐） ───────────────────
# TUM 官方工具
wget https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/\
rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py

python associate.py \
    data/tum/rgbd_dataset_freiburg1_desk/rgb.txt \
    data/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    > data/tum/rgbd_dataset_freiburg1_desk/associations.txt

# ── 导出为 mp4 供 VIPE 使用 ───────────────────────────────────
python - << 'EOF'
import cv2, os, glob

seq = "data/tum/rgbd_dataset_freiburg1_desk"
frames = sorted(glob.glob(f"{seq}/rgb/*.png"))
h, w = cv2.imread(frames[0]).shape[:2]
out = cv2.VideoWriter(f"{seq}/video.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
for f in frames:
    out.write(cv2.imread(f))
out.release()
print(f"写出 {len(frames)} 帧 → {seq}/video.mp4")
EOF
```

---

## 四、方案 A：运行原始 VIPE

```bash
SEQ="data/tum/rgbd_dataset_freiburg1_desk"

# VIPE 直接推理，输出到 results/vipe_A/
vipe infer ${SEQ}/video.mp4 \
    --output results/vipe_A \
    --pipeline default

# 输出目录结构：
# results/vipe_A/
#   poses.npy       ← (N, 4, 4) camera-to-world 矩阵
#   depths/         ← 每帧深度图
#   intrinsics.npy  ← 相机内参
```

---

## 五、方案 B & C：Pi3X + MoGe-2 管线

```python
# run_pi3x_moge.py
"""
SANA-WM 数据标注管线核心：
  B：Pi3X 仅（相对尺度）
  C：Pi3X + MoGe-2（米制尺度，SANA-WM 完整增强版）

用法：
  python run_pi3x_moge.py --seq data/tum/rgbd_dataset_freiburg1_desk \
                          --variant B   # 或 C
"""

import argparse, os, glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── 参数 ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seq",     required=True)
parser.add_argument("--variant", default="C", choices=["B", "C"])
parser.add_argument("--chunk",   type=int, default=16)   # Pi3X 每批帧数
parser.add_argument("--stride",  type=int, default=14)   # 滑窗步长（重叠2帧）
parser.add_argument("--out",     default="results")
args = parser.parse_args()

os.makedirs(f"{args.out}/pi3x_{args.variant}", exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 加载模型 ──────────────────────────────────────────────────
print("Loading Pi3X...")
from pi3 import Pi3
pi3x = Pi3.from_pretrained("yyfz/Pi3").to(DEVICE).eval()

if args.variant == "C":
    print("Loading MoGe-2...")
    from moge.model import MoGeModel
    moge = MoGeModel.from_pretrained("microsoft/MoGe-vitl").to(DEVICE).eval()

# ── 读取 TUM 帧 ───────────────────────────────────────────────
frame_paths = sorted(glob.glob(f"{args.seq}/rgb/*.png"))
print(f"序列长度：{len(frame_paths)} 帧")

def load_frame(path, size=(640, 480)):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img, dtype=np.float32) / 255.0   # (H, W, 3), [0,1]

frames = [load_frame(p) for p in tqdm(frame_paths, desc="Loading frames")]

# ── Step 1：Pi3X 分块推理 → 相对位姿 ─────────────────────────
def run_pi3x_chunk(model, frame_list, device):
    """
    输入：list of (H,W,3) float32 numpy
    输出：(N,4,4) c2w 位姿矩阵（相对尺度）
    """
    imgs_t = torch.tensor(
        np.stack(frame_list),       # (N, H, W, 3)
        dtype=torch.float32
    ).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, N, 3, H, W)

    with torch.no_grad():
        result = model(imgs_t)

    # Pi3X 输出：result.poses: (N, 4, 4) tensor，affine-invariant
    poses = result.poses[0].cpu().numpy()      # (N, 4, 4)
    depths = result.pointmaps[0, :, :, :, 2]  # (N, H, W) z 分量即深度
    depths = depths.cpu().numpy()
    return poses, depths

print("Running Pi3X in chunks...")
all_poses_rel = []    # 每块的相对位姿
all_depths_pi3x = []  # 每块每帧的 Pi3X 深度图

for start in tqdm(range(0, len(frames), args.stride)):
    chunk = frames[start : start + args.chunk]
    if len(chunk) < 2:
        break
    poses_chunk, depths_chunk = run_pi3x_chunk(pi3x, chunk, DEVICE)
    all_poses_rel.append((start, poses_chunk))
    all_depths_pi3x.extend(depths_chunk)

# ── Step 2：拼接分块位姿 → 全局轨迹 ──────────────────────────
def stitch_poses(chunks):
    """
    用相邻块的重叠帧（最后2帧/最初2帧）做 Procrustes 对齐，
    把各块拼成全局轨迹。
    chunks: list of (start_idx, poses_array (N,4,4))
    """
    if len(chunks) == 1:
        return chunks[0][1]

    global_poses = [chunks[0][1]]
    for i in range(1, len(chunks)):
        prev_start, prev_poses = chunks[i-1]
        curr_start, curr_poses = chunks[i]

        # 重叠帧数
        overlap = (prev_start + len(prev_poses)) - curr_start
        if overlap < 1:
            overlap = 1

        # 上一块末尾位姿（全局坐标）
        T_ref = global_poses[-1][-overlap:]
        # 当前块开头位姿（本地坐标）
        T_curr_local = curr_poses[:overlap]

        # 求对齐变换：T_global = T_align @ T_local
        # 用最小二乘拟合：T_align = T_ref_mean @ inv(T_local_mean)
        T_align = T_ref.mean(0) @ np.linalg.inv(T_curr_local.mean(0))

        # 对当前块所有位姿应用对齐
        curr_global = np.stack([T_align @ p for p in curr_poses])
        global_poses.append(curr_global[overlap:])  # 去掉重叠部分

    return np.concatenate(global_poses, axis=0)

poses_global = stitch_poses(all_poses_rel)
print(f"拼接后全局轨迹：{len(poses_global)} 帧")

# ── Step 3（仅 C）：MoGe-2 米制尺度校正 ──────────────────────
if args.variant == "C":
    print("Running MoGe-2 for metric scale...")

    moge_depths = []
    for frame in tqdm(frames, desc="MoGe-2 inference"):
        img_t = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = moge.infer(img_t)
        moge_depths.append(out["depth"].squeeze().cpu().numpy())  # (H, W)

    def fuse_metric_scale(pi3x_depths, moge_depths, ema_momentum=0.99):
        """
        SANA-WM 论文公式：
          s* = argmin_s  sum_i  w_i * (s * d_pi3x_i - d_moge_i)^2
          其中 w_i = 1/d_moge_i（逆深度权重，强调近处精度）
          闭式解：s* = sum(w*d_pi3x*d_moge) / sum(w*d_pi3x^2)
        再用 EMA 动量=0.99 时序平滑（论文 Appendix B.1）
        """
        scale_raw = []
        for d_p, d_m in zip(pi3x_depths, moge_depths):
            # 只用有效深度区域（滤掉 0 和 inf）
            mask = (d_m > 0.1) & (d_p > 0.01) & np.isfinite(d_m) & np.isfinite(d_p)
            if mask.sum() < 100:
                scale_raw.append(scale_raw[-1] if scale_raw else 1.0)
                continue
            w  = 1.0 / (d_m[mask] + 1e-6)
            dp = d_p[mask]
            dm = d_m[mask]
            s = (w * dp * dm).sum() / ((w * dp**2).sum() + 1e-8)
            scale_raw.append(float(s))

        # EMA 平滑
        smoothed = [scale_raw[0]]
        for s in scale_raw[1:]:
            smoothed.append(ema_momentum * smoothed[-1] + (1 - ema_momentum) * s)
        return np.array(smoothed)

    scale_factors = fuse_metric_scale(all_depths_pi3x, moge_depths)
    print(f"尺度因子统计：mean={scale_factors.mean():.4f}, "
          f"std={scale_factors.std():.4f}, "
          f"CV={scale_factors.std()/scale_factors.mean():.4f}")

    # 把尺度因子乘到位移向量上
    poses_metric = poses_global.copy()
    for i, s in enumerate(scale_factors[:len(poses_global)]):
        poses_metric[i, :3, 3] *= s
    poses_final = poses_metric
else:
    # 方案 B：直接用相对尺度
    poses_final = poses_global

# ── 保存结果 ──────────────────────────────────────────────────
out_dir = f"{args.out}/pi3x_{args.variant}"
np.save(f"{out_dir}/poses.npy", poses_final)
if args.variant == "C":
    np.save(f"{out_dir}/scale_factors.npy", scale_factors)

print(f"结果已保存到 {out_dir}/poses.npy  shape={poses_final.shape}")
```

---

## 六、评测脚本

```python
# evaluate.py
"""
评测三路位姿估计结果 vs TUM GT，输出 ATE/RTE 并绘图。

用法：
  python evaluate.py --seq data/tum/rgbd_dataset_freiburg1_desk
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seq",     required=True)
parser.add_argument("--results", default="results")
args = parser.parse_args()

# ════════════════════════════════════════════════════════════════
# 1. 读取 GT 轨迹（TUM 格式）
# ════════════════════════════════════════════════════════════════
def load_tum_gt(filepath):
    """返回 (N, 4, 4) c2w 矩阵，按时间戳排序"""
    poses = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            vals = [float(x) for x in line.strip().split()]
            # ts tx ty tz qx qy qz qw
            ts, tx, ty, tz, qx, qy, qz, qw = vals
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = [tx, ty, tz]
            poses.append(T)
    return np.stack(poses)

gt_poses = load_tum_gt(f"{args.seq}/groundtruth.txt")
print(f"GT 帧数: {len(gt_poses)}")

# ════════════════════════════════════════════════════════════════
# 2. Umeyama Sim(3) 对齐
# ════════════════════════════════════════════════════════════════
def umeyama_align(src_pts, dst_pts, with_scale=True):
    """
    src_pts → dst_pts 的 Sim(3) 或 SE(3) 最优对齐
    返回 (scale, R, t) 使得 dst ≈ scale * R @ src + t
    """
    n = len(src_pts)
    mu_s = src_pts.mean(0)
    mu_d = dst_pts.mean(0)
    s_c  = src_pts - mu_s
    d_c  = dst_pts - mu_d
    var_s = (s_c**2).sum() / n
    cov   = (d_c.T @ s_c) / n
    U, D, Vt = np.linalg.svd(cov)
    det_sgn = np.sign(np.linalg.det(U @ Vt))
    S_mat = np.diag([1.0, 1.0, det_sgn])
    R = U @ S_mat @ Vt
    scale = float((D * np.diag(S_mat)).sum() / var_s) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t

def align_trajectory(pred_poses, gt_poses, with_scale=True):
    """对齐预测轨迹到 GT，返回对齐后的预测轨迹"""
    # 截取相同帧数
    n = min(len(pred_poses), len(gt_poses))
    pred_t = pred_poses[:n, :3, 3]
    gt_t   = gt_poses[:n,  :3, 3]
    scale, R, t = umeyama_align(pred_t, gt_t, with_scale=with_scale)
    aligned = pred_poses[:n].copy()
    aligned[:, :3, 3] = (scale * (R @ pred_t.T)).T + t
    aligned[:, :3,:3] = R @ pred_poses[:n, :3, :3]
    return aligned, scale

# ════════════════════════════════════════════════════════════════
# 3. ATE（绝对轨迹误差）
# ════════════════════════════════════════════════════════════════
def compute_ate(pred_poses, gt_poses, with_scale=True):
    aligned, scale = align_trajectory(pred_poses, gt_poses, with_scale)
    n = len(aligned)
    errs = np.linalg.norm(aligned[:, :3, 3] - gt_poses[:n, :3, 3], axis=1)
    return {
        "rmse":   float(np.sqrt((errs**2).mean())),
        "mean":   float(errs.mean()),
        "median": float(np.median(errs)),
        "max":    float(errs.max()),
        "scale":  scale,
        "errors": errs,
    }

# ════════════════════════════════════════════════════════════════
# 4. RTE（相对轨迹误差，重点测长视频后段漂移）
# ════════════════════════════════════════════════════════════════
def compute_rte(pred_poses, gt_poses, delta=30):
    """
    delta: 相隔帧数（default=30，约1秒）
    分前半段/后半段分别报告，体现长视频漂移
    """
    n = min(len(pred_poses), len(gt_poses))
    rot_errs, trans_errs = [], []
    for i in range(0, n - delta, delta):
        dT_gt   = np.linalg.inv(gt_poses[i])   @ gt_poses[i+delta]
        dT_pred = np.linalg.inv(pred_poses[i]) @ pred_poses[i+delta]
        dT_err  = np.linalg.inv(dT_gt) @ dT_pred
        R_err   = dT_err[:3, :3]
        trace   = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_errs.append(float(np.degrees(np.arccos(trace))))
        trans_errs.append(float(np.linalg.norm(dT_err[:3, 3])))

    half = len(rot_errs) // 2
    return {
        "rot_mean":       float(np.mean(rot_errs)),
        "trans_mean":     float(np.mean(trans_errs)),
        # 后半段专项（体现长视频漂移）
        "rot_2nd_half":   float(np.mean(rot_errs[half:])),
        "trans_2nd_half": float(np.mean(trans_errs[half:])),
    }

# ════════════════════════════════════════════════════════════════
# 5. 尺度误差（仅 Variant C，有米制尺度时）
# ════════════════════════════════════════════════════════════════
def compute_scale_error(pred_poses, gt_poses):
    """
    用 SE(3) 对齐（不纠正尺度）后计算平均平移误差
    与 Sim(3) 对齐结果的差值即为"尺度误差贡献"
    """
    ate_sim3 = compute_ate(pred_poses, gt_poses, with_scale=True)
    ate_se3  = compute_ate(pred_poses, gt_poses, with_scale=False)
    return {
        "ate_sim3_rmse": ate_sim3["rmse"],   # 尺度被纠正后的误差
        "ate_se3_rmse":  ate_se3["rmse"],    # 未纠正尺度的误差
        "scale_error":   ate_se3["rmse"] - ate_sim3["rmse"],  # 尺度误差贡献
        "estimated_scale": ate_se3["scale"],
    }

# ════════════════════════════════════════════════════════════════
# 6. 汇总评测
# ════════════════════════════════════════════════════════════════
variants = {
    "A_VIPE_orig":    f"{args.results}/vipe_A/poses.npy",
    "B_Pi3X_only":    f"{args.results}/pi3x_B/poses.npy",
    "C_Pi3X_MoGe2":   f"{args.results}/pi3x_C/poses.npy",
}

print("\n" + "="*70)
print(f"{'指标':<28} {'A: VIPE原版':>14} {'B: Pi3X only':>14} {'C: Pi3X+MoGe2':>14}")
print("="*70)

all_results = {}
for name, path in variants.items():
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: 文件不存在 {path}")
        continue
    pred = np.load(path)
    # 截到与GT等长
    n = min(len(pred), len(gt_poses))
    pred, gt = pred[:n], gt_poses[:n]

    ate  = compute_ate(pred, gt, with_scale=True)
    rte  = compute_rte(pred, gt, delta=30)
    sclr = compute_scale_error(pred, gt)
    all_results[name] = {"ate": ate, "rte": rte, "scale": sclr}

# 打印核心指标
for row_name, key, sub in [
    ("ATE RMSE (Sim3, 全段↓)",     "ate",   "rmse"),
    ("ATE RMSE (SE3, 含尺度误差↓)", "scale", "ate_se3_rmse"),
    ("尺度误差贡献 (↓)",            "scale", "scale_error"),
    ("RTE 旋转均值° (↓)",           "rte",   "rot_mean"),
    ("RTE 平移均值  (↓)",           "rte",   "trans_mean"),
    ("RTE 后半段旋转° (漂移↓)",      "rte",   "rot_2nd_half"),
    ("RTE 后半段平移  (漂移↓)",      "rte",   "trans_2nd_half"),
]:
    vals = []
    for name in variants:
        if name in all_results:
            vals.append(f"{all_results[name][key][sub]:>14.4f}")
        else:
            vals.append(f"{'N/A':>14}")
    print(f"{row_name:<28} {'  '.join(vals)}")

print("="*70)

# ════════════════════════════════════════════════════════════════
# 7. 可视化：轨迹对比 + 逐帧误差曲线
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {"A_VIPE_orig": "tab:red", "B_Pi3X_only": "tab:orange", "C_Pi3X_MoGe2": "tab:blue"}
labels = {"A_VIPE_orig": "A: VIPE 原版", "B_Pi3X_only": "B: Pi3X only", "C_Pi3X_MoGe2": "C: Pi3X+MoGe-2 (SANA-WM)"}

# (左上) 俯视轨迹图
ax = axes[0, 0]
ax.set_title("轨迹俯视图 (X-Z 平面)", fontsize=11)
gt_t = gt_poses[:, :3, 3]
ax.plot(gt_t[:, 0], gt_t[:, 2], "k-", lw=2, label="GT", zorder=10)
for name, path in variants.items():
    if name not in all_results:
        continue
    pred = np.load(path)
    aligned, _ = align_trajectory(pred[:len(gt_poses)], gt_poses, with_scale=True)
    ax.plot(aligned[:, 0, 3], aligned[:, 2, 3],
            color=colors[name], lw=1.2, label=labels[name])
ax.legend(fontsize=8); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")

# (右上) 逐帧 ATE 误差曲线
ax = axes[0, 1]
ax.set_title("逐帧绝对平移误差 ATE (Sim3 对齐)", fontsize=11)
for name in variants:
    if name not in all_results:
        continue
    errs = all_results[name]["ate"]["errors"]
    ax.plot(errs, color=colors[name], lw=0.8, label=labels[name])
ax.axvline(x=len(errs)//2, color="gray", ls="--", lw=0.8, label="中点（后半段漂移分界）")
ax.legend(fontsize=8); ax.set_xlabel("帧序号"); ax.set_ylabel("误差 (m)")

# (左下) 柱状图：核心 ATE 对比
ax = axes[1, 0]
ax.set_title("ATE RMSE 对比（越低越好）", fontsize=11)
names_clean  = [labels[n] for n in variants if n in all_results]
ate_sim3_vals = [all_results[n]["ate"]["rmse"]          for n in variants if n in all_results]
ate_se3_vals  = [all_results[n]["scale"]["ate_se3_rmse"] for n in variants if n in all_results]
x = np.arange(len(names_clean))
ax.bar(x - 0.2, ate_sim3_vals, 0.4, label="ATE (Sim3, 纠正尺度)", alpha=0.8)
ax.bar(x + 0.2, ate_se3_vals,  0.4, label="ATE (SE3, 含尺度误差)", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(names_clean, fontsize=7, rotation=10)
ax.legend(fontsize=8); ax.set_ylabel("RMSE (m)")

# (右下) 尺度因子时序曲线（仅 C）
ax = axes[1, 1]
ax.set_title("尺度因子时序 (仅 C: Pi3X+MoGe-2)", fontsize=11)
scale_path = f"{args.results}/pi3x_C/scale_factors.npy"
if os.path.exists(scale_path):
    sf = np.load(scale_path)
    ax.plot(sf, color="tab:blue", lw=0.8)
    ax.axhline(sf.mean(), color="k", ls="--", lw=1, label=f"均值={sf.mean():.3f}")
    ax.fill_between(range(len(sf)), sf.mean()-sf.std(), sf.mean()+sf.std(),
                    alpha=0.2, color="tab:blue", label=f"±1σ (CV={sf.std()/sf.mean():.3f})")
    ax.legend(fontsize=8); ax.set_xlabel("帧序号"); ax.set_ylabel("尺度因子 s")
    ax.set_title(f"尺度因子时序 | CV={sf.std()/sf.mean():.4f}（越小越稳定）", fontsize=10)
else:
    ax.text(0.5, 0.5, "方案 C 未运行", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
os.makedirs(f"{args.results}/plots", exist_ok=True)
plt.savefig(f"{args.results}/plots/comparison.png", dpi=150)
print(f"\n图表已保存到 {args.results}/plots/comparison.png")
plt.show()
```

---

## 七、一键运行脚本

```bash
#!/bin/bash
# run_all.sh —— 顺序执行三路实验并评测
set -e

SEQ="data/tum/rgbd_dataset_freiburg1_desk"  # 改成 freiburg2_desk 做长视频测试

echo "=== Step 1: 方案 A —— 原始 VIPE ==="
vipe infer ${SEQ}/video.mp4 --output results/vipe_A --pipeline default

echo "=== Step 2: 方案 B —— Pi3X 仅 (相对尺度) ==="
python run_pi3x_moge.py --seq ${SEQ} --variant B --out results

echo "=== Step 3: 方案 C —— Pi3X + MoGe-2 (SANA-WM) ==="
python run_pi3x_moge.py --seq ${SEQ} --variant C --out results

echo "=== Step 4: 评测 ==="
python evaluate.py --seq ${SEQ} --results results
```

---

## 八、预期结果与解读

### 8.1 数字预期（基于各论文报告值的推断）

```
指标                        A: VIPE原版    B: Pi3X only   C: Pi3X+MoGe2
─────────────────────────  ─────────────  ─────────────  ─────────────
ATE RMSE Sim3↓ (m)         ~0.025         ~0.030         ~0.020
ATE RMSE SE3↓  (m)         ~0.030         ~0.150         ~0.022   ← C 的尺度准
尺度误差贡献↓  (m)          ~0.005         ~0.120         ~0.002
RTE 后半段旋转°↓           ~1.8           ~2.5           ~1.2
RTE 后半段平移↓             ~0.018         ~0.080         ~0.012
```

数字仅供参考，实际结果因序列而异。核心模式应该是：
- **ATE SE3**：方案 C 远优于 B（MoGe-2 的米制尺度锚点发挥作用）
- **RTE 后半段**：方案 C 优于 A（Pi3X 长序列一致性减少漂移）
- **尺度 CV**：方案 C 的尺度因子时序图应平稳（低 CV），体现 EMA 平滑效果

### 8.2 结果解读指南

| 观察到的现象 | 说明 |
|---|---|
| C 的 ATE-SE3 ≈ ATE-Sim3 | MoGe-2 米制尺度准确，与 GT 尺度高度匹配 ✓ |
| B 的 ATE-SE3 >> ATE-Sim3 | Pi3X 相对尺度存在漂移，需要 MoGe-2 锚点 ✓ |
| C 后半段 RTE 低于 A | Pi3X 长序列一致深度减少了长视频漂移 ✓ |
| 尺度因子 CV < 0.2 | EMA 平滑后尺度稳定，SANA-WM 过滤阈值 2.0 合理 ✓ |

### 8.3 与 SANA-WM 论文的对应关系

| 实验结果 | 对应论文声明 |
|---|---|
| ATE SE3 改善 | "Pi3X provides long-sequence-consistent depth" |
| 尺度误差减少 | "MoGe-2 provides accurate per-frame metric scale" |
| 尺度 CV 时序稳定 | EMA smoothing (momentum 0.99) 的效果可视化 |

---

## 九、注意事项

1. **VIPE 输出格式**：`vipe infer` 的输出目录里有 `poses.npy`，格式可能需要根据实际输出做调整（参考 `vipe_results/` 目录结构）。

2. **Pi3X 的帧数限制**：Pi3X 每次处理 N 帧（通常 8~16），超长视频需要滑窗，重叠帧数 ≥ 2 以保证轨迹连续。

3. **GT 帧对齐**：TUM 的 RGB 帧和 GT 位姿时间戳不完全对应，用 `associate.py` 做最近邻匹配，容忍误差 0.02s。

4. **MoGe-2 可用性**：如果 `microsoft/MoGe-vitl` 仅有 MoGe-1，可用 MoGe-1 代替——它同样提供米制深度，只是在细节精度上略低，不影响实验的宏观结论。

5. **fr2/desk 长视频实验**：这是最能体现 SANA-WM 声明的场景。前 500 帧 A/B/C 差距可能不明显，1500 帧后差距会拉大——这正是 Pi3X 长序列一致性的价值所在。

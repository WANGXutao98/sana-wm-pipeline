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

import matplotlib
matplotlib.use("Agg")  # 无 display 环境下保存图片
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
    """
    d = np.load(npz_path)
    poses = d["data"].astype(np.float32)   # (T, 4, 4)
    inds  = d["inds"]                       # (T,)
    T_full = int(inds.max()) + 1
    if len(poses) == T_full:
        return poses
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
    """ATE with Sim(3) alignment."""
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
    """RTE：相隔 delta 帧的相对运动误差，分前后半段。"""
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

METHODS = {
    "A: VIPE + metric3d-small (论文 baseline)": RESULTS / "method_A_m3d" / "pose" / "video.npz",
    "B: VIPE + Pi3X+MoGe-2 (cached, SANA-WM)": RESULTS / "method_B_cached" / "pose" / "video.npz",
}

METHOD_COLORS = {
    "A: VIPE + metric3d-small (论文 baseline)": "tab:red",
    "B: VIPE + Pi3X+MoGe-2 (cached, SANA-WM)": "tab:green",
}

results: dict[str, dict] = {}
for name, npz_path in METHODS.items():
    if not npz_path.exists():
        print(f"[skip] {name}: {npz_path} not found")
        continue
    pred = load_vipe_poses(npz_path)
    ate  = compute_ate(pred, gt_poses)
    rte  = compute_rte(pred, gt_poses, delta=30)
    results[name] = {"ate": ate, "rte": rte, "path": npz_path}
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
    print(f"  RTE 后半漂移旋转: {rte['rot_2nd_half']:.3f}°")
    print(f"  RTE 后半漂移平移: {rte['trans_2nd_half']:.4f} m")

if len(results) < 2:
    print("\n[warn] 少于两组结果，跳过对比图（运行完推理后重新执行）")
else:
    n_methods = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 轨迹俯视图 ──
    ax = axes[0]
    ax.set_title("轨迹对比（俯视 X-Z 平面）", fontsize=10)
    gt_t = gt_poses[:, :3, 3]
    ax.plot(gt_t[:, 0], gt_t[:, 2], "k-", lw=2, label="GT", zorder=10)
    for name, data in results.items():
        pred = load_vipe_poses(data["path"])
        n = min(len(pred), len(gt_poses))
        aligned, _ = align_to_gt(pred[:n], gt_poses[:n])
        ax.plot(aligned[:, 0, 3], aligned[:, 2, 3],
                color=METHOD_COLORS[name], lw=1.2, label=name.split("(")[0].strip(), alpha=0.8)
    ax.legend(fontsize=7); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")

    # ── 逐帧 ATE ──
    ax = axes[1]
    ax.set_title("逐帧 ATE（Sim3 对齐）", fontsize=10)
    for name, data in results.items():
        errs = data["ate"]["per_frame"]
        ax.plot(errs, color=METHOD_COLORS[name], lw=0.8,
                label=name.split("(")[0].strip(), alpha=0.9)
    half_n = len(list(results.values())[0]["ate"]["per_frame"]) // 2
    ax.axvline(x=half_n, color="gray", ls="--", lw=0.8, label="中点")
    ax.legend(fontsize=7); ax.set_xlabel("帧序号"); ax.set_ylabel("误差 (m)")

    # ── ATE RMSE 柱状图 ──
    ax = axes[2]
    ax.set_title("ATE RMSE（越低越好）", fontsize=10)
    names = list(results.keys())
    vals  = [results[n]["ate"]["rmse"] for n in names]
    x     = np.arange(len(names))
    bars  = ax.bar(x, vals, color=[METHOD_COLORS[n] for n in names], alpha=0.8)
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=7)
    ax.set_ylabel("RMSE (m)")

    plt.tight_layout()
    out_fig = PLOT_DIR / "comparison.png"
    plt.savefig(out_fig, dpi=150)
    print(f"\n[plot] 图表保存到 {out_fig}")

    # ── 汇总表 ──
    col_names = list(results.keys())
    header_width = 28
    col_width = 16
    print("\n" + "=" * (header_width + col_width * len(col_names) + 2))
    header = f"{'指标':<{header_width}}" + "".join(f"{n.split('(')[0].strip():>{col_width}}" for n in col_names)
    print(header)
    print("=" * (header_width + col_width * len(col_names) + 2))
    rows = [
        ("ATE RMSE (Sim3) ↓ (m)",   "ate", "rmse"),
        ("ATE mean ↓ (m)",           "ate", "mean"),
        ("ATE max ↓ (m)",            "ate", "max"),
        ("估计尺度 (→1.0)",           "ate", "scale"),
        ("RTE 旋转均值 ↓ (°)",        "rte", "rot_mean"),
        ("RTE 平移均值 ↓ (m)",        "rte", "trans_mean"),
        ("RTE 后半旋转 ↓ (°)",        "rte", "rot_2nd_half"),
        ("RTE 后半平移 ↓ (m)",        "rte", "trans_2nd_half"),
    ]
    for label, key, sub in rows:
        vals_row = [results[n][key][sub] for n in col_names]
        row_str = f"{label:<{header_width}}"
        for v in vals_row:
            row_str += f"{v:>{col_width}.4f}"
        print(row_str)
    print("=" * (header_width + col_width * len(col_names) + 2))

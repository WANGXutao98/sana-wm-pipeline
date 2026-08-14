#!/usr/bin/env python
"""SpatialVID冒烟测试质量检查脚本

用法：
    python scripts/validate_smoke_output.py \
        --output-dir /path/to/smoke_result \
        --samples selected_samples.txt
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ValidationReport:
    """单样本验证报告"""
    sample_id: str
    n_frames: int

    # Pose检查
    pose_shape_ok: bool
    rotation_orthogonal: bool
    rotation_max_error: float
    first_frame_identity_error: float
    translation_smoothness_max: float
    trajectory_length: float

    # 与VIPE对比
    vipe_trajectory_length: Optional[float]
    trajectory_length_ratio: Optional[float]
    rotation_error_deg: Optional[float]
    translation_error_m: Optional[float]

    # 内参检查
    intrinsics_shape_ok: bool
    focal_aspect_ratio_ok: bool
    focal_range_ok: bool
    principal_point_ok: bool
    focal_consistency_ok: bool
    focal_diff_pct: Optional[float]

    # Scale检查
    scale_all_ones: bool
    scale_mean: float
    scale_std: float

    # 完整性
    shard_exists: bool

    def summary(self) -> str:
        """生成可读报告"""
        lines = [f"\n{'='*70}"]
        lines.append(f"样本: {self.sample_id} ({self.n_frames}帧)")
        lines.append(f"{'='*70}")

        # Pose检查
        lines.append("\n[Pose检查]")
        lines.append(f"  {'✅' if self.pose_shape_ok else '❌'} Shape: {self.pose_shape_ok}")
        lines.append(f"  {'✅' if self.rotation_orthogonal else '❌'} 旋转矩阵正交性: max_error={self.rotation_max_error:.2e}")
        lines.append(f"  {'✅' if self.first_frame_identity_error < 1e-3 else '⚠️ '} 第一帧归一化: |pose[0]-I|={self.first_frame_identity_error:.4f}")
        lines.append(f"  {'✅' if self.translation_smoothness_max < 0.01 else '⚠️ '} 平移平滑性: max_Δt={self.translation_smoothness_max:.6f}")
        lines.append(f"  轨迹长度: {self.trajectory_length:.4f}")

        if self.vipe_trajectory_length is not None:
            ratio_icon = '⚠️ ' if self.trajectory_length_ratio > 2.0 else '✅'
            lines.append(f"  {ratio_icon} vs VIPE参考: {self.vipe_trajectory_length:.4f} (比例={self.trajectory_length_ratio:.2f}x)")

        if self.rotation_error_deg is not None:
            rot_icon = '✅' if self.rotation_error_deg < 5.0 else '⚠️ '
            lines.append(f"  {rot_icon} 旋转误差: {self.rotation_error_deg:.2f}°")

        if self.translation_error_m is not None:
            trans_icon = '✅' if self.translation_error_m < 0.05 else '⚠️ '
            lines.append(f"  {trans_icon} 平移误差(ATE): {self.translation_error_m:.4f}m")

        # 内参检查
        lines.append("\n[内参检查]")
        lines.append(f"  {'✅' if self.intrinsics_shape_ok else '❌'} Shape: {self.intrinsics_shape_ok}")
        lines.append(f"  {'✅' if self.focal_aspect_ratio_ok else '⚠️ '} fx≈fy: {self.focal_aspect_ratio_ok}")
        lines.append(f"  {'✅' if self.focal_range_ok else '⚠️ '} 焦距范围: {self.focal_range_ok}")
        lines.append(f"  {'✅' if self.principal_point_ok else '⚠️ '} 主点位置: {self.principal_point_ok}")
        lines.append(f"  {'✅' if self.focal_consistency_ok else '⚠️ '} 时序一致性: {self.focal_consistency_ok}")

        if self.focal_diff_pct is not None:
            focal_icon = '✅' if self.focal_diff_pct < 5.0 else '⚠️ '
            lines.append(f"  {focal_icon} vs VIPE焦距: 差异={self.focal_diff_pct:.1f}%")

        # Scale检查
        lines.append("\n[Scale检查]")
        scale_icon = '⚠️ ' if self.scale_all_ones else '✅'
        lines.append(f"  {scale_icon} 全为1.0: {self.scale_all_ones}")
        lines.append(f"  mean={self.scale_mean:.4f}, std={self.scale_std:.4f}")

        # 完整性
        lines.append("\n[完整性]")
        lines.append(f"  {'✅' if self.shard_exists else '❌'} Shard文件: {self.shard_exists}")

        return '\n'.join(lines)


def load_our_output(sample_dir: Path):
    """加载我们的输出"""
    artifact_json = sample_dir / "pose_artifact_default.json"
    art = json.loads(artifact_json.read_text())

    poses = np.array(art['poses_c2w'], np.float32)
    intr = np.array(art['intrinsics'], np.float32)
    scale = np.array(art['scale_per_frame'], np.float32)

    return poses, intr, scale


def load_vipe_reference(camera_file: Path):
    """加载VIPE参考标注"""
    data = np.load(camera_file)
    return {
        'c2w': data['vipe_c2w'],
        'K_px': data['vipe_K_px'],
    }


def check_rotation_orthogonality(poses: np.ndarray) -> tuple[bool, float]:
    """检查旋转矩阵正交性"""
    R = poses[:, :3, :3]
    I_target = np.eye(3)[None, :, :]
    RRT = R @ R.transpose(0, 2, 1)
    errors = np.abs(RRT - I_target)
    max_error = errors.max()

    is_valid = max_error < 1e-4
    return is_valid, float(max_error)


def compute_trajectory_length(poses: np.ndarray) -> float:
    """计算轨迹总长度"""
    translations = poses[:, :3, 3]
    diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    return float(diffs.sum())


def compute_translation_smoothness(poses: np.ndarray) -> float:
    """计算平移平滑性（最大帧间位移）"""
    translations = poses[:, :3, 3]
    diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    return float(diffs.max())


def umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Umeyama对齐算法（用于对齐两组3D点）

    Args:
        src_pts: (N, 3)
        dst_pts: (N, 3)

    Returns:
        s: scale
        R: (3, 3) rotation
        t: (3,) translation
    """
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    src_var = (src_centered ** 2).sum() / len(src_pts)
    s = S.sum() / src_var if src_var > 0 else 1.0

    t = dst_mean - s * R @ src_mean

    return s, R, t


def compare_with_vipe(our_poses: np.ndarray, vipe_ref: dict, original_fps: int = 25, our_fps: int = 16) -> tuple:
    """与VIPE参考标注对比

    Args:
        our_poses: (T_ours, 4, 4) 我们的poses (归一化后16fps)
        vipe_ref: VIPE参考 (原始25fps)
        original_fps: 原始fps
        our_fps: 归一化后fps

    Returns:
        rotation_error_deg, translation_error_m
    """
    vipe_c2w = vipe_ref['c2w']

    # 由于帧率不同，采样VIPE参考以匹配我们的帧数
    T_our = len(our_poses)
    T_vipe = len(vipe_c2w)

    # 时间对齐：our的第i帧对应vipe的哪一帧
    # our: 0, 1, 2, ..., T_our-1 @ 16fps
    # vipe: 0, 1, 2, ..., T_vipe-1 @ 25fps
    # 假设视频长度一致
    vipe_indices = np.linspace(0, T_vipe - 1, T_our).round().astype(int)
    vipe_sampled = vipe_c2w[vipe_indices]

    # 提取平移部分做Umeyama对齐
    our_t = our_poses[:, :3, 3]
    vipe_t = vipe_sampled[:, :3, 3]

    s, R_align, t_align = umeyama_alignment(our_t, vipe_t)

    # 对齐后的轨迹
    our_t_aligned = s * (R_align @ our_t.T).T + t_align

    # 计算ATE (Average Translation Error)
    ate = np.linalg.norm(our_t_aligned - vipe_t, axis=1).mean()

    # 旋转误差：比较旋转矩阵
    our_R = our_poses[:, :3, :3]
    vipe_R = vipe_sampled[:, :3, :3]

    # R_error = our_R @ vipe_R.T，计算相对旋转的角度
    R_errors = []
    for i in range(len(our_R)):
        R_diff = our_R[i] @ vipe_R[i].T
        # 从旋转矩阵提取角度：angle = arccos((trace(R)-1)/2)
        trace = np.trace(R_diff)
        # 限制trace到[-1, 3]避免数值误差
        trace = np.clip(trace, -1, 3)
        angle_rad = np.arccos((trace - 1) / 2)
        R_errors.append(np.rad2deg(angle_rad))

    mean_rot_error = np.mean(R_errors)

    return float(mean_rot_error), float(ate)


def validate_sample(sample_id: str, n_frames: int, output_dir: Path, raw_dir: Path) -> ValidationReport:
    """验证单个样本"""
    sample_dir = output_dir / sample_id
    camera_file = raw_dir / f"{sample_id}.camera.npz"

    # 加载数据
    our_poses, our_intr, our_scale = load_our_output(sample_dir)

    vipe_ref = None
    if camera_file.exists():
        vipe_ref = load_vipe_reference(camera_file)

    # Pose检查
    pose_shape_ok = our_poses.shape[1:] == (4, 4)
    rotation_orthogonal, rotation_max_error = check_rotation_orthogonality(our_poses)

    first_frame_identity_error = float(np.abs(our_poses[0] - np.eye(4)).max())
    translation_smoothness_max = compute_translation_smoothness(our_poses)
    trajectory_length = compute_trajectory_length(our_poses)

    # 与VIPE对比
    vipe_trajectory_length = None
    trajectory_length_ratio = None
    rotation_error_deg = None
    translation_error_m = None

    if vipe_ref is not None:
        vipe_trajectory_length = compute_trajectory_length(vipe_ref['c2w'])
        trajectory_length_ratio = trajectory_length / vipe_trajectory_length if vipe_trajectory_length > 0 else None
        rotation_error_deg, translation_error_m = compare_with_vipe(our_poses, vipe_ref)

    # 内参检查
    intrinsics_shape_ok = our_intr.shape[1:] == (1, 4)

    fx_fy_cx_cy = our_intr[:, 0, :]
    fx = fx_fy_cx_cy[:, 0]
    fy = fx_fy_cx_cy[:, 1]
    cx = fx_fy_cx_cy[:, 2]
    cy = fx_fy_cx_cy[:, 3]

    focal_aspect_ratio_ok = np.all(np.abs(fx - fy) / fx < 0.01)
    focal_range_ok = np.all((fx > 400) & (fx < 1200))
    principal_point_ok = np.all((np.abs(cx - 640) < 50) & (np.abs(cy - 360) < 50))

    focal_diffs = np.abs(np.diff(fx))
    focal_consistency_ok = np.all(focal_diffs < 10)

    focal_diff_pct = None
    if vipe_ref is not None:
        vipe_fx = vipe_ref['K_px'][0, 0]
        our_fx_mean = fx.mean()
        focal_diff_pct = abs(our_fx_mean - vipe_fx) / vipe_fx * 100

    # Scale检查
    scale_all_ones = np.allclose(our_scale, 1.0)
    scale_mean = float(our_scale.mean())
    scale_std = float(our_scale.std())

    # 完整性
    shard_exists = (sample_dir / f"{sample_id}.tar").exists()

    return ValidationReport(
        sample_id=sample_id,
        n_frames=len(our_poses),
        pose_shape_ok=pose_shape_ok,
        rotation_orthogonal=rotation_orthogonal,
        rotation_max_error=rotation_max_error,
        first_frame_identity_error=first_frame_identity_error,
        translation_smoothness_max=translation_smoothness_max,
        trajectory_length=trajectory_length,
        vipe_trajectory_length=vipe_trajectory_length,
        trajectory_length_ratio=trajectory_length_ratio,
        rotation_error_deg=rotation_error_deg,
        translation_error_m=translation_error_m,
        intrinsics_shape_ok=intrinsics_shape_ok,
        focal_aspect_ratio_ok=focal_aspect_ratio_ok,
        focal_range_ok=focal_range_ok,
        principal_point_ok=principal_point_ok,
        focal_consistency_ok=focal_consistency_ok,
        focal_diff_pct=focal_diff_pct,
        scale_all_ones=scale_all_ones,
        scale_mean=scale_mean,
        scale_std=scale_std,
        shard_exists=shard_exists,
    )


def main():
    parser = argparse.ArgumentParser(description="SpatialVID冒烟测试质量检查")
    parser.add_argument("--output-dir", required=True, help="smoke_result输出目录")
    parser.add_argument("--samples", required=True, help="selected_samples.txt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples_file = Path(args.samples)
    raw_dir = output_dir / "raw_samples"

    # 读取样本列表
    samples = []
    with samples_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            sample_id, n_frames = parts[0], int(parts[1])
            samples.append((sample_id, n_frames))

    print(f"\n{'='*70}")
    print(f"SpatialVID冒烟测试质量报告")
    print(f"{'='*70}")
    print(f"共 {len(samples)} 个样本")

    reports = []
    for sample_id, n_frames in samples:
        report = validate_sample(sample_id, n_frames, output_dir, raw_dir)
        reports.append(report)
        print(report.summary())

    # 汇总统计
    print(f"\n{'='*70}")
    print(f"汇总统计")
    print(f"{'='*70}")

    total_checks = 0
    passed_checks = 0
    warnings = []

    for report in reports:
        checks = [
            ("Pose shape", report.pose_shape_ok),
            ("旋转正交性", report.rotation_orthogonal),
            ("第一帧归一化", report.first_frame_identity_error < 1e-3),
            ("平移平滑性", report.translation_smoothness_max < 0.01),
            ("内参shape", report.intrinsics_shape_ok),
            ("焦距长宽比", report.focal_aspect_ratio_ok),
            ("焦距范围", report.focal_range_ok),
            ("主点位置", report.principal_point_ok),
            ("焦距一致性", report.focal_consistency_ok),
            ("Shard存在", report.shard_exists),
        ]

        for name, passed in checks:
            total_checks += 1
            if passed:
                passed_checks += 1

        if report.trajectory_length_ratio and report.trajectory_length_ratio > 2.0:
            warnings.append(f"{report.sample_id}: 轨迹长度比VIPE大{report.trajectory_length_ratio:.1f}倍")

        if report.scale_all_ones:
            warnings.append(f"{report.sample_id}: Scale全为1.0")

        if report.rotation_error_deg and report.rotation_error_deg > 5.0:
            warnings.append(f"{report.sample_id}: 旋转误差{report.rotation_error_deg:.1f}°")

    print(f"\n通过: {passed_checks}/{total_checks} 检查项")
    print(f"警告: {len(warnings)}项")

    if warnings:
        print("\n⚠️  警告列表:")
        for w in warnings:
            print(f"  - {w}")

    if passed_checks == total_checks and len(warnings) == 0:
        print("\n✅ 全部检查通过！")
        sys.exit(0)
    elif passed_checks >= total_checks * 0.8:
        print("\n⚠️  大部分检查通过，有少量警告")
        sys.exit(0)
    else:
        print("\n❌ 存在严重问题，需要调查")
        sys.exit(1)


if __name__ == "__main__":
    main()

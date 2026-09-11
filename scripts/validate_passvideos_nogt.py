#!/usr/bin/env python3
"""无GT验证 - 只检查内部一致性和物理合理性"""
import argparse
import json
import numpy as np
from pathlib import Path


def check_sample(sample_dir):
    """检查单个样本的质量指标"""
    artifact_path = sample_dir / "pose_artifact_default.json"
    if not artifact_path.exists():
        return None

    with open(artifact_path) as f:
        data = json.load(f)

    # ponytail: 字段名对齐（poses_c2w不是poses）
    poses = np.array(data["poses_c2w"])  # (T, 4, 4)
    intrinsics = np.array(data["intrinsics"])  # (T, 1, 4)
    scales = np.array(data["scale_per_frame"])  # (T,)
    T = len(poses)

    results = {"sample": sample_dir.name, "frames": T}

    # 1. 旋转正交性
    R_matrices = poses[:, :3, :3]
    ortho_errors = []
    for R in R_matrices:
        should_be_I = R @ R.T
        error = np.abs(should_be_I - np.eye(3)).max()
        ortho_errors.append(error)
    results["ortho_max"] = max(ortho_errors)
    results["ortho_pass"] = results["ortho_max"] < 1e-5

    # 2. Scale CoV（变异系数）
    scale_cov = scales.std() / scales.mean() if scales.mean() > 0 else 999
    results["scale_cov"] = scale_cov
    results["scale_cov_pass"] = scale_cov < 2.0

    # 3. 轨迹长度（物理合理性）
    translations = poses[:, :3, 3]
    traj_length = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
    results["traj_length"] = traj_length
    # 10秒视频，相机移动应该在0.1m-50m之间（静态-快速移动）
    results["traj_reasonable"] = 0.1 <= traj_length <= 50.0

    # 4. 焦距范围
    fx_values = intrinsics[:, 0, 0]
    results["fx_mean"] = fx_values.mean()
    results["fx_range"] = (fx_values.min(), fx_values.max())
    # 归一化后1280x720，焦距应该在500-1500之间
    results["fx_reasonable"] = 500 <= results["fx_mean"] <= 1500

    # 5. 输出完整性
    results["has_shard"] = (sample_dir / f"{sample_dir.name}.tar").exists()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples_file = Path(args.samples)

    # 读取样本列表
    sample_ids = []
    with open(samples_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                sample_ids.append(parts[0])

    print(f"验证 {len(sample_ids)} 个样本（无GT模式）\n")

    results_list = []
    for sample_id in sample_ids:
        sample_dir = output_dir / sample_id
        result = check_sample(sample_dir)
        if result:
            results_list.append(result)

    if not results_list:
        print("无有效样本")
        return

    # 统计汇总
    print(f"{'='*80}")
    print(f"质量检查汇总（{len(results_list)}个样本）")
    print(f"{'='*80}\n")

    ortho_pass = sum(1 for r in results_list if r["ortho_pass"])
    scale_pass = sum(1 for r in results_list if r["scale_cov_pass"])
    traj_pass = sum(1 for r in results_list if r["traj_reasonable"])
    fx_pass = sum(1 for r in results_list if r["fx_reasonable"])
    shard_pass = sum(1 for r in results_list if r["has_shard"])

    print(f"1. 旋转正交性: {ortho_pass}/{len(results_list)} 通过 (<1e-5)")
    ortho_max = max(r["ortho_max"] for r in results_list)
    print(f"   最大误差: {ortho_max:.2e}\n")

    print(f"2. Scale CoV: {scale_pass}/{len(results_list)} 通过 (<2.0)")
    scale_covs = [r["scale_cov"] for r in results_list]
    print(f"   范围: {min(scale_covs):.4f} - {max(scale_covs):.4f}")
    print(f"   中位数: {np.median(scale_covs):.4f}\n")

    print(f"3. 轨迹长度合理性: {traj_pass}/{len(results_list)} 通过 (0.1-50m)")
    traj_lengths = [r["traj_length"] for r in results_list]
    print(f"   范围: {min(traj_lengths):.2f} - {max(traj_lengths):.2f}m")
    print(f"   中位数: {np.median(traj_lengths):.2f}m\n")

    print(f"4. 焦距合理性: {fx_pass}/{len(results_list)} 通过 (500-1500)")
    fx_means = [r["fx_mean"] for r in results_list]
    print(f"   范围: {min(fx_means):.1f} - {max(fx_means):.1f}")
    print(f"   中位数: {np.median(fx_means):.1f}\n")

    print(f"5. Shard完整性: {shard_pass}/{len(results_list)} 生成\n")

    # 异常样本
    print(f"{'='*80}")
    print("异常样本（任一检查失败）:")
    print(f"{'='*80}")
    abnormal = [r for r in results_list if not (
        r["ortho_pass"] and r["scale_cov_pass"] and
        r["traj_reasonable"] and r["fx_reasonable"] and r["has_shard"]
    )]
    if abnormal:
        for r in abnormal[:10]:  # 只显示前10个
            issues = []
            if not r["ortho_pass"]: issues.append(f"ortho={r['ortho_max']:.2e}")
            if not r["scale_cov_pass"]: issues.append(f"cov={r['scale_cov']:.2f}")
            if not r["traj_reasonable"]: issues.append(f"traj={r['traj_length']:.2f}m")
            if not r["fx_reasonable"]: issues.append(f"fx={r['fx_mean']:.1f}")
            if not r["has_shard"]: issues.append("no_shard")
            print(f"  {r['sample']}: {', '.join(issues)}")
        if len(abnormal) > 10:
            print(f"  ... 还有 {len(abnormal)-10} 个异常样本")
    else:
        print("  无异常样本 ✅")

    # 保存结果（ponytail: 转换numpy bool为Python bool）
    output_json = output_dir / "validation_results_nogt.json"
    with open(output_json, 'w') as f:
        # 转换numpy类型为Python原生类型
        json_results = []
        for r in results_list:
            json_r = {}
            for k, v in r.items():
                if isinstance(v, (np.bool_, np.generic)):
                    json_r[k] = bool(v)
                elif isinstance(v, tuple):
                    json_r[k] = [float(x) for x in v]
                elif isinstance(v, (np.floating, np.integer)):
                    json_r[k] = float(v)
                else:
                    json_r[k] = v
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    print(f"\n详细结果已保存: {output_json}")


if __name__ == "__main__":
    main()

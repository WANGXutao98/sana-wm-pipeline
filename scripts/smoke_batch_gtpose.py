#!/usr/bin/env python3
"""DL3DV GT-Pose 模式批量冒烟测试

用法：
    python scripts/smoke_batch_gtpose.py \
        --samples /path/to/samples.tsv \
        --dl3dv-dir /path/to/Dl3dv \
        --output-dir /path/to/output
"""
import argparse
import json
import sys
import time
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sana_wm_pipeline.stage02_pose.mode_gtpose import run_gtpose


def process_sample(
    sample_id: str,
    n_frames: int,
    dl3dv_dir: Path,
    output_dir: Path
) -> dict:
    """处理单个 DL3DV 样本

    Args:
        sample_id: 样本 ID（不含扩展名）
        n_frames: 帧数（用于验证）
        dl3dv_dir: DL3DV 数据目录
        output_dir: 输出目录

    Returns:
        结果字典，包含 status, elapsed_time 等
    """
    print("=" * 60)
    print(f"样本: {sample_id}")
    print(f"预期帧数: {n_frames}")
    print("=" * 60)

    # 输入文件
    video_path = dl3dv_dir / f"{sample_id}.mp4"
    camera_path = dl3dv_dir / f"{sample_id}.camera.npz"

    # 验证文件存在
    if not video_path.exists():
        return {
            "sample_id": sample_id,
            "status": "failed",
            "error": f"Video file not found: {video_path.name}",
        }
    if not camera_path.exists():
        return {
            "sample_id": sample_id,
            "status": "failed",
            "error": f"Camera file not found: {camera_path.name}",
        }

    # 工作目录
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 开始计时
    start_time = time.time()

    try:
        # 从 camera.npz 提取 GT poses
        print("--- 加载 GT 数据 ---")
        camera_data = np.load(camera_path)
        gt_c2w = camera_data['c2w']  # (N, 4, 4)
        gt_K_px = camera_data.get('K_px', None)  # (N, 4) or None

        actual_frames = gt_c2w.shape[0]
        print(f"  GT poses (c2w): {gt_c2w.shape}")
        if gt_K_px is not None:
            print(f"  GT intrinsics (K_px): {gt_K_px.shape}")
        else:
            print(f"  GT intrinsics: None (将使用 seed intrinsics)")

        # 验证帧数
        if actual_frames != n_frames:
            print(f"  ⚠️  实际帧数 {actual_frames} != 预期 {n_frames}")

        # 保存为临时 GT poses 文件
        temp_gt_poses = sample_dir / "gt_poses.npy"
        np.save(temp_gt_poses, gt_c2w)
        print(f"  临时 GT poses 已保存: {temp_gt_poses.name}")

        # 运行 GT-pose 模式
        print("\n--- Stage 2: GT-Pose 模式 ---")
        artifact = run_gtpose(
            clip_path=video_path,
            gt_poses_path=temp_gt_poses,
            work_dir=sample_dir,
            inlier_percentile=80.0,
            gt_intrinsics_path=camera_path,  # 显式传入 camera.npz 路径
        )

        elapsed = time.time() - start_time

        # 保存结果
        print("\n--- 保存结果 ---")
        output_poses = sample_dir / "poses.npy"
        output_intrinsics = sample_dir / "intrinsics.npy"
        output_scale = sample_dir / "scale_per_frame.npy"

        np.save(output_poses, artifact.poses_c2w)
        np.save(output_intrinsics, artifact.intrinsics)
        np.save(output_scale, artifact.scale_per_frame)

        print(f"  poses.npy: {artifact.poses_c2w.shape}")
        print(f"  intrinsics.npy: {artifact.intrinsics.shape}")
        print(f"  scale_per_frame.npy: {artifact.scale_per_frame.shape}")

        # 生成报告
        result = {
            "sample_id": sample_id,
            "status": "success",
            "elapsed_time": f"{elapsed:.2f}s",
            "expected_frames": n_frames,
            "actual_frames": actual_frames,
            "output_poses_shape": str(artifact.poses_c2w.shape),
            "output_intrinsics_shape": str(artifact.intrinsics.shape),
            "output_scale_shape": str(artifact.scale_per_frame.shape),
            "scale_mean": float(artifact.scale_per_frame.mean()),
            "scale_std": float(artifact.scale_per_frame.std()),
            "scale_min": float(artifact.scale_per_frame.min()),
            "scale_max": float(artifact.scale_per_frame.max()),
        }

        print(f"\n✅ 处理成功")
        print(f"   耗时: {elapsed:.2f}s")
        print(f"   Scale: {result['scale_mean']:.6f} ± {result['scale_std']:.6f}")
        print(f"   Scale 范围: [{result['scale_min']:.6f}, {result['scale_max']:.6f}]")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            "sample_id": sample_id,
            "status": "failed",
            "error": str(e),
            "elapsed_time": f"{elapsed:.2f}s",
            "expected_frames": n_frames,
        }


def main():
    parser = argparse.ArgumentParser(
        description="DL3DV GT-Pose 模式批量冒烟测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/smoke_batch_gtpose.py \\
        --samples /path/to/samples.tsv \\
        --dl3dv-dir /mnt/afs/davidwang/workspace/sana_test_data/Dl3dv \\
        --output-dir /mnt/afs/davidwang/workspace/sana_test_data/smoke_result_dl3dv

samples.tsv 格式:
    sample_id<TAB>n_frames
    DL3DV-ALL-2K_10K__a1cc9c41...__images_2<TAB>303
    DL3DV-ALL-2K_10K__b137b3eb...__images_2<TAB>303
"""
    )
    parser.add_argument("--samples", required=True, help="样本列表文件 (TSV 格式)")
    parser.add_argument("--dl3dv-dir", required=True, help="DL3DV 数据目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()

    samples_file = Path(args.samples)
    dl3dv_dir = Path(args.dl3dv_dir)
    output_dir = Path(args.output_dir)

    # 验证路径
    if not samples_file.exists():
        print(f"❌ 样本列表文件不存在: {samples_file}")
        sys.exit(1)
    if not dl3dv_dir.exists():
        print(f"❌ DL3DV 目录不存在: {dl3dv_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取样本列表
    samples = []
    with samples_file.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"⚠️  行 {line_num}: 格式错误，跳过: {line}")
                continue
            sample_id, n_frames = parts[0], int(parts[1])
            samples.append((sample_id, n_frames))

    if not samples:
        print("❌ 没有有效的样本")
        sys.exit(1)

    # 打印测试配置
    print("=" * 60)
    print("DL3DV GT-Pose 模式批量测试")
    print("=" * 60)
    print(f"样本数量: {len(samples)}")
    print(f"DL3DV 目录: {dl3dv_dir}")
    print(f"输出目录: {output_dir}")
    print(f"Pi3X 最大帧数: {os.environ.get('SANA_WM_MAX_FRAMES', '64')}")
    print("=" * 60)
    print()

    # 批量处理（单进程，模型只加载一次，利用 @lru_cache）
    results = []
    success_count = 0
    fail_count = 0

    for i, (sample_id, n_frames) in enumerate(samples, 1):
        print(f"\n进度: [{i}/{len(samples)}]")
        result = process_sample(sample_id, n_frames, dl3dv_dir, output_dir)
        results.append(result)

        if result["status"] == "success":
            success_count += 1
        else:
            fail_count += 1

    # 保存结果摘要
    summary_file = output_dir / "smoke_test_summary.json"
    summary = {
        "total_samples": len(samples),
        "successful": success_count,
        "failed": fail_count,
        "results": results,
    }

    with summary_file.open('w') as f:
        json.dump(summary, f, indent=2)

    # 打印总结
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总样本数: {len(samples)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"\n结果已保存到: {summary_file}")

    if fail_count > 0:
        print(f"\n❌ 有 {fail_count} 个样本失败:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['sample_id']}: {r.get('error', 'unknown error')}")
        print()
        sys.exit(1)
    else:
        print(f"\n🎉 所有样本测试通过！")

        # 打印 scale 统计
        scales = [r["scale_mean"] for r in results if r["status"] == "success"]
        if scales:
            print(f"\nScale 统计:")
            print(f"  平均: {np.mean(scales):.6f}")
            print(f"  标准差: {np.std(scales):.6f}")
            print(f"  范围: [{min(scales):.6f}, {max(scales):.6f}]")

        print()
        sys.exit(0)


if __name__ == "__main__":
    import os
    main()

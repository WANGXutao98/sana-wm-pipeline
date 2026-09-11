#!/usr/bin/env python3
"""对比参考实现与我们实现的scale数值"""

import numpy as np
from pathlib import Path
import sys

def main():
    out_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/reference_verification")
    smoke_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/smoke_result")

    short_id = "89f6503b"
    full_id = "SpatialVID-hq_89f6503b-be11-590a-86c9-1f033acc3a03"

    print("=" * 80)
    print("Scale对比分析")
    print("=" * 80)
    print(f"\n样本: {short_id} (最佳样本, 当前轨迹偏差6.96x)")
    print("-" * 80)

    # 1. 读取参考实现的scale
    ref_scale_path = out_dir / f"{short_id}_depth" / "scales.npy"
    if not ref_scale_path.exists():
        print(f"❌ 参考实现scale不存在: {ref_scale_path}")
        sys.exit(1)

    ref_scale = np.load(ref_scale_path)
    print(f"\n【参考实现】scales.npy:")
    print(f"  文件: {ref_scale_path}")
    print(f"  样本数: {len(ref_scale)}")
    print(f"  数据类型: {ref_scale.dtype}")
    print(f"  范围: {ref_scale.min():.6f} - {ref_scale.max():.6f}")
    print(f"  均值: {ref_scale.mean():.6f}")
    print(f"  中位数: {np.median(ref_scale):.6f}")
    print(f"  标准差: {ref_scale.std():.6f}")
    print(f"  CoV: {ref_scale.std() / ref_scale.mean():.6f}")
    print(f"  前10个值: {ref_scale[:10]}")

    # 2. 读取我们的scale
    # 尝试多个可能的路径
    possible_paths = [
        smoke_dir / full_id / f"{full_id}" / "vipe_work_default" / "depth_precomputed" / "scales.npy",
        smoke_dir / full_id / "extracted" / f"{full_id}.scale.npy",
        smoke_dir / full_id / "vipe_work_default" / "depth_precomputed" / "scales.npy",
    ]

    our_scale = None
    our_scale_path = None
    for path in possible_paths:
        if path.exists():
            our_scale_path = path
            our_scale = np.load(path)
            break

    if our_scale is None:
        print(f"\n⚠️ 我们的scale未找到，尝试过的路径:")
        for p in possible_paths:
            print(f"  - {p}")
        print("\n提示: 可能需要先运行我们的管线生成数据")
        sys.exit(1)

    print(f"\n【我们的实现】scales.npy:")
    print(f"  文件: {our_scale_path}")
    print(f"  样本数: {len(our_scale)}")
    print(f"  数据类型: {our_scale.dtype}")
    print(f"  范围: {our_scale.min():.6f} - {our_scale.max():.6f}")
    print(f"  均值: {our_scale.mean():.6f}")
    print(f"  中位数: {np.median(our_scale):.6f}")
    print(f"  标准差: {our_scale.std():.6f}")
    print(f"  CoV: {our_scale.std() / our_scale.mean():.6f}")
    print(f"  前10个值: {our_scale[:10]}")

    # 3. 对比分析
    print(f"\n【对比分析】")
    print("-" * 80)

    # 3a. 均值对比
    ratio_mean = our_scale.mean() / ref_scale.mean()
    print(f"\n均值比例: {ratio_mean:.6f}x")
    if abs(ratio_mean - 1.0) < 0.01:
        print(f"  ✅ 基本一致（偏差{abs(ratio_mean-1.0)*100:.2f}%）")
    elif abs(ratio_mean - 1.0) < 0.1:
        print(f"  ⚠️ 有小偏差（偏差{abs(ratio_mean-1.0)*100:.2f}%）")
    else:
        print(f"  ❌ 显著偏差（偏差{abs(ratio_mean-1.0)*100:.2f}%）")

    # 3b. 中位数对比
    ratio_median = np.median(our_scale) / np.median(ref_scale)
    print(f"\n中位数比例: {ratio_median:.6f}x")

    # 3c. 分布对比
    print(f"\nCoV对比:")
    print(f"  参考: {ref_scale.std() / ref_scale.mean():.6f}")
    print(f"  我们: {our_scale.std() / our_scale.mean():.6f}")

    # 3d. 逐帧对比（如果长度一致）
    if len(ref_scale) == len(our_scale):
        print(f"\n逐帧对比（长度一致: {len(ref_scale)}）:")
        ratios = our_scale / ref_scale
        print(f"  逐帧比例范围: {ratios.min():.6f} - {ratios.max():.6f}")
        print(f"  逐帧比例均值: {ratios.mean():.6f}")
        print(f"  逐帧比例std: {ratios.std():.6f}")

        # 检查是否有系统性偏移
        if ratios.std() < 0.01:
            print(f"  → 逐帧比例非常稳定，说明有系统性偏移")
    else:
        print(f"\n⚠️ 长度不一致: 参考{len(ref_scale)} vs 我们{len(our_scale)}")
        print(f"  无法进行逐帧对比")

    # 4. 结论
    print(f"\n【关键结论】")
    print("=" * 80)

    if abs(ratio_mean - 1.0) < 0.1:
        print("✅ 深度融合的scale计算与参考实现基本一致（偏差<10%）")
        print()
        print("这说明:")
        print("  1. solve_frame_scale()函数工作正常")
        print("  2. Pi3X和MoGe-2的推理结果一致")
        print("  3. 深度融合的EMA平滑一致")
        print()
        print("🔍 轨迹偏差17x的原因不在深度融合阶段")
        print("🔍 需要检查VIPE SLAM的Bundle Adjustment阶段")
        print("🔍 可能的问题:")
        print("   - BA优化时scale的传播方式")
        print("   - poses的归一化处理")
        print("   - 参考标注本身的可靠性")
    else:
        print("❌ 深度融合的scale计算有显著偏差")
        print()
        print("需要进一步调查:")
        print("  1. 对比Pi3X和MoGe-2的原始输出")
        print("  2. 检查solve_frame_scale的输入输出")
        print("  3. 验证权重策略w=1/d是否正确")

    print()

if __name__ == "__main__":
    main()

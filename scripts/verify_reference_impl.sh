#!/bin/bash
# 验证方案：对比参考实现与我们实现的scale计算

set -e

echo "========================================================================"
echo "验证方案：对比scale计算"
echo "========================================================================"
echo ""
echo "策略调整："
echo "  1. 参考实现只有precompute_fused_depth.py（计算深度+scale）"
echo "  2. 没有完整的VIPE SLAM运行脚本"
echo "  3. 所以我们对比：深度融合阶段的scale.npy"
echo ""
echo "验证样本："
echo "  - 89f6503b (最佳, 6.96x)"
echo "  - c0a56c15 (中等, 9.10x)"
echo "  - 2fa6ae85 (较差, 22.86x)"
echo ""

SAMPLE_IDS=(
    "SpatialVID-hq_89f6503b-be11-590a-86c9-1f033acc3a03"
    "SpatialVID-hq_c0a56c15-579b-5ecf-a1e4-547bc35af51e"
    "SpatialVID-hq_2fa6ae85-c441-5700-9f98-4e08e24bdf9c"
)

RAW_DIR="/mnt/afs/davidwang/workspace/sana_test_data/smoke_result/raw_samples"
OUT_DIR="/mnt/afs/davidwang/workspace/sana_test_data/reference_verification"
REPO_DIR="/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean"

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "步骤1: 用参考实现计算scale"
echo "========================================================================"
echo ""

for sample_id in "${SAMPLE_IDS[@]}"; do
    short_id="${sample_id##*_}"
    short_id="${short_id:0:8}"

    video="$RAW_DIR/${sample_id}.mp4"
    out_depth="$OUT_DIR/${short_id}_depth"

    if [ ! -f "$video" ]; then
        echo "⚠️  视频不存在: $video"
        continue
    fi

    echo "处理样本: $short_id"
    echo "  视频: $video"
    echo "  输出: $out_depth"

    # 运行参考实现的precompute_fused_depth.py
    cd "$REPO_DIR"
    python scripts/precompute_fused_depth.py "$video" "$out_depth" 64

    # 检查输出
    if [ -f "$out_depth/scales.npy" ]; then
        echo "  ✅ scales.npy 生成成功"
    else
        echo "  ❌ scales.npy 生成失败"
    fi

    echo ""
done

echo "========================================================================"
echo "步骤2: 对比scale统计"
echo "========================================================================"
echo ""

python3 << 'PYTHON_EOF'
import numpy as np
from pathlib import Path

out_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/reference_verification")
smoke_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/smoke_result")

samples = [
    ("89f6503b", "SpatialVID-hq_89f6503b-be11-590a-86c9-1f033acc3a03"),
    ("c0a56c15", "SpatialVID-hq_c0a56c15-579b-5ecf-a1e4-547bc35af51e"),
    ("2fa6ae85", "SpatialVID-hq_2fa6ae85-c441-5700-9f98-4e08e24bdf9c"),
]

print("对比结果:")
print("=" * 80)

for short_id, full_id in samples:
    print(f"\n样本: {short_id}")
    print("-" * 80)

    # 参考实现的scale
    ref_scale_path = out_dir / f"{short_id}_depth" / "scales.npy"
    if ref_scale_path.exists():
        ref_scale = np.load(ref_scale_path)
        print(f"参考实现 scale:")
        print(f"  样本数: {len(ref_scale)}")
        print(f"  范围: {ref_scale.min():.4f} - {ref_scale.max():.4f}")
        print(f"  均值: {ref_scale.mean():.4f}")
        print(f"  中位数: {np.median(ref_scale):.4f}")
    else:
        print(f"⚠️  参考实现scale不存在: {ref_scale_path}")
        ref_scale = None

    # 我们的scale
    our_scale_path = smoke_dir / full_id / "extracted" / f"{full_id}.scale.npy"
    if our_scale_path.exists():
        our_scale = np.load(our_scale_path)
        print(f"\n我们的实现 scale:")
        print(f"  样本数: {len(our_scale)}")
        print(f"  范围: {our_scale.min():.4f} - {our_scale.max():.4f}")
        print(f"  均值: {our_scale.mean():.4f}")
        print(f"  中位数: {np.median(our_scale):.4f}")
    else:
        print(f"⚠️  我们的scale不存在: {our_scale_path}")
        our_scale = None

    # 对比
    if ref_scale is not None and our_scale is not None:
        print(f"\n对比:")
        ratio = our_scale.mean() / ref_scale.mean()
        print(f"  均值比例: {ratio:.4f}")
        if abs(ratio - 1.0) < 0.01:
            print(f"  ✅ 基本一致（偏差<1%）")
        elif abs(ratio - 1.0) < 0.1:
            print(f"  ⚠️ 有小偏差（偏差{abs(ratio-1.0)*100:.1f}%）")
        else:
            print(f"  ❌ 显著偏差（偏差{abs(ratio-1.0)*100:.1f}%）")

PYTHON_EOF

echo ""
echo "========================================================================"
echo "验证完成"
echo "========================================================================"

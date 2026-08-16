#!/bin/bash
# 验证方案：对比参考实现与我们实现的scale计算
# 需要在sana_wm环境下运行

set -e

echo "========================================================================"
echo "验证方案：对比scale计算"
echo "========================================================================"
echo ""

# 激活conda环境
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

echo "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo ""

# 设置环境变量
export SANA_WM_PI3X_WEIGHTS="/mnt/afs/davidwang/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="/mnt/afs/davidwang/models/moge2"
export SANA_WM_WEIGHTS="/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-wm-data-clean/weights"

echo "环境变量:"
echo "  SANA_WM_PI3X_WEIGHTS=$SANA_WM_PI3X_WEIGHTS"
echo "  SANA_WM_MOGE2_WEIGHTS=$SANA_WM_MOGE2_WEIGHTS"
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
echo "步骤1: 用参考实现计算scale（仅1个样本快速验证）"
echo "========================================================================"
echo ""

# 只处理第一个样本（最快验证）
sample_id="${SAMPLE_IDS[0]}"
short_id="${sample_id##*_}"
short_id="${short_id:0:8}"

video="$RAW_DIR/${sample_id}.mp4"
out_depth="$OUT_DIR/${short_id}_depth"

echo "处理样本: $short_id (89f6503b - 最佳样本)"
echo "  视频: $video"
echo "  输出: $out_depth"
echo ""

if [ ! -f "$video" ]; then
    echo "❌ 视频不存在: $video"
    exit 1
fi

# 运行参考实现的precompute_fused_depth.py
cd "$REPO_DIR"
echo "运行参考实现..."
python scripts/precompute_fused_depth.py "$video" "$out_depth" 64 2>&1 | tee "$OUT_DIR/${short_id}_log.txt"

# 检查输出
if [ -f "$out_depth/scales.npy" ]; then
    echo ""
    echo "✅ scales.npy 生成成功"
else
    echo ""
    echo "❌ scales.npy 生成失败"
    exit 1
fi

echo ""
echo "========================================================================"
echo "步骤2: 对比scale统计"
echo "========================================================================"
echo ""

python3 << 'PYTHON_EOF'
import numpy as np
from pathlib import Path

out_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/reference_verification")
smoke_dir = Path("/mnt/afs/davidwang/workspace/sana_test_data/smoke_result")

short_id = "89f6503b"
full_id = "SpatialVID-hq_89f6503b-be11-590a-86c9-1f033acc3a03"

print("对比结果:")
print("=" * 80)
print(f"样本: {short_id} (最佳样本, 当前偏差6.96x)")
print("-" * 80)

# 参考实现的scale
ref_scale_path = out_dir / f"{short_id}_depth" / "scales.npy"
if ref_scale_path.exists():
    ref_scale = np.load(ref_scale_path)
    print(f"\n参考实现 scale:")
    print(f"  样本数: {len(ref_scale)}")
    print(f"  范围: {ref_scale.min():.4f} - {ref_scale.max():.4f}")
    print(f"  均值: {ref_scale.mean():.4f}")
    print(f"  中位数: {np.median(ref_scale):.4f}")
    print(f"  标准差: {ref_scale.std():.4f}")
else:
    print(f"❌ 参考实现scale不存在: {ref_scale_path}")
    exit(1)

# 我们的scale (从depth_precomputed读取)
our_scale_path = smoke_dir / full_id / f"{full_id}" / "vipe_work_default" / "depth_precomputed" / "scales.npy"
if not our_scale_path.exists():
    # 尝试另一个路径
    our_scale_path = smoke_dir / full_id / "extracted" / f"{full_id}.scale.npy"

if our_scale_path.exists():
    our_scale = np.load(our_scale_path)
    print(f"\n我们的实现 scale:")
    print(f"  样本数: {len(our_scale)}")
    print(f"  范围: {our_scale.min():.4f} - {our_scale.max():.4f}")
    print(f"  均值: {our_scale.mean():.4f}")
    print(f"  中位数: {np.median(our_scale):.4f}")
    print(f"  标准差: {our_scale.std():.4f}")
else:
    print(f"⚠️ 我们的scale不存在: {our_scale_path}")
    our_scale = None

# 对比
if our_scale is not None:
    print(f"\n对比:")
    print("-" * 80)
    ratio = our_scale.mean() / ref_scale.mean()
    print(f"  均值比例: {ratio:.4f}x")

    if abs(ratio - 1.0) < 0.01:
        print(f"  ✅ 基本一致（偏差<1%）")
        print(f"  → scale计算正确，问题在其他地方")
    elif abs(ratio - 1.0) < 0.1:
        print(f"  ⚠️ 有小偏差（偏差{abs(ratio-1.0)*100:.1f}%）")
        print(f"  → scale计算有轻微差异")
    else:
        print(f"  ❌ 显著偏差（偏差{abs(ratio-1.0)*100:.1f}%）")
        print(f"  → scale计算有明显问题")

    print()
    print("关键结论:")
    if abs(ratio - 1.0) < 0.1:
        print("  ✅ 深度融合的scale计算与参考实现一致")
        print("  🔍 轨迹偏差17x的原因不在深度融合")
        print("  🔍 需要检查VIPE SLAM的BA阶段")
    else:
        print("  ❌ 深度融合的scale计算有问题")
        print("  🔍 需要详细对比solve_frame_scale的输入输出")

print()
print("=" * 80)

PYTHON_EOF

echo ""
echo "========================================================================"
echo "验证完成"
echo "========================================================================"
echo ""
echo "如果需要处理其他2个样本，请手动修改脚本中的sample_id"

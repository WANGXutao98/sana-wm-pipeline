#!/bin/bash
# Stage3 冒烟测试对比实验
# 对比 2s分块 vs 5s分块+降采样

VIDEO="/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00eb7564-d5e8-54a1-b8bd-52ab85334924.mp4"

echo "========================================="
echo "Stage3 分块策略对比实验"
echo "========================================="
echo "测试视频: $(basename $VIDEO)"
echo ""

source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建临时测试目录
TMP_DIR="/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_test"
mkdir -p "$TMP_DIR"
rm -f "$TMP_DIR"/*.mp4
ln -sf "$VIDEO" "$TMP_DIR/test.mp4"

echo "========================================="
echo "测试 1: 2s 分块（当前实现）"
echo "========================================="
echo "配置: 官方采样器 + 原始720p分辨率"
echo ""

python scripts/stage3_batch_minimal.py \
  --input_dir "$TMP_DIR" \
  --output /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl \
  --device cuda 2>&1 | grep -E "Loading|✅|Processing|UniMatch|DOVER|TQE|AQE|fused|Verdict"

echo ""
echo "结果文件: /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl"
cat /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl | python3 -m json.tool 2>/dev/null || cat /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl
echo ""

echo "========================================="
echo "测试 2: 5s 分块 + 480p 降采样（论文配置）"
echo "========================================="
echo "配置: 官方采样器 + 降采样到480p"
echo ""

python scripts/stage3_test_5s.py \
  --video "$VIDEO" \
  --device cuda

echo ""
echo "结果文件: /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl"
cat /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl | python3 -m json.tool 2>/dev/null || cat /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl
echo ""

echo "========================================="
echo "对比分析"
echo "========================================="

python3 << 'PYTHON_EOF'
import json

with open("/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl") as f:
    result_2s = json.loads(f.read().strip())

with open("/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl") as f:
    result_5s = json.loads(f.read().strip())

print("指标对比:")
print(f"{'指标':<20} {'2s分块':>15} {'5s+480p':>15} {'差异':>15}")
print("-" * 70)

metrics = [
    ("UniMatch Flow", "unimatch_flow"),
    ("DOVER TQE", "dover_tqe"),
    ("DOVER AQE", "dover_aqe"),
    ("DOVER Fused", "dover_fused"),
    ("Verdict", "verdict"),
]

for label, key in metrics:
    val_2s = result_2s.get(key)
    val_5s = result_5s.get(key)

    if isinstance(val_2s, (int, float)) and isinstance(val_5s, (int, float)):
        diff = val_5s - val_2s
        print(f"{label:<20} {val_2s:>15.4f} {val_5s:>15.4f} {diff:>+15.4f}")
    else:
        print(f"{label:<20} {str(val_2s):>15} {str(val_5s):>15} {'':>15}")

print("\n影响分析:")
print("-" * 70)

dover_diff = result_5s["dover_fused"] - result_2s["dover_fused"]
unimatch_diff = result_5s["unimatch_flow"] - result_2s["unimatch_flow"]

print(f"DOVER 分数变化:   {dover_diff:+.4f} ({abs(dover_diff)/result_2s['dover_fused']*100:+.1f}%)")
print(f"UniMatch 分数变化: {unimatch_diff:+.3f} ({abs(unimatch_diff)/result_2s['unimatch_flow']*100:+.1f}%)")

print("\n结论:")
if abs(dover_diff) > abs(unimatch_diff):
    print("✅ DOVER 受影响更大")
    if dover_diff > 0:
        print("   5s分块 提升了 DOVER 分数（更好的时序建模）")
    else:
        print("   降采样 降低了 DOVER 分数（信息损失）")
else:
    print("✅ UniMatch 受影响更大")

if abs(dover_diff) > 0.05:
    print(f"\n⚠️ DOVER 差异显著 (>{0.05})，建议:")
    if dover_diff < 0:
        print("   - 降采样损失过大，考虑提高到540p或600p")
    else:
        print("   - 5s分块效果更好，建议采用")
else:
    print(f"\n✅ DOVER 差异可接受 (<{0.05})，两种方案均可")

PYTHON_EOF

echo ""
echo "========================================="
echo "完成"
echo "========================================="

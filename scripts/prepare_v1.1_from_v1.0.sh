#!/bin/bash
# v1.0 清理 + v1.1 构建脚本

set -e

echo "========================================================================"
echo "v1.0 清理 + v1.1 增量构建"
echo "========================================================================"
echo "开始时间: $(date)"
echo ""

# ==================== Step 1: 清理 v1.0 ====================
echo "Step 1: 清理 v1.0 多余文件..."
cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0

# 删除 3 个孤立的 caption.txt
rm -f RealEstate10K-360p_train__a68112570746fce2.caption.txt
rm -f RealEstate10K-360p_train__63ec3dfd04ee6970.caption.txt
rm -f RealEstate10K-360p_train__f1f2ab311bf11cce.caption.txt

echo "✅ 已删除 3 个孤立的 caption.txt"

# 验证清理结果
echo ""
echo "=== v1.0 清理后统计 ==="
for ext in .mp4 .poses_c2w.npy .intrinsics.npy .scale.npy .caption.txt; do
    count=$(ls *$ext 2>/dev/null | wc -l)
    echo "  $ext: $count"
done

total=$(find . -type f ! -name "*.log" ! -name "*report*" ! -name "*samples.jsonl" ! -name "need_reextract*" | wc -l)
echo "  总文件数: $total"

# ==================== Step 2: 生成 v1.1 增量列表 ====================
echo ""
echo "Step 2: 生成 v1.1 增量列表..."
cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/scripts

python3 << 'PYTHON_EOF'
import json
from pathlib import Path

# 读取 v1.0 实际成功的样本（基于 .mp4 文件）
v1_0_dir = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0")
v1_0_ids = set()

for file in v1_0_dir.glob("*.mp4"):
    sample_id = file.stem  # 去掉 .mp4 后缀
    v1_0_ids.add(sample_id)

print(f"v1.0 实际样本数: {len(v1_0_ids)}")

# 读取 v1.1 筛选列表
v1_1_all = []
with open('/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/sana-human-feedback/filtered_training_samples_v1.1_with_acceptable.jsonl', 'r') as f:
    for line in f:
        v1_1_all.append(json.loads(line))

print(f"v1.1 筛选列表总数: {len(v1_1_all)}")

# 找出需要新增的样本
new_samples = []
for sample in v1_1_all:
    if sample['sample_id'] not in v1_0_ids:
        new_samples.append(sample)

print(f"v1.1 需要新增: {len(new_samples)} 个样本")
print(f"v1.1 预期总数: {len(v1_0_ids) + len(new_samples)}")

# 保存增量列表
output = '/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/sana-human-feedback/filtered_training_samples_v1.1_incremental.jsonl'
with open(output, 'w') as f:
    for sample in new_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\n✅ 增量列表已保存: {output}")
print(f"预计提取时间: {len(new_samples) * 26.4 / 3600:.1f} 小时")
PYTHON_EOF

# ==================== Step 3: 复制 v1.0 到 v1.1 ====================
echo ""
echo "Step 3: 复制 v1.0 到 v1.1..."

# 创建 v1.1 目录
mkdir -p /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.1

# 使用 rsync 复制（排除日志和报告文件）
rsync -a \
  /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0/ \
  /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.1/ \
  --exclude="*.log" \
  --exclude="*report*.txt" \
  --exclude="*samples.jsonl" \
  --exclude="need_reextract*" \
  --exclude="corrupted_tars.txt"

echo "✅ 复制完成"

# 验证复制结果
v1_1_count=$(find /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.1 -name "*.mp4" | wc -l)
echo "v1.1 当前样本数: $v1_1_count"

# ==================== 完成 ====================
echo ""
echo "========================================================================"
echo "✅ 准备完成！"
echo "========================================================================"
echo ""
echo "v1.0 最终状态:"
echo "  - 完整样本: 1964"
echo "  - 成功率: 99.2% (1964/1980)"
echo "  - 总文件数: 9820 (1964 × 5)"
echo ""
echo "v1.1 增量提取准备就绪"
echo "  - 基础样本: 1964 (来自 v1.0)"
echo "  - 需要新增: ~687 个样本"
echo "  - 预期总数: ~2651 个样本"
echo ""
echo "下一步: 在 tmux 中启动 v1.1 增量提取"
echo ""
echo "  tmux new-session -s v1.1_incremental"
echo "  cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/scripts"
echo "  python3 extract_training_data_from_filtered_corrected.py \\"
echo "    --filtered_list /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/sana-human-feedback/filtered_training_samples_v1.1_incremental.jsonl \\"
echo "    --data_root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \\"
echo "    --output_dir /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.1 \\"
echo "    2>&1 | tee v1.1_incremental.log"
echo ""
echo "========================================================================"

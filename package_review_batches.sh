#!/bin/bash
# 打包所有审查样本，按批次分组

set -e

cd /root/work/david_work/sana_wm_qc

OUTPUT_DIR="review_packages"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  打包审查样本（分批）"
echo "========================================"
echo ""

REVIEW_SAMPLES="human_review_samples/review_samples.jsonl"
TOTAL_SAMPLES=$(wc -l < "$REVIEW_SAMPLES")

echo "总样本数: $TOTAL_SAMPLES"
echo ""

# 计算每批样本数（建议每批 500-1000 样本）
BATCH_SIZE=800
NUM_BATCHES=$(( ($TOTAL_SAMPLES + $BATCH_SIZE - 1) / $BATCH_SIZE ))

echo "分批策略: 每批 $BATCH_SIZE 样本，共 $NUM_BATCHES 批"
echo ""

# 分批
for ((batch=1; batch<=$NUM_BATCHES; batch++)); do
  start=$(( ($batch - 1) * $BATCH_SIZE + 1 ))
  end=$(( $batch * $BATCH_SIZE ))

  if [ $end -gt $TOTAL_SAMPLES ]; then
    end=$TOTAL_SAMPLES
  fi

  actual_count=$(( $end - $start + 1 ))

  echo "处理批次 $batch/$NUM_BATCHES (样本 $start-$end, 共 $actual_count 个)..."

  # 创建批次目录
  batch_dir="$OUTPUT_DIR/batch_$(printf '%02d' $batch)"
  mkdir -p "$batch_dir"

  # 提取该批次样本
  sed -n "${start},${end}p" "$REVIEW_SAMPLES" > "$batch_dir/samples.jsonl"

  # 生成批次统计
  python3 << EOF
import json
from collections import Counter

samples = []
with open('$batch_dir/samples.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

# 统计
by_group = Counter(s['review_group'] for s in samples)
by_verdict = Counter(s['verdict'] for s in samples)

# 保存统计
with open('$batch_dir/batch_stats.txt', 'w') as f:
    f.write(f"批次 $batch 统计信息\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"样本数: {len(samples)}\n")
    f.write(f"样本范围: $start - $end\n\n")

    f.write("按 Group 分布:\n")
    for group, count in sorted(by_group.items()):
        f.write(f"  {group:<30} {count:>5}\n")

    f.write("\n按 Verdict 分布:\n")
    for verdict, count in sorted(by_verdict.items()):
        f.write(f"  {verdict:<30} {count:>5}\n")

print(f"  ✅ 批次 $batch: {len(samples)} 个样本")
EOF

  # 生成批次说明
  cat > "$batch_dir/README.txt" << EOFREADME
========================================
审查批次 $batch/$NUM_BATCHES
========================================

样本范围: $start - $end
样本数: $actual_count

文件说明:
---------
- samples.jsonl: 该批次的所有样本数据
- batch_stats.txt: 批次统计信息
- annotation_results.jsonl: 标注结果（请填写后返回）

标注格式:
---------
对每个样本，在 annotation_results.jsonl 中添加一行：

{
  "sample_id": "<样本ID>",
  "quality_rating": "excellent|good|acceptable|poor",
  "use_for_training": true|false,
  "issues": ["问题1", "问题2", ...],
  "notes": "备注",
  "annotator": "<标注者姓名>"
}

标注指南:
---------
1. 查看样本的视频、pose、caption
2. 参考 flag_reasons 和 metrics
3. 根据质量标准做出判断
4. 记录主要问题和备注

请参考黄金样本中的质量标准！
EOFREADME

  # 创建空的标注结果文件
  touch "$batch_dir/annotation_results.jsonl"

  # 打包该批次
  batch_package="$OUTPUT_DIR/batch_$(printf '%02d' $batch).tar.gz"
  tar -czf "$batch_package" -C "$OUTPUT_DIR" "batch_$(printf '%02d' $batch)"

  echo "  ✅ 已打包: $batch_package ($(du -h "$batch_package" | cut -f1))"

  # 清理临时目录（保留打包文件）
  rm -rf "$batch_dir"

done

echo ""
echo "========================================"
echo "  打包完成"
echo "========================================"
echo ""
echo "生成的文件："
ls -lh "$OUTPUT_DIR"/*.tar.gz | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# 生成总体说明
cat > "$OUTPUT_DIR/REVIEW_DISTRIBUTION_GUIDE.txt" << 'EOF'
========================================
人工审查分发指南
========================================

总体情况:
---------
- 总样本数: 6,702
- 分批数: 9 批
- 每批约: 700-800 样本

分发建议:
---------
1. 每个测试人员分配 1-2 批（约 800-1600 样本）
2. 预计每人每天审查 200-300 样本（取决于熟练度）
3. 完成时间: 3-5 天/人

测试人员要求:
-------------
1. 已完成黄金样本培训
2. 通过一致性测试（与专家判断一致性 > 80%）
3. 理解质量标准

工作流程:
---------
1. 下载分配的批次包
2. 解压后查看 README.txt
3. 按顺序审查 samples.jsonl 中的样本
4. 将标注结果写入 annotation_results.jsonl
5. 完成后返回 annotation_results.jsonl

质量控制:
---------
1. 随机抽查 10% 样本进行复核
2. 多人标注重叠样本计算一致性
3. 一致性低于 75% 的标注者需要重新培训

时间估算:
---------
- 培训: 1-2 小时
- 一致性测试: 20 个样本，约 30 分钟
- 正式审查: 800 样本 × 15-20秒/样本 = 3-4 小时
- 每批建议分 2-3 天完成，避免疲劳

注意事项:
---------
1. 保持一致的判断标准
2. 遇到困难案例做标记
3. 定期休息，避免视觉疲劳
4. 有疑问及时沟通
EOF

echo "✅ 分发指南已创建: $OUTPUT_DIR/REVIEW_DISTRIBUTION_GUIDE.txt"
echo ""

echo "下载所有批次："
echo "  scp -r user@cmcc:/root/work/david_work/sana_wm_qc/$OUTPUT_DIR /local/path/"
echo ""

echo "或单独下载："
echo "  scp user@cmcc:/root/work/david_work/sana_wm_qc/$OUTPUT_DIR/batch_01.tar.gz /local/path/"

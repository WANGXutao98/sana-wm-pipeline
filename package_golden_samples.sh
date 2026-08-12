#!/bin/bash
# 打包黄金样本供手动下载

set -e

cd /root/work/david_work/sana_wm_qc

OUTPUT_DIR="review_packages"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  打包黄金样本"
echo "========================================"
echo ""

# 1. 打包黄金样本
GOLDEN_PACKAGE="$OUTPUT_DIR/golden_samples_package.tar.gz"

echo "正在打包黄金样本..."
tar -czf "$GOLDEN_PACKAGE" \
  human_review_samples/golden_samples/

echo "✅ 黄金样本已打包: $GOLDEN_PACKAGE"
ls -lh "$GOLDEN_PACKAGE"
echo ""

# 2. 创建说明文件
cat > "$OUTPUT_DIR/golden_samples_README.txt" << 'EOF'
================================
黄金样本包说明
================================

这个包包含 20 个精心挑选的代表性样本，用于：
1. 标注者培训
2. 质量标准制定
3. 一致性测试

文件内容：
---------
- golden_samples.jsonl: 20个样本的完整数据（JSON Lines格式）
- golden_samples_summary.txt: 人类可读的样本摘要
- annotation_template.txt: 标注模板

样本分布：
---------
- Pass 样本（高质量基准）: 5个
- Fail 样本（明确低质量）: 3个
- Flag 样本（边界案例）: 7个
- 难度分级样本: 5个

使用方法：
---------
1. 先由专家标注这20个样本，建立质量标准
2. 使用这些样本培训标注者
3. 让多个标注者标注相同样本，测试一致性
4. 达到一致性要求后，开始大规模审查

下一步：
---------
查看 golden_samples_summary.txt 了解样本详情
使用 annotation_template.txt 进行标注
EOF

echo "✅ 说明文件已创建: $OUTPUT_DIR/golden_samples_README.txt"
echo ""

echo "========================================"
echo "  打包完成"
echo "========================================"
echo ""
echo "下载文件："
echo "  scp user@cmcc:/root/work/david_work/sana_wm_qc/$GOLDEN_PACKAGE /local/path/"
echo ""
echo "或使用以下命令查看："
echo "  cd /root/work/david_work/sana_wm_qc/$OUTPUT_DIR"
echo "  ls -lh"

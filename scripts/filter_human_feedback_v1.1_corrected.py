#!/usr/bin/env python3
"""
SANA-WM 人工反馈数据筛选脚本 v1.1（修正版）

功能：从 9 个 JSONL 文件中筛选符合条件的样本
筛选条件：
  1. 数据集归属：RealEstate10K / SpatialVID-hq / sekai-real-walking-hq
  2. 质量评分：good / excellent / acceptable（v1.1 新增）

输出：合并后的 JSONL 文件 + 统计报表
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# ==================== 配置参数 ====================
INPUT_DIR = Path("/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback")
OUTPUT_FILE = INPUT_DIR / "filtered_training_samples_v1.1_with_acceptable.jsonl"
STATS_FILE = INPUT_DIR / "filter_statistics_v1.1_corrected.txt"

# 筛选条件
TARGET_DATASETS = {
    "RealEstate10K",
    "SpatialVID-hq",
    "sekai-real-walking-hq"
}

# v1.1 新增 acceptable（可接受）评级
TARGET_RATINGS = {"good", "excellent", "acceptable"}

# ==================== 数据筛选逻辑 ====================

def extract_dataset_name(sample_id):
    """
    从 sample_id 提取数据集名称

    示例：
      RealEstate10K-360p_train__11034893f72fe474 -> RealEstate10K
      SpatialVID-hq_val__abc123 -> SpatialVID-hq
      sekai-real-walking-hq_train__xyz789 -> sekai-real-walking-hq
    """
    # 特殊处理：SpatialVID-hq, sekai-real-walking-hq 包含破折号
    for target in TARGET_DATASETS:
        if sample_id.startswith(target):
            return target

    # 处理 RealEstate10K-360p_train 这种情况
    if '_' in sample_id:
        parts = sample_id.split('_')
        if '-' in parts[0]:
            dataset = parts[0].split('-')[0]
        else:
            dataset = parts[0]
    else:
        dataset = sample_id.split('-')[0]

    return dataset


def filter_samples():
    """主筛选函数"""

    # 统计数据
    stats = {
        'total_samples': 0,
        'filtered_samples': 0,
        'by_dataset': defaultdict(lambda: {'good': 0, 'excellent': 0, 'acceptable': 0, 'total': 0}),
        'by_rating': defaultdict(int),
        'rejected_dataset': 0,
        'rejected_rating': 0
    }

    filtered_samples = []

    # 遍历 9 个 JSONL 文件
    jsonl_files = sorted(INPUT_DIR.glob("annotation_results_*.jsonl"))

    print(f"📁 数据源目录: {INPUT_DIR}")
    print(f"📄 发现 {len(jsonl_files)} 个 JSONL 文件")
    print(f"🆕 版本 v1.1 - 新增 acceptable（可接受）评级\n")

    for jsonl_file in jsonl_files:
        print(f"⏳ 处理文件: {jsonl_file.name}")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                    stats['total_samples'] += 1

                    # 提取字段
                    sample_id = sample.get('sample_id', '')
                    quality_rating = sample.get('quality_rating', '')

                    # 提取数据集名称
                    dataset_name = extract_dataset_name(sample_id)

                    # 条件 1：数据集归属检查
                    if dataset_name not in TARGET_DATASETS:
                        stats['rejected_dataset'] += 1
                        continue

                    # 条件 2：质量评分检查
                    if quality_rating not in TARGET_RATINGS:
                        stats['rejected_rating'] += 1
                        continue

                    # 通过筛选，保留样本
                    filtered_samples.append(sample)
                    stats['filtered_samples'] += 1
                    stats['by_dataset'][dataset_name][quality_rating] += 1
                    stats['by_dataset'][dataset_name]['total'] += 1
                    stats['by_rating'][quality_rating] += 1

                except json.JSONDecodeError as e:
                    print(f"  ⚠️  JSON 解析错误 (行 {line_num}): {e}")
                    continue

        print(f"  ✅ 完成\n")

    return filtered_samples, stats


def write_filtered_samples(samples):
    """写入筛选后的样本"""
    print(f"💾 写入筛选结果到: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"  ✅ 成功写入 {len(samples)} 条样本\n")


def generate_statistics_report(stats):
    """生成统计报表"""

    report = []
    report.append("=" * 70)
    report.append("SANA-WM 人工反馈数据筛选统计报表 v1.1（修正版）")
    report.append("=" * 70)
    report.append("")
    report.append("【版本说明】")
    report.append("  v1.1 - 新增 acceptable（可接受）评级，放宽筛选条件")
    report.append("  注：原始数据使用 'acceptable' 而非 'average'")
    report.append("")

    # 总体统计
    report.append("【一、总体统计】")
    report.append(f"  源文件总样本数：{stats['total_samples']}")
    report.append(f"  筛选通过样本数：{stats['filtered_samples']}")
    report.append(f"  通过率：{stats['filtered_samples'] / stats['total_samples'] * 100:.2f}%")
    report.append("")

    # 拒绝原因统计
    report.append("【二、拒绝原因统计】")
    report.append(f"  数据集不匹配：{stats['rejected_dataset']}")
    report.append(f"  评分不达标：{stats['rejected_rating']}")
    report.append("")

    # 按数据集统计
    report.append("【三、按数据集分类统计】")
    for dataset in sorted(stats['by_dataset'].keys()):
        data = stats['by_dataset'][dataset]
        report.append(f"  {dataset}:")
        report.append(f"    - Excellent 样本：{data['excellent']}")
        report.append(f"    - Good 样本：{data['good']}")
        report.append(f"    - Acceptable 样本：{data['acceptable']}")
        report.append(f"    - 小计：{data['total']}")
    report.append("")

    # 按评分统计
    report.append("【四、按评分分类统计】")
    for rating in ['excellent', 'good', 'acceptable']:
        count = stats['by_rating'].get(rating, 0)
        percentage = count / stats['filtered_samples'] * 100 if stats['filtered_samples'] > 0 else 0
        report.append(f"  {rating.capitalize()}：{count} ({percentage:.1f}%)")
    report.append("")

    # 筛选条件
    report.append("【五、筛选条件】")
    report.append(f"  目标数据集：{', '.join(sorted(TARGET_DATASETS))}")
    report.append(f"  目标评分：{', '.join(sorted(TARGET_RATINGS))}")
    report.append("")

    # 输出文件
    report.append("【六、输出产物】")
    report.append(f"  筛选结果文件：{OUTPUT_FILE}")
    report.append(f"  统计报表文件：{STATS_FILE}")
    report.append("")

    # 版本对比
    report.append("【七、版本对比】")
    report.append(f"  v1.0 (good + excellent)：1,980 样本")
    report.append(f"  v1.1 (good + excellent + acceptable)：{stats['filtered_samples']} 样本")
    increment = stats['filtered_samples'] - 1980
    report.append(f"  增量：+{increment} 样本 ({increment / 1980 * 100:.1f}% 增长)")
    report.append("")

    report.append("=" * 70)

    report_text = "\n".join(report)

    # 打印到控制台
    print(report_text)

    # 保存到文件
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n📊 统计报表已保存到: {STATS_FILE}")


def main():
    print("\n" + "=" * 70)
    print("SANA-WM 人工反馈数据筛选任务 v1.1（修正版）")
    print("=" * 70 + "\n")

    # 1. 筛选样本
    filtered_samples, stats = filter_samples()

    # 2. 写入结果
    write_filtered_samples(filtered_samples)

    # 3. 生成统计报表
    generate_statistics_report(stats)

    print("\n✅ 筛选任务完成！\n")


if __name__ == "__main__":
    main()

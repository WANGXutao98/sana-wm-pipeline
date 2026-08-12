#!/usr/bin/env python3
"""
从 CMCC 原始数据集中提取筛选后的训练样本

输入：filtered_training_samples.jsonl 或 filtered_training_samples_v1.1_with_acceptable.jsonl
输出：打包后的训练数据集

功能：
  1. 读取筛选后的样本 ID 列表
  2. 在 CMCC 原始数据集中查找对应的 5 个文件
  3. 复制到输出目录
  4. 生成缺失样本报告
  5. 验证数据完整性
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# ==================== 配置参数 ====================

# 数据集路径映射
DATASET_MAPPING = {
    "RealEstate10K": "wds-RealEstate10K-360p",
    "SpatialVID-hq": "wds-SpatialVID-hq",
    "sekai-real-walking-hq": "wds-sekai-real-walking-hq"
}

# 每个样本包含的 5 个必需文件
REQUIRED_FILES = [
    "{sample_id}.video.mp4",
    "{sample_id}.poses_c2w.npy",
    "{sample_id}.intrinsics.npy",
    "{sample_id}.scale.npy",
    "{sample_id}.caption.txt"
]

# ==================== 工具函数 ====================

def extract_dataset_name(sample_id):
    """从 sample_id 提取数据集名称"""
    for dataset in DATASET_MAPPING.keys():
        if sample_id.startswith(dataset):
            return dataset
    return None


def find_sample_files(sample_id, dataset_name, data_root):
    """
    在 CMCC 原始数据集中查找样本文件

    返回：
      - 如果找到：返回 5 个文件路径的列表
      - 如果未找到：返回 None
    """
    dataset_dir = Path(data_root) / DATASET_MAPPING[dataset_name]

    if not dataset_dir.exists():
        print(f"  ⚠️  数据集目录不存在: {dataset_dir}")
        return None

    # 搜索所有 worker 目录
    for worker_dir in sorted(dataset_dir.glob("w*")):
        # 检查 5 个必需文件
        required_files = [
            worker_dir / f"{sample_id}.video.mp4",
            worker_dir / f"{sample_id}.poses_c2w.npy",
            worker_dir / f"{sample_id}.intrinsics.npy",
            worker_dir / f"{sample_id}.scale.npy",
            worker_dir / f"{sample_id}.caption.txt"
        ]

        # 检查文件是否全部存在
        if all(f.exists() for f in required_files):
            return required_files

    return None


def extract_training_data(filtered_list, data_root, output_dir, dry_run=False):
    """主提取函数"""

    # 创建输出目录
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # 读取筛选后的样本列表
    print(f"📋 读取筛选列表: {filtered_list}")
    with open(filtered_list, 'r', encoding='utf-8') as f:
        filtered_samples = [json.loads(line) for line in f]

    print(f"  ✅ 加载 {len(filtered_samples)} 个样本\n")

    # 统计数据
    stats = {
        'total': len(filtered_samples),
        'success': 0,
        'missing': 0,
        'by_dataset': defaultdict(lambda: {'success': 0, 'missing': 0}),
        'by_rating': defaultdict(lambda: {'success': 0, 'missing': 0})
    }

    missing_samples = []

    # 逐条提取
    print(f"🔍 开始提取样本...")
    print(f"  数据源: {data_root}")
    print(f"  输出到: {output_dir}")
    print(f"  模式: {'DRY RUN（仅检查）' if dry_run else '正式提取'}\n")

    for i, sample in enumerate(filtered_samples, 1):
        sample_id = sample['sample_id']
        quality_rating = sample.get('quality_rating', 'unknown')
        dataset_name = extract_dataset_name(sample_id)

        # 进度显示
        if i % 100 == 0 or i == len(filtered_samples):
            print(f"  ⏳ 进度：{i}/{len(filtered_samples)} ({i/len(filtered_samples)*100:.1f}%)")

        if not dataset_name:
            print(f"  ⚠️  无法识别数据集：{sample_id}")
            stats['missing'] += 1
            missing_samples.append({
                'sample_id': sample_id,
                'reason': 'unknown_dataset',
                'quality_rating': quality_rating
            })
            continue

        # 查找源文件
        source_files = find_sample_files(sample_id, dataset_name, data_root)

        if source_files:
            # 复制文件到输出目录
            if not dry_run:
                for src_file in source_files:
                    dst_file = output_path / src_file.name
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        print(f"  ❌ 复制失败：{src_file.name} - {e}")
                        continue

            stats['success'] += 1
            stats['by_dataset'][dataset_name]['success'] += 1
            stats['by_rating'][quality_rating]['success'] += 1
        else:
            stats['missing'] += 1
            stats['by_dataset'][dataset_name]['missing'] += 1
            stats['by_rating'][quality_rating]['missing'] += 1
            missing_samples.append({
                'sample_id': sample_id,
                'reason': 'files_not_found',
                'dataset': dataset_name,
                'quality_rating': quality_rating
            })

    return stats, missing_samples


def generate_report(stats, missing_samples, output_dir):
    """生成提取报告"""

    report = []
    report.append("=" * 70)
    report.append("CMCC 训练数据提取报告")
    report.append("=" * 70)
    report.append("")

    # 总体统计
    report.append("【一、总体统计】")
    report.append(f"  总样本数：{stats['total']}")
    report.append(f"  成功提取：{stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    report.append(f"  缺失样本：{stats['missing']} ({stats['missing']/stats['total']*100:.1f}%)")
    report.append("")

    # 按数据集统计
    report.append("【二、按数据集分类】")
    for dataset in sorted(stats['by_dataset'].keys()):
        data = stats['by_dataset'][dataset]
        total = data['success'] + data['missing']
        report.append(f"  {dataset}:")
        report.append(f"    - 成功：{data['success']}/{total} ({data['success']/total*100:.1f}%)")
        report.append(f"    - 缺失：{data['missing']}/{total}")
    report.append("")

    # 按评级统计
    report.append("【三、按评级分类】")
    for rating in sorted(stats['by_rating'].keys()):
        data = stats['by_rating'][rating]
        total = data['success'] + data['missing']
        report.append(f"  {rating.capitalize()}:")
        report.append(f"    - 成功：{data['success']}/{total} ({data['success']/total*100:.1f}%)")
        report.append(f"    - 缺失：{data['missing']}/{total}")
    report.append("")

    # 输出文件统计
    if stats['success'] > 0:
        report.append("【四、输出产物】")
        report.append(f"  输出目录：{output_dir}")
        report.append(f"  文件总数：{stats['success'] * 5} (每样本 5 个文件)")
        report.append(f"  预估大小：~{stats['success'] * 50} MB")
        report.append("")

    # 缺失样本列表
    if missing_samples:
        report.append("【五、缺失样本清单】")
        report.append(f"  共 {len(missing_samples)} 个样本")
        report.append("")

        # 按数据集分组
        by_dataset = defaultdict(list)
        for sample in missing_samples:
            dataset = sample.get('dataset', 'unknown')
            by_dataset[dataset].append(sample['sample_id'])

        for dataset, sample_ids in sorted(by_dataset.items()):
            report.append(f"  {dataset} ({len(sample_ids)} 个):")
            for sample_id in sample_ids[:10]:  # 最多显示 10 个
                report.append(f"    - {sample_id}")
            if len(sample_ids) > 10:
                report.append(f"    ... 还有 {len(sample_ids) - 10} 个")

    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)

    # 打印到控制台
    print("\n" + report_text)

    # 保存到文件
    report_file = Path(output_dir) / "extraction_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n📊 提取报告已保存到: {report_file}")

    # 保存缺失样本列表
    if missing_samples:
        missing_file = Path(output_dir) / "missing_samples.jsonl"
        with open(missing_file, 'w', encoding='utf-8') as f:
            for sample in missing_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"📋 缺失样本清单已保存到: {missing_file}")


def main():
    parser = argparse.ArgumentParser(
        description="从 CMCC 原始数据集中提取筛选后的训练样本"
    )

    parser.add_argument(
        '--filtered_list',
        type=str,
        required=True,
        help='筛选后的样本列表（JSONL 格式）'
    )

    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='CMCC 原始数据集根目录（包含 wds-* 子目录）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='仅检查不复制（测试模式）'
    )

    args = parser.parse_args()

    # 验证输入文件
    if not Path(args.filtered_list).exists():
        print(f"❌ 错误：筛选列表文件不存在：{args.filtered_list}")
        sys.exit(1)

    if not Path(args.data_root).exists():
        print(f"❌ 错误：数据根目录不存在：{args.data_root}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("CMCC 训练数据提取任务")
    print("=" * 70 + "\n")

    # 执行提取
    stats, missing_samples = extract_training_data(
        args.filtered_list,
        args.data_root,
        args.output_dir,
        args.dry_run
    )

    # 生成报告
    generate_report(stats, missing_samples, args.output_dir)

    # 总结
    print("\n" + "=" * 70)
    if stats['missing'] == 0:
        print("✅ 提取完成！所有样本均成功提取")
    elif stats['missing'] < stats['total'] * 0.05:
        print(f"⚠️  提取完成，但有 {stats['missing']} 个样本缺失（< 5%）")
    else:
        print(f"❌ 提取完成，但有 {stats['missing']} 个样本缺失（> 5%），请检查数据源")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

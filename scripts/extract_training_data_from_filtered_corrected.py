#!/usr/bin/env python3
"""
从 CMCC 原始 WDS 数据集中提取筛选后的训练样本（修正版）

适配 CMCC 实际目录结构：
  /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/
    ├── final_wds-RealEstate10K-360p/
    │   └── wds-RealEstate10K-360p/
    │       ├── w000/
    │       │   ├── shard-0000.tar
    │       │   └── ...
    │       └── w001/
    └── final_wds-DL3DV-ALL-2K/
        └── wds-DL3DV-ALL-2K/
            └── w000/

功能：
  1. 读取筛选后的样本 ID 列表
  2. 在 WDS tar 分片中查找对应的 5 个文件
  3. 提取到输出目录
  4. 生成详细报告
"""

import json
import shutil
import argparse
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict
import sys
import logging

# ==================== 配置参数 ====================

# 数据集路径映射（适配 CMCC 实际目录结构）
DATASET_MAPPING = {
    "RealEstate10K": "final_wds-RealEstate10K-360p/wds-RealEstate10K-360p",
    "SpatialVID-hq": "final_wds-SpatialVID-hq/wds-SpatialVID-hq",
    "sekai-real-walking-hq": "final_wds-sekai-real-walking-hq/wds-sekai-real-walking-hq",
    "DL3DV-ALL-2K": "final_wds-DL3DV-ALL-2K/wds-DL3DV-ALL-2K"
}

# 每个样本包含的 5 个必需文件后缀
REQUIRED_EXTENSIONS = [
    ".video.mp4",
    ".poses_c2w.npy",
    ".intrinsics.npy",
    ".scale.npy",
    ".caption.txt"
]

# ==================== 日志配置 ====================

def setup_logging(log_file):
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ==================== 工具函数 ====================

def extract_dataset_name(sample_id):
    """从 sample_id 提取数据集名称"""
    for dataset in DATASET_MAPPING.keys():
        if sample_id.startswith(dataset):
            return dataset
    return None


def find_sample_in_tar_shards(sample_id, dataset_dir):
    """
    在 tar 分片中查找样本文件

    返回：
      - 如果找到：返回 (tar_path, member_names) 元组
      - 如果未找到：返回 None
    """
    if not dataset_dir.exists():
        logging.warning(f"数据集目录不存在: {dataset_dir}")
        return None

    # 遍历所有 worker 目录 (w000, w001, ...)
    for worker_dir in sorted(dataset_dir.glob("w*")):
        if not worker_dir.is_dir():
            continue

        # 遍历该 worker 目录下的所有 tar 文件
        for tar_path in sorted(worker_dir.glob("*.tar")):
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    # 获取 tar 中所有文件名
                    members = tar.getnames()

                    # 检查是否包含该样本的所有 5 个文件
                    required_files = [
                        f"{sample_id}{ext}" for ext in REQUIRED_EXTENSIONS
                    ]

                    if all(f in members for f in required_files):
                        return (tar_path, required_files)

            except Exception as e:
                logging.warning(f"读取 tar 文件失败 {tar_path}: {e}")
                continue

    return None


def extract_files_from_tar(tar_path, member_names, output_dir):
    """从 tar 文件中提取指定文件"""
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member_name in member_names:
                member = tar.getmember(member_name)
                # 提取到输出目录
                tar.extract(member, output_dir)
        return True
    except Exception as e:
        logging.error(f"从 {tar_path} 提取文件失败: {e}")
        return False


def extract_training_data(filtered_list, data_root, output_dir, dry_run=False):
    """主提取函数"""

    # 创建输出目录
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # 读取筛选后的样本列表
    logging.info(f"📋 读取筛选列表: {filtered_list}")
    with open(filtered_list, 'r', encoding='utf-8') as f:
        filtered_samples = [json.loads(line) for line in f]

    logging.info(f"  ✅ 加载 {len(filtered_samples)} 个样本\n")

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
    logging.info(f"🔍 开始提取样本...")
    logging.info(f"  数据源: {data_root}")
    logging.info(f"  输出到: {output_dir}")
    logging.info(f"  模式: {'DRY RUN（仅检查）' if dry_run else '正式提取'}\n")

    for i, sample in enumerate(filtered_samples, 1):
        sample_id = sample['sample_id']
        quality_rating = sample.get('quality_rating', 'unknown')
        dataset_name = extract_dataset_name(sample_id)

        # 进度显示
        if i % 100 == 0 or i == len(filtered_samples):
            logging.info(f"  ⏳ 进度：{i}/{len(filtered_samples)} ({i/len(filtered_samples)*100:.1f}%)")

        if not dataset_name:
            logging.warning(f"  ⚠️  无法识别数据集：{sample_id}")
            stats['missing'] += 1
            missing_samples.append({
                'sample_id': sample_id,
                'reason': 'unknown_dataset',
                'quality_rating': quality_rating
            })
            continue

        # 构建数据集目录路径
        dataset_rel_path = DATASET_MAPPING[dataset_name]
        dataset_dir = Path(data_root) / dataset_rel_path

        # 在 tar 分片中查找样本
        result = find_sample_in_tar_shards(sample_id, dataset_dir)

        if result:
            tar_path, member_names = result

            # 提取文件
            if not dry_run:
                success = extract_files_from_tar(tar_path, member_names, output_path)
                if not success:
                    stats['missing'] += 1
                    missing_samples.append({
                        'sample_id': sample_id,
                        'reason': 'extraction_failed',
                        'dataset': dataset_name,
                        'quality_rating': quality_rating
                    })
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
    logging.info("\n" + report_text)

    # 保存到文件
    report_file = Path(output_dir) / "extraction_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logging.info(f"\n📊 提取报告已保存到: {report_file}")

    # 保存缺失样本列表
    if missing_samples:
        missing_file = Path(output_dir) / "missing_samples.jsonl"
        with open(missing_file, 'w', encoding='utf-8') as f:
            for sample in missing_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logging.info(f"📋 缺失样本清单已保存到: {missing_file}")


def main():
    parser = argparse.ArgumentParser(
        description="从 CMCC WDS tar 分片中提取筛选后的训练样本"
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
        help='CMCC 原始数据集根目录（包含 final_wds-* 子目录）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )

    parser.add_argument(
        '--log_file',
        type=str,
        default='extraction.log',
        help='日志文件路径'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='仅检查不提取（测试模式）'
    )

    args = parser.parse_args()

    # 设置日志
    log_path = Path(args.output_dir) / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    # 验证输入文件
    if not Path(args.filtered_list).exists():
        logging.error(f"❌ 错误：筛选列表文件不存在：{args.filtered_list}")
        sys.exit(1)

    if not Path(args.data_root).exists():
        logging.error(f"❌ 错误：数据根目录不存在：{args.data_root}")
        sys.exit(1)

    logging.info("\n" + "=" * 70)
    logging.info("CMCC 训练数据提取任务")
    logging.info("=" * 70 + "\n")

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
    logging.info("\n" + "=" * 70)
    if stats['missing'] == 0:
        logging.info("✅ 提取完成！所有样本均成功提取")
        sys.exit(0)
    elif stats['missing'] < stats['total'] * 0.05:
        logging.info(f"⚠️  提取完成，但有 {stats['missing']} 个样本缺失（< 5%）")
        sys.exit(0)
    else:
        logging.error(f"❌ 提取完成，但有 {stats['missing']} 个样本缺失（> 5%），请检查数据源")
        sys.exit(1)
    logging.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()

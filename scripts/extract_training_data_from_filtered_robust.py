#!/usr/bin/env python3
"""
从 CMCC 原始 WDS 数据集中提取筛选后的训练样本（增强容错版本）

增强功能：
  1. 跳过损坏的 tar 文件，继续处理其他文件
  2. 对部分损坏的 tar 尽可能提取完好的内容
  3. 详细记录损坏文件信息
  4. 多次尝试机制
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
import io

# ==================== 配置参数 ====================

# 数据集路径映射
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


# ==================== 容错工具函数 ====================

def safe_list_tar_members(tar_path):
    """
    安全地列出 tar 文件中的成员，处理部分损坏的情况

    返回：(member_names, is_corrupted)
    """
    member_names = []
    is_corrupted = False

    try:
        # 方法 1：尝试正常读取
        with tarfile.open(tar_path, 'r') as tar:
            member_names = tar.getnames()
            return member_names, False
    except Exception as e1:
        # 方法 2：尝试忽略错误，读取尽可能多的内容
        try:
            with tarfile.open(tar_path, 'r', errorlevel=0) as tar:
                # errorlevel=0 表示忽略所有错误
                member_names = []
                try:
                    for member in tar:
                        member_names.append(member.name)
                except Exception:
                    # 读取到损坏部分，返回已读取的内容
                    pass

                if member_names:
                    logging.debug(f"部分读取 {tar_path}: 获取 {len(member_names)} 个文件（tar 可能部分损坏）")
                    return member_names, True
        except Exception as e2:
            logging.debug(f"无法读取 {tar_path}: {e2}")

    return [], True


def safe_extract_from_tar(tar_path, member_names, output_dir):
    """
    安全地从 tar 文件中提取指定文件，处理部分损坏的情况

    返回：(success, extracted_count)
    """
    extracted = []

    try:
        # 方法 1：尝试正常提取
        with tarfile.open(tar_path, 'r') as tar:
            for member_name in member_names:
                try:
                    member = tar.getmember(member_name)
                    tar.extract(member, output_dir)
                    extracted.append(member_name)
                except KeyError:
                    logging.debug(f"文件 {member_name} 不在 tar 中")
                    continue
                except Exception as e:
                    logging.debug(f"提取 {member_name} 失败: {e}")
                    continue

        return len(extracted) == len(member_names), len(extracted)

    except Exception as e1:
        # 方法 2：尝试使用容错模式提取
        try:
            with tarfile.open(tar_path, 'r', errorlevel=0) as tar:
                for member_name in member_names:
                    try:
                        member = tar.getmember(member_name)
                        tar.extract(member, output_dir)
                        extracted.append(member_name)
                    except Exception:
                        continue

            return len(extracted) == len(member_names), len(extracted)

        except Exception as e2:
            logging.error(f"从 {tar_path} 提取失败: {e2}")
            return False, len(extracted)


# ==================== 核心函数 ====================

def extract_dataset_name(sample_id):
    """从 sample_id 提取数据集名称"""
    for dataset in DATASET_MAPPING.keys():
        if sample_id.startswith(dataset):
            return dataset
    return None


def find_sample_in_tar_shards(sample_id, dataset_dir, corrupted_tars):
    """
    在 tar 分片中查找样本文件（增强容错版本）

    返回：(tar_path, member_names) 或 None
    """
    if not dataset_dir.exists():
        logging.warning(f"数据集目录不存在: {dataset_dir}")
        return None

    required_files = [f"{sample_id}{ext}" for ext in REQUIRED_EXTENSIONS]

    # 遍历所有 worker 目录
    for worker_dir in sorted(dataset_dir.glob("w*")):
        if not worker_dir.is_dir():
            continue

        # 遍历该 worker 目录下的所有 tar 文件
        for tar_path in sorted(worker_dir.glob("*.tar")):
            # 跳过已知完全损坏的 tar
            if str(tar_path) in corrupted_tars.get('fatal', set()):
                continue

            # 安全地列出 tar 成员
            members, is_corrupted = safe_list_tar_members(tar_path)

            if is_corrupted and members:
                # 记录部分损坏的 tar
                corrupted_tars['partial'].add(str(tar_path))
            elif is_corrupted and not members:
                # 记录完全损坏的 tar
                corrupted_tars['fatal'].add(str(tar_path))
                continue

            # 检查是否包含所有必需文件
            if all(f in members for f in required_files):
                return (tar_path, required_files)

    return None


def extract_training_data(filtered_list, data_root, output_dir, dry_run=False):
    """主提取函数（增强容错版本）"""

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
        'partial_success': 0,
        'by_dataset': defaultdict(lambda: {'success': 0, 'missing': 0, 'partial': 0}),
        'by_rating': defaultdict(lambda: {'success': 0, 'missing': 0, 'partial': 0}),
        'corrupted_tars': {'partial': set(), 'fatal': set()}
    }

    missing_samples = []

    # 逐条提取
    logging.info(f"🔍 开始提取样本...")
    logging.info(f"  数据源: {data_root}")
    logging.info(f"  输出到: {output_dir}")
    logging.info(f"  模式: {'DRY RUN（仅检查）' if dry_run else '正式提取'}")
    logging.info(f"  容错: 启用（跳过损坏的 tar 文件）\n")

    for i, sample in enumerate(filtered_samples, 1):
        sample_id = sample['sample_id']
        quality_rating = sample.get('quality_rating', 'unknown')
        dataset_name = extract_dataset_name(sample_id)

        # 进度显示
        if i % 100 == 0 or i == len(filtered_samples):
            logging.info(f"  ⏳ 进度：{i}/{len(filtered_samples)} ({i/len(filtered_samples)*100:.1f}%) | "
                        f"成功: {stats['success']} | 缺失: {stats['missing']}")

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
        result = find_sample_in_tar_shards(sample_id, dataset_dir, stats['corrupted_tars'])

        if result:
            tar_path, member_names = result

            # 提取文件
            if not dry_run:
                success, extracted_count = safe_extract_from_tar(tar_path, member_names, output_path)

                if success:
                    stats['success'] += 1
                    stats['by_dataset'][dataset_name]['success'] += 1
                    stats['by_rating'][quality_rating]['success'] += 1
                elif extracted_count > 0:
                    # 部分成功
                    stats['partial_success'] += 1
                    stats['by_dataset'][dataset_name]['partial'] += 1
                    stats['by_rating'][quality_rating]['partial'] += 1
                    missing_samples.append({
                        'sample_id': sample_id,
                        'reason': 'partial_extraction',
                        'extracted_count': extracted_count,
                        'total_files': len(member_names),
                        'dataset': dataset_name,
                        'quality_rating': quality_rating
                    })
                else:
                    stats['missing'] += 1
                    stats['by_dataset'][dataset_name]['missing'] += 1
                    stats['by_rating'][quality_rating]['missing'] += 1
                    missing_samples.append({
                        'sample_id': sample_id,
                        'reason': 'extraction_failed',
                        'dataset': dataset_name,
                        'quality_rating': quality_rating
                    })
            else:
                # dry-run 模式
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
    """生成提取报告（增强版）"""

    report = []
    report.append("=" * 70)
    report.append("CMCC 训练数据提取报告（增强容错版本）")
    report.append("=" * 70)
    report.append("")

    # 总体统计
    report.append("【一、总体统计】")
    report.append(f"  总样本数：{stats['total']}")
    report.append(f"  完全成功：{stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    if stats['partial_success'] > 0:
        report.append(f"  部分成功：{stats['partial_success']} ({stats['partial_success']/stats['total']*100:.1f}%)")
    report.append(f"  缺失样本：{stats['missing']} ({stats['missing']/stats['total']*100:.1f}%)")
    report.append("")

    # Tar 文件损坏统计
    if stats['corrupted_tars']['partial'] or stats['corrupted_tars']['fatal']:
        report.append("【二、Tar 文件损坏统计】")
        report.append(f"  部分损坏（可读取部分内容）：{len(stats['corrupted_tars']['partial'])} 个")
        report.append(f"  完全损坏（无法读取）：{len(stats['corrupted_tars']['fatal'])} 个")
        report.append("")

        if stats['corrupted_tars']['partial']:
            report.append("  部分损坏的 tar 文件（前 10 个）：")
            for tar in list(stats['corrupted_tars']['partial'])[:10]:
                report.append(f"    - {tar}")
            if len(stats['corrupted_tars']['partial']) > 10:
                report.append(f"    ... 还有 {len(stats['corrupted_tars']['partial']) - 10} 个")

        if stats['corrupted_tars']['fatal']:
            report.append("  完全损坏的 tar 文件（前 10 个）：")
            for tar in list(stats['corrupted_tars']['fatal'])[:10]:
                report.append(f"    - {tar}")
            if len(stats['corrupted_tars']['fatal']) > 10:
                report.append(f"    ... 还有 {len(stats['corrupted_tars']['fatal']) - 10} 个")
        report.append("")

    # 按数据集统计
    report.append("【三、按数据集分类】")
    for dataset in sorted(stats['by_dataset'].keys()):
        data = stats['by_dataset'][dataset]
        total = data['success'] + data['missing'] + data.get('partial', 0)
        report.append(f"  {dataset}:")
        report.append(f"    - 成功：{data['success']}/{total} ({data['success']/total*100:.1f}%)")
        if data.get('partial', 0) > 0:
            report.append(f"    - 部分：{data['partial']}/{total}")
        report.append(f"    - 缺失：{data['missing']}/{total}")
    report.append("")

    # 按评级统计
    report.append("【四、按评级分类】")
    for rating in sorted(stats['by_rating'].keys()):
        data = stats['by_rating'][rating]
        total = data['success'] + data['missing'] + data.get('partial', 0)
        report.append(f"  {rating.capitalize()}:")
        report.append(f"    - 成功：{data['success']}/{total} ({data['success']/total*100:.1f}%)")
        if data.get('partial', 0) > 0:
            report.append(f"    - 部分：{data['partial']}/{total}")
        report.append(f"    - 缺失：{data['missing']}/{total}")
    report.append("")

    # 输出文件统计
    if stats['success'] > 0:
        report.append("【五、输出产物】")
        report.append(f"  输出目录：{output_dir}")
        report.append(f"  文件总数：{stats['success'] * 5} (每样本 5 个文件)")
        report.append(f"  预估大小：~{stats['success'] * 50} MB")
        report.append("")

    # 缺失样本列表
    if missing_samples:
        report.append("【六、缺失/异常样本清单】")
        report.append(f"  共 {len(missing_samples)} 个样本")
        report.append("")

        # 按原因分组
        by_reason = defaultdict(list)
        for sample in missing_samples:
            reason = sample.get('reason', 'unknown')
            by_reason[reason].append(sample['sample_id'])

        for reason, sample_ids in sorted(by_reason.items()):
            report.append(f"  原因: {reason} ({len(sample_ids)} 个):")
            for sample_id in sample_ids[:5]:  # 最多显示 5 个
                report.append(f"    - {sample_id}")
            if len(sample_ids) > 5:
                report.append(f"    ... 还有 {len(sample_ids) - 5} 个")
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

    # 保存损坏的 tar 列表
    if stats['corrupted_tars']['partial'] or stats['corrupted_tars']['fatal']:
        corrupted_file = Path(output_dir) / "corrupted_tars.txt"
        with open(corrupted_file, 'w', encoding='utf-8') as f:
            f.write("部分损坏的 tar 文件:\n")
            for tar in sorted(stats['corrupted_tars']['partial']):
                f.write(f"{tar}\n")
            f.write("\n完全损坏的 tar 文件:\n")
            for tar in sorted(stats['corrupted_tars']['fatal']):
                f.write(f"{tar}\n")
        logging.info(f"📋 损坏 tar 清单已保存到: {corrupted_file}")


def main():
    parser = argparse.ArgumentParser(
        description="从 CMCC WDS tar 分片中提取筛选后的训练样本（增强容错版本）"
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
    logging.info("CMCC 训练数据提取任务（增强容错版本）")
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
    success_rate = stats['success'] / stats['total'] * 100

    if stats['missing'] == 0:
        logging.info("✅ 提取完成！所有样本均成功提取")
        sys.exit(0)
    elif success_rate >= 95:
        logging.info(f"✅ 提取完成！成功率: {success_rate:.1f}% (>= 95%)")
        if stats['corrupted_tars']['partial'] or stats['corrupted_tars']['fatal']:
            logging.info(f"⚠️  发现 {len(stats['corrupted_tars']['partial']) + len(stats['corrupted_tars']['fatal'])} 个损坏的 tar 文件，已跳过")
        sys.exit(0)
    else:
        logging.warning(f"⚠️  提取完成，成功率: {success_rate:.1f}% (< 95%)")
        logging.warning(f"   缺失样本: {stats['missing']} 个")
        if stats['corrupted_tars']['partial'] or stats['corrupted_tars']['fatal']:
            logging.warning(f"   损坏 tar: {len(stats['corrupted_tars']['partial']) + len(stats['corrupted_tars']['fatal'])} 个")
        sys.exit(1)
    logging.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()

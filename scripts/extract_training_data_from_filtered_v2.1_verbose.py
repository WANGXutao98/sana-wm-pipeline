#!/usr/bin/env python3
"""
从 CMCC 原始 WDS 数据集中提取筛选后的训练样本（增强容错 + 实时进度版本）

v2.1 改进：
  - 增加实时进度输出（每个样本都显示）
  - 增加 tar 文件遍历进度
  - 增加耗时统计
  - 优化日志输出
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
import time

# ==================== 配置参数 ====================

DATASET_MAPPING = {
    "RealEstate10K": "final_wds-RealEstate10K-360p/wds-RealEstate10K-360p",
    "SpatialVID-hq": "final_wds-SpatialVID-hq/wds-SpatialVID-hq",
    "sekai-real-walking-hq": "final_wds-sekai-real-walking-hq/wds-sekai-real-walking-hq",
    "DL3DV-ALL-2K": "final_wds-DL3DV-ALL-2K/wds-DL3DV-ALL-2K"
}

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
    """安全地列出 tar 文件中的成员"""
    member_names = []
    is_corrupted = False

    try:
        with tarfile.open(tar_path, 'r') as tar:
            member_names = tar.getnames()
            return member_names, False
    except Exception:
        try:
            with tarfile.open(tar_path, 'r', errorlevel=0) as tar:
                member_names = []
                try:
                    for member in tar:
                        member_names.append(member.name)
                except Exception:
                    pass
                if member_names:
                    return member_names, True
        except Exception:
            pass

    return [], True


def safe_extract_from_tar(tar_path, member_names, output_dir):
    """安全地从 tar 文件中提取指定文件"""
    extracted = []

    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member_name in member_names:
                try:
                    member = tar.getmember(member_name)
                    tar.extract(member, output_dir)
                    extracted.append(member_name)
                except:
                    continue
        return len(extracted) == len(member_names), len(extracted)
    except Exception:
        try:
            with tarfile.open(tar_path, 'r', errorlevel=0) as tar:
                for member_name in member_names:
                    try:
                        member = tar.getmember(member_name)
                        tar.extract(member, output_dir)
                        extracted.append(member_name)
                    except:
                        continue
            return len(extracted) == len(member_names), len(extracted)
        except Exception:
            return False, len(extracted)


# ==================== 核心函数 ====================

def extract_dataset_name(sample_id):
    """从 sample_id 提取数据集名称"""
    for dataset in DATASET_MAPPING.keys():
        if sample_id.startswith(dataset):
            return dataset
    return None


def find_sample_in_tar_shards(sample_id, dataset_dir, corrupted_tars, sample_idx, total_samples):
    """在 tar 分片中查找样本文件（带进度输出）"""
    if not dataset_dir.exists():
        logging.warning(f"数据集目录不存在: {dataset_dir}")
        return None

    required_files = [f"{sample_id}{ext}" for ext in REQUIRED_EXTENSIONS]

    # 遍历所有 worker 目录
    worker_dirs = sorted(dataset_dir.glob("w*"))
    logging.info(f"  [{sample_idx}/{total_samples}] 查找样本: {sample_id}")
    logging.info(f"    扫描 {len(worker_dirs)} 个 worker 目录...")

    for worker_idx, worker_dir in enumerate(worker_dirs, 1):
        if not worker_dir.is_dir():
            continue

        # 获取 tar 文件列表
        tar_files = sorted(worker_dir.glob("*.tar"))
        logging.info(f"    Worker {worker_dir.name}: {len(tar_files)} 个 tar 文件")

        # 遍历 tar 文件
        for tar_idx, tar_path in enumerate(tar_files, 1):
            # 每 10 个 tar 显示进度
            if tar_idx % 10 == 0:
                logging.info(f"      进度: {tar_idx}/{len(tar_files)} tar 文件已扫描")

            # 跳过已知损坏的 tar
            if str(tar_path) in corrupted_tars.get('fatal', set()):
                continue

            # 安全地列出 tar 成员
            start_time = time.time()
            members, is_corrupted = safe_list_tar_members(tar_path)
            elapsed = time.time() - start_time

            if elapsed > 2:  # 读取耗时超过 2 秒
                logging.info(f"      ⚠️  {tar_path.name} 读取耗时 {elapsed:.1f}s")

            if is_corrupted and members:
                corrupted_tars['partial'].add(str(tar_path))
            elif is_corrupted and not members:
                corrupted_tars['fatal'].add(str(tar_path))
                continue

            # 检查是否包含所有必需文件
            if all(f in members for f in required_files):
                logging.info(f"    ✅ 找到！位置: {tar_path.name}")
                return (tar_path, required_files)

    logging.warning(f"    ❌ 未找到样本: {sample_id}")
    return None


def extract_training_data(filtered_list, data_root, output_dir, dry_run=False):
    """主提取函数（增强进度输出版本）"""

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
    overall_start_time = time.time()

    # 逐条提取
    logging.info(f"🔍 开始提取样本...")
    logging.info(f"  数据源: {data_root}")
    logging.info(f"  输出到: {output_dir}")
    logging.info(f"  模式: {'DRY RUN（仅检查）' if dry_run else '正式提取'}")
    logging.info(f"  容错: 启用（跳过损坏的 tar 文件）")
    logging.info(f"  进度: 实时显示每个样本\n")

    for i, sample in enumerate(filtered_samples, 1):
        sample_start_time = time.time()

        sample_id = sample['sample_id']
        quality_rating = sample.get('quality_rating', 'unknown')
        dataset_name = extract_dataset_name(sample_id)

        # 显示总体进度
        elapsed_total = time.time() - overall_start_time
        avg_time = elapsed_total / i if i > 0 else 0
        remaining = (len(filtered_samples) - i) * avg_time

        logging.info(f"\n{'='*70}")
        logging.info(f"样本 {i}/{len(filtered_samples)} ({i/len(filtered_samples)*100:.1f}%)")
        logging.info(f"已用时间: {elapsed_total/60:.1f} 分钟 | 预计剩余: {remaining/60:.1f} 分钟")
        logging.info(f"成功: {stats['success']} | 缺失: {stats['missing']}")
        logging.info(f"{'='*70}")

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
        result = find_sample_in_tar_shards(
            sample_id, dataset_dir, stats['corrupted_tars'], i, len(filtered_samples)
        )

        if result:
            tar_path, member_names = result

            # 提取文件
            if not dry_run:
                success, extracted_count = safe_extract_from_tar(tar_path, member_names, output_path)

                if success:
                    stats['success'] += 1
                    stats['by_dataset'][dataset_name]['success'] += 1
                    stats['by_rating'][quality_rating]['success'] += 1
                    logging.info(f"  ✅ 提取成功")
                elif extracted_count > 0:
                    stats['partial_success'] += 1
                    stats['by_dataset'][dataset_name]['partial'] += 1
                    stats['by_rating'][quality_rating]['partial'] += 1
                    logging.warning(f"  ⚠️  部分提取: {extracted_count}/{len(member_names)} 文件")
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
                    logging.error(f"  ❌ 提取失败")
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
                logging.info(f"  ✅ 找到（dry-run 模式）")
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

        # 显示单样本耗时
        sample_elapsed = time.time() - sample_start_time
        logging.info(f"  本样本耗时: {sample_elapsed:.1f} 秒")

    return stats, missing_samples


def generate_report(stats, missing_samples, output_dir):
    """生成提取报告"""
    report = []
    report.append("=" * 70)
    report.append("CMCC 训练数据提取报告（增强容错 + 实时进度版本）")
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
        report.append(f"  部分损坏：{len(stats['corrupted_tars']['partial'])} 个")
        report.append(f"  完全损坏：{len(stats['corrupted_tars']['fatal'])} 个")
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

    report.append("=" * 70)

    report_text = "\n".join(report)
    logging.info("\n" + report_text)

    # 保存报告
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
        description="从 CMCC WDS tar 分片中提取筛选后的训练样本（v2.1 实时进度版）"
    )

    parser.add_argument('--filtered_list', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_file', type=str, default='extraction.log')
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    # 设置日志
    log_path = Path(args.output_dir) / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    if not Path(args.filtered_list).exists():
        logging.error(f"❌ 筛选列表文件不存在：{args.filtered_list}")
        sys.exit(1)

    if not Path(args.data_root).exists():
        logging.error(f"❌ 数据根目录不存在：{args.data_root}")
        sys.exit(1)

    logging.info("\n" + "=" * 70)
    logging.info("CMCC 训练数据提取任务 v2.1（增强容错 + 实时进度版本）")
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
        sys.exit(0)
    else:
        logging.warning(f"⚠️  提取完成，成功率: {success_rate:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()

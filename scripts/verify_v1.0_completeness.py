#!/usr/bin/env python3
"""
v1.0 数据完整性验证脚本

功能：
1. 检查每个样本是否有完整的 5 个文件
2. 识别缺失文件的样本
3. 生成详细报告
"""

import json
from pathlib import Path
from collections import defaultdict

# ==================== 配置 ====================

V1_0_FILTERED_LIST = "/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/sana-human-feedback/filtered_training_samples.jsonl"
V1_0_OUTPUT_DIR = "/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0"
EXTRACTION_REPORT = f"{V1_0_OUTPUT_DIR}/extraction_report.txt"
MISSING_SAMPLES = f"{V1_0_OUTPUT_DIR}/missing_samples.jsonl"

REQUIRED_EXTENSIONS = [
    ".mp4",
    ".poses_c2w.npy",
    ".intrinsics.npy",
    ".scale.npy",
    ".caption.txt"
]

# ==================== 主函数 ====================

def load_filtered_samples():
    """加载筛选列表"""
    samples = []
    with open(V1_0_FILTERED_LIST, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def scan_output_directory():
    """扫描输出目录，按 sample_id 分组文件"""
    output_path = Path(V1_0_OUTPUT_DIR)

    # ponytail: 精确跳过已知报告文件，不要用扩展名判断
    SKIP_FILES = {
        'extraction_report.txt',
        'missing_samples.jsonl',
        'completeness_verification_report.txt',
        'incomplete_samples.jsonl'
    }

    sample_files = defaultdict(list)

    for file_path in output_path.glob("*"):
        if file_path.is_file():
            # 跳过已知报告文件
            if file_path.name in SKIP_FILES:
                continue

            filename = file_path.name

            # 匹配必需扩展名
            for ext in REQUIRED_EXTENSIONS:
                if filename.endswith(ext):
                    sample_id = filename[:-len(ext)]
                    sample_files[sample_id].append(ext)
                    break

    return sample_files


def verify_completeness(filtered_samples, sample_files):
    """验证每个样本的文件完整性"""

    results = {
        'total_samples': len(filtered_samples),
        'complete_samples': 0,
        'incomplete_samples': 0,
        'missing_samples': 0,
        'complete_list': [],
        'incomplete_list': [],
        'missing_list': []
    }

    for sample in filtered_samples:
        sample_id = sample['sample_id']

        if sample_id not in sample_files:
            # 完全缺失（0 个文件）
            results['missing_samples'] += 1
            results['missing_list'].append({
                'sample_id': sample_id,
                'found_files': 0,
                'missing_files': REQUIRED_EXTENSIONS,
                'quality_rating': sample.get('quality_rating', 'unknown')
            })
        else:
            found_files = sample_files[sample_id]

            if len(found_files) == 5:
                # 完整（5 个文件）
                results['complete_samples'] += 1
                results['complete_list'].append(sample_id)
            else:
                # 不完整（1-4 个文件）
                missing_files = [ext for ext in REQUIRED_EXTENSIONS if ext not in found_files]
                results['incomplete_samples'] += 1
                results['incomplete_list'].append({
                    'sample_id': sample_id,
                    'found_files': len(found_files),
                    'found_extensions': found_files,
                    'missing_files': missing_files,
                    'quality_rating': sample.get('quality_rating', 'unknown')
                })

    return results


def generate_report(results):
    """生成详细报告"""

    report = []
    report.append("=" * 80)
    report.append("v1.0 数据完整性验证报告")
    report.append("=" * 80)
    report.append("")

    # 总体统计
    report.append("【一、总体统计】")
    report.append(f"  总样本数：{results['total_samples']}")
    report.append(f"  完整样本：{results['complete_samples']} ({results['complete_samples']/results['total_samples']*100:.1f}%)")
    report.append(f"  不完整样本：{results['incomplete_samples']} ({results['incomplete_samples']/results['total_samples']*100:.1f}%)")
    report.append(f"  完全缺失：{results['missing_samples']} ({results['missing_samples']/results['total_samples']*100:.1f}%)")
    report.append("")

    # 文件数统计
    total_expected = results['total_samples'] * 5
    total_found = results['complete_samples'] * 5 + sum(item['found_files'] for item in results['incomplete_list'])
    report.append("【二、文件数统计】")
    report.append(f"  预期文件总数：{total_expected}")
    report.append(f"  实际文件数：{total_found}")
    report.append(f"  缺失文件数：{total_expected - total_found}")
    report.append("")

    # 完全缺失的样本
    if results['missing_list']:
        report.append("【三、完全缺失的样本】")
        report.append(f"  共 {len(results['missing_list'])} 个样本")
        report.append("")
        for item in results['missing_list'][:20]:  # 最多显示 20 个
            report.append(f"  • {item['sample_id']} ({item['quality_rating']})")
        if len(results['missing_list']) > 20:
            report.append(f"  ... 还有 {len(results['missing_list']) - 20} 个")
        report.append("")

    # 不完整的样本
    if results['incomplete_list']:
        report.append("【四、不完整的样本（部分文件缺失）】")
        report.append(f"  共 {len(results['incomplete_list'])} 个样本")
        report.append("")
        for item in results['incomplete_list'][:20]:  # 最多显示 20 个
            report.append(f"  • {item['sample_id']} ({item['quality_rating']})")
            report.append(f"    - 已有文件：{item['found_files']}/5")
            report.append(f"    - 缺失扩展名：{', '.join(item['missing_files'])}")
        if len(results['incomplete_list']) > 20:
            report.append(f"  ... 还有 {len(results['incomplete_list']) - 20} 个")
        report.append("")

    # 问题诊断
    report.append("【五、问题诊断】")
    if results['missing_samples'] > 0:
        report.append(f"  ✓ {results['missing_samples']} 个样本完全缺失")
        report.append(f"    原因：提取报告中标记为 'files_not_found' 或 'extraction_failed'")
    if results['incomplete_samples'] > 0:
        report.append(f"  ✓ {results['incomplete_samples']} 个样本部分文件缺失")
        report.append(f"    原因：tar 文件损坏或提取过程中断")
    if results['complete_samples'] == results['total_samples']:
        report.append("  ✅ 所有样本文件完整！")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def save_incomplete_list(results):
    """保存不完整样本列表，用于重新提取"""

    output_file = f"{V1_0_OUTPUT_DIR}/incomplete_samples.jsonl"

    # 合并完全缺失和不完整的样本
    all_incomplete = []

    # 从原始筛选列表中读取完整信息
    with open(V1_0_FILTERED_LIST, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            sample_id = sample['sample_id']

            # 检查是否在缺失或不完整列表中
            if any(item['sample_id'] == sample_id for item in results['missing_list']):
                all_incomplete.append(sample)
            elif any(item['sample_id'] == sample_id for item in results['incomplete_list']):
                all_incomplete.append(sample)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_incomplete:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    return output_file, len(all_incomplete)


def main():
    print("=" * 80)
    print("v1.0 数据完整性验证")
    print("=" * 80)
    print("")

    # 1. 加载筛选列表
    print("Step 1: 加载筛选列表...")
    filtered_samples = load_filtered_samples()
    print(f"  ✓ 加载 {len(filtered_samples)} 个样本")

    # 2. 扫描输出目录
    print("\nStep 2: 扫描输出目录...")
    sample_files = scan_output_directory()
    print(f"  ✓ 发现 {len(sample_files)} 个样本的文件")

    # 3. 验证完整性
    print("\nStep 3: 验证文件完整性...")
    results = verify_completeness(filtered_samples, sample_files)
    print(f"  ✓ 完整样本：{results['complete_samples']}")
    print(f"  ✓ 不完整样本：{results['incomplete_samples']}")
    print(f"  ✓ 完全缺失：{results['missing_samples']}")

    # 4. 生成报告
    print("\nStep 4: 生成报告...")
    report = generate_report(results)
    print(report)

    # 保存报告
    report_file = f"{V1_0_OUTPUT_DIR}/completeness_verification_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n📊 报告已保存：{report_file}")

    # 5. 保存不完整样本列表
    if results['incomplete_samples'] > 0 or results['missing_samples'] > 0:
        print("\nStep 5: 保存不完整样本列表...")
        output_file, count = save_incomplete_list(results)
        print(f"  ✓ 已保存 {count} 个不完整样本到：{output_file}")
        print(f"\n💡 提示：可使用此列表重新提取缺失样本：")
        print(f"  python3 extract_training_data_from_filtered_corrected.py \\")
        print(f"    --filtered_list {output_file} \\")
        print(f"    --data_root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \\")
        print(f"    --output_dir {V1_0_OUTPUT_DIR}")
    else:
        print("\n✅ 所有样本文件完整，无需重新提取！")

    print("\n" + "=" * 80)
    print("验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

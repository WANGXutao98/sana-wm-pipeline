#!/usr/bin/env python3
"""
提取审查样本的实际数据文件（视频、pose、caption等）
并打包成可分发的批次
"""
import json
import tarfile
import shutil
from pathlib import Path
from collections import defaultdict
import tempfile

REVIEW_SAMPLES_FILE = Path("human_review_samples/review_samples.jsonl")
OUTPUT_BASE = Path("review_packages_with_data")
BATCH_SIZE = 800
TEMP_EXTRACT_DIR = Path("/tmp/review_extract")

def load_samples():
    """加载所有审查样本"""
    samples = []
    with open(REVIEW_SAMPLES_FILE, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_sample_files(sample, extract_dir):
    """
    从 tar 文件中提取一个样本的所有文件

    返回：提取的文件列表
    """
    sample_id = sample['sample_id']
    tar_path = sample['tar_path']

    if not Path(tar_path).exists():
        print(f"  ⚠️  Tar 文件不存在: {tar_path}")
        return []

    # 需要提取的文件后缀
    extensions = ['.mp4', '.poses_c2w.npy', '.intrinsics.npy', '.scale.npy', '.caption.txt', '.meta.json']

    extracted_files = []

    try:
        with tarfile.open(tar_path, 'r') as tar:
            # 查找该样本的所有文件
            for member in tar.getmembers():
                if member.name.startswith(sample_id):
                    # 提取文件
                    tar.extract(member, path=extract_dir)
                    extracted_files.append(extract_dir / member.name)

    except Exception as e:
        print(f"  ❌ 提取失败 {sample_id}: {e}")
        return []

    return extracted_files


def create_batch_package(batch_num, batch_samples, total_batches):
    """创建一个批次的完整包（包含实际数据文件）"""

    print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_samples)} 个样本)...")

    # 创建批次目录
    batch_dir = OUTPUT_BASE / f"batch_{batch_num:02d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据目录
    data_dir = batch_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # 创建临时提取目录
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 保存样本元数据
    samples_file = batch_dir / "samples.jsonl"
    with open(samples_file, 'w') as f:
        for sample in batch_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 2. 提取实际数据文件
    success_count = 0
    fail_count = 0

    for i, sample in enumerate(batch_samples, 1):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(batch_samples)}...")

        sample_id = sample['sample_id']

        # 提取文件
        extracted = extract_sample_files(sample, TEMP_EXTRACT_DIR)

        if extracted:
            # 移动到批次数据目录
            for src_file in extracted:
                dst_file = data_dir / src_file.name
                shutil.move(str(src_file), str(dst_file))
            success_count += 1
        else:
            fail_count += 1

    # 清理临时目录
    if TEMP_EXTRACT_DIR.exists():
        shutil.rmtree(TEMP_EXTRACT_DIR)

    print(f"  ✅ 成功提取: {success_count}, 失败: {fail_count}")

    # 3. 生成批次统计
    stats_file = batch_dir / "batch_stats.txt"
    by_group = defaultdict(int)
    by_verdict = defaultdict(int)

    for s in batch_samples:
        by_group[s['review_group']] += 1
        by_verdict[s['verdict']] += 1

    with open(stats_file, 'w') as f:
        f.write(f"批次 {batch_num} 统计信息\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"样本数: {len(batch_samples)}\n")
        f.write(f"成功提取: {success_count}\n")
        f.write(f"提取失败: {fail_count}\n\n")

        f.write("按 Group 分布:\n")
        for group, count in sorted(by_group.items()):
            f.write(f"  {group:<30} {count:>5}\n")

        f.write("\n按 Verdict 分布:\n")
        for verdict, count in sorted(by_verdict.items()):
            f.write(f"  {verdict:<30} {count:>5}\n")

    # 4. 生成 README
    readme_file = batch_dir / "README.txt"
    with open(readme_file, 'w') as f:
        f.write(f"========================================\n")
        f.write(f"审查批次 {batch_num}/{total_batches}\n")
        f.write(f"========================================\n\n")
        f.write(f"样本数: {len(batch_samples)}\n")
        f.write(f"成功提取: {success_count}\n\n")

        f.write(f"文件结构:\n")
        f.write(f"---------\n")
        f.write(f"- samples.jsonl: 样本元数据\n")
        f.write(f"- data/: 实际数据文件\n")
        f.write(f"  - <sample_id>.mp4: 视频\n")
        f.write(f"  - <sample_id>.poses_c2w.npy: Pose 数据\n")
        f.write(f"  - <sample_id>.caption.txt: Caption\n")
        f.write(f"  - <sample_id>.meta.json: 元数据\n")
        f.write(f"- annotation_results.jsonl: 标注结果（请填写）\n")
        f.write(f"- batch_stats.txt: 统计信息\n\n")

        f.write(f"审查方法:\n")
        f.write(f"---------\n")
        f.write(f"1. 读取 samples.jsonl 中的一行\n")
        f.write(f"2. 根据 sample_id 找到 data/ 下的对应文件\n")
        f.write(f"3. 观看视频，查看 metrics，阅读 caption\n")
        f.write(f"4. 做出质量判断\n")
        f.write(f"5. 将结果写入 annotation_results.jsonl\n\n")

        f.write(f"详细指南请参考: ANNOTATOR_TRAINING_GUIDE.md\n")

    # 5. 创建空的标注结果文件
    (batch_dir / "annotation_results.jsonl").touch()

    # 6. 打包
    print(f"  正在打包...")
    batch_tar = OUTPUT_BASE / f"batch_{batch_num:02d}.tar.gz"
    with tarfile.open(batch_tar, 'w:gz') as tar:
        tar.add(batch_dir, arcname=f"batch_{batch_num:02d}")

    # 清理解压的目录
    shutil.rmtree(batch_dir)

    size_mb = batch_tar.stat().st_size / (1024 * 1024)
    print(f"  ✅ 已打包: {batch_tar.name} ({size_mb:.1f} MB)")

    return batch_tar, success_count, fail_count


def main():
    print("=" * 60)
    print("提取并打包审查样本（包含实际数据文件）")
    print("=" * 60)
    print()
    print("⚠️  警告：此过程会提取大量文件，可能需要 30-60 分钟")
    print()

    # 加载样本
    samples = load_samples()
    print(f"✅ 加载 {len(samples)} 个审查样本")
    print()

    # 创建输出目录
    OUTPUT_BASE.mkdir(exist_ok=True)

    # 分批
    num_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"分批策略: 每批 {BATCH_SIZE} 样本，共 {num_batches} 批")
    print()

    total_success = 0
    total_fail = 0

    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(batch_num * BATCH_SIZE, len(samples))
        batch_samples = samples[start_idx:end_idx]

        batch_tar, success, fail = create_batch_package(batch_num, batch_samples, num_batches)
        total_success += success
        total_fail += fail

    print()
    print("=" * 60)
    print("打包完成")
    print("=" * 60)
    print()
    print(f"总样本数: {len(samples)}")
    print(f"成功提取: {total_success}")
    print(f"提取失败: {total_fail}")
    print()
    print(f"生成的文件:")
    for tar_file in sorted(OUTPUT_BASE.glob("batch_*.tar.gz")):
        size_mb = tar_file.stat().st_size / (1024 * 1024)
        print(f"  {tar_file.name} ({size_mb:.1f} MB)")
    print()
    print(f"下载命令:")
    print(f"  scp -r user@cmcc:/root/work/david_work/sana_wm_qc/{OUTPUT_BASE}/*.tar.gz /local/path/")


if __name__ == "__main__":
    main()

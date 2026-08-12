#!/usr/bin/env python3
"""
打包黄金样本的完整数据（包含视频、pose等）
"""
import json
import tarfile
import shutil
from pathlib import Path
from collections import defaultdict

GOLDEN_SAMPLES_FILE = Path("human_review_samples/golden_samples/golden_samples.jsonl")
OUTPUT_DIR = Path("review_packages/golden_samples_with_data")
TEMP_EXTRACT_DIR = Path("/tmp/golden_extract")

def load_golden_samples():
    """加载黄金样本"""
    samples = []
    with open(GOLDEN_SAMPLES_FILE, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def extract_sample_files(sample, extract_dir):
    """从 tar 文件中提取一个样本的所有文件"""
    sample_id = sample['sample_id']
    tar_path = sample['tar_path']

    if not Path(tar_path).exists():
        print(f"  ⚠️  Tar 文件不存在: {tar_path}")
        return []

    # 从 sample_id 提取关键部分用于匹配
    # 格式: DL3DV-ALL-2K_1K__<hash>__images_2
    # 在 tar 中可能是: <hash>.mp4, <hash>.json 等
    parts = sample_id.split('__')
    if len(parts) >= 2:
        hash_key = parts[1]  # 提取 hash
    else:
        hash_key = sample_id

    extracted_files = []
    sample_dir = extract_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, 'r') as tar:
            # 查找匹配的文件
            for member in tar.getmembers():
                # 匹配包含 hash 的文件
                if hash_key in member.name:
                    # 提取到样本目录
                    tar.extract(member, path=sample_dir)
                    extracted_path = sample_dir / member.name
                    if extracted_path.exists():
                        extracted_files.append(extracted_path)

                    # 找到足够的文件就停止（提高速度）
                    if len(extracted_files) >= 3:  # mp4, json, txt 等
                        break

    except Exception as e:
        print(f"  ❌ 提取失败 {sample_id}: {e}")
        return []

    return extracted_files


def main():
    print("=" * 60)
    print("打包黄金样本（包含完整数据）")
    print("=" * 60)
    print()

    # 加载黄金样本
    samples = load_golden_samples()
    print(f"✅ 加载 {len(samples)} 个黄金样本")
    print()

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 创建数据目录
    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    # 1. 保存样本元数据
    samples_file = OUTPUT_DIR / "golden_samples.jsonl"
    with open(samples_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print("✅ 元数据已保存")
    print()

    # 2. 提取实际数据文件
    print("开始提取数据文件...")
    print()

    success_count = 0
    fail_count = 0

    for i, sample in enumerate(samples, 1):
        sample_id = sample['sample_id']
        print(f"[{i}/{len(samples)}] 提取 {sample_id[:60]}...")

        # 提取文件
        extracted = extract_sample_files(sample, TEMP_EXTRACT_DIR)

        if extracted:
            # 为每个样本创建子目录
            sample_data_dir = data_dir / sample_id
            sample_data_dir.mkdir(exist_ok=True)

            # 移动文件
            for src_file in extracted:
                if src_file.is_file():
                    dst_file = sample_data_dir / src_file.name
                    shutil.move(str(src_file), str(dst_file))

            print(f"  ✅ 提取 {len(extracted)} 个文件")
            success_count += 1
        else:
            print(f"  ❌ 提取失败")
            fail_count += 1

    print()
    print("=" * 60)
    print("提取完成")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print()

    # 3. 复制培训文档
    doc_files = [
        "human_review_samples/golden_samples/golden_samples_summary.txt",
        "human_review_samples/golden_samples/annotation_template.txt",
    ]

    for doc in doc_files:
        src = Path(doc)
        if src.exists():
            shutil.copy(src, OUTPUT_DIR / src.name)
            print(f"✅ 复制文档: {src.name}")

    print()

    # 4. 创建说明文件
    readme = OUTPUT_DIR / "README.txt"
    with open(readme, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("黄金样本包（包含完整数据）\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"样本数: {len(samples)}\n")
        f.write(f"成功提取: {success_count}\n")
        f.write(f"提取失败: {fail_count}\n\n")
        f.write("文件结构:\n")
        f.write("-" * 60 + "\n")
        f.write("golden_samples.jsonl - 样本元数据\n")
        f.write("golden_samples_summary.txt - 人类可读摘要\n")
        f.write("annotation_template.txt - 标注模板\n")
        f.write("data/ - 实际数据文件\n")
        f.write("  └─ <sample_id>/ - 每个样本的数据\n")
        f.write("      ├─ *.mp4 - 视频文件\n")
        f.write("      ├─ *.json - Pose/元数据\n")
        f.write("      └─ *.txt - Caption\n\n")
        f.write("使用方法:\n")
        f.write("-" * 60 + "\n")
        f.write("1. 阅读 ANNOTATOR_TRAINING_GUIDE.md\n")
        f.write("2. 查看 golden_samples_summary.txt 了解样本\n")
        f.write("3. 对每个样本:\n")
        f.write("   - 查看 data/<sample_id>/ 中的视频和数据\n")
        f.write("   - 参考 golden_samples.jsonl 中的 metrics\n")
        f.write("   - 使用 annotation_template.txt 进行标注\n")
        f.write("4. 与专家结果对比，达到一致性 > 80%\n")

    print(f"✅ README 已创建")
    print()

    # 5. 打包
    print("开始打包...")
    package_file = OUTPUT_DIR.parent / "golden_samples_with_data.tar.gz"

    with tarfile.open(package_file, 'w:gz') as tar:
        tar.add(OUTPUT_DIR, arcname='golden_samples_with_data')

    package_size_mb = package_file.stat().st_size / (1024 * 1024)

    print()
    print("=" * 60)
    print("打包完成")
    print("=" * 60)
    print(f"输出文件: {package_file}")
    print(f"文件大小: {package_size_mb:.1f} MB")
    print()
    print("下载命令:")
    print(f"  scp user@cmcc:{package_file.absolute()} /local/path/")
    print()

    # 清理临时目录
    if TEMP_EXTRACT_DIR.exists():
        shutil.rmtree(TEMP_EXTRACT_DIR)
        print("✅ 临时文件已清理")


if __name__ == "__main__":
    main()

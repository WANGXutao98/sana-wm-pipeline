#!/bin/bash
# CMCC 机器上的 ModelScope 上传准备脚本
# 功能：生成 manifest.jsonl + 打包 tar + 验证完整性

set -e

# ==================== 配置 ====================
SOURCE_DIR="$HOME/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0"
OUTPUT_DIR="$HOME/work/filestorage/shangaoooooo/davidwang/jdvbbfb_modelscope_upload"
DATASET_NAME="jdvbbfb-guiyu-v1.0"

echo "=========================================="
echo "ModelScope 数据上传准备脚本"
echo "=========================================="
echo "源目录: $SOURCE_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "数据集名称: $DATASET_NAME"
echo ""

# 检查源目录
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ 错误：源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# ==================== 步骤 1: 生成 manifest.jsonl ====================
echo "步骤 1/4: 生成样本清单 manifest.jsonl"

python3 << 'EOF'
import os
import json
from pathlib import Path
from collections import defaultdict

SOURCE_DIR = os.path.expanduser("~/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0")
OUTPUT_DIR = os.path.expanduser("~/work/filestorage/shangaoooooo/davidwang/jdvbbfb_modelscope_upload")

# 扫描所有 mp4 文件
mp4_files = list(Path(SOURCE_DIR).glob("*.mp4"))
print(f"  发现 {len(mp4_files)} 个 mp4 文件")

samples = []
stats = defaultdict(int)
missing_files = []

for mp4 in sorted(mp4_files):
    sample_id = mp4.stem  # 去掉 .mp4

    # 检查 5 个必需文件
    required = {
        'mp4': mp4,
        'caption': mp4.with_suffix('.caption.txt'),
        'poses_c2w': Path(str(mp4).replace('.mp4', '.poses_c2w.npy')),
        'intrinsics': Path(str(mp4).replace('.mp4', '.intrinsics.npy')),
        'scale': Path(str(mp4).replace('.mp4', '.scale.npy'))
    }

    # 检查完整性
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        missing_files.append({'sample_id': sample_id, 'missing': missing})
        continue

    # 提取数据集名称
    dataset = sample_id.split('_')[0]
    stats[dataset] += 1

    # 收集文件大小
    total_size = sum(p.stat().st_size for p in required.values())

    samples.append({
        'sample_id': sample_id,
        'dataset': dataset,
        'files': {k: p.name for k, p in required.items()},
        'total_size_bytes': total_size
    })

# 写入 manifest.jsonl
manifest_path = Path(OUTPUT_DIR) / 'manifest.jsonl'
with open(manifest_path, 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"  ✅ 生成 manifest.jsonl: {len(samples)} 个完整样本")

# 写入统计信息
stats_path = Path(OUTPUT_DIR) / 'dataset_stats.json'
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump({
        'total_samples': len(samples),
        'total_size_gb': sum(s['total_size_bytes'] for s in samples) / 1e9,
        'by_dataset': dict(stats),
        'missing_samples': len(missing_files)
    }, f, indent=2, ensure_ascii=False)

print(f"  ✅ 生成 dataset_stats.json")

if missing_files:
    print(f"  ⚠️  发现 {len(missing_files)} 个不完整样本")
    with open(Path(OUTPUT_DIR) / 'missing_files.jsonl', 'w') as f:
        for item in missing_files:
            f.write(json.dumps(item) + '\n')

EOF

# ==================== 步骤 2: 打包 tar ====================
echo ""
echo "步骤 2/4: 打包数据为 ${DATASET_NAME}.tar"

# 读取样本数
SAMPLE_COUNT=$(wc -l < manifest.jsonl)
echo "  打包 $SAMPLE_COUNT 个样本（1980 × 5 = 9900 个文件）"

# 创建文件列表（从 manifest 提取所有文件名）
python3 << 'EOF'
import json
from pathlib import Path

OUTPUT_DIR = Path.home() / "work/filestorage/shangaoooooo/davidwang/jdvbbfb_modelscope_upload"
manifest = OUTPUT_DIR / 'manifest.jsonl'

with open(manifest, 'r') as f_in, open(OUTPUT_DIR / 'files_to_pack.txt', 'w') as f_out:
    for line in f_in:
        sample = json.loads(line)
        for filename in sample['files'].values():
            f_out.write(filename + '\n')
    # 添加元数据文件
    f_out.write('manifest.jsonl\n')
    f_out.write('dataset_stats.json\n')
EOF


# 方法：先复制元数据到源目录，打包后删除
cp manifest.jsonl dataset_stats.json "$SOURCE_DIR/"

if command -v pigz &> /dev/null; then
    echo "  使用 pigz 多线程压缩"
    tar -C "$SOURCE_DIR" -cf - -T files_to_pack.txt | pigz -p 8 > "${DATASET_NAME}.tar.gz"
else
    echo "  使用标准 gzip 压缩（可通过 'yum install pigz' 加速）"
    tar -czf "${DATASET_NAME}.tar.gz" -C "$SOURCE_DIR" -T files_to_pack.txt
fi

# 清理
rm "$SOURCE_DIR/manifest.jsonl" "$SOURCE_DIR/dataset_stats.json"


echo "  ✅ 打包完成: ${DATASET_NAME}.tar.gz"

# ==================== 步骤 3: 生成校验和 ====================
echo ""
echo "步骤 3/4: 生成 MD5 校验和"

md5sum "${DATASET_NAME}.tar.gz" > "${DATASET_NAME}.tar.gz.md5"
echo "  ✅ 校验和已保存: ${DATASET_NAME}.tar.gz.md5"

# ==================== 步骤 4: 验证完整性 ====================
echo ""
echo "步骤 4/4: 验证 tar 包完整性"

# 测试解压（只列出文件，不实际解压）
tar -tzf "${DATASET_NAME}.tar.gz" > tar_contents.txt
FILE_COUNT=$(wc -l < tar_contents.txt)
EXPECTED_COUNT=$((SAMPLE_COUNT * 5 + 2))  # 5 文件/样本 + 2 元数据

echo "  tar 包内文件数: $FILE_COUNT"
echo "  预期文件数: $EXPECTED_COUNT"

if [ "$FILE_COUNT" -eq "$EXPECTED_COUNT" ]; then
    echo "  ✅ 验证通过"
else
    echo "  ⚠️  文件数不匹配，请检查"
fi

# ==================== 最终报告 ====================
echo ""
echo "=========================================="
echo "打包完成！"
echo "=========================================="
ls -lh "${DATASET_NAME}.tar.gz"
echo ""
echo "下一步：上传到 ModelScope"
echo "----------------------------------------"
echo "1. 安装 modelscope 客户端（如果未安装）："
echo "   pip install modelscope"
echo ""
echo "2. 登录 ModelScope："
echo "   modelscope login"
echo ""
echo "3. 创建数据集（在 ModelScope 网页操作）："
echo "   https://modelscope.cn/datasets/create"
echo ""
echo "4. 上传文件："
echo "   modelscope dataset upload \\"
echo "     --dataset_name <你的用户名>/${DATASET_NAME} \\"
echo "     --local_path ${OUTPUT_DIR}/${DATASET_NAME}.tar.gz"
echo ""
echo "5. 或使用 Git LFS 方式上传："
echo "   git clone https://www.modelscope.cn/datasets/<你的用户名>/${DATASET_NAME}.git"
echo "   cd ${DATASET_NAME}"
echo "   git lfs install"
echo "   cp ${OUTPUT_DIR}/${DATASET_NAME}.tar.gz ."
echo "   git add ${DATASET_NAME}.tar.gz"
echo "   git commit -m 'Add dataset v1.0'"
echo "   git push"
echo "=========================================="

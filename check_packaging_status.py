#!/usr/bin/env python3
"""
检查打包脚本的执行状态
"""
import sys
import time
from pathlib import Path

def check_status():
    print("=" * 60)
    print("检查打包脚本状态")
    print("=" * 60)
    print()

    # 检查输出目录
    output_dir = Path("review_packages_with_data")
    if output_dir.exists():
        print(f"✅ 输出目录存在: {output_dir}")

        # 统计批次数
        batches = list(output_dir.glob("batch_*"))
        print(f"   已创建批次数: {len(batches)}")

        # 检查每个批次的大小
        for batch in sorted(batches):
            size_mb = sum(f.stat().st_size for f in batch.rglob('*') if f.is_file()) / (1024*1024)
            print(f"   {batch.name}: {size_mb:.1f} MB")

    else:
        print("⚠️  输出目录还未创建")

    print()

    # 检查临时目录
    temp_dir = Path("/tmp/review_extract")
    if temp_dir.exists():
        temp_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file()) / (1024*1024)
        temp_files = len(list(temp_dir.rglob('*')))
        print(f"临时目录: {temp_dir}")
        print(f"   文件数: {temp_files}")
        print(f"   大小: {temp_size:.1f} MB")
    else:
        print("⚠️  临时目录还未创建")

    print()
    print("=" * 60)
    print("建议:")
    print("=" * 60)
    print("1. 如果输出目录没有变化 > 5分钟，脚本可能卡住了")
    print("2. 可以 kill 掉进程，使用优化版本重新运行")
    print("3. 运行: kill $(ps aux | grep package_review_with_data.py | grep -v grep | awk '{print $2}')")

if __name__ == "__main__":
    check_status()

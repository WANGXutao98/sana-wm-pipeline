#!/usr/bin/env python3
"""
快速验证 run_stage3_cmcc.py 的文件定位逻辑

用法：
  python verify_file_location.py /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
"""

import sys
from pathlib import Path


def test_file_location(data_root: str):
    """测试文件定位逻辑"""
    data_root = Path(data_root)

    print("=" * 80)
    print("验证文件定位逻辑")
    print("=" * 80)
    print(f"数据根目录: {data_root}")
    print()

    # 1. 扫描解压目录
    print("[1/4] 扫描解压目录...")
    shard_dirs = list(data_root.glob("final_wds-*/wds-*/w*/shard-*"))

    # 过滤掉 .tar 和 .SUCCESS
    shard_dirs = [d for d in shard_dirs if d.is_dir() and d.suffix not in ['.tar', '.SUCCESS']]

    print(f"  找到 {len(shard_dirs)} 个解压目录")

    if len(shard_dirs) == 0:
        print("  ❌ 错误：没有找到解压目录！")
        print(f"  请检查路径: {data_root}")
        return False

    # 2. 显示前 3 个目录
    print(f"\n[2/4] 前 3 个解压目录示例:")
    for i, shard_dir in enumerate(shard_dirs[:3]):
        print(f"  {i+1}. {shard_dir}")

    # 3. 测试路径构造逻辑
    print(f"\n[3/4] 测试路径构造逻辑...")
    test_shard = shard_dirs[0]

    print(f"  测试 shard: {test_shard}")
    print(f"  shard.name: {test_shard.name}")
    print(f"  shard.parent: {test_shard.parent}")

    # 构造 fake_tar_path（修复后的逻辑）
    fake_tar_path = test_shard.parent / f"{test_shard.name}.tar"
    print(f"  fake_tar_path: {fake_tar_path}")

    # stage3_gpu.py 的逻辑
    extracted_dir = fake_tar_path.parent / fake_tar_path.stem
    print(f"  extracted_dir (stage3_gpu.py 计算): {extracted_dir}")

    # 检查是否匹配
    if extracted_dir == test_shard:
        print(f"  ✅ 路径匹配正确！")
    else:
        print(f"  ❌ 路径不匹配！")
        print(f"     期望: {test_shard}")
        print(f"     实际: {extracted_dir}")
        return False

    # 4. 测试样本文件
    print(f"\n[4/4] 测试样本文件定位...")
    mp4_files = list(test_shard.glob("*.mp4"))

    if len(mp4_files) == 0:
        print(f"  ❌ 错误：shard 目录中没有 .mp4 文件！")
        return False

    test_mp4 = mp4_files[0]
    sample_id = test_mp4.stem

    print(f"  测试样本 ID: {sample_id}")
    print(f"  视频文件: {test_mp4}")

    # 检查 caption 文件
    caption_file = test_mp4.with_suffix('.caption.txt')
    if caption_file.exists():
        print(f"  Caption 文件: {caption_file}")
        print(f"  ✅ Caption 文件存在")
    else:
        print(f"  ❌ Caption 文件不存在: {caption_file}")

    # stage3_gpu.py 的文件读取逻辑
    print(f"\n  stage3_gpu.py 文件读取逻辑:")
    mp4_path = extracted_dir / f"{sample_id}.mp4"
    cap_path = extracted_dir / f"{sample_id}.caption.txt"

    print(f"    mp4_path: {mp4_path}")
    print(f"    exists: {mp4_path.exists()}")
    print(f"    cap_path: {cap_path}")
    print(f"    exists: {cap_path.exists()}")

    if mp4_path.exists() and cap_path.exists():
        print(f"  ✅ 文件定位成功！")

        # 显示文件大小
        mp4_size = mp4_path.stat().st_size / 1024 / 1024
        cap_size = cap_path.stat().st_size
        print(f"\n  文件信息:")
        print(f"    视频大小: {mp4_size:.2f} MB")
        print(f"    Caption 大小: {cap_size} bytes")

        return True
    else:
        print(f"  ❌ 文件定位失败！")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_file_location.py <data_root>")
        print("示例: python verify_file_location.py /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output")
        sys.exit(1)

    data_root = sys.argv[1]

    success = test_file_location(data_root)

    print()
    print("=" * 80)
    if success:
        print("✅ 验证通过！run_stage3_cmcc.py 应该可以正确读取解压文件")
        print()
        print("你现在可以运行:")
        print("python scripts/run_stage3_cmcc.py \\")
        print("  --stage1-jsonl /root/work/david_work/qc_output_new/smoke_test_manifest.jsonl \\")
        print("  --data-root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \\")
        print("  --output-dir /root/work/david_work/sana_qc_pipeline/scripts/stage3_smoke_test \\")
        print("  --qwen-dir /root/work/david_work/models/Qwen3.5-9B \\")
        print("  --unimatch-dir /root/work/david_work/models/unimatch \\")
        print("  --worker-id 0 \\")
        print("  --total-workers 1 \\")
        print("  --table6-cfg src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml")
    else:
        print("❌ 验证失败！需要进一步检查数据结构")
        print()
        print("请在 CMCC 机器上执行以下命令，并将输出发给我：")
        print(f"find {data_root} -maxdepth 4 -type d | head -20")
        print(f"find {data_root}/final_wds-SpatialVID-hq/wds-SpatialVID-hq/w000 -maxdepth 1 -type d -name 'shard-*' | head -1 | xargs ls -lh | head -10")
    print("=" * 80)


if __name__ == "__main__":
    main()

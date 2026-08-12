#!/usr/bin/env python3
"""测试从解压目录 vs tar 读取的性能差异 - CMCC 环境

对比：
1. 从 tar 文件读取（当前实现）
2. 从解压目录读取（优化后）

预期：从目录读取快 10-100 倍
"""
import sys
import time
import tarfile
from pathlib import Path

# 测试配置（根据你的实际路径调整）
TAR_PATH = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/final_wds-sekai-game-drone/wds-sekai-game-drone/w003/shard-000003-000001.tar")
SAMPLE_ID = "sekai-game-drone_00400110001_0006450_0006750"


def read_from_tar(tar_path: Path, sample_id: str) -> tuple[bytes, bytes, float]:
    """从 tar 文件读取（当前方式）"""
    start = time.perf_counter()
    try:
        with tarfile.open(tar_path, "r") as tf:
            mp4_bytes = tf.extractfile(tf.getmember(f"{sample_id}.mp4")).read()
            cap_bytes = tf.extractfile(tf.getmember(f"{sample_id}.caption.txt")).read()
    except Exception as e:
        print(f"  ❌ 从 tar 读取失败: {e}")
        return None, None, -1
    elapsed = time.perf_counter() - start
    return mp4_bytes, cap_bytes, elapsed


def read_from_directory(tar_path: Path, sample_id: str) -> tuple[bytes, bytes, float]:
    """从解压目录读取（优化方式）"""
    extracted_dir = tar_path.with_suffix('')
    mp4_path = extracted_dir / f"{sample_id}.mp4"
    cap_path = extracted_dir / f"{sample_id}.caption.txt"

    start = time.perf_counter()
    try:
        mp4_bytes = mp4_path.read_bytes()
        cap_bytes = cap_path.read_bytes()
    except Exception as e:
        print(f"  ❌ 从目录读取失败: {e}")
        return None, None, -1
    elapsed = time.perf_counter() - start
    return mp4_bytes, cap_bytes, elapsed


def main():
    print("=" * 80)
    print("I/O 性能测试：tar vs 解压目录")
    print("=" * 80)

    # 检查文件是否存在
    print("\n[1/4] 环境检查")
    if not TAR_PATH.exists():
        print(f"  ❌ tar 文件不存在: {TAR_PATH}")
        print("  请修改脚本中的 TAR_PATH")
        return

    extracted_dir = TAR_PATH.with_suffix('')
    if not extracted_dir.exists():
        print(f"  ❌ 解压目录不存在: {extracted_dir}")
        print("  请先运行 tar 解压")
        return

    print(f"  ✅ tar 文件: {TAR_PATH}")
    print(f"  ✅ 解压目录: {extracted_dir}")
    print(f"  ✅ 测试样本: {SAMPLE_ID}")

    # 测试从 tar 读取
    print("\n[2/4] 测试从 tar 读取（当前实现）")
    mp4_tar, cap_tar, time_tar = read_from_tar(TAR_PATH, SAMPLE_ID)
    if mp4_tar is not None:
        print(f"  ✅ 读取成功")
        print(f"     mp4 大小: {len(mp4_tar) / 1024 / 1024:.2f} MB")
        print(f"     caption 大小: {len(cap_tar)} bytes")
        print(f"     耗时: {time_tar * 1000:.2f} ms")
    else:
        print(f"  ❌ 读取失败")
        return

    # 测试从目录读取
    print("\n[3/4] 测试从解压目录读取（优化实现）")
    mp4_dir, cap_dir, time_dir = read_from_directory(TAR_PATH, SAMPLE_ID)
    if mp4_dir is not None:
        print(f"  ✅ 读取成功")
        print(f"     mp4 大小: {len(mp4_dir) / 1024 / 1024:.2f} MB")
        print(f"     caption 大小: {len(cap_dir)} bytes")
        print(f"     耗时: {time_dir * 1000:.2f} ms")
        print(f"     加速比: {time_tar / time_dir:.1f}x")
    else:
        print(f"  ❌ 读取失败")
        return

    # 验证内容一致
    print("\n[4/4] 验证内容一致性")
    if mp4_tar == mp4_dir and cap_tar == cap_dir:
        print(f"  ✅ 内容完全一致")
    else:
        print(f"  ⚠️ 内容不一致")
        print(f"     mp4 一致: {mp4_tar == mp4_dir}")
        print(f"     caption 一致: {cap_tar == cap_dir}")

    # 总结
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print(f"\n从 tar 读取:      {time_tar * 1000:8.2f} ms  (基线)")
    print(f"从目录读取:      {time_dir * 1000:8.2f} ms  ({time_tar / time_dir:.1f}x 加速)")

    print(f"\n预估 Stage 3 性能提升：")
    if time_tar > 0 and time_dir > 0:
        # 假设 I/O 占 90% 时间（视频解码实际很快）
        current_io_time = 258 * 0.9  # 秒
        new_io_time = current_io_time * (time_dir / time_tar)
        compute_time = 258 * 0.1
        new_total = new_io_time + compute_time

        print(f"  当前单样本耗时：~258 秒")
        print(f"  预期单样本耗时：~{new_total:.0f} 秒")
        print(f"  总加速比：{258 / new_total:.1f}x")
        print(f"  139 样本处理时间：{139 * new_total / 60:.1f} 分钟（当前 10 小时）")

    print(f"\n结论：")
    if time_tar / time_dir > 10:
        print(f"  ✅ I/O 加速 {time_tar / time_dir:.1f}x，这是主要瓶颈！")
        print(f"  ✅ 使用解压目录可以解决性能问题")
    else:
        print(f"  ⚠️ I/O 加速仅 {time_tar / time_dir:.1f}x，可能还有其他瓶颈")


if __name__ == "__main__":
    main()

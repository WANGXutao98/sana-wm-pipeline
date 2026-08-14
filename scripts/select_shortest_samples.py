#!/usr/bin/env python3
"""从SpatialVID-hq tar包中选择帧数最短的N个样本

用法:
    python scripts/select_shortest_samples.py \\
        /path/to/SpatialVID-hq-000000.tar \\
        --num-samples 3 \\
        --output /tmp/selected_samples.txt
"""
import argparse
import tarfile
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="选择最短的N个样本")
    parser.add_argument("tar_path", type=Path, help="输入tar包路径")
    parser.add_argument("--num-samples", type=int, default=3, help="选择样本数量")
    parser.add_argument("--output", type=Path, required=True, help="输出文件路径")
    args = parser.parse_args()

    # 收集所有样本及其帧数
    samples = []

    print(f"扫描tar包: {args.tar_path}")
    with tarfile.open(args.tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".camera.npz")]
        print(f"找到 {len(members)} 个样本")

        for i, member in enumerate(members):
            if (i + 1) % 50 == 0:
                print(f"  处理进度: {i+1}/{len(members)}")

            # 提取sample_id
            sample_id = member.name.replace(".camera.npz", "")

            # 读取camera.npz获取帧数
            f = tf.extractfile(member)
            npz = np.load(f)
            n_frames = len(npz["frame_indices"])

            samples.append((sample_id, n_frames))

    # 按帧数排序
    samples.sort(key=lambda x: x[1])

    # 选择最短的N个
    selected = samples[:args.num_samples]

    print(f"\n选中的{args.num_samples}个最短样本:")
    for sample_id, n_frames in selected:
        print(f"  {sample_id}: {n_frames} 帧")

    # 写入输出文件
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample_id, n_frames in selected:
            f.write(f"{sample_id}\t{n_frames}\n")

    print(f"\n结果已写入: {args.output}")


if __name__ == "__main__":
    main()

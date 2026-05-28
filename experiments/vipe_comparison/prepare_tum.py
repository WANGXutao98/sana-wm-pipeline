#!/usr/bin/env python3
"""
下载 TUM RGB-D fr1/desk，生成 MP4 和 GT 对齐文件。

运行:
  python experiments/vipe_comparison/prepare_tum.py \
      --out experiments/vipe_comparison/data
"""
import argparse, os, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

URL = "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
ASSOC_URL = "https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py"

def download(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[download] {url}")
    subprocess.check_call(["wget", "-q", "-O", str(dest), url])

def read_file_list(filename: Path):
    """读取 TUM 格式时间戳文件，返回 {timestamp: [data...]} dict"""
    result = {}
    for line in filename.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace("\t", " ").split()
        if len(parts) >= 2:
            result[float(parts[0])] = parts[1:]
    return result

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """Python-3 兼容的时间戳关联（TUM benchmark 算法）"""
    first_keys = list(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    used_first = set()
    used_second = set()
    matches = []
    for diff, a, b in potential_matches:
        if a not in used_first and b not in used_second:
            used_first.add(a)
            used_second.add(b)
            matches.append((a, b))
    matches.sort()
    return matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/vipe_comparison/data")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 下载 TGZ
    tgz = out / "rgbd_dataset_freiburg1_desk.tgz"
    download(URL, tgz)

    # 2. 解压
    seq_dir = out / "rgbd_dataset_freiburg1_desk"
    if not seq_dir.exists():
        print("[extract] decompressing...")
        subprocess.check_call(["tar", "xzf", str(tgz), "-C", str(out), "--no-same-owner"])
    else:
        print("[skip] already extracted")

    # 3. 下载 associate.py（仅备用，实际使用内嵌 Python-3 兼容版）
    assoc_py = out / "associate.py"
    download(ASSOC_URL, assoc_py)

    # 4. 生成 associations.txt（RGB 时间戳 ↔ GT 时间戳，容忍 0.02s）
    assoc_file = seq_dir / "associations.txt"
    if not assoc_file.exists():
        print("[assoc] generating associations.txt...")
        rgb_list = read_file_list(seq_dir / "rgb.txt")
        gt_list  = read_file_list(seq_dir / "groundtruth.txt")
        matches  = associate(rgb_list, gt_list, offset=0.0, max_difference=0.02)
        lines_out = []
        for a, b in matches:
            rgb_data = " ".join(rgb_list[a])
            gt_data  = " ".join(gt_list[b])
            lines_out.append(f"{a:.6f} {rgb_data} {b:.6f} {gt_data}")
        assoc_file.write_text("\n".join(lines_out) + "\n")
        print(f"[assoc] {assoc_file}: {len(lines_out)} matched pairs")
    else:
        print("[skip] associations.txt exists")

    # 5. 生成 MP4（按 associations.txt 中 RGB 帧顺序）
    mp4_path = seq_dir / "video.mp4"
    if not mp4_path.exists():
        print("[mp4] generating video.mp4...")
        lines = [l for l in assoc_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
        frame_paths = []
        gt_poses_lines = []
        for line in lines:
            parts = line.split()
            # associate.py 输出: ts_rgb rgb_path ts_gt tx ty tz qx qy qz qw
            rgb_file = seq_dir / parts[1]
            frame_paths.append(str(rgb_file))
            gt_poses_lines.append(" ".join(parts[2:]))  # ts tx ty tz qx qy qz qw

        # 保存 GT 对齐序列（按帧序号排列）
        gt_aligned = seq_dir / "gt_aligned.txt"
        gt_aligned.write_text("# timestamp tx ty tz qx qy qz qw\n" + "\n".join(gt_poses_lines))
        print(f"[gt] gt_aligned.txt: {len(gt_poses_lines)} poses")

        # 读取第一帧获取分辨率
        sample = cv2.imread(frame_paths[0])
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, 30.0, (w, h))
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                print(f"[warn] cannot read {fp}")
                continue
            writer.write(frame)
        writer.release()
        print(f"[mp4] {mp4_path}: {len(frame_paths)} frames @ {w}×{h} 30fps")
    else:
        print("[skip] video.mp4 exists")

    print("\n[done] Data ready:")
    print(f"  MP4:  {mp4_path}")
    print(f"  GT:   {seq_dir}/gt_aligned.txt")
    print(f"  Stem: {mp4_path.stem}")

if __name__ == "__main__":
    main()

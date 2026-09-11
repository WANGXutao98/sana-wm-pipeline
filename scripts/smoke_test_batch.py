#!/usr/bin/env python
"""单进程批量处理冒烟测试（解决@lru_cache跨进程失效问题）

用法：
    python scripts/smoke_test_batch.py \
        --samples /path/to/selected_samples.txt \
        --extract-dir /path/to/raw_samples \
        --output-dir /path/to/smoke_result
"""
import argparse
import json
import sys
import tarfile
import io
from pathlib import Path
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from sana_wm_pipeline.stage02_pose.mode_default import run_default


def pack_shard(scene_id: str, artifact_json: Path, norm_video: Path, shard_path: Path):
    """打包WebDataset shard"""
    art = json.loads(artifact_json.read_text())
    poses = np.array(art["poses_c2w"], np.float32)
    intr = np.array(art["intrinsics"], np.float32)
    scale = np.array(art["scale_per_frame"], np.float32)

    cap = f"A video from SpatialVID-hq dataset with {len(poses)} frames."

    def add_npy(tf, key, arr):
        b = io.BytesIO()
        np.save(b, arr)
        raw = b.getvalue()
        ti = tarfile.TarInfo(f"{scene_id}.{key}")
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))

    with tarfile.open(shard_path, "w") as tf:
        # video
        vb = norm_video.read_bytes()
        ti = tarfile.TarInfo(f"{scene_id}.mp4")
        ti.size = len(vb)
        tf.addfile(ti, io.BytesIO(vb))

        # arrays
        add_npy(tf, "poses_c2w.npy", poses)
        add_npy(tf, "intrinsics.npy", intr)
        add_npy(tf, "scale.npy", scale)

        # caption
        cb = cap.encode()
        ti = tarfile.TarInfo(f"{scene_id}.caption.txt")
        ti.size = len(cb)
        tf.addfile(ti, io.BytesIO(cb))

        # meta
        meta = json.dumps({
            "scene_id": scene_id,
            "T": len(poses),
            "mode": "default",
            "dataset": "SpatialVID-hq",
            "group": "wds-SpatialVID-hq"
        }).encode()
        ti = tarfile.TarInfo(f"{scene_id}.meta.json")
        ti.size = len(meta)
        tf.addfile(ti, io.BytesIO(meta))

    print(f"  Shard written: {shard_path.name} ({len(poses)} frames)")


def process_sample(sample_id: str, n_frames: int, extract_dir: Path, output_dir: Path) -> bool:
    """处理单个样本，返回是否成功"""
    print("=" * 60)
    print(f"样本: {sample_id} ({n_frames} 帧)")
    print("=" * 60)

    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    video_path = extract_dir / f"{sample_id}.mp4"
    norm_video = sample_dir / "normalized.mp4"
    vipe_work = sample_dir / "vipe_work_default"
    artifact_json = sample_dir / "pose_artifact_default.json"

    try:
        # Stage 1: 归一化
        print("--- Stage 1: 归一化 ---")
        info = normalize_video(video_path, norm_video)
        print(f"  Normalized: {info.n_frames} frames @ {info.fps}fps ({info.width}x{info.height})")

        # Stage 2: VIPE SLAM
        print("--- Stage 2: VIPE SLAM ---")
        art = run_default(norm_video, vipe_work)
        print(f"  Poses {art.poses_c2w.shape}  Intr {art.intrinsics.shape}")

        artifact_json.write_text(json.dumps({
            "poses_c2w": art.poses_c2w.tolist(),
            "intrinsics": art.intrinsics.tolist(),
            "scale_per_frame": art.scale_per_frame.tolist(),
        }))
        print("  Stage 2 SUCCESS")

        # Stage 3: 打包shard
        print("--- Stage 3: 打包 shard ---")
        shard_path = sample_dir / f"{sample_id}.tar"
        pack_shard(sample_id, artifact_json, norm_video, shard_path)

        print(f"✅ 样本处理成功: {sample_id}\n")
        return True

    except Exception as e:
        print(f"❌ 样本处理失败: {sample_id}")
        print(f"   错误: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="单进程批量冒烟测试")
    parser.add_argument("--samples", required=True, help="selected_samples.txt路径")
    parser.add_argument("--extract-dir", required=True, help="解包后的样本目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()

    samples_file = Path(args.samples)
    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)

    # 读取样本列表
    samples = []
    with samples_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"⚠️  跳过格式错误的行: {line}")
                continue
            sample_id, n_frames = parts[0], int(parts[1])
            samples.append((sample_id, n_frames))

    print(f"共 {len(samples)} 个样本待处理\n")

    # 批量处理（单进程，模型只加载一次）
    success_count = 0
    fail_count = 0

    for sample_id, n_frames in samples:
        if process_sample(sample_id, n_frames, extract_dir, output_dir):
            success_count += 1
        else:
            fail_count += 1

    # 汇总结果
    print("=" * 60)
    print("冒烟测试完成")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print()

    if success_count == len(samples):
        print("✅ 全部样本通过！")
        sys.exit(0)
    elif success_count > 0:
        print("⚠️  部分样本失败，需要调查")
        sys.exit(1)
    else:
        print("❌ 全部失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

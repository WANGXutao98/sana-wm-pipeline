#!/usr/bin/env python3
"""批量处理pass_videos（无GT标注）- 单进程模式确保@lru_cache生效"""
import argparse
from pathlib import Path
import sys
import time
import json
import tarfile
import io
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ponytail: 复用原脚本的导入
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from sana_wm_pipeline.stage02_pose.mode_default import run_default


# ponytail: 完整复制原smoke_test_batch.py的打包函数（100%一致）
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="样本列表文件(sample_id<TAB>n_frames)")
    parser.add_argument("--video-dir", required=True, help="视频目录")
    parser.add_argument("--output-dir", required=True, help="输出根目录")
    args = parser.parse_args()

    samples_file = Path(args.samples)
    video_dir = Path(args.video_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # 读取样本列表
    samples = []
    with open(samples_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            sample_id = parts[0]
            n_frames = int(parts[1]) if len(parts) > 1 else 0
            samples.append((sample_id, n_frames))

    print(f"[批量处理] 共 {len(samples)} 个样本")
    print()

    success_count = 0
    skipped_count = 0
    failed_samples = []
    start_time = time.time()

    for idx, (sample_id, n_frames) in enumerate(samples, 1):
        print(f"{'='*80}")
        print(f"[{idx}/{len(samples)}] 处理样本: {sample_id} ({n_frames}帧)")
        print(f"{'='*80}")

        try:
            # ponytail: 断点续传 - 检查已有输出
            sample_out = output_base / sample_id
            shard_path = sample_out / f"{sample_id}.tar"
            pose_artifact = sample_out / "pose_artifact_default.json"

            if shard_path.exists() and pose_artifact.exists():
                print(f"✅ 已完成，跳过: {sample_id}")
                success_count += 1
                skipped_count += 1
                continue

            video_path = video_dir / f"{sample_id}.mp4"
            if not video_path.exists():
                print(f"⚠️  视频不存在，跳过: {video_path}")
                failed_samples.append((sample_id, "视频不存在"))
                continue

            sample_out.mkdir(parents=True, exist_ok=True)

            # Stage 1: 归一化
            print(f"\n[Stage 1] 归一化视频...")
            normalized_mp4 = sample_out / "normalized.mp4"
            vipe_work = sample_out / "vipe_work_default"
            # ponytail: 复用原脚本的调用方式
            info = normalize_video(video_path, normalized_mp4)
            print(f"  Normalized: {info.n_frames} frames @ {info.fps}fps ({info.width}x{info.height})")

            # Stage 2: VIPE SLAM
            print(f"\n[Stage 2] VIPE SLAM...")
            # ponytail: 完全复制原脚本的调用方式
            art = run_default(normalized_mp4, vipe_work)
            print(f"  Poses {art.poses_c2w.shape}  Intr {art.intrinsics.shape}")

            # 序列化artifact到JSON（原脚本第100-104行）
            pose_artifact.write_text(json.dumps({
                "poses_c2w": art.poses_c2w.tolist(),
                "intrinsics": art.intrinsics.tolist(),
                "scale_per_frame": art.scale_per_frame.tolist(),
            }))
            print(f"  Stage 2 SUCCESS")

            # Stage 3: 打包
            print(f"\n[Stage 3] 打包 shard...")
            pack_shard(sample_id, pose_artifact, normalized_mp4, shard_path)

            success_count += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            eta = avg_time * (len(samples) - idx)
            print(f"\n✅ [{idx}/{len(samples)}] 成功 | 累计耗时: {elapsed/60:.1f}分钟 | ETA: {eta/60:.1f}分钟")

        except Exception as e:
            print(f"\n❌ 处理失败: {sample_id}")
            print(f"   错误: {e}")
            failed_samples.append((sample_id, str(e)))
            continue

    # 总结
    print(f"\n{'='*80}")
    print(f"批量处理完成")
    print(f"{'='*80}")
    print(f"✅ 成功: {success_count}/{len(samples)} (其中跳过: {skipped_count})")
    print(f"❌ 失败: {len(failed_samples)}/{len(samples)}")
    print(f"⏱️  总耗时: {(time.time() - start_time)/60:.1f}分钟")

    if failed_samples:
        print(f"\n失败样本列表:")
        for sample_id, reason in failed_samples:
            print(f"  - {sample_id}: {reason}")

    sys.exit(0 if success_count == len(samples) else 1)


if __name__ == "__main__":
    main()

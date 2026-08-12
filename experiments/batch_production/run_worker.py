#!/usr/bin/env python3
"""
单 GPU Worker：处理分配到的输入 shard 列表，产出 WebDataset shard。

用法:
  CUDA_VISIBLE_DEVICES=0 python run_worker.py \
    --group wds-sekai-real-walking-hq \
    --data-root /root/work/externalstorage/.../jdvbbfb-v3-full \
    --out-base  /root/work/filestorage/jdvbbfb_output \
    --worker-id 0 \
    --shard-indices 0,8,16,24,32,40,48,56,64,72,80,88 \
    --samples-per-shard 200
"""
from __future__ import annotations

import argparse, json, shutil, sys, time, traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--group",             required=True)
    p.add_argument("--data-root",         required=True, type=Path)
    p.add_argument("--out-base",          required=True, type=Path)
    p.add_argument("--worker-id",         required=True, type=int)
    p.add_argument("--shard-indices",     required=True,
                   help="逗号分隔 shard 下标，如 '0,8,16'")
    p.add_argument("--samples-per-shard", type=int, default=200)
    return p.parse_args()


def _shard_basename(group: str, shard_idx: int) -> str:
    prefix = group[len("wds-"):] if group.startswith("wds-") else group
    return f"{prefix}-{shard_idx:06d}.tar"


def _load_captions(index_path: Path) -> dict[str, str]:
    """启动时一次性加载 index.jsonl → {key: caption}，消除每样本磁盘读取。"""
    if not index_path.exists():
        print(f"[WARN] index.jsonl not found: {index_path}")
        return {}
    caps: dict[str, str] = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key", "")
            text = (rec.get("manifest", {}).get("prompt") or {}).get("text", "") or ""
            if key:
                caps[key] = text
    print(f"[index] 加载 {len(caps)} 条 caption")
    return caps


def process_input_shard(
    shard_path: Path,
    shard_idx: int,
    group: str,
    captions: dict[str, str],
    tmp_dir: Path,
    worker_out: Path,
    progress_dir: Path,
    samples_per_shard: int,
    shard_writer_cls=None,
) -> tuple[int, int]:
    import shard_io

    if shard_io.shard_is_complete(worker_out, progress_dir, shard_idx):
        print(f"[SKIP] shard {shard_idx:06d} 已完成（.done + tar 均在），跳过")
        return 0, 0

    if shard_writer_cls is None:
        from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter
        shard_writer_cls = ShardWriter

    from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import iter_tar_samples
    from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
    from sana_wm_pipeline.stage02_pose.mode_default import run_default
    from sana_wm_pipeline.stage06_pack.schema import Sample

    STUB = "A real-world scene captured by a moving camera."
    n_ok = n_fail = 0
    t0_shard = time.time()

    # 该输入 shard 独占一组确定性命名的输出 tar：shard-{idx:06d}-{part:06d}.tar
    prefix = shard_io.output_shard_prefix(shard_idx)
    with shard_writer_cls(worker_out, samples_per_shard=samples_per_shard,
                          prefix=prefix, strict_frames=False) as writer:
        with open(shard_path, "rb") as fobj:
            for key, mp4_bytes, camera_bytes in iter_tar_samples(fobj, limit=None):
                sample_tmp = tmp_dir / key
                sample_tmp.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                try:
                    raw_video = sample_tmp / "video.mp4"
                    raw_video.write_bytes(mp4_bytes)
                    norm_video = sample_tmp / "normalized.mp4"
                    info = normalize_video(raw_video, norm_video)
                    raw_video.unlink()
                    vipe_work = sample_tmp / "vipe_work"
                    art = run_default(norm_video, vipe_work)
                    caption = captions.get(key, "").strip() or STUB
                    sample = Sample(
                        sample_id=key,
                        video_path=str(norm_video),
                        poses_c2w=art.poses_c2w,
                        intrinsics_NVD=art.intrinsics,
                        scale_per_frame=art.scale_per_frame,
                        caption=caption,
                        meta={
                            "scene_id": key,
                            "T": int(art.poses_c2w.shape[0]),
                            "mode": "default",
                            "dataset": "jdvbbfb-v3-full",
                            "group": group,
                            "source_shard": shard_path.name,
                        },
                    )
                    writer.write(sample)
                    n_ok += 1
                    print(f"  [OK]   {key}  T={info.n_frames}  {time.time()-t0:.0f}s")
                except Exception as exc:
                    n_fail += 1
                    print(f"  [FAIL] {key}: {exc}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                finally:
                    shutil.rmtree(sample_tmp, ignore_errors=True)

    shard_io.mark_shard_done(progress_dir, shard_idx, n_ok, n_fail,
                             round(time.time() - t0_shard, 1))
    print(f"[shard {shard_idx:06d}] DONE  ok={n_ok}  fail={n_fail}  "
          f"{time.time()-t0_shard:.0f}s")
    return n_ok, n_fail


def main() -> None:
    args = parse_args()

    # 每个 worker 独立输出目录，无需文件锁（w000, w001, ...）
    worker_out  = args.out_base / args.group / f"w{args.worker_id:03d}"
    progress_dir = args.out_base / args.group / "progress"
    worker_out.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    # /tmp 本地 NVMe 临时目录，重启丢失无所谓
    tmp_dir = Path(f"/tmp/sana_wm_w{args.worker_id:03d}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.data_root / args.group / "index.jsonl"
    captions = _load_captions(index_path)

    shard_indices = [int(x) for x in args.shard_indices.split(",") if x.strip()]
    shard_dir = args.data_root / args.group / "shards"
    print(f"[worker {args.worker_id}] group={args.group}  "
          f"shards={shard_indices}  out={worker_out}")

    total_ok = total_fail = 0
    for shard_idx in shard_indices:
        shard_path = shard_dir / _shard_basename(args.group, shard_idx)
        if not shard_path.exists():
            print(f"[WARN] shard not found: {shard_path}，跳过")
            continue
        n_ok, n_fail = process_input_shard(
            shard_path, shard_idx, args.group, captions, tmp_dir,
            worker_out, progress_dir, args.samples_per_shard,
        )
        total_ok += n_ok
        total_fail += n_fail

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n[worker {args.worker_id}] 完成  ok={total_ok}  fail={total_fail}")
    sys.exit(1 if total_fail > 0 and total_ok == 0 else 0)


if __name__ == "__main__":
    main()

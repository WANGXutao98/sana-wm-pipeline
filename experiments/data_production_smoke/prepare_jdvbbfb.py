#!/usr/bin/env python3
"""Extract samples from one jdvbbfb-v3-full shard into scene dirs.

Each sample → {out_base}/{scene_id}/ with video.mp4 + gt_poses.npy +
gt_intrinsics.npy + orig_fps.txt + caption.txt (+ vipe_ref_poses.npy).
Default mode never uses the GT; gt_poses.npy is for verify_and_eval ATE only.

Two input modes (mutually exclusive):
  A) --local-root <DIR>   read {DIR}/{group}/shards/<prefix>-NNNNNN.tar locally
                          (CMCC production: data already on externalstorage,
                          no HF token / no network)
  B) --repo <REPO>        stream the shard over HTTPS from Hugging Face
                          (H100 dev/smoke; needs a valid HF token)

Usage (CMCC, local):
  python prepare_jdvbbfb.py \\
    --local-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full \\
    --group wds-DL3DV-ALL-2K --shard-idx 0 --sample-limit 0 \\
    --out-base /root/work/<userspace>/jdvbbfb_out

Usage (H100, HF stream):
  python prepare_jdvbbfb.py \\
    --repo junchaoh-cs/jdvbbfb-v3-full \\
    --group wds-DL3DV-ALL-2K --shard-idx 0 --sample-limit 1 \\
    --out-base /mnt/afs/davidwang/workspace/data/jdvbbfb_smoke

--sample-limit 0 (or negative) means ALL samples in the shard (batch production).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import (
    iter_tar_samples, read_index, write_scene_dir,
)


def _shard_basename(group: str, shard_idx: int) -> str:
    # "wds-DL3DV-ALL-2K" -> "DL3DV-ALL-2K-000000.tar"
    prefix = group[len("wds-"):] if group.startswith("wds-") else group
    return f"{prefix}-{shard_idx:06d}.tar"


def _open_local(local_root: Path, group: str, shard_idx: int):
    """Return (index_path, fileobj) for a local shard."""
    root = Path(local_root) / group
    index_path = root / "index.jsonl"
    shard = root / "shards" / _shard_basename(group, shard_idx)
    if not shard.exists():
        raise SystemExit(f"local shard not found: {shard}")
    return index_path, open(shard, "rb")


def _open_remote(repo: str, group: str, shard_idx: int):
    """Return (index_path, fileobj) for an HF-streamed shard."""
    import requests
    from huggingface_hub import get_token, hf_hub_download
    index_path = Path(hf_hub_download(repo, f"{group}/index.jsonl",
                                      repo_type="dataset"))
    shard_name = f"{group}/shards/{_shard_basename(group, shard_idx)}"
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard_name}"
    tok = get_token()
    if not tok:
        raise SystemExit("No HF token. Run: "
                         "HF_HOME=/mnt/afs/davidwang/cache/huggingface hf auth login")
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                     stream=True, timeout=300)
    r.raise_for_status()
    return index_path, r.raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-root", type=Path,
                     help="local dataset root (CMCC). Reads {root}/{group}/...")
    src.add_argument("--repo", help="HF repo id (stream over HTTPS)")
    ap.add_argument("--group", required=True, help="e.g. wds-DL3DV-ALL-2K")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--sample-limit", type=int, default=1,
                    help="0 or negative = all samples in shard")
    ap.add_argument("--out-base", required=True, type=Path)
    args = ap.parse_args()

    limit = None if args.sample_limit <= 0 else args.sample_limit

    if args.local_root is not None:
        index_path, fobj = _open_local(args.local_root, args.group, args.shard_idx)
        print(f"[local] {args.local_root}/{args.group} shard {args.shard_idx}")
    else:
        index_path, fobj = _open_remote(args.repo, args.group, args.shard_idx)
        print(f"[stream] {args.repo} {args.group} shard {args.shard_idx}")

    refs = read_index(index_path)
    cap_by_key = {r.key: r.caption for r in refs}
    print(f"[index] {len(refs)} samples in {args.group}")

    n = 0
    try:
        for key, mp4_bytes, camera_bytes in iter_tar_samples(fobj, limit=limit):
            scene = write_scene_dir(args.out_base, key, mp4_bytes,
                                    camera_bytes, cap_by_key.get(key, ""))
            n += 1
            print(f"  [{n}] {key}  →  {scene}  "
                  f"(video {len(mp4_bytes)/1e6:.1f} MB, "
                  f"caption {'yes' if cap_by_key.get(key) else 'stub'})")
    finally:
        fobj.close()
    print(f"[done] wrote {n} scene dir(s) under {args.out_base}")


if __name__ == "__main__":
    main()

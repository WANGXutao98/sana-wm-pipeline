#!/usr/bin/env python3
"""SANA-WM inference and evaluation on DL3DV smoke test shards.

Steps:
  1. Extract sample data from WebDataset shard (.tar)
  2. Prepare inputs: first_frame.png, camera poses, intrinsics (reshaped)
  3. Call inference_sana_wm.py via subprocess
  4. Compare generated video vs DL3DV GT video (PSNR/SSIM per frame)
  5. Generate side-by-side comparison video

Usage:
  python run_sana_wm_inference.py \
    --shards-dir /path/to/shards \
    --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke \
    --sana-dir /mnt/afs/davidwang/workspace/Sana \
    --output-dir /path/to/results \
    [--sample-limit 1]
    [--model-path /mnt/afs/davidwang/models/sana_wm]
"""

import argparse
import json
import logging
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: extract one sample from a tar shard
# ---------------------------------------------------------------------------

def extract_shard_sample(tar_path: Path, sample_id: str, out_dir: Path) -> dict:
    """Extract one sample from tar to out_dir.

    Returns dict of {suffix: path} for mp4, poses_c2w.npy, intrinsics.npy,
    caption.txt, meta.json, scale.npy (if present).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        members = t.getmembers()
        for m in members:
            if m.name.startswith(f"{sample_id}."):
                t.extract(m, out_dir)

    result = {}
    for suffix in ["mp4", "poses_c2w.npy", "intrinsics.npy", "scale.npy",
                   "caption.txt", "meta.json"]:
        p = out_dir / f"{sample_id}.{suffix}"
        if p.exists():
            result[suffix] = p
    return result


# ---------------------------------------------------------------------------
# Step 2: prepare inference inputs
# ---------------------------------------------------------------------------

def prepare_inputs(sample_dir: Path, sample_id: str, files: dict) -> dict:
    """Prepare inference inputs: first_frame.png + intrinsics_flat.npy.

    intrinsics reshape: (T, 1, 4) -> mean over T -> (4,) = [fx, fy, cx, cy]
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        log.warning("imageio not available — cannot extract first frame.")
        raise

    # Extract first frame from GT video
    mp4_path = files["mp4"]
    first_frame_path = sample_dir / "first_frame.png"
    frames = iio.imread(mp4_path, index=None)  # (T, H, W, 3)
    iio.imwrite(first_frame_path, frames[0])
    log.info("  first_frame saved: %s  shape=%s", first_frame_path, frames[0].shape)

    # Reshape intrinsics: (T, 1, 4) -> (4,)
    intr_arr = np.load(files["intrinsics.npy"])  # (T, 1, 4) expected
    if intr_arr.ndim == 3:
        intr_mean = intr_arr[:, 0, :].mean(axis=0)  # (4,)
    elif intr_arr.ndim == 2:
        intr_mean = intr_arr.mean(axis=0)            # (4,) fallback
    else:
        intr_mean = intr_arr.flatten()[:4]           # last-resort
    intr_flat_path = sample_dir / "intrinsics_flat.npy"
    np.save(intr_flat_path, intr_mean.astype(np.float32))
    log.info("  intrinsics_flat: %s", intr_mean)

    num_frames = int(np.load(files["poses_c2w.npy"]).shape[0])
    log.info("  num_frames from poses: %d", num_frames)

    return {
        "first_frame": first_frame_path,
        "poses": files["poses_c2w.npy"],
        "intrinsics_flat": intr_flat_path,
        "caption": files["caption.txt"],
        "num_frames": num_frames,
        "gt_mp4": files["mp4"],
    }


# ---------------------------------------------------------------------------
# Step 3: run inference
# ---------------------------------------------------------------------------

def run_inference(
    inputs: dict,
    sample_id: str,
    sana_dir: Path,
    output_dir: Path,
    model_path: str | None,
) -> Path:
    """Call inference_sana_wm.py via subprocess.

    Returns the expected generated video path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    script = sana_dir / "inference_video_scripts" / "inference_sana_wm.py"
    local_config = Path("/mnt/afs/davidwang/models/SANA-WM_bidirectional/config.yaml")

    cmd = [
        sys.executable,
        str(script),
        "--image",       str(inputs["first_frame"]),
        "--prompt",      str(inputs["caption"]),
        "--camera",      str(inputs["poses"]),
        "--intrinsics",  str(inputs["intrinsics_flat"]),
        "--num_frames",  str(inputs["num_frames"]),
        "--fps",         "16",
        "--output_dir",  str(output_dir),
        "--name",        sample_id,
        "--step",        "60",
    ]
    if local_config.exists():
        cmd += [
            "--config",            str(local_config),
            "--refiner_root",      "/mnt/afs/davidwang/models/SANA-WM_bidirectional/refiner",
            "--refiner_gemma_root","/mnt/afs/davidwang/models/SANA-WM_bidirectional/refiner/text_encoder",
            "--offload_vae",
            "--offload_refiner",
        ]
    if model_path:
        cmd += ["--model_path", model_path]

    import os as _os
    env = _os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(sana_dir) + (":" + existing_pp if existing_pp else "")
    env["DISABLE_XFORMERS"] = "1"

    log.info("Running inference for sample %s", sample_id)
    log.info("  cmd: %s", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

    gen_mp4 = output_dir / f"{sample_id}_generated.mp4"
    if not gen_mp4.exists():
        raise FileNotFoundError(f"Expected generated video not found: {gen_mp4}")
    return gen_mp4


# ---------------------------------------------------------------------------
# Step 4 & 5: evaluate vs GT and produce side-by-side video
# ---------------------------------------------------------------------------

def evaluate_vs_gt(
    gen_mp4: Path,
    gt_mp4: Path,
    output_dir: Path,
    sample_id: str,
) -> dict:
    """Compute per-frame PSNR/SSIM and generate side-by-side video.

    inference_sana_wm.py outputs 704x1280; GT video is resized to match.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        log.warning("imageio not available — skipping evaluation.")
        return {"sample_id": sample_id, "error": "imageio not available"}

    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        has_skimage = True
    except ImportError:
        log.warning("scikit-image not available — skipping PSNR/SSIM computation.")
        has_skimage = False

    from PIL import Image

    gen_frames = iio.imread(gen_mp4, index=None)   # (T, H, W, 3)
    gt_frames  = iio.imread(gt_mp4,  index=None)   # (T', H', W', 3)
    T = min(len(gen_frames), len(gt_frames))
    log.info("  gen_frames=%d  gt_frames=%d  -> using T=%d", len(gen_frames), len(gt_frames), T)

    # Resize GT to match generated resolution (704 x 1280 = H x W)
    H, W = gen_frames.shape[1], gen_frames.shape[2]
    gt_resized = np.stack([
        np.array(Image.fromarray(gt_frames[i]).resize((W, H), Image.LANCZOS))
        for i in range(T)
    ])
    gen_trim = gen_frames[:T]

    metrics: dict = {
        "sample_id": sample_id,
        "num_frames": T,
        "gen_resolution": [H, W],
    }

    if has_skimage:
        psnrs, ssims = [], []
        for i in range(T):
            p = psnr(gt_resized[i], gen_trim[i], data_range=255)
            s = ssim(gt_resized[i], gen_trim[i], channel_axis=2, data_range=255)
            psnrs.append(float(p))
            ssims.append(float(s))

        metrics.update({
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std":  float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std":  float(np.std(ssims)),
        })
        log.info("  PSNR=%.2f±%.2f  SSIM=%.4f±%.4f",
                 metrics["psnr_mean"], metrics["psnr_std"],
                 metrics["ssim_mean"], metrics["ssim_std"])

    # Side-by-side video: horizontally concatenate GT (resized) and generated
    sbs_frames = np.concatenate([gt_resized, gen_trim], axis=2)  # (T, H, 2*W, 3)
    sbs_path = output_dir / f"{sample_id}_sbs.mp4"
    try:
        iio.imwrite(str(sbs_path), sbs_frames, fps=16, codec="libx264")
        metrics["sbs_video"] = str(sbs_path)
        log.info("  side-by-side video: %s", sbs_path)
    except Exception as e:
        log.warning("  side-by-side video write failed: %s", e)
        metrics["sbs_video_error"] = str(e)

    return metrics


# ---------------------------------------------------------------------------
# Helper: list sample ids in a tar
# ---------------------------------------------------------------------------

def list_sample_ids(tar_path: Path) -> list[str]:
    """Return unique sample IDs (stem before first dot) in the tar."""
    ids = set()
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            name = Path(m.name).name  # strip any sub-directory prefix
            parts = name.split(".", 1)
            if len(parts) >= 2:
                ids.add(parts[0])
    return sorted(ids)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SANA-WM inference + evaluation on DL3DV smoke test shards."
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        required=True,
        help="Directory containing WebDataset .tar shards.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("/mnt/afs/davidwang/workspace/data/dl3dv_smoke"),
        help="DL3DV raw scene directory (for GT videos if needed). "
             "Currently the GT video is taken from the shard .mp4.",
    )
    parser.add_argument(
        "--sana-dir",
        type=Path,
        default=Path("/mnt/afs/davidwang/workspace/Sana"),
        help="Root directory of the Sana repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated videos and evaluation results.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local path to SANA-WM model weights. "
             "If omitted, HF auto-download is used.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Maximum number of samples to process (0 = all).",
    )
    args = parser.parse_args()

    # Check for default model path
    model_path = args.model_path
    default_model = Path("/mnt/afs/davidwang/models/SANA-WM_bidirectional/dit/sana_wm_1600m_720p.safetensors")
    if model_path is None and default_model.exists():
        model_path = str(default_model)
        log.info("Found local model weights at %s", model_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(args.shards_dir.glob("*.tar"))
    if not tar_files:
        log.error("No .tar shards found in %s", args.shards_dir)
        sys.exit(1)
    log.info("Found %d shard(s) in %s", len(tar_files), args.shards_dir)

    all_metrics: list[dict] = []
    processed = 0

    for tar_path in tar_files:
        log.info("=== Processing shard: %s ===", tar_path.name)
        try:
            sample_ids = list_sample_ids(tar_path)
        except Exception as e:
            log.error("  Failed to list samples in %s: %s", tar_path, e)
            continue
        log.info("  %d sample(s) in shard: %s", len(sample_ids), sample_ids[:5])

        for sample_id in sample_ids:
            if args.sample_limit > 0 and processed >= args.sample_limit:
                log.info("Reached sample limit (%d). Stopping.", args.sample_limit)
                break

            log.info("--- Sample: %s ---", sample_id)
            sample_out = args.output_dir / sample_id
            sample_out.mkdir(parents=True, exist_ok=True)

            # --- step 1: extract ---
            try:
                files = extract_shard_sample(tar_path, sample_id, sample_out)
                log.info("  Extracted files: %s", list(files.keys()))
                if "mp4" not in files or "poses_c2w.npy" not in files:
                    log.warning("  Missing required files (mp4/poses_c2w.npy) — skipping.")
                    all_metrics.append({"sample_id": sample_id, "error": "missing required files"})
                    continue
            except Exception:
                log.error("  extract_shard_sample failed:\n%s", traceback.format_exc())
                all_metrics.append({"sample_id": sample_id, "error": "extract failed"})
                continue

            # --- step 2: prepare inputs ---
            try:
                inputs = prepare_inputs(sample_out, sample_id, files)
            except Exception:
                log.error("  prepare_inputs failed:\n%s", traceback.format_exc())
                all_metrics.append({"sample_id": sample_id, "error": "prepare_inputs failed"})
                continue

            # --- step 3: run inference ---
            gen_mp4 = None
            try:
                gen_mp4 = run_inference(inputs, sample_id, args.sana_dir, sample_out, model_path)
                log.info("  Generated video: %s", gen_mp4)
            except Exception:
                log.error("  run_inference failed:\n%s", traceback.format_exc())
                all_metrics.append({"sample_id": sample_id, "error": "inference failed"})
                continue

            # --- step 4 & 5: evaluate ---
            try:
                metrics = evaluate_vs_gt(gen_mp4, inputs["gt_mp4"], sample_out, sample_id)
                all_metrics.append(metrics)
                log.info("  Metrics: %s", {k: v for k, v in metrics.items() if k != "sbs_video"})
            except Exception:
                log.error("  evaluate_vs_gt failed:\n%s", traceback.format_exc())
                all_metrics.append({"sample_id": sample_id, "error": "evaluation failed"})

            processed += 1

        if args.sample_limit > 0 and processed >= args.sample_limit:
            break

    # --- write summary ---
    summary_path = args.output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Wrote eval summary to %s (%d samples)", summary_path, len(all_metrics))

    # Print aggregate stats if PSNR available
    psnr_vals = [m["psnr_mean"] for m in all_metrics if "psnr_mean" in m]
    ssim_vals = [m["ssim_mean"] for m in all_metrics if "ssim_mean" in m]
    if psnr_vals:
        log.info("Aggregate PSNR: %.2f ± %.2f over %d samples",
                 float(np.mean(psnr_vals)), float(np.std(psnr_vals)), len(psnr_vals))
    if ssim_vals:
        log.info("Aggregate SSIM: %.4f ± %.4f over %d samples",
                 float(np.mean(ssim_vals)), float(np.std(ssim_vals)), len(ssim_vals))

    errors = [m for m in all_metrics if "error" in m]
    if errors:
        log.warning("%d sample(s) had errors.", len(errors))
    log.info("Done. Processed %d sample(s).", processed)


if __name__ == "__main__":
    main()

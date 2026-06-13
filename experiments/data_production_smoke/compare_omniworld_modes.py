#!/usr/bin/env python3
"""
compare_omniworld_modes.py

Compares GT-depth vs Default mode results from the SANA-WM OmniWorld pipeline.

Produces:
  - Printed comparison table (pose accuracy + video quality)
  - 3-panel side-by-side comparison video: [GT original | GT-depth generated | Default generated]
  - Markdown report

Usage:
  python compare_omniworld_modes.py [--sample-id SAMPLE_ID] [--out-dir OUT_DIR]
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants / Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_ID = "020c2bed1dbb"
WORK_BASE = "/mnt/afs/davidwang/workspace/data/omniworld_smoke"
DEFAULT_OUT_DIR = "/mnt/afs/davidwang/workspace/data/omniworld_smoke/comparison"
GTDEPTH_RESULTS = "/mnt/afs/davidwang/workspace/data/sana_wm_results_gtdepth"
DEFAULT_RESULTS = "/mnt/afs/davidwang/workspace/data/sana_wm_results_default"

TARGET_H = 704
TARGET_W = 1280
VIDEO_FPS = 16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resize_frame(frame: np.ndarray, h: int = TARGET_H, w: int = TARGET_W) -> np.ndarray:
    """Resize an HWC uint8 frame to (h, w) using PIL LANCZOS."""
    from PIL import Image

    img = Image.fromarray(frame)
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def load_video_frames(path: str | Path, target_h: int = TARGET_H, target_w: int = TARGET_W):
    """
    Load all frames from a video file and resize to (target_h, target_w).
    Returns list of np.ndarray (H, W, 3) uint8, or None on failure.
    """
    import imageio.v3 as iio

    path = Path(path)
    if not path.exists():
        warnings.warn(f"Video not found: {path}")
        return None
    try:
        frames = list(iio.imiter(str(path), plugin="pyav"))
        frames = [resize_frame(f, target_h, target_w) for f in frames]
        return frames
    except Exception as e:
        warnings.warn(f"Failed to load video {path}: {e}")
        return None


def black_frames(n: int, h: int = TARGET_H, w: int = TARGET_W) -> list:
    """Return a list of n black frames."""
    return [np.zeros((h, w, 3), dtype=np.uint8)] * n


def add_label(frame: np.ndarray, label: str, font_size: int = 24) -> np.ndarray:
    """Overlay text label on the top-left corner of a frame (in-place copy)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    # Shadow for readability
    draw.text((11, 11), label, fill=(0, 0, 0), font=font)
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return np.array(img)


def build_3panel_frame(gt_frame, gtdepth_frame, default_frame, labels=None) -> np.ndarray:
    """Concatenate three (H, W, 3) frames horizontally with optional labels."""
    if labels is None:
        labels = ["GT Original", "GT-depth", "Default"]
    panels = [gt_frame, gtdepth_frame, default_frame]
    labeled = [add_label(p, lbl) for p, lbl in zip(panels, labels)]
    return np.concatenate(labeled, axis=1)  # (H, 3*W, 3)


def write_video(frames: list, out_path: Path, fps: int = VIDEO_FPS) -> bool:
    """
    Write list of (H, W, 3) uint8 frames to out_path as mp4 (libx264).
    Uses static_ffmpeg for reliable encoding.
    Returns True on success, False on failure.
    """
    import tempfile, subprocess, shutil
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="cmp_frames_"))
    try:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(tmp_dir / f"frame_{i:05d}.png")

        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
        except ImportError:
            pass

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            warnings.warn(f"ffmpeg encoding failed: {result.stderr[-300:]}")
            return False
        return True
    except Exception as e:
        warnings.warn(f"Video encoding failed ({e}).")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_video_quality(
    gt_frames: list,
    gen_frames: list,
) -> dict:
    """
    Compute per-frame PSNR and SSIM between gt_frames and gen_frames.
    Returns dict with psnr_mean, psnr_std, ssim_mean, ssim_std.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    n = min(len(gt_frames), len(gen_frames))
    if n == 0:
        return {"psnr_mean": None, "psnr_std": None, "ssim_mean": None, "ssim_std": None}

    psnrs, ssims = [], []
    for gt, gen in zip(gt_frames[:n], gen_frames[:n]):
        p = psnr_fn(gt, gen, data_range=255)
        s = ssim_fn(gt, gen, channel_axis=2, data_range=255)
        psnrs.append(p)
        ssims.append(s)

    return {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_pose_ate(summary_json: Path, sample_id: str) -> float | None:
    """
    Load ATE RMSE for sample_id from pose_eval_summary.json.
    Returns float in metres, or None if unavailable.
    """
    if not summary_json.exists():
        warnings.warn(f"Pose eval summary not found: {summary_json}")
        return None
    with open(summary_json) as f:
        data = json.load(f)
    # data may be a list or a dict keyed by sample_id
    if isinstance(data, list):
        for entry in data:
            if entry.get("sample_id") == sample_id:
                return entry.get("ate_rmse")
    elif isinstance(data, dict):
        if sample_id in data:
            return data[sample_id].get("ate_rmse")
        # maybe the dict IS the single entry
        if data.get("sample_id") == sample_id:
            return data.get("ate_rmse")
    warnings.warn(f"sample_id {sample_id!r} not found in {summary_json}")
    return None


def load_video_quality_from_summary(eval_json: Path, sample_id: str) -> dict | None:
    """
    Load pre-computed PSNR/SSIM from eval_summary.json.
    Returns dict with psnr_mean, psnr_std, ssim_mean, ssim_std, or None.
    """
    if not eval_json.exists():
        return None
    with open(eval_json) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        if entry.get("sample_id") == sample_id:
            return {
                "psnr_mean": entry.get("psnr_mean"),
                "psnr_std": entry.get("psnr_std"),
                "ssim_mean": entry.get("ssim_mean"),
                "ssim_std": entry.get("ssim_std"),
            }
    return None


def get_video_quality(
    results_dir: Path,
    sample_id: str,
    gt_frames: list | None,
) -> dict:
    """
    Attempt to load video quality from eval_summary.json; fall back to inline compute.
    Returns dict with psnr_mean / psnr_std / ssim_mean / ssim_std (may be None values).
    """
    summary = load_video_quality_from_summary(results_dir / "eval_summary.json", sample_id)
    if summary and summary.get("psnr_mean") is not None:
        return summary

    gen_path = results_dir / sample_id / f"{sample_id}_generated.mp4"
    gen_frames = load_video_frames(gen_path)
    if gen_frames is None:
        warnings.warn(f"Generated video missing for inline compute: {gen_path}")
        return {"psnr_mean": None, "psnr_std": None, "ssim_mean": None, "ssim_std": None}
    if gt_frames is None:
        warnings.warn("GT frames not available; cannot compute video quality.")
        return {"psnr_mean": None, "psnr_std": None, "ssim_mean": None, "ssim_std": None}

    print(f"  Computing PSNR/SSIM inline for {results_dir.name} (n_frames={min(len(gt_frames), len(gen_frames))})...")
    return compute_video_quality(gt_frames, gen_frames)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt_m(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.6f}"


def fmt_mm(v) -> str:
    if v is None:
        return "N/A"
    return f"{v * 1000:.2f}"


def fmt_db(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}"


def fmt_ssim(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def print_comparison_table(
    sample_id: str,
    gtdepth_ate: float | None,
    default_ate: float | None,
    gtdepth_vq: dict,
    default_vq: dict,
    out_video_path: Path,
):
    print()
    print(f"=== OmniWorld Mode Comparison ({sample_id}) ===")
    print()
    print("Pose Accuracy (ATE RMSE vs OmniWorld GT):")
    print(f"  GT-depth:  {fmt_m(gtdepth_ate)} m   ({fmt_mm(gtdepth_ate)} mm)")
    print(f"  Default:   {fmt_m(default_ate)} m   ({fmt_mm(default_ate)} mm)")

    if gtdepth_ate is not None and default_ate is not None and gtdepth_ate > 0:
        ratio = default_ate / gtdepth_ate
        direction = "Default worse than GT-depth" if ratio > 1.0 else "Default better than GT-depth"
        print(f"  Ratio:     {ratio:.1f}x ({direction})")
    else:
        print("  Ratio:     N/A")

    print()
    psnr_gt = gtdepth_vq.get("psnr_mean")
    ssim_gt = gtdepth_vq.get("ssim_mean")
    psnr_def = default_vq.get("psnr_mean")
    ssim_def = default_vq.get("ssim_mean")

    print("Video Quality (PSNR / SSIM vs GT original):")
    print(f"  GT-depth:  PSNR = {fmt_db(psnr_gt)} dB,  SSIM = {fmt_ssim(ssim_gt)}")
    print(f"  Default:   PSNR = {fmt_db(psnr_def)} dB,  SSIM = {fmt_ssim(ssim_def)}")

    if psnr_gt is not None and psnr_def is not None:
        delta_psnr = psnr_def - psnr_gt
        sign = "+" if delta_psnr >= 0 else ""
        print(f"  Delta PSNR: {sign}{delta_psnr:.2f} dB (positive = Default better)")
    else:
        print("  Delta PSNR: N/A")

    print()
    print(f"3-panel comparison video: {out_video_path}")
    print()


def write_markdown_report(
    out_dir: Path,
    sample_id: str,
    gtdepth_ate: float | None,
    default_ate: float | None,
    gtdepth_vq: dict,
    default_vq: dict,
    out_video_path: Path,
):
    psnr_gt = gtdepth_vq.get("psnr_mean")
    psnr_gt_std = gtdepth_vq.get("psnr_std")
    ssim_gt = gtdepth_vq.get("ssim_mean")
    ssim_gt_std = gtdepth_vq.get("ssim_std")
    psnr_def = default_vq.get("psnr_mean")
    psnr_def_std = default_vq.get("psnr_std")
    ssim_def = default_vq.get("ssim_mean")
    ssim_def_std = default_vq.get("ssim_std")

    def _psnr_cell(mean, std):
        if mean is None:
            return "N/A"
        if std is not None:
            return f"{mean:.2f} ± {std:.2f} dB"
        return f"{mean:.2f} dB"

    def _ssim_cell(mean, std):
        if mean is None:
            return "N/A"
        if std is not None:
            return f"{mean:.4f} ± {std:.4f}"
        return f"{mean:.4f}"

    ratio_str = "N/A"
    if gtdepth_ate is not None and default_ate is not None and gtdepth_ate > 0:
        ratio = default_ate / gtdepth_ate
        direction = "worse" if ratio > 1.0 else "better"
        ratio_str = f"{ratio:.2f}x (Default {direction} than GT-depth)"

    delta_psnr_str = "N/A"
    interpretation_vq = ""
    if psnr_gt is not None and psnr_def is not None:
        delta = psnr_def - psnr_gt
        sign = "+" if delta >= 0 else ""
        delta_psnr_str = f"{sign}{delta:.2f} dB"
        if delta > 0:
            interpretation_vq = f"Default mode achieves {delta:.2f} dB higher PSNR than GT-depth mode."
        else:
            interpretation_vq = f"GT-depth mode achieves {-delta:.2f} dB higher PSNR than Default mode."

    interpretation_pose = ""
    if gtdepth_ate is not None and default_ate is not None and gtdepth_ate > 0:
        ratio = default_ate / gtdepth_ate
        if ratio > 1:
            interpretation_pose = (
                f"Default mode has {ratio:.1f}x higher ATE RMSE, likely due to SLAM drift "
                f"without GT depth anchor. GT-depth mode benefits from ground-truth depth "
                f"supervision which constrains the VIPE SLAM trajectory."
            )
        else:
            interpretation_pose = (
                f"Default mode achieves {1/ratio:.1f}x lower ATE RMSE than GT-depth mode, "
                f"suggesting the Pi3X + MoGe-2 depth estimates are well-calibrated for this sample."
            )

    lines = [
        "# OmniWorld Mode Comparison Report",
        "",
        f"**Date generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Sample ID:** `{sample_id}`",
        f"**Dataset:** OmniWorld-Game",
        "",
        "---",
        "",
        "## 1. Pose Accuracy (ATE RMSE vs OmniWorld GT)",
        "",
        "| Mode | ATE RMSE (m) | ATE RMSE (mm) |",
        "|------|-------------|--------------|",
        f"| GT-depth | {fmt_m(gtdepth_ate)} | {fmt_mm(gtdepth_ate)} |",
        f"| Default  | {fmt_m(default_ate)} | {fmt_mm(default_ate)} |",
        f"| **Ratio** | | {ratio_str} |",
        "",
    ]
    if interpretation_pose:
        lines += [f"**Interpretation:** {interpretation_pose}", ""]

    lines += [
        "---",
        "",
        "## 2. Video Quality (vs GT Original)",
        "",
        "| Mode | PSNR | SSIM |",
        "|------|------|------|",
        f"| GT-depth | {_psnr_cell(psnr_gt, psnr_gt_std)} | {_ssim_cell(ssim_gt, ssim_gt_std)} |",
        f"| Default  | {_psnr_cell(psnr_def, psnr_def_std)} | {_ssim_cell(ssim_def, ssim_def_std)} |",
        f"| **Delta (Default - GT-depth)** | {delta_psnr_str} | |",
        "",
    ]
    if interpretation_vq:
        lines += [f"**Interpretation:** {interpretation_vq}", ""]

    lines += [
        "---",
        "",
        "## 3. Output Files",
        "",
        f"- 3-panel comparison video: `{out_video_path}`",
        f"- This report: `{out_dir / 'comparison_report.md'}`",
        "",
        "---",
        "",
        "## 4. Pipeline Configuration",
        "",
        "| | GT-depth mode | Default mode |",
        "|---|---|---|",
        "| Depth source | OmniWorld GT depth | Pi3X + MoGe-2 |",
        "| Pose estimation | VIPE SLAM | VIPE SLAM |",
        "| Camera model | GT intrinsics | Estimated |",
        "",
    ]

    report_path = out_dir / "comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Markdown report written to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GT-depth vs Default OmniWorld pipeline results.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID, help="OmniWorld sample ID")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for comparison artifacts")
    parser.add_argument(
        "--work-base",
        default=WORK_BASE,
        help="Base directory containing shards_gtdepth/ and shards_default/",
    )
    parser.add_argument(
        "--gtdepth-results",
        default=GTDEPTH_RESULTS,
        help="Directory of SANA-WM GT-depth inference outputs",
    )
    parser.add_argument(
        "--default-results",
        default=DEFAULT_RESULTS,
        help="Directory of SANA-WM Default inference outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_id = args.sample_id
    out_dir = Path(args.out_dir)
    work_base = Path(args.work_base)
    gtdepth_results_dir = Path(args.gtdepth_results)
    default_results_dir = Path(args.default_results)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load pose eval results
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading pose eval results for sample {sample_id!r}...")

    gtdepth_pose_json = work_base / "shards_gtdepth" / "eval_output" / "pose_eval_summary.json"
    default_pose_json = work_base / "shards_default" / "eval_output" / "pose_eval_summary.json"

    gtdepth_ate = load_pose_ate(gtdepth_pose_json, sample_id)
    default_ate = load_pose_ate(default_pose_json, sample_id)

    print(f"  GT-depth ATE RMSE: {fmt_m(gtdepth_ate)} m")
    print(f"  Default  ATE RMSE: {fmt_m(default_ate)} m")

    # ------------------------------------------------------------------
    # 2 & 3. Load / compute video quality metrics
    # ------------------------------------------------------------------
    print(f"\n[2/4] Loading video frames for sample {sample_id!r}...")

    gt_video_path = gtdepth_results_dir / sample_id / f"{sample_id}.mp4"
    print(f"  Loading GT video: {gt_video_path}")
    gt_frames = load_video_frames(gt_video_path)
    if gt_frames is None:
        print(f"  WARNING: GT video not found at {gt_video_path}; falling back to black frames for video panel.")

    print("\n[3/4] Computing / loading video quality metrics...")
    gtdepth_vq = get_video_quality(gtdepth_results_dir, sample_id, gt_frames)
    default_vq = get_video_quality(default_results_dir, sample_id, gt_frames)
    print(f"  GT-depth VQ: PSNR={fmt_db(gtdepth_vq.get('psnr_mean'))} dB, SSIM={fmt_ssim(gtdepth_vq.get('ssim_mean'))}")
    print(f"  Default  VQ: PSNR={fmt_db(default_vq.get('psnr_mean'))} dB, SSIM={fmt_ssim(default_vq.get('ssim_mean'))}")

    # ------------------------------------------------------------------
    # 4. Generate 3-panel comparison video
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating 3-panel comparison video...")

    gtdepth_gen_path = gtdepth_results_dir / sample_id / f"{sample_id}_generated.mp4"
    default_gen_path = default_results_dir / sample_id / f"{sample_id}_generated.mp4"

    gtdepth_gen_frames = load_video_frames(gtdepth_gen_path)
    default_gen_frames = load_video_frames(default_gen_path)

    available = [x is not None for x in [gt_frames, gtdepth_gen_frames, default_gen_frames]]
    if sum(available) < 2:
        print("  WARNING: Fewer than 2 video sources available; skipping 3-panel video generation.")
        out_video_path = out_dir / f"{sample_id}_comparison_3panel.mp4"
    else:
        # Determine number of frames (use the shortest non-None sequence)
        n_frames = min(
            len(f) for f in [gt_frames, gtdepth_gen_frames, default_gen_frames] if f is not None
        )

        if gt_frames is None:
            gt_frames = black_frames(n_frames)
        if gtdepth_gen_frames is None:
            gtdepth_gen_frames = black_frames(n_frames)
        if default_gen_frames is None:
            default_gen_frames = black_frames(n_frames)

        panel_frames = []
        for i in range(n_frames):
            panel = build_3panel_frame(
                gt_frames[i],
                gtdepth_gen_frames[i],
                default_gen_frames[i],
                labels=["GT Original", "GT-depth", "Default"],
            )
            panel_frames.append(panel)

        out_video_path = out_dir / f"{sample_id}_comparison_3panel.mp4"
        success = write_video(panel_frames, out_video_path, fps=VIDEO_FPS)
        if success:
            print(f"  3-panel video written: {out_video_path}")
        else:
            print(f"  3-panel frames (PNG fallback) written to: {out_dir / 'frames'}")

    # ------------------------------------------------------------------
    # 5. Print comparison table
    # ------------------------------------------------------------------
    print_comparison_table(
        sample_id=sample_id,
        gtdepth_ate=gtdepth_ate,
        default_ate=default_ate,
        gtdepth_vq=gtdepth_vq,
        default_vq=default_vq,
        out_video_path=out_video_path,
    )

    # ------------------------------------------------------------------
    # 6. Write Markdown report
    # ------------------------------------------------------------------
    write_markdown_report(
        out_dir=out_dir,
        sample_id=sample_id,
        gtdepth_ate=gtdepth_ate,
        default_ate=default_ate,
        gtdepth_vq=gtdepth_vq,
        default_vq=default_vq,
        out_video_path=out_video_path,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify WebDataset shards and evaluate pose accuracy vs DL3DV GT.

Modes:
  --mode schema     : verify each .tar contains exactly 6 required files
  --mode pose-eval  : compute ATE RMSE / RTE vs GT using evo library
                      (generates 3-view trajectory comparison plot)

Usage:
  # Schema check
  python verify_and_eval.py --mode schema --shards-dir /path/to/shards

  # Pose evaluation
  python verify_and_eval.py --mode pose-eval \
    --shards-dir /path/to/shards \
    --scenes-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke
"""

import argparse
import io
import json
import tarfile
from pathlib import Path
from collections import defaultdict

import numpy as np

REQUIRED_SUFFIXES = {
    "mp4",
    "poses_c2w.npy",
    "intrinsics.npy",
    "scale.npy",
    "caption.txt",
    "meta.json",
}


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------

def run_schema_check(shards_dir: Path) -> None:
    """Verify each .tar in shards_dir contains exactly the 6 required files."""
    tar_files = sorted(shards_dir.glob("*.tar"))
    if not tar_files:
        print(f"[ERROR] No .tar files found in {shards_dir}")
        return

    n_valid = 0
    n_total = len(tar_files)

    for tar_path in tar_files:
        try:
            valid, report = _check_tar_schema(tar_path)
        except Exception as e:
            print(f"  [FAIL] {tar_path.name}: exception — {e}")
            continue

        if valid:
            n_valid += 1
            print(f"  [OK]   {tar_path.name}  ({report})")
        else:
            print(f"  [FAIL] {tar_path.name}  {report}")

    print(f"\nResult: {n_valid}/{n_total} shards valid")


def _check_tar_schema(tar_path: Path):
    """Return (is_valid, report_str) for one tar file."""
    with tarfile.open(tar_path, "r") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]

    # Group by sample_id (strip the first extension component)
    sample_files: dict[str, set] = defaultdict(set)
    for name in names:
        p = Path(name)
        # Handle compound extensions like "foo.poses_c2w.npy"
        stem = p.name
        for suffix in REQUIRED_SUFFIXES:
            if stem.endswith("." + suffix):
                sample_id = stem[: -(len(suffix) + 1)]
                sample_files[sample_id].add(suffix)
                break
        else:
            # Unknown extension — record as-is so we can report
            sample_files[p.stem].add(p.suffix.lstrip("."))

    n_samples = len(sample_files)
    bad_samples = []
    for sid, found in sample_files.items():
        missing = REQUIRED_SUFFIXES - found
        extra = found - REQUIRED_SUFFIXES
        if missing or extra:
            bad_samples.append((sid, missing, extra))

    if bad_samples:
        details = "; ".join(
            f"{sid} missing={m} extra={e}" for sid, m, e in bad_samples
        )
        return False, f"{n_samples} samples, {len(bad_samples)} invalid: {details}"

    return True, f"{n_samples} samples all valid"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_from_tar(tar_path: Path, sample_id: str, key: str) -> bytes:
    """Extract raw bytes for '{sample_id}.{key}' from a tar archive."""
    target = f"{sample_id}.{key}"
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name == target or member.name.endswith("/" + target):
                f = tf.extractfile(member)
                if f is None:
                    raise ValueError(f"Cannot extract {target} from {tar_path}")
                return f.read()
    raise KeyError(f"{target} not found in {tar_path}")


def downsample_gt_poses(
    gt_poses: np.ndarray, orig_fps: float, T_16fps: int
) -> np.ndarray:
    """Downsample GT poses from orig_fps to 16 fps.

    Args:
        gt_poses:  (T', 4, 4) float32, at orig_fps.
        orig_fps:  original frame rate of GT.
        T_16fps:   number of frames at 16 fps (i.e. len of estimated poses).

    Returns:
        (T_16fps, 4, 4) float32
    """
    T_prime = len(gt_poses)
    gt_inds = [
        min(round(i * orig_fps / 16), T_prime - 1) for i in range(T_16fps)
    ]
    return gt_poses[gt_inds]


def _poses_to_quats(poses: np.ndarray) -> np.ndarray:
    """(T, 4, 4) -> (T, 4) wxyz quaternions via scipy."""
    from scipy.spatial.transform import Rotation

    rots = Rotation.from_matrix(poses[:, :3, :3])
    xyzw = rots.as_quat()  # scipy gives xyzw
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return wxyz.astype(np.float64)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_trajectory(
    est_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    scene_id: str,
    out_dir: Path,
) -> None:
    """Save a 3-view (xy, xz, yz) trajectory comparison plot.

    Red = GT, Blue = estimated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available; skipping trajectory plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [
        (0, 1, "xy"),
        (0, 2, "xz"),
        (1, 2, "yz"),
    ]
    for ax, (i, j, label) in zip(axes, views):
        ax.plot(gt_xyz[:, i], gt_xyz[:, j], "r-o", markersize=2, linewidth=1,
                label="GT")
        ax.plot(est_xyz[:, i], est_xyz[:, j], "b-o", markersize=2, linewidth=1,
                label="Estimated")
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_title(f"{label} plane")
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"Trajectory comparison — {scene_id}", fontsize=12)
    plt.tight_layout()

    out_path = out_dir / f"{scene_id}_traj_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] Saved {out_path}")


# ---------------------------------------------------------------------------
# Pose evaluation
# ---------------------------------------------------------------------------

def _compute_ate(
    poses_est: np.ndarray,
    gt_poses_16fps: np.ndarray,
) -> float | None:
    """Compute ATE RMSE using evo. Returns None if evo is unavailable."""
    try:
        from evo.core.trajectory import PoseTrajectory3D
        from evo.core import metrics
        import evo.main_ape as evo_ape
    except ImportError:
        print("  [WARN] evo library not available; skipping ATE computation")
        return None

    est_xyz = poses_est[:, :3, 3].astype(np.float64)
    gt_xyz = gt_poses_16fps[:, :3, 3].astype(np.float64)
    timestamps = np.arange(len(est_xyz), dtype=np.float64)

    traj_est = PoseTrajectory3D(
        positions_xyz=est_xyz,
        orientations_quat_wxyz=_poses_to_quats(poses_est),
        timestamps=timestamps,
    )
    traj_ref = PoseTrajectory3D(
        positions_xyz=gt_xyz,
        orientations_quat_wxyz=_poses_to_quats(gt_poses_16fps),
        timestamps=timestamps,
    )

    result = evo_ape.ape(
        traj_ref,
        traj_est,
        est_name="estimated",
        pose_relation=metrics.PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        verbose=False,
    )
    return float(result.stats["rmse"])


def _eval_sample(
    tar_path: Path,
    sample_id: str,
    scenes_dir: Path,
    out_dir: Path,
) -> dict | None:
    """Evaluate one sample. Returns a dict with results, or None on failure."""
    # --- load estimated poses from tar ---
    try:
        poses_bytes = extract_from_tar(tar_path, sample_id, "poses_c2w.npy")
        poses_est = np.load(io.BytesIO(poses_bytes)).astype(np.float32)
    except Exception as e:
        print(f"    [ERROR] Cannot load poses_c2w.npy for {sample_id}: {e}")
        return None

    # --- load meta to get scene_id ---
    try:
        meta_bytes = extract_from_tar(tar_path, sample_id, "meta.json")
        meta = json.loads(meta_bytes.decode("utf-8"))
        scene_id = meta["scene_id"]
    except Exception as e:
        print(f"    [ERROR] Cannot load meta.json for {sample_id}: {e}")
        return None

    # --- load GT ---
    scene_dir = scenes_dir / scene_id
    gt_poses_path = scene_dir / "gt_poses.npy"
    orig_fps_path = scene_dir / "orig_fps.txt"

    if not gt_poses_path.exists():
        print(f"    [WARN] GT poses not found: {gt_poses_path}")
        return None
    if not orig_fps_path.exists():
        print(f"    [WARN] orig_fps.txt not found: {orig_fps_path}")
        return None

    try:
        gt_poses = np.load(gt_poses_path).astype(np.float32)
        orig_fps = float(orig_fps_path.read_text().strip())
    except Exception as e:
        print(f"    [ERROR] Cannot load GT for scene {scene_id}: {e}")
        return None

    # --- downsample GT ---
    T_16fps = len(poses_est)
    gt_poses_16fps = downsample_gt_poses(gt_poses, orig_fps, T_16fps)

    # --- ATE ---
    ate_rmse = _compute_ate(poses_est, gt_poses_16fps)

    # --- trajectory plot ---
    est_xyz = poses_est[:, :3, 3]
    gt_xyz = gt_poses_16fps[:, :3, 3]
    plot_trajectory(est_xyz, gt_xyz, f"{scene_id}_{sample_id}", out_dir)

    result = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "T_est": T_16fps,
        "T_gt_orig": len(gt_poses),
        "orig_fps": orig_fps,
        "ate_rmse": ate_rmse,
    }
    return result


def run_pose_eval(shards_dir: Path, scenes_dir: Path, out_dir: Path) -> None:
    """Run ATE evaluation for all samples in all shards."""
    if scenes_dir is None:
        print("[ERROR] --scenes-dir is required for pose-eval mode")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tar_files = sorted(shards_dir.glob("*.tar"))
    if not tar_files:
        print(f"[ERROR] No .tar files found in {shards_dir}")
        return

    all_results = []

    for tar_path in tar_files:
        print(f"\n[SHARD] {tar_path.name}")
        # Find sample ids in this tar
        try:
            with tarfile.open(tar_path, "r") as tf:
                names = [m.name for m in tf.getmembers() if m.isfile()]
        except Exception as e:
            print(f"  [ERROR] Cannot open {tar_path}: {e}")
            continue

        # Collect sample ids (identify by meta.json presence)
        sample_ids = []
        for name in names:
            p = Path(name)
            stem = p.name
            if stem.endswith(".meta.json"):
                sid = stem[: -len(".meta.json")]
                sample_ids.append(sid)

        if not sample_ids:
            print(f"  [WARN] No samples found (no .meta.json entries)")
            continue

        print(f"  Found {len(sample_ids)} samples")

        for sample_id in sample_ids:
            print(f"  [SAMPLE] {sample_id}")
            result = _eval_sample(tar_path, sample_id, scenes_dir, out_dir)
            if result is not None:
                all_results.append(result)
                ate_str = (
                    f"{result['ate_rmse']:.6f}"
                    if result["ate_rmse"] is not None
                    else "N/A"
                )
                print(
                    f"    ATE RMSE={ate_str}  "
                    f"T_est={result['T_est']}  "
                    f"T_gt_orig={result['T_gt_orig']}  "
                    f"orig_fps={result['orig_fps']:.2f}"
                )

    # Summary
    print("\n" + "=" * 60)
    print(f"Evaluated {len(all_results)} samples total")
    ate_values = [r["ate_rmse"] for r in all_results if r["ate_rmse"] is not None]
    if ate_values:
        print(f"ATE RMSE — mean: {np.mean(ate_values):.6f}  "
              f"median: {np.median(ate_values):.6f}  "
              f"max: {np.max(ate_values):.6f}")
    else:
        print("No ATE values computed (evo unavailable or all samples failed)")

    # Save summary JSON
    summary_path = out_dir / "pose_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify WebDataset shards and evaluate pose accuracy vs DL3DV GT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["schema", "pose-eval"],
        required=True,
        help="Operation mode.",
    )
    parser.add_argument(
        "--shards-dir",
        required=True,
        type=Path,
        help="Directory containing .tar shard files.",
    )
    parser.add_argument(
        "--scenes-dir",
        default=None,
        type=Path,
        help="Directory containing per-scene GT data (required for pose-eval).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Output directory for plots and summary JSON (default: shards-dir/eval_output).",
    )
    args = parser.parse_args()

    shards_dir = args.shards_dir
    if not shards_dir.exists():
        parser.error(f"--shards-dir does not exist: {shards_dir}")

    out_dir = args.out_dir or (shards_dir / "eval_output")

    if args.mode == "schema":
        run_schema_check(shards_dir)
    elif args.mode == "pose-eval":
        if args.scenes_dir is None:
            parser.error("--scenes-dir is required for pose-eval mode")
        if not args.scenes_dir.exists():
            parser.error(f"--scenes-dir does not exist: {args.scenes_dir}")
        run_pose_eval(shards_dir, args.scenes_dir, out_dir)


if __name__ == "__main__":
    main()

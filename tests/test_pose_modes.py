"""Tests for stage02_pose three annotation modes (default/gt-depth/gt-pose).

Subprocess invocations are intercepted via monkeypatch — these tests verify
input/output plumbing and shapes without needing VIPE or Pi3X installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from sana_wm_pipeline.stage02_pose import mode_default, mode_gtdepth, mode_gtpose
from sana_wm_pipeline.stage02_pose._common import PoseArtifact


T = 961   # paper camera-frame count


# ---- Helpers ---------------------------------------------------------------
def _ok_intrinsics(t: int = T) -> np.ndarray:
    return np.tile([[[700.0, 700.0, 640.0, 360.0]]], (t, 1, 1)).astype(np.float32)


def _eye_poses(t: int = T) -> np.ndarray:
    return np.tile(np.eye(4, dtype=np.float32), (t, 1, 1))


# ---- Default mode ----------------------------------------------------------
def test_default_mode_loads_artifact(monkeypatch, tmp_path: Path):
    """VIPE now writes npz artifacts under pose/ and intrinsics/ subdirs."""
    T_frames = T
    poses = _eye_poses(T_frames)  # (T, 4, 4)
    intr_raw = np.tile(
        np.array([700.0, 700.0, 640.0, 360.0], dtype=np.float32), (T_frames, 1)
    )  # (T, 4)

    def fake_call(cmd: Sequence[str], **kw):
        # vipe infer <clip> --output <work_dir> --pipeline <name>
        cmd = list(cmd)
        out_idx = cmd.index("--output") + 1
        work_dir = Path(cmd[out_idx])
        stem = "clip"   # clip_path.stem = "clip"

        # Write VIPE's pose/ and intrinsics/ artifacts.
        pose_dir = work_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pose_dir / f"{stem}.npz",
                 data=poses, inds=np.arange(T_frames))

        intr_dir = work_dir / "intrinsics"
        intr_dir.mkdir(parents=True, exist_ok=True)
        np.savez(intr_dir / f"{stem}.npz",
                 data=intr_raw, inds=np.arange(T_frames))

    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_default._precompute_depth_cache",
        lambda *a, **kw: None,
    )
    monkeypatch.setenv("SANA_WM_PI3X_WEIGHTS", "/fake/pi3x")
    monkeypatch.setenv("SANA_WM_MOGE2_WEIGHTS", "/fake/moge2")
    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_default.subprocess.check_call",
        fake_call,
    )
    art = mode_default.run_default(Path("clip.mp4"), tmp_path)
    art.validate(T)
    assert isinstance(art, PoseArtifact)
    # intrinsics must be (T, 1, 4)
    assert art.intrinsics.shape == (T, 1, 4)
    # scale_per_frame ones when no Pi3X
    assert art.scale_per_frame.shape == (T,)


# ---- GT-depth mode ---------------------------------------------------------
def test_gtdepth_mode_recovers_scale(monkeypatch, tmp_path: Path):
    """mode_gtdepth: GT depth=1, MoGe-2 depth=2 → scale_per_frame ≈ 2.0."""
    gt_depth = tmp_path / "gt_depth.npy"
    np.save(gt_depth, np.full((T, 90, 160), 1.0, dtype=np.float32))

    def fake_moge2(clip_path, moge_out, moge2_weights, fov_x_deg=60.0, device="cuda"):
        # MoGe-2 returns 2.0 m everywhere → fuse_metric_scale gives s=2.0
        depths = np.full((T, 90, 160), 2.0, dtype=np.float32)
        np.save(str(moge_out), depths)
        return depths

    def fake_vipe(cmd: Sequence[str], **kw):
        # VIPE writes pose/<stem>.npz + intrinsics/<stem>.npz
        cmd = list(cmd)
        out_idx = cmd.index("--output") + 1
        work_dir = Path(cmd[out_idx])
        # cmd = ["vipe", "infer", "<clip>", "--output", "<work_dir>", "--pipeline", ...]
        out_idx = cmd.index("--output")
        stem = Path(cmd[out_idx - 1]).stem  # clip is the arg just before --output

        pose_dir = work_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        np.savez(pose_dir / f"{stem}.npz",
                 data=_eye_poses(), inds=np.arange(T))

        intr_dir = work_dir / "intrinsics"
        intr_dir.mkdir(parents=True, exist_ok=True)
        intr_raw = np.tile(
            np.array([700.0, 700.0, 640.0, 360.0], dtype=np.float32), (T, 1)
        )
        np.savez(intr_dir / f"{stem}.npz",
                 data=intr_raw, inds=np.arange(T))

    monkeypatch.setenv("SANA_WM_MOGE2_WEIGHTS", "/fake/moge2")
    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtdepth._run_moge2",
        fake_moge2,
    )
    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtdepth.subprocess.check_call",
        fake_vipe,
    )

    art = mode_gtdepth.run_gtdepth(Path("clip.mp4"), gt_depth, tmp_path)
    art.validate(T)
    # fuse_metric_scale(d_gt=1, d_moge=2): closed-form s* = 2.0, EMA stays at 2.0
    assert art.scale_per_frame.mean() == pytest.approx(2.0, rel=1e-3)
    assert art.intrinsics.shape == (T, 1, 4)


# ---- GT-pose mode ----------------------------------------------------------
def test_gtpose_mode_aligns_via_umeyama(monkeypatch, tmp_path: Path):
    gt_path = tmp_path / "gt_poses.npy"
    # Build GT trajectory: a line scaled by factor 5 vs Pi3X.
    rng = np.random.default_rng(0)
    pi3x_centers = rng.standard_normal((T, 3))
    gt_centers = 5.0 * pi3x_centers
    gt_poses = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    gt_poses[:, :3, 3] = gt_centers.astype(np.float32)
    np.save(gt_path, gt_poses)

    cams = {"frames": []}
    for c in pi3x_centers:
        K = np.eye(3).tolist()
        K[0][0] = 800.0; K[1][1] = 800.0; K[0][2] = 640.0; K[1][2] = 360.0
        cams["frames"].append({"center": c.tolist(), "K": K})
    cams_json_path = tmp_path / "cams_pi3x.json"
    cams_json_path.write_text(json.dumps(cams))

    def fake_call(cmd: Sequence[str], **kw):
        # Pi3X stub: cams + (empty) points already written.
        cmd = list(cmd)
        emit_cams = cmd[cmd.index("--emit-cams") + 1]
        emit_pts = cmd[cmd.index("--emit-points") + 1]
        Path(emit_cams).write_text(json.dumps(cams))
        np.save(emit_pts, np.zeros((1, 3), dtype=np.float32))

    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_gtpose.subprocess.check_call",
        fake_call,
    )
    art = mode_gtpose.run_gtpose(Path("clip.mp4"), gt_path, tmp_path)
    # Default-mode shape check
    assert art.poses_c2w.shape == (T, 4, 4)
    assert art.intrinsics.shape == (T, 1, 4)
    assert art.scale_per_frame.shape == (T,)
    # Sim(3) scale ≈ 5
    assert art.scale_per_frame.mean() == pytest.approx(5.0, rel=1e-3)


# ---- _common.PoseArtifact validation --------------------------------------
def test_pose_artifact_validate_rejects_bad_first_frame():
    bad_poses = _eye_poses()
    bad_poses[0, 0, 3] = 10.0   # translate first frame off origin
    art = PoseArtifact(
        poses_c2w=bad_poses,
        intrinsics=_ok_intrinsics(),
        scale_per_frame=np.ones(T, np.float32),
    )
    with pytest.raises(AssertionError, match="identity"):
        art.validate(T)

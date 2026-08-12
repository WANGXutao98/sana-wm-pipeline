"""Pose-stage dry-run tests across the three annotation modes."""

import numpy as np
import pytest

from sana_wm_data.manifest import ClipRecord
from sana_wm_data.pose.stage import annotate_pose

MODELS = {"dry_run": True, "depth_fusion": {"ema_momentum": 0.99}}


@pytest.mark.parametrize("mode", ["default", "gt_depth", "gt_pose"])
def test_pose_stage_emits_artifacts(tmp_path, mode):
    rec = ClipRecord(
        clip_id=f"clip_{mode}", source="dl3dv", video_path=str(tmp_path / "v.mp4"),
        mode=mode, num_frames=12, width=128, height=72,
    )
    out = annotate_pose(rec, tmp_path / "pose", MODELS)

    poses = np.load(out.pose_path)
    intr = np.load(out.intrinsics_path)
    assert poses.shape == (12, 4, 4)
    assert intr.shape == (12, 4)  # per-frame (fx,fy,cx,cy)
    assert len(out.scale_factors) == 12
    assert out.pose_mode == mode
    # scales are finite & positive
    assert all(np.isfinite(s) and s > 0 for s in out.scale_factors)


def test_gt_pose_recovers_known_scale(tmp_path):
    # dry-run gt_pose proxy: GT = 1.7 * predicted -> recovered scale ~ 1.7
    rec = ClipRecord(
        clip_id="gtp", source="sekai_game", video_path=str(tmp_path / "v.mp4"),
        mode="gt_pose", num_frames=20, width=128, height=72,
    )
    out = annotate_pose(rec, tmp_path / "pose", MODELS)
    assert out.scale_factors[0] == pytest.approx(1.7, rel=1e-3)

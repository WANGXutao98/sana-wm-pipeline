"""Tests for orchestrate.ray_pipeline — DAG plumbing only (no Ray cluster)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sana_wm_pipeline.orchestrate.ray_pipeline import (
    ClipJob,
    _SOURCE_TO_POSE_MODE,
    enumerate_jobs,
)


def test_pose_mode_dispatch_table():
    # Paper App. B.1: 3 modes mapped to specific sources.
    assert _SOURCE_TO_POSE_MODE["SpatialVID_HQ"] == "default"
    assert _SOURCE_TO_POSE_MODE["MiraData"] == "default"
    assert _SOURCE_TO_POSE_MODE["Sekai_Walking_HQ"] == "default"
    assert _SOURCE_TO_POSE_MODE["OmniWorld"] == "gtdepth"
    assert _SOURCE_TO_POSE_MODE["DL3DV"] == "gtpose"
    assert _SOURCE_TO_POSE_MODE["Sekai_Game"] == "gtpose"


def test_enumerate_jobs_smoke_yields_one_per_source_with_example():
    # Sources with local_path_example configured produce 1 job in smoke mode;
    # sources without it are skipped to avoid crashing on nonexistent /tmp sentinels.
    sources_cfg = {"sources": {
        "A": {"target_clips": 100, "local_path_example": "/tmp/a.mp4"},
        "B": {"target_clips": 50},   # no local_path_example — skipped in smoke
        "C": {"target_clips": 0, "local_path_example": "/tmp/c.mp4"},
    }}
    jobs = enumerate_jobs(sources_cfg, smoke=True)
    by_src = {j.source for j in jobs}
    assert by_src == {"A", "C"}
    assert len(jobs) == 2


def test_enumerate_jobs_smoke_skips_sources_without_path():
    sources_cfg = {"sources": {
        "NoPath": {"target_clips": 10},
    }}
    jobs = enumerate_jobs(sources_cfg, smoke=True)
    assert len(jobs) == 0


def test_enumerate_jobs_full_uses_target_clips():
    sources_cfg = {"sources": {
        "X": {"target_clips": 3},
        "Y": {"target_clips": 1},
    }}
    jobs = enumerate_jobs(sources_cfg, smoke=False)
    assert len(jobs) == 4
    assert sum(1 for j in jobs if j.source == "X") == 3
    assert sum(1 for j in jobs if j.source == "Y") == 1


def test_clipjob_default_pose_mode_fallback():
    # An unknown source maps to "default" rather than raising.
    sources_cfg = {"sources": {"NeverHeardOf": {"target_clips": 2}}}
    jobs = enumerate_jobs(sources_cfg, smoke=False)
    assert all(j.pose_mode == "default" for j in jobs)

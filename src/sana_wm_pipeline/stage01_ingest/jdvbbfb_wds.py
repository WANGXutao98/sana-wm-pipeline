"""Ingest adapter for the junchaoh-cs/jdvbbfb-v3-full WebDataset corpus.

Per-sample layout inside each shard tar:
  {key}.mp4          — RGB video (H264)
  {key}.camera.npz   — per_frame_camera_npz_v1 (GT c2w/K_px + vipe_* refs)
Caption text lives in <group>/index.jsonl  →  record["manifest"]["prompt"]["text"]
(json_members_in_shards=false, so prompts are NOT inside the tar).

This module holds only pure / unit-testable helpers. Network + HF download
glue lives in experiments/data_production_smoke/prepare_jdvbbfb.py.
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

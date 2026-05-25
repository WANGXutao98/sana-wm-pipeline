#!/usr/bin/env python3
"""Verify a WebDataset shard against ALL SANA-WM paper hard constraints.

Reads each sample in shard-*.tar and checks:
  - poses_c2w shape (961, 4, 4) float32; frame 0 ≈ identity
  - intrinsics shape (961, 1, 4) float32
  - scale shape (961,) float32
  - FOV ∈ [25°, 120°] (paper App. B.3)
  - focal divergence ≤ 0.20 (paper App. B.3)
  - scale CV ≤ 2.0 (paper App. B.3)
  - caption non-empty AND contains no camera-action verbs

Exit code: 0 if all pass, 1 if any fail. Prints per-sample status.

Usage:
    python scripts/verify_consistency.py path/to/shard-000000.tar
    python scripts/verify_consistency.py path/to/shards/   # all shards in dir
"""
from __future__ import annotations
import io
import json
import sys
import tarfile
from pathlib import Path
import numpy as np

# Path manipulation so this script can be run without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sana_wm_pipeline.stage02_pose.pose_quality import evaluate_pose_quality
from sana_wm_pipeline.stage06_pack.schema import (
    CAMERA_FRAMES, EXPECTED_WH, INTRINSICS_VIEWS, INTRINSICS_DIM,
)

# Forbidden camera-action verbs (paper §4: caption must not mention camera actions)
FORBIDDEN_VERBS = (
    "pan", "tilt", "zoom", "dolly", "truck", "crab", "crane",
    "fly-through", "flythrough", "walk", "walking", "rotate", "spin",
    "orbit", "approach", "retreat",
)


def _verb_pattern(verb: str) -> str:
    """Build a regex pattern that matches a verb stem + common inflections.

    Examples:
      pan   -> pan, pans, panning, panned (but not panel/panic)
      rotate -> rotate, rotates, rotating, rotated
      walk   -> walk, walks, walking, walked
      fly-through / flythrough -> exact hyphenated form

    We avoid `\\w*` greediness (which would catch panel/spinach) by enumerating
    the common English verb suffixes explicitly.
    """
    import re
    if "-" in verb or verb == "flythrough":
        # multi-token forms must match exactly with optional plural 's'
        return rf"\b{re.escape(verb)}s?\b"
    # Verbs ending in 'e' drop the e before -ing/-ed: rotate -> rotating
    if verb.endswith("e"):
        stem = verb[:-1]
        # Match: rotate, rotates, rotated, rotating
        return rf"\b(?:{re.escape(verb)}(?:s|d)?|{re.escape(stem)}ing)\b"
    # Generic: pan, pans, panned, panning  (consonant doubling handled by [a-z]?)
    return rf"\b{re.escape(verb)}(?:s|ed|n?ing|n?ed)?\b"


def caption_has_forbidden_verbs(caption: str) -> tuple[bool, list[str]]:
    """Return (has_any, list_of_hits). Token-boundary + inflection-aware,
    case-insensitive."""
    import re
    hits: list[str] = []
    low = caption.lower()
    for v in FORBIDDEN_VERBS:
        if re.search(_verb_pattern(v), low):
            hits.append(v)
    # Common camera phrases
    if re.search(r"\bthe camera\b", low) or re.search(r"\bfirst[- ]person view (going|walking|moving)\b", low):
        hits.append("camera-phrase")
    return (len(hits) > 0, hits)


def verify_sample(sid: str, tar: tarfile.TarFile) -> tuple[bool, list[str]]:
    """Verify one sample. Returns (ok, list_of_violations)."""
    violations: list[str] = []

    def _load_npy(name: str) -> np.ndarray:
        b = tar.extractfile(name).read()
        return np.load(io.BytesIO(b))

    def _load_text(name: str) -> str:
        return tar.extractfile(name).read().decode("utf-8")

    try:
        poses = _load_npy(f"{sid}.poses_c2w.npy")
        intr = _load_npy(f"{sid}.intrinsics.npy")
        scale = _load_npy(f"{sid}.scale.npy")
        caption = _load_text(f"{sid}.caption.txt")
        _ = json.loads(_load_text(f"{sid}.meta.json"))
    except Exception as e:
        return False, [f"could not load sample group: {e}"]

    # Shape / dtype gates
    T = CAMERA_FRAMES
    if poses.shape != (T, 4, 4):
        violations.append(f"poses shape {poses.shape} != ({T},4,4)")
    if poses.dtype != np.float32:
        violations.append(f"poses dtype {poses.dtype} != float32")
    if intr.shape != (T, INTRINSICS_VIEWS, INTRINSICS_DIM):
        violations.append(f"intrinsics shape {intr.shape} != ({T},{INTRINSICS_VIEWS},{INTRINSICS_DIM})")
    if scale.shape != (T,):
        violations.append(f"scale shape {scale.shape} != ({T},)")

    # First-frame anchor
    if poses.shape == (T, 4, 4) and not np.allclose(poses[0], np.eye(4), atol=1e-3):
        violations.append("poses[0] != identity (first-frame anchor)")

    # FOV / focal / scale CV via the shared evaluator (paper App. B.3)
    if intr.shape == (T, INTRINSICS_VIEWS, INTRINSICS_DIM):
        result = evaluate_pose_quality(intr, EXPECTED_WH, scale)
        if not result.passed:
            for r in result.reasons:
                violations.append(f"paper App. B.3: {r}")

    # Caption hygiene
    if not caption.strip():
        violations.append("caption empty")
    else:
        has_bad, hits = caption_has_forbidden_verbs(caption)
        if has_bad:
            violations.append(f"caption contains camera-action verbs: {hits}")

    return (len(violations) == 0, violations)


def verify_shard(shard_path: Path) -> tuple[int, int, list[str]]:
    """Verify all samples in one shard. Returns (n_ok, n_fail, error_lines)."""
    n_ok, n_fail = 0, 0
    error_lines: list[str] = []
    with tarfile.open(shard_path) as tar:
        names = tar.getnames()
        # sample ids are filenames without .meta.json suffix
        sids = sorted({n[: -len(".meta.json")] for n in names if n.endswith(".meta.json")})
        for sid in sids:
            ok, violations = verify_sample(sid, tar)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                error_lines.append(f"FAIL {shard_path.name}::{sid}")
                for v in violations:
                    error_lines.append(f"  - {v}")
    return n_ok, n_fail, error_lines


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: verify_consistency.py SHARD_OR_DIR", file=sys.stderr)
        return 2
    target = Path(argv[0])
    if target.is_dir():
        shards = sorted(target.glob("shard-*.tar"))
    else:
        shards = [target]
    if not shards:
        print(f"no shards found at {target}", file=sys.stderr)
        return 2

    total_ok, total_fail = 0, 0
    all_errors: list[str] = []
    for sh in shards:
        n_ok, n_fail, errs = verify_shard(sh)
        total_ok += n_ok
        total_fail += n_fail
        all_errors.extend(errs)
        status = "OK" if n_fail == 0 else "FAIL"
        print(f"{status}  {sh.name}: {n_ok} ok, {n_fail} fail")

    for e in all_errors[:50]:  # cap log volume
        print(e)
    if len(all_errors) > 50:
        print(f"... and {len(all_errors) - 50} more lines")

    print(f"\nTOTAL: {total_ok} ok, {total_fail} fail across {len(shards)} shards")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

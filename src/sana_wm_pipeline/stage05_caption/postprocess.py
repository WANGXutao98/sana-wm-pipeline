"""Caption post-processing — enforce paper §4 "no camera verbs" rule.

This module is the canonical home of the forbidden-verb regex used both by
the stage-05 caption runner (rejection retry) and by
`scripts/verify_consistency.py` (shard-level audit).
"""
from __future__ import annotations

import re
from typing import List, Tuple


# Forbidden verbs (paper §4) — must match `scripts/verify_consistency.py`.
FORBIDDEN_VERBS: Tuple[str, ...] = (
    "pan", "tilt", "zoom", "dolly", "truck", "crab", "crane",
    "fly-through", "flythrough", "walk", "walking", "rotate", "spin",
    "orbit", "approach", "retreat",
)


def _verb_pattern(verb: str) -> str:
    """Build an inflection-aware word-bounded regex for a verb stem.

    Examples:
      pan    -> pan, pans, panned, panning   (not panel/panic)
      rotate -> rotate, rotates, rotated, rotating
      walk   -> walk, walks, walked, walking
      fly-through / flythrough -> exact (+ optional plural 's')
    """
    if "-" in verb or verb == "flythrough":
        return rf"\b{re.escape(verb)}s?\b"
    if verb.endswith("e"):
        stem = verb[:-1]
        return rf"\b(?:{re.escape(verb)}(?:s|d)?|{re.escape(stem)}ing)\b"
    return rf"\b{re.escape(verb)}(?:s|ed|n?ing|n?ed)?\b"


_CAMERA_PHRASE_RE = re.compile(
    r"\bthe camera\b"
    r"|\bfirst[- ]person view (?:going|walking|moving)\b"
    r"|\bcamera (?:pans|tilts|zooms|moves|rotates)\b",
    flags=re.IGNORECASE,
)


def find_camera_verbs(caption: str) -> List[str]:
    """Return the list of forbidden hits (verb stems + 'camera-phrase')."""
    hits: List[str] = []
    low = caption.lower()
    for v in FORBIDDEN_VERBS:
        if re.search(_verb_pattern(v), low):
            hits.append(v)
    if _CAMERA_PHRASE_RE.search(low):
        hits.append("camera-phrase")
    return hits


def has_camera_verb(caption: str) -> bool:
    """True iff caption contains any forbidden camera-motion phrase."""
    return len(find_camera_verbs(caption)) > 0


def strip_offending_sentences(caption: str) -> str:
    """Drop sentences (split on .!?) that contain a forbidden phrase.

    Used as the fallback when the VLM keeps regenerating bad captions and we
    have exhausted retries.  Returns an empty string if every sentence is bad.
    """
    pieces = re.split(r"(?<=[.!?])\s+", caption.strip())
    keep = [s for s in pieces if s and not has_camera_verb(s)]
    return " ".join(keep)

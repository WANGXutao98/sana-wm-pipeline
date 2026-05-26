"""Tests for stage05_caption.postprocess + qwen35_vl_runner retry logic."""
from __future__ import annotations

import numpy as np
import pytest

from sana_wm_pipeline.stage05_caption.postprocess import (
    FORBIDDEN_VERBS,
    find_camera_verbs,
    has_camera_verb,
    strip_offending_sentences,
)
from sana_wm_pipeline.stage05_caption.qwen35_vl_runner import caption_clip


# ---- has_camera_verb / find_camera_verbs ----------------------------------
@pytest.mark.parametrize("bad", [
    "The camera pans left across the room.",
    "Camera tilts up to reveal the spire.",
    "A wide dolly shot follows the runner.",
    "First-person view walking through a dense forest.",
    "The drone is rotating around the centerpiece.",
    "An orbit around the statue at golden hour.",
    "Slow zoom on the painting reveals brushwork.",
])
def test_rejects_camera_verbs(bad):
    assert has_camera_verb(bad)


@pytest.mark.parametrize("good", [
    "A wooden table with two ceramic mugs sits under warm afternoon light.",
    "A serene alpine lake at dawn with snow-dusted peaks.",
    "An industrial workshop filled with brass tools and oily rags.",
    "A neon-lit alley with puddles reflecting signs in Korean and English.",
    "A medieval stone bridge spans a foggy ravine at dusk.",
])
def test_accepts_pure_scene(good):
    assert not has_camera_verb(good), find_camera_verbs(good)


def test_no_false_positive_on_substring():
    # 'transit' ⊃ 'sit'+'transit' should NOT trigger 'tilt', 'pan' etc.
    assert not has_camera_verb("a busy rail transit station with concrete platforms")
    # 'panel' should not trigger 'pan'
    assert not has_camera_verb("a chess board with elaborate wooden panels")
    # 'orbital' is a noun — but 'orbit' as a verb is the issue.  In English the
    # adjective 'orbital' technically contains the stem.  Our regex is verb
    # focused; we accept that 'orbital' is rare in scene captions and a small
    # over-rejection is preferable to under-rejection (paper requires strict).
    # So we do NOT assert 'orbital' is clean here.


def test_paper_verb_set_unchanged():
    # Mirror of paper §4 example list; if this changes we want the test to fail loudly.
    assert "pan" in FORBIDDEN_VERBS and "tilt" in FORBIDDEN_VERBS
    assert "zoom" in FORBIDDEN_VERBS and "dolly" in FORBIDDEN_VERBS
    assert "walking" in FORBIDDEN_VERBS and "rotate" in FORBIDDEN_VERBS


# ---- strip_offending_sentences --------------------------------------------
def test_strip_keeps_clean_sentences():
    cap = ("A wooden table with two ceramic mugs sits under afternoon light. "
           "The camera pans across the table.")
    out = strip_offending_sentences(cap)
    assert "wooden table" in out
    assert not has_camera_verb(out)


def test_strip_empty_when_all_bad():
    cap = "The camera pans left. The camera zooms in. Drone orbits."
    assert strip_offending_sentences(cap) == ""


# ---- caption_clip retry mechanism (DI'd generator) ------------------------
def test_caption_clip_returns_first_clean():
    frames = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    out = caption_clip(frames, generate_fn=lambda p, i: "a tranquil meadow at noon")
    assert out == "a tranquil meadow at noon"


def test_caption_clip_retries_until_clean():
    frames = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    attempts = iter([
        "The camera pans left across a meadow.",
        "First-person view walking through grass.",
        "a tranquil meadow at noon",
    ])

    def gen(prompt, images):
        return next(attempts)

    out = caption_clip(frames, max_retries=2, generate_fn=gen)
    assert out == "a tranquil meadow at noon"


def test_caption_clip_falls_back_to_strip():
    frames = np.zeros((20, 4, 4, 3), dtype=np.uint8)
    # All attempts are bad — runner must return the stripped fallback.
    out = caption_clip(
        frames, max_retries=1,
        generate_fn=lambda p, i: ("The camera pans left. "
                                  "A polished wooden floor reflects warm light."),
    )
    # The good sentence must survive and the bad one must be gone.
    assert "polished wooden floor" in out
    assert not has_camera_verb(out)

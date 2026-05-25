"""Shared test fixtures.

Auto-generates ``tests/fixtures/tiny_video_5s.mp4`` (a 5s 640x480@24fps
testsrc clip) before any test runs, so the binary is never committed.

Also prepends the project-local ``.bin/`` (where the static-ffmpeg binaries
are symlinked) to ``PATH`` so subprocess invocations of bare ``ffmpeg``/
``ffprobe`` resolve correctly on this host.
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOCAL_BIN = ROOT / ".bin"
FIXTURE = Path(__file__).parent / "fixtures" / "tiny_video_5s.mp4"

# Make ffmpeg/ffprobe discoverable for all subprocess calls in this session.
if LOCAL_BIN.is_dir():
    os.environ["PATH"] = f"{LOCAL_BIN}{os.pathsep}{os.environ.get('PATH', '')}"


@pytest.fixture(scope="session", autouse=True)
def ensure_tiny_video():
    if not FIXTURE.exists():
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=5:size=640x480:rate=24",
            "-loglevel", "error", str(FIXTURE),
        ])
    yield

"""Scene-cut counting via PySceneDetect (paper App. B.3 — used for the
`scene_cuts_max` Table-6 rule on MiraData / DL3DV_GS)."""
from __future__ import annotations

from typing import Optional


def count_scene_cuts(
    video_path: str,
    threshold: float = 27.0,
    detect_fn: Optional[object] = None,
) -> int:
    """Count detected scene transitions inside `video_path`.

    Uses PySceneDetect's ContentDetector by default. `detect_fn` can be
    injected with a callable `(video_path, threshold) -> int` for tests.
    """
    if detect_fn is not None:
        return int(detect_fn(video_path, threshold))

    from scenedetect import open_video, SceneManager  # type: ignore
    from scenedetect.detectors import ContentDetector  # type: ignore

    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    # PySceneDetect returns one scene per uninterrupted segment;
    # transitions == len(scenes) - 1 (a clip with no cuts -> 1 scene -> 0 cuts).
    return max(0, len(scenes) - 1)

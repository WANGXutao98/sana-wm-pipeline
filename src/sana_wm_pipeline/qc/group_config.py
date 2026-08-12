"""Per-group QC threshold config and Stage 1 verdict logic."""
from __future__ import annotations
from dataclasses import dataclass, field

VERDICT_PASS = "pass"
VERDICT_FLAG = "flag"
VERDICT_FAIL = "fail"

_HARD_FAIL_KEYS = ("t_aligned", "no_nan_inf", "so3_valid", "first_frame_ok", "pose_quality_ok")


@dataclass(frozen=True)
class GroupConfig:
    # Stage 1 pose thresholds
    jump_threshold_m: float
    max_jumps_flag: int       # n_jumps > this → FLAG
    max_jumps_fail: int       # n_jumps > this → FAIL
    min_caption_len: int
    # Stage 1 caption: check camera action words (True = flag if present)
    check_camera_words: bool = True
    # Stage 1 saturation: None = not checked
    saturation_min: float | None = None
    saturation_max: float | None = None
    # Stage 2: None = not checked
    max_scene_cuts: int | None = None
    # Stage 3 source key for filter_thresholds.yaml (None = skip Stage 3 table6)
    table6_source: str | None = None


# ── Registry ─────────────────────────────────────────────────────────────────

_REAL_STRICT = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=3, max_jumps_fail=50,
    min_caption_len=0,  # DL3DV has no captions
    check_camera_words=True,
    saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=None, table6_source="DL3DV",
)
_REALESTATE = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=3, max_jumps_fail=50,
    min_caption_len=50,
    check_camera_words=True,
    saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=1, table6_source="RealEstate10K",
)
_SEKAI_WALKING = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=3, max_jumps_fail=15,
    min_caption_len=50, saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=None, table6_source="Sekai_Walking",
)
_SPATIALVID = GroupConfig(
    jump_threshold_m=0.5, max_jumps_flag=0, max_jumps_fail=5,
    min_caption_len=50, saturation_min=0.0, saturation_max=180.0,
    max_scene_cuts=None, table6_source="SpatialVID",
)
_OMNIWORLD = GroupConfig(
    jump_threshold_m=2.0, max_jumps_flag=15, max_jumps_fail=50,
    min_caption_len=10,  # Game captions are shorter
    check_camera_words=False,  # Game captions contain camera words
    saturation_min=None, saturation_max=None,
    max_scene_cuts=None, table6_source="OmniWorld",
)
_SEKAI_DRONE = GroupConfig(
    jump_threshold_m=5.0, max_jumps_flag=20, max_jumps_fail=80,
    min_caption_len=10,  # Game captions are shorter
    check_camera_words=False,  # Game captions contain camera words
    saturation_min=None, saturation_max=None,
    max_scene_cuts=None, table6_source="Sekai_Game_Drone",
)
_SEKAI_GAME_WALKING = GroupConfig(
    jump_threshold_m=2.0, max_jumps_flag=15, max_jumps_fail=50,
    min_caption_len=10,  # Game captions are shorter
    check_camera_words=False,  # Game captions contain camera words
    saturation_min=None, saturation_max=None,
    max_scene_cuts=None, table6_source="Sekai_Game_Walking",
)
_DEFAULT = GroupConfig(
    jump_threshold_m=1.0, max_jumps_flag=10, max_jumps_fail=50,
    min_caption_len=50, table6_source=None,
)

_REGISTRY: dict[str, GroupConfig] = {
    "wds-DL3DV-ALL-2K": _REAL_STRICT,
    "wds-sekai-real-walking-hq": _SEKAI_WALKING,
    "wds-OmniWorld-Game": _OMNIWORLD,
    "wds-SpatialVID-hq": _SPATIALVID,
    "wds-RealEstate10K-360p": _REALESTATE,
    "wds-sekai-game-drone": _SEKAI_DRONE,
    "wds-sekai-game-walking": _SEKAI_GAME_WALKING,
}


def get_group_config(group_name: str) -> GroupConfig:
    return _REGISTRY.get(group_name, _DEFAULT)


def compute_verdict(metrics: dict, cfg: GroupConfig) -> tuple[str, list[str]]:
    """Classify Stage 1 metrics as PASS / FLAG / FAIL."""
    flag_reasons: list[str] = []

    # Hard structural failures
    for key in _HARD_FAIL_KEYS:
        if not metrics.get(key, True):
            return VERDICT_FAIL, list(metrics.get("reasons", []))
    # Jump count
    n_jumps = metrics.get("n_jumps", 0)
    if n_jumps > cfg.max_jumps_fail:
        return VERDICT_FAIL, [f"n_jumps={n_jumps} > max_jumps_fail={cfg.max_jumps_fail}"]
    if n_jumps > cfg.max_jumps_flag:
        flag_reasons.append(f"n_jumps={n_jumps} > max_jumps_flag={cfg.max_jumps_flag}")

    # Caption basic quality
    if not metrics.get("caption_ok", True):
        flag_reasons.extend([r for r in metrics.get("reasons", []) if "caption" in r])

    # Camera action words → flag for Qwen rewrite in Stage 3
    cw = metrics.get("camera_words", [])
    if cw and cfg.check_camera_words:
        flag_reasons.append(f"camera_word: {cw[0]!r} (+{len(cw)-1} more)")

    # Saturation (only for groups with saturation check, only flag not fail)
    sat = metrics.get("saturation")
    if sat is not None and cfg.saturation_min is not None:
        if not (cfg.saturation_min <= sat <= cfg.saturation_max):
            flag_reasons.append(f"saturation={sat:.1f} out of [{cfg.saturation_min},{cfg.saturation_max}]")

    return (VERDICT_FLAG, flag_reasons) if flag_reasons else (VERDICT_PASS, [])

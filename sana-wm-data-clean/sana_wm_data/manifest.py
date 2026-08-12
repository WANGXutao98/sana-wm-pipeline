"""Minimal camera-estimation record schema and JSONL helpers."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class CameraMetrics:
    """Camera-only quality-control quantities."""

    fov_x: float | None = None
    fov_y: float | None = None
    focal_div: float | None = None
    scale_cov: float | None = None


@dataclass
class ClipRecord:
    """One video and the camera artifacts produced for it."""

    clip_id: str
    source: str
    video_path: str
    mode: str = "default"
    fps: float | None = None
    num_frames: int | None = None
    width: int | None = None
    height: int | None = None
    pose_path: str | None = None
    intrinsics_path: str | None = None
    scale_factors: list[float] | None = None
    pose_mode: str | None = None
    camera: CameraMetrics = field(default_factory=CameraMetrics)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClipRecord":
        data = dict(value)
        camera = data.pop("camera", None) or {}
        extra = data.pop("extra", None) or {}
        known = {item.name for item in dataclasses.fields(cls)}
        for key in list(data):
            if key not in known:
                extra[key] = data.pop(key)
        rec = cls(**data)
        camera_fields = {item.name for item in dataclasses.fields(CameraMetrics)}
        rec.camera = CameraMetrics(**{k: v for k, v in camera.items() if k in camera_fields})
        rec.extra = extra
        return rec


def read_manifest(path: str | Path) -> list[ClipRecord]:
    return list(iter_manifest(path))


def iter_manifest(path: str | Path) -> Iterator[ClipRecord]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield ClipRecord.from_dict(json.loads(line))


def write_manifest(path: str | Path, records: Iterable[ClipRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    temporary.replace(destination)

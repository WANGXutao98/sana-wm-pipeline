"""WebDataset .tar shard writer for SANA-WM samples."""
from __future__ import annotations
import io
import json
import tarfile
from pathlib import Path
import numpy as np
from .schema import Sample


class ShardWriter:
    """Writes Samples to a sequence of .tar shards, rotating when capacity hit.

    Each shard is `shard-{shard_id:06d}.tar` in `out_dir`.
    """

    def __init__(self, out_dir: Path | str, samples_per_shard: int = 1000,
                 prefix: str = "shard", strict_frames: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if samples_per_shard < 1:
            raise ValueError(f"samples_per_shard must be >= 1; got {samples_per_shard}")
        self.samples_per_shard = samples_per_shard
        self.prefix = prefix
        self.shard_id = 0
        self.count_in_shard = 0
        self._tar: tarfile.TarFile | None = None
        self._strict_frames = strict_frames
        self._open_new_shard()

    @property
    def current_shard_path(self) -> Path:
        return self.out_dir / f"{self.prefix}-{self.shard_id:06d}.tar"

    def _open_new_shard(self) -> None:
        self._tar = tarfile.open(self.current_shard_path, "w")
        self.count_in_shard = 0

    def _add_bytes(self, name: str, data: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))

    def _add_npy(self, name: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        self._add_bytes(name, buf.getvalue())

    def write(self, sample: Sample) -> None:
        sample.validate(strict_frames=self._strict_frames)
        if self._tar is None:
            self._open_new_shard()
        sid = sample.sample_id
        # Video bytes
        vpath = Path(sample.video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"video_path missing: {vpath}")
        self._add_bytes(f"{sid}.mp4", vpath.read_bytes())
        # Arrays
        self._add_npy(f"{sid}.poses_c2w.npy", sample.poses_c2w)
        self._add_npy(f"{sid}.intrinsics.npy", sample.intrinsics_NVD)
        self._add_npy(f"{sid}.scale.npy", sample.scale_per_frame)
        # Text
        self._add_bytes(f"{sid}.caption.txt", sample.caption.encode("utf-8"))
        self._add_bytes(
            f"{sid}.meta.json",
            json.dumps(sample.meta, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
        self.count_in_shard += 1
        if self.count_in_shard >= self.samples_per_shard:
            self._rotate()

    def _rotate(self) -> None:
        assert self._tar is not None
        self._tar.close()
        self._tar = None
        self.shard_id += 1

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

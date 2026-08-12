"""run_worker 续跑判定集成测试（不触发 GPU 管线）。

只验证 process_input_shard 在 shard 已完成时直接跳过、不 import GPU 栈。
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

BP = Path(__file__).resolve().parents[1] / "experiments" / "batch_production"
sys.path.insert(0, str(BP))
import shard_io  # noqa: E402
import run_worker  # noqa: E402


def test_process_input_shard_skips_when_complete(tmp_path):
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()
    # 造一个"已完成"状态：.done + per-idx tar 都在
    (progress / "000003.done").write_text("{}")
    (worker_out / "shard-000003-000000.tar").write_bytes(b"x")

    called = {"opened": False}

    class _BoomWriter:  # 若被调用即说明没跳过
        def __init__(self, *a, **k): called["opened"] = True

    n_ok, n_fail = run_worker.process_input_shard(
        shard_path=tmp_path / "does-not-matter.tar",
        shard_idx=3, group="wds-x", captions={},
        tmp_dir=tmp_path / "tmp", worker_out=worker_out,
        progress_dir=progress, samples_per_shard=200,
        shard_writer_cls=_BoomWriter,
    )
    assert (n_ok, n_fail) == (0, 0)
    assert called["opened"] is False  # 完成的 shard 不应新建 writer


def test_process_input_shard_writes_and_marks_done(tmp_path, monkeypatch):
    """写路径测试：验证 per-idx prefix + mark_shard_done，不触发任何 GPU 代码。"""
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()

    # ── stub: iter_tar_samples 返回一条假样本 ──────────────────────────────────
    monkeypatch.setattr(
        "sana_wm_pipeline.stage01_ingest.jdvbbfb_wds.iter_tar_samples",
        lambda fobj, limit=None: iter([("key1", b"mp4bytes", b"cambytes")]),
    )

    # ── stub: normalize_video 写一个空文件，返回带 n_frames 的命名空间 ───────────
    def fake_normalize(raw, norm):
        norm.write_bytes(b"")
        return types.SimpleNamespace(n_frames=10)

    monkeypatch.setattr(
        "sana_wm_pipeline.stage01_ingest.normalize.normalize_video",
        fake_normalize,
    )

    # ── stub: run_default 返回假位姿 ───────────────────────────────────────────
    def fake_run_default(norm, work):
        return types.SimpleNamespace(
            poses_c2w=np.zeros((10, 4, 4), dtype=np.float32),
            intrinsics=np.zeros((10, 1, 4), dtype=np.float32),
            scale_per_frame=np.ones(10, dtype=np.float32),
        )

    monkeypatch.setattr(
        "sana_wm_pipeline.stage02_pose.mode_default.run_default",
        fake_run_default,
    )

    # ── RecordingWriter：记录 prefix、收集 write() 调用 ──────────────────────
    class RecordingWriter:
        def __init__(self, out_dir, *, samples_per_shard, prefix, strict_frames=False):
            self.prefix = prefix
            self.written = []
            # create a dummy tar so shard_is_complete can glob for it
            (out_dir / f"{prefix}-000000.tar").write_bytes(b"")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, sample):
            self.written.append(sample)

    # ── shard_path 必须存在（stub 忽略内容）──────────────────────────────────
    shard_file = tmp_path / "fake-000004.tar"
    shard_file.write_bytes(b"")

    recording_writer = None

    class CapturingWriter(RecordingWriter):
        def __init__(self, *a, **k):
            nonlocal recording_writer
            super().__init__(*a, **k)
            recording_writer = self

    n_ok, n_fail = run_worker.process_input_shard(
        shard_path=shard_file,
        shard_idx=4,
        group="wds-x",
        captions={"key1": "a caption"},
        tmp_dir=tmp_path / "tmp",
        worker_out=worker_out,
        progress_dir=progress,
        samples_per_shard=200,
        shard_writer_cls=CapturingWriter,
    )

    # 返回值
    assert (n_ok, n_fail) == (1, 0)

    # prefix 是 per-idx 格式
    assert recording_writer is not None
    assert recording_writer.prefix == "shard-000004"

    # 恰好写入了 1 条 sample
    assert len(recording_writer.written) == 1

    # .done 文件存在且 n_ok == 1
    done_file = progress / "000004.done"
    assert done_file.exists(), ".done 文件应存在"
    rec = json.loads(done_file.read_text())
    assert rec["n_ok"] == 1

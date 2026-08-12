"""Tests for batch_production shard_io resume/naming helpers."""
import json
import sys
from pathlib import Path

# shard_io 在 experiments/batch_production，不在安装包里
BP = Path(__file__).resolve().parents[1] / "experiments" / "batch_production"
sys.path.insert(0, str(BP))
import shard_io  # noqa: E402


def test_output_shard_prefix_zero_padded():
    assert shard_io.output_shard_prefix(0) == "shard-000000"
    assert shard_io.output_shard_prefix(8) == "shard-000008"
    assert shard_io.output_shard_prefix(123456) == "shard-123456"


def test_incomplete_when_no_done_and_no_tar(tmp_path):
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()
    assert shard_io.shard_is_complete(worker_out, progress, 5) is False


def test_incomplete_when_done_but_tar_missing(tmp_path):
    # 模拟旧截断 bug 遗留：.done 在，但没有 per-idx 输出 tar
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()
    (progress / "000005.done").write_text("{}")
    assert shard_io.shard_is_complete(worker_out, progress, 5) is False


def test_incomplete_when_tar_but_no_done(tmp_path):
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()
    (worker_out / "shard-000005-000000.tar").write_bytes(b"x")
    assert shard_io.shard_is_complete(worker_out, progress, 5) is False


def test_complete_when_done_and_tar_present(tmp_path):
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()
    (progress / "000005.done").write_text("{}")
    (worker_out / "shard-000005-000000.tar").write_bytes(b"x")
    assert shard_io.shard_is_complete(worker_out, progress, 5) is True


def test_done_idx_does_not_match_other_idx_tar(tmp_path):
    # 关键：idx=5 的 .done 不能被 idx=8 的 tar 满足（防旧共享命名误判）
    worker_out = tmp_path / "w000"; worker_out.mkdir()
    progress = tmp_path / "progress"; progress.mkdir()
    (progress / "000005.done").write_text("{}")
    (worker_out / "shard-000008-000000.tar").write_bytes(b"x")
    assert shard_io.shard_is_complete(worker_out, progress, 5) is False


def test_mark_shard_done_writes_json(tmp_path):
    progress = tmp_path / "progress"; progress.mkdir()
    shard_io.mark_shard_done(progress, 7, n_ok=3, n_fail=1, elapsed_s=12.5)
    rec = json.loads((progress / "000007.done").read_text())
    assert rec == {"shard_idx": 7, "n_ok": 3, "n_fail": 1, "elapsed_s": 12.5}

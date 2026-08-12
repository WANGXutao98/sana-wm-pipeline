"""纯函数：输出 shard 命名 + 续跑完成判定 + .done 标记写入。

不 import 任何 GPU/重负载依赖，可在无 GPU 机器上单测。
输出 tar 命名为 `shard-{input_idx:06d}-{part:06d}.tar`（part 由 ShardWriter 追加），
使 `.done` 标记与输出文件按输入 shard 下标 1:1 对应，续跑幂等、重启不丢数。
"""
from __future__ import annotations

import json
from pathlib import Path


def output_shard_prefix(shard_idx: int) -> str:
    """传给 ShardWriter(prefix=...)；writer 会追加 `-{part:06d}.tar`。"""
    return f"shard-{shard_idx:06d}"


def shard_is_complete(worker_out: Path, progress_dir: Path, shard_idx: int) -> bool:
    """输入 shard 完成 ⟺ .done 存在 且 对应 per-idx 输出 tar 存在。

    第二个条件让旧的「共享 shard-000000.tar」截断 bug 遗留的 .done 自动判为未完成，
    从而无需手动清空目录即可重算自愈。
    """
    done = progress_dir / f"{shard_idx:06d}.done"
    if not done.exists():
        return False
    prefix = output_shard_prefix(shard_idx)
    return any(worker_out.glob(f"{prefix}-*.tar"))


def mark_shard_done(progress_dir: Path, shard_idx: int,
                    n_ok: int, n_fail: int, elapsed_s: float) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / f"{shard_idx:06d}.done").write_text(json.dumps({
        "shard_idx": shard_idx,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "elapsed_s": elapsed_s,
    }))

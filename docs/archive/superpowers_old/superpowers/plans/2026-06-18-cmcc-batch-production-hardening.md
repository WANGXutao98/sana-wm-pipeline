# CMCC 多任务组批量生产加固 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 CMCC 集群上 `experiments/batch_production` 的两个生产事故（输出只有 `shard-000000.tar` 的数据丢失 bug、多节点单节点 CUDA 失效），并把脚本套件改造成「5 个独立任务组、每组带数据集名 + 自定义输出路径双参数、可断点续跑」的标准形态。

**Architecture:** 保持现有 embarrassingly-parallel 架构（worker `W = NODE_RANK×8 + LOCAL_GPU`，按全局 shard 下标 round-robin，无 NCCL）。核心改两处：(1) 输出 tar 命名从「全 worker 共享 `shard-000000.tar` 累加」改为「每个输入 shard 一个确定性命名的输出 tar」，让 `.done` 标记与输出文件 1:1 对应，根治重启截断丢数 + 让续跑真正幂等；(2) `launch_all_nodes.sh` 远程环境改为唯一真源 `source config.sh`，删除把 master 本机 `PATH/LD_LIBRARY_PATH` 注入远程的缺陷行，根治单节点 CUDA 失效。参数化：数据集名 + 输出路径双入参从 `launch_all_nodes.sh` 一路透传到 `run_worker.py`。

**Tech Stack:** Bash (set -euo pipefail)、Python 3.10、pytest、WebDataset(.tar)、SSH 编排、CMCC 自研 "VC" 调度器（无 SLURM/torchrun 自动 rank 注入）。

## Global Constraints

- 路径变量恒定：`NEW_BASE=/root/work/david_work`、`ENV_DIR=$NEW_BASE/sana_wm_env`、`PROJ_DIR=$NEW_BASE/sana_wm_pipeline`，远程节点镜像一致。
- 离线模式必须保留：`VIPE_EXT_JIT=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`（CMCC 无外网）。
- CUDA：远程非交互 SSH shell **必须** `unset CUDA_VISIBLE_DEVICES`（节点镜像把它预置为空串=0卡可见，见 progress.md 会话7）。
- worker 输出目录命名 `w{NODE_RANK*8+LOCAL_GPU:03d}`，不含 job 标识 → 不同任务组**必须**写到不同 `OUT_BASE`，靠双参数中的「自定义输出路径」天然隔离。
- 所有 bash 脚本顶部 `set -euo pipefail`（`watch_progress.sh` 例外，注释已说明）。
- 测试运行器：`cd $PROJ_DIR && python -m pytest`（`pyproject.toml` testpaths=["tests"]）。
- 重负载 GPU 依赖（Pi3X/MoGe-2/vipe）在测试机不可用 → 续跑/打包逻辑必须抽成**不 import GPU 栈**的纯函数才能单测。

---

## File Structure

| 文件 | 职责 | 改动类型 |
|------|------|---------|
| `src/sana_wm_pipeline/stage06_pack/webdataset_writer.py` | `.tar` 写出 | 不改（已支持自定义 `prefix`，沿用）|
| `experiments/batch_production/shard_io.py` | **新增** 纯函数模块：续跑判定 `shard_is_complete()` + 输出 tar 命名 `output_shard_prefix()` + 写 `.done` `mark_shard_done()`。无 GPU import，可单测 | 新建 |
| `experiments/batch_production/run_worker.py` | 单 GPU worker | 改：每输入 shard 独立输出 tar + 调 `shard_io` 续跑判定 |
| `experiments/batch_production/config.sh` | 路径/环境集中配置 | 改：`OUT_BASE`/`DATA_ROOT` 可被外部环境覆盖；删 `CUDA_HOME=torch/lib` 误配；移除调试 echo |
| `experiments/batch_production/launch_single_node.sh` | 单节点 8 卡启动 | 改：移除热路径里的 `pip install`；进度统计用新 tar 命名 |
| `experiments/batch_production/run_groups_sequential.sh` | 串行跑 group | 改：保留 `--groups`，确认透传 `OUT_BASE` |
| `experiments/batch_production/launch_all_nodes.sh` | 多节点编排总入口 | **重写**：新双参数签名；远程统一 `source config.sh`；删 master env 注入行；pip 安装移到预检串行执行一次 |
| `experiments/batch_production/stop_all_nodes.sh` | 多节点一键停止 | 改：支持多 hostfile；加固 pattern |
| `experiments/batch_production/watch_progress.sh` | 监控面板 | 改：接受 `GROUP` + 可选 `OUT_BASE` 参数 |
| `tests/test_shard_io.py` | **新增** `shard_io.py` 单测 | 新建 |
| `experiments/batch_production/README.md` | 标准输出目录规范 + 5 组启动模板 + 迁移说明 | 新建 |

---

## 根因诊断（写入 README，先在此固定结论）

### 问题1：输出只有 `shard-000000.tar` —— 重启截断丢数（Critical）

`webdataset_writer.py:36` `_open_new_shard()` 用 `tarfile.open(path, "w")` 打开 `w{wid}/shard-000000.tar`，**"w" 模式会截断**。`run_worker.py` 旧逻辑用**一个**贯穿所有输入 shard 的 `ShardWriter`，`shard_id` 永远从 0 起。worker 每次重启（OOM、节点被踢、`gg` 保活抢显存、两天里必然多次）：

1. `ShardWriter.__init__` → 截断 `shard-000000.tar`，**抹掉上次已写入的样本**；
2. `process_input_shard` 跳过所有有 `.done` 的输入 shard（这些样本不会重算）；
3. 净结果：被截断掉的样本永久丢失，且 `.done` 已存在不再补算。

两天 + 多次重启 ⇒ 每个 worker 只剩「最后一段未重启期间」的少量样本，且 200 样本/输出 shard 的阈值远未触达，所以恒为单个 `shard-000000.tar`。**这不是慢，是每次重启都在毁数据，续跑根本没生效。**

> `w000~w031`（仅 32 个而非 64）是问题2 的连带结果：`launch_all_nodes.sh` 预检「自动剔除坏点」把 CUDA 失效节点踢出，`NUM_NODES` 缩水、rank 重排稠密化 → 只生成存活节点对应的 worker 目录（4 节点存活 = w000~w031）。

**修复：** 输出 tar 改为按输入 shard 下标确定性命名 `shard-{input_idx:06d}-{part:06d}.tar`（每输入 shard 独立 writer，`prefix=f"shard-{idx:06d}"`），`.done` 与输出 tar 1:1。续跑判定升级为「`.done` 存在 **且** 对应 `shard-{idx:06d}-*.tar` 存在」才算完成——旧的全部 `.done`（共享命名，无 per-idx tar）会自动判为未完成并重算，**无需手动清空目录即可自愈两天的损坏产出**。

### 问题2：多节点恒有一台 CUDA 不可用 —— master 环境注入污染远程（Critical）

`launch_all_nodes.sh` 预检(79-80)与拉起(167-168)的 SSH 负载里：

```bash
export PATH="${PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
```

SSH 负载是**外层双引号字符串**，`${PATH:-}`/`${LD_LIBRARY_PATH:-}` 由 **master 本机** shell 先展开再发往远程 ⇒ 把 master 的 `LD_LIBRARY_PATH`（可能含系统 Python3.12 torch，正是 findings F-0 的 `undefined symbol` 根因）强行 export 到远程，污染/抢在 env libtorch 之前。该值随「你从哪台机器发起」而变，表现就是「source 不生效 / 环境没完全覆盖 / 稳定有一台跟 master 环境不合的节点挂」。

此外预检只 `source activate_sana_wm.sh`，**没 source `config.sh`**，与真实生产路径（`run_groups_sequential.sh`→`config.sh`）环境不一致；`launch_single_node.sh:40` 还在热路径里 `pip install -e`，多节点并发对共享 `$PROJ_DIR` 装同一个 editable 包会竞态损坏 import。

**修复：** ① 删掉两处 master env 注入行；② 远程统一 `source config.sh`（唯一真源，含 activate + alloc + offline + 后续 OUT_BASE 覆盖）后 `unset CUDA_VISIBLE_DEVICES`；③ `pip install -e` 移到预检阶段**串行**每节点跑一次，从 worker 热路径删除。

---

## 标准输出目录规范（写入 README）

```
$OUT_BASE/                         # = 启动参数2「自定义输出路径」，每个数据集组独占一个
└── <group>/                       # 如 wds-sekai-real-walking-hq
    ├── w000/                      # worker 0 输出（NODE_RANK*8+LOCAL_GPU）
    │   ├── shard-000000-000000.tar   # 输入 shard 0 → 输出（part 0）
    │   ├── shard-000008-000000.tar   # 输入 shard 8（round-robin 步长=全局worker数）
    │   └── ...
    ├── w001/ ... w0NN/
    ├── progress/
    │   ├── 000000.done            # JSON: {shard_idx,n_ok,n_fail,elapsed_s}
    │   └── ...                    # 每个"已完成输入 shard"一个标记
    ├── logs/
    │   ├── node0_main_<ts>.log    # 单节点总日志
    │   └── node0_gpu0.log ...     # 每 worker 日志
    └── driver_logs/               # （在 OUT_BASE 根，非 group 下）launch_all_nodes 各节点 driver 日志
```

判完成：输入 shard `idx` 完成 ⟺ `progress/{idx:06d}.done` 存在 **且** 某个 `w*/shard-{idx:06d}-*.tar` 存在。

---

### Task 1: `shard_io.py` 纯函数模块 + 单测（续跑判定 / 命名 / 标记）

**Files:**
- Create: `experiments/batch_production/shard_io.py`
- Test: `tests/test_shard_io.py`

**Interfaces:**
- Consumes: 无（stdlib only：`json`, `pathlib`, `time`）。
- Produces（供 Task 2 `run_worker.py` 调用）：
  - `output_shard_prefix(shard_idx: int) -> str` → `f"shard-{shard_idx:06d}"`（传给 `ShardWriter(prefix=...)`，writer 会再追加 `-{part:06d}.tar`）
  - `shard_is_complete(worker_out: Path, progress_dir: Path, shard_idx: int) -> bool` → `.done` 存在 **且** `worker_out/shard-{shard_idx:06d}-*.tar` glob 非空
  - `mark_shard_done(progress_dir: Path, shard_idx: int, n_ok: int, n_fail: int, elapsed_s: float) -> None` → 写 `{idx:06d}.done`（JSON）

- [ ] **Step 1: 写失败测试** `tests/test_shard_io.py`

```python
"""Tests for batch_production shard_io resume/naming helpers."""
import json
import sys
from pathlib import Path
import pytest

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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $PROJ_DIR && python -m pytest tests/test_shard_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shard_io'`

- [ ] **Step 3: 写实现** `experiments/batch_production/shard_io.py`

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $PROJ_DIR && python -m pytest tests/test_shard_io.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: 提交**

```bash
git add experiments/batch_production/shard_io.py tests/test_shard_io.py
git commit -m "feat(batch): add shard_io resume helpers (per-input-shard tar naming)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `run_worker.py` 改为每输入 shard 独立输出 tar + 自愈续跑

**Files:**
- Modify: `experiments/batch_production/run_worker.py`
- Test: `tests/test_run_worker_resume.py`（新增；只测可在无 GPU 下跑的续跑分支）

**Interfaces:**
- Consumes: `shard_io.output_shard_prefix / shard_is_complete / mark_shard_done`（Task 1）；`ShardWriter(out_dir, samples_per_shard, prefix, strict_frames)`（不变）。
- Produces: CLI 不变（`--group --data-root --out-base --worker-id --shard-indices --samples-per-shard`）。输出文件命名变为 `w{wid}/shard-{input_idx:06d}-{part:06d}.tar`。

**关键改动说明：** 删除「整个 worker 一个 ShardWriter」的写法，改为 `for shard_idx: 先 shard_is_complete 判跳过 → 否则为该输入 shard 新建 ShardWriter(prefix=output_shard_prefix(idx)) → 处理完 close → mark_shard_done`。`samples_per_shard` 仍生效（单输入 shard 样本数超阈值会自动滚动 part）。

- [ ] **Step 1: 写失败测试** `tests/test_run_worker_resume.py`

```python
"""run_worker 续跑判定集成测试（不触发 GPU 管线）。

只验证 process_input_shard 在 shard 已完成时直接跳过、不 import GPU 栈。
"""
import sys
from pathlib import Path

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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd $PROJ_DIR && python -m pytest tests/test_run_worker_resume.py -v`
Expected: FAIL — `process_input_shard` 签名不含 `worker_out/shard_writer_cls`（TypeError）

- [ ] **Step 3: 改写 `run_worker.py`**

完整替换 `process_input_shard` 与 `main` 中调用处。新 `process_input_shard`：

```python
def process_input_shard(
    shard_path: Path,
    shard_idx: int,
    group: str,
    captions: dict[str, str],
    tmp_dir: Path,
    worker_out: Path,
    progress_dir: Path,
    samples_per_shard: int,
    shard_writer_cls=None,
) -> tuple[int, int]:
    import shard_io

    if shard_io.shard_is_complete(worker_out, progress_dir, shard_idx):
        print(f"[SKIP] shard {shard_idx:06d} 已完成（.done + tar 均在），跳过")
        return 0, 0

    if shard_writer_cls is None:
        from sana_wm_pipeline.stage06_pack.webdataset_writer import ShardWriter
        shard_writer_cls = ShardWriter

    from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import iter_tar_samples
    from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
    from sana_wm_pipeline.stage02_pose.mode_default import run_default
    from sana_wm_pipeline.stage06_pack.schema import Sample

    STUB = "A real-world scene captured by a moving camera."
    n_ok = n_fail = 0
    t0_shard = time.time()

    # 该输入 shard 独占一组确定性命名的输出 tar：shard-{idx:06d}-{part:06d}.tar
    prefix = shard_io.output_shard_prefix(shard_idx)
    with shard_writer_cls(worker_out, samples_per_shard=samples_per_shard,
                          prefix=prefix, strict_frames=False) as writer:
        with open(shard_path, "rb") as fobj:
            for key, mp4_bytes, camera_bytes in iter_tar_samples(fobj, limit=None):
                sample_tmp = tmp_dir / key
                sample_tmp.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                try:
                    raw_video = sample_tmp / "video.mp4"
                    raw_video.write_bytes(mp4_bytes)
                    norm_video = sample_tmp / "normalized.mp4"
                    info = normalize_video(raw_video, norm_video)
                    raw_video.unlink()
                    vipe_work = sample_tmp / "vipe_work"
                    art = run_default(norm_video, vipe_work)
                    caption = captions.get(key, "").strip() or STUB
                    sample = Sample(
                        sample_id=key,
                        video_path=str(norm_video),
                        poses_c2w=art.poses_c2w,
                        intrinsics_NVD=art.intrinsics,
                        scale_per_frame=art.scale_per_frame,
                        caption=caption,
                        meta={
                            "scene_id": key,
                            "T": int(art.poses_c2w.shape[0]),
                            "mode": "default",
                            "dataset": "jdvbbfb-v3-full",
                            "group": group,
                            "source_shard": shard_path.name,
                        },
                    )
                    writer.write(sample)
                    n_ok += 1
                    print(f"  [OK]   {key}  T={info.n_frames}  {time.time()-t0:.0f}s")
                except Exception as exc:
                    n_fail += 1
                    print(f"  [FAIL] {key}: {exc}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                finally:
                    shutil.rmtree(sample_tmp, ignore_errors=True)

    shard_io.mark_shard_done(progress_dir, shard_idx, n_ok, n_fail,
                             round(time.time() - t0_shard, 1))
    print(f"[shard {shard_idx:06d}] DONE  ok={n_ok}  fail={n_fail}  "
          f"{time.time()-t0_shard:.0f}s")
    return n_ok, n_fail
```

`main()` 中：删除原 `with ShardWriter(...) as writer:` 包裹整个循环的写法，改为循环内调用：

```python
    total_ok = total_fail = 0
    for shard_idx in shard_indices:
        shard_path = shard_dir / _shard_basename(args.group, shard_idx)
        if not shard_path.exists():
            print(f"[WARN] shard not found: {shard_path}，跳过")
            continue
        n_ok, n_fail = process_input_shard(
            shard_path, shard_idx, args.group, captions, tmp_dir,
            worker_out, progress_dir, args.samples_per_shard,
        )
        total_ok += n_ok
        total_fail += n_fail

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n[worker {args.worker_id}] 完成  ok={total_ok}  fail={total_fail}")
    sys.exit(1 if total_fail > 0 and total_ok == 0 else 0)
```

（`main` 顶部 `worker_out`/`progress_dir`/`tmp_dir`/`captions` 计算不变；删除原 `from ... import ShardWriter` 顶层那行，改由 `process_input_shard` 内部按需 import。）

- [ ] **Step 4: 跑测试确认通过**

Run: `cd $PROJ_DIR && python -m pytest tests/test_run_worker_resume.py tests/test_shard_io.py -v`
Expected: PASS

- [ ] **Step 5: 语法自检 + 提交**

```bash
cd $PROJ_DIR && python -c "import ast; ast.parse(open('experiments/batch_production/run_worker.py').read()); print('OK')"
git add experiments/batch_production/run_worker.py tests/test_run_worker_resume.py
git commit -m "fix(batch): per-input-shard output tar to stop restart truncation data loss

Each input shard now writes shard-{idx}-{part}.tar; .done is validated
against the per-idx tar so legacy truncated outputs auto-reprocess.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `config.sh` —— 输出/数据路径可覆盖 + 清理 CUDA 误配

**Files:**
- Modify: `experiments/batch_production/config.sh`
- Test: `tests/test_config_overridable.sh`（新增 bash 断言脚本）

**Interfaces:**
- Produces: `source config.sh` 后 `OUT_BASE`/`DATA_ROOT` 若调用前已 export 则保留外部值，否则用默认；保留 `BATCH1_GROUPS`/`SAMPLES_PER_OUTPUT_SHARD`/offline flags/alloc conf。

- [ ] **Step 1: 写失败测试** `tests/test_config_overridable.sh`

```bash
#!/bin/bash
# 断言 config.sh 尊重外部预设的 OUT_BASE / DATA_ROOT
set -euo pipefail
CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../experiments/batch_production" && pwd)/config.sh"

# 屏蔽 activate（测试机无该环境）：用一个假的 NEW_BASE 指向临时空 activate
TMP="$(mktemp -d)"; mkdir -p "$TMP/sana_wm_env/bin"
printf '#!/bin/bash\n' > "$TMP/activate_sana_wm.sh"
printf '#!/bin/bash\necho 13.0\n' > "$TMP/sana_wm_env/bin/python"  # 假 python -c 的输出
chmod +x "$TMP/sana_wm_env/bin/python"

export NEW_BASE="$TMP"
export OUT_BASE="/custom/out/path"
export DATA_ROOT="/custom/data/path"
# shellcheck disable=SC1090
source "$CFG" >/dev/null 2>&1 || true

[[ "$OUT_BASE" == "/custom/out/path" ]]   || { echo "FAIL: OUT_BASE 被覆盖成 $OUT_BASE"; exit 1; }
[[ "$DATA_ROOT" == "/custom/data/path" ]] || { echo "FAIL: DATA_ROOT 被覆盖成 $DATA_ROOT"; exit 1; }
echo "PASS: config.sh 尊重外部 OUT_BASE / DATA_ROOT"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `bash tests/test_config_overridable.sh`
Expected: FAIL: OUT_BASE 被覆盖成 /root/work/externalstorage/...（当前 `export OUT_BASE=...` 硬覆盖）

- [ ] **Step 3: 改 `config.sh`**

把硬编码 `export` 改为 `:=` 默认赋值；删除 `CUDA_HOME=torch/lib` 误配与调试 echo。完整新版：

```bash
#!/bin/bash
# ── 基础路径 ──────────────────────────────────────────────────────────────────
export NEW_BASE="${NEW_BASE:-/root/work/david_work}"
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"

# DATA_ROOT / OUT_BASE 可被启动脚本（双参数）外部 export 覆盖，未设时用默认
export DATA_ROOT="${DATA_ROOT:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full}"
export OUT_BASE="${OUT_BASE:-/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output}"

# ── 模型权重路径 ──────────────────────────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"

# ── conda 环境激活（唯一真源；远程 SSH 也只 source 本文件）────────────────────
source "$NEW_BASE/activate_sana_wm.sh"

# ── 离线模式（CMCC 无外网）────────────────────────────────────────────────────
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── 显存优化：减少 CUDA allocator 碎片 ────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 批次 1 优先 group（仅在不传 --groups 时作默认队列）────────────────────────
BATCH1_GROUPS=(
    "wds-sekai-real-walking-hq"
    "wds-DL3DV-ALL-2K"
    "wds-SpatialVID-hq"
)
export BATCH1_GROUPS

# 每个输出 shard 最多样本数
export SAMPLES_PER_OUTPUT_SHARD=200
```

> 删除原 40-46 行：`CUDA_HOME=$ENV_DIR/.../torch/lib`（把 torch/lib 当 CUDA_HOME 是误配，且 `LD_LIBRARY_PATH` 由 `activate_sana_wm.sh` 统一管理）与两行调试 echo（会污染 SSH 返回、干扰解析）。

- [ ] **Step 4: 跑测试确认通过**

Run: `bash tests/test_config_overridable.sh && bash -n experiments/batch_production/config.sh`
Expected: `PASS: config.sh 尊重外部 OUT_BASE / DATA_ROOT`

- [ ] **Step 5: 提交**

```bash
git add experiments/batch_production/config.sh tests/test_config_overridable.sh
git commit -m "fix(batch): make OUT_BASE/DATA_ROOT overridable, drop bad CUDA_HOME hack

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `launch_single_node.sh` —— 移除热路径 pip 安装 + 对齐新 tar 命名统计

**Files:**
- Modify: `experiments/batch_production/launch_single_node.sh`

**Interfaces:**
- Consumes: `config.sh`（含覆盖后的 `OUT_BASE`/`DATA_ROOT`）、`run_worker.py`。
- Produces: 给 `run_worker.py` 传 `--out-base "$OUT_BASE"`（不变）；不再在本脚本内 `pip install`。

- [ ] **Step 1: 删除热路径 pip 安装段**

删除第 38-41 行（`echo "=== 安装最新sana_wm_pipeline ==="` 与 `pip install --no-user -e "$PROJ_DIR" ...`）。安装改由 `launch_all_nodes.sh` 预检阶段每节点串行执行一次（Task 5）。单节点手动调试时用户已在 smoke 流程装过，README 会注明手动单节点入口需先 `pip install -e $PROJ_DIR --no-deps`。

- [ ] **Step 2: 环境验证段保留**（第 44-51 行 `python -c "import ...; cuda=..."`）作为启动前自检，不动。

- [ ] **Step 3: 进度统计无需改**

`DONE_CNT=$(ls "$OUT_BASE/$GROUP/progress/"*.done ...)` 仍按 `.done` 计数，命名未变，正确。保留 `|| true` 防空目录。

- [ ] **Step 4: 语法检查 + 干跑模拟**

```bash
cd $PROJ_DIR && bash -n experiments/batch_production/launch_single_node.sh && echo "syntax OK"
```

模拟分片分配（无需 GPU，stub 掉 run_worker / config）：

```bash
# 用临时 stub 验证 round-robin：NUM_GPUS=2 NUM_NODES=2 NODE_RANK=1 → worker 2,3
cd $(mktemp -d) && mkdir -p shards && for i in $(seq -w 0 9); do : > "shards/g-0000$i.tar" 2>/dev/null || touch "shards/g-00000$i.tar"; done
echo "（人工核对：GLOBAL_WORKER=NODE_RANK*NUM_GPUS+LOCAL_GPU，步长=NUM_NODES*NUM_GPUS）"
```

Expected: `syntax OK`

- [ ] **Step 5: 提交**

```bash
git add experiments/batch_production/launch_single_node.sh
git commit -m "fix(batch): drop pip install from worker hot path (avoid concurrent editable-install race)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `run_groups_sequential.sh` —— 确认单数据集 + OUT_BASE 透传

**Files:**
- Modify: `experiments/batch_production/run_groups_sequential.sh`

**Interfaces:**
- Consumes: `config.sh`、`launch_single_node.sh`。`--groups <DATASET>` 指定单一数据集（5 组架构下每次只跑一个）。
- Produces: 调 `launch_single_node.sh "$GROUP" "$NODE_RANK" "$NUM_NODES"`，`OUT_BASE` 经环境继承（远程已 export，config.sh `:=` 保留）。

- [ ] **Step 1: 确认现有 `--groups` 路径正确**

现脚本已支持 `--groups G1,G2,...`。5 组架构下 `launch_all_nodes.sh` 会传 `--groups <单个dataset>`。`OUT_BASE` 由远程 SSH 在 `source config.sh` 前 export，config.sh `:=` 保留，`launch_single_node.sh` 读到正确值。**无需逻辑改动**，仅在头部用法注释补充双参数说明：

```bash
# 用法: bash run_groups_sequential.sh [--batch1-only | --groups G1,G2,...] [NODE_RANK=0] [NUM_NODES=1]
# OUT_BASE/DATA_ROOT 通过调用前 export 覆盖（见 launch_all_nodes.sh 双参数透传）。
```

- [ ] **Step 2: 语法检查**

Run: `cd $PROJ_DIR && bash -n experiments/batch_production/run_groups_sequential.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add experiments/batch_production/run_groups_sequential.sh
git commit -m "docs(batch): clarify OUT_BASE/DATA_ROOT override in run_groups_sequential header

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `launch_all_nodes.sh` 重写 —— 双参数 + 远程 config.sh 唯一真源 + 修 CUDA

**Files:**
- Modify (full rewrite): `experiments/batch_production/launch_all_nodes.sh`
- Test: `tests/test_launch_all_nodes_parse.sh`（新增；mock `ssh`/`scp` 验证参数解析与 rank 分配，不真连节点）

**Interfaces:**
- 新签名：`bash launch_all_nodes.sh [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>`
  - `<DATASET>`：jdvbbfb-v3-full 内的 group 名（如 `wds-sekai-real-walking-hq`）→ 透传为 `run_groups_sequential.sh --groups <DATASET>`
  - `<OUT_PATH>`：本次自定义输出根 → 远程 `export OUT_BASE=<OUT_PATH>`
  - `<HOSTFILE>`：本任务组独占的 hostfile
- 远程环境：仅 `source config.sh`（已含 activate+offline+alloc）→ `unset CUDA_VISIBLE_DEVICES`；**不再**注入 master `PATH/LD_LIBRARY_PATH`。
- 预检阶段每节点**串行** `pip install -e $PROJ_DIR --no-deps --no-build-isolation -q` 各一次。
- rank = 存活节点稠密下标，`NUM_NODES` = 存活节点数。

- [ ] **Step 1: 写失败测试** `tests/test_launch_all_nodes_parse.sh`

```bash
#!/bin/bash
# 用 PATH 前置 mock 掉 ssh/scp，验证 launch_all_nodes.sh 的参数解析 + rank 拉起命令
set -euo pipefail
BP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../experiments/batch_production" && pwd)"
WORK="$(mktemp -d)"

# mock ssh：预检永远成功(回显 OK)，拉起阶段把远程命令记录到文件
cat > "$WORK/ssh" <<'EOF'
#!/bin/bash
# 末参是远程命令串
CMD="${@: -1}"
echo "$CMD" >> "$MOCK_SSH_LOG"
# pgrep 计数返回 0；其余回显成功
case "$CMD" in
  *pgrep*) echo 0 ;;
  *) echo "[OK] mock" ;;
esac
EOF
chmod +x "$WORK/ssh"
cp "$WORK/ssh" "$WORK/scp"   # scp 同样吞掉

# 假 config.sh：不 source activate，只设最小变量
cat > "$WORK/config.sh" <<EOF
export NEW_BASE=/fake/base
export ENV_DIR=/fake/base/sana_wm_env
export PROJ_DIR=/fake/base/sana_wm_pipeline
export DATA_ROOT="\${DATA_ROOT:-/fake/data}"
export OUT_BASE="\${OUT_BASE:-/fake/out}"
EOF

# hostfile：3 个有效节点
printf 'nodeA slots=8\nnodeB slots=8\nnodeC slots=8\n' > "$WORK/hostfile"

export MOCK_SSH_LOG="$WORK/ssh.log"; : > "$MOCK_SSH_LOG"
# 用 mock config 覆盖：把 launch_all_nodes 里的 source 指到我们的假 config
SCRIPT="$WORK/launch_all_nodes.sh"
sed "s#source \"\$SCRIPT_DIR/config.sh\"#source \"$WORK/config.sh\"#" \
    "$BP/launch_all_nodes.sh" > "$SCRIPT"

PATH="$WORK:$PATH" bash "$SCRIPT" wds-sekai-real-walking-hq /my/out "$WORK/hostfile" \
    > "$WORK/out.log" 2>&1 || true

# 断言1：拉起命令含 --groups wds-sekai-real-walking-hq
grep -q -- "--groups wds-sekai-real-walking-hq" "$MOCK_SSH_LOG" \
    || { echo "FAIL: 未透传 --groups dataset"; cat "$WORK/out.log"; exit 1; }
# 断言2：远程 export OUT_BASE=/my/out
grep -q "OUT_BASE=/my/out\|OUT_BASE='/my/out'" "$MOCK_SSH_LOG" \
    || { echo "FAIL: 未 export 自定义 OUT_BASE"; exit 1; }
# 断言3：3 节点 → 出现 rank 0/1/2 且 NUM_NODES=3
grep -q "run_groups_sequential.sh --groups wds-sekai-real-walking-hq 0 3" "$MOCK_SSH_LOG" \
    || { echo "FAIL: rank0/NUM_NODES=3 拉起命令不对"; grep run_groups "$MOCK_SSH_LOG"; exit 1; }
grep -q " 2 3$" "$MOCK_SSH_LOG" \
    || { echo "FAIL: rank2 拉起命令不对"; exit 1; }
# 断言4：不得把 master 的 LD_LIBRARY_PATH 注入远程
grep -q 'export LD_LIBRARY_PATH="' "$MOCK_SSH_LOG" \
    && { echo "FAIL: 仍在向远程注入 master LD_LIBRARY_PATH"; exit 1; }
# 断言5：远程必须 unset CUDA_VISIBLE_DEVICES
grep -q "unset CUDA_VISIBLE_DEVICES" "$MOCK_SSH_LOG" \
    || { echo "FAIL: 远程未 unset CUDA_VISIBLE_DEVICES"; exit 1; }
echo "PASS: launch_all_nodes 参数解析 + rank + 环境修复 全部正确"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `bash tests/test_launch_all_nodes_parse.sh`
Expected: FAIL（旧脚本签名是 `[flags] <HOSTFILE>`，无 DATASET/OUT_PATH 位置参数，且仍注入 LD_LIBRARY_PATH）

- [ ] **Step 3: 重写 `launch_all_nodes.sh`**（完整文件）

```bash
#!/bin/bash
# 多节点批量生产总入口（CMCC VC 调度器，无 SLURM/torchrun 自动 rank）
#
# 用法:
#   bash launch_all_nodes.sh [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>
# 示例:
#   bash launch_all_nodes.sh wds-sekai-real-walking-hq \
#        /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_sekai_realwalking_hq \
#        /root/work/david_work/sana-wm-48-hostfiles/hostfile
#
# 设计要点（见 docs/.../2026-06-18-cmcc-batch-production-hardening.md 根因诊断）：
#   - 远程环境唯一真源 = source config.sh（含 activate + offline + alloc），
#     绝不向远程注入 master 本机 PATH/LD_LIBRARY_PATH（旧 bug：单节点 CUDA 失效）
#   - source config.sh 后 unset CUDA_VISIBLE_DEVICES（节点镜像预置为空串=0卡可见）
#   - pip install -e 在预检阶段每节点串行装一次（不在 worker 热路径并发装）
#   - rank = 存活节点稠密下标；NUM_NODES = 存活节点数
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CHECK_ONLY=0
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=1 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

DATASET="${1:?用法: $0 [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>}"
OUT_PATH="${2:?用法: $0 [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>}"
HOSTFILE="${3:?用法: $0 [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>}"
[[ -f "$HOSTFILE" ]] || { echo "[ERROR] hostfile 不存在: $HOSTFILE"; exit 1; }

# 本次运行的输出根：覆盖 config.sh 默认，并向所有远程 SSH 透传
export OUT_BASE="$OUT_PATH"

echo "========================================"
echo "数据集(group): $DATASET"
echo "输出根 OUT_BASE: $OUT_BASE"
echo "hostfile: $HOSTFILE"
echo "数据根 DATA_ROOT: $DATA_ROOT"
echo "========================================"

# ── 解析 hostfile（防 CRLF / 注释 / 空行 / 纯数字误填）────────────────────────
mapfile -t RAW_LINES < "$HOSTFILE"
LINES=()
for line in "${RAW_LINES[@]}"; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    NODE_ID=$(awk '{print $1}' <<< "$line")
    if [[ "$NODE_ID" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] 行节点名 '$NODE_ID' 是纯数字，会被 SSH 当无效 IP，请修 hostfile"; exit 1
    fi
    LINES+=("$line")
done
ORIGINAL_NUM_NODES=${#LINES[@]}
[[ $ORIGINAL_NUM_NODES -ge 1 ]] || { echo "[ERROR] hostfile 为空: $HOSTFILE"; exit 1; }

DRIVER_LOG_DIR="$OUT_BASE/driver_logs"
mkdir -p "$DRIVER_LOG_DIR"

##############################################################################
# 阶段1/2：预检（每节点串行：装包一次 + 环境/CUDA/GPU数 校验），剔除坏点
##############################################################################
echo -e "\n===== 阶段1/2：预检 $ORIGINAL_NUM_NODES 节点 ====="
VALID_LINES=()
for i in "${!LINES[@]}"; do
    NODE_ID=$(awk '{print $1}' <<< "${LINES[$i]}")
    SLOTS=$(awk -F'slots=' 'NF>1{print $2}' <<< "${LINES[$i]}"); SLOTS="${SLOTS:-8}"
    echo -e "\n--- 探测节点: $NODE_ID (期望 $SLOTS 卡) ---"

    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" "
        set -e
        export OUT_BASE='$OUT_BASE'
        export DATA_ROOT='$DATA_ROOT'
        source '$PROJ_DIR/experiments/batch_production/config.sh'
        unset CUDA_VISIBLE_DEVICES

        test -x '$ENV_DIR/bin/python' || { echo '[FAIL] ENV_DIR 缺失'; exit 1; }
        test -f '$PROJ_DIR/experiments/batch_production/run_worker.py' || { echo '[FAIL] 代码缺失'; exit 1; }
        test -d '$DATA_ROOT/$DATASET/shards' || { echo '[FAIL] 数据集 shards 不存在: $DATASET'; exit 1; }
        mkdir -p '$OUT_BASE' || { echo '[FAIL] OUT_BASE 不可写'; exit 1; }

        # 每节点串行安装一次（不在 worker 热路径并发装，避免 editable-install 竞态）
        pip install --no-user -e '$PROJ_DIR' --no-deps --no-build-isolation --quiet

        python -c \"
import sana_wm_pipeline, vipe_ext, vipe, torch
assert torch.cuda.is_available(), 'CUDA 不可用'
n = torch.cuda.device_count()
assert n == $SLOTS, f'GPU 数量={n}，期望={$SLOTS}'
print(f'[OK] cuda={torch.version.cuda} gpus={n}')
\"
        echo '[OK] 节点就绪'
    "; then
        VALID_LINES+=("${LINES[$i]}")
    else
        echo "[WARN] $NODE_ID 预检未通过，已从本次任务池剔除"
    fi
done

LINES=("${VALID_LINES[@]}")
NUM_NODES=${#LINES[@]}
echo -e "\n===== 预检汇总：健康 $NUM_NODES / 原始 $ORIGINAL_NUM_NODES（剔除 $((ORIGINAL_NUM_NODES-NUM_NODES))）====="
[[ $NUM_NODES -ge 1 ]] || { echo "[ERROR] 无可用节点，中止"; exit 1; }

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo "（--check-only：预检完成，不拉起任务）"; exit 0
fi

##############################################################################
# 阶段2/2：稠密 rank 拉起
##############################################################################
echo -e "\n===== 阶段2/2：用 $NUM_NODES 个健康节点拉起 ====="
for i in "${!LINES[@]}"; do
    NODE_ID=$(awk '{print $1}' <<< "${LINES[$i]}")
    LOG="$DRIVER_LOG_DIR/node${i}_driver.log"

    RUNNING=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
        "pgrep -f 'run_groups_sequential.sh|run_worker.py' | wc -l" 2>/dev/null) || RUNNING=0
    if [[ "${RUNNING:-0}" -gt 0 ]]; then
        echo "[WARN] rank $i ($NODE_ID) 已有任务在跑，先清理旧进程"
        ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
            "pkill -9 -f 'run_groups_sequential.sh|run_worker.py' 2>/dev/null || true" || true
        sleep 2
    fi

    echo "Rank $i → $NODE_ID (driver: $LOG)"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" "
        set -e
        export OUT_BASE='$OUT_BASE'
        export DATA_ROOT='$DATA_ROOT'
        source '$PROJ_DIR/experiments/batch_production/config.sh'
        unset CUDA_VISIBLE_DEVICES
        cd '$PROJ_DIR'
        nohup bash experiments/batch_production/run_groups_sequential.sh \
            --groups '$DATASET' $i $NUM_NODES > '$LOG' 2>&1 < /dev/null &
        disown
    " || echo "[WARN] rank $i ($NODE_ID) SSH 启动失败"
done

echo -e "\n===== 拉起完毕：$NUM_NODES 节点 ====="
echo "监控: bash $SCRIPT_DIR/watch_progress.sh $DATASET $OUT_BASE"
echo "停止: bash $SCRIPT_DIR/stop_all_nodes.sh $HOSTFILE"
```

- [ ] **Step 4: 跑测试确认通过 + 语法**

Run: `cd $PROJ_DIR && bash -n experiments/batch_production/launch_all_nodes.sh && bash tests/test_launch_all_nodes_parse.sh`
Expected: `PASS: launch_all_nodes 参数解析 + rank + 环境修复 全部正确`

- [ ] **Step 5: 提交**

```bash
git add experiments/batch_production/launch_all_nodes.sh tests/test_launch_all_nodes_parse.sh
git commit -m "fix(batch): rewrite launch_all_nodes — dual-arg, config.sh-only remote env, fix per-node CUDA

Remove master PATH/LD_LIBRARY_PATH injection into remote shells (root cause
of one-node CUDA failure); source config.sh as single env source + unset
CUDA_VISIBLE_DEVICES; serial per-node pip install in precheck.
New signature: launch_all_nodes.sh [--check-only] <DATASET> <OUT_PATH> <HOSTFILE>.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `stop_all_nodes.sh` —— 多 hostfile + 监控参数化

**Files:**
- Modify: `experiments/batch_production/stop_all_nodes.sh`
- Modify: `experiments/batch_production/watch_progress.sh`

**Interfaces:**
- `stop_all_nodes.sh <HOSTFILE> [<HOSTFILE2> ...]`：遍历多份 hostfile（一条命令停所有任务组），SIGTERM→SIGKILL。
- `watch_progress.sh [GROUP] [OUT_BASE]`：可指定自定义 OUT_BASE 看对应任务组。

- [ ] **Step 1: `stop_all_nodes.sh` 支持多 hostfile**

把第 7 行 `HOSTFILE="${1:?...}"` 与单文件 `while read < "$HOSTFILE"` 改为遍历 `"$@"`：

```bash
[[ $# -ge 1 ]] || { echo "用法: $0 <HOSTFILE> [<HOSTFILE2> ...]"; exit 1; }
KILL_PATTERN="run_worker\.py|run_groups_sequential\.sh|launch_single_node\.sh"

for HOSTFILE in "$@"; do
    [[ -f "$HOSTFILE" ]] || { echo "ERROR: hostfile 不存在: $HOSTFILE"; exit 1; }
    echo -e "\n===== 停止 hostfile: $HOSTFILE ====="
    while read -r NODE_ID _; do
        NODE_ID="${NODE_ID%$'\r'}"
        [[ -z "$NODE_ID" || "$NODE_ID" =~ ^# ]] && continue
        echo "--- 节点 $NODE_ID ---"
        COUNT=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
            "pgrep -f '$KILL_PATTERN' | wc -l" 2>/dev/null) || COUNT=0
        if [[ "${COUNT:-0}" -gt 0 ]]; then
            echo "[1/2] $COUNT 进程 → SIGTERM"
            ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
                "pkill -TERM -f '$KILL_PATTERN' 2>/dev/null || true" || true
            sleep 2
            REMAIN=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
                "pgrep -f '$KILL_PATTERN' | wc -l" 2>/dev/null) || REMAIN=0
            if [[ "${REMAIN:-0}" -gt 0 ]]; then
                echo "[2/2] 残留 $REMAIN → SIGKILL"
                ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -n root@"$NODE_ID" \
                    "pkill -KILL -f '$KILL_PATTERN' 2>/dev/null || true" || true
            fi
            echo "节点 $NODE_ID 已停 ✅"
        else
            echo "节点 $NODE_ID 无进程，跳过 ✅"
        fi
    done < "$HOSTFILE"
done
echo -e "\n===== 全部 hostfile 处理完成 ====="
```

（保留文件顶部 `#!/bin/bash` + `set -euo pipefail`。）

- [ ] **Step 2: `watch_progress.sh` 加 OUT_BASE 参数**

第 11 行后插入；让 OUT_BASE 可由第二参数覆盖（在 source config.sh 之后）：

```bash
GROUP="${1:-wds-sekai-real-walking-hq}"
OUT_BASE="${2:-$OUT_BASE}"   # 可指定自定义输出路径对应的任务组
```

- [ ] **Step 3: 语法检查**

Run: `cd $PROJ_DIR && bash -n experiments/batch_production/stop_all_nodes.sh && bash -n experiments/batch_production/watch_progress.sh && echo OK`
Expected: `OK`

- [ ] **Step 4: 提交**

```bash
git add experiments/batch_production/stop_all_nodes.sh experiments/batch_production/watch_progress.sh
git commit -m "feat(batch): stop_all_nodes accepts multiple hostfiles; watch_progress takes OUT_BASE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `README.md` —— 输出规范 + 5 组启动模板 + 迁移说明

**Files:**
- Create: `experiments/batch_production/README.md`

- [ ] **Step 1: 写 README**

```markdown
# CMCC 批量生产脚本套件

## 标准输出目录规范
（粘贴本计划「标准输出目录规范」整节）

## 两个历史事故根因
（粘贴本计划「根因诊断」整节：问题1 重启截断丢数、问题2 master 环境注入）

## 5 个独立任务组启动模板

每个任务组 = 一个数据集 + 一份独立 hostfile + 一个独立输出路径（彼此隔离，无 worker 目录写冲突）：

\`\`\`bash
cd /root/work/david_work/sana_wm_pipeline
BP=experiments/batch_production
OUT=/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output

# 组1
bash $BP/launch_all_nodes.sh wds-sekai-real-walking-hq $OUT/final_sekai_realwalking_hq  /path/hostfile_group1
# 组2
bash $BP/launch_all_nodes.sh wds-DL3DV-ALL-2K          $OUT/final_dl3dv_all_2k          /path/hostfile_group2
# 组3
bash $BP/launch_all_nodes.sh wds-SpatialVID-hq         $OUT/final_spatialvid_hq         /path/hostfile_group3
# 组4 / 组5：同理换 <DATASET> <OUT_PATH> <HOSTFILE>

# 先只预检不拉起：加 --check-only
bash $BP/launch_all_nodes.sh --check-only wds-sekai-real-walking-hq $OUT/final_sekai_realwalking_hq /path/hostfile_group1

# 监控 / 停止
bash $BP/watch_progress.sh wds-sekai-real-walking-hq $OUT/final_sekai_realwalking_hq
bash $BP/stop_all_nodes.sh /path/hostfile_group1                      # 停单组
bash $BP/stop_all_nodes.sh /path/hostfile_group1 /path/hostfile_group2 ...  # 一键停多组
\`\`\`

## 断点续跑
任意中断后**重跑同一条命令即可**：worker 按输入 shard 下标判定，
`progress/{idx}.done` + 对应 `shard-{idx}-*.tar` 同时存在才跳过，否则重算。

## 从旧版本迁移（重要）
旧脚本因「共享 shard-000000.tar 截断」bug 产出的输出**不可信**。新命名
（`shard-{idx}-*.tar`）会让旧的全部 `.done` 自动判为未完成并重算，**无需手动删目录**；
但旧的 `w*/shard-000000.tar` 残留文件建议手动清掉以免与新文件混淆：
\`\`\`bash
find $OUT_BASE/<group> -name 'shard-000000.tar' -path '*/w*' -delete   # 仅删旧共享命名残留
\`\`\`
```

- [ ] **Step 2: 提交**

```bash
git add experiments/batch_production/README.md
git commit -m "docs(batch): output dir spec, 5-group launch templates, migration notes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: 全量回归 + sync_to_nodes 分发提醒

**Files:** 无新增（验证 + 文档）

- [ ] **Step 1: 跑全部相关测试**

Run: `cd $PROJ_DIR && python -m pytest tests/test_shard_io.py tests/test_run_worker_resume.py tests/test_webdataset_writer.py -v`
Expected: 全 PASS（确认既有 writer 测试未被破坏）

- [ ] **Step 2: 全脚本语法检查**

Run: `cd $PROJ_DIR && for f in experiments/batch_production/*.sh; do bash -n "$f" && echo "OK $f"; done`
Expected: 每个脚本 `OK`

- [ ] **Step 3: 跑 bash 断言测试**

Run: `cd $PROJ_DIR && bash tests/test_config_overridable.sh && bash tests/test_launch_all_nodes_parse.sh`
Expected: 两个 `PASS`

- [ ] **Step 4: 提醒分发（人工，不自动执行）**

CMCC 镜像不可重发布，改动须用 `sync_to_nodes.sh` 分发到全部节点（含被改的 `config.sh`/`run_worker.py`/`shard_io.py`/`launch_*`/`stop_*`/`watch_*`）：

```bash
# 先把本次改动从 AFS 传到 CMCC 任一可 SSH 全节点的 master，再：
bash experiments/batch_production/sync_to_nodes.sh /path/hostfile_group1 /path/hostfile_group2 ...
```

> `sync_to_nodes.sh` 已 `scp` 整个 `batch_production/*.sh *.py`，新增的 `shard_io.py` 会一并分发——无需改 sync 脚本。

---

## Self-Review

**1. Spec coverage:**
- 标准输出目录规范 → Task 8 README + 文中「标准输出目录规范」节 ✅
- 单 shard tar 根因分析 → 「根因诊断 问题1」+ Task 2 修复 ✅
- 完整审计 launch_all_nodes + 修跨节点 conda/CUDA → 「根因诊断 问题2」+ Task 6 全重写 ✅
- 双参数（数据集名 + 输出路径）→ Task 6 签名 + Task 3 config 覆盖 + 透传链 ✅
- 分片完成度检测 / 自动续跑 → Task 1 `shard_is_complete` + Task 2 集成 ✅
- 多节点一键停止 → Task 7 `stop_all_nodes.sh`（多 hostfile）✅
- 带参集群启动命令模板 → Task 8 README 5 组模板 ✅

**2. Placeholder scan:** 各 step 均含完整代码/命令/期望输出，无 TBD/TODO。

**3. Type consistency:** `output_shard_prefix`/`shard_is_complete`/`mark_shard_done` 三函数签名在 Task 1 定义、Task 2 调用一致；`process_input_shard` 新签名 `(shard_path, shard_idx, group, captions, tmp_dir, worker_out, progress_dir, samples_per_shard, shard_writer_cls=None)` 在 Task 2 测试与 main 调用一致；`launch_all_nodes.sh <DATASET> <OUT_PATH> <HOSTFILE>` 与 README 模板、监控/停止命令一致。

## 待用户确认的设计取舍（执行前可推翻）
- **输出路径嵌套：** 采用 `OUT_BASE=<参数2>` 后仍保留 `<group>/` 子目录（即最终 `<参数2>/<dataset>/w*`）。好处：`progress`/`logs`/`driver_logs` 分门别类、续跑 key 稳定；代价：路径比示例多一层 group 名。若希望直接落在参数2 下不加 group 层，改 `run_worker.py` 的 `worker_out = args.out_base / f"w{...}"`（去掉 `/ args.group`）即可，但会牺牲同一 OUT_BASE 跑多 group 的能力——鉴于「一组一输出路径」架构二者等价，默认保留 group 层更稳。

# jdvbbfb-v3-full Default 模式适配 + CMCC 打包 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 SANA-WM 数据管线的 **default 模式**能直接消费 Hugging Face 数据集 `junchaoh-cs/jdvbbfb-v3-full`（WebDataset 打包格式）里的 RGB 视频，产出 SANA-WM 训练 shard，并完成可在 CMCC 中移动无外网机器上跑大规模处理的完整环境+代码打包。

**Architecture:** `mode_default.py` 是数据集无关的（只吃一个 `normalized.mp4`），**不修改**。适配工作全部落在「数据摄取层」：新增一个可单测的纯函数 WDS 读取模块（`src/.../stage01_ingest/jdvbbfb_wds.py`）+ 一个 CLI 准备脚本（`prepare_jdvbbfb.py`，**双输入模式**：HF 流式 `--repo`（H100 开发用）/ 本地目录 `--local-root`（CMCC 生产用））+ 一个端到端编排脚本（`run_e2e_default_jdvbbfb.sh`）+ `configs/sources.yaml` 新增源条目。数据集每个样本 = `{key}.mp4` + `{key}.camera.npz`（含 GT 位姿 + 已跑好的 VIPE 位姿），caption 文本在 `index.jsonl` 的 `manifest.prompt.text`。准备脚本把每个样本落成 `prepare_omniworld.py` 同构的 scene 目录（`video.mp4 / gt_poses.npy / gt_intrinsics.npy / orig_fps.txt / caption.txt`），后续复用现成 default E2E 流程。

> **2026-06-14 更新（关键）**：数据集**已在 CMCC 集群持久盘就位**，无需从 HF 下载+传输：
> `/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full/`（`externalstorage` 是 CMCC 两个持久化目录之一，重启不丢）。
> 目录树与 HF repo 一致：`{root}/wds-<NAME>/{index.jsonl, shards/<NAME>-NNNNNN.tar}`。
> 因此 CMCC 上 `prepare_jdvbbfb.py --local-root <root>` 直接读本地 tar，**不需要 HF token、不需要 modelscope 传数据**（env/代码/模型仍需打包传输）。

**Tech Stack:** Python 3.10, huggingface_hub 0.36, numpy, tarfile（流式）, requests, ffmpeg(static), conda env `sana_wm`；下游 Pi3X + MoGe-2 + VIPE SLAM（H100）；CMCC 部署用 conda-pack + modelscope。

---

## 背景事实（已实测确认，2026-06-14）

### 数据集结构 `junchaoh-cs/jdvbbfb-v3-full`（gated: manual，已授权）

顶层 8 个 WebDataset 子集 + 若干顶层日志/校验 JSON。每个子集目录 `wds-<NAME>/`：

```
wds-<NAME>/
├── build_config.json            # 打包配置（shard_size=512, max_shard_bytes=2GB, json_members_in_shards=false）
├── stats.json                   # samples / shards / bytes / passed
├── index.jsonl                  # 每行一个样本：sample_id,key,shard,video_member,camera_member,manifest{...}
├── reversible_manifest.json
├── file/
│   ├── manifest.jsonl           # 与 index.jsonl 的 manifest 字段同源（含 prompt.text 全文）
│   ├── metadata.csv
│   ├── stats.json
│   └── _meta/...                # 各类校验报告
└── shards/<NAME>-NNNNNN.tar     # 每个 tar 内：{key}.mp4 + {key}.camera.npz（无 json，json_members_in_shards=false）
```

8 子集规模（实测）：

| 子集 | samples | shards | 分辨率 | fps | frames |
|------|--------:|-------:|--------|-----|-------:|
| wds-Context-as-Memory | 100 | 9 | 640×360 | 30 | 7601 |
| wds-DL3DV-ALL-2K | 9,993 | 87 | 1920×1080 | 30 | 300 |
| wds-OmniWorld-Game | 6,576 | 65 | 1280×720 | 24 | 1508 |
| wds-RealEstate10K-360p | 73,165 | 143 | 640×360 | 30 | 70 |
| wds-SpatialVID-hq | 365,362 | 714 | 1280×720 | ~59.94 | 740 |
| wds-sekai-game-drone | 932 | 5 | 1920×1080 | 30 | 300 |
| wds-sekai-game-walking | 1,618 | 43 | 1920×1080 | 30 | 1800 |
| wds-sekai-real-walking-hq | 18,208 | 287 | 1280×720 | 30 | 1800 |
| **合计** | **475,954** | **1,353** | | | |

### 每个样本的 `{key}.camera.npz` 精确 schema（实测，`per_frame_camera_npz_v1`）

```
c2w               (T,4,4) float32   GT camera-to-world, opencv 约定
w2c               (T,4,4) float32   GT world-to-camera
K_px              (T,4)   float32   [fx,fy,cx,cy]，原始分辨率像素单位
frame_indices     (T,)    int32
raw_frame_indices (T,)    int32
width/height      scalar  int32     例 1920 / 1080
fps               scalar  float32   例 30.0
source_width/height        int32    原生分辨率（例 3840×2160）
source_camera_model        <U6      'OPENCV'
source_pose_format         str      'dl3dv_transforms_json_opengl_c2w_converted_to_opencv_c2w'
pose_convention            <U10     'opencv_c2w'
intrinsics_format          str      'fx_fy_cx_cy_pixels_original_resolution'
# —— 已跑好的 VIPE 参考位姿（可作 default 模式 baseline 对照）——
vipe_c2w          (T,4,4) float32
vipe_w2c          (T,4,4) float32
vipe_K_px         (T,4)   float32
vipe_frame_indices(T,)    int32
vipe_sparse_c2w   (S,4,4) float32   关键帧稀疏位姿（例 S=75）
vipe_sparse_indices (S,)  int32
vipe_frame_skip   scalar  int32     例 4
vipe_interpolation str             'linear_translation_slerp_rotation'
vipe_run_id        str             'dl3dv_vipe_full_skip4_v1'
vipe_status        str             'ok'
...（其余 vipe_* 元信息）
```

> **设计含义**：数据集已自带 GT 位姿（`c2w`）与一套 VIPE 位姿（`vipe_c2w`）。本任务的「default 模式」指**用本管线自己的 Pi3X+MoGe-2+VIPE 流程重新标注**，`c2w` 用作 ATE 评估的 GT，`vipe_c2w` 可作为现成 baseline 对照。

### index.jsonl 每条记录字段（实测）

```json
{
  "sample_id": "DL3DV-ALL-2K/6K__<hash>__images_2",
  "key": "DL3DV-ALL-2K_6K__<hash>__images_2",
  "shard": "shards/DL3DV-ALL-2K-000000.tar",
  "video_member": "<key>.mp4",
  "camera_member": "<key>.camera.npz",
  "manifest": {
    "video": {"num_frames":300,"fps":30.0,"width":1920,"height":1080,...},
    "prompt": {"text": "<caption 全文>", "source": "target_prompt.tar.gz"},
    "camera": {"pose_convention":"opencv_c2w","intrinsics_format":"fx_fy_cx_cy_pixels_original_resolution",...}
  }
}
```

### 下游 default 流程接口（已确认，不改）

- `normalize_video(in_path: Path, out_path: Path) -> VideoInfo`（统一到 1280×720 @16fps）
- `run_default(clip_path: Path, work_dir: Path) -> PoseArtifact`（需要环境变量 `SANA_WM_PI3X_WEIGHTS` / `SANA_WM_MOGE2_WEIGHTS`）
- `verify_and_eval.py --mode pose-eval` 读取 `scenes-dir/{scene_id}/gt_poses.npy` + `orig_fps.txt`，把 GT 下采样到 16fps 后算 ATE。
- stage06 pack 的 shard 样本命名：`{id}.mp4 / {id}.poses_c2w.npy / {id}.intrinsics.npy / {id}.scale.npy / {id}.caption.txt / {id}.meta.json`。

---

## 文件结构（本计划新增/修改）

| 操作 | 路径 | 职责 |
|------|------|------|
| Create | `src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py` | 纯函数：解析 camera.npz → GT 数组；读 index.jsonl → 样本记录；流式从 shard 提取 `{key}.mp4`/`.camera.npz`。**可单测，无网络依赖于纯函数部分。** |
| Create | `tests/test_jdvbbfb_wds.py` | 单测纯函数（用内存合成 npz / 临时 tar，无网络）。 |
| Create | `experiments/data_production_smoke/prepare_jdvbbfb.py` | CLI：从 HF 一个 shard 取前 N 个样本 → 写成 scene 目录（video.mp4/gt_poses.npy/gt_intrinsics.npy/orig_fps.txt/caption.txt/vipe_ref_poses.npy）。 |
| Create | `experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh` | 端到端：normalize → run_default → pack shard → schema check → pose-eval（vs gt_poses.npy）。 |
| Modify | `configs/sources.yaml` | 新增 `jdvbbfb_v3_full` 源条目（不破坏 total_clips 断言：放进独立顶层键 `external_corpora`，不计入 `sources` 总和）。 |
| Create | `docs/JDVBBFB_DEFAULT_GUIDE.md` | 该数据集 default 模式的运行 + CMCC 打包说明。 |

> `mode_default.py` / `schema.py` / `verify_and_eval.py` / `normalize.py` **均不修改**——适配是纯增量摄取层。

---

## Task 1：新分支 + 占位模块骨架

**Files:**
- Create: `src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py`

- [ ] **Step 1.1：从 master 切新分支**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git checkout master
git pull --ff-only origin master || true
git checkout -b feat/jdvbbfb-default-adapt
git rev-parse --abbrev-ref HEAD   # 预期：feat/jdvbbfb-default-adapt
```

- [ ] **Step 1.2：创建空模块（仅 import 可用，函数稍后 TDD 填充）**

```python
# src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py
"""Ingest adapter for the junchaoh-cs/jdvbbfb-v3-full WebDataset corpus.

Per-sample layout inside each shard tar:
  {key}.mp4          — RGB video (H264)
  {key}.camera.npz   — per_frame_camera_npz_v1 (GT c2w/K_px + vipe_* refs)
Caption text lives in <group>/index.jsonl  →  record["manifest"]["prompt"]["text"]
(json_members_in_shards=false, so prompts are NOT inside the tar).

This module holds only pure / unit-testable helpers. Network + HF download
glue lives in experiments/data_production_smoke/prepare_jdvbbfb.py.
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
```

- [ ] **Step 1.3：提交骨架**

```bash
git add src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py
git commit -m "feat(ingest): scaffold jdvbbfb-v3-full WDS adapter module"
```

---

## Task 2：`load_camera_gt` —— camera.npz → GT 数组（TDD）

**Files:**
- Modify: `src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py`
- Test: `tests/test_jdvbbfb_wds.py`

- [ ] **Step 2.1：写失败测试**

```python
# tests/test_jdvbbfb_wds.py
import io
import numpy as np
import pytest
from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import load_camera_gt, CameraGT


def _synth_camera_npz(T=5) -> bytes:
    """Build an in-memory per_frame_camera_npz_v1 like the real dataset."""
    c2w = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    for t in range(T):
        c2w[t, 0, 3] = float(t)          # translate along x so it's non-trivial
    buf = io.BytesIO()
    np.savez(
        buf,
        c2w=c2w,
        w2c=np.linalg.inv(c2w).astype(np.float32),
        K_px=np.tile(np.array([500, 500, 320, 240], np.float32), (T, 1)),
        frame_indices=np.arange(T, dtype=np.int32),
        width=np.int32(1920), height=np.int32(1080), fps=np.float32(30.0),
        pose_convention=np.array("opencv_c2w"),
        vipe_c2w=c2w.copy(),
    )
    return buf.getvalue()


def test_load_camera_gt_basic():
    gt = load_camera_gt(_synth_camera_npz(T=5))
    assert isinstance(gt, CameraGT)
    assert gt.c2w.shape == (5, 4, 4) and gt.c2w.dtype == np.float32
    assert gt.k_px.shape == (5, 4)
    assert gt.fps == pytest.approx(30.0)
    assert gt.width == 1920 and gt.height == 1080
    assert gt.vipe_c2w is not None and gt.vipe_c2w.shape == (5, 4, 4)


def test_load_camera_gt_missing_vipe_is_none():
    buf = io.BytesIO()
    np.savez(buf,
             c2w=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
             K_px=np.zeros((3, 4), np.float32),
             width=np.int32(640), height=np.int32(360), fps=np.float32(24.0))
    gt = load_camera_gt(buf.getvalue())
    assert gt.vipe_c2w is None
    assert gt.fps == pytest.approx(24.0)
```

- [ ] **Step 2.2：运行确认失败**

Run: `cd /mnt/afs/davidwang/workspace/sana_wm_pipeline && conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm && pytest tests/test_jdvbbfb_wds.py -v`
Expected: FAIL —「cannot import name 'load_camera_gt'」

- [ ] **Step 2.3：实现**

在 `jdvbbfb_wds.py` 追加：

```python
@dataclass(frozen=True)
class CameraGT:
    """Parsed GT camera state from a {key}.camera.npz member."""
    c2w: np.ndarray          # (T,4,4) float32 opencv c2w
    k_px: np.ndarray         # (T,4)   float32 [fx,fy,cx,cy] original-res pixels
    fps: float
    width: int
    height: int
    vipe_c2w: np.ndarray | None = None   # (T,4,4) float32 reference VIPE poses


def load_camera_gt(npz_bytes: bytes) -> CameraGT:
    """Parse a per_frame_camera_npz_v1 byte blob into CameraGT.

    Robust to the optional vipe_* fields (RealEstate/Context groups may differ).
    """
    z = np.load(io.BytesIO(npz_bytes))
    files = set(z.files)
    if "c2w" not in files:
        raise ValueError(f"camera npz missing 'c2w' (have: {sorted(files)})")
    return CameraGT(
        c2w=z["c2w"].astype(np.float32),
        k_px=z["K_px"].astype(np.float32) if "K_px" in files
             else np.zeros((len(z["c2w"]), 4), np.float32),
        fps=float(z["fps"]) if "fps" in files else 30.0,
        width=int(z["width"]) if "width" in files else 0,
        height=int(z["height"]) if "height" in files else 0,
        vipe_c2w=z["vipe_c2w"].astype(np.float32) if "vipe_c2w" in files else None,
    )
```

- [ ] **Step 2.4：运行确认通过**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: PASS（2 passed）

- [ ] **Step 2.5：提交**

```bash
git add src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py tests/test_jdvbbfb_wds.py
git commit -m "feat(ingest): load_camera_gt parses per_frame_camera_npz_v1"
```

---

## Task 3：`read_index` —— 解析 index.jsonl → 样本记录（TDD）

**Files:**
- Modify: `src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py`
- Test: `tests/test_jdvbbfb_wds.py`

- [ ] **Step 3.1：写失败测试**

```python
# 追加到 tests/test_jdvbbfb_wds.py
from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import read_index, SampleRef


def test_read_index_parses_records(tmp_path):
    rec = {
        "sample_id": "DL3DV-ALL-2K/6K__abc__images_2",
        "key": "DL3DV-ALL-2K_6K__abc__images_2",
        "shard": "shards/DL3DV-ALL-2K-000000.tar",
        "video_member": "DL3DV-ALL-2K_6K__abc__images_2.mp4",
        "camera_member": "DL3DV-ALL-2K_6K__abc__images_2.camera.npz",
        "manifest": {"video": {"fps": 30.0, "num_frames": 300},
                     "prompt": {"text": "a calm indoor lounge"}},
    }
    p = tmp_path / "index.jsonl"
    p.write_text(json.dumps(rec) + "\n" + json.dumps({**rec, "key": "k2",
                 "shard": "shards/DL3DV-ALL-2K-000001.tar"}) + "\n")

    refs = read_index(p)
    assert len(refs) == 2
    r0 = refs[0]
    assert isinstance(r0, SampleRef)
    assert r0.key == "DL3DV-ALL-2K_6K__abc__images_2"
    assert r0.shard == "shards/DL3DV-ALL-2K-000000.tar"
    assert r0.video_member.endswith(".mp4")
    assert r0.camera_member.endswith(".camera.npz")
    assert r0.caption == "a calm indoor lounge"
    assert r0.fps == pytest.approx(30.0)


def test_read_index_caption_fallback(tmp_path):
    rec = {"key": "k", "shard": "s.tar",
           "video_member": "k.mp4", "camera_member": "k.camera.npz",
           "manifest": {"video": {}}}     # no prompt
    p = tmp_path / "index.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    refs = read_index(p)
    assert refs[0].caption == ""          # graceful empty, not KeyError
```

需要在测试文件顶部确保 `import json`（已在 Step 2.1 之外，补一行）。

- [ ] **Step 3.2：运行确认失败**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: FAIL —「cannot import name 'read_index'」

- [ ] **Step 3.3：实现**

```python
@dataclass(frozen=True)
class SampleRef:
    """One row of <group>/index.jsonl, with caption/fps hoisted for convenience."""
    sample_id: str
    key: str
    shard: str                 # e.g. "shards/DL3DV-ALL-2K-000000.tar"
    video_member: str          # tar member name "{key}.mp4"
    camera_member: str         # tar member name "{key}.camera.npz"
    caption: str
    fps: float


def read_index(index_path: Path) -> list[SampleRef]:
    """Parse a group's index.jsonl into SampleRef rows."""
    refs: list[SampleRef] = []
    for line in Path(index_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        man = rec.get("manifest", {})
        prompt = man.get("prompt", {}) or {}
        video = man.get("video", {}) or {}
        refs.append(SampleRef(
            sample_id=rec.get("sample_id", rec["key"]),
            key=rec["key"],
            shard=rec["shard"],
            video_member=rec["video_member"],
            camera_member=rec["camera_member"],
            caption=prompt.get("text", "") or "",
            fps=float(video.get("fps", 0.0) or 0.0),
        ))
    return refs
```

- [ ] **Step 3.4：运行确认通过**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: PASS（4 passed）

- [ ] **Step 3.5：提交**

```bash
git add src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py tests/test_jdvbbfb_wds.py
git commit -m "feat(ingest): read_index parses jdvbbfb index.jsonl rows"
```

---

## Task 4：`iter_tar_samples` —— 从一个 tar 流式取 (key, mp4_bytes, camera_bytes)（TDD）

**Files:**
- Modify: `src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py`
- Test: `tests/test_jdvbbfb_wds.py`

- [ ] **Step 4.1：写失败测试（用临时 tar，无网络）**

```python
# 追加到 tests/test_jdvbbfb_wds.py
import tarfile
from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import iter_tar_samples


def _make_shard(tmp_path, keys):
    shard = tmp_path / "shard.tar"
    with tarfile.open(shard, "w") as tf:
        for k in keys:
            for ext, payload in [(".mp4", b"FAKEVIDEO" + k.encode()),
                                 (".camera.npz", b"FAKENPZ" + k.encode())]:
                data = payload
                ti = tarfile.TarInfo(f"{k}{ext}"); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return shard


def test_iter_tar_samples_pairs_mp4_and_npz(tmp_path):
    shard = _make_shard(tmp_path, ["sampleA", "sampleB"])
    got = list(iter_tar_samples(open(shard, "rb"), limit=None))
    assert [k for k, _, _ in got] == ["sampleA", "sampleB"]
    k0, mp4_0, npz_0 = got[0]
    assert mp4_0 == b"FAKEVIDEOsampleA"
    assert npz_0 == b"FAKENPZsampleA"


def test_iter_tar_samples_respects_limit(tmp_path):
    shard = _make_shard(tmp_path, ["a", "b", "c"])
    got = list(iter_tar_samples(open(shard, "rb"), limit=2))
    assert len(got) == 2
```

- [ ] **Step 4.2：运行确认失败**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: FAIL —「cannot import name 'iter_tar_samples'」

- [ ] **Step 4.3：实现（流式，按 key 聚合 .mp4 + .camera.npz）**

```python
def _split_key(member_name: str) -> tuple[str, str]:
    """'foo.bar.camera.npz' -> ('foo.bar', '.camera.npz'); 'foo.mp4' -> ('foo', '.mp4')."""
    if member_name.endswith(".camera.npz"):
        return member_name[: -len(".camera.npz")], ".camera.npz"
    if member_name.endswith(".mp4"):
        return member_name[: -len(".mp4")], ".mp4"
    # ignore any other extension
    return member_name, ""


def iter_tar_samples(fileobj, limit: int | None = None
                     ) -> Iterator[tuple[str, bytes, bytes]]:
    """Stream a shard tar, yielding (key, mp4_bytes, camera_npz_bytes).

    Works on any binary fileobj (local file or requests.raw HTTP stream).
    Pairs the two members per sample-key; yields once both are seen.
    Stops after `limit` complete samples (None = all).
    """
    pending: dict[str, dict[str, bytes]] = {}
    n = 0
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = _split_key(m.name)
            if ext not in (".mp4", ".camera.npz"):
                continue
            data = tf.extractfile(m).read()
            slot = pending.setdefault(key, {})
            slot[ext] = data
            if ".mp4" in slot and ".camera.npz" in slot:
                yield key, slot[".mp4"], slot[".camera.npz"]
                pending.pop(key, None)
                n += 1
                if limit is not None and n >= limit:
                    return
```

- [ ] **Step 4.4：运行确认通过**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: PASS（6 passed）

- [ ] **Step 4.5：提交**

```bash
git add src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py tests/test_jdvbbfb_wds.py
git commit -m "feat(ingest): iter_tar_samples streams mp4+camera pairs from shard"
```

---

## Task 5：`write_scene_dir` —— 一个样本 → scene 目录（TDD）

**Files:**
- Modify: `src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py`
- Test: `tests/test_jdvbbfb_wds.py`

落盘布局必须与 `prepare_omniworld.py` 同构，才能复用 default E2E + pose-eval：
`{out}/{scene_id}/{video.mp4, gt_poses.npy, gt_intrinsics.npy, orig_fps.txt, caption.txt, vipe_ref_poses.npy}`。

- [ ] **Step 5.1：写失败测试**

```python
# 追加到 tests/test_jdvbbfb_wds.py
from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import write_scene_dir


def test_write_scene_dir_layout(tmp_path):
    cam = _synth_camera_npz(T=4)
    scene = write_scene_dir(
        out_base=tmp_path,
        scene_id="DL3DV-ALL-2K_6K__abc__images_2",
        mp4_bytes=b"FAKEVIDEO",
        camera_npz_bytes=cam,
        caption="a calm indoor lounge",
    )
    assert (scene / "video.mp4").read_bytes() == b"FAKEVIDEO"
    assert (scene / "caption.txt").read_text() == "a calm indoor lounge"
    assert (scene / "orig_fps.txt").read_text().strip() == "30.0"
    gt = np.load(scene / "gt_poses.npy")
    assert gt.shape == (4, 4, 4) and gt.dtype == np.float32
    intr = np.load(scene / "gt_intrinsics.npy")
    assert intr.shape == (4, 4)                  # (T,4) [fx,fy,cx,cy]
    assert (scene / "vipe_ref_poses.npy").exists()   # synth npz has vipe_c2w


def test_write_scene_dir_empty_caption_gets_stub(tmp_path):
    buf = io.BytesIO()
    np.savez(buf, c2w=np.tile(np.eye(4, np.float32), (2, 1, 1)),
             K_px=np.zeros((2, 4), np.float32), fps=np.float32(24.0))
    scene = write_scene_dir(tmp_path, "scene_x", b"v", buf.getvalue(), caption="")
    txt = (scene / "caption.txt").read_text()
    assert txt.strip()                            # non-empty stub, not blank
```

- [ ] **Step 5.2：运行确认失败**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: FAIL —「cannot import name 'write_scene_dir'」

- [ ] **Step 5.3：实现**

```python
_STUB_CAPTION = "A static real-world scene with no camera-action description."


def write_scene_dir(out_base: Path, scene_id: str, mp4_bytes: bytes,
                    camera_npz_bytes: bytes, caption: str) -> Path:
    """Materialize one sample into a prepare_omniworld-compatible scene dir.

    Layout: {out_base}/{scene_id}/{video.mp4, gt_poses.npy, gt_intrinsics.npy,
             orig_fps.txt, caption.txt, vipe_ref_poses.npy?}
    Returns the scene directory path.
    """
    gt = load_camera_gt(camera_npz_bytes)
    scene = Path(out_base) / scene_id
    scene.mkdir(parents=True, exist_ok=True)

    (scene / "video.mp4").write_bytes(mp4_bytes)
    np.save(scene / "gt_poses.npy", gt.c2w)            # (T,4,4) c2w GT
    np.save(scene / "gt_intrinsics.npy", gt.k_px)      # (T,4) original-res px
    (scene / "orig_fps.txt").write_text(str(gt.fps))
    (scene / "caption.txt").write_text(caption.strip() or _STUB_CAPTION)
    if gt.vipe_c2w is not None:
        np.save(scene / "vipe_ref_poses.npy", gt.vipe_c2w)
    return scene
```

- [ ] **Step 5.4：运行确认通过**

Run: `pytest tests/test_jdvbbfb_wds.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5.5：提交**

```bash
git add src/sana_wm_pipeline/stage01_ingest/jdvbbfb_wds.py tests/test_jdvbbfb_wds.py
git commit -m "feat(ingest): write_scene_dir materializes omniworld-compatible scene"
```

---

## Task 6：`prepare_jdvbbfb.py` CLI（双输入模式：HF 流式 + 本地目录）

**Files:**
- Create: `experiments/data_production_smoke/prepare_jdvbbfb.py`

此脚本是 I/O 胶水，不做单测；正确性由 Task 8 的单样本端到端验证覆盖。
**两种输入互斥**：`--local-root`（CMCC 生产，读本地 tar，无需 token）优先；否则 `--repo`（H100 开发，HF 流式）。

- [ ] **Step 6.1：写脚本**

```python
#!/usr/bin/env python3
"""Extract samples from one jdvbbfb-v3-full shard into scene dirs.

Each sample → {out_base}/{scene_id}/ with video.mp4 + gt_poses.npy +
gt_intrinsics.npy + orig_fps.txt + caption.txt (+ vipe_ref_poses.npy).
Default mode never uses the GT; gt_poses.npy is for verify_and_eval ATE only.

Two input modes (mutually exclusive):
  A) --local-root <DIR>   read {DIR}/{group}/shards/<prefix>-NNNNNN.tar locally
                          (CMCC production: data already on externalstorage,
                          no HF token / no network)
  B) --repo <REPO>        stream the shard over HTTPS from Hugging Face
                          (H100 dev/smoke; needs a valid HF token)

Usage (CMCC, local):
  python prepare_jdvbbfb.py \\
    --local-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full \\
    --group wds-DL3DV-ALL-2K --shard-idx 0 --sample-limit 0 \\
    --out-base /root/work/<userspace>/jdvbbfb_out

Usage (H100, HF stream):
  python prepare_jdvbbfb.py \\
    --repo junchaoh-cs/jdvbbfb-v3-full \\
    --group wds-DL3DV-ALL-2K --shard-idx 0 --sample-limit 1 \\
    --out-base /mnt/afs/davidwang/workspace/data/jdvbbfb_smoke

--sample-limit 0 (or negative) means ALL samples in the shard (batch production).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sana_wm_pipeline.stage01_ingest.jdvbbfb_wds import (
    iter_tar_samples, read_index, write_scene_dir,
)


def _shard_basename(group: str, shard_idx: int) -> str:
    # "wds-DL3DV-ALL-2K" -> "DL3DV-ALL-2K-000000.tar"
    prefix = group[len("wds-"):] if group.startswith("wds-") else group
    return f"{prefix}-{shard_idx:06d}.tar"


def _open_local(local_root: Path, group: str, shard_idx: int):
    """Return (index_path, fileobj) for a local shard."""
    root = Path(local_root) / group
    index_path = root / "index.jsonl"
    shard = root / "shards" / _shard_basename(group, shard_idx)
    if not shard.exists():
        raise SystemExit(f"local shard not found: {shard}")
    return index_path, open(shard, "rb")


def _open_remote(repo: str, group: str, shard_idx: int):
    """Return (index_path, fileobj) for an HF-streamed shard."""
    import requests
    from huggingface_hub import get_token, hf_hub_download
    index_path = Path(hf_hub_download(repo, f"{group}/index.jsonl",
                                      repo_type="dataset"))
    shard_name = f"{group}/shards/{_shard_basename(group, shard_idx)}"
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{shard_name}"
    tok = get_token()
    if not tok:
        raise SystemExit("No HF token. Run: "
                         "HF_HOME=/mnt/afs/davidwang/cache/huggingface hf auth login")
    r = requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                     stream=True, timeout=300)
    r.raise_for_status()
    return index_path, r.raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-root", type=Path,
                     help="local dataset root (CMCC). Reads {root}/{group}/...")
    src.add_argument("--repo", help="HF repo id (stream over HTTPS)")
    ap.add_argument("--group", required=True, help="e.g. wds-DL3DV-ALL-2K")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--sample-limit", type=int, default=1,
                    help="0 or negative = all samples in shard")
    ap.add_argument("--out-base", required=True, type=Path)
    args = ap.parse_args()

    limit = None if args.sample_limit <= 0 else args.sample_limit

    if args.local_root is not None:
        index_path, fobj = _open_local(args.local_root, args.group, args.shard_idx)
        print(f"[local] {args.local_root}/{args.group} shard {args.shard_idx}")
    else:
        index_path, fobj = _open_remote(args.repo, args.group, args.shard_idx)
        print(f"[stream] {args.repo} {args.group} shard {args.shard_idx}")

    refs = read_index(index_path)
    cap_by_key = {r.key: r.caption for r in refs}
    print(f"[index] {len(refs)} samples in {args.group}")

    n = 0
    try:
        for key, mp4_bytes, camera_bytes in iter_tar_samples(fobj, limit=limit):
            scene = write_scene_dir(args.out_base, key, mp4_bytes,
                                    camera_bytes, cap_by_key.get(key, ""))
            n += 1
            print(f"  [{n}] {key}  →  {scene}  "
                  f"(video {len(mp4_bytes)/1e6:.1f} MB, "
                  f"caption {'yes' if cap_by_key.get(key) else 'stub'})")
    finally:
        fobj.close()
    print(f"[done] wrote {n} scene dir(s) under {args.out_base}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2：语法 + import 自检（无网络）**

Run: `cd /mnt/afs/davidwang/workspace/sana_wm_pipeline && python -c "import ast; ast.parse(open('experiments/data_production_smoke/prepare_jdvbbfb.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 6.3：本地模式冒烟（用临时合成 shard，无网络，验证 --local-root 路径拼接）**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python - <<'PY'
import io, tarfile, numpy as np, subprocess, tempfile
from pathlib import Path
# 造一个临时本地数据根：{root}/wds-FAKE/{index.jsonl, shards/FAKE-000000.tar}
root = Path(tempfile.mkdtemp()) / "ds"
g = root / "wds-FAKE"; (g / "shards").mkdir(parents=True)
key = "FAKE_sample0"
cam = io.BytesIO()
np.savez(cam, c2w=np.tile(np.eye(4, np.float32), (3, 1, 1)),
         K_px=np.zeros((3, 4), np.float32), fps=np.float32(30.0),
         width=np.int32(640), height=np.int32(360))
with tarfile.open(g / "shards" / "FAKE-000000.tar", "w") as tf:
    for ext, data in [(".mp4", b"VID"), (".camera.npz", cam.getvalue())]:
        ti = tarfile.TarInfo(f"{key}{ext}"); ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
(g / "index.jsonl").write_text(
    '{"sample_id":"x","key":"FAKE_sample0","shard":"shards/FAKE-000000.tar",'
    '"video_member":"FAKE_sample0.mp4","camera_member":"FAKE_sample0.camera.npz",'
    '"manifest":{"video":{"fps":30.0},"prompt":{"text":"hi"}}}\n')
out = root / "out"
subprocess.check_call(["python", "experiments/data_production_smoke/prepare_jdvbbfb.py",
                       "--local-root", str(root), "--group", "wds-FAKE",
                       "--shard-idx", "0", "--sample-limit", "0", "--out-base", str(out)])
assert (out / "FAKE_sample0" / "video.mp4").read_bytes() == b"VID"
assert (out / "FAKE_sample0" / "caption.txt").read_text() == "hi"
print("LOCAL MODE SMOKE OK")
PY
```
Expected: `LOCAL MODE SMOKE OK`

- [ ] **Step 6.4：提交**

```bash
git add experiments/data_production_smoke/prepare_jdvbbfb.py
git commit -m "feat(prep): prepare_jdvbbfb.py dual-mode (local-root + HF stream)"
```

---

## Task 7：`run_e2e_default_jdvbbfb.sh` 端到端编排

**Files:**
- Create: `experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh`

镜像 `run_e2e_default_omniworld.sh`，但 Stage 0 改为「调用 prepare_jdvbbfb.py 拉一个样本 + normalize」。

- [ ] **Step 7.1：写脚本**

```bash
#!/usr/bin/env bash
# Default 模式端到端：jdvbbfb-v3-full 单样本 → WebDataset shard + ATE 评估
#
# 用法：
#   bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh \
#     <group> <shard_idx> [<out_base>]
# 例：
#   bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh wds-DL3DV-ALL-2K 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 环境（CMCC 部署时由 sed 重写为 <YOUR_BASE>）──────────────────────────────
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
export DISABLE_XFORMERS=1
export VIPE_EXT_JIT=1

GROUP="${1:?Usage: $0 <group> <shard_idx> [out_base]}"
SHARD_IDX="${2:?Usage: $0 <group> <shard_idx> [out_base]}"
OUT_BASE="${3:-/mnt/afs/davidwang/workspace/data/jdvbbfb_smoke}"
SHARDS_DIR="${OUT_BASE}/shards_default"
mkdir -p "${OUT_BASE}" "${SHARDS_DIR}"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo " jdvbbfb Default E2E: group=${GROUP} shard=${SHARD_IDX}"
echo "========================================================================"

# ── Stage 0: 拉取一个样本 → scene 目录 ───────────────────────────────────────
# 数据来源：若 JDVBBFB_LOCAL_ROOT 已设（CMCC，数据在 externalstorage）→ 读本地；
#           否则走 HF 流式（H100 开发）。
echo "=== Stage 0: prepare 1 sample from ${GROUP} shard ${SHARD_IDX} ==="
if [ -n "${JDVBBFB_LOCAL_ROOT:-}" ]; then
  SRC_ARGS=(--local-root "${JDVBBFB_LOCAL_ROOT}")
  echo "  source: LOCAL ${JDVBBFB_LOCAL_ROOT}"
else
  SRC_ARGS=(--repo junchaoh-cs/jdvbbfb-v3-full)
  echo "  source: HF stream"
fi
python experiments/data_production_smoke/prepare_jdvbbfb.py \
  "${SRC_ARGS[@]}" --group "${GROUP}" --shard-idx "${SHARD_IDX}" \
  --sample-limit 1 --out-base "${OUT_BASE}"

# 取刚写出的 scene 目录（最新修改的、含 video.mp4 的子目录）
SCENE_DIR="$(find "${OUT_BASE}" -mindepth 1 -maxdepth 1 -type d -name "${GROUP#wds-}*" \
             -exec test -f '{}/video.mp4' \; -print | sort | tail -1)"
SCENE_ID="$(basename "${SCENE_DIR}")"
echo "scene: ${SCENE_DIR}"

# ── Stage 1: normalize → 1280x720 @16fps ─────────────────────────────────────
echo "=== Stage 1: normalize ==="
NORM_VIDEO="${SCENE_DIR}/normalized.mp4"
if [ ! -f "${NORM_VIDEO}" ]; then
  python - <<PYEOF
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
info = normalize_video(Path("${SCENE_DIR}/video.mp4"), Path("${NORM_VIDEO}"))
print(f"Normalized: {info.n_frames} frames @ {info.fps}fps ({info.width}x{info.height})")
PYEOF
fi

# ── Stage 2: Default mode VIPE SLAM (Pi3X + MoGe-2) ──────────────────────────
echo "=== Stage 2: Default mode (Pi3X + MoGe-2 + VIPE) ==="
VIPE_WORK="${SCENE_DIR}/vipe_work_default"
mkdir -p "${VIPE_WORK}"
ARTIFACT_JSON="${VIPE_WORK}/pose_artifact_default.json"
if [ ! -f "${ARTIFACT_JSON}" ]; then
  python - <<PYEOF
import json
from pathlib import Path
from sana_wm_pipeline.stage02_pose.mode_default import run_default
art = run_default(Path("${NORM_VIDEO}"), Path("${VIPE_WORK}"))
print(f"Poses {art.poses_c2w.shape}  Intr {art.intrinsics.shape}")
Path("${ARTIFACT_JSON}").write_text(json.dumps({
    "poses_c2w": art.poses_c2w.tolist(),
    "intrinsics": art.intrinsics.tolist(),
    "scale_per_frame": art.scale_per_frame.tolist(),
}))
PYEOF
fi

# ── Stage 6: pack WebDataset shard ───────────────────────────────────────────
echo "=== Stage 6: pack shard ==="
SHARD="${SHARDS_DIR}/shard-000001.tar"
python - <<PYEOF
import io, json, numpy as np, tarfile
from pathlib import Path
scene_id="${SCENE_ID}"
art=json.loads(Path("${ARTIFACT_JSON}").read_text())
poses=np.array(art["poses_c2w"],np.float32)
intr=np.array(art["intrinsics"],np.float32)        # (T,1,4)
scale=np.array(art["scale_per_frame"],np.float32)
cap=Path("${SCENE_DIR}/caption.txt").read_text()
def add_npy(tf,key,arr):
    b=io.BytesIO(); np.save(b,arr); raw=b.getvalue()
    ti=tarfile.TarInfo(f"{scene_id}.{key}"); ti.size=len(raw); tf.addfile(ti,io.BytesIO(raw))
with tarfile.open("${SHARD}","w") as tf:
    vb=Path("${NORM_VIDEO}").read_bytes()
    ti=tarfile.TarInfo(f"{scene_id}.mp4"); ti.size=len(vb); tf.addfile(ti,io.BytesIO(vb))
    add_npy(tf,"poses_c2w.npy",poses)
    add_npy(tf,"intrinsics.npy",intr)
    add_npy(tf,"scale.npy",scale)
    cb=cap.encode(); ti=tarfile.TarInfo(f"{scene_id}.caption.txt"); ti.size=len(cb); tf.addfile(ti,io.BytesIO(cb))
    meta=json.dumps({"scene_id":scene_id,"T":len(poses),"mode":"default","dataset":"jdvbbfb-v3-full","group":"${GROUP}"}).encode()
    ti=tarfile.TarInfo(f"{scene_id}.meta.json"); ti.size=len(meta); tf.addfile(ti,io.BytesIO(meta))
print(f"Shard: ${SHARD}")
PYEOF

# ── Schema check + Pose eval (vs gt_poses.npy) ───────────────────────────────
echo "=== Schema check ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode schema --shards-dir "${SHARDS_DIR}"
echo "=== Pose eval (vs GT c2w) ==="
python experiments/data_production_smoke/verify_and_eval.py \
  --mode pose-eval --shards-dir "${SHARDS_DIR}" \
  --scenes-dir "${OUT_BASE}" --out-dir "${SHARDS_DIR}/eval_output" || \
  echo "[note] pose-eval needs meta.scene_id == scene dir name"

echo "✓ jdvbbfb Default E2E 完成: ${SCENE_ID}"
```

- [ ] **Step 7.2：语法自检**

Run: `bash -n experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh && echo "bash syntax OK"`
Expected: `bash syntax OK`

> **注意**：verify_and_eval pose-eval 用 `meta.json` 的 `scene_id` 去 `scenes-dir/{scene_id}/gt_poses.npy` 找 GT。本脚本里 `scene_id == 目录名`，已对齐。GT c2w 帧率 = 原始 fps，脚本内 `orig_fps.txt` 已写入，eval 会自动下采样到 16fps。

- [ ] **Step 7.3：提交**

```bash
git add experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh
git commit -m "feat(e2e): run_e2e_default_jdvbbfb.sh single-sample default pipeline"
```

---

## Task 8：单样本端到端验证（真数据，H100）

**Files:**（无新增，运行验证）

- [ ] **Step 8.1：确认权重与环境就绪**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
ls /mnt/afs/davidwang/models/pi3x/model.safetensors
ls /mnt/afs/davidwang/models/moge2/model.pt
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"  # 预期 DavidxWang
```

- [ ] **Step 8.2：只跑 prepare（最快验证数据读取正确）**

```bash
python experiments/data_production_smoke/prepare_jdvbbfb.py \
  --group wds-DL3DV-ALL-2K --shard-idx 0 --sample-limit 1 \
  --out-base /mnt/afs/davidwang/workspace/data/jdvbbfb_smoke
# 预期：写出 1 个 scene 目录，video.mp4 ~10MB，caption=yes
ls /mnt/afs/davidwang/workspace/data/jdvbbfb_smoke/*/
# 预期含：video.mp4 gt_poses.npy gt_intrinsics.npy orig_fps.txt caption.txt vipe_ref_poses.npy
python -c "
import numpy as np, glob
d=sorted(glob.glob('/mnt/afs/davidwang/workspace/data/jdvbbfb_smoke/*/gt_poses.npy'))[0]
p=np.load(d); print('gt_poses', p.shape, p.dtype)   # 预期 (300,4,4) float32
"
```

- [ ] **Step 8.3：跑完整 default 端到端**

```bash
bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh wds-DL3DV-ALL-2K 0
# H100 约 20-30 分钟（首次 VIPE JIT ~2min + Pi3X ~10min + SLAM ~10min）
```

- [ ] **Step 8.4：确认产物**

```bash
SMK=/mnt/afs/davidwang/workspace/data/jdvbbfb_smoke
ls ${SMK}/shards_default/shard-000001.tar
python -c "
import tarfile
tf=tarfile.open('${SMK}/shards_default/shard-000001.tar')
print([m.name for m in tf.getmembers()])
# 预期 6 个成员: .mp4 .poses_c2w.npy .intrinsics.npy .scale.npy .caption.txt .meta.json
"
cat ${SMK}/shards_default/eval_output/pose_eval_summary.json 2>/dev/null || echo "(eval 可选)"
```

Expected: shard 含 6 个成员；schema check 通过；pose-eval 输出一个 ATE 数值（DL3DV 无 GT depth 约束，default 模式 ATE 量级 ~0.1m，与 REPRODUCTION_GUIDE 中 DL3DV default=127.7mm 同量级即算合理）。

- [ ] **Step 8.5：跑全量单测确认无回归**

```bash
pytest tests/ -v --tb=short
# 预期：原有测试全过 + 新增 8 个 jdvbbfb 测试全过
```

- [ ] **Step 8.6：提交验证记录**

```bash
git add -A
git commit -m "test(e2e): validate jdvbbfb default mode on DL3DV-ALL-2K shard0 sample0"
```

---

## Task 9：sources.yaml 登记 + 运行/打包文档

**Files:**
- Modify: `configs/sources.yaml`
- Create: `docs/JDVBBFB_DEFAULT_GUIDE.md`

- [ ] **Step 9.1：在 sources.yaml 末尾追加独立顶层键（不触碰 `sources` 总和断言）**

```yaml
# ── External pre-packed WDS corpus (not counted in paper Table 1 totals) ──
# 实测 2026-06-14：8 子集 / 475,954 样本 / 1,353 shards。
# 每样本 = {key}.mp4 + {key}.camera.npz(per_frame_camera_npz_v1)；caption 在 index.jsonl。
external_corpora:
  jdvbbfb_v3_full:
    type: huggingface_gated
    repo_id: junchaoh-cs/jdvbbfb-v3-full
    pose_mode: default                      # 用本管线 Pi3X+MoGe-2+VIPE 重标注
    groups:
      - {name: wds-DL3DV-ALL-2K,        samples: 9993,   shards: 87,  res: 1920x1080, fps: 30}
      - {name: wds-OmniWorld-Game,      samples: 6576,   shards: 65,  res: 1280x720,  fps: 24}
      - {name: wds-SpatialVID-hq,       samples: 365362, shards: 714, res: 1280x720,  fps: 59.94}
      - {name: wds-sekai-real-walking-hq, samples: 18208, shards: 287, res: 1280x720, fps: 30}
      - {name: wds-sekai-game-walking,  samples: 1618,   shards: 43,  res: 1920x1080, fps: 30}
      - {name: wds-sekai-game-drone,    samples: 932,    shards: 5,   res: 1920x1080, fps: 30}
      - {name: wds-RealEstate10K-360p,  samples: 73165,  shards: 143, res: 640x360,   fps: 30}
      - {name: wds-Context-as-Memory,   samples: 100,    shards: 9,   res: 640x360,   fps: 30}
    sample_layout:
      video_member: "{key}.mp4"
      camera_member: "{key}.camera.npz"     # c2w/w2c/K_px + vipe_* refs
      caption: "index.jsonl → manifest.prompt.text"
```

- [ ] **Step 9.2：验证 sources loader 仍通过（断言不破）**

Run: `pytest tests/test_sources.py -v`
Expected: PASS（`external_corpora` 是 loader 忽略的未知顶层键，不影响 `sources` 总和断言）

- [ ] **Step 9.3：写 `docs/JDVBBFB_DEFAULT_GUIDE.md`**

内容须含：数据集结构表（Task 背景的两张表）、单机运行命令（Task 8）、CMCC 打包/运行流程（见本计划「CMCC 执行步骤」节，逐条复制为可执行命令）。

- [ ] **Step 9.4：提交**

```bash
git add configs/sources.yaml docs/JDVBBFB_DEFAULT_GUIDE.md
git commit -m "docs(sources): register jdvbbfb-v3-full external corpus + default guide"
```

---

## Task 10：推送分支 + 开 PR（不合并到 master）

- [ ] **Step 10.1：推送**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
git push -u origin feat/jdvbbfb-default-adapt
```

- [ ] **Step 10.2：开 PR（draft）**

```bash
gh pr create --draft --base master --head feat/jdvbbfb-default-adapt \
  --title "feat: jdvbbfb-v3-full default-mode ingest adapter" \
  --body "新增 WDS 摄取层（jdvbbfb_wds.py + prepare_jdvbbfb.py + run_e2e_default_jdvbbfb.sh），
mode_default.py 不变。单样本 E2E 已在 DL3DV-ALL-2K shard0 验证。
🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## CMCC 大规模处理执行步骤（部署阶段，Task 8 验证通过后）

> 来源：`docker-images/cmcc/docs/1_RETROSPECTIVE.md` SOP + `2_RUNBOOK.md`。CMCC = 无外网、5×8 卡 K8s 容器。**核心铁律**：env 必须带工具链；env 解压到热盘；持久化只信 `filestorage`/`externalstorage`；`tar --dereference`；conda-pack env 用 `source bin/activate`。

### A. 源机器打包（H100，有外网）

- [ ] **A.1：克隆并装工具链的 cmcc 版 env**

```bash
SRC=/mnt/afs/davidwang/miniconda3/envs/sana_wm
CMCC=/mnt/afs/davidwang/miniconda3/envs/sana_wm-cmcc
conda create --clone $SRC --prefix $CMCC -y
conda install -p $CMCC -c nvidia/label/cuda-12.4.1 -c conda-forge \
  'gcc_linux-64=13' 'gxx_linux-64=13' \
  cuda-nvcc cuda-cudart cuda-cudart-dev cuda-cudart-static \
  cuda-curand cuda-curand-dev cuda-cublas cuda-cublas-dev -y
mkdir -p $CMCC/etc/conda/activate.d
cat > $CMCC/etc/conda/activate.d/cc_nvcc.sh <<'EOF'
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
EOF
```

- [ ] **A.2：JIT gate-keeper（必过）**——见 `1_RETROSPECTIVE.md` 第三步 JIT 编译脚本，确认输出 `JIT PASS`。

- [ ] **A.3：conda-pack**

```bash
source $SRC/../base/bin/activate 2>/dev/null; pip install conda-pack 2>/dev/null || true
conda-pack -p $CMCC -o /mnt/afs/davidwang/workspace/docker-images/out/sana_wm-cmcc.tar.gz -j 16 --compress-level 5
md5sum /mnt/afs/davidwang/workspace/docker-images/out/sana_wm-cmcc.tar.gz > /mnt/afs/davidwang/workspace/docker-images/out/sana_wm-cmcc.tar.gz.md5
```

- [ ] **A.4：项目代码打包（`--dereference`，排除大目录）**

```bash
cd /mnt/afs/davidwang/workspace
tar --dereference -czf docker-images/out/sana_wm-deploy.tar.gz \
  --exclude='*/.git' --exclude='*/.git/*' \
  --exclude='*/__pycache__' --exclude='*/data' --exclude='*/data/*' \
  --exclude='*/.pytest_cache' \
  -C /mnt/afs/davidwang/workspace sana_wm_pipeline
# 验证无 broken symlink（应空输出）
tar -tzvf docker-images/out/sana_wm-deploy.tar.gz | grep "^l" | head
```

- [ ] **A.5：管线权重打包（必须含 VIPE+MoGe-2+Pi3X；不含 SANA-WM 推理权重）**

> ⚠️ **SANA-WM 推理权重（`models/SANA-WM_bidirectional` 96G + `gemma-2-2b-it`）不打包**——用户后续在 CMCC 单独下载。
> **管线运行权重全部必打**，实测清单（路径来自 H100，2026-06-14）：
>
> | 权重 | 路径 | 大小 | 用途 |
> |------|------|-----:|------|
> | Pi3X | `models/pi3x/` | 5.1G | 相对深度 |
> | MoGe-2 | `models/moge2/` | 1.3G | metric anchor |
> | VIPE/SLAM priors | `cache/torch/hub/` | 1.5G | DROID-SLAM + GroundingDINO(662M) + SAM(358M) + AOT(226M) + UniDepth(144M) + GeoCalib(111M) |
> | bert-base-uncased | `cache/huggingface/hub/models--bert-base-uncased/` | ~0.4G | GroundingDINO 文本编码器 |

```bash
# 权重模型（pi3x + moge2），目标 CMCC 落点 $NEW_BASE/models/
tar --dereference -czf docker-images/out/sana_wm-models.tar.gz \
  -C /mnt/afs/davidwang/models pi3x moge2

# VIPE/SLAM + bert 缓存，目标 CMCC 落点 $NEW_BASE/cache/（对应 TORCH_HOME/HF_HOME）
tar --dereference -czf docker-images/out/sana_wm-caches.tar.gz \
  -C /mnt/afs/davidwang/cache \
  torch/hub \
  huggingface/hub/models--bert-base-uncased

# 校验：确认 6 个关键 VIPE 权重都在包里
tar -tzf docker-images/out/sana_wm-caches.tar.gz | grep -E \
  "droid_slam|groundingdino_swint_ogc|sam_vit_b|R50_DeAOTL|metric_depth_vit_small|geocalib/pinhole" | sort
md5sum docker-images/out/sana_wm-models.tar.gz docker-images/out/sana_wm-caches.tar.gz \
  > docker-images/out/weights.md5
```

- [ ] **A.6：上传 modelscope（本机已登录 modelscope，无需再设 token）**

```bash
# 本机 modelscope 已登录 → transfer_via_modelscope.sh 直接用
bash docker-images/transfer_via_modelscope.sh upload davidxwang/conda-cmcc docker-images/out/sana_wm-cmcc.tar.gz
bash docker-images/transfer_via_modelscope.sh upload davidxwang/conda-cmcc docker-images/out/sana_wm-deploy.tar.gz
bash docker-images/transfer_via_modelscope.sh upload davidxwang/conda-cmcc docker-images/out/sana_wm-models.tar.gz
bash docker-images/transfer_via_modelscope.sh upload davidxwang/conda-cmcc docker-images/out/sana_wm-caches.tar.gz
```

### B. 目标机器部署（CMCC，无外网，能访问 modelscope）

- [ ] **B.1：选热盘（`isfast` 实测 <1s）→ 记为 `NEW_BASE`**；关键产物周期 rsync 到 `externalstorage`/`filestorage`。
- [ ] **B.2：下载 + 解压 env 到热盘 + `./bin/conda-unpack` + `source bin/activate`**（不可用 `conda activate`）。
- [ ] **B.3：解压 deploy + models + caches，落点须与环境变量一致：**

```bash
# 代码 → $NEW_BASE/workspace/sana_wm_pipeline
mkdir -p $NEW_BASE/workspace && tar -xzf $NEW_BASE/sana_wm-deploy.tar.gz -C $NEW_BASE/workspace
# 权重 → $NEW_BASE/models/{pi3x,moge2}  (= SANA_WM_PI3X/MOGE2_WEIGHTS)
mkdir -p $NEW_BASE/models && tar -xzf $NEW_BASE/sana_wm-models.tar.gz -C $NEW_BASE/models
# VIPE/SLAM + bert 缓存 → $NEW_BASE/cache/{torch,huggingface}  (= TORCH_HOME / HF_HOME)
mkdir -p $NEW_BASE/cache && tar -xzf $NEW_BASE/sana_wm-caches.tar.gz -C $NEW_BASE/cache
# 校验 VIPE 6 件套就位
ls $NEW_BASE/cache/torch/hub/{droid_slam,sam,aot,geocalib,checkpoints}
```

- [ ] **B.4：`sed` 把 `/mnt/afs/davidwang` 全量重写为 `$NEW_BASE` 对应前缀**（含 `run_e2e_default_jdvbbfb.sh` 顶部 conda/TORCH_HOME/HF_HOME/权重路径；注意 `$NEW_BASE/cache` 对应原 `/mnt/afs/davidwang/cache`，`$NEW_BASE/models` 对应原 `/mnt/afs/davidwang/models`）。或直接在运行前用 C.1 的 `export` 覆盖，免 sed。
- [ ] **B.5：linker 软链补漏 + 清 `~/.cache/torch_extensions`**（见 RETROSPECTIVE 第十步）。

### C. CMCC 上跑大规模处理（数据已在持久盘，无需传输）

> ✅ **数据集已在 CMCC 就位**：`/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full/`
> （`externalstorage` = 两个持久化目录之一，重启不丢）。目录树与 HF repo 一致：
> `{root}/wds-<NAME>/{index.jsonl, shards/<NAME>-NNNNNN.tar}`。
> 因此 **C 阶段不需要 HF token、不需要 modelscope 传数据**——`prepare_jdvbbfb.py --local-root` 直接读。
> A/B 阶段仍需传输的只有：env tarball、项目代码、模型权重（Pi3X/MoGe-2）。

- [ ] **C.1：设数据根 + 单样本验证（先确认本地读取通）**

```bash
source $NEW_BASE/start_env.sh                       # source bin/activate（conda-pack env）
export JDVBBFB_LOCAL_ROOT=/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full
export TORCH_HOME=$NEW_BASE/cache/torch
export HF_HOME=$NEW_BASE/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=$NEW_BASE/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=$NEW_BASE/models/moge2
export DISABLE_XFORMERS=1 VIPE_EXT_JIT=1

cd $NEW_BASE/workspace/sana_wm_pipeline
# run_e2e 脚本检测到 JDVBBFB_LOCAL_ROOT → 自动走本地模式
bash experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh \
  wds-DL3DV-ALL-2K 0 $NEW_BASE/work/jdvbbfb_out      # 工作输出放热盘
```

- [ ] **C.2：批量准备一个 group 的全部 shard → scene 目录（本地读，全量 sample-limit 0）**

```bash
ROOT=$JDVBBFB_LOCAL_ROOT
GROUP=wds-DL3DV-ALL-2K
OUT=$NEW_BASE/work/jdvbbfb_out/$GROUP                 # 热盘
N_SHARDS=$(ls $ROOT/$GROUP/shards/*.tar | wc -l)
for i in $(seq 0 $((N_SHARDS-1))); do
  python experiments/data_production_smoke/prepare_jdvbbfb.py \
    --local-root "$ROOT" --group "$GROUP" --shard-idx "$i" \
    --sample-limit 0 --out-base "$OUT"
done
```

- [ ] **C.3：多卡并行跑 default 标注**——按 GPU 数把 scene 目录列表分片，每卡一个 `run_default` 进程：

```bash
# 列出所有待处理 scene，按 GPU 数轮转分配（8 卡示例）
mapfile -t SCENES < <(find $OUT -mindepth 1 -maxdepth 1 -type d | sort)
for gpu in $(seq 0 7); do
  ( for idx in "${!SCENES[@]}"; do
      [ $((idx % 8)) -ne $gpu ] && continue
      sc="${SCENES[$idx]}"
      CUDA_VISIBLE_DEVICES=$gpu python - <<PYEOF
from pathlib import Path
from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
from sana_wm_pipeline.stage02_pose.mode_default import run_default
import json
sc = Path("$sc")
nv = sc / "normalized.mp4"
if not nv.exists():
    normalize_video(sc / "video.mp4", nv)
wd = sc / "vipe_work_default"; wd.mkdir(exist_ok=True)
art = run_default(nv, wd)
(wd / "pose_artifact_default.json").write_text(json.dumps({
    "poses_c2w": art.poses_c2w.tolist(),
    "intrinsics": art.intrinsics.tolist(),
    "scale_per_frame": art.scale_per_frame.tolist()}))
print("done", sc.name)
PYEOF
    done ) &
done
wait
```

- [ ] **C.4：打包产出 shard + 周期备份到持久盘**

```bash
# Stage 6 打包逻辑同 run_e2e_default_jdvbbfb.sh，对每个 scene 产 shard 到热盘，
# 再 rsync 到 externalstorage（持久，不丢）：
rsync -a --info=progress2 $NEW_BASE/work/jdvbbfb_out/$GROUP/shards_default/ \
  /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_default_out/$GROUP/
```

> **铁律提醒**：工作目录用热盘（速度），关键产物（shard）周期 rsync 到 `externalstorage`/`filestorage`（持久）。
> 多机扩展时参考 `docker-images/cmcc/docs/5_MULTINODE_EXPLAINED.md` 的副本/RANK 注入方式按 GPU 分片。

---

## Self-Review

**Spec coverage：**
- ✅ CMCC 打包流程要点 → 「CMCC 大规模处理执行步骤」节 + 背景引用 RETROSPECTIVE。
- ✅ SANA-WM 项目状态 → 背景节已整合 REPRODUCTION_GUIDE 三模式 + default 内部接口。
- ✅ 数据集轻量化探索 → 背景节两张表 + camera.npz schema（全部实测，非猜测）。
- ✅ default 模式代码适配 → Task 1-7 新分支、新摄取层、单样本验证；mode_default.py 不污染。
- ✅ 输出要求四项 → CMCC 要点 / 结构报告 / 改动文件清单（文件结构表）/ 打包运行步骤 全覆盖。

**Placeholder scan：** 无 TBD/TODO；所有代码步骤含完整代码；CMCC 命令引用 RETROSPECTIVE 具体脚本路径。`prepare_jdvbbfb.py` 的本地模式（`--local-root`）已是 Task 6 一等实现并有 Step 6.3 本地冒烟覆盖，CMCC C 阶段直接调用，非占位。

**2026-06-14 修订一致性：** 数据集已在 CMCC `externalstorage` 就位 → `--local-root` 为 CMCC 主路径，`--repo`(HF 流式) 仅 H100 开发用；`run_e2e_default_jdvbbfb.sh` 用 `JDVBBFB_LOCAL_ROOT` 环境变量切换两模式；A.4 deploy tar 已 `--exclude '*/data'`（数据不打包）。

**Type consistency：** `CameraGT`(c2w/k_px/fps/width/height/vipe_c2w)、`SampleRef`(key/shard/video_member/camera_member/caption/fps)、`iter_tar_samples`→(key,mp4_bytes,camera_bytes)、`write_scene_dir`(out_base,scene_id,mp4_bytes,camera_npz_bytes,caption) 在 Task 2-7 各调用点签名一致。scene 目录布局与 `prepare_omniworld.py` / `verify_and_eval.py`（读 `gt_poses.npy`+`orig_fps.txt`）一致。

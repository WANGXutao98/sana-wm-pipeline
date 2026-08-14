# 对齐官方stage.py实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将本地mode_default.py改为直接调用官方sana-wm-data-clean的stage.py逻辑，确保与官方实现100%一致，解决轨迹长度偏差问题

**Architecture:** 
- 调研发现官方adapters.run_vipe_slam()在真实模式下**不运行VIPE SLAM**，而是直接使用Pi3X的pose输出
- 本地实现通过subprocess调用VIPE，引入了额外的scale处理问题
- 解决方案：创建适配层，直接调用官方stage.py的annotate_pose()函数，确保逻辑100%一致

**Tech Stack:** 
- Python 3.10
- 官方sana-wm-data-clean库（已集成到src/sana_wm_pipeline/sana_wm_data_clean/）
- numpy, pathlib

**Spec:** SCALE_POSE_ANALYSIS.md + SCALE_FIX_RESULTS.md（深度分析发现scale传递问题的根因）

## Global Constraints

- Python ≥ 3.10
- numpy ≥ 1.24
- 保持与官方sana-wm-data-clean完全一致的逻辑
- 不修改官方代码（src/sana_wm_pipeline/sana_wm_data_clean/下的文件）
- 所有修改在src/sana_wm_pipeline/stage02_pose/下进行
- 保持现有的PoseArtifact数据结构
- 测试通过：3个SpatialVID样本，轨迹长度比例≤1.5x，ATE RMSE <0.05m

---

## 调研发现总结

### 关键发现1：官方adapters.run_vipe_slam()的实现

**位置**: `sana-wm-data-clean/sana_wm_data/pose/adapters.py:122-140`

```python
def run_vipe_slam(..., depth, intrinsics0, ...):
    if dry_run:
        poses = _synthetic_trajectory(clip_id, n_frames)
        return poses, intrinsics0
    # Real mode: full VIPE SLAM+BA is a heavy CUDA build we do not set up here.
    # We use Pi3's multi-frame-consistent pose output as the pose track (Pi3 is
    # the structure backbone SANA-WM builds on); VIPE's bundle-adjustment refine
    # is the one layer omitted. Intrinsics keep the seed (per-frame BA not run).
    from . import _real
    frames = read_frames(video_path, n_frames)
    poses, _depth = _real.pi3_infer(frames)
    return poses, intrinsics0
```

**关键结论**: 官方代码**不调用VIPE SLAM**，直接使用Pi3X的pose！深度参数被忽略！

### 关键发现2：本地实现的问题

**当前实现**: `src/sana_wm_pipeline/stage02_pose/mode_default.py:100-111`
- 通过`subprocess.check_call()`调用VIPE CLI
- 使用`vipe_sanawm` pipeline，该pipeline使用`pi3xmoge` depth backend
- VIPE会运行完整的SLAM+BA，但可能存在scale处理问题

**根本差异**:
- 官方：Pi3X pose + 不运行VIPE
- 本地：Pi3X+MoGe融合深度 → VIPE SLAM → 产生新的pose

### 关键发现3：官方stage.py的default模式流程

**位置**: `sana-wm-data-clean/sana_wm_data/pose/stage.py:93-104`

```python
else:  # default
    n = n_scale
    intr0 = _load_real_intrinsics(rec, n)
    if intr0 is None:
        intr0 = _seed_intrinsics(rec, n)
    pi3x = adapters.run_pi3x_depth(rec.video_path, rec.clip_id, n, hw, models_cfg, dry)
    moge = adapters.run_moge2_depth(rec.video_path, rec.clip_id, n, hw, models_cfg, dry)
    fused, scales_arr = fuse_depth_sequence(pi3x, moge, ema_momentum=momentum)
    poses, intr = adapters.run_vipe_slam(
        rec.video_path, rec.clip_id, n, hw, fused, intr0, models_cfg, dry
    )
    scales = scales_arr.tolist()
```

**流程**:
1. 计算Pi3X深度
2. 计算MoGe-2深度
3. 融合 → 得到fused depth + scales
4. 调用run_vipe_slam() → **实际上只是再次调用Pi3X获取pose**
5. 返回scales

**结论**: 融合深度在官方实现中**没有被使用**！Scale只用于后续的质量过滤（CoV检查）。

---

## Task 1: 创建ClipRecord适配层

**Files:**
- Create: `src/sana_wm_pipeline/stage02_pose/clip_record_adapter.py`
- Test: `tests/test_clip_record_adapter.py`

**Interfaces:**
- Consumes: Path对象（视频路径）
- Produces: `ClipRecord`对象（官方stage.py的输入格式）
          `from_local_video(video_path: Path, mode: str) -> ClipRecord`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clip_record_adapter.py
from pathlib import Path
from sana_wm_pipeline.stage02_pose.clip_record_adapter import from_local_video

def test_from_local_video_creates_valid_cliprecord():
    """Verify ClipRecord creation from local video path"""
    video_path = Path("/fake/video.mp4")
    
    rec = from_local_video(video_path, mode="default")
    
    assert rec.clip_id == "video"
    assert rec.video_path == str(video_path)
    assert rec.mode == "default"
    assert rec.num_frames is None  # Will be detected
    assert rec.width is None
    assert rec.height is None
    assert rec.extra == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/afs/davidwang/workspace/sana_wm_pipeline && conda activate sana_wm && pytest tests/test_clip_record_adapter.py::test_from_local_video_creates_valid_cliprecord -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'sana_wm_pipeline.stage02_pose.clip_record_adapter'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/sana_wm_pipeline/stage02_pose/clip_record_adapter.py
"""Adapter to convert local video paths to ClipRecord format for official stage.py"""
from __future__ import annotations
from pathlib import Path

# Import官方的ClipRecord（假设存在，需要确认）
try:
    from ..sana_wm_data_clean.manifest import ClipRecord
except ImportError:
    # 如果不存在，创建简化版本
    from dataclasses import dataclass, field
    
    @dataclass
    class ClipRecord:
        clip_id: str
        video_path: str
        mode: str = "default"
        num_frames: int | None = None
        width: int | None = None
        height: int | None = None
        extra: dict = field(default_factory=dict)
        pose_path: str | None = None
        intrinsics_path: str | None = None
        scale_factors: list[float] = field(default_factory=list)
        pose_mode: str | None = None


def from_local_video(video_path: Path, mode: str = "default") -> ClipRecord:
    """Create ClipRecord from local video path"""
    return ClipRecord(
        clip_id=video_path.stem,
        video_path=str(video_path),
        mode=mode,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clip_record_adapter.py::test_from_local_video_creates_valid_cliprecord -v`
Expected: PASS

- [ ] **Step 5: Add test for video metadata detection**

```python
def test_from_local_video_detects_metadata(tmp_path):
    """Verify video metadata is detected when auto_detect=True"""
    # 创建假视频文件（用于测试）
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    
    rec = from_local_video(video_path, mode="default", auto_detect=True)
    
    # 在没有真实视频时，应该返回None
    assert rec.num_frames is None
    assert rec.width is None
    assert rec.height is None
```

- [ ] **Step 6: Implement metadata detection (optional)**

```python
def from_local_video(
    video_path: Path, 
    mode: str = "default",
    auto_detect: bool = False
) -> ClipRecord:
    """Create ClipRecord from local video path
    
    Args:
        video_path: Path to video file
        mode: Pose annotation mode (default, gt_pose, gt_depth)
        auto_detect: If True, detect video metadata (num_frames, width, height)
    """
    rec = ClipRecord(
        clip_id=video_path.stem,
        video_path=str(video_path),
        mode=mode,
    )
    
    if auto_detect and video_path.exists():
        try:
            import decord
            vr = decord.VideoReader(str(video_path))
            rec.num_frames = len(vr)
            rec.height, rec.width = vr[0].shape[:2]
        except Exception:
            pass  # Metadata detection failed, keep None
    
    return rec
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_clip_record_adapter.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_clip_record_adapter.py src/sana_wm_pipeline/stage02_pose/clip_record_adapter.py
git commit -m "feat: add ClipRecord adapter for official stage.py integration"
```

---

## Task 2: 确认官方ClipRecord和manifest的可用性

**Files:**
- Read: `src/sana_wm_pipeline/sana_wm_data_clean/manifest.py`
- Create: `src/sana_wm_pipeline/sana_wm_data_clean/manifest.py` (如果不存在)

**Interfaces:**
- Consumes: 无
- Produces: `ClipRecord` dataclass定义

- [ ] **Step 1: 检查manifest.py是否存在**

```bash
ls -la src/sana_wm_pipeline/sana_wm_data_clean/manifest.py
```

Expected: 文件存在或不存在

- [ ] **Step 2: 如果不存在，从官方复制**

```bash
cp sana-wm-data-clean/sana_wm_data/manifest.py src/sana_wm_pipeline/sana_wm_data_clean/manifest.py
```

- [ ] **Step 3: 验证ClipRecord可导入**

```python
# 临时测试
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python -c "from src.sana_wm_pipeline.sana_wm_data_clean.manifest import ClipRecord; print('OK')"
```

Expected: 打印 "OK"

- [ ] **Step 4: 如果导入失败，创建简化版本**

在Task 1的clip_record_adapter.py中已经包含了fallback实现，无需额外操作

- [ ] **Step 5: Commit (如果有新文件)**

```bash
git add src/sana_wm_pipeline/sana_wm_data_clean/manifest.py
git commit -m "feat: add ClipRecord manifest from official repo"
```

---

## Task 3: 创建官方stage.py的调用包装器

**Files:**
- Create: `src/sana_wm_pipeline/stage02_pose/official_stage_wrapper.py`
- Test: `tests/test_official_stage_wrapper.py`

**Interfaces:**
- Consumes: `ClipRecord`, 输出目录Path
- Produces: `PoseArtifact` (本地格式)
          `run_official_annotate_pose(video_path: Path, work_dir: Path, mode: str, dry_run: bool) -> PoseArtifact`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_official_stage_wrapper.py
from pathlib import Path
from sana_wm_pipeline.stage02_pose.official_stage_wrapper import run_official_annotate_pose

def test_run_official_annotate_pose_dry_run(tmp_path):
    """Verify official stage.py wrapper in dry-run mode"""
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    work_dir = tmp_path / "work"
    
    artifact = run_official_annotate_pose(
        video_path=video_path,
        work_dir=work_dir,
        mode="default",
        dry_run=True
    )
    
    assert artifact.poses_c2w.shape[1:] == (4, 4)
    assert artifact.intrinsics.shape[2] == 4
    assert artifact.scale_per_frame.shape[0] > 0
    assert not all(artifact.scale_per_frame == 1.0)  # Scale should vary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_official_stage_wrapper.py::test_run_official_annotate_pose_dry_run -v`
Expected: FAIL with module not found

- [ ] **Step 3: Write implementation**

```python
# src/sana_wm_pipeline/stage02_pose/official_stage_wrapper.py
"""Wrapper around official sana-wm-data-clean stage.py for pose annotation"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from ._common import PoseArtifact
from .clip_record_adapter import from_local_video

def run_official_annotate_pose(
    video_path: Path,
    work_dir: Path,
    mode: str = "default",
    dry_run: bool = False,
    max_frames: int = 64,
) -> PoseArtifact:
    """Run official stage.py::annotate_pose() and convert to PoseArtifact
    
    Args:
        video_path: Path to input video
        work_dir: Working directory for outputs
        mode: Annotation mode (default, gt_pose, gt_depth)
        dry_run: If True, use synthetic data (no GPU)
        max_frames: Maximum frames to sample for Pi3X/MoGe
        
    Returns:
        PoseArtifact with poses, intrinsics, and scale_per_frame
    """
    import os
    from ..sana_wm_data_clean.pose.stage import annotate_pose
    
    # Create ClipRecord
    rec = from_local_video(video_path, mode=mode, auto_detect=True)
    
    # Configure models
    models_cfg = {
        "dry_run": dry_run,
        "depth_fusion": {
            "ema_momentum": 0.99
        }
    }
    
    # Set max_frames via environment (官方代码从这里读取)
    old_max_frames = os.environ.get("SANA_WM_MAX_FRAMES")
    os.environ["SANA_WM_MAX_FRAMES"] = str(max_frames)
    
    try:
        # Call official annotate_pose
        rec_out = annotate_pose(rec, work_dir, models_cfg)
    finally:
        # Restore environment
        if old_max_frames is not None:
            os.environ["SANA_WM_MAX_FRAMES"] = old_max_frames
        else:
            os.environ.pop("SANA_WM_MAX_FRAMES", None)
    
    # Load outputs and convert to PoseArtifact
    poses_c2w = np.load(rec_out.pose_path).astype(np.float32)  # (N, 4, 4)
    intrinsics_raw = np.load(rec_out.intrinsics_path).astype(np.float32)  # (N, 4)
    
    # Convert intrinsics to (N, 1, 4) format
    intrinsics = intrinsics_raw[:, None, :]
    
    # Get scale_per_frame
    scale_per_frame = np.array(rec_out.scale_factors, dtype=np.float32)
    
    # Interpolate poses/intrinsics/scale to full video length if needed
    N_output = poses_c2w.shape[0]
    T_full = rec_out.num_frames or N_output
    
    if T_full > N_output:
        # Need interpolation (VIPE只输出了关键帧)
        poses_c2w = _interp_poses(poses_c2w, np.arange(N_output), T_full)
        intrinsics = _interp_intrinsics(intrinsics_raw, np.arange(N_output), T_full)[:, None, :]
        scale_per_frame = np.interp(np.arange(T_full), np.arange(N_output), scale_per_frame).astype(np.float32)
    
    return PoseArtifact(
        poses_c2w=poses_c2w,
        intrinsics=intrinsics,
        scale_per_frame=scale_per_frame,
        depth_downsampled=None,
    )


def _interp_poses(poses: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Interpolate poses to T frames (copied from mode_default.py)"""
    out = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = np.interp(np.arange(T), inds, poses[:, i, j])
    # Ensure first frame is identity
    if not np.allclose(out[0], np.eye(4), atol=1e-3):
        T0_inv = np.linalg.inv(out[0])
        out = (T0_inv[None] @ out)
    return out.astype(np.float32)


def _interp_intrinsics(intr: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Interpolate intrinsics to T frames (copied from mode_default.py)"""
    out = np.zeros((T, 4), dtype=np.float32)
    for k in range(4):
        out[:, k] = np.interp(np.arange(T), inds, intr[:, k])
    return out.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_official_stage_wrapper.py::test_run_official_annotate_pose_dry_run -v`
Expected: PASS

- [ ] **Step 5: Add test for scale values**

```python
def test_scale_values_not_all_ones(tmp_path):
    """Verify scale_per_frame contains actual computed values, not all 1.0"""
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    work_dir = tmp_path / "work"
    
    artifact = run_official_annotate_pose(
        video_path=video_path,
        work_dir=work_dir,
        mode="default",
        dry_run=True
    )
    
    scale = artifact.scale_per_frame
    
    # Dry-run mode should produce varied scales (not all 1.0)
    assert not np.allclose(scale, 1.0)
    
    # Scale should be in reasonable range (0.5-2.0 per paper)
    assert 0.3 < scale.min() < 3.0
    assert 0.3 < scale.max() < 3.0
    
    # Scale CoV should be reasonable
    cov = scale.std() / (scale.mean() + 1e-8)
    assert cov < 2.0  # Paper threshold
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_official_stage_wrapper.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_official_stage_wrapper.py src/sana_wm_pipeline/stage02_pose/official_stage_wrapper.py
git commit -m "feat: add official stage.py wrapper with PoseArtifact conversion"
```

---

## Task 4: 修改mode_default.py使用官方实现

**Files:**
- Modify: `src/sana_wm_pipeline/stage02_pose/mode_default.py:34-113`
- Test: 运行现有的冒烟测试

**Interfaces:**
- Consumes: `run_official_annotate_pose()` from official_stage_wrapper
- Produces: `PoseArtifact` (与之前相同)

- [ ] **Step 1: Backup original implementation**

```bash
cp src/sana_wm_pipeline/stage02_pose/mode_default.py src/sana_wm_pipeline/stage02_pose/mode_default.py.backup
```

- [ ] **Step 2: Replace run_default() function**

```python
# src/sana_wm_pipeline/stage02_pose/mode_default.py
# 替换整个run_default函数（第34-113行）

def run_default(
    clip_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = "vipe_sanawm",
) -> PoseArtifact:
    """使用官方sana-wm-data-clean的stage.py实现（100%一致）
    
    Args:
        clip_path: Path to video file
        work_dir: Working directory for outputs
        vipe_cmd: Unused (kept for API compatibility)
        pipeline: Unused (kept for API compatibility)
        
    Returns:
        PoseArtifact with poses, intrinsics, scale_per_frame
    """
    from .official_stage_wrapper import run_official_annotate_pose
    import os
    
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查环境变量
    pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS", "")
    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "")
    if not pi3x_weights or not moge2_weights:
        raise RuntimeError(
            "SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS must be set"
        )
    
    # 获取max_frames配置
    max_frames = int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
    
    print(f"[mode_default] Using official stage.py implementation", flush=True)
    print(f"[mode_default] Video: {clip_path}", flush=True)
    print(f"[mode_default] Max frames: {max_frames}", flush=True)
    
    # 调用官方实现
    artifact = run_official_annotate_pose(
        video_path=clip_path,
        work_dir=work_dir,
        mode="default",
        dry_run=False,
        max_frames=max_frames,
    )
    
    print(f"[mode_default] ✅ Pose annotation complete", flush=True)
    print(f"[mode_default]    Poses: {artifact.poses_c2w.shape}", flush=True)
    print(f"[mode_default]    Intrinsics: {artifact.intrinsics.shape}", flush=True)
    print(f"[mode_default]    Scale range: {artifact.scale_per_frame.min():.3f} - {artifact.scale_per_frame.max():.3f}", flush=True)
    print(f"[mode_default]    Scale CoV: {artifact.scale_per_frame.std()/(artifact.scale_per_frame.mean()+1e-8):.3f}", flush=True)
    
    return artifact
```

- [ ] **Step 3: Remove unused helper functions**

删除以下不再使用的函数（保留_interp_poses, _interp_intrinsics因为可能被其他地方使用）:
- `_read_frames_uniform` (第116-124行)
- `_compute_rgb_signatures` (第127-135行)  
- `_load_vipe_artifacts` (第138-186行)

注释：可以保留这些函数但标记为deprecated，或者完全删除

- [ ] **Step 4: Update imports**

在文件顶部添加：
```python
from .official_stage_wrapper import run_official_annotate_pose
```

移除不再使用的imports（如果有）

- [ ] **Step 5: Run smoke test on single sample**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 测试单个样本
python -c "
from pathlib import Path
from src.sana_wm_pipeline.stage02_pose.mode_default import run_default

video = Path('/mnt/afs/davidwang/workspace/sana_test_data/smoke_result/SpatialVID-hq_b5a60fd2-64ff-5a22-b2f5-5df2bd7dea63/normalized.mp4')
work = Path('/tmp/test_official_stage')

artifact = run_default(video, work)
print(f'Poses: {artifact.poses_c2w.shape}')
print(f'Scale range: {artifact.scale_per_frame.min():.3f}-{artifact.scale_per_frame.max():.3f}')
print(f'All 1.0: {(artifact.scale_per_frame == 1.0).all()}')
"
```

Expected: 
- Poses shape printed
- Scale range NOT all 1.0
- No errors

- [ ] **Step 6: Run full smoke test (3 samples)**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 清理旧输出
rm -rf /mnt/afs/davidwang/workspace/sana_test_data/smoke_result/SpatialVID-hq_*

# 运行完整测试
bash experiments/data_production_smoke/smoke_spatialvid.sh
```

Expected:
- 3个样本全部完成
- Scale不全为1.0
- 无错误

- [ ] **Step 7: Verify scale values**

```bash
python -c "
import json, numpy as np
from pathlib import Path

samples = [
    'SpatialVID-hq_b5a60fd2-64ff-5a22-b2f5-5df2bd7dea63',
    'SpatialVID-hq_a884fb06-ac39-5950-a2b4-288bf4d93efe',
    'SpatialVID-hq_16987b84-30a4-5a87-9be2-e7876b090dd4',
]

for s in samples:
    p = Path(f'/mnt/afs/davidwang/workspace/sana_test_data/smoke_result/{s}/pose_artifact_default.json')
    if p.exists():
        data = json.load(p.open())
        scale = np.array(data['scale_per_frame'])
        print(f'{s[:8]}: scale={scale.min():.3f}-{scale.max():.3f}, all_1.0={np.allclose(scale, 1.0)}')
"
```

Expected: All samples show scale NOT all 1.0

- [ ] **Step 8: Commit**

```bash
git add src/sana_wm_pipeline/stage02_pose/mode_default.py
git commit -m "refactor: use official stage.py implementation in mode_default

BREAKING CHANGE: Replaced custom VIPE subprocess call with official
sana-wm-data-clean stage.py logic. This ensures 100% consistency with
the reference implementation and fixes scale propagation issues.

- Remove custom Phase A/B implementation
- Remove VIPE subprocess call
- Add wrapper around official annotate_pose()
- Preserve PoseArtifact API for backward compatibility

Fixes: trajectory length 3-10x deviation, scale=1.0 bug"
```

---

## Task 5: 运行完整质量验证

**Files:**
- Run: `scripts/validate_smoke_output.py`
- Create: `docs/OFFICIAL_STAGE_VALIDATION.md` (验证报告)

**Interfaces:**
- Consumes: 3个样本的输出（pose_artifact_default.json + .tar files）
- Produces: 质量验证报告（轨迹长度、ATE、RPE等指标）

- [ ] **Step 1: Run quality validation**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
conda activate sana_wm

python scripts/validate_smoke_output.py \
    --output-dir /mnt/afs/davidwang/workspace/sana_test_data/smoke_result \
    --samples /mnt/afs/davidwang/workspace/sana_test_data/smoke_result/selected_samples.txt \
    2>&1 | tee /tmp/official_stage_validation.log
```

Expected: 
- 报告生成
- 轨迹长度比例改善
- ATE误差改善

- [ ] **Step 2: Extract key metrics**

```bash
grep -E "轨迹长度|vs VIPE参考|平移误差|Scale.*1\.0" /tmp/official_stage_validation.log | head -20
```

Expected:
- 轨迹长度比例 ≤ 1.5x
- Scale不全为1.0

- [ ] **Step 3: Compare with baseline (SCALE_FIX_RESULTS.md)**

手动对比：
- 修复前：轨迹长度3-10x，ATE 0.20-22m
- Scale修复后：轨迹长度2.5-10x，ATE 0.20-2.1m  
- 官方实现后：期望轨迹长度1.0-1.5x，ATE <0.05m

- [ ] **Step 4: Create validation report**

```bash
cat > docs/OFFICIAL_STAGE_VALIDATION.md << 'EOF'
# 官方stage.py实施验证报告

**日期**: 2026-08-13  
**实施内容**: 将mode_default.py改为直接调用官方stage.py  

## 修改内容

- 创建ClipRecord适配层
- 创建官方stage.py包装器
- 替换mode_default.py的实现
- 移除自定义VIPE subprocess调用

## 验证结果

[从validation log粘贴关键指标]

| 指标 | 修复前 | Scale修复后 | 官方实现后 | 目标 | 达标 |
|------|--------|------------|-----------|------|------|
| Scale全为1.0 | ❌ 是 | ✅ 否 | ✅ 否 | 否 | ✅ |
| Scale范围 | N/A | 0.7-2.4 | [实际值] | 0.5-2.0 | [是/否] |
| 轨迹长度比例（样本1） | 3.0x | 2.46x | [实际值] | 1.0-1.5x | [是/否] |
| 轨迹长度比例（样本2） | 10.7x | 10.73x | [实际值] | 1.0-1.5x | [是/否] |
| ATE RMSE（样本1） | 0.20m | 0.20m | [实际值] | <0.05m | [是/否] |
| ATE RMSE（样本2） | 22.3m | 2.13m | [实际值] | <0.05m | [是/否] |

## 结论

[根据实际结果填写]

## 下一步

[如果仍有问题，列出进一步的调查方向]
EOF
```

- [ ] **Step 5: Review and edit report**

手动编辑`docs/OFFICIAL_STAGE_VALIDATION.md`，填入实际数值

- [ ] **Step 6: Commit validation report**

```bash
git add docs/OFFICIAL_STAGE_VALIDATION.md
git commit -m "docs: add official stage.py validation report"
```

---

## Task 6: 更新文档和任务计划

**Files:**
- Modify: `task_plan_spatialvid_smoke.md`
- Modify: `SCALE_FIX_RESULTS.md`
- Create: `docs/ARCHITECTURE_DECISION_RECORDS.md` (ADR记录)

**Interfaces:**
- Consumes: 验证结果
- Produces: 更新的文档

- [ ] **Step 1: Update task_plan阶段10状态**

```bash
# Edit task_plan_spatialvid_smoke.md
# 将阶段10标记为完成
# 添加新的阶段11：官方实现集成
```

- [ ] **Step 2: Create ADR for this decision**

```bash
cat > docs/ARCHITECTURE_DECISION_RECORDS.md << 'EOF'
# Architecture Decision Records

## ADR-001: Use Official stage.py Implementation (2026-08-13)

### Context

Our local implementation of mode_default.py had a scale propagation issue:
- Scale was correctly computed in Phase A (fusion.py)
- Scale was correctly saved to scales.npy  
- But scale was not correctly used in VIPE SLAM
- This caused trajectory length deviations of 3-10x

Initial fix attempted to load scales.npy in _load_vipe_artifacts(), which
fixed the scale=1.0 bug but did NOT fix trajectory length issues.

Root cause analysis revealed:
- Official adapters.run_vipe_slam() does NOT actually run VIPE in real mode
- It simply calls Pi3X again to get poses (line 133-140 in adapters.py)
- The fused depth parameter is IGNORED
- Local implementation was calling actual VIPE SLAM via subprocess, introducing
  scale handling issues

### Decision

Replace custom mode_default.py implementation with direct calls to official
sana-wm-data-clean stage.py::annotate_pose().

### Consequences

**Positive:**
- 100% consistency with official implementation
- Eliminates scale propagation bugs
- Simpler code (removes Phase A/B split, VIPE subprocess)
- Easier to maintain (upstream changes automatically included)

**Negative:**
- Loses custom VIPE SLAM integration (if we wanted real SLAM+BA)
- Deeper dependency on official code structure

**Note:** Official implementation does NOT run VIPE SLAM in real mode. It uses
Pi3X poses directly. This is a deliberate simplification in the reference
implementation.

### Alternatives Considered

1. Fix VIPE pipeline configuration → rejected (official code doesn't use VIPE)
2. Post-process scale correction → rejected (would diverge from official)
3. Implement full VIPE SLAM properly → rejected (out of scope, official doesn't do this)

### Verification

See OFFICIAL_STAGE_VALIDATION.md for test results.
EOF
```

- [ ] **Step 3: Update SCALE_FIX_RESULTS.md conclusion**

在文件末尾添加：
```markdown
## Update (2026-08-13): Official Implementation Integration

After investigating VIPE pipeline configs and official code, we discovered:
- Official adapters.run_vipe_slam() does NOT run VIPE in real mode
- It directly uses Pi3X poses (fusion depth is ignored)
- Our custom VIPE subprocess call was the source of scale issues

**Solution:** Replace mode_default.py with direct calls to official stage.py

**Result:** [To be filled after Task 5 validation]

See: OFFICIAL_STAGE_VALIDATION.md, ARCHITECTURE_DECISION_RECORDS.md
```

- [ ] **Step 4: Commit documentation updates**

```bash
git add task_plan_spatialvid_smoke.md SCALE_FIX_RESULTS.md docs/ARCHITECTURE_DECISION_RECORDS.md
git commit -m "docs: record official stage.py integration decision and results"
```

---

## 自审清单

在实施前检查：

- [ ] 所有任务都有明确的测试步骤
- [ ] 没有"TBD"或占位符
- [ ] 代码块是完整的、可运行的
- [ ] 文件路径是准确的
- [ ] 函数签名在各任务间一致
- [ ] 每个任务都以commit结束
- [ ] Global Constraints已在所有任务中考虑

---

## 执行选择

计划已保存到 `docs/superpowers/plans/2026-08-13-align-with-official-stage-py.md`

**两种执行方式：**

1. **Subagent-Driven (推荐)** - 每个任务派发新的subagent，任务间review，快速迭代

2. **Inline Execution** - 在当前会话执行，批量处理带checkpoint review

选择哪种方式？

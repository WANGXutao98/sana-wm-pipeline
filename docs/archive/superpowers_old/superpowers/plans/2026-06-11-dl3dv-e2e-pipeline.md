# 计划：SANA-WM 数据标注管线 DL3DV 端到端打通 + SANA-WM 世界模型推理验证

> **创建日期**：2026-06-11  
> **工作目录**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/`  
> **持久化说明**：`~/.claude` → `/mnt/afs/davidwang/workspace/.claude`，方案文件跨重启持久

---

## 一、背景与目标

**现状**：管线代码完成（140 pytest），但从未在真实数据上端到端跑过。5 个硬阻塞项阻止在 DL3DV 上运行。

**目标**：
1. 修复阻塞项，用 5–8 个 DL3DV-ALL-2K 场景跑通 Stage 01–06（两种 PoseMode）
2. 产出 WebDataset shard（video.mp4 + poses_c2w.npy + intrinsics.npy + scale.npy + caption.txt + meta.json）
3. 从 shard 提取标注数据，调用 SANA-WM 推理（`Sana/inference_video_scripts/inference_sana_wm.py`）
4. 对比生成视频 vs DL3DV GT 视频，验证管线标注质量

---

## 二、关键已知事实（探索阶段确认）

| 事实 | 细节 |
|---|---|
| Pi3X 输出 key | `local_points`(1,T,H,W,3)、`camera_poses`(1,T,4,4)、`rays`(1,T,H,W,3)、`conf`、`metric` |
| `camera_poses[0,t,:3,3]` | = 相机中心（世界坐标，全局一致坐标系，可直接用于 Umeyama） |
| `pts_pi3x.npy` | 写出但从不读回（mode_gtpose.py 只用 cams_pi3x.json）|
| `sana_wm_pi3x_moge2.yaml` SLAM 阶段 | 调用 `_estimate_single()`（只有 MoGe-2，Pi3X 不进 BA）|
| 论文对齐方案 | 预计算缓存 → `vipe_cached_depth.yaml`（与 TUM 实验一致）|
| SANA-WM 推理接口 | `--image first_frame.png --prompt caption.txt --camera poses.npy --intrinsics intrinsics.npy --num_frames N` |
| `--intrinsics` 接受格式 | `(3,3)` / `(F,3,3)` / `(4,)=(fx,fy,cx,cy)`，我们的 `(T,1,4)` 需 reshape |
| SANA-WM 权重 | 本地无，需从 HF 下载：`Efficient-Large-Model/SANA-WM_bidirectional` |
| SANA-WM 推理代码 | `/mnt/afs/davidwang/workspace/Sana/inference_video_scripts/inference_sana_wm.py` |
| DL3DV 约定 | OpenCV c2w（X 右 Y 下 Z 前），与管线一致，无需坐标转换 |

---

## 三、变更文件汇总

### 修改的已有文件（6 个）

| 文件 | 改动 |
|---|---|
| `src/sana_wm_pipeline/stage02_pose/mode_default.py` | 两阶段：预计算深度缓存 → `vipe_cached_depth.yaml` SLAM |
| `src/sana_wm_pipeline/stage06_pack/schema.py` | `validate()` 加 `strict_frames=True` 参数 |
| `src/sana_wm_pipeline/stage06_pack/webdataset_writer.py` | `ShardWriter.__init__` 加 `strict_frames` 并传给 `validate()` |
| `src/sana_wm_pipeline/stage05_caption/qwen35_vl_runner.py` | stub 返回非空 fallback（不再是 None）|
| `configs/filter_thresholds.yaml` | 新增 `DL3DV` 源（unimatch_flow/dover 设 null）|
| `src/sana_wm_pipeline/orchestrate/ray_pipeline.py` | `_SOURCE_TO_POSE_MODE` 已有 `"DL3DV": "gtpose"`，smoke test 需用 `"default"` 时可命令行覆盖 |

### 新建文件（7 个）

| 文件 | 用途 |
|---|---|
| `scripts/pi3x_infer_cli.py` | Pi3X CLI，为 gt_pose 模式提供 `cams_pi3x.json` |
| `experiments/data_production_smoke/download_dl3dv.sh` | HF 下载 5-8 个场景 |
| `experiments/data_production_smoke/prepare_dl3dv.py` | 图像序列→MP4 + transforms.json→gt_poses.npy |
| `experiments/data_production_smoke/run_e2e_default.sh` | Default 模式端到端（Stage 01→06）|
| `experiments/data_production_smoke/run_e2e_gtpose.sh` | GT-pose 模式端到端 |
| `experiments/data_production_smoke/verify_and_eval.py` | shard 校验 + ATE/RTE 位姿对比 |
| `experiments/data_production_smoke/run_sana_wm_inference.py` | shard → 提取第一帧+位姿 → SANA-WM 推理 → 对比 GT |

---

## 四、各修改详解

### Fix 1：`mode_default.py`——两阶段论文对齐

**原因**：`sana_wm_pi3x_moge2.yaml` 的 SLAM 阶段只调 MoGe-2（Pi3X 不进 BA）。真正的论文方案（App.B.1）是预计算缓存注入 BA，与 TUM 实验架构一致。

```python
# 关键改动（完整函数体见下方）
VIPE_PIPELINE = "vipe_cached_depth"  # 从 sana_wm_pose_only 改

def run_default(clip_path, work_dir, vipe_cmd=VIPE_CMD, pipeline=VIPE_PIPELINE):
    # Phase A：预计算 Pi3X+MoGe-2 深度缓存
    cache_path = work_dir / "_depth_cache.npz"
    _precompute_depth_cache(clip_path, cache_path,
                            pi3x_weights=os.environ["SANA_WM_PI3X_WEIGHTS"],
                            moge2_weights=os.environ["SANA_WM_MOGE2_WEIGHTS"])
    
    # Phase B：VIPE SLAM 用缓存深度（CachedDepthModel 注入 BA）
    os.environ["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path)
    try:
        subprocess.check_call([*vipe_cmd, str(clip_path),
                               "--output", str(work_dir), "--pipeline", pipeline])
    finally:
        os.environ.pop("SANA_WM_CACHED_DEPTH_PATH", None)
        cache_path.unlink(missing_ok=True)  # 释放 ~600 MB 临时文件
    
    return _load_vipe_artifacts(clip_path, work_dir)
```

`_precompute_depth_cache` 直接复用 `experiments/vipe_comparison/precompute_pi3x_depths.py` 的核心逻辑（chunk Pi3X + MoGe-2 per frame + EMA scale fusion），**不重复实现**。

---

### Fix 2：`scripts/pi3x_infer_cli.py`——启用 gt_pose 模式

**原因**：`mode_gtpose.py` 调用 `python -m pi3x.infer`，该模块不存在。Pi3X 实际输出了所需的全部信息。

**`cams_pi3x.json` 需要格式**：
```json
{"frames": [{"center": [x, y, z], "K": [[fx,0,cx],[0,fy,cy],[0,0,1]]}, ...]}
```

**实现**：
```python
# scripts/pi3x_infer_cli.py

def recover_intrinsics_from_rays(rays: np.ndarray) -> np.ndarray:
    """rays: (H, W, 3) 单位方向向量 → K (3, 3)
    
    原理：rays[v, u] = normalize((u-cx, v-cy, f))
    取中间行做最小二乘：us = cx + fx * (rays[H//2,:,0]/rays[H//2,:,2])
    """
    H, W = rays.shape[:2]
    # fx, cx from center row
    row = rays[H // 2]  # (W, 3)
    us = np.arange(W, dtype=np.float64)
    ratios_x = row[:, 0] / np.clip(row[:, 2], 1e-9, None)
    A = np.stack([ratios_x, np.ones(W)], axis=1)
    fx, cx = np.linalg.lstsq(A, us, rcond=None)[0]
    # fy, cy from center column
    col = rays[:, W // 2]  # (H, 3)
    vs = np.arange(H, dtype=np.float64)
    ratios_y = col[:, 1] / np.clip(col[:, 2], 1e-9, None)
    A = np.stack([ratios_y, np.ones(H)], axis=1)
    fy, cy = np.linalg.lstsq(A, vs, rcond=None)[0]
    K = np.eye(3)
    K[0, 0] = abs(fx); K[1, 1] = abs(fy)
    K[0, 2] = cx;      K[1, 2] = cy
    return K

# 主流程：chunk 推理（chunk=16, stride=8），合并 camera_poses 和 rays
# out["camera_poses"][0, t, :3, 3] → center
# recover_intrinsics_from_rays(rays[0, t]) → K
# 写 cams_pi3x.json + pts_pi3x.npy
```

**调用方式**（mode_gtpose.py 兼容）：
```python
pi3x_cmd=("python", "scripts/pi3x_infer_cli.py")
```

---

### Fix 3：`configs/filter_thresholds.yaml`

```yaml
DL3DV:                        # default 模式，Stage-03 跳过
  vmaf_motion:      [0.5, 100]
  unimatch_flow:    null       # UniMatch 未安装
  dover:            null       # DOVER 未安装
  color_saturation: [0, 180]
  scene_cuts_max:   2
  vlm_entity:       null
  vlm_quality:      null
```

---

### Fix 4：`qwen35_vl_runner.py` stub

```python
# 两处 return None 改为非空字符串
return "a scene captured by a moving camera with natural indoor lighting"
```

---

### Fix 5：`schema.py` + `webdataset_writer.py`

```python
# schema.py
def validate(self, strict_frames: bool = True) -> None:
    if strict_frames:
        assert self.poses_c2w.shape == (CAMERA_FRAMES, 4, 4), ...
    else:
        assert self.poses_c2w.ndim == 3 and self.poses_c2w.shape[1:] == (4, 4), ...
    # 其余逻辑不变（caption 非空、first-frame identity 等）

# webdataset_writer.py ShardWriter
def __init__(self, out_dir, samples_per_shard=1000, strict_frames=True):
    self._strict_frames = strict_frames

def write(self, sample):
    sample.validate(strict_frames=self._strict_frames)
    ...
```

---

## 五、数据准备脚本

### `download_dl3dv.sh`

从 `DL3DV/DL3DV-ALL-2K`（HF）下载 5-8 个场景：
```bash
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
# 选景原则：室内+室外混合，时长 ≥60s，相机运动明显
huggingface-cli download DL3DV/DL3DV-ALL-2K \
  --include "1K/${SCENE_ID}/*" \
  --repo-type dataset \
  --local-dir /mnt/afs/davidwang/workspace/data/dl3dv_smoke \
  --local-dir-use-symlinks False
```
磁盘估算：5 场景 × 3–5 GB ≈ 15–25 GB（AFS 496 TB 可用）。

### `prepare_dl3dv.py`

```python
"""
输入：<scene>/images/*.png + <scene>/transforms.json
输出：<scene>/video.mp4         （ffmpeg，原始帧率）
      <scene>/gt_poses.npy      （T,4,4，OpenCV c2w）
      <scene>/gt_intrinsics.npy （4,）= [fx,fy,cx,cy]（来自 transforms.json 顶层）
      <scene>/orig_fps.txt      （评测时对齐帧号用）
"""

def transforms_to_poses(path) -> np.ndarray:
    data = json.load(open(path))
    frames = sorted(data["frames"], key=lambda f: f["file_path"])
    return np.array([f["transform_matrix"] for f in frames], dtype=np.float32)
    # DL3DV 使用 OpenCV c2w 约定，与管线一致，无需转换

def transforms_to_intrinsics(path) -> np.ndarray:
    d = json.load(open(path))
    return np.array([d["fl_x"], d["fl_y"], d["cx"], d["cy"]], dtype=np.float32)
```

**帧率对齐（评测用）**：`normalize.py` 从 orig_fps 重采样到 16fps，frame i（16fps）对应 GT frame `round(i * orig_fps / 16)`。

---

## 六、端到端运行脚本

### 环境变量（必须在运行前 export）

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
```

### `run_e2e_default.sh`（Default 模式，VIPE+Pi3X+MoGe-2 SLAM）

```
Stage 01 normalize → Stage 02 mode_default（预计算+cached SLAM）
→ Stage 04 apply_table6(source=DL3DV) → Stage 05(stub caption)
→ Stage 06 pack(strict_frames=False)
```

### `run_e2e_gtpose.sh`（GT-pose 模式，论文 DL3DV 对齐）

```
prepare_dl3dv.py → Stage 01 normalize
→ Stage 02 mode_gtpose(pi3x_cmd=scripts/pi3x_infer_cli.py, gt_poses=gt_poses.npy)
→ Stage 04 → Stage 05 → Stage 06
```

---

## 七、SANA-WM 推理验证脚本

### 前置：下载权重

```bash
huggingface-cli download Efficient-Large-Model/SANA-WM_bidirectional \
  --local-dir /mnt/afs/davidwang/models/sana_wm \
  --local-dir-use-symlinks False
# 估计大小：~10-20 GB，需确认 AFS 空间
```

### `run_sana_wm_inference.py`

**接口适配**：从 WebDataset shard 提取数据，调用现有的 `inference_sana_wm.py`。

```python
"""
inference_sana_wm.py 接口：
  --image        first_frame.png       （从 video.mp4 提取第 0 帧）
  --prompt       caption.txt           （从 caption.txt 读取）
  --camera       poses_c2w.npy         （T,4,4，直接用 shard 中的文件）
  --intrinsics   intrinsics.npy        （需 reshape：(T,1,4) → (T,4) 取第 0 帧 → (4,)=[fx,fy,cx,cy]）
  --num_frames   T                     （与标注帧数一致）
  --fps          16
"""
import subprocess, tarfile, io, numpy as np
from pathlib import Path
import imageio.v3 as iio

def extract_shard(shard_path, sample_id, out_dir):
    with tarfile.open(shard_path) as t:
        for key in ["mp4", "poses_c2w.npy", "intrinsics.npy", "caption.txt"]:
            t.extract(f"{sample_id}.{key}", out_dir)

def run_inference(sample_dir, sana_dir, output_dir, model_path):
    mp4     = sample_dir / "video.mp4"
    poses   = sample_dir / "poses_c2w.npy"     # (T, 4, 4) 直接可用
    intrs   = sample_dir / "intrinsics.npy"    # (T, 1, 4) → 取 [0,0,:] → (4,) = [fx,fy,cx,cy]
    caption = sample_dir / "caption.txt"
    
    # 提取第一帧
    first_frame = sample_dir / "first_frame.png"
    frames = iio.imread(mp4)  # (T, H, W, 3)
    iio.imwrite(first_frame, frames[0])
    
    # intrinsics reshape：(T,1,4) → 取全局均值 → (4,) = [fx,fy,cx,cy]
    intr_arr = np.load(intrs)  # (T, 1, 4)
    intr_mean = intr_arr[:, 0, :].mean(axis=0)  # (4,)
    intr_path = sample_dir / "intrinsics_flat.npy"
    np.save(intr_path, intr_mean)
    
    num_frames = np.load(poses).shape[0]
    subprocess.check_call([
        "python", str(sana_dir / "inference_video_scripts/inference_sana_wm.py"),
        "--image",       str(first_frame),
        "--prompt",      str(caption),
        "--camera",      str(poses),
        "--intrinsics",  str(intr_path),
        "--num_frames",  str(num_frames),
        "--fps",         "16",
        "--output_dir",  str(output_dir),
        "--name",        sample_dir.name,
        "--step",        "60",
    ])
```

**对比指标**：
- 生成视频 vs DL3DV GT 视频（same scene, same trajectory）
- PSNR / SSIM（逐帧）
- 主观：side-by-side MP4

---

## 八、位姿验证（`verify_and_eval.py`）

1. **Schema 校验**：每个 tar 包含 6 个必要文件
2. **ATE/RTE**：
   - 加载 shard `poses_c2w.npy`（T, 4, 4，16fps）
   - 加载 `gt_poses.npy`（T', 4, 4，orig_fps）→ 降采样到 16fps
   - 用 `evo` 计算 ATE RMSE / RTE（与 TUM 实验一致）
3. **轨迹可视化**：生成 3-view 对比图

---

## 九、已知风险

| 风险 | 概率 | 应对 |
|---|---|---|
| Pi3X `camera_poses` 在 chunk 边界不连续 | 中 | 用 stride=8 overlap + 线性插值平滑边界；先在 2 帧测试确认 |
| DL3DV 场景 <60s → schema 961 帧不满 | 低 | strict_frames=False 绕过，或选更长场景 |
| SANA-WM 权重 HF 访问受限 | 中 | 先确认 `huggingface-cli download`；`run_sana_wm.sh` 注释说"first use 自动下载" |
| `inference_sana_wm.py` 输出分辨率固定 704×1280 | 确定 | DL3DV GT 视频 resize 到同分辨率再计算 PSNR/SSIM |
| Stage 02 default 模式：预计算缓存 ~600 MB/clip | 确定 | 运行完后立即删除（finally 块），5 场景峰值磁盘 ≤ 3 GB |

---

## 十、执行顺序

```
Day 1 上午：Fix 1–5（代码修改）+ pytest 确认 140 passed
Day 1 下午：download_dl3dv.sh（~2h 下载）+ prepare_dl3dv.py
Day 1 晚：  run_e2e_default.sh（5 场景，每场景约 30-60 min）
Day 2 上午：pi3x_infer_cli.py + run_e2e_gtpose.sh
Day 2 下午：verify_and_eval.py（shard 校验 + ATE/RTE 对比图）
Day 3 上午：SANA-WM 权重下载（~15-20 GB）
Day 3 下午：run_sana_wm_inference.py（5 场景推理 + 对比视频）
```

---

## 十一、验证命令

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 1. 管线代码不受影响
python -m pytest -q
# 期望：140 passed, 0 failed

# 2. Shard schema 合法
python experiments/data_production_smoke/verify_and_eval.py --mode schema
# 期望：N/N shards valid

# 3. 位姿精度对比 GT
python experiments/data_production_smoke/verify_and_eval.py --mode pose-eval
# 期望：ATE RMSE 输出（DL3DV 尺度比 TUM 大，绝对值不可比，但趋势合理）

# 4. SANA-WM 推理跑通
python experiments/data_production_smoke/run_sana_wm_inference.py --sample-limit 1
# 期望：output_dir 下生成 {sample_id}.mp4 + side-by-side 对比
```

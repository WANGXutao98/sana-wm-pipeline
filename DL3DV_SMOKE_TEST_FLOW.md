# DL3DV 冒烟测试流程图

## 📂 目录结构

```
输入目录: /mnt/afs/davidwang/workspace/sana_test_data/Dl3dv/
├── DL3DV-ALL-2K_10K__a1cc9c41...__images_2.mp4          (5.6M, 303帧)
├── DL3DV-ALL-2K_10K__a1cc9c41...__images_2.camera.npz   (包含 c2w, K_px)
├── DL3DV-ALL-2K_10K__b137b3eb...__images_2.mp4          (6.5M, 303帧)
├── DL3DV-ALL-2K_10K__b137b3eb...__images_2.camera.npz
├── DL3DV-ALL-2K_10K__9eb0f6c5...__images_2.mp4          (19M, 301帧)
├── DL3DV-ALL-2K_10K__9eb0f6c5...__images_2.camera.npz
├── DL3DV-ALL-2K_10K__9d44d654...__images_2.mp4          (28M, 303帧)
├── DL3DV-ALL-2K_10K__9d44d654...__images_2.camera.npz
├── DL3DV-ALL-2K_10K__a42b0f54...__images_2.mp4          (48M, 301帧)
└── DL3DV-ALL-2K_10K__a42b0f54...__images_2.camera.npz

输出目录: /mnt/afs/davidwang/workspace/sana_test_data/smoke_result_dl3dv/
├── samples.tsv                                          (样本列表)
├── smoke_test_summary.json                              (测试结果摘要)
└── DL3DV-ALL-2K_10K__a1cc9c41...__images_2/            (每个样本一个子目录)
    ├── gt_poses.npy                                     (临时：从 camera.npz 提取的 c2w)
    ├── poses.npy                                        (输出：完整 GT poses)
    ├── intrinsics.npy                                   (输出：GT 或 seed intrinsics)
    └── scale_per_frame.npy                              (输出：metric scale)
```

---

## 🔄 执行流程

### 1. Shell 脚本入口
**文件**: `experiments/data_production_smoke/smoke_dl3dv.sh`

```bash
# 环境变量
export DL3DV_DIR="/mnt/afs/.../Dl3dv"
export OUT_BASE="/mnt/afs/.../smoke_result_dl3dv"
export SANA_WM_MAX_FRAMES=64                    # Pi3X 子采样帧数
export SANA_WM_PI3X_WEIGHTS="/mnt/afs/.../pi3x"

# 生成样本列表 (TSV 格式)
cat > samples.tsv
sample_id<TAB>n_frames
DL3DV-...__images_2<TAB>303
DL3DV-...__images_2<TAB>301
...

# 验证文件存在性
for sample in samples.tsv:
    检查 ${DL3DV_DIR}/${sample_id}.mp4
    检查 ${DL3DV_DIR}/${sample_id}.camera.npz

# 调用 Python 脚本
python scripts/smoke_batch_gtpose.py \
    --samples "$OUT_BASE/samples.tsv" \
    --dl3dv-dir "$DL3DV_DIR" \
    --output-dir "$OUT_BASE"
```

---

### 2. Python 批处理脚本
**文件**: `scripts/smoke_batch_gtpose.py`

```python
def main():
    # 读取样本列表
    samples = parse_tsv("samples.tsv")
    # [(sample_id, n_frames), ...]
    
    # 逐个处理（单进程，利用 @lru_cache）
    for sample_id, n_frames in samples:
        process_sample(sample_id, n_frames, dl3dv_dir, output_dir)

def process_sample(sample_id, n_frames, dl3dv_dir, output_dir):
    # 输入文件
    video_path = dl3dv_dir / f"{sample_id}.mp4"
    camera_path = dl3dv_dir / f"{sample_id}.camera.npz"
    
    # 1. 加载 camera.npz
    camera_data = np.load(camera_path)
    gt_c2w = camera_data['c2w']         # (N, 4, 4) GT poses
    # gt_K_px = camera_data['K_px']     # (N, 4) GT intrinsics (暂不使用)
    
    # 2. 保存临时 GT poses
    temp_gt_poses = output_dir / sample_id / "gt_poses.npy"
    np.save(temp_gt_poses, gt_c2w)
    
    # 3. 调用 GT-pose 模式
    artifact = run_gtpose(
        clip_path=video_path,
        gt_poses_path=temp_gt_poses,
        work_dir=output_dir / sample_id,
        inlier_percentile=80.0,
        gt_intrinsics_path=camera_path,     # ← 修复后新增
    )
    
    # 4. 保存结果
    np.save("poses.npy", artifact.poses_c2w)
    np.save("intrinsics.npy", artifact.intrinsics)
    np.save("scale_per_frame.npy", artifact.scale_per_frame)
    
    # 5. 返回结果字典
    return {
        "status": "success",
        "scale_mean": float(artifact.scale_per_frame.mean()),
        ...
    }
```

---

### 3. GT-Pose 模式核心
**文件**: `src/sana_wm_pipeline/stage02_pose/mode_gtpose.py`

```python
def run_gtpose(clip_path, gt_poses_path, work_dir, inlier_percentile, gt_intrinsics_path):
    # 步骤 1: 获取视频帧数
    cap = cv2.VideoCapture(str(clip_path))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 24
    cap.release()
    
    # 步骤 2: 计算 Pi3X 运行帧数
    max_frames = int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
    n_scale = min(N, max_frames)                    # n_scale = 64
    
    # 步骤 3: Pi3X 推理（子采样）
    from ..sana_wm_data_clean.pose import _real, adapters
    
    frames = adapters.read_frames(str(clip_path), n_scale)  # 读取 64 帧
    poses_pi3x, _depth = _real.pi3_infer(frames)            # Pi3X 推理
    pred_positions = poses_pi3x[:, :3, 3]                   # (64, 3) 相机中心
    n_scale = pred_positions.shape[0]                       # 可能 < 64
    
    # 步骤 4: 加载完整 GT poses
    gt_poses_full = _load_gt_poses(gt_poses_path)          # (N, 4, 4), N=301/303
    N_gt = len(gt_poses_full)
    
    # 步骤 5: GT 子采样匹配 Pi3X
    gt_indices = adapters.even_indices(N_gt, n_scale)       # 子采样索引
    poses_gt_sub = gt_poses_full[gt_indices]                # (64, 4, 4)
    gt_positions_sub = poses_gt_sub[:, :3, 3]               # (64, 3)
    
    # 步骤 6: Umeyama Sim(3) 恢复 scale
    s = recover_metric_scale(
        pred_positions,      # Pi3X 预测的 64 个相机中心
        gt_positions_sub,    # GT 子采样的 64 个相机中心
        inlier_percentile=80.0
    )
    
    # 步骤 7: 加载 GT intrinsics（修复后）
    m = gt_poses_full.shape[0]
    
    if gt_intrinsics_path is not None and gt_intrinsics_path.exists():
        intr = _load_gt_intrinsics(gt_intrinsics_path, m)  # 从 camera.npz 加载
    else:
        intr = _load_gt_intrinsics(gt_poses_path, m)       # Fallback 原有逻辑
    
    if intr is None:
        intr = _seed_intrinsics(clip_path, m)              # Fallback seed
    
    # 步骤 8: Scale 广播
    scale_per_frame = np.full(m, float(s), dtype=np.float32)
    
    # 步骤 9: 返回结果
    return PoseArtifact(
        poses_c2w=gt_poses_full,        # 完整 N 帧 GT poses（未修改）
        intrinsics=intr,                # (N, 1, 4)
        scale_per_frame=scale_per_frame,# (N,) 单值广播
        depth_downsampled=None,
    )
```

---

### 4. GT Intrinsics 加载（修复后）
**文件**: `src/sana_wm_pipeline/stage02_pose/mode_gtpose.py`

```python
def _load_gt_intrinsics(gt_poses_path: Path, n_frames: int):
    # 情况 1: 直接传入 camera.npz（修复后新增）
    if gt_poses_path.suffix == '.npz' and gt_poses_path.exists():
        data = np.load(gt_poses_path)
        if 'K_px' in data:
            K_gt = data['K_px']             # (M, 4) [fx, fy, cx, cy]
            
            # 子采样到 n_frames
            idx = np.linspace(0, K_gt.shape[0] - 1, n_frames).round().astype(int)
            return K_gt[idx][:, None, :].astype(np.float32)  # (n_frames, 1, 4)
    
    # 情况 2: 从 gt_poses_path 推断（原有逻辑）
    gt_dir = gt_poses_path.parent
    candidates = [
        gt_dir / f"{gt_dir.name}.camera.npz",
        gt_dir.parent / f"{gt_dir.name}.camera.npz",
    ]
    
    for camera_npz in candidates:
        if camera_npz.exists():
            # 加载 K_px...
```

---

## 📊 数据流

```
输入 (Dl3dv/)
  ├─ video.mp4 (303帧)
  └─ camera.npz
       ├─ c2w: (303, 4, 4)        ← GT poses
       └─ K_px: (303, 4)          ← GT intrinsics

↓ smoke_batch_gtpose.py

临时文件 (smoke_result_dl3dv/sample/)
  └─ gt_poses.npy: (303, 4, 4)    ← 从 camera.npz['c2w'] 提取

↓ run_gtpose()

Pi3X 处理
  ├─ 读取视频: 64 帧（子采样）
  ├─ Pi3X 推理: (64, 4, 4) → 提取中心 (64, 3)
  └─ GT 子采样: (303, 4, 4) → even_indices → (64, 4, 4) → 提取中心 (64, 3)

↓ Umeyama 对齐

  pred_positions (64, 3) vs gt_positions_sub (64, 3)
  → scale = 1.717549

↓ 输出 (smoke_result_dl3dv/sample/)

  ├─ poses.npy: (303, 4, 4)       ← 完整 GT poses（未修改）
  ├─ intrinsics.npy: (303, 1, 4)  ← GT K_px（修复后）
  └─ scale_per_frame.npy: (303,)  ← scale 广播
```

---

## 🔑 关键点

### Pi3X 子采样（SANA_WM_MAX_FRAMES=64）
- **输入视频**: 301/303 帧
- **Pi3X 处理**: 64 帧（节省 GPU 内存）
- **输出 poses**: 301/303 帧（完整，不截断）

### GT Intrinsics 加载（修复）
- **修复前**: 从 `gt_poses_path` 推断 → 失败（输出目录）
- **修复后**: 显式传入 `gt_intrinsics_path=camera.npz` → 成功

### Umeyama 对齐
- **输入**: Pi3X 64 个中心 vs GT 64 个中心（子采样）
- **输出**: 单个 scale 值（广播到所有帧）

---

## 📁 文件依赖关系

```
smoke_dl3dv.sh
  └─ scripts/smoke_batch_gtpose.py
       └─ src/sana_wm_pipeline/stage02_pose/mode_gtpose.py
            ├─ _load_gt_poses()
            ├─ _load_gt_intrinsics()          ← 修复点
            ├─ _seed_intrinsics()
            └─ src/sana_wm_pipeline/sana_wm_data_clean/pose/
                 ├─ _real.pi3_infer()         ← Pi3X 推理
                 ├─ adapters.read_frames()    ← 读取视频帧
                 └─ adapters.even_indices()   ← 子采样索引
```

---

## 📈 性能指标（实测）

| 样本 | 文件大小 | 帧数 | 耗时 | Scale |
|------|---------|------|------|-------|
| a1cc9c41 | 5.6M | 303 | 70.53s | 1.7175 |
| b137b3eb | 6.5M | 303 | 2.55s | 2.8002 |
| 9eb0f6c5 | 19M | 301 | 2.80s | 2.4155 |
| 9d44d654 | 28M | 303 | 3.38s | 1.3995 |
| a42b0f54 | 48M | 301 | 3.46s | 2.7001 |

**注**: 第一个样本耗时长（加载模型），后续样本快（@lru_cache）

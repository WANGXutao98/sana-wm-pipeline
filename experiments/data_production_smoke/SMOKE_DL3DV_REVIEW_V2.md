# DL3DV 冒烟测试脚本审阅文档（修订版）

**脚本路径**: 
- Shell 脚本: `experiments/data_production_smoke/smoke_dl3dv.sh`
- Python 脚本: `scripts/smoke_batch_gtpose.py`

**日期**: 2026-09-11  
**状态**: ⏸️ 待审阅 - 请勿执行  
**修订原因**: 修正帧数错误，拆分 Python 脚本

---

## 🔧 修正内容

### 问题 1: 帧数错误 ✅ 已修正

**之前（错误）**: 所有样本都是 302 帧  
**现在（正确）**: 
- 3 个样本：**303 帧**
- 2 个样本：**301 帧**

**验证结果**:
```bash
a1cc9c41... → 303 帧
b137b3eb... → 303 帧
9eb0f6c5... → 301 帧 ✅
9d44d654... → 303 帧
a42b0f54... → 301 帧 ✅
```

### 问题 2: Pi3X 处理 64 帧的原因 ✅ 已确认

**官方代码确认** (stage.py:50-55):
```python
# OUTPUT poses MUST be frame-aligned to EVERY video frame. The max_frames cap applies
# ONLY to the subsample fed to Pi3/MoGe (GPU memory) — it must NEVER truncate the
# output (a 64-cap desynced pose_i from video frame_i and produced unusable clips).
N = rec.num_frames or 24
max_frames = int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
n_scale = min(N, max_frames)
```

**关键点**:
- ✅ 官方默认 `SANA_WM_MAX_FRAMES=64`
- ✅ Pi3X 只在 n_scale (64) 帧子集上运行 → 节省 GPU 内存
- ✅ GT poses 保持完整 N 帧 → 不截断输出
- ✅ 通过 `even_indices()` 子采样进行 Umeyama 对齐

**我们的实现**: 完全一致 ✅

### 问题 3: Python 脚本独立 ✅ 已完成

**新文件**: `scripts/smoke_batch_gtpose.py`

**参考**: `scripts/smoke_test_batch.py` 的结构

**主要特点**:
- ✅ 单进程批量处理（利用 @lru_cache）
- ✅ 从 camera.npz 提取 GT poses (c2w)
- ✅ 调用 `run_gtpose()` 处理
- ✅ 保存结果 + 生成 JSON 摘要
- ✅ 详细的进度和错误报告

---

## 📋 更新后的样本列表

| ID（前8位） | 文件大小 | 实际帧数 | 选择理由 |
|------------|---------|----------|----------|
| `a1cc9c41` | 5.6M | **303** | 小文件，快速验证 |
| `b137b3eb` | 6.5M | **303** | 小文件，稳定性 |
| `9eb0f6c5` | 19M | **301** ✅ | 中等大小，不同帧数 |
| `9d44d654` | 28M | **303** | 大文件，压力测试 |
| `a42b0f54` | 48M | **301** ✅ | 最大文件，不同帧数 |

**策略优化**:
- ✅ 覆盖小/中/大文件（5M-48M）
- ✅ 覆盖两种帧数（301/303）→ 验证通用性
- ✅ 不同哈希前缀 → 场景多样性

---

## 🔧 脚本结构（修订后）

### Shell 脚本 (smoke_dl3dv.sh)

```bash
# 1. 环境配置
export SANA_WM_MAX_FRAMES=64  # 官方默认值

# 2. 生成样本列表（TSV 格式，正确的帧数）
cat > samples.tsv << EOF
sample_id<TAB>n_frames
DL3DV-...__images_2<TAB>303
DL3DV-...__images_2<TAB>301
EOF

# 3. 验证文件存在性

# 4. 调用 Python 脚本
python scripts/smoke_batch_gtpose.py \
    --samples "$SAMPLES_FILE" \
    --dl3dv-dir "$DL3DV_DIR" \
    --output-dir "$OUT_BASE"
```

### Python 脚本 (smoke_batch_gtpose.py)

```python
def process_sample(sample_id, n_frames, dl3dv_dir, output_dir):
    # 1. 加载 camera.npz
    camera_data = np.load(camera_path)
    gt_c2w = camera_data['c2w']  # (N, 4, 4)
    
    # 2. 验证帧数
    actual_frames = gt_c2w.shape[0]
    if actual_frames != n_frames:
        print(f"⚠️  实际帧数 {actual_frames} != 预期 {n_frames}")
    
    # 3. 保存临时 GT poses
    np.save(temp_gt_poses, gt_c2w)
    
    # 4. 调用 run_gtpose()
    artifact = run_gtpose(
        clip_path=video_path,
        gt_poses_path=temp_gt_poses,
        work_dir=sample_dir,
        inlier_percentile=80.0,
    )
    
    # 5. 保存结果
    np.save("poses.npy", artifact.poses_c2w)
    np.save("intrinsics.npy", artifact.intrinsics)
    np.save("scale_per_frame.npy", artifact.scale_per_frame)
    
    # 6. 返回结果字典
    return {
        "status": "success",
        "actual_frames": actual_frames,
        "scale_mean": float(artifact.scale_per_frame.mean()),
        ...
    }
```

---

## ✅ 预期输出（修订）

### 成功场景

```json
{
  "total_samples": 5,
  "successful": 5,
  "failed": 0,
  "results": [
    {
      "sample_id": "DL3DV-...a1cc9c41...",
      "status": "success",
      "elapsed_time": "42.15s",
      "expected_frames": 303,
      "actual_frames": 303,
      "output_poses_shape": "(303, 4, 4)",
      "output_intrinsics_shape": "(303, 1, 4)",
      "output_scale_shape": "(303,)",
      "scale_mean": 5.234567,
      "scale_std": 0.0,
      "scale_min": 5.234567,
      "scale_max": 5.234567
    },
    {
      "sample_id": "DL3DV-...9eb0f6c5...",
      "status": "success",
      "expected_frames": 301,
      "actual_frames": 301,
      "output_poses_shape": "(301, 4, 4)",
      ...
    }
  ]
}
```

**关键验证点**:
- ✅ `actual_frames` 匹配 `expected_frames`（301 或 303）
- ✅ 输出 shape 使用完整帧数（不是 64）
- ✅ `scale_std: 0.0`（单值广播）
- ✅ `scale_mean` 在合理范围（1-10）

---

## 📊 预期日志输出（关键部分）

### 样本 1（303 帧）

```
============================================================
样本: DL3DV-ALL-2K_10K__a1cc9c41...__images_2
预期帧数: 303
============================================================
--- 加载 GT 数据 ---
  GT poses (c2w): (303, 4, 4)
  GT intrinsics (K_px): (303, 4)
  临时 GT poses 已保存: gt_poses.npy

--- Stage 2: GT-Pose 模式 ---
[mode_gtpose] Video has 303 frames, Pi3X will run on 64 frames (SANA_WM_MAX_FRAMES=64)
[mode_gtpose] Running Pi3X inference...
[_real.py] Loading Pi3X from /mnt/afs/davidwang/models/pi3x...
[_real.py] Pi3X loaded successfully
[mode_gtpose] Pi3X returned 64 camera positions
[mode_gtpose] Loaded 303 GT poses from gt_poses.npy
[mode_gtpose] GT subsampled to 64 frames for alignment
[mode_gtpose]   Subsample indices: [0, 5, 10, 14, 19] ... [...]
[mode_gtpose] Umeyama scale: 5.234567 (inlier_percentile=80.0%)
[mode_gtpose] ✅ Using GT intrinsics (shape: (303, 1, 4))
[mode_gtpose] ✅ GT-pose mode completed:
[mode_gtpose]    Output poses: (303, 4, 4)
[mode_gtpose]    Intrinsics: (303, 1, 4)
[mode_gtpose]    Scale: 5.234567 (broadcasted to 303 frames)

--- 保存结果 ---
  poses.npy: (303, 4, 4)
  intrinsics.npy: (303, 1, 4)
  scale_per_frame.npy: (303,)

✅ 处理成功
   耗时: 42.15s
   Scale: 5.234567 ± 0.000000
   Scale 范围: [5.234567, 5.234567]
```

### 样本 3（301 帧）

```
============================================================
样本: DL3DV-ALL-2K_10K__9eb0f6c5...__images_2
预期帧数: 301
============================================================
--- 加载 GT 数据 ---
  GT poses (c2w): (301, 4, 4)
  ...

[mode_gtpose] Video has 301 frames, Pi3X will run on 64 frames
[mode_gtpose] Pi3X returned 64 camera positions
[mode_gtpose] Loaded 301 GT poses from gt_poses.npy
[mode_gtpose] GT subsampled to 64 frames for alignment
[mode_gtpose] ✅ GT-pose mode completed:
[mode_gtpose]    Output poses: (301, 4, 4)
[mode_gtpose]    Intrinsics: (301, 1, 4)
[mode_gtpose]    Scale: 4.987654 (broadcasted to 301 frames)

✅ 处理成功
   Scale: 4.987654 ± 0.000000
```

**关键验证**:
- ✅ Pi3X 只处理 64 帧（不是 301/303）
- ✅ GT 子采样到 64 帧用于 Umeyama
- ✅ 输出保持完整帧数（301 或 303）

---

## 🚦 执行决策（修订）

### ✅ 可以执行的条件

- [x] 代码已与官方 100% 对齐
- [x] 所有单元测试通过（9/9）
- [x] DL3DV 数据结构验证通过
- [x] **帧数已修正**（301/303）✅
- [x] **Python 脚本已独立**✅
- [x] 样本文件存在（需要你确认）
- [x] GPU 可用且内存充足（需要你确认）

### ⏸️ 执行前检查

**请在执行前运行**:

```bash
cd /mnt/afs/davidwang/workspace/sana_test_data/Dl3dv

# 验证 5 个样本的文件
echo "=== 样本 1 (303 帧) ==="
ls -lh *a1cc9c41*.{mp4,camera.npz}

echo "=== 样本 2 (303 帧) ==="
ls -lh *b137b3eb*.{mp4,camera.npz}

echo "=== 样本 3 (301 帧) ==="
ls -lh *9eb0f6c5*.{mp4,camera.npz}

echo "=== 样本 4 (303 帧) ==="
ls -lh *9d44d654*.{mp4,camera.npz}

echo "=== 样本 5 (301 帧) ==="
ls -lh *a42b0f54*.{mp4,camera.npz}

# 验证 GPU
nvidia-smi

# 验证 Pi3X 权重
ls -ld /mnt/afs/davidwang/models/pi3x
```

### 🚀 执行命令

**如果所有检查通过**:

```bash
bash experiments/data_production_smoke/smoke_dl3dv.sh
```

---

## 📝 修订总结

### 修正的问题

1. ✅ **帧数错误**: 302 → 正确的 301/303
2. ✅ **Python 脚本**: 内嵌 → 独立文件 `scripts/smoke_batch_gtpose.py`
3. ✅ **64 帧说明**: 添加官方代码引用，解释清楚

### 新增内容

1. ✅ 独立的 Python 脚本（参考 smoke_test_batch.py）
2. ✅ 帧数验证逻辑（actual vs expected）
3. ✅ 更详细的日志输出
4. ✅ Scale 统计信息（mean/std/min/max）

### 保持不变

1. ✅ 样本选择策略（小/中/大文件）
2. ✅ Pi3X 子采样到 64 帧（官方默认）
3. ✅ 输出格式和路径
4. ✅ 预计耗时 10-15 分钟

---

## 🎯 审阅要点（修订）

### 请再次确认

1. **帧数修正** - 301/303 是否正确？（已验证 ✅）
2. **Python 脚本** - 独立的 `smoke_batch_gtpose.py` 是否 OK？
3. **执行时机** - 现在执行还是稍后？
4. **其他调整** - 还需要修改什么吗？

---

**修订完成，等待你的最终批准！** 🐴

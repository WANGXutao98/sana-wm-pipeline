# SANA-WM Pipeline 各阶段详解

> **文档状态**：当前有效  
> **最后更新**：2026-08-01  
> **维护者**：David Wang

---

## 目录

1. [Stage 01: 数据摄取](#stage-01-数据摄取)
2. [Stage 02: 位姿标注](#stage-02-位姿标注)
3. [Stage 03: 3DGS 增强](#stage-03-3dgs-增强)
4. [Stage 04: 视觉过滤](#stage-04-视觉过滤)
5. [Stage 05: Caption 生成](#stage-05-caption-生成)
6. [Stage 06: WebDataset 打包](#stage-06-webdataset-打包)
7. [端到端示例](#端到端示例)

---

## Stage 01: 数据摄取

### 目标

将 7 个公开数据源的原始视频标准化为统一格式，供后续处理。

### 输入

| 数据源 | 原始格式 | 样本数 | 来源 |
|--------|---------|--------|------|
| RealEstate10K | MP4, 多种分辨率 | 70,938 | YouTube 室内视频 |
| DL3DV-ALL-2K | MP4, 多种分辨率 | 9,937 | 3D 场景重建数据集 |
| SpatialVID-hq | MP4, 高分辨率 | 37,622 | 空间视频数据集 |
| sekai-real-walking-hq | MP4 | 19,238 | 真实街景行走 |
| OmniWorld-Game | MP4, 游戏引擎 | 6,145 | Unity/UE5 合成 |
| sekai-game-walking | MP4, 游戏引擎 | 1,602 | 游戏角色行走 |
| sekai-game-drone | MP4, 游戏引擎 | 931 | 游戏无人机视角 |

### 处理流程

```python
# 伪代码
def stage01_ingest(raw_video_path, output_dir):
    # 1. 读取原始视频
    video = VideoReader(raw_video_path)
    
    # 2. 检测原始参数
    original_fps = video.fps
    original_resolution = video.resolution
    original_frames = video.num_frames
    
    # 3. 标准化处理
    target_fps = 16  # 论文固定值
    target_resolution = (1280, 720)  # 720p
    target_frames = 961  # 论文固定值 (60s @ 16fps)
    
    # 4. 时间重采样
    if original_fps != target_fps:
        video = resample_fps(video, target_fps)
    
    # 5. 空间缩放
    if original_resolution != target_resolution:
        video = resize(video, target_resolution, method='lanczos')
    
    # 6. 裁剪/填充到固定帧数
    if len(video) > target_frames:
        video = video[:target_frames]  # 截断
    elif len(video) < target_frames:
        video = pad_frames(video, target_frames)  # 重复最后一帧
    
    # 7. 重新编码
    video.save(output_dir / 'video_normalized.mp4', codec='h264', crf=23)
    
    # 8. 元数据
    metadata = {
        'original_fps': original_fps,
        'original_resolution': original_resolution,
        'original_frames': original_frames,
        'transform': 'resample+resize+crop'
    }
    json.dump(metadata, open(output_dir / 'ingest_meta.json', 'w'))
```

### 输出

```
output/stage01_ingest/
├── video_normalized.mp4       # 720p, 16fps, 961 帧, H.264
└── ingest_meta.json           # 原始参数记录
```

### 关键参数

| 参数 | 值 | 来源 |
|------|---|------|
| `target_fps` | 16 | 论文 §5.1 |
| `target_resolution` | (1280, 720) | 论文 §5.1 |
| `target_frames` | 961 | 论文 App. D.1 |
| `codec` | H.264 | 兼容性最佳 |
| `crf` | 23 | 平衡质量与大小 |

### 常见问题

**Q: 为什么是 961 帧？**  
A: 60 秒 × 16 fps = 960 帧，+1 帧用于首尾对齐（某些算法需要）。实际生产中 `strict_frames=False`，允许 960 或 961。

**Q: 原始视频不足 60 秒怎么办？**  
A: 填充策略：重复最后一帧。但 QC Stage 1 会标记 `video_duration_too_short`，后续可能被拒绝。

---

## Stage 02: 位姿标注

### 目标

为每一帧估计 6-DoF 相机位姿（米制），生成 `poses_c2w.npy` (camera-to-world 变换矩阵)。

### 技术方案

**核心算法**：VIPE SLAM + Pi3X + MoGe-2 深度融合

```
输入视频 (961 帧)
    ↓
VIPE SLAM (SuperPoint + SuperGlue + BA)
    → 相对尺度轨迹 (相机坐标系)
    ↓
Pi3X (多帧深度估计)
    → 时序一致深度图 (961 张)
    ↓
MoGe-2 (单帧米制深度)
    → 关键帧绝对深度 (每 10 帧 1 张)
    ↓
深度融合 (MoGe-2 深度注入 BA)
    → 米制尺度轨迹
    ↓
输出: poses_c2w.npy (T×4×4)
```

### 实现细节

**代码入口**：`src/sana_wm_pipeline/stage02_pose/run_worker.py`

```python
def stage02_pose_default(video_path, output_dir):
    # 1. 加载视频
    frames = load_video(video_path)  # (T, H, W, 3)
    
    # 2. VIPE SLAM 初始化
    vipe = VIPESlam(
        feature_extractor='superpoint',
        matcher='superglue',
        backend='ceres'  # Bundle Adjustment 求解器
    )
    
    # 3. Pi3X 深度估计 (多帧)
    model_pi3x = load_model('pi3x', device='cuda:0')
    depth_pi3x = model_pi3x(frames)  # (T, H, W)
    del model_pi3x  # ⚠️ 立即释放 25GB 显存
    torch.cuda.empty_cache()
    
    # 4. MoGe-2 深度估计 (关键帧)
    model_moge2 = load_model('moge2', device='cuda:0')
    keyframe_indices = range(0, len(frames), 10)  # 每 10 帧
    depth_moge2 = []
    for idx in keyframe_indices:
        depth = model_moge2(frames[idx])  # 单帧推理
        depth_moge2.append(depth)
    del model_moge2  # ⚠️ 立即释放 30GB 显存
    torch.cuda.empty_cache()
    
    # 5. 深度注入 VIPE (关键步骤)
    vipe.set_depth_prior(
        mode='cached',  # 使用预计算深度
        depth_maps=depth_moge2,
        keyframe_indices=keyframe_indices
    )
    
    # 6. VIPE 运行
    for t, frame in enumerate(frames):
        vipe.track(frame, timestamp=t)
    
    # 7. 提取轨迹
    poses_c2w = vipe.get_poses()  # (T, 4, 4) 米制
    intrinsics = vipe.get_intrinsics()  # (T, 4) [fx, fy, cx, cy]
    
    # 8. 保存
    np.save(output_dir / 'poses_c2w.npy', poses_c2w)
    np.save(output_dir / 'intrinsics.npy', intrinsics)
    np.save(output_dir / 'scale.npy', np.ones(len(frames)))  # Default 模式固定为 1.0
```

### 三种模式对比

| 模式 | 深度模型 | 尺度精度 | 速度 | 显存 | 使用场景 |
|------|---------|---------|------|------|---------|
| **default** | Pi3X + MoGe-2 | 3.3% 偏差 | 30s | 55GB | ✅ 生产模式 |
| **metric3d** | metric3d-small | 18.5% 偏差 | 15s | 25GB | Baseline 对比 |
| **gt_depth** | 真实深度 | 0% 偏差 | 10s | 10GB | 仅用于验证 |

### 输出

```
output/stage02_pose/
├── poses_c2w.npy              # (T, 4, 4) 米制相机到世界变换
├── intrinsics.npy             # (T, 4) [fx, fy, cx, cy]
├── scale.npy                  # (T,) 每帧尺度因子（Default 模式全 1.0）
└── pose_meta.json             # 元数据（模式/模型版本/耗时）
```

### 关键 API 陷阱

**陷阱 1：深度注入必须用 `cached` 模式**

```python
# ❌ 错误：VIPE 会忽略外部深度
slam_config.keyframe_depth = 'moge2'

# ✅ 正确：使用 CachedDepthModel
slam_config.keyframe_depth = 'cached'
vipe.set_depth_prior(mode='cached', depth_maps=depth_moge2)
```

**陷阱 2：关键帧索引必须传递**

```python
# ❌ 错误：VIPE 无法匹配深度图与帧
for kf in keyframes:
    buffer.add_keyframe(kf, depth_map)

# ✅ 正确：显式传递帧索引
for kf in keyframes:
    buffer.add_keyframe(kf, depth_map, frame_idx=int(kf.timestamp))
```

**陷阱 3：显存泄漏**

```python
# ❌ 错误：Pi3X + MoGe-2 同时存在显存会爆
depth_pi3x = model_pi3x(frames)
depth_moge2 = model_moge2(keyframes)  # OOM!

# ✅ 正确：用完立即释放
depth_pi3x = model_pi3x(frames)
del model_pi3x
torch.cuda.empty_cache()
depth_moge2 = model_moge2(keyframes)
```

### 性能基准

**H100 80GB 单卡**：
- **Pi3X 推理**：961 帧 → 12 秒
- **MoGe-2 推理**：97 关键帧 → 8 秒
- **VIPE SLAM**：961 帧 → 5 秒
- **总耗时**：~25 秒
- **显存峰值**：55GB

---

## Stage 03: 3DGS 增强

### 目标

**仅针对 DL3DV 数据集**，通过 3D 高斯点云重建 (3DGS) 合成新的相机轨迹，将单个样本扩增 40 倍。

### 适用范围

| 数据集 | 是否使用 Stage 3 | 原因 |
|--------|-----------------|------|
| DL3DV-ALL-2K | ✅ 是 | 样本稀缺，且场景静态适合重建 |
| 其他 6 个数据集 | ❌ 否 | 样本充足，或场景动态不适合 |

### 技术流程

```
原始样本 (1 条轨迹)
    ↓
FCGS (Fast 3D Gaussian Splatting)
    → 场景点云 (.ply 文件)
    ↓
轨迹采样器 (40 条新轨迹)
    → 螺旋/圆周/直线组合
    ↓
新视角渲染 (每条轨迹 961 帧)
    ↓
DiFix3D (深度修复)
    → 修复遮挡/空洞
    ↓
输出: 40 个新样本 (video + poses)
```

### 实现代码

```python
def stage03_3dgs_augmentation(original_video, original_poses, output_dir):
    # 1. FCGS 重建
    point_cloud = fcgs_reconstruct(
        images=original_video,
        poses=original_poses,
        iterations=30000,  # 论文推荐值
        sh_degree=3        # 球谐函数阶数
    )
    point_cloud.save(output_dir / 'scene.ply')
    
    # 2. 生成 40 条新轨迹
    trajectory_sampler = TrajectorySampler(
        scene_bounds=point_cloud.bounds,
        num_trajectories=40
    )
    new_trajectories = trajectory_sampler.sample([
        {'type': 'spiral', 'count': 10},
        {'type': 'circular', 'count': 10},
        {'type': 'linear', 'count': 20}
    ])
    
    # 3. 渲染新视角
    renderer = GaussianSplattingRenderer(point_cloud)
    for i, traj in enumerate(new_trajectories):
        video_frames = []
        for pose in traj:
            frame = renderer.render(pose, resolution=(1280, 720))
            video_frames.append(frame)
        
        # 4. DiFix3D 深度修复
        video_frames = difix3d_repair(video_frames)
        
        # 5. 保存
        save_video(video_frames, output_dir / f'aug_{i:03d}_video.mp4')
        np.save(output_dir / f'aug_{i:03d}_poses.npy', traj)
```

### 输出

```
output/stage03_3dgs_aug/
├── scene.ply                  # 3D 高斯点云
├── aug_000_video.mp4          # 增强样本 1
├── aug_000_poses.npy
├── aug_001_video.mp4          # 增强样本 2
├── aug_001_poses.npy
└── ...                        # 共 40 个新样本
```

### 限制与注意事项

1. **仅适用静态场景**：动态物体（行人/车辆）会产生鬼影
2. **计算成本高**：单个场景重建需 10-20 分钟（FCGS 30k 迭代）
3. **质量依赖原始轨迹**：如果原始 pose 不准，重建会失败

**生产决策**：本项目仅对 DL3DV 启用 Stage 3，其他数据集跳过。

---

## Stage 04: 视觉过滤

### 目标

过滤低质量样本：光流不连续、画面模糊、编码错误等。

### 三项检查

| 检查项 | 工具 | 阈值 | 用途 |
|--------|------|------|------|
| **光流连续性** | UniMatch | mean_flow ∈ [3, 100] | 检测运动合理性 |
| **视频质量** | DOVER | score > 0.5 | 检测模糊/抖动 |
| **VLM 验证** | Qwen3.5-VL | - | 场景理解检查 |

### 实现代码

```python
def stage04_filter(video_path, output_dir):
    frames = load_video(video_path)
    
    # 1. UniMatch 光流
    unimatch = load_model('unimatch')
    flows = []
    for t in range(len(frames) - 1):
        flow = unimatch(frames[t], frames[t+1])
        flows.append(flow)
    mean_flow = np.mean([np.linalg.norm(f) for f in flows])
    
    # 2. DOVER 质量评分
    dover = load_model('dover')
    quality_score = dover(video_path)
    
    # 3. Qwen VLM 验证
    qwen = load_model('qwen3.5-vl')
    first_frame = frames[0]
    scene_description = qwen.generate(first_frame, prompt="Describe this scene")
    
    # 4. 判定
    is_pass = (
        3 <= mean_flow <= 100 and
        quality_score > 0.5 and
        'error' not in scene_description.lower()
    )
    
    # 5. 保存结果
    result = {
        'mean_flow': mean_flow,
        'quality_score': quality_score,
        'scene_description': scene_description,
        'pass': is_pass
    }
    json.dump(result, open(output_dir / 'filter_result.json', 'w'))
```

### 输出

```
output/stage04_filter/
└── filter_result.json         # {"mean_flow": 15.3, "quality_score": 0.72, "pass": true}
```

---

## Stage 05: Caption 生成

### 目标

为每个样本生成 **scene-static** 的文字描述（静态场景描述，不包含相机运动词汇）。

### 关键约束

| 约束 | 值 | 原因 |
|------|---|------|
| **最小长度** | 50 字符 | 避免过于笼统（如 "A video"） |
| **禁止相机词** | "camera moves", "pans", "follows" 等 | 训练时相机运动由 pose 控制 |
| **场景聚焦** | 描述物体/环境/光照/天气 | 而非动作/情节 |

### 实现代码

```python
def stage05_caption(video_path, output_dir):
    # 1. 采样关键帧（首/中/尾）
    frames = load_video(video_path)
    keyframes = [frames[0], frames[len(frames)//2], frames[-1]]
    
    # 2. Qwen3.5-VL 生成描述
    qwen = load_model('qwen3.5-vl-27b')
    prompt = """Describe the scene in this video. Focus on:
    - Physical environment (indoor/outdoor, time of day)
    - Main objects and their spatial arrangement
    - Lighting and atmosphere
    - Visual style (realistic/game/animation)
    
    Do NOT describe camera movement or actions. Minimum 50 characters."""
    
    caption = qwen.generate(keyframes, prompt=prompt, max_tokens=200)
    
    # 3. 检查相机词
    camera_words = ['camera', 'pan', 'zoom', 'tilt', 'follow', 'track']
    has_camera_words = any(word in caption.lower() for word in camera_words)
    
    # 4. 如果包含相机词，重新生成
    if has_camera_words:
        caption = qwen.generate(
            keyframes,
            prompt=prompt + "\n\nIMPORTANT: Do NOT mention camera movement.",
            max_tokens=200
        )
    
    # 5. 保存
    with open(output_dir / 'caption.txt', 'w') as f:
        f.write(caption)
```

### 输出

```
output/stage05_caption/
└── caption.txt                # "A modern office interior with large windows..."
```

### Caption 质量示例

**✅ 好的 Caption**：
> "A modern office interior with large windows overlooking a city skyline. The space features minimalist furniture, warm wooden flooring, and soft natural lighting from the afternoon sun. Several potted plants are placed near the windows."

**❌ 差的 Caption（包含相机词）**：
> "The camera pans across a modern office, following the movement of people walking through the space."

---

## Stage 06: WebDataset 打包

### 目标

将所有中间产物打包为 WebDataset 格式（tar shards），供训练加载。

### Schema

每个样本包含 5 个文件：

```
<sample_id>.video.mp4          # 视频 (720p, 16fps, 961 帧)
<sample_id>.poses_c2w.npy      # 相机轨迹 (T, 4, 4)
<sample_id>.intrinsics.npy     # 内参 (T, 4)
<sample_id>.scale.npy          # 尺度 (T,)
<sample_id>.caption.txt        # 文字描述
```

### 打包规则

- **每个 shard**: 100 个样本（论文推荐值）
- **Shard 命名**: `000000.tar`, `000001.tar`, ...
- **压缩**: 不压缩（训练时随机访问需要）

### 实现代码

```python
def stage06_pack(input_dirs, output_dir, samples_per_shard=100):
    import tarfile
    
    shard_idx = 0
    samples_in_current_shard = 0
    current_tar = None
    
    for sample_id, sample_dir in enumerate(input_dirs):
        # 1. 创建新 shard
        if samples_in_current_shard == 0:
            if current_tar:
                current_tar.close()
            shard_path = output_dir / f'{shard_idx:06d}.tar'
            current_tar = tarfile.open(shard_path, 'w')
        
        # 2. 添加 5 个文件
        files = {
            'video.mp4': sample_dir / 'video_normalized.mp4',
            'poses_c2w.npy': sample_dir / 'poses_c2w.npy',
            'intrinsics.npy': sample_dir / 'intrinsics.npy',
            'scale.npy': sample_dir / 'scale.npy',
            'caption.txt': sample_dir / 'caption.txt'
        }
        
        for ext, filepath in files.items():
            arcname = f'{sample_id:08d}.{ext}'
            current_tar.add(filepath, arcname=arcname)
        
        # 3. 更新计数
        samples_in_current_shard += 1
        if samples_in_current_shard >= samples_per_shard:
            samples_in_current_shard = 0
            shard_idx += 1
    
    # 4. 关闭最后一个 shard
    if current_tar:
        current_tar.close()
```

### 输出

```
output/stage06_pack/
├── 000000.tar                 # Shard 0 (样本 0-99)
├── 000001.tar                 # Shard 1 (样本 100-199)
└── ...
```

### 训练加载示例

```python
import webdataset as wds

dataset = wds.WebDataset('output/stage06_pack/{000000..001000}.tar')
dataset = dataset.decode('rgb')
dataset = dataset.to_tuple('video.mp4', 'poses_c2w.npy', 'caption.txt')

for video, poses, caption in dataset:
    # video: (T, H, W, 3) tensor
    # poses: (T, 4, 4) array
    # caption: str
    train_step(video, poses, caption)
```

---

## 端到端示例

### 单样本完整流程

```bash
# 环境准备
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export VIPE_EXT_JIT=0

# 输入
INPUT_VIDEO=/path/to/raw_video.mp4
OUTPUT_ROOT=./output/sample_001

# Stage 1: Ingest
python -m sana_wm_pipeline.stage01_ingest.run \
  --input $INPUT_VIDEO \
  --output $OUTPUT_ROOT/stage01

# Stage 2: Pose
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --mode default \
  --input_video $OUTPUT_ROOT/stage01/video_normalized.mp4 \
  --output_dir $OUTPUT_ROOT/stage02 \
  --model_pi3x /mnt/afs/davidwang/models/pi3x/pi3x_full.pth \
  --model_moge2 /mnt/afs/davidwang/models/moge2/moge2_weights.pth

# Stage 4: Filter (跳过 Stage 3，非 DL3DV)
python -m sana_wm_pipeline.stage04_filter.run \
  --input_video $OUTPUT_ROOT/stage01/video_normalized.mp4 \
  --output_dir $OUTPUT_ROOT/stage04

# Stage 5: Caption
python -m sana_wm_pipeline.stage05_caption.run \
  --input_video $OUTPUT_ROOT/stage01/video_normalized.mp4 \
  --output_dir $OUTPUT_ROOT/stage05

# Stage 6: Pack
python -m sana_wm_pipeline.stage06_pack.run \
  --input_dirs $OUTPUT_ROOT/stage02 $OUTPUT_ROOT/stage05 \
  --output_dir $OUTPUT_ROOT/stage06

# 最终产物
ls $OUTPUT_ROOT/stage06/
# 00000000.video.mp4
# 00000000.poses_c2w.npy
# 00000000.intrinsics.npy
# 00000000.scale.npy
# 00000000.caption.txt
```

### 批量生产流程

```bash
# 1. 准备样本列表
python scripts/prepare_batch_list.py \
  --dataset wds-SpatialVID-hq \
  --output batch_list.txt

# 2. 启动 48 个 worker (CMCC)
python experiments/batch_production/run_batch.py \
  --sample_list batch_list.txt \
  --num_workers 48 \
  --output_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/

# 3. 监控进度
tail -f logs/worker_*.log

# 4. 合并结果
python scripts/merge_shards.py \
  --input_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
  --output_dir /root/work/filestorage/shangaoooooo/davidwang/final_shards/
```

---

## 性能优化建议

### 瓶颈识别

| Stage | 耗时占比 | 瓶颈 | 优化方案 |
|-------|---------|------|---------|
| Stage 1 | 10% | ffmpeg CPU | 使用 NVENC GPU 编码 |
| Stage 2 | 70% | Pi3X + MoGe-2 推理 | TensorRT 加速 |
| Stage 4 | 15% | UniMatch 光流 | Batch 推理 |
| Stage 5 | 5% | Qwen VLM | 量化为 INT4 |

### 已实现优化

1. ✅ **显存管理**：用后立即释放模型
2. ✅ **混合精度**：FP16 推理
3. ✅ **Batch 推理**：Pi3X 支持 batch=4

### 待实现优化

1. ⏳ **TensorRT**：推理加速 2-3 倍
2. ⏳ **异步 I/O**：预读取下一批样本
3. ⏳ **模型量化**：Qwen-27B → Qwen-14B-INT4

---

**相关文档**：
- 快速开始：`../QUICKSTART.md`
- 系统架构：`01-ARCHITECTURE.md`
- 质检系统：`03-QC_SYSTEM.md`
- 故障排查：`05-TROUBLESHOOTING.md`

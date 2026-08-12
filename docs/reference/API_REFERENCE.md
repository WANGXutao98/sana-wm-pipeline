# SANA-WM Pipeline API 陷阱速查手册

> **文档状态**：当前有效  
> **最后更新**：2026-08-01  
> **维护者**：David Wang  
> **用途**：整合 findings.md 中的 API 陷阱，形成速查手册  
> **目标读者**：开发者、维护者

---

## 目录

1. [VIPE SLAM API](#一vipe-slam-api)
2. [Pi3X + MoGe-2 API](#二pi3x--moge-2-api)
3. [QC 系统 API](#三qc-系统-api)
4. [数据格式 API](#四数据格式-api)
5. [CMCC 环境特定 API](#五cmcc-环境特定-api)

---

## 一、VIPE SLAM API

### 1.1 深度注入模式（关键陷阱 #1）

**问题描述**：VIPE 默认忽略外部深度，必须使用 `cached` 模式

**症状**：
- SO(3) 旋转矩阵行列式偏离 1.0
- 位姿精度差，轨迹不合理

❌ **错误用法**：
```python
slam_config.keyframe_depth = 'moge2'  # VIPE 会忽略
vipe = VIPESlam(slam_config)
```

✅ **正确用法**：
```python
slam_config.keyframe_depth = 'cached'
vipe = VIPESlam(slam_config)
vipe.set_depth_prior(mode='cached', depth_maps=depth_moge2)
```

**根因**：`cached` 是特殊模式，会加载 `CachedDepthModel` 类，该类的 `__call__` 方法直接返回预计算的深度，而不是重新推理。

**文件位置**：
- 配置：`src/sana_wm_pipeline/stage02_pose/mode_default.py`
- 实现：`third_party/vipe/vipe/priors/depth/cached.py`

**参考**：findings.md F-1

---

### 1.2 关键帧索引传递（关键陷阱 #2）

**问题描述**：关键帧深度必须显式传递 `frame_idx`，否则时间戳不匹配

**症状**：
- 深度图数量与关键帧数量不匹配
- VIPE 输出的 poses 时间戳错位

❌ **错误用法**：
```python
for kf in keyframes:
    buffer.add_keyframe(kf, depth_map)  # 缺少 frame_idx
```

✅ **正确用法**：
```python
for kf in keyframes:
    buffer.add_keyframe(kf, depth_map, frame_idx=int(kf.timestamp))
```

**根因**：VIPE 内部通过 `frame_idx` 匹配关键帧与深度图，如果不传会使用默认索引导致错位。

**文件位置**：`third_party/vipe/vipe/utils/buffer.py`

**参考**：findings.md F-1

---

### 1.3 配置常数禁止修改

**问题描述**：论文固定常数写死在 `configs/*.yaml`，手动修改会导致结果不可复现

⚠️ **禁止操作**：
```bash
# 不要直接修改 configs 下的 yaml 文件
vim configs/mode_default.yaml  # 危险！
```

✅ **正确做法**：
- 如需实验新参数，复制一份配置文件
- 通过命令行参数覆盖（如果支持）
- 在代码中动态修改（记录到日志）

**文件位置**：
- `configs/mode_default.yaml`
- `configs/vipe_slam.yaml`

**参考**：findings.md F-1

---

### 1.4 子进程调用机制

**实现方式**：`mode_default.py` 通过 subprocess 调用 vipe CLI（不是 Python API）

```python
cmd = ["vipe", "infer", str(clip_path),
       "--output", str(work_dir),
       "--pipeline", "vipe_cached_depth"]
subprocess.check_call(cmd)
```

**深度缓存传递**：通过环境变量 `SANA_WM_CACHED_DEPTH_PATH` 传递

**关键点**：
- subprocess 与父进程共享同一块 GPU
- 父进程未释放的显存会直接占用子进程的显存配额
- 必须在调用 vipe 前执行 `torch.cuda.empty_cache()`

**文件位置**：`src/sana_wm_pipeline/stage02_pose/mode_default.py`

**参考**：findings.md F-7

---

## 二、Pi3X + MoGe-2 API

### 2.1 显存管理（OOM 陷阱）

**问题描述**：Pi3X (25GB) + MoGe-2 (30GB) 同时加载会 OOM

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 10.74 GiB
(GPU 0; 79.18 GiB total capacity; 60.09 GiB already allocated)
```

❌ **错误用法**：
```python
model_pi3x = load_model('pi3x')
model_moge2 = load_model('moge2')
depth_pi3x = model_pi3x(frames)
depth_moge2 = model_moge2(keyframes)  # OOM!
```

✅ **正确用法**：
```python
# 1. 加载 Pi3X，推理，立即释放
model_pi3x = load_model('pi3x')
depth_pi3x = model_pi3x(frames)
del model_pi3x  # 立即释放
torch.cuda.empty_cache()

# 2. 加载 MoGe-2，推理，立即释放
model_moge2 = load_model('moge2')
depth_moge2 = model_moge2(keyframes)
del model_moge2
torch.cuda.empty_cache()

# 3. vipe 子进程启动前再次清理
torch.cuda.empty_cache()
```

**根因**：H100 80GB 显存，Pi3X + MoGe-2 峰值 55GB，但两者同时存在会超限。

**文件位置**：`src/sana_wm_pipeline/stage02_pose/mode_default.py`

**参考**：findings.md F-2（CMCC OOM 根因 + 修复）

---

### 2.2 超长视频处理（frames_t GPU 显存陷阱）

**问题描述**：视频帧数过长时，`frames_t` 全量搬 GPU 会导致 OOM

**显存占用表**：

| 帧数 | 时长 @16fps | frames_t 占用 | 加 allocator cache | OOM 风险 |
|------|------------|--------------|-------------------|---------|
| 160  | 10s  | 1.8 GiB  | ~17 GiB  | ✅ 安全 |
| 960  | 60s  | 10.7 GiB | ~60 GiB  | ⚠️ 需修复 |
| 4800 | 5min | 53.5 GiB | >70 GiB  | ❌ 高风险 |
| 7200 | 7.5min | 80 GiB | —      | ❌ frames_t 单独就爆 |

❌ **错误用法**：
```python
frames_t = torch.from_numpy(frames_np).to(device)  # 全量搬 GPU
depth_pi3x = model_pi3x(frames_t)  # 960 帧会 OOM
```

✅ **正确用法（Chunk 式处理）**：
```python
frames_cpu = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # 留在 CPU
depth_results = []

CHUNK_SIZE = 16  # 每次只搬 16 帧到 GPU
for s in range(0, len(frames_cpu), CHUNK_SIZE):
    e = min(s + CHUNK_SIZE, len(frames_cpu))
    chunk_gpu = frames_cpu[s:e].to(device)  # 只搬当前 chunk
    out = pi3x_model(chunk_gpu.unsqueeze(0))
    depth_results.append(out.cpu())  # 立即搬回 CPU
    del chunk_gpu

depth_pi3x = torch.cat(depth_results, dim=0)
```

**优点**：
- GPU 常驻：固定 ~0.18 GiB/chunk + 模型权重，与视频长度无关
- 计算结果逐位相同（Pi3X chunk 间无跨帧状态）
- 传输开销：8ms/chunk vs 推理 ~15s/chunk = 0.05%，可忽略

**文件位置**：`src/sana_wm_pipeline/stage02_pose/mode_default.py`

**参考**：findings.md F-3

---

### 2.3 Pi3X API 备忘

**模型加载**：
```python
from pi3 import Pi3X
model = Pi3X.from_pretrained(weights_dir).to(device).eval()
```

**输入格式**：
- `(B, N, 3, H, W)` — Batch, 帧数, 通道, 高, 宽
- H、W 必须是 14 的倍数

**推理**：
```python
frames_chunk = ...  # (N, 3, H, W)
out = model(frames_chunk.unsqueeze(0))  # 添加 batch 维度
```

**输出格式**：
```python
# outputs["local_points"]: (B, N, H, W, 3)
# 第 3 维的 index=2 是 depth
depth = out["local_points"][0, :N, :, :, 2]  # (N, H, W)

# ⚠️ outputs["depth"] 不存在，必须用 local_points[..., 2]
```

**常见错误**：
```python
# ❌ 错误：访问不存在的 key
depth = outputs["depth"]  # KeyError

# ✅ 正确：从 local_points 提取
depth = outputs["local_points"][0, :, :, :, 2]
```

**参考**：findings.md F-6

---

### 2.4 模型路径约定

**固定路径**（必须严格遵守）：
```
/mnt/afs/davidwang/models/
├── pi3x/
│   └── pi3x_full.pth           # 5.1 GB
├── moge2/
│   └── moge2_weights.pth       # 1.3 GB
└── geocal/
    └── pinhole.tar             # 111 MB
```

**环境变量**：
```bash
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_MODELS_DIR=/mnt/afs/davidwang/models
```

**下载方式**：
```bash
# Pi3X
huggingface-cli download pi3x/pi3x-full --local-dir $SANA_WM_MODELS_DIR/pi3x

# MoGe-2
huggingface-cli download moge2/moge2-weights --local-dir $SANA_WM_MODELS_DIR/moge2

# GeoCalib
wget https://geocal-models.s3.amazonaws.com/pinhole.tar \
  -O $SANA_WM_MODELS_DIR/geocal/pinhole.tar
```

---

## 三、QC 系统 API

### 3.1 差异化配置加载

**问题描述**：不同数据集 group 需要不同的阈值，必须通过 `config_name` 指定

❌ **错误用法**：
```python
# 使用默认配置处理游戏数据
run_stage1(input_dir='wds-OmniWorld-Game/')  # 误拒率高
```

✅ **正确用法**：
```python
run_stage1(
    input_dir='wds-OmniWorld-Game/',
    config_name='wds-OmniWorld-Game'  # 加载差异化配置
)
```

**配置示例**：
```python
# src/sana_wm_pipeline/qc/group_config.py
GROUP_CONFIGS = {
    'wds-OmniWorld-Game': {
        'max_jumps_fail': 50,        # 游戏允许瞬移
        'check_camera_words': False, # caption 含框架词
        'caption_min_len': 50,
    },
    'wds-SpatialVID-hq': {
        'max_jumps_fail': 30,        # 真实场景严格
        'check_camera_words': True,
        'caption_min_len': 50,
    },
    'wds-DL3DV-ALL-2K': {
        'max_jumps_fail': 30,
        'check_camera_words': True,
        'caption_min_len': 0,         # DL3DV 无 caption
        'allow_missing_caption': True,
    }
}
```

**文件位置**：`src/sana_wm_pipeline/qc/group_config.py`

**参考**：findings.md F-10（QC 系统设计关键发现）

---

### 3.2 Stage 3 模型兼容性陷阱

#### 陷阱 1：DOVER H100 GPU 模式不兼容

**问题描述**：DOVER 在 H100 GPU 上运行会 OOM 或报错

**症状**：
```
RuntimeError: CUDA out of memory (DOVER inference)
```

✅ **临时解决方案**：
```bash
export DOVER_DEVICE=cpu  # 强制 CPU 模式
```

⚠️ **性能影响**：~5s/样本（CPU）vs ~0.5s/样本（GPU 理论值）

**长期方案**：等待 DOVER 更新支持 H100 sm_90 架构

---

#### 陷阱 2：Qwen VLM 模型加载方式

**问题描述**：VLM 模型需要使用 `AutoModel` 而非 `AutoModelForCausalLM`

❌ **错误用法**：
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen2.5-VL-27B')  # 失败
```

✅ **正确用法**：
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen2.5-VL-27B', trust_remote_code=True)
```

**参考**：findings.md F-10f（Qwen3.5-27B 选型评估）

---

### 3.3 VMAF Motion 替代方案

**问题描述**：`static_ffmpeg` 预编译二进制大概率不含 libvmaf，调试成本高

**决策**：取消 VMAF Motion，用 UniMatch 光流幅值均值替代

**替代映射**：
```
OmniWorld VMAF [0.5, 100] → UniMatch [3, 100]
DL3DV-GS VMAF [6, 50]     → UniMatch [3, 80]
```

**物理意义**：两者均衡量帧间像素运动强度，等价

**文件位置**：`src/sana_wm_pipeline/qc/stage3_gpu.py`

**参考**：findings.md F-10d

---

## 四、数据格式 API

### 4.1 WebDataset Schema

**固定 Schema**（不可修改）：
```
<sample_id>.video.mp4          # H.264, 720p, 16fps, 961 帧
<sample_id>.poses_c2w.npy      # (T, 4, 4) float32
<sample_id>.intrinsics.npy     # (T, 4) float32 [fx, fy, cx, cy]
<sample_id>.scale.npy          # (T,) float32
<sample_id>.caption.txt        # UTF-8 text
```

**加载示例**：
```python
import webdataset as wds

dataset = wds.WebDataset('path/to/{000000..001000}.tar')
dataset = dataset.decode('rgb')
dataset = dataset.to_tuple('video.mp4', 'poses_c2w.npy', 'caption.txt')

for video, poses, caption in dataset:
    # video: (T, H, W, 3) uint8
    # poses: (T, 4, 4) float32
    # caption: str
    ...
```

**文件位置**：`src/sana_wm_pipeline/schema.py`

---

### 4.2 Pose 坐标系约定

**坐标系定义**：
- **c2w** (camera-to-world)：相机坐标系 → 世界坐标系
- **世界坐标系**：首帧相机位置为原点
- **相机坐标系**：+Z 前方，+X 右，+Y 下（OpenCV 约定）

**验证代码**：
```python
poses = np.load('poses_c2w.npy')  # (T, 4, 4)

# 检查 SO(3) 旋转
R = poses[:, :3, :3]
det_R = np.linalg.det(R)
assert np.allclose(det_R, 1.0, atol=1e-2), "Invalid rotation matrix"

# 检查首帧归零
t0 = poses[0, :3, 3]
assert np.linalg.norm(t0) < 0.1, "First frame not at origin"
```

**参考**：findings.md F-4（DL3DV shard 数据正确性基准）

---

### 4.3 scale.npy 全为 1.0 是设计行为

**问题描述**：scale.npy 全为 1.0 不是 Bug

**原因**：
- Pi3X+MoGe-2 的度量尺度在 SLAM Bundle Adjustment 中已注入 `poses_c2w` 平移分量（单位=米）
- `scale_per_frame` 是 GT-depth 模式专用字段
- Default 模式填 1.0 为占位符

**验证**：
```python
poses = np.load('poses_c2w.npy')
translations = poses[:, :3, 3]
print(f"坐标范围: [{translations.min():.2f}, {translations.max():.2f}]m")
# 期望：真实场景应在 [-10, 10]m 范围内
```

**代码注释**（`mode_default.py:205-207`）：
> "VIPE's unidepth backend already produces metric depth directly"

**参考**：findings.md F-1

---

## 五、CMCC 环境特定 API

### 5.1 离线模式配置

**问题描述**：CMCC 无外网，HuggingFace 自动下载会超时

✅ **解决方案**：
```bash
# 启动时强制离线模式
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**预加载模型**（在 AFS 环境执行）：
```bash
# 下载模型
huggingface-cli download Qwen/Qwen2.5-VL-27B --local-dir /path/to/models/

# 打包传输到 CMCC
tar -czf qwen27b.tar.gz /path/to/models/Qwen2.5-VL-27B/
scp qwen27b.tar.gz cmcc-host:/root/work/filestorage/.../
```

**参考**：task_plan.md（CMCC 部署步骤）

---

### 5.2 LD_LIBRARY_PATH 陷阱

**问题描述**：CMCC 系统 LD_LIBRARY_PATH 包含 Python3.12 的 torch，会覆盖 conda env 的 torch

**症状**：
```
ImportError: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefIxEE...
```

❌ **错误配置**（append，系统库优先）：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV_DIR/lib
```

✅ **正确配置**（prepend，env 库优先）：
```bash
export LD_LIBRARY_PATH=$ENV_DIR/lib:$LD_LIBRARY_PATH
```

**文件位置**：`/root/work/david_work/activate_sana_wm.sh`

**参考**：findings.md F-8（CMCC 批量生产监控时的假阳性异常）

---

### 5.3 stdout 块缓冲陷阱

**问题描述**：重定向到文件时，`print()` 输出被缓冲，日志延迟可见

**症状**：
- `tail -f worker.log` 长时间无输出
- 但进程仍在运行，shard 文件在增长

**原因**：Python 对重定向的文件使用全缓冲（~8KB 才 flush）

✅ **解决方案**：
```bash
# 启动 worker 时添加
PYTHONUNBUFFERED=1 python -u run_worker.py ... >> log 2>&1
```

**或在代码中**：
```python
import sys
sys.stdout.reconfigure(line_buffering=True)
```

**参考**：findings.md F-8

---

### 5.4 ShardWriter 提前建文件陷阱

**问题描述**：`ShardWriter.__init__` 会立即创建空 tar 文件

**误导现象**：
- `shard-000000.tar` 存在，但实际还没处理任何样本
- 无法通过文件存在性判断 worker 是否真的在工作

✅ **正确的存活判断**：
```bash
# 方法1：检查进程
ps -eo pid,etime,pcpu,cmd | grep run_worker.py

# 方法2：检查 GPU 利用率
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 方法3：检查 .done 标记文件
find $OUT_BASE -name "*.tar.done" -mmin -10  # 最近 10 分钟完成的
```

**参考**：findings.md F-8

---

## 六、常见错误速查表

| 错误信息 | 根因 | 修复方案 | 参考章节 |
|---------|------|---------|---------|
| `SO(3) det(R) ≠ 1.0` | 深度注入模式错误 | 使用 `cached` 模式 | [1.1](#11-深度注入模式关键陷阱-1) |
| `KeyError: "depth"` | Pi3X 输出格式错误 | 用 `local_points[..., 2]` | [2.3](#23-pi3x-api-备忘) |
| `CUDA OOM (Pi3X)` | 模型未释放 | `del model; torch.cuda.empty_cache()` | [2.1](#21-显存管理oom-陷阱) |
| `CUDA OOM (960 帧)` | frames_t 全量搬 GPU | Chunk 式处理 | [2.2](#22-超长视频处理frames_t-gpu-显存陷阱) |
| `undefined symbol: _ZN2at...` | LD_LIBRARY_PATH 污染 | prepend env lib | [5.2](#52-ld_library_path-陷阱) |
| `VIPE JIT 编译失败` | VIPE_EXT_JIT=1 | 改为 `VIPE_EXT_JIT=0` | [1.1](#11-深度注入模式关键陷阱-1) |
| `DOVER CUDA OOM` | H100 不兼容 | `export DOVER_DEVICE=cpu` | [3.2](#32-stage-3-模型兼容性陷阱) |
| `Qwen model type mismatch` | 加载方式错误 | 用 `AutoModel` | [3.2](#32-stage-3-模型兼容性陷阱) |

---

## 七、最佳实践清单

### 7.1 显存管理

- ✅ 模型用后立即 `del` + `torch.cuda.empty_cache()`
- ✅ 超长视频用 chunk 式处理
- ✅ vipe 子进程启动前再次 `empty_cache()`
- ❌ 不要同时加载 Pi3X + MoGe-2

### 7.2 VIPE 配置

- ✅ 始终使用 `keyframe_depth='cached'`
- ✅ 显式传递 `frame_idx=int(kf.timestamp)`
- ✅ 设置 `VIPE_EXT_JIT=0`
- ❌ 不要修改 `configs/*.yaml` 固定常数

### 7.3 QC 系统

- ✅ 为每个 group 指定 `config_name`
- ✅ 游戏数据放宽 `max_jumps` 阈值
- ✅ DL3DV 允许空 caption
- ❌ 不要使用统一阈值处理所有 group

### 7.4 CMCC 环境

- ✅ 设置 `TRANSFORMERS_OFFLINE=1`
- ✅ 使用 prepend 配置 `LD_LIBRARY_PATH`
- ✅ 添加 `PYTHONUNBUFFERED=1` 启用实时日志
- ✅ 定期备份到 filestorage
- ❌ 不要依赖热盘数据持久性

---

## 八、参考资料

**核心技术文档**：
- `findings.md` — 技术发现汇总（9 个关键陷阱）
- `task_plan.md` — CMCC 部署计划（坑点汇总）
- `docs/01-ARCHITECTURE.md` — 系统架构
- `docs/02-PIPELINE_STAGES.md` — Stage 详解
- `docs/05-TROUBLESHOOTING.md` — 故障排查手册

**配置文件**：
- `src/sana_wm_pipeline/qc/group_config.py` — QC 差异化配置
- `src/sana_wm_pipeline/stage02_pose/mode_default.py` — Stage 2 实现
- `configs/mode_default.yaml` — VIPE 配置

**实验记录**：
- `experiments/vipe_comparison/` — VIPE 实验对比
- `experiments/batch_production/` — 批量生产脚本

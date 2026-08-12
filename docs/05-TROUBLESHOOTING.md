# SANA-WM Pipeline 故障排查手册

> **文档状态**：当前有效  
> **最后更新**：2026-08-01  
> **维护者**：David Wang

---

## 目录

1. [环境问题](#一环境问题)
2. [Stage 2 位姿标注问题](#二stage-2-位姿标注问题)
3. [QC 系统问题](#三qc-系统问题)
4. [数据问题](#四数据问题)
5. [性能问题](#五性能问题)
6. [CMCC 特定问题](#六cmcc-特定问题)
7. [调试技巧](#七调试技巧)
8. [紧急恢复](#八紧急恢复)

---

## 一、环境问题

### 1.1 Conda 环境

#### 问题：`ModuleNotFoundError: No module named 'sana_wm_pipeline'`

**症状**：
```
Traceback (most recent call last):
  File "run_worker.py", line 3, in <module>
    from sana_wm_pipeline import ...
ModuleNotFoundError: No module named 'sana_wm_pipeline'
```

**原因**：项目未安装到 Python 环境

**解决方案**：
```bash
cd /path/to/sana_wm_pipeline
pip install -e . --no-deps
```

---

#### 问题：`CUDA out of memory`

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 10.70 GiB (GPU 0; 79.18 GiB total capacity)
```

**原因**：
1. Pi3X + MoGe-2 同时加载（55GB+）
2. 超长视频 frames_t 全量搬 GPU（~10GB/960帧）
3. 父进程显存未释放，vipe 子进程无法申请

**解决方案**：

参考 **findings.md F-2**，应用 OOM 修复：

```python
# 1. Pi3X 后立即释放
del pi3x_model, src, accum, count
torch.cuda.empty_cache()

# 2. MoGe-2 后立即释放
del moge2_model, frames_t
torch.cuda.empty_cache()

# 3. vipe 子进程启动前再次清理
torch.cuda.empty_cache()
```

**超长视频优化**（参考 F-3）：

```python
# 改为 chunk 式逐批搬帧
frames_cpu = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
for s in range(0, len(frames_cpu), CHUNK_SIZE):
    chunk_gpu = frames_cpu[s:s+CHUNK_SIZE].to(device)
    out = pi3x_model(chunk_gpu.unsqueeze(0))
    depth_results.append(out.cpu())
```

---

#### 问题：`libcudart.so.11.0: cannot open shared object file`

**症状**：
```
OSError: libcudart.so.11.0: cannot open shared object file: No such file or directory
```

**原因**：PyTorch 版本与 CUDA 版本不匹配

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version  # 应为 13.0

# 重新安装正确版本的 PyTorch
pip uninstall torch torchvision
pip install torch==2.12.0+cu130 torchvision==0.17.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
```

---

### 1.2 模型加载

#### 问题：Pi3X 模型加载失败

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/torch/hub/pi3x_full.pth'
```

**原因**：模型权重路径不正确或未设置 TORCH_HOME

**解决方案**：
```bash
# 方法1：设置环境变量
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export SANA_WM_MODELS_DIR=/mnt/afs/davidwang/models

# 方法2：手动指定模型路径
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --pi3x_weights /path/to/pi3x_full.pth \
  --moge2_weights /path/to/moge2_weights.pth
```

---

#### 问题：MoGe-2 权重不匹配

**症状**：
```
RuntimeError: Error(s) in loading state_dict for MoGe2Model:
  size mismatch for encoder.layer0.weight: copying a param with shape ...
```

**原因**：模型权重版本与代码不匹配

**解决方案**：
```bash
# 重新下载正确版本的权重
huggingface-cli download moge2/moge2-weights --revision main \
  --local-dir /path/to/models/moge2

# 或使用 ModelScope 镜像
modelscope download --model moge2/moge2-weights \
  --local_dir /path/to/models/moge2
```

---

#### 问题：GeoCalib pinhole.tar 解压错误

**症状**：
```
EOFError: Compressed file ended before the end-of-stream marker was reached
```

**原因**：下载不完整或文件损坏

**解决方案**：
```bash
# 验证文件完整性
md5sum /path/to/geocal/pinhole.tar
# 期望：<正确的MD5值>

# 重新下载
rm /path/to/geocal/pinhole.tar
wget https://geocal-models.s3.amazonaws.com/pinhole.tar \
  -O /path/to/models/geocal/pinhole.tar
```

---

## 二、Stage 2 位姿标注问题

### 2.1 VIPE SLAM 错误

#### 问题：`VIPE_EXT_JIT=1` 触发 JIT 编译失败

**症状**：
```
RuntimeError: NVCC JIT compilation failed
```

**原因**：JIT 编译绕过预编译产物，在 CMCC 环境易失败

**解决方案**：
```bash
# 强制使用预编译扩展
export VIPE_EXT_JIT=0
```

---

#### 问题：深度注入未生效（det(R) 偏差大）

**症状**：
```
检测到 SO(3) 旋转矩阵不合法：det(R) = 0.87（期望 1.0）
```

**原因**：未正确配置 VIPE 深度注入模式（参考 **findings.md F-6**）

**解决方案**：

❌ **错误用法**：
```python
slam_config.keyframe_depth = 'moge2'  # VIPE 会忽略
```

✅ **正确用法**：
```python
slam_config.keyframe_depth = 'cached'  # 必须用 cached 模式
vipe = VIPESlam(slam_config)
vipe.set_depth_prior(mode='cached', depth_maps=depth_moge2)
```

**文件位置**：`src/sana_wm_pipeline/stage02_pose/mode_default.py`

---

#### 问题：关键帧索引不匹配

**症状**：
```
AssertionError: 深度图数量 (15) 与关键帧数量 (16) 不匹配
```

**原因**：未显式传递 `frame_idx`（参考 **findings.md F-6**）

**解决方案**：

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

---

### 2.2 显存管理

参考 [一、环境问题 → 1.1 → CUDA OOM](#问题cuda-out-of-memory)

---

### 2.3 API 陷阱汇总

#### 陷阱 1：深度注入模式错误

**问题描述**：VIPE 默认忽略外部深度，必须使用 `cached` 模式

**错误代码**：
```python
slam_config.keyframe_depth = 'moge2'  # 不生效
vipe = VIPESlam(slam_config)
```

**正确代码**：
```python
slam_config.keyframe_depth = 'cached'
vipe = VIPESlam(slam_config)
vipe.set_depth_prior(mode='cached', depth_maps=depth_moge2)
```

**根因**：`cached` 是特殊模式，加载 `CachedDepthModel` 类直接返回预计算深度

**文件位置**：`third_party/vipe/vipe/priors/depth/cached.py`

---

#### 陷阱 2：关键帧索引传递

**问题描述**：关键帧深度必须显式传递 `frame_idx`

**错误代码**：
```python
for kf in keyframes:
    buffer.add_keyframe(kf, depth_map)  # 时间戳不匹配
```

**正确代码**：
```python
for kf in keyframes:
    buffer.add_keyframe(kf, depth_map, frame_idx=int(kf.timestamp))
```

**根因**：VIPE 内部通过 `frame_idx` 匹配关键帧与深度图

**文件位置**：`third_party/vipe/vipe/utils/buffer.py`

---

#### 陷阱 3：显存泄漏

**问题描述**：Pi3X + MoGe-2 同时存在会 OOM

**错误代码**：
```python
depth_pi3x = model_pi3x(frames)
depth_moge2 = model_moge2(keyframes)  # OOM!
```

**正确代码**：
```python
depth_pi3x = model_pi3x(frames)
del model_pi3x
torch.cuda.empty_cache()

depth_moge2 = model_moge2(keyframes)
del model_moge2
torch.cuda.empty_cache()
```

**根因**：H100 80GB 显存，Pi3X + MoGe-2 峰值 55GB

---

## 三、QC 系统问题

### 3.1 Stage 1 问题

#### 问题：`KeyError: 'poses_c2w'`

**症状**：
```
KeyError: 'poses_c2w' not found in npz file
```

**原因**：WebDataset tar 文件中的 `.npz` 文件格式不正确

**解决方案**：
```python
# 检查 npz 文件内容
import numpy as np
data = np.load('sample.camera.npz')
print(data.files)  # 应包含 'poses_c2w'

# 如果缺失，检查数据生产脚本
```

---

#### 问题：SO(3) 旋转检查过严

**症状**：
```
大量样本因 det(R) = 0.999998 被拒绝（阈值 1e-6 过严）
```

**原因**：浮点精度导致旋转矩阵行列式略微偏离 1.0

**解决方案**：
```python
# 放宽阈值（在 metrics.py 中）
def check_so3_validity(poses_c2w, tolerance=1e-2):  # 从 1e-6 改为 1e-2
    R = poses_c2w[:, :3, :3]
    det_R = np.linalg.det(R)
    is_valid = np.abs(det_R - 1.0) < tolerance
    return is_valid
```

---

#### 问题：Caption 长度检查失败（DL3DV）

**症状**：
```
DL3DV 数据集 65% 样本因 caption 长度 < 50 被拒绝
```

**原因**：DL3DV 数据源无 caption（设计问题）

**解决方案**：
```python
# 在 group_config.py 中添加差异化配置
GROUP_CONFIGS = {
    'wds-DL3DV-ALL-2K': {
        'caption_min_len': 0,  # 允许空 caption
        'allow_missing_caption': True,
    },
    # 其他 group 保持 caption_min_len=50
}
```

---

### 3.2 Stage 2 问题

#### 问题：黑帧检测误判（低光场景）

**症状**：
```
夜景视频被误判为黑帧比例过高（>5%）
```

**原因**：固定阈值 10 过高，低光场景平均亮度 < 10

**解决方案**：
```python
# 使用自适应阈值
def check_black_frames_adaptive(video_path, percentile=5, max_ratio=0.05):
    frames = load_video(video_path)
    brightness = np.mean(frames, axis=(1, 2, 3))
    threshold = np.percentile(brightness, percentile)  # 动态阈值
    black_frames = (brightness < threshold).sum()
    black_ratio = black_frames / len(frames)
    return black_ratio < max_ratio
```

---

#### 问题：场景切换阈值过低（游戏数据）

**症状**：
```
游戏数据场景切换检测过于敏感，正常镜头切换被误判
```

**原因**：游戏引擎渲染的场景切换与真实世界不同

**解决方案**：
```python
# 差异化阈值配置
GROUP_CONFIGS = {
    'wds-OmniWorld-Game': {
        'scene_cut_threshold': 800,  # 游戏数据更宽松
        'max_cuts': 5,
    },
    'wds-SpatialVID-hq': {
        'scene_cut_threshold': 500,  # 真实数据更严格
        'max_cuts': 3,
    },
}
```

---

### 3.3 Stage 3 问题

#### 问题：DOVER CUDA out of memory

**症状**：
```
RuntimeError: CUDA out of memory (DOVER 模型加载失败)
```

**原因**：H100 GPU 模式不兼容（驱动问题）

**临时解决方案**：
```bash
# 强制 CPU 模式
export DOVER_DEVICE=cpu
```

**性能影响**：~5s/样本（CPU）vs ~0.5s/样本（GPU 理论值）

**长期方案**：
- 等待 DOVER 官方修复 H100 兼容性
- 或使用其他视频质量评估模型（如 BRISQUE）

---

#### 问题：UniMatch 导入失败

**症状**：
```
ModuleNotFoundError: No module named 'unimatch'
```

**原因**：sys.path 未正确设置

**解决方案**：
```python
# 在 stage3_gpu.py 开头添加
import sys
sys.path.insert(0, '/path/to/UniMatch')
import unimatch
```

---

#### 问题：Qwen model type mismatch

**症状**：
```
ValueError: Qwen2.5-VL-27B is not a CausalLM model
```

**原因**：使用了错误的模型加载方式

**解决方案**：

❌ **错误用法**：
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen2.5-VL-27B')
```

✅ **正确用法**：
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen2.5-VL-27B', trust_remote_code=True)
```

---

## 四、数据问题

### 4.1 视频格式

#### 问题：ffprobe 无法解析视频

**症状**：
```
Error: Invalid data found when processing input
```

**原因**：视频文件损坏或编码格式不支持

**解决方案**：
```bash
# 检查视频完整性
ffprobe -v error -show_format sample.mp4

# 重新编码
ffmpeg -i input.mp4 -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k output.mp4
```

---

#### 问题：帧率不是 16fps

**症状**：
```
AssertionError: 期望帧率 16fps，实际 30fps
```

**原因**：数据源视频帧率不符合要求

**解决方案**：
```bash
# 转换帧率
ffmpeg -i input.mp4 -r 16 -c:v libx264 -preset medium output.mp4
```

---

#### 问题：分辨率不是 720p

**症状**：
```
AssertionError: 期望分辨率 1280x720，实际 1920x1080
```

**解决方案**：
```bash
# 缩放到 720p
ffmpeg -i input.mp4 -vf scale=1280:720 -c:v libx264 -preset medium output.mp4
```

---

### 4.2 Pose 数据

#### 问题：poses_c2w.npy 形状不匹配

**症状**：
```
AssertionError: poses_c2w shape (T, 3, 4), expected (T, 4, 4)
```

**原因**：数据生产脚本输出格式错误

**解决方案**：
```python
# 检查 poses 形状
poses = np.load('poses_c2w.npy')
print(poses.shape)  # 应为 (T, 4, 4)

# 如果是 (T, 3, 4)，需要补齐第 4 行
poses_full = np.zeros((T, 4, 4))
poses_full[:, :3, :] = poses
poses_full[:, 3, 3] = 1.0
np.save('poses_c2w_fixed.npy', poses_full)
```

---

#### 问题：旋转矩阵非正交

**症状**：
```
检测到旋转矩阵不正交：R @ R.T != I
```

**原因**：SLAM 输出不稳定或数值精度问题

**解决方案**：
```python
# 正交化修复（Gram-Schmidt）
def orthogonalize_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    # 确保 det(R) = 1（非反射）
    if np.linalg.det(R_ortho) < 0:
        Vt[-1, :] *= -1
        R_ortho = U @ Vt
    return R_ortho

poses_c2w[:, :3, :3] = np.array([orthogonalize_rotation(R) for R in poses_c2w[:, :3, :3]])
```

---

#### 问题：平移向量异常（>100m）

**症状**：
```
检测到异常轨迹：平移向量超过 100m
```

**原因**：SLAM 尺度估计错误或数据损坏

**解决方案**：
```python
# 检查轨迹合理性
translations = poses_c2w[:, :3, 3]
max_translation = np.max(np.linalg.norm(translations, axis=1))
print(f"最大平移: {max_translation:.2f}m")

# 如果异常，重新运行 Stage 2
```

---

## 五、性能问题

### 5.1 速度慢

#### 症状：单样本耗时 > 60s

**排查步骤**：

**1. 检查 GPU 利用率**
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv --loop=1
```

期望：GPU 利用率 > 80%，显存占用 > 40GB

**2. 检查 I/O 瓶颈**
```bash
iostat -x 1
```

期望：await < 10ms

**3. 检查显存是否充足**
```bash
nvidia-smi
```

如果显存占用 > 75GB，参考 [CUDA OOM 解决方案](#问题cuda-out-of-memory)

---

### 5.2 吞吐量低

#### 症状：48 GPU 吞吐量 < 1000 样本/h

**排查步骤**：

**1. 检查 worker 负载均衡**
```bash
# 查看每个 worker 处理的样本数
for log in $OUT_BASE/logs/node*_gpu*.log; do
  echo "$log: $(grep -c '[OK]' $log)"
done
```

期望：各 worker 样本数基本均衡（±10%）

**2. 检查数据预读取是否启用**
```python
# 在 run_worker.py 中启用异步 I/O
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_data, path) for path in paths]
```

**3. 检查网络 I/O（共享存储）**
```bash
# 测试读写速度
dd if=/dev/zero of=$OUT_BASE/testfile bs=1G count=1 oflag=direct
dd if=$OUT_BASE/testfile of=/dev/null bs=1G iflag=direct

# 期望：写入 > 500 MB/s，读取 > 1 GB/s
```

---

## 六、CMCC 特定问题

### 6.1 环境问题

#### 问题：LD_LIBRARY_PATH 污染

**症状**：
```
ImportError: undefined symbol: _ZN2at4_ops...
```

**原因**：系统 Python3.12 torch 覆盖了 conda env torch

**解决方案**：

❌ **错误配置**（append，系统库优先）：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV_DIR/lib
```

✅ **正确配置**（prepend，env 库优先）：
```bash
export LD_LIBRARY_PATH=$ENV_DIR/lib:$LD_LIBRARY_PATH
```

**文件位置**：`activate_sana_wm.sh`

---

#### 问题：HuggingFace 模型下载超时

**症状**：
```
requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='huggingface.co', port=443)
```

**原因**：CMCC 环境无外网访问

**解决方案**：
```bash
# 启用离线模式
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 预先在 AFS 下载模型，打包传输到 CMCC
```

---

### 6.2 数据持久化问题

#### 问题：重启后数据丢失

**症状**：
```
重启机器后，/root/work/david_work/ 目录为空
```

**原因**：数据存在热盘（非持久化）

**解决方案**：
```bash
# 备份到持久盘
rsync -av /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
          /root/work/filestorage/shangaoooooo/davidwang/backup/

# 添加 cron 定时备份
crontab -e
0 2 * * * rsync -av /root/work/externalstorage/.../jdvbbfb_output/ /root/work/filestorage/.../backup/
```

---

## 七、调试技巧

### 7.1 启用详细日志

```bash
# 设置环境变量
export SANA_WM_DEBUG=1
export VIPE_VERBOSE=1

# 运行 worker
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --input_video test.mp4 \
  --output_dir ./debug_output \
  --verbose
```

---

### 7.2 单样本调试

```bash
# 只处理一个样本，快速定位问题
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --input_video test.mp4 \
  --output_dir ./debug_output \
  --verbose \
  --max_samples 1
```

---

### 7.3 性能分析

```bash
# 使用 py-spy 分析性能瓶颈
pip install py-spy

# 实时监控
py-spy top -- python -m sana_wm_pipeline.stage02_pose.run_worker ...

# 生成火焰图
py-spy record -o profile.svg -- python -m sana_wm_pipeline.stage02_pose.run_worker ...
```

---

### 7.4 显存分析

```bash
# 实时显存监控
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv --loop=1 > gpu_mem.log

# 分析显存峰值
grep "MiB" gpu_mem.log | awk -F',' '{print $2}' | sort -n | tail -1
```

---

## 八、紧急恢复

### 8.1 Worker 崩溃

**检查日志**：
```bash
tail -f $OUT_BASE/logs/worker_*.log
```

**重启单个 worker**：
```bash
# 查找崩溃的 worker
ps aux | grep run_worker.py

# 杀死僵尸进程
kill -9 <PID>

# 重启（假设是 node0 gpu3）
CUDA_VISIBLE_DEVICES=3 \
python -m sana_wm_pipeline.experiments.batch_production.run_worker \
  --group wds-DL3DV-ALL-2K \
  --worker_id 3 \
  --output_dir $OUT_BASE
```

---

### 8.2 数据损坏

**验证 MD5**：
```bash
md5sum -c md5sum.txt | grep -v OK
```

**从备份恢复**：
```bash
rsync -av /root/work/filestorage/.../backup/ \
          /root/work/externalstorage/.../jdvbbfb_output/
```

---

### 8.3 系统资源耗尽

**磁盘空间不足**：
```bash
# 检查磁盘使用
df -h

# 清理临时文件
rm -rf /tmp/*
rm -rf $NEW_BASE/cache/*

# 压缩旧日志
gzip $OUT_BASE/logs/*.log
```

**内存不足**：
```bash
# 检查内存使用
free -h

# 清理页缓存
sync; echo 3 > /proc/sys/vm/drop_caches
```

---

## 九、联系支持

**维护者**：David Wang  
**技术文档**：
- `docs/01-ARCHITECTURE.md` — 系统架构
- `docs/02-PIPELINE_STAGES.md` — Stage 详解
- `docs/04-DEPLOYMENT.md` — 部署指南

**实验记录**：
- `experiments/vipe_comparison/` — VIPE 实验
- `findings.md` — 技术发现汇总（9 个关键陷阱）

**常见问题汇总**：
- `task_plan.md` — CMCC 坑点汇总（9 个常见错误）

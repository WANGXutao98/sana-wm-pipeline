# SANA-WM Pipeline 快速开始

> **目标读者**：首次接触本项目的开发者  
> **预计时间**：30 分钟从零到产出第一个样本  
> **最后更新**：2026-08-01

---

## 前置条件

- ✅ **H100 GPU**（80GB 显存）或同等算力
- ✅ **200GB+ 可用磁盘空间**
- ✅ **Conda** 已安装（推荐 Miniconda3）
- ✅ **持久化目录**：`/mnt/afs/davidwang/workspace`（AFS 存储）

---

## 30 分钟上手流程

### Step 1: 环境准备（10 分钟）

```bash
# 1.1 进入工作目录
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 1.2 创建 Conda 环境（必须在 AFS 路径下，机器重启后不丢失）
conda create -p /mnt/afs/davidwang/miniconda3/envs/sana_wm python=3.10 -y

# 1.3 激活环境
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm

# 1.4 安装依赖（约 5 分钟）
pip install -e . --no-deps
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: 下载模型权重（15 分钟）

```bash
# 2.1 创建模型目录
mkdir -p /mnt/afs/davidwang/models/{pi3x,moge2,geocal}

# 2.2 下载 Pi3X（视频深度估计）
# 访问 https://github.com/yyfz/Pi3 下载 pi3x_full.pth (~5.1 GB)
# 放置到 /mnt/afs/davidwang/models/pi3x/

# 2.3 下载 MoGe-2（单帧米制深度）
# 访问 https://github.com/microsoft/MoGe 下载 moge2_weights.pth (~1.3 GB)
# 放置到 /mnt/afs/davidwang/models/moge2/

# 2.4 下载 GeoCalib（相机内参估计）
wget https://huggingface.co/geopose/GeoCalib/resolve/main/pinhole.tar \
  -O /mnt/afs/davidwang/models/geocal/pinhole.tar

# 2.5 设置环境变量（必须！）
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
```

**重要提示**：模型权重总计约 6.5 GB，下载时间取决于网络速度。权重放在 AFS 路径下，机器重启后无需重新下载。

### Step 3: 单样本测试（5 分钟）

```bash
# 3.1 准备测试视频（使用项目自带样本）
export TEST_VIDEO="tests/data/sample_video.mp4"

# 3.2 运行 Stage 2（姿态标注）- 核心管线
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --mode default \
  --input_video $TEST_VIDEO \
  --output_dir ./output/test_run \
  --model_pi3x /mnt/afs/davidwang/models/pi3x/pi3x_full.pth \
  --model_moge2 /mnt/afs/davidwang/models/moge2/moge2_weights.pth

# 预期输出：
# ✅ poses_c2w.npy         # 相机轨迹（米制 6-DoF）
# ✅ intrinsics.npy        # 内参 [fx, fy, cx, cy]
# ✅ scale.npy             # 度量尺度（Default 模式为 1.0）
# ✅ video_normalized.mp4  # 720p@16fps 标准化视频
```

**验证成功标志**：
- `poses_c2w.npy` 形状为 `(T, 4, 4)`，其中 T 为帧数
- 无 CUDA OOM 错误（Pi3X + MoGe-2 显存峰值 < 60GB）
- 运行时间：60 帧视频约 30 秒

---

## 常见问题速查

### Q1: 找不到 `sana_wm_pipeline` 模块
**原因**：未以可编辑模式安装  
**解决**：`pip install -e . --no-deps`

### Q2: CUDA out of memory
**原因**：Pi3X + MoGe-2 显存峰值约 55GB  
**解决**：确保使用 H100 80GB 或修改 `mode_default.py` 启用显存释放补丁

### Q3: 模型下载超时
**原因**：HuggingFace 网络不稳定  
**解决**：使用镜像站或手动下载后放置到指定路径

### Q4: `VIPE_EXT_JIT=1` 报错
**原因**：JIT 编译与某些环境不兼容  
**解决**：启动前执行 `export VIPE_EXT_JIT=0`

---

## 下一步

✅ **单样本测试通过**后，可进入以下流程：

1. **批量生产**：参考 `docs/02-PIPELINE_STAGES.md` 了解 6 个 Stage 全流程
2. **质检系统**：参考 `docs/03-QC_SYSTEM.md` 对产出数据进行质量评估
3. **CMCC 部署**：参考 `docs/04-DEPLOYMENT.md` 在生产环境部署

---

## 技术支持

- **详细架构**：`docs/01-ARCHITECTURE.md`
- **故障排查**：`docs/05-TROUBLESHOOTING.md`
- **API 陷阱**：`docs/reference/API_REFERENCE.md`
- **联系维护者**：David Wang

---

**最后提示**：所有路径必须使用 AFS 持久化存储（`/mnt/afs/davidwang/`），避免机器重启后数据丢失。

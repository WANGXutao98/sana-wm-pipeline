# DOVER H100 部署执行方案（CMCC 机器）

> **目标**：在 CMCC 机器上成功用 H100 GPU 运行 DOVER 视频质量评估  
> **环境选择**：优先在现有 `sana_wm_qc_env` 上安装，失败则传输 `dover_h100_test` 环境  
> **关键难点**：DOVER 在 H100 GPU 上的兼容性已通过本地验证（2026-07-02）  
> **创建日期**：2026-08-03  
> **验证状态**：✅ 已在 CMCC 机器验证通过（2026-08-03）

---

## 📋 方案总览

```
方案 A（优先）: 在 CMCC 现有 sana_wm_qc_env 环境上安装 DOVER
    ├─ 步骤 1: 环境验证（PyTorch + CUDA）
    ├─ 步骤 2: 传输 DOVER 代码和权重
    ├─ 步骤 3: 安装依赖包
    ├─ 步骤 4: 运行测试脚本验证
    └─ 步骤 5: 性能基准测试
    
方案 B（备用）: 打包传输 dover_h100_test 完整环境
    ├─ 步骤 1: 本地打包环境（AFS）
    ├─ 步骤 2: 传输到 CMCC
    ├─ 步骤 3: 解压并激活
    └─ 步骤 4: 运行测试验证
```

---

## 🎯 方案 A：在 sana_wm_qc_env 上安装（优先执行）

### 前置准备：传输文件到 CMCC

**在 AFS 机器执行（需要你本地操作）：**

```bash
# 1. 打包 DOVER 代码和权重
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/models/
tar -czf dover_package.tar.gz DOVER/

# 2. 验证打包大小（应该约 240MB）
ls -lh dover_package.tar.gz

# 3. 记录 MD5（用于传输验证）
md5sum dover_package.tar.gz > dover_package.md5

# 4. 通过你的方式传输到 CMCC（例如：U盘、内网传输等）
# 目标路径：/root/work/filestorage/shangaoooooo/davidwang/dover_package.tar.gz
```

---

### 步骤 1：CMCC 环境验证（30 分钟）

**在 CMCC 机器执行：**

```bash
# 1.1 激活目标环境
conda activate sana_wm_qc_env

# 1.2 验证 Python 版本
python --version
# 期望：Python 3.10.x 或 3.11.x

# 1.3 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**期望输出：**
```
PyTorch: 2.x.x+cu130 (或其他版本)
CUDA available: True
CUDA version: 13.0 (或其他版本)
GPU: NVIDIA H100 80GB
```

**⚠️ 关键检查点：**
- [ ] `CUDA available` 必须是 `True`
- [ ] GPU 名称包含 `H100`
- [ ] CUDA 版本与 PyTorch 匹配

**如果 CUDA available: False：**
```bash
# 检查环境变量
echo $CUDA_VISIBLE_DEVICES

# 如果为空字符串（不是未设置），取消设置
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    unset CUDA_VISIBLE_DEVICES
fi

# 重新测试
python -c "import torch; print(torch.cuda.is_available())"
```

---

### 步骤 2：解压 DOVER 代码和权重（5 分钟）

```bash
# 2.1 创建目标目录
mkdir -p /root/work/david_work/models/

# 2.2 解压
cd /root/work/david_work/models/
tar -xzf /root/work/filestorage/shangaoooooo/davidwang/dover_package.tar.gz

# 2.3 验证解压结果
ls -la DOVER/
# 期望看到：README.md, dover/, evaluate_one_video.py, pretrained_weights/ 等

# 2.4 验证权重文件
ls -lh DOVER/pretrained_weights/DOVER.pth
# 期望：239729097 字节（约 229MB）

# 2.5 验证 MD5（可选）
md5sum DOVER/pretrained_weights/DOVER.pth
# 期望：与 AFS 上的文件一致
```

---

### 步骤 3：安装 DOVER 依赖（10 分钟）

```bash
# 3.1 确保在正确环境
conda activate sana_wm_qc_env

# 3.2 进入 DOVER 目录
cd /root/work/david_work/models/DOVER

# 3.3 安装依赖（分批安装，便于排查问题）
# 第一批：核心依赖
pip install decord opencv-python scipy numpy tqdm

# 第二批：模型相关
pip install timm einops

# 第三批：视频处理（关键依赖，必须安装）
pip install scikit-video

# 第四批：其他工具（可选，测试不强制要求）
pip install thop==0.0.31-2005241907 || echo "thop 安装失败，跳过（非关键依赖）"

# 3.4 验证安装
python -c "import decord; import cv2; import timm; import einops; import skvideo; print('✅ 核心依赖安装成功')"
```

**期望输出：**
```
✅ 核心依赖安装成功
```

**⚠️ 常见问题**：
- **缺少 `skvideo`**：必须安装 `pip install scikit-video`，否则 DOVER 导入失败
- **pip 超时**：使用 `pip install --timeout=300 <package>`
- **权限问题**：使用 `pip install --user <package>`

---

### 步骤 4：运行测试脚本验证（核心步骤，15 分钟）

**4.1 准备测试脚本**

```bash
# 创建测试脚本
cat > /tmp/test_dover_cmcc.py << 'EOF'
#!/usr/bin/env python3
"""
DOVER H100 测试脚本 - CMCC 版本
测试 DOVER 在 H100 GPU 上的完整工作流
"""
import torch
import yaml
import sys
from pathlib import Path

print("=" * 60)
print("DOVER H100 兼容性测试 - CMCC 环境")
print("=" * 60)

# 步骤 1: 环境检查
print("\n[1/5] 环境检查")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 版本: {torch.version.cuda}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 计算能力: {torch.cuda.get_device_capability(0)}")  # H100 应该是 (9, 0)
    print(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("  ❌ 错误：CUDA 不可用")
    sys.exit(1)

# 步骤 2: 加载 DOVER 代码
print("\n[2/5] 加载 DOVER 代码")
dover_dir = Path("/root/work/david_work/models/DOVER")
sys.path.insert(0, str(dover_dir))

try:
    from dover import DOVER
    print("  ✅ DOVER 模块加载成功")
except ImportError as e:
    print(f"  ❌ 错误：{e}")
    sys.exit(1)

# 步骤 3: 加载配置和权重
print("\n[3/5] 加载配置和权重")
dover_config = dover_dir / "dover.yml"
dover_weight = dover_dir / "pretrained_weights/DOVER.pth"

if not dover_config.exists():
    print(f"  ❌ 配置文件不存在: {dover_config}")
    sys.exit(1)
if not dover_weight.exists():
    print(f"  ❌ 权重文件不存在: {dover_weight}")
    sys.exit(1)

print(f"  配置文件: {dover_config}")
print(f"  权重文件: {dover_weight} ({dover_weight.stat().st_size / 1024**2:.1f} MB)")

with open(dover_config, "r") as f:
    dover_opt = yaml.safe_load(f)

print("  ✅ 配置加载成功")

# 步骤 4: 初始化模型并移到 GPU（关键测试点）
print("\n[4/5] 初始化模型并移到 GPU（关键测试）")
device = "cuda"

try:
    model = DOVER(**dover_opt["model"]["args"])
    print("  ✅ 模型对象创建成功")
    
    model.load_state_dict(
        torch.load(dover_weight, map_location="cpu", weights_only=False)
    )
    print("  ✅ 权重加载成功")
    
    # 关键：移到 GPU（如果不兼容会在这里报错或 coredump）
    model = model.to(device)
    print("  ✅ 模型成功移到 GPU（H100 兼容性验证通过）")
    
    model.eval()
    print("  ✅ 模型设置为评估模式")
    
except RuntimeError as e:
    print(f"  ❌ 运行时错误: {e}")
    print("\n建议：")
    print("  1. 检查 PyTorch 版本与 CUDA 版本是否匹配")
    print("  2. 检查显存是否充足（需要 >5GB）")
    print("  3. 尝试重启 Python 进程")
    sys.exit(1)
except Exception as e:
    print(f"  ❌ 其他错误: {e}")
    sys.exit(1)

# 步骤 5: 测试推理（使用随机数据）
print("\n[5/5] 测试推理")
import numpy as np

try:
    # 创建随机视频数据（32 帧，224×224，RGB）
    dummy_frames = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
    
    # 转为 tensor 并归一化
    t = torch.from_numpy(dummy_frames).float() / 255.0
    t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, 32, 224, 224)
    
    print(f"  输入形状: {t.shape}")
    print(f"  输入设备: {t.device}")
    
    # 构造 DOVER 需要的输入格式
    views = {
        "technical": t,
        "aesthetic": t
    }
    
    # 推理（关键：验证 H100 计算正确性）
    with torch.no_grad():
        results = model(views)
    
    # 计算融合分数
    score = sum(r.mean().item() for r in results) / len(results)
    
    print(f"  ✅ 推理成功")
    print(f"  质量分数: {score:.4f}")
    print(f"  分数范围检查: {'✅ 合理' if 0 <= score <= 1 else '❌ 异常'}")
    
except RuntimeError as e:
    print(f"  ❌ 推理失败: {e}")
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("🎉 测试完成：DOVER 在 H100 GPU 上运行成功！")
print("=" * 60)
print("\n下一步：")
print("  1. 使用真实视频测试：python evaluate_one_video.py -v <video_path> -f")
print("  2. 批量处理测试：python evaluate_a_set_of_videos.py -in <video_dir> -out results.csv")
print("  3. 集成到 Stage 3 管线")
EOF

chmod +x /tmp/test_dover_cmcc.py
```

**4.2 运行测试**

```bash
cd /root/work/david_work/models/DOVER
conda activate sana_wm_qc_env

python /tmp/test_dover_cmcc.py
```

**期望输出（关键检查点）：**
```
============================================================
DOVER H100 兼容性测试 - CMCC 环境
============================================================

[1/5] 环境检查
  PyTorch 版本: 2.x.x
  CUDA 版本: 13.0
  CUDA 可用: True
  GPU 名称: NVIDIA H100 80GB
  GPU 计算能力: (9, 0)
  显存总量: 80.00 GB

[2/5] 加载 DOVER 代码
  ✅ DOVER 模块加载成功

[3/5] 加载配置和权重
  配置文件: /root/work/david_work/models/DOVER/dover.yml
  权重文件: /root/work/david_work/models/DOVER/pretrained_weights/DOVER.pth (228.6 MB)
  ✅ 配置加载成功

[4/5] 初始化模型并移到 GPU（关键测试）
  ✅ 模型对象创建成功
  ✅ 权重加载成功
  ✅ 模型成功移到 GPU（H100 兼容性验证通过）  <-- 关键！
  ✅ 模型设置为评估模式

[5/5] 测试推理
  输入形状: torch.Size([1, 3, 32, 224, 224])
  输入设备: cuda:0
  ✅ 推理成功
  质量分数: 0.xxxx
  分数范围检查: ✅ 合理

============================================================
🎉 测试完成：DOVER 在 H100 GPU 上运行成功！
============================================================
```

**✅ 成功标志：**
- [4/5] 所有 ✅ 都出现
- [5/5] 推理成功且分数在 [0, 1] 范围内
- 没有 `RuntimeError` 或 `CUDA error`

**❌ 如果失败，记录错误信息并跳转到方案 B**

---

### 步骤 5：真实视频测试（10 分钟）

**如果步骤 4 成功，继续测试真实视频：**

```bash
cd /root/work/david_work/models/DOVER

# 测试 demo 视频（如果存在）
if [ -f "./demo/17734.mp4" ]; then
    echo "测试 demo 视频..."
    python evaluate_one_video.py -v ./demo/17734.mp4 -f
fi

# 测试 CMCC 实际数据（选一个小视频）
TEST_VIDEO="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/wds-DL3DV-ALL-2K/w000/sample_video.mp4"

if [ -f "$TEST_VIDEO" ]; then
    echo "测试 CMCC 实际视频..."
    python evaluate_one_video.py -v "$TEST_VIDEO" -f
else
    echo "实际视频不存在，跳过"
fi
```

**期望输出：**
```
Normalized fused overall score (scale in [0,1]): 0.xxxx
```

**记录性能数据：**
- 视频长度：XX 秒
- 推理时间：XX 秒
- 质量分数：0.xxxx

---

## 🔄 方案 B：传输完整 dover_h100_test 环境（备用）

**仅在方案 A 失败时执行**

### 步骤 1：本地打包环境（需要你在 AFS 机器操作）

```bash
# 1.1 激活环境
conda activate dover_h100_test

# 1.2 打包环境
cd /mnt/afs/davidwang/miniconda3/envs/
conda pack -n dover_h100_test -o /tmp/dover_h100_test.tar.gz

# 1.3 验证打包
ls -lh /tmp/dover_h100_test.tar.gz
# 期望：约 1-2 GB

# 1.4 打包 DOVER 代码和权重（如果还没打包）
tar -czf /tmp/dover_package.tar.gz -C /mnt/afs/davidwang/workspace/sana_wm_pipeline/models/ DOVER/

# 1.5 生成 MD5
cd /tmp
md5sum dover_h100_test.tar.gz dover_package.tar.gz > dover_transfer.md5

# 1.6 传输到 CMCC
# 目标路径：/root/work/filestorage/shangaoooooo/davidwang/
```

### 步骤 2：CMCC 解压并激活

```bash
# 2.1 验证传输完整性
cd /root/work/filestorage/shangaoooooo/davidwang/
md5sum -c dover_transfer.md5

# 2.2 创建环境目录
mkdir -p /root/work/david_work/conda_envs/

# 2.3 解压环境
cd /root/work/david_work/conda_envs/
tar -xzf /root/work/filestorage/shangaoooooo/davidwang/dover_h100_test.tar.gz
mv dover_h100_test dover_h100_test_env

# 2.4 激活脚本修复路径
cd dover_h100_test_env
./bin/conda-unpack

# 2.5 激活环境
source /root/work/david_work/conda_envs/dover_h100_test_env/bin/activate

# 2.6 验证
python -c "import torch; print(torch.cuda.is_available())"

# 2.7 解压 DOVER（如果方案 A 没做）
cd /root/work/david_work/models/
tar -xzf /root/work/filestorage/shangaoooooo/davidwang/dover_package.tar.gz

# 2.8 运行测试（同方案 A 步骤 4）
python /tmp/test_dover_cmcc.py
```

---

## 📊 预期性能基准

| 指标 | 预期值 | 备注 |
|------|--------|------|
| 模型加载时间 | 5-10 秒 | 首次加载 |
| 单视频推理（10秒） | 0.5-1 秒 | H100 GPU 模式 |
| 单视频推理（30秒） | 1-2 秒 | H100 GPU 模式 |
| 单视频推理（60秒） | 2-4 秒 | H100 GPU 模式 |
| 显存占用 | 5-8 GB | 推理时 |
| 质量分数范围 | 0-1 | 浮点数 |

---

## ⚠️ 常见问题排查

### 问题 1：CUDA available: False

**症状：**
```python
torch.cuda.is_available()  # False
```

**排查步骤：**
```bash
# 1. 检查 nvidia-smi
nvidia-smi

# 2. 检查环境变量
echo $CUDA_VISIBLE_DEVICES
# 如果是空字符串（不是未设置），执行：
unset CUDA_VISIBLE_DEVICES

# 3. 检查 PyTorch 版本
python -c "import torch; print(torch.version.cuda)"
ls -la /usr/local/ | grep cuda

# 4. 如果版本不匹配，重装 PyTorch
pip uninstall torch torchvision
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 问题 2：ModuleNotFoundError: No module named 'dover'

**原因：** Python 路径未设置

**解决：**
```bash
cd /root/work/david_work/models/DOVER
export PYTHONPATH=/root/work/david_work/models/DOVER:$PYTHONPATH
python /tmp/test_dover_cmcc.py
```

### 问题 3：RuntimeError: CUDA out of memory

**原因：** 显存不足或有其他进程占用

**排查：**
```bash
# 查看显存占用
nvidia-smi

# 查看占用显存的进程
fuser -v /dev/nvidia*

# 如果有其他进程，考虑：
# 1. 停止其他进程
# 2. 使用其他 GPU
export CUDA_VISIBLE_DEVICES=1
```

### 问题 4：模型加载后推理报错

**症状：**
```
RuntimeError: CUDA error: xxx
```

**排查：**
```bash
# 1. 清理显存
python -c "import torch; torch.cuda.empty_cache()"

# 2. 重启 Python 进程

# 3. 降低 batch size 或视频分辨率

# 4. 检查驱动版本
nvidia-smi | grep "Driver Version"
```

---

## ✅ 成功检查清单

完成以下所有项目即为成功：

- [ ] PyTorch CUDA 可用（`torch.cuda.is_available() == True`）
- [ ] GPU 识别为 H100（`torch.cuda.get_device_name(0)` 包含 "H100"）
- [ ] DOVER 代码解压到 `/root/work/david_work/models/DOVER/`
- [ ] DOVER 权重存在（`DOVER/pretrained_weights/DOVER.pth` 约 229MB）
- [ ] 核心依赖安装成功（decord, opencv-python, timm, einops）
- [ ] 测试脚本运行成功（5 个步骤全部 ✅）
- [ ] 真实视频推理成功（输出 0-1 之间的质量分数）
- [ ] 性能符合预期（10秒视频 <2秒推理时间）

---

## 📝 执行日志模板

**✅ CMCC 实际执行已完成，详见：`DOVER_H100_部署方案_CMCC实际执行记录.md`**

---

**如需在其他机器重新部署，请记录以下信息：**

```
=== DOVER H100 部署日志 ===
执行日期：2026-08-XX
执行人员：XXX
CMCC 节点：nodeXXX

【环境配置】
export TORCH_HOME=/root/work/david_work/cache/torch  # ⚠️ 必需

【方案选择】
☑ 方案 A：sana_wm_qc_env 环境
☐ 方案 B：dover_h100_test 环境

【方案 A 执行记录】
步骤 1 - 环境验证：
  PyTorch 版本：_______________
  CUDA 可用：☑ 是 ☐ 否
  GPU 名称：_______________
  
步骤 2 - DOVER 路径：
  实际路径：_______________
  权重文件大小：_______________ MB
  
步骤 3 - 安装依赖：
  decord：☑ 成功 ☐ 失败
  timm：☑ 成功 ☐ 失败
  einops：☑ 成功 ☐ 失败
  scikit-video：☑ 成功 ☐ 失败  # ⚠️ 必需
  
步骤 4 - 测试脚本：
  模型加载：☑ 成功 ☐ 失败
  移到 GPU：☑ 成功 ☐ 失败
  推理测试：☑ 成功 ☐ 失败
  推理耗时：_______________ ms
  
步骤 5 - 真实视频：
  视频路径：_______________
  归一化分数：_______________

【遇到的问题】
问题 1：_______________
解决方案：_______________

【最终状态】
☑ 成功：DOVER 在 H100 GPU 上运行正常
☐ 失败：需要进一步排查

【性能数据】
随机数据推理：_______________ ms
真实视频推理：_______________ ms
显存占用：_______________ GB
```

---

## 📞 支持资源

- **本地验证文档**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/H100_INSTALLATION_GUIDE.md`
- **测试脚本**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/test_dover_h100.py`
- **DOVER 官方仓库**：https://github.com/VQAssessment/DOVER
- **技术负责人**：David Wang

---

**下一步**：请按照"方案 A"开始执行，从步骤 1 开始逐步验证。

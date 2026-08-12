# DOVER 测试进度记录

## 当前状态：等待 GPU 机器验证

**日期**：2026-07-02  
**环境**：sana_wm-cmcc conda 环境  
**位置**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/`

---

## 已完成的工作

### 1. 调研总结（已完成 ✅）

**DOVER 核心功能**：
- 无参考视频质量评估（No-Reference VQA）
- 双视角分解：美学质量 + 技术质量
- SOTA 性能，专为 UGC 设计
- DOVER-Mobile 版本：参数量减少 5.7 倍，速度快 2.5 倍

**H100 Coredump 问题调研结果**：
- ❌ 官方仓库无任何相关记录（检查了 42 个 issues + git 历史）
- 可能原因：PyTorch 版本不兼容、内存管理问题、decord 库问题

**替代方案推荐**：
1. **Q-Align** ⭐（推荐）- ICML 2024，统一 IQA+VQA，易用，H100 兼容性好
2. **DOVER 修复版** - 最高精度但需调试
3. **VMAF + FAST-VQA** - 工业级稳定

### 2. 环境配置（已完成 ✅）

**已安装依赖**：
```bash
conda activate sana_wm-cmcc
# 已安装：torch 2.12.0, decord 0.6.0, thop 0.0.31, wandb, onnx, scikit-video
```

**模型权重**：
- ✅ DOVER.pth (229MB) 已下载到 `pretrained_weights/`
- ✅ ConvNext backbone 权重已自动下载到 `/root/.cache/torch/hub/`

**测试视频**：
- ✅ 7 个示例视频已存在于 `demo/` 目录

### 3. 问题诊断（已完成 ✅）

**当前机器状态**：
- ❌ **CPU 机器**（PyTorch 检测不到 GPU）
- 系统有 CUDA 13.0，但 PyTorch 2.12+cu130 无法访问
- `CUDA_VISIBLE_DEVICES` 为空

**测试结果**：
- GPU 模式：`RuntimeError: Found no NVIDIA driver`
- CPU 模式：进程被 Killed（内存不足或超时）

---

## 下一步计划（GPU 机器上执行）

### 测试脚本 A：基础推理验证
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm-cmcc

# 确认 GPU 可见
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 测试单个视频（融合分数模式）
python evaluate_one_video.py -v ./demo/17734.mp4 -f
```

**期望输出**：
```
The quality score of video [./demo/17734.mp4] is [0.XXX].
```

### 测试脚本 B：验证是否 Coredump
```bash
# 如果基础测试通过，用更长视频或批量测试压力
python evaluate_a_set_of_videos.py -in ./demo/ -out ./test_output.csv
```

### 测试脚本 C：如果 Coredump，尝试修复
```bash
# 方案 1：使用 DOVER-Mobile（内存占用小）
python evaluate_one_video.py -v ./demo/17734.mp4 -f -o dover-mobile.yml

# 方案 2：禁用 torch.compile（修改代码）
# 编辑 default_infer.py 第 65 行，注释掉：
# try:
#     model = torch.compile(model)
# except:
#     pass

# 方案 3：降级 PyTorch
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 关键文件位置

| 文件 | 路径 |
|------|------|
| DOVER 代码 | `/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/` |
| 模型权重 | `pretrained_weights/DOVER.pth` |
| 测试视频 | `demo/*.mp4` |
| Conda 环境 | `sana_wm-cmcc` |
| 进度记录 | 本文件 |

---

## 问题排查 Checklist

GPU 机器上执行时按顺序检查：

- [ ] `nvidia-smi` 能看到 H100
- [ ] PyTorch `torch.cuda.is_available()` 返回 `True`
- [ ] 单视频推理成功（无 coredump）
- [ ] 批量推理成功
- [ ] 如果失败，记录完整错误栈和 `dmesg | tail -50` 输出

---

## 备注

- improved-aesthetic-predictor 不适合视频（仅图像美学评分）
- Q-Align 是更现代的备选方案，建议并行测试
- 如需技术支持，官方仓库 issues 活跃度一般，建议先自行调试

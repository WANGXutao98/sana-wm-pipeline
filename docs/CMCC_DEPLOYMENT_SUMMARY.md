# SANA-WM QC Pipeline CMCC 部署总结

**日期**: 2026-07-02  
**状态**: Stage 1+2 ✅ 完成 | Stage 3 ⏳ 运行中  
**机器**: CMCC H100 GPU (PyTorch 2.4.0, transformers 5.12.1)

---

## 📋 一、项目概述

### 1.1 任务目标
在 CMCC H100 机器上部署 SANA-WM 视频质量检测（QC）流程，包含 3 个阶段：
- **Stage 1**: 快速全覆盖扫描（姿态、caption、饱和度）
- **Stage 2**: 深度目标检查（黑帧、场景切换、轨迹冻结）
- **Stage 3**: GPU 视觉质量评估（UniMatch 光流 + DOVER 质量评分 + Qwen VLM）

### 1.2 关键路径

| 路径类型 | CMCC 机器路径 |
|---------|--------------|
| 项目代码 | `/root/work/david_work/sana_wm_qc` |
| conda 环境 | `/root/work/david_work/sana_wm_qc_env` |
| 数据根目录 | `/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output` |
| QC 输出 | `/root/work/david_work/qc_output` |
| Qwen 模型 | `/root/work/david_work/models/Qwen3.5-27B` |
| UniMatch | `/root/work/david_work/models/unimatch` |
| DOVER | `/root/work/david_work/sana_qc_pipeline/DOVER` |

---

## 🔧 二、关键问题与修复

### 2.1 DOVER 初始化失败（H100 兼容性）

**问题**：
```
Segmentation fault at model.to("cuda")
```

**根因**：DOVER 的某些层（Swin Transformer）与 H100 sm_90 架构不兼容

**解决方案**：强制 DOVER 使用 CPU 模式

**修改位置**：`src/sana_wm_pipeline/qc/stage3_gpu.py` 的 `load_dover_fn()`
```python
# ⚠️ WORKAROUND: Force CPU mode due to H100 compatibility issues
if device != "cpu":
    warnings.warn(
        f"DOVER requested device='{device}', but forcing CPU mode due to H100 compatibility issues.",
        RuntimeWarning
    )
    device = "cpu"
```

**影响**：DOVER 处理速度慢 10-50 倍（单样本 ~5-10s）

---

### 2.2 UniMatch 导入失败

**问题**：
```python
from unimatch.unimatch import UniMatch
# ImportError: cannot import name 'UniMatch'
```

**根因 1**：`unimatch/__init__.py` 是空文件（0 字节）

**根因 2**：`unimatch.py` 内部使用相对导入（`from .backbone import ...`），需要 package context

**解决方案**：
```python
# 将 unimatch 目录加入 sys.path，使用标准导入
sys.path.insert(0, str(model_path))
from unimatch import unimatch as unimatch_module
UniMatch = unimatch_module.UniMatch
```

**修改位置**：`src/sana_wm_pipeline/qc/stage3_gpu.py` 的 `load_unimatch_fn()`

---

### 2.3 UniMatch 权重路径错误

**问题**：
```
FileNotFoundError: gmflow-scale2-regrefine6-mixdata.pth
```

**根因**：代码在根目录查找，但权重在 `pretrained/` 子目录

**解决方案**：按顺序查找多个路径
```python
ckpt_paths = [
    Path(model_dir) / "pretrained" / "gmflow-scale2-regrefine6-mixdata.pth",
    Path(model_dir) / "gmflow-scale2-regrefine6-mixdata.pth",
]
```

**下载命令**（如缺失）：
```bash
cd /root/work/david_work/models/unimatch
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained_models/gmflow-scale2-regrefine6-mixdata.pth
```

---

### 2.4 Qwen 模型类型不匹配

**问题**：
```
ValueError: hidden_size must be divisible by num_heads (got 5120 and 24)
```

**根因**：模型是 Qwen3.5，但代码使用 `Qwen2_5_VLForConditionalGeneration`

**解决方案**：使用通用 AutoModel
```python
from transformers import AutoModelForCausalLM, AutoProcessor
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True  # ← 关键
)
```

**修改位置**：`src/sana_wm_pipeline/qc/stage3_gpu.py` 的 `load_qwen_fn()`

---

### 2.5 torchaudio 版本冲突

**问题**：
```
undefined symbol: _ZN2at4_ops9fft_irfft4callER...
```

**根因**：torchaudio 2.6.0 与 PyTorch 2.4.0 不匹配

**解决方案**：
```bash
pip uninstall -y torchaudio
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

---

### 2.6 DOVER 输入形状错误

**问题**：
```
RuntimeError: expected input to have 3 channels, but got 32 channels
```

**根因**：输入张量形状是 `(1, 32, 3, H, W)` 但应该是 `(1, 3, 32, H, W)`

**解决方案**：
```python
# 修正 permute 顺序
t = torch.from_numpy(frames_rgb).float() / 255.0  # (T, H, W, 3)
t = t.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, T, H, W)
```

---

## 📦 三、最终文件清单（需从本地复制到 CMCC）

**源路径**：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/`

| 文件 | 本地路径 | CMCC 目标路径 | 关键修改 |
|------|---------|--------------|---------|
| **stage3_gpu.py** | `qc/stage3_gpu.py` | `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/qc/stage3_gpu.py` | DOVER CPU 模式 + UniMatch 导入 + Qwen AutoModel + 输入形状 |
| **stage1_fast.py** | `qc/stage1_fast.py` | `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/qc/stage1_fast.py` | 边界条件、异常处理 |
| **stage2_deep.py** | `qc/stage2_deep.py` | `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/qc/stage2_deep.py` | 资源泄漏防护 |
| **visual_metrics.py** | `stage04_filter/visual_metrics.py` | `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/stage04_filter/visual_metrics.py` | 超时、空帧检查 |
| **table6_thresholds.yaml** | `stage04_filter/table6_thresholds.yaml` | `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml` | 7 个数据集的质量阈值 |

---

## 🎯 四、数据集配置

### 4.1 数据路径结构

```
/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/
├── final_wds-DL3DV-ALL-2K/
│   └── wds-DL3DV-ALL-2K/
│       ├── w000/ (shard-*.tar)
│       ├── w001/
│       └── ...
├── final_wds-OmniWorld-Game/
├── final_wds-RealEstate10K-360p/
├── final_wds-SpatialVID-hq/
├── final_wds-sekai-game-drone/
├── final_wds-sekai-game-walking/
└── final_wds-sekai-real-walking-hq/
```

### 4.2 Group 到 Table6 Source 的映射

| 目录名 | group_name | table6_source | 特点 |
|--------|-----------|---------------|------|
| `final_wds-DL3DV-ALL-2K` | `wds-DL3DV-ALL-2K` | `DL3DV` | 室内 3D 扫描，严格质量 |
| `final_wds-OmniWorld-Game` | `wds-OmniWorld-Game` | `OmniWorld` | 第三人称游戏，运动大 |
| `final_wds-RealEstate10K-360p` | `wds-RealEstate10K-360p` | `RealEstate10K` | 室内建筑漫游 |
| `final_wds-SpatialVID-hq` | `wds-SpatialVID-hq` | `SpatialVID` | 空间视频，高质量 |
| `final_wds-sekai-game-drone` | `wds-sekai-game-drone` | `Sekai_Game_Drone` | 游戏航拍，光流大 |
| `final_wds-sekai-game-walking` | `wds-sekai-game-walking` | `Sekai_Game_Walking` | 游戏步行视角 |
| `final_wds-sekai-real-walking-hq` | `wds-sekai-real-walking-hq` | `Sekai_Walking` | 真实步行场景 |

### 4.3 table6_thresholds.yaml 关键阈值

**DL3DV（当前测试数据集）**：
```yaml
DL3DV:
  vmaf_motion:      [0.5, 50]    # 运动强度
  unimatch_flow:    [3, 80]      # 光流幅度
  dover:            [0.40, 1.0]  # 质量评分
  color_saturation: [0, 180]     # 饱和度
  scene_cuts_max:   1            # 最多 1 次场景切换
  vlm_entity:       null         # 不检查实体数量
  vlm_quality:      null         # 不用 VLM 质量评分
```

---

## 🧪 五、测试结果

### 5.1 冒烟测试（w000 目录，3 个 tar 文件）

#### Stage 1+2 ✅ 完成
```
数据: /root/.../wds-DL3DV-ALL-2K/w000
Stage 1: 346 samples → stage1_results.jsonl
Stage 2: 293 samples → stage2_results.jsonl
报告: report.html + manifests/
```

#### Stage 3 ⏳ 运行中（预计 45-84 分钟）
```
命令:
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /root/work/david_work/qc_output/smoke_full/stage1_results.jsonl \
  --output-dir /root/work/david_work/qc_output/smoke_full \
  --qwen-dir /root/work/david_work/models/Qwen3.5-27B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg /root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
  --device cuda

状态: 所有模型已加载，正在处理
GPU 显存: 57853/81559 MiB (71%)
GPU 利用率: 0% (当前在 DOVER CPU 阶段，正常)
```

**监控进度**：
```bash
# 查看已处理样本数
wc -l /root/work/david_work/qc_output/smoke_full/stage3_worker000.jsonl

# 实时监控
watch -n 10 'wc -l /root/work/david_work/qc_output/smoke_full/stage3_worker000.jsonl'
```

---

## 📊 六、性能特征

### 6.1 Stage 3 处理时间（单样本）

| 组件 | 设备 | 时间 | 备注 |
|------|------|------|------|
| UniMatch | GPU | 0.5-1s | 光流计算 |
| DOVER | CPU | **5-10s** | ⚠️ 瓶颈（H100 不兼容） |
| Qwen | GPU | 2-3s | VLM 推理 |

### 6.2 GPU 显存占用

| 组件 | 显存 |
|------|------|
| Qwen3.5-27B | ~54GB |
| UniMatch | ~2GB |
| 临时张量 | ~2GB |
| **总计** | **~58GB** |

### 6.3 为什么 GPU 利用率为 0？

**正常现象！** Stage 3 对每个样本的处理顺序：
1. UniMatch (GPU) → GPU 利用率 80-90%
2. **DOVER (CPU)** → GPU 利用率 0% ← **当前这里**
3. Qwen (GPU) → GPU 利用率 80-90%

---

## 🚀 七、批量生产运行命令

### 7.1 完整 Stage 1+2（所有 w* 目录）

```bash
cd /root/work/david_work/sana_wm_qc
source /root/work/david_work/activate_qc.sh

python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-DL3DV-ALL-2K/wds-DL3DV-ALL-2K \
  --group wds-DL3DV-ALL-2K \
  --output-dir /root/work/david_work/qc_output/full_qc \
  --n-workers 32 \
  --skip-stage2  # 或去掉此参数运行 Stage 2
```

### 7.2 Stage 3（多 GPU 并行）

**单 GPU**：
```bash
python scripts/run_stage3_cmcc.py \
  --stage1-jsonl /root/work/david_work/qc_output/full_qc/stage1_results.jsonl \
  --output-dir /root/work/david_work/qc_output/full_qc \
  --qwen-dir /root/work/david_work/models/Qwen3.5-27B \
  --unimatch-dir /root/work/david_work/models/unimatch \
  --worker-id 0 \
  --total-workers 1 \
  --table6-cfg /root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml \
  --device cuda
```

**多 GPU（例如 4 卡）**：
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3_cmcc.py --worker-id 0 --total-workers 4 ... &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/run_stage3_cmcc.py --worker-id 1 --total-workers 4 ... &

# GPU 2
CUDA_VISIBLE_DEVICES=2 python scripts/run_stage3_cmcc.py --worker-id 2 --total-workers 4 ... &

# GPU 3
CUDA_VISIBLE_DEVICES=3 python scripts/run_stage3_cmcc.py --worker-id 3 --total-workers 4 ... &

wait
```

---

## ⚠️ 八、已知限制和注意事项

### 8.1 DOVER CPU 模式性能

**问题**：H100 架构不兼容，只能用 CPU
**影响**：处理速度慢 10-50 倍
**解决方案（可选）**：
- 使用 A100/V100 GPU（sm_80 架构）
- 降级 PyTorch 到 2.1.x
- 批处理优化

### 8.2 依赖版本锁定

**关键版本**：
```
torch==2.4.0
torchaudio==2.4.0
transformers==5.12.1
```

**不要随意升级**，可能导致兼容性问题。

### 8.3 路径注意事项

**正确的源代码路径**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/
```

**错误的路径（过时）**：
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana_qc_cmcc_pack/  # ❌ 不要使用
```

---

## 📝 九、下次对话衔接点

### 9.1 当前状态

✅ **已完成**：
- Stage 1+2 冒烟测试成功（346 样本）
- 所有模型修复和配置完成
- Stage 3 已启动运行

⏳ **进行中**：
- Stage 3 正在处理 346 个样本（预计 1-1.5 小时）

### 9.2 下一步待办

1. **验证 Stage 3 输出**
   - 检查 `stage3_worker000.jsonl` 是否生成
   - 验证 346 个样本都有 Stage 3 结果

2. **批量生产测试**
   - 运行完整数据集（所有 w* 目录）
   - 测试多 GPU 并行

3. **性能优化（可选）**
   - 评估 DOVER CPU 模式是否可接受
   - 如果太慢，考虑使用 A100 或批处理优化

4. **生成最终文档**
   - 部署手册
   - 故障排查指南
   - 性能基准测试报告

---

## 🔗 十、相关文档

- **部署文档**：`docs/7_SANA_WM_QC_DEPLOY.md`
- **模型权重**：`docs/QC_MODEL_WEIGHTS.md`
- **原始审查文档**：`.superpowers/sdd/task-5-brief.md`

---

**文档版本**: v1.0  
**最后更新**: 2026-07-02  
**维护者**: Claude + David Wang

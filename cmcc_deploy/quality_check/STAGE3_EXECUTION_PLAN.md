# Stage 3 数据质检管线执行计划（CMCC 机器）

> **文档状态**：当前有效  
> **创建日期**：2026-08-03  
> **目标**：在 CMCC 机器上成功跑通 Stage 3 数据质检管线（UniMatch + DOVER + Qwen Caption 改写）

---

## 一、执行计划总览

### 1.1 当前项目状态

**已完成**：
- ✅ 批量生产完成（7 个 group，~20 万样本）
- ✅ QC Stage 1+2 执行完毕（Pass 率 77.1%）
- ✅ 人工审查完成（6,702 样本标注）
- ✅ dover_h100_test 环境已克隆到持久化路径

**待完成**：
- ⏳ Stage 3 环境验证与依赖补全
- ⏳ Stage 3 单样本端到端测试
- ⏳ Stage 3 全量执行（48 GPU）
- ⏳ QC 结果汇总与最终数据提取

### 1.2 Stage 3 核心模块

| 模块 | 工具 | 功能 | 单样本耗时 | 状态 |
|------|------|------|-----------|------|
| 光流检测 | UniMatch | 检测运动连续性 | ~3s | ⏳ 待安装 |
| 质量评分 | DOVER | 检测模糊/抖动 | ~5s (CPU) | ⚠️ GPU 不兼容 |
| Caption改写 | Qwen VLM | 去除相机动作词 | ~2s (27B) | ⏳ 待选型 |

**预估总耗时**：10s/样本 × 18万样本 / 48 GPU = **10.4 小时**（实际预留 12-14h）

---

## 二、环境校验与补全（第一步）

### 2.1 dover_h100_test 环境验证

**执行机器**：CMCC 任意节点  
**环境路径**：`/mnt/afs/davidwang/miniconda3/envs/dover_h100_test`

#### 验证脚本

```bash
# 1. 激活环境
conda activate /mnt/afs/davidwang/miniconda3/envs/dover_h100_test

# 2. 验证 Python + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. 验证 DOVER 安装
python -c "import dover; print('DOVER installed')" 2>&1 | tee /tmp/dover_check.log

# 4. 验证 GPU 显存
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv

# 5. 记录结果
echo "===== 环境验证结果 =====" > /tmp/stage3_env_check.txt
conda list | grep -E "(torch|dover|numpy)" >> /tmp/stage3_env_check.txt
```

#### 预期输出

```
PyTorch: 2.x.x, CUDA: True
DOVER installed
GPU 0, 81920 MiB, >70000 MiB
```

#### 如果失败

- **CUDA: False** → 检查 `LD_LIBRARY_PATH` 是否包含 CUDA 路径
- **DOVER 未安装** → 执行 `pip install dover-lap`
- **显存不足** → 检查是否有其他进程占用 GPU

---

### 2.2 依赖补全清单

#### 补全脚本

```bash
# 激活环境
conda activate /mnt/afs/davidwang/miniconda3/envs/dover_h100_test

# 设置工作目录
cd /root/work/david_work/sana_wm_pipeline

# 1. 安装 UniMatch（如未安装）
if [ ! -d "third_party/unimatch" ]; then
    cd third_party
    git clone https://github.com/autonomousvision/unimatch.git
    cd unimatch
    pip install -e . --no-deps
    cd ../..
fi

# 2. 下载 UniMatch 权重（~200MB）
mkdir -p /root/work/david_work/models/unimatch
# 从 AFS 传输或 ModelScope 下载
# scp user@afs:/mnt/afs/davidwang/models/unimatch/* /root/work/david_work/models/unimatch/

# 3. 确认 DOVER 安装
pip show dover-lap || pip install dover-lap

# 4. 补全 Stage 2 遗留依赖
pip install av scenedetect

# 5. 验证所有依赖
python -c "import unimatch; import dover; import av; import scenedetect; print('All dependencies OK')"
```

#### 关键检查点

- ☐ UniMatch 代码已克隆
- ☐ UniMatch 权重已下载（~200MB）
- ☐ DOVER 已安装
- ☐ PyAV 已安装
- ☐ scenedetect 已安装

---

## 三、DOVER H100 兼容性测试（第二步）

### 3.1 GPU 模式测试

```python
# 文件：/tmp/test_dover_gpu.py
import torch
from dover import DOVERModel

print("测试 DOVER GPU 模式...")

try:
    # 强制 GPU
    model = DOVERModel(device='cuda:0')
    print("✅ DOVER GPU 模式加载成功")
    
    # 测试推理（需要一个测试视频）
    # score = model.predict('/path/to/test_video.mp4')
    # print(f'DOVER score: {score}')
    
except Exception as e:
    print(f"❌ DOVER GPU 模式失败: {e}")
    print("将回退到 CPU 模式")
```

```bash
python /tmp/test_dover_gpu.py
```

### 3.2 CPU 模式回退

如果 GPU 模式失败，在所有 Stage 3 脚本中添加：

```bash
export DOVER_DEVICE=cpu
```

**性能影响**：
- GPU 模式：~0.5s/样本（理论值）
- CPU 模式：~5s/样本（实测）
- 全量耗时增加：约 10 倍

---

## 四、Qwen 模型选型与测试（第三步）

### 4.1 当前模型问题

- **Qwen2.5-VL-27B**：~55GB 显存，~2s/样本
- **目标**：<20GB 显存，<1s/样本

### 4.2 候选模型（按优先级）

| 模型 | 显存 | 推理速度（预估） | 下载大小 |
|------|------|----------------|---------|
| Qwen2-14B-VL-INT4 | ~15GB | ~0.8s | ~8GB |
| Qwen2-7B-VL | ~10GB | ~0.5s | ~14GB |
| InternVL2-8B | ~12GB | ~0.7s | ~16GB |
| Llama-3.2-11B-Vision | ~14GB | ~0.9s | ~22GB |

### 4.3 选型测试流程（在 AFS 执行）

```bash
# 1. 下载候选模型
cd /mnt/afs/davidwang/models
modelscope download --model Qwen/Qwen2-7B-VL --local_dir Qwen2-7B-VL

# 2. 测试推理速度
python /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/benchmark_qwen.py \
  --model_path /mnt/afs/davidwang/models/Qwen2-7B-VL \
  --test_samples 10 \
  --output /tmp/qwen_benchmark.json

# 3. 评估 Caption 质量（人工评审 10 个样本）
```

### 4.4 模型传输到 CMCC

```bash
# 选定后打包
tar -czf qwen2-7b-vl.tar.gz /mnt/afs/davidwang/models/Qwen2-7B-VL/

# 传输到 CMCC filestorage
scp qwen2-7b-vl.tar.gz cmcc-host:/root/work/filestorage/shangaoooooo/davidwang/

# CMCC 解压
cd /root/work/david_work/models
tar -xzf /root/work/filestorage/shangaoooooo/davidwang/qwen2-7b-vl.tar.gz
```

---

## 五、Stage 3 单样本端到端测试（第四步）

### 5.1 测试脚本

```bash
# 文件：/root/work/david_work/sana_wm_pipeline/scripts/test_stage3_single.sh

# 激活环境
conda activate /mnt/afs/davidwang/miniconda3/envs/dover_h100_test

# 设置环境变量
export DOVER_DEVICE=cpu  # 如 GPU 失败
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 选择一个测试样本
TEST_SAMPLE="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/wds-DL3DV-ALL-2K/w000/shard-000000.tar"

# 运行 Stage 3
python -m sana_wm_pipeline.qc.stage3_gpu \
  --input_tar $TEST_SAMPLE \
  --output_dir /tmp/stage3_test_output \
  --model_unimatch /root/work/david_work/models/unimatch \
  --model_qwen /root/work/david_work/models/Qwen2-7B-VL \
  --gpu_id 0

# 检查输出
cat /tmp/stage3_test_output/stage3_results.jsonl
```

### 5.2 验证检查点

- ☐ UniMatch 光流计算成功（mean_flow 值合理）
- ☐ DOVER 质量评分成功（score ∈ [0, 1]）
- ☐ Qwen Caption 改写成功（原始 vs 改写版本）
- ☐ 输出 jsonl 格式正确
- ☐ 单样本耗时 <15s（含 I/O）

---

## 六、Stage 3 全量执行（第五步）

### 6.1 执行脚本

```bash
# 文件：/root/work/david_work/sana_wm_pipeline/scripts/run_stage3_cmcc.py

import os
import sys
from pathlib import Path

# 配置
STAGE2_RESULTS = "/root/work/david_work/qc_output/*/stage2_results.jsonl"
VIDEO_ROOT = "/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/"
OUTPUT_DIR = "/root/work/david_work/qc_output/stage3/"
NUM_GPUS = 48

# 模型路径
MODEL_UNIMATCH = "/root/work/david_work/models/unimatch"
MODEL_QWEN = "/root/work/david_work/models/Qwen2-7B-VL"

# 执行命令（伪代码，需完整实现）
# for gpu_id in range(NUM_GPUS):
#     启动 worker(gpu_id, samples_subset)
```

### 6.2 启动命令

```bash
# 在 CMCC master 节点执行
cd /root/work/david_work/sana_wm_pipeline

# 启动 48 GPU 并行任务
python scripts/run_stage3_cmcc.py \
  --stage2_results qc_output/*/stage2_results.jsonl \
  --video_root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
  --output_dir qc_output/stage3/ \
  --num_gpus 48 \
  --model_unimatch /root/work/david_work/models/unimatch \
  --model_qwen /root/work/david_work/models/Qwen2-7B-VL

# 预计耗时：10-14 小时
```

### 6.3 监控命令

```bash
# 查看进度
watch -n 10 'find qc_output/stage3/ -name "*.jsonl" | wc -l'

# 查看 GPU 利用率
watch -n 5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'

# 查看单个 worker 日志
tail -f qc_output/stage3/logs/worker_gpu00.log
```

---

## 七、QC 结果汇总与数据提取（第六步）

### 7.1 合并三阶段结果

```bash
cd /root/work/david_work/sana_wm_pipeline

python scripts/merge_qc_results.py \
  --stage1 qc_output/*/stage1_results.jsonl \
  --stage2 qc_output/*/stage2_results.jsonl \
  --stage3 qc_output/stage3/stage3_results.jsonl \
  --human qc_output/human_review_results.jsonl \
  --output qc_final_output/

# 输出：
#   pass_final.txt          # ~14 万条样本 ID
#   reject_final.txt        # ~4 万条样本 ID
#   qc_summary_report.html  # 可视化报告
```

### 7.2 提取训练数据

```bash
python scripts/extract_training_data.py \
  --pass_list qc_final_output/pass_final.txt \
  --source_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
  --output_dir /root/work/filestorage/shangaoooooo/davidwang/training_data_final/

# 验证完整性
python scripts/verify_training_data.py \
  --data_dir /root/work/filestorage/shangaoooooo/davidwang/training_data_final/ \
  --expected_count 141550
```

---

## 八、故障排查预案

### 8.1 高频问题

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| DOVER OOM | `CUDA out of memory` | `export DOVER_DEVICE=cpu` |
| UniMatch 导入失败 | `ModuleNotFoundError` | `pip install -e third_party/unimatch` |
| Qwen 加载失败 | `model type mismatch` | 用 `AutoModel.from_pretrained(trust_remote_code=True)` |
| 离线模式报错 | `HTTPSConnectionPool timeout` | 检查 `TRANSFORMERS_OFFLINE=1` |
| LD_LIBRARY_PATH 污染 | `undefined symbol` | prepend env lib：`export LD_LIBRARY_PATH=$ENV_DIR/lib:$LD_LIBRARY_PATH` |

### 8.2 紧急联系人

- **技术负责人**：David Wang
- **文档位置**：
  - `docs/03-QC_SYSTEM.md` — QC 系统详解
  - `docs/05-TROUBLESHOOTING.md` — 故障排查手册
  - `findings.md` — 技术发现汇总

---

## 九、执行检查清单

### 启动前检查（必须全部完成）

- [ ] dover_h100_test 环境已验证（Python/CUDA/DOVER）
- [ ] UniMatch 已安装（代码 + 权重）
- [ ] DOVER GPU 兼容性已测试（或确认使用 CPU 模式）
- [ ] Qwen 模型已选型并传输到 CMCC
- [ ] Stage 3 单样本测试已通过
- [ ] 离线环境变量已设置（`TRANSFORMERS_OFFLINE=1`）
- [ ] 数据备份已完成（stage1+stage2 结果）

### 执行中监控

- [ ] 每小时检查一次 GPU 利用率
- [ ] 每 2 小时检查一次产出样本数
- [ ] 每 4 小时检查一次日志错误数

### 完成后验证

- [ ] stage3_results.jsonl 样本数 = stage2 pass 样本数
- [ ] pass_final.txt 生成成功
- [ ] 训练数据提取完成
- [ ] MD5 清单已生成

---

## 十、时间线预估

| 阶段 | 预估耗时 | 备注 |
|------|---------|------|
| 环境验证 | 30 分钟 | dover_h100_test 校验 |
| 依赖补全 | 1 小时 | UniMatch + 其他依赖 |
| Qwen 选型 | 4-8 小时 | 在 AFS 测试候选模型 |
| 单样本测试 | 30 分钟 | 端到端验证 |
| 全量执行 | 10-14 小时 | 48 GPU 并行 |
| 结果汇总 | 1 小时 | 合并 + 提取 |
| **总计** | **17-25 小时** | 约 1-2 个工作日 |

---

**最后更新**：2026-08-03  
**下一步**：执行"二、环境校验与补全"

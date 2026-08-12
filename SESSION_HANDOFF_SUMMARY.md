# SANA-WM QC Pipeline 会话交接文档

**会话日期**: 2026-08-05  
**当前状态**: Stage 1+2 完成 ✅，Stage 3 准备中 🚧  
**文档用途**: 供新的 Claude 会话无缝接续工作

---

## 📍 当前进度概览

### 已完成阶段

| 阶段 | 状态 | 样本数 | 执行位置 | 结果位置 |
|------|------|--------|----------|----------|
| **Stage 1** | ✅ 完成 | 333,248 | CMCC 机器 | `/root/work/david_work/qc_output_new/wds-*/stage1_results.jsonl` |
| **Stage 2** | ✅ 完成 | 27,526 (8.3%) | CMCC 机器 | `/root/work/david_work/qc_output_new/wds-*/stage2_results.jsonl` |
| **Stage 3** | 🚧 准备中 | 282,222 | 待执行 | 待生成 |

### 数据流转路径

```
CMCC 机器执行 Stage 1+2
    ↓
手动传输结果到本机
    ↓
本机分析路径: /mnt/afs/davidwang/workspace/sana_wm_pipeline/stage2_result/qc_output_new/
    ↓
生成 Stage 3 输入 manifest (CMCC)
    ↓
Stage 3 执行 (CMCC, 16×H100)
```

---

## 🗂️ 项目结构与关键文件

### 本机 (分析环境)
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/
├── src/sana_wm_pipeline/qc/
│   ├── stage1_fast.py         # Stage 1 核心逻辑
│   ├── stage2_deep.py         # Stage 2 核心逻辑
│   ├── metrics.py             # Stage 1 指标计算
│   ├── group_config.py        # 各数据集配置
│   └── report.py              # 报告生成
├── scripts/
│   └── run_qc.py              # Stage 1+2 CLI 入口
├── stage2_result/qc_output_new/  # ← Stage 1+2 结果 (从 CMCC 传输)
│   ├── stage1_summary.json
│   ├── STAGE2_ANALYSIS_REPORT.md  # ← 本次分析报告
│   └── wds-*/
│       ├── stage1_results.jsonl
│       └── stage2_results.jsonl
└── SESSION_HANDOFF_SUMMARY.md  # ← 本文档
```

### CMCC 机器 (执行环境)
```
/root/work/david_work/
├── sana_wm_qc/                # QC 代码仓库
│   ├── src/                   # 同本机
│   └── scripts/               # 同本机
├── sana_wm_qc_env/            # Python 虚拟环境
├── qc_output_new/             # Stage 1+2 执行结果
│   ├── stage3_input_manifest.jsonl  # ← Stage 3 输入 (已生成)
│   └── wds-*/
└── stage1_batch_*.log         # Stage 1 执行日志
```

### 数据源 (CMCC 机器)
```
/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/
├── final_wds-sekai-game-drone/
├── final_wds-sekai-game-walking/
├── final_wds-OmniWorld-Game/
├── final_wds-DL3DV-ALL-2K/
├── final_wds-sekai-real-walking-hq/
├── final_wds-RealEstate10K-360p/
└── final_wds-SpatialVID-hq/
```

---

## 📊 Stage 1+2 执行结果汇总

### Stage 1 统计 (CPU 全量扫描)

| 数据集 | 总样本 | Pass | Flag | Fail | Pass率 |
|--------|--------|------|------|------|--------|
| wds-sekai-game-drone | 931 | 898 | 7 | 26 | 96.4% |
| wds-sekai-game-walking | 1,602 | 1,421 | 11 | 170 | 88.7% |
| wds-OmniWorld-Game | 6,378 | 6,203 | 39 | 136 | 97.2% |
| wds-DL3DV-ALL-2K | 9,937 | 6,464 | 2,805 | 668 | 65.0% |
| wds-sekai-real-walking-hq | 20,154 | 18,700 | 256 | 1,198 | 92.7% |
| wds-RealEstate10K-360p | 73,738 | 68,594 | 227 | 4,917 | 93.0% |
| wds-SpatialVID-hq | 220,508 | 193,229 | 9,597 | 17,682 | 87.6% |
| **总计** | **333,248** | **294,509** | **12,942** | **24,797** | **88.4%** |

**Stage 1 检查内容** (CPU-only):
1. ✅ 结构完整性: 6 个必需文件
2. ✅ Pose 几何有效性: SO(3), 第一帧, NaN/Inf
3. ✅ 轨迹质量: 跳跃检测 (不同数据集阈值 0.5m-5.0m)
4. ✅ 内参质量: FOV, 焦距散度, Scale 稳定性
5. ✅ Caption 质量: 长度 (game=10, real=50, DL3DV=0), 占位符, 相机动作词
6. ⚠️ 饱和度: 未启用 (需要 `--read-video-frames`)

---

### Stage 2 统计 (CPU/IO 采样深度检查)

**采样策略**:
- Flag 样本: 100% 检查
- Pass 样本: 5% 随机采样

| 数据集 | 检查数 | 问题数 | 问题率 | Tar错误 | 黑帧 | 场景切换 | 轨迹冻结 |
|--------|--------|--------|--------|---------|------|---------|---------|
| wds-sekai-game-drone | 44 | 4 | 9.1% | 0 | 4 | 0 | 0 |
| wds-sekai-game-walking | 72 | 4 | 5.6% | 0 | 2 | 0 | 2 |
| wds-OmniWorld-Game | 321 | 14 | 4.4% | 13 | 0 | 0 | 1 |
| wds-DL3DV-ALL-2K | 3,099 | 0 | 0.0% | 0 | 0 | 0 | 0 |
| wds-sekai-real-walking-hq | 1,133 | 1 | 0.1% | 0 | 0 | 0 | 1 |
| wds-RealEstate10K-360p | 3,641 | 134 | 3.7% | 113 | 0 | 19 | 2 |
| wds-SpatialVID-hq | 19,216 | 316 | 1.6% | 294 | 0 | 0 | 22 |
| **总计** | **27,526** | **473** | **1.7%** | **420** | **6** | **19** | **28** |

**Stage 2 检查内容** (CPU/IO 密集):
1. ✅ 视频帧数验证: PyAV 解码实际帧数与 npy 对齐
2. ✅ 黑帧检测: brightness < 10 的帧比例 (> 30% 标记)
3. ✅ Tar 文件完整性: 发现 420 个损坏文件 (Stage 1 无法检测)
4. ✅ 场景切换: PySceneDetect (仅 RealEstate10K, 阈值=1)
5. ✅ 轨迹冻结: 检测 > 50% 帧静止 (移动 < 0.0001m)

**关键发现**:
- ✅ **问题率 1.7% < 2% 阈值**: 验证 Pass 样本质量稳定，无需全量检查
- ✅ **Tar 错误 420 个**: Stage 2 独有发现，Stage 1 无法检测
- ✅ **黑帧问题 6 个**: sekai-game 数据集，最严重 100% 黑帧

---

### Stage 3 输入 manifest

**文件**: `/root/work/david_work/qc_output_new/stage3_input_manifest.jsonl` (CMCC)

**统计**:
```
总样本数: 282,222

各数据集:
  wds-DL3DV-ALL-2K: 9,269
  wds-OmniWorld-Game: 6,228
  wds-RealEstate10K-360p: 62,081
  wds-SpatialVID-hq: 185,280
  wds-sekai-game-drone: 901
  wds-sekai-game-walking: 1,428
  wds-sekai-real-walking-hq: 17,035

Stage 1 verdict 分布:
  pass: 270,119 (95.7%)
  flag: 12,103 (4.3%)
```

**筛选规则**:
```python
选入条件:
1. Stage 1 verdict in ["pass", "flag"]  # 排除 fail
2. Stage 2 未检查 OR Stage 2 无问题 (reasons == [])

排除样本:
- Stage 1 fail: 24,797
- Stage 2 有问题 (tar_error/black_frame/traj_frozen/scene_cuts): 473 (采样) → 推算全量 ~9,460
- Stage 2 未抽样的 Pass 中潜在问题: 按 1.7% 问题率估算

最终: 282,222 样本进入 Stage 3
```

---

## 🎯 Stage 3 执行计划 (当前任务)

### Stage 3 检查内容 (GPU 密集型)

1. **UniMatch 光流一致性** (GPU)
   - 前向/后向光流计算
   - 循环一致性检测
   - 与 camera poses 的一致性验证

2. **DOVER 视频质量评分** (GPU)
   - 技术质量分数 (清晰度、稳定性)
   - 美学质量分数

3. **Qwen3.5-9B Caption 重写** (GPU)
   - 仅针对 Flag 样本 (12,103 个)
   - 主要目标: DL3DV-ALL-2K (无 caption)
   - 移除相机动作词
   - 参数: `enable_thinking=False` (关键！17.91× 加速)

---

### 执行方案: 16 卡 H100 数据并行

#### 方案设计
- **策略**: Manifest 分片 + 多卡独立执行
- **GPU**: 2 节点 × 8 卡 = 16 卡 H100
- **加速比**: ~15× (理论)
- **预估耗时**: 4-5 小时 (vs 单卡 72 小时)

#### 执行流程

**Step 1: Manifest 分片** (CMCC)
```bash
cd /root/work/david_work/qc_output_new

python3 << 'PYEOF'
import json
from pathlib import Path

manifest_file = Path("stage3_input_manifest.jsonl")
output_dir = Path("stage3_manifests_shards")
output_dir.mkdir(exist_ok=True)

samples = []
with open(manifest_file, 'r') as f:
    for line in f:
        samples.append(line)

total = len(samples)
n_gpus = 16
samples_per_gpu = (total + n_gpus - 1) // n_gpus

for gpu_id in range(n_gpus):
    start = gpu_id * samples_per_gpu
    end = min(start + samples_per_gpu, total)
    
    shard_file = output_dir / f"shard_{gpu_id:02d}_of_{n_gpus:02d}.jsonl"
    with open(shard_file, 'w') as f:
        f.writelines(samples[start:end])
    
    print(f"Shard {gpu_id:02d}: {end - start} samples")

print(f"✅ 分片完成: {output_dir}")
PYEOF
```

**预期输出**:
```
Shard 00: 17639 samples
Shard 01: 17639 samples
...
Shard 15: 17638 samples
✅ 分片完成: stage3_manifests_shards
```

---

**Step 2: 多卡并行启动脚本**

```bash
cd /root/work/david_work

cat > launch_stage3_multi_gpu.sh << 'EOF'
#!/bin/bash
set -e

N_GPUS=16
MANIFEST_DIR="/root/work/david_work/qc_output_new/stage3_manifests_shards"
OUTPUT_DIR="/root/work/david_work/qc_output_new/stage3_results"
SCRIPT="sana_wm_qc/scripts/run_stage3.py"

mkdir -p "$OUTPUT_DIR"

cd /root/work/david_work/sana_wm_qc
source /root/work/david_work/sana_wm_qc_env/bin/activate
export PYTHONPATH=/root/work/david_work/sana_wm_qc/src:$PYTHONPATH

# 并行启动 16 个进程
for gpu_id in $(seq 0 15); do
    shard_file="$MANIFEST_DIR/shard_$(printf '%02d' $gpu_id)_of_16.jsonl"
    output_file="$OUTPUT_DIR/stage3_results_gpu$(printf '%02d' $gpu_id).jsonl"
    log_file="$OUTPUT_DIR/gpu$(printf '%02d' $gpu_id).log"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT" \
        --manifest "$shard_file" \
        --output "$output_file" \
        --gpu-id $gpu_id \
        2>&1 | tee "$log_file" &
done

wait

# 合并结果
cat "$OUTPUT_DIR"/stage3_results_gpu*.jsonl > "$OUTPUT_DIR/stage3_results_all.jsonl"
echo "✅ Stage 3 完成: $(wc -l < "$OUTPUT_DIR/stage3_results_all.jsonl") 样本"
EOF

chmod +x launch_stage3_multi_gpu.sh
```

---

**Step 3: Stage 3 核心脚本框架**

`/root/work/david_work/sana_wm_qc/scripts/run_stage3.py`:

```python
#!/usr/bin/env python3
"""Stage 3: GPU-intensive quality checks."""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

# 需要用户提供的模型接口
# from sana_wm_pipeline.stage03_quality.unimatch import UniMatchModel
# from sana_wm_pipeline.stage03_quality.dover import DOVERModel
# from sana_wm_pipeline.stage03_quality.caption_rewrite import QwenCaptionRewriter


def load_models(gpu_id: int):
    """加载 Stage 3 模型"""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] 加载模型...")
    
    # TODO: 实际模型加载
    # unimatch = UniMatchModel(device=device)
    # dover = DOVERModel(device=device)
    # qwen = QwenCaptionRewriter(
    #     model_path="/path/to/qwen3.5-9b",
    #     device=device,
    #     enable_thinking=False  # 关键！
    # )
    
    return {}  # 占位


def process_one_sample(sample: dict, models: dict, gpu_id: int) -> dict:
    """处理单个样本"""
    result = {
        "sample_id": sample["sample_id"],
        "group": sample["group"],
        "stage3": {
            "unimatch_consistency": None,
            "dover_technical": None,
            "dover_aesthetic": None,
            "caption_rewritten": False,
            "new_caption": None,
            "reasons": []
        }
    }
    
    try:
        # TODO: 实际检查逻辑
        # 1. 从 tar 提取 video + poses
        # 2. UniMatch 光流一致性
        # 3. DOVER 质量评分
        # 4. Qwen Caption 重写 (仅 Flag)
        pass
        
    except Exception as e:
        result["stage3"]["reasons"].append(f"error: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    
    models = load_models(args.gpu_id)
    
    samples = []
    with open(args.manifest, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"[GPU {args.gpu_id}] 处理 {len(samples)} 样本")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f_out:
        for sample in tqdm(samples, desc=f"GPU {args.gpu_id}"):
            result = process_one_sample(sample, models, args.gpu_id)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
    
    print(f"[GPU {args.gpu_id}] 完成")


if __name__ == "__main__":
    main()
```

---

### 冒烟测试方案

**推荐**: wds-sekai-game-drone (901 样本)

```bash
# 提取冒烟测试 manifest
cd /root/work/david_work/qc_output_new

python3 << 'PYEOF'
import json
from pathlib import Path

with open("stage3_input_manifest.jsonl", 'r') as f_in, \
     open("stage3_smoke_test_manifest.jsonl", 'w') as f_out:
    for line in f_in:
        rec = json.loads(line)
        if rec["group"] == "wds-sekai-game-drone":
            f_out.write(line)

print("✅ 冒烟测试 manifest: 901 samples")
PYEOF

# 执行冒烟测试 (单卡)
CUDA_VISIBLE_DEVICES=0 python /root/work/david_work/sana_wm_qc/scripts/run_stage3.py \
    --manifest stage3_smoke_test_manifest.jsonl \
    --output stage3_smoke_test_results.jsonl \
    --gpu-id 0

# 预计耗时: 10-15 分钟
```

---

## 🔧 待解决问题 (需要用户提供)

### 1. 模型路径和接口

**UniMatch**:
```python
# 需要提供:
# - 模型权重路径
# - 输入格式 (video frames shape)
# - 输出格式 (consistency score)
```

**DOVER**:
```python
# 需要提供:
# - 模型权重路径
# - 输入格式
# - 输出格式 (technical_score, aesthetic_score)
```

**Qwen3.5-9B**:
```python
# 需要提供:
# - 模型路径: /path/to/Qwen3.5-9B-Instruct
# - 是否已下载
# - 加载方式 (transformers/vllm/other)
```

### 2. 当前 Stage 3 代码状态

- 是否已有 `sana_wm_pipeline/stage03_quality/` 目录？
- 是否已有部分模型代码？
- 需要从头实现还是集成现有代码？

### 3. Tar 文件读取

- Stage 3 如何从 tar 中提取 video + poses？
- 复用 Stage 1/2 的 `_extract_samples_from_tar()` 函数？

---

## 📝 各数据集配置参数

### group_config.py 配置汇总

| 数据集 | jump_threshold_m | max_jumps_flag | max_jumps_fail | min_caption_len | check_camera_words | max_scene_cuts | table6_source |
|--------|------------------|----------------|----------------|-----------------|-------------------|---------------|---------------|
| wds-DL3DV-ALL-2K | 0.5 | 3 | 50 | 0 | True | None | DL3DV |
| wds-RealEstate10K-360p | 0.5 | 3 | 50 | 50 | True | 1 | RealEstate10K |
| wds-sekai-real-walking-hq | 0.5 | 3 | 15 | 50 | True | None | Sekai_Walking |
| wds-SpatialVID-hq | 0.5 | 0 | 5 | 50 | True | None | SpatialVID |
| wds-OmniWorld-Game | 2.0 | 15 | 50 | 10 | False | None | OmniWorld |
| wds-sekai-game-drone | 5.0 | 20 | 80 | 10 | False | None | Sekai_Game_Drone |
| wds-sekai-game-walking | 2.0 | 15 | 50 | 10 | False | None | Sekai_Game_Walking |

**注意**:
- **游戏数据集**: `check_camera_words=False` (caption 包含相机动作是正常的)
- **DL3DV**: `min_caption_len=0` (无 caption，需 Stage 3 生成)
- **RealEstate10K**: `max_scene_cuts=1` (唯一检查场景切换)

---

## 📚 关键文档位置

### 本机
1. **Stage 2 分析报告**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/stage2_result/STAGE2_ANALYSIS_REPORT.md`
2. **会话交接文档**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/SESSION_HANDOFF_SUMMARY.md` (本文档)
3. **Stage 1+2 代码**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/`

### CMCC 机器
1. **Stage 1 日志**: `/root/work/david_work/stage1_batch_20260804_095828.log`
2. **Stage 2 日志**: 各数据集目录下 `stage2_run.log`
3. **Stage 3 输入**: `/root/work/david_work/qc_output_new/stage3_input_manifest.jsonl`

---

## 🎯 下一步行动清单

### 立即执行 (冒烟测试)
1. ☐ 用户提供 UniMatch/DOVER/Qwen3.5 模型路径和接口
2. ☐ 完善 `run_stage3.py` 脚本
3. ☐ 在 CMCC 单卡执行冒烟测试 (901 样本, 10-15 分钟)
4. ☐ 验证结果格式和质量

### 后续执行 (全量)
5. ☐ Manifest 分片 (16 份)
6. ☐ 多卡并行执行 (16×H100, 4-5 小时)
7. ☐ 合并结果并验证
8. ☐ 生成最终训练数据清单

### 最终交付
9. ☐ Stage 1+2+3 完整质量报告
10. ☐ 最终通过率统计
11. ☐ WebDataset 重新打包 (仅保留高质量样本)

---

## 💡 重要提示

### 工具调用注意事项
- ✅ Write 工具: 必须提供 `file_path` 和 `content` 两个参数
- ✅ 大文件: 内容超过 150 行需分块写入
- ✅ 失败处理: 第一次失败就停止，运行诊断，不要重试

### CMCC 机器访问
- ⚠️ **本机无法直接访问 CMCC 机器**
- ⚠️ 所有 CMCC 命令需要用户手动执行
- ✅ 提供完整的可复制粘贴命令

### 数据持久化
- ✅ 本机持久路径: `/mnt/afs/davidwang/workspace`
- ✅ CMCC 热盘易丢失，需要 rsync 到 filestorage

---

## 📞 会话恢复检查清单

新的 Claude 会话开始时，请确认:

1. ☐ 已阅读本文档 (`SESSION_HANDOFF_SUMMARY.md`)
2. ☐ 已阅读 Stage 2 分析报告 (`STAGE2_ANALYSIS_REPORT.md`)
3. ☐ 已了解当前进度: Stage 1+2 完成，Stage 3 准备中
4. ☐ 已知晓 Stage 3 输入: 282,222 样本
5. ☐ 已明确下一步: 完善 `run_stage3.py` 并执行冒烟测试

---

**文档版本**: v1.0  
**最后更新**: 2026-08-05  
**创建者**: Claude Sonnet 4.6

# Stage 3 质检管线快速开始指南

> 在 CMCC 机器上跑通 Stage 3（UniMatch + DOVER + Qwen Caption 改写）

---

## 一、前置条件检查

- ✅ **环境已准备**：`dover_h100_test` 已克隆到 `/mnt/afs/davidwang/miniconda3/envs/dover_h100_test`
- ✅ **数据已产出**：7 个 group 在 `/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/`
- ✅ **Stage 1+2 已完成**：QC 结果存在 `qc_output/*/stage{1,2}_results.jsonl`

---

## 二、三步快速启动

### Step 1：环境验证（30 分钟）

```bash
# 激活环境
conda activate /mnt/afs/davidwang/miniconda3/envs/dover_h100_test

# 验证核心组件
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import dover; print('DOVER OK')" || pip install dover-lap

# 检查显存
nvidia-smi
```

### Step 2：补全依赖（1 小时）

```bash
cd /root/work/david_work/sana_wm_pipeline

# 安装 UniMatch
git clone https://github.com/autonomousvision/unimatch.git third_party/unimatch
pip install -e third_party/unimatch --no-deps

# 下载 UniMatch 权重（从 AFS 传输）
mkdir -p /root/work/david_work/models/unimatch
# scp user@afs:/path/to/unimatch/weights/* /root/work/david_work/models/unimatch/

# 补全其他依赖
pip install av scenedetect

# 验证
python -c "import unimatch; import dover; import av; import scenedetect; print('ALL OK')"
```

### Step 3：单样本测试（30 分钟）

```bash
# 设置环境变量（离线模式）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 选择测试样本
TEST_SAMPLE="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/wds-DL3DV-ALL-2K/w000/shard-000000.tar"

# 运行测试
python -m sana_wm_pipeline.qc.stage3_gpu \
  --input_tar $TEST_SAMPLE \
  --output_dir /tmp/stage3_test \
  --model_unimatch /root/work/david_work/models/unimatch \
  --model_qwen /root/work/david_work/models/Qwen2-7B-VL \
  --gpu_id 0

# 检查结果
cat /tmp/stage3_test/stage3_results.jsonl
```

**期望输出**：
```json
{
  "sample_id": "...",
  "unimatch_flow": {"pass": true, "mean_flow": 15.3},
  "dover_quality": {"pass": true, "score": 0.72},
  "caption_rewrite": {...},
  "stage3_pass": true
}
```

---

## 三、全量执行（10-14 小时）

```bash
# 启动 48 GPU 并行任务
python scripts/run_stage3_cmcc.py \
  --stage2_results qc_output/*/stage2_results.jsonl \
  --video_root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
  --output_dir qc_output/stage3/ \
  --num_gpus 48

# 实时监控
watch -n 10 'find qc_output/stage3/ -name "*.jsonl" | wc -l'
```

---

## 四、常见问题速查

| 问题 | 解决方案 |
|------|---------|
| `CUDA out of memory` | 检查 nvidia-smi，释放其他进程显存 |
| `ModuleNotFoundError: unimatch` | `pip install -e third_party/unimatch` |
| `HTTPSConnectionPool timeout` | `export TRANSFORMERS_OFFLINE=1` |
| DOVER 加载失败 | 检查权重路径和 PyTorch 版本 |
| GPU 利用率低 | 检查 worker 数量是否匹配 GPU 数量 |

---

## 五、关键注意事项

### ✅ DOVER 使用 H100 GPU 模式（已验证）
- 2026-07-02 验证通过，性能：1-2s/10秒视频
- 详见 `models/DOVER/H100_INSTALLATION_GUIDE.md`
- CMCC 部署指南：`DOVER_H100_部署方案.md`

### ✅ Qwen 模型已选定
- **Qwen3.5-9B**（~18GB，推理 0.5-1s/样本）
- 正在传输到 CMCC 机器
- 部署路径：`/root/work/david_work/models/Qwen3.5-9B/`

### ✅ 离线模式必须开启
CMCC 无外网，必须设置：
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## 六、完成后下一步

1. **合并 QC 结果**：
   ```bash
   python scripts/merge_qc_results.py \
     --stage1 qc_output/*/stage1_results.jsonl \
     --stage2 qc_output/*/stage2_results.jsonl \
     --stage3 qc_output/stage3/stage3_results.jsonl \
     --human qc_output/human_review_results.jsonl \
     --output qc_final_output/
   ```

2. **提取训练数据**：
   ```bash
   python scripts/extract_training_data.py \
     --pass_list qc_final_output/pass_final.txt \
     --source_dir /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/ \
     --output_dir /root/work/filestorage/shangaoooooo/davidwang/training_data_final/
   ```

---

## 七、文档索引

- **完整计划**：`STAGE3_EXECUTION_PLAN.md`
- **QC 系统详解**：`docs/03-QC_SYSTEM.md`
- **故障排查**：`docs/05-TROUBLESHOOTING.md`
- **API 陷阱**：`docs/reference/API_REFERENCE.md`

---

**预估总耗时**（更新）：7-10 小时（约半个工作日，基于 H100 GPU 模式）  
**DOVER 部署重点**：详见 `DOVER_H100_部署方案.md`  
**下一步**：执行 Step 1 环境验证

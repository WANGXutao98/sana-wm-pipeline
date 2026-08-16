# Stage3 批量处理执行指南

## 📋 当前状态诊断

### 问题确认
- ❌ **原批量任务已中断**: 只处理了 1/5000 个样本后停止
- ✅ **1 个样本成功处理**: 输出正常，模型推理正常
- ⚠️ **中断原因**: 后台进程被 tee 或 SSH 断开导致停止

### 已处理样本
```json
{"sample_id": "00094653-a9c6-5558-8e2a-4119e7d64f36", 
 "unimatch_flow": 41.528, 
 "dover_score": -0.0479, 
 "verdict": "fail"}
```

---

## 🔍 模型权重加载验证

### ✅ 确认：权重仅加载一次

**代码逻辑** (`stage3_batch.py` L17-42 & L141-148):

```python
# L17-42: 定义 load_models() 函数
def load_models(device="cuda"):
    dover = DOVER(...)
    dover.load_state_dict(torch.load(...))  # 一次性加载
    unimatch = UniMatch(...)
    unimatch.load_state_dict(torch.load(...))  # 一次性加载
    return dover, unimatch

# L141-148: 主循环
dover_model, unimatch_model = load_models(args.device)  # ← 仅此一次
for video_path in tqdm(videos):
    result = process_one_video(video_path, dover_model, unimatch_model, ...)  # ← 复用模型
```

**结论**: 
- ✅ **无重复加载问题**
- ✅ 模型对象在循环外初始化，循环内仅调用 `model()`
- ✅ 权重文件仅读取一次（启动时）

---

## 📐 完整执行流程

### 阶段 1: 环境准备
```bash
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
```

### 阶段 2: 模型加载（一次性）
```python
# 程序启动时执行一次
dover_model = DOVER(...)  # 785 MB GPU 显存
dover_model.load_state_dict(torch.load("DOVER.pth"))

unimatch_model = UniMatch(...)  # 20 MB GPU 显存
unimatch_model.load_state_dict(torch.load("gmflow-scale2-regrefine6-mixdata.pth"))

# 总显存占用: ~1 GB (模型权重)
```

### 阶段 3: 批量推理循环
```python
for video in all_5000_videos:
    # 3.1 解码视频 (CPU)
    frames = decord.VideoReader(video)[:]  # 87 帧 @ 1280×720
    
    # 3.2 UniMatch 光流计算
    flow_mag = compute_unimatch_flow(frames, unimatch_model, device)
    # - 每 0.5s 采样一对帧
    # - 推理时间: ~7s
    # - GPU 峰值: +2 GB (临时激活值)
    
    # 3.3 DOVER 质量评分
    dover_score = compute_dover_score(frames, dover_model, device)
    # - 32 帧分块，共 3 块
    # - 推理时间: ~2.7s
    # - GPU 峰值: +3.5 GB / 块 (分块后清理)
    
    # 3.4 判定并写入结果
    verdict = "pass" if (flow_mag in [3,80] and dover_score in [0.35,1.0]) else "fail"
    write_jsonl({"sample_id": ..., "verdict": verdict, ...})
    
    # 3.5 清理 GPU 缓存
    torch.cuda.empty_cache()

# 总循环时间: 5000 × 10s = 13.9 小时
```

### 阶段 4: 汇总统计
```python
# 读取所有结果
results = read_jsonl("stage3_results.jsonl")

# 统计
pass_count = sum(r["verdict"] == "pass" for r in results)
fail_count = sum(r["verdict"] == "fail" for r in results)

print(f"Pass: {pass_count} ({pass_count/5000*100:.1f}%)")
print(f"Fail: {fail_count} ({fail_count/5000*100:.1f}%)")
```

---

## 🚀 手动启动命令（推荐）

### 方案 A: 改进版脚本（支持断点续传）

```bash
# 激活环境
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc

# 进入工作目录
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 后台运行（使用 nohup 避免 SSH 断开）
nohup python scripts/stage3_batch_robust.py \
  --input_dir /mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001 \
  --output /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_v2.jsonl \
  --resume \
  > /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_v2.nohup 2>&1 &

# 记录进程 ID
echo $! > /tmp/stage3_batch.pid

# 实时监控进度
tail -f /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_v2.log
```

**优势**:
- ✅ 支持断点续传（`--resume` 自动跳过已处理样本）
- ✅ 独立日志文件，不依赖 tee
- ✅ nohup 确保 SSH 断开后继续运行
- ✅ 每 10 个样本输出进度

### 方案 B: 原始脚本（简单但无续传）

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

nohup python scripts/stage3_batch.py \
  --input_dir /mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001 \
  --output /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results.jsonl \
  > /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3.nohup 2>&1 &

echo $! > /tmp/stage3_batch.pid
```

---

## 📊 监控命令

### 实时查看进度
```bash
# 方案 A (改进版)
tail -f /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_v2.log

# 方案 B (原始版)
tail -f /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3.nohup
```

### 查看已处理样本数
```bash
wc -l /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_v2.jsonl
```

### 查看最新结果
```bash
tail -3 /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_v2.jsonl | jq
```

### 检查进程是否还在运行
```bash
ps -p $(cat /tmp/stage3_batch.pid) || echo "进程已停止"
```

### GPU 显存监控
```bash
watch -n 5 nvidia-smi
```

---

## ⏱️ 预期时间

| 阶段 | 单样本耗时 | 5000 样本总耗时 |
|------|-----------|----------------|
| 视频解码 | 0.3s | 25 分钟 |
| UniMatch | 7s | 9.7 小时 |
| DOVER | 2.7s | 3.75 小时 |
| 写入结果 | 0.01s | 1 分钟 |
| **总计** | **~10s** | **~13.9 小时** |

实际可能更快（GPU 预热后加速）。

---

## 🔧 故障恢复

### 如果任务中途停止

**方案 A (改进版)**: 直接加 `--resume` 重新运行，自动跳过已处理样本
```bash
nohup python scripts/stage3_batch_robust.py \
  --input_dir /mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001 \
  --output /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_v2.jsonl \
  --resume \
  > /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_v2.nohup 2>&1 &
```

**方案 B (原始版)**: 手动编辑脚本，添加已处理样本过滤
```python
# 在 L138 后添加
processed_ids = set()
if Path(args.output).exists():
    with open(args.output) as f:
        for line in f:
            data = json.loads(line)
            processed_ids.add(data["sample_id"])
videos = [v for v in videos if v.stem not in processed_ids]
```

---

## ✅ 验证清单

执行前确认：
- [ ] conda 环境 `sana_qc` 已激活
- [ ] GPU 空闲（`nvidia-smi` 显示 0 MB 已用）
- [ ] 输入目录包含 5000 个 `.mp4` 文件
- [ ] 输出目录可写
- [ ] 磁盘空间充足（日志文件 ~100MB）

---

## 📝 最终输出格式

**JSONL 文件** (`stage3_results_v2.jsonl`):
```json
{"sample_id": "xxx", "unimatch_flow": 41.528, "dover_score": -0.0479, "verdict": "fail", "reasons": ["dover=-0.0479 not in [0.35, 1.0]"]}
{"sample_id": "yyy", "unimatch_flow": 12.3, "dover_score": 0.67, "verdict": "pass", "reasons": []}
...
```

**统计信息** (日志末尾):
```
Total:  5000
Pass:   3850 (77.0%)
Fail:   1120 (22.4%)
Error:  30 (0.6%)
Time:   13.5h
```

---

**推荐执行**: 使用方案 A（改进版脚本），支持断点续传，更稳健。

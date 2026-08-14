# Stage 2 V2 - 死锁修复方案

## 问题诊断

**症状**：wds-RealEstate10K-360p 处理 18+ 小时无输出，所有 32 个 worker 进程状态为 'T' (Stopped/traced)，CPU 0%

**根本原因**：`find_sample_files()` 在每个样本调用时都执行 `glob()` 扫描，68,821 个样本导致主进程在构建任务列表时阻塞，所有 worker 进入等待状态

## 解决方案

### 修改点

**v1 (旧版本)**：
```python
def deep_check_sample_extracted(sample_id, data_root, group_name):
    files = find_sample_files(sample_id, data_root)  # ← 每次都 glob
    # ... 处理
```

**v2 (新版本)**：
```python
def build_sample_index(data_root, sample_ids):
    """一次性构建索引"""
    index = {}
    for shard_dir in data_root.glob("final_wds-*/wds-*/w*/shard-*"):
        for mp4_file in shard_dir.glob("*.mp4"):
            if mp4_file.stem in sample_ids:
                index[mp4_file.stem] = {
                    'mp4': mp4_file,
                    'poses': mp4_file.with_suffix('.poses_c2w.npy')
                }
    return index

def run_stage2_extracted_v2(...):
    # 1. 选择样本
    selected_ids = set(...)
    
    # 2. 构建索引（主进程，Pool 创建前）
    sample_index = build_sample_index(data_root, selected_ids)
    
    # 3. 准备任务（直接传文件路径）
    tasks = [(sid, sample_index[sid], group) for sid in selected_ids if sid in sample_index]
    
    # 4. 创建 Pool（worker 不需要 glob）
    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker_fn_with_index, tasks):
            # ...
```

### 关键改进

1. **索引提前构建**：在 Pool 创建前一次性扫描所有文件
2. **Worker 无 I/O**：worker 直接接收文件路径，不需要文件查找
3. **进度可见**：索引构建时每 100 个 shard 输出进度

## 部署步骤

### 1. 停止当前进程

```bash
# 查找 stuck 进程
ps aux | grep stage2 | grep RealEstate

# 终止所有相关进程
pkill -f "stage2.*RealEstate"

# 验证
ps aux | grep stage2
```

### 2. 部署 v2 代码

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 打包
tar czf stage2_v2_fix.tar.gz \
    src/sana_wm_pipeline/qc/stage2_deep_extracted_v2.py \
    scripts/run_stage2_full_extracted_v2.sh \
    STAGE2_V2_FIX.md

# 传输到 CMCC（根据实际配置调整）
scp stage2_v2_fix.tar.gz user@cmcc:/root/work/david_work/
```

### 3. 在 CMCC 机器上部署

```bash
cd /root/work/david_work/sana_wm_qc
tar xzf ../stage2_v2_fix.tar.gz
chmod +x scripts/run_stage2_full_extracted_v2.sh

# 验证
ls -l src/sana_wm_pipeline/qc/stage2_deep_extracted_v2.py
python -c "from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2; print('OK')"
```

### 4. 重新执行 RealEstate10K

#### 方法 A：单独执行 RealEstate10K

```bash
cd /root/work/david_work/sana_wm_qc
source /root/work/david_work/sana_wm_qc_env/bin/activate
export PYTHONPATH=/root/work/david_work/sana_wm_qc/src:$PYTHONPATH

# 单独执行
python -u << 'PYEOF' | tee /root/work/david_work/qc_output_new/wds-RealEstate10K-360p/stage2_run_full_v2.log
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2

s1 = Path("/root/work/david_work/qc_output_new/wds-RealEstate10K-360p/stage1_results.jsonl")
s2 = Path("/root/work/david_work/qc_output_new/wds-RealEstate10K-360p/stage2_results_full_v2.jsonl")
data = Path("/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output")

n = run_stage2_extracted_v2(s1, s2, data, sample_frac=1.0, n_workers=32)
print(f"\n✅ 完成！处理了 {n} 个样本")
PYEOF
```

#### 方法 B：执行剩余数据集（RealEstate10K + SpatialVID）

```bash
# 修改脚本只处理剩余数据集
cd /root/work/david_work/sana_wm_qc
cat > scripts/run_remaining_v2.sh << 'BASH'
#!/bin/bash
set -e
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"
OUTPUT_ROOT="/root/work/david_work/qc_output_new"

for group in wds-RealEstate10K-360p wds-SpatialVID-hq; do
    echo "处理: $group - $(date)"
    python -u -c "
from pathlib import Path
from sana_wm_pipeline.qc.stage2_deep_extracted_v2 import run_stage2_extracted_v2
n = run_stage2_extracted_v2(
    Path('$OUTPUT_ROOT/$group/stage1_results.jsonl'),
    Path('$OUTPUT_ROOT/$group/stage2_results_full_v2.jsonl'),
    Path('$DATA_ROOT'),
    sample_frac=1.0, n_workers=32
)
print(f'[完成] {n} 个样本')
    " 2>&1 | tee "$OUTPUT_ROOT/$group/stage2_run_full_v2.log"
done
BASH

chmod +x scripts/run_remaining_v2.sh
nohup bash scripts/run_remaining_v2.sh > /root/work/david_work/stage2_remaining_v2.log 2>&1 &
```

## 预期性能

| 阶段 | RealEstate10K (68,821 样本) | SpatialVID (214,216 样本) |
|------|------------------------|------------------------|
| **索引构建** | ~2-3 分钟 | ~5-8 分钟 |
| **深度检查** | ~1-1.5 小时 | ~3-4 小时 |
| **总计** | **~1.5 小时** | **~4 小时** |

v1 版本卡在索引阶段（18+ 小时无输出），v2 版本将索引时间缩短到分钟级别。

## 监控命令

```bash
# 实时日志
tail -f /root/work/david_work/qc_output_new/wds-RealEstate10K-360p/stage2_run_full_v2.log

# 进度统计
wc -l /root/work/david_work/qc_output_new/wds-RealEstate10K-360p/stage2_results_full_v2.jsonl

# 进程状态
ps aux | grep stage2 | grep -v grep

# CPU 使用
top -b -n 1 | grep python | head -5
```

## 输出文件

```
/root/work/david_work/qc_output_new/
├── wds-RealEstate10K-360p/
│   ├── stage2_results_full.jsonl      # v1 输出（空文件，0 行）
│   ├── stage2_results_full_v2.jsonl   # v2 输出（新）✅
│   └── stage2_run_full_v2.log         # v2 日志
└── wds-SpatialVID-hq/
    ├── stage2_results_full_v2.jsonl   # v2 输出
    └── stage2_run_full_v2.log         # v2 日志
```

## 验证清单

执行前：
- [ ] v1 进程已全部终止
- [ ] v2 代码已部署
- [ ] 模块导入测试通过

执行后：
- [ ] 索引构建有进度输出（每 100 个 shard）
- [ ] 深度检查有进度输出（每 1000 个样本）
- [ ] `stage2_results_full_v2.jsonl` 行数合理（~69k / ~214k）
- [ ] CPU 利用率正常（32 进程 × ~100%）

---

**版本**: v2  
**日期**: 2026-08-08  
**状态**: ✅ 修复完成，可立即部署

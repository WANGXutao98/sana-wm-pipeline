# Stage 2 执行总结 - 快速参考

> **1 分钟速览**：Stage 2 已完成，数据已解压，可直接进入 Stage 3

---

## ✅ Stage 2 执行状态

| 项目 | 状态 | 数值 |
|------|------|------|
| **执行状态** | ✅ 完成 | 2026-08-04 |
| **检查样本数** | ✅ 完成 | 27,526 / 333,248 (8.3%) |
| **问题样本数** | ✅ 发现 | 473 (1.7%) |
| **数据解压** | ✅ 完成 | 所有 tar 已解压 |
| **进入 Stage 3** | ✅ 就绪 | ~295,000 样本 |

---

## 📊 关键数据

### 样本统计
```
原始样本:     333,248
├─ Stage 1 Pass:  294,509 (88.4%)
├─ Stage 1 Flag:   12,942 (3.9%)
└─ Stage 1 Fail:   24,797 (7.4%)

Stage 2 检查:  27,526 (Flag 100% + Pass 5%)
└─ 新增问题:      473 (1.7%)
    ├─ Tar 损坏:     420 (88.8%)
    ├─ 黑帧:           6 (1.3%)
    ├─ 场景切换:      19 (4.0%)
    └─ 轨迹冻结:      28 (5.9%)

Stage 3 输入: ~295,000 (高质量样本)
```

---

## 🗂️ 数据路径结构

### 解压后的目录结构
```
/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/
├── final_wds-DL3DV-ALL-2K/
│   └── wds-DL3DV-ALL-2K/
│       └── w000/ ... w047/           # 48 个 worker 分片
│           └── shard-000000-000000/  # 解压后的目录 ✅
│               ├── {dataset}_{UUID}.mp4
│               ├── {dataset}_{UUID}.caption.txt
│               ├── {dataset}_{UUID}.intrinsics.npy
│               ├── {dataset}_{UUID}.poses_c2w.npy
│               ├── {dataset}_{UUID}.meta.json
│               └── {dataset}_{UUID}.scale.npy
├── final_wds-OmniWorld-Game/
├── final_wds-RealEstate10K-360p/
├── final_wds-SpatialVID-hq/
├── final_wds-sekai-game-drone/
├── final_wds-sekai-game-walking/
├── final_wds-sekai-real-walking-hq/
└── sample_completeness.csv
```

### 样本文件示例
```
sample_id = "SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18"

文件：
- SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18.mp4
- SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18.caption.txt
- SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18.intrinsics.npy
- SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18.poses_c2w.npy
- SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18.meta.json
- SpatialVID-hq_05b84042-799c-55b1-8a0a-77a2911ecd18.scale.npy
```

---

## 🔗 Stage 2 结果位置

```bash
# Stage 2 结果（本地）
/mnt/afs/davidwang/workspace/sana_wm_pipeline/stage2_result/qc_output_new/
├── wds-DL3DV-ALL-2K/stage2_results.jsonl
├── wds-OmniWorld-Game/stage2_results.jsonl
├── wds-RealEstate10K-360p/stage2_results.jsonl
├── wds-SpatialVID-hq/stage2_results.jsonl
├── wds-sekai-game-drone/stage2_results.jsonl
├── wds-sekai-game-walking/stage2_results.jsonl
└── wds-sekai-real-walking-hq/stage2_results.jsonl

# 分析报告
/mnt/afs/davidwang/workspace/sana_wm_pipeline/stage2_result/STAGE2_ANALYSIS_REPORT.md
```

---

## 🚀 数据路径映射机制

### 旧方案（已弃用）
```python
# ❌ 慢：每次都要解压 tar
with tarfile.open("shard.tar", "r") as tf:
    video_bytes = tf.extractfile(f"{sample_id}.mp4").read()
# 耗时：~100-200 ms/样本
```

### 新方案（已实现）✅
```python
# ✅ 快：直接从解压目录读取
from scripts.data_loader_cmcc import Stage3DataLoaderCMCC

loader = Stage3DataLoaderCMCC(data_root)
files = loader.get_sample_files(sample_id)
# files['mp4'] → Path to .mp4
# files['caption'] → Path to .caption.txt
# 耗时：~1-5 ms/样本（20-200x 加速）
```

### 快速验证
```bash
# 验证索引构建
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/data_loader_cmcc.py \
    /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

# 输出示例：
# 索引构建完成，共 333248 个样本
# 统计信息:
#   索引样本数: 333248
#   完整样本数: 0 (如果没有 CSV)
#   可处理样本数: 333248
```

---

## 📋 Stage 3 准备清单

### ✅ 已完成
- [x] Stage 1+2 执行完成
- [x] 所有 tar 文件已解压
- [x] 数据索引机制已实现（`data_loader_cmcc.py`）
- [x] Stage 3 执行脚本已完成（`run_stage3_cmcc.py`）
- [x] 单样本测试脚本就绪（`run_stage3_single_sample.py`）
- [x] 三大模块已验证（UniMatch、DOVER、Qwen）
- [x] 性能优化完成（17.91x Qwen 加速）

### ⏳ 待执行
- [ ] 生成 Stage 3 输入 manifest（5 分钟）
- [ ] 单样本端到端测试（2 分钟）
- [ ] 小批量测试（10 分钟，100 样本）
- [ ] 全量执行（7-8 小时，48 GPU）

---

## 🎯 立即可执行的命令

### 1. 生成 Stage 3 输入 manifest
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/stage2_result/qc_output_new

python3 << 'PYEOF'
import json
from pathlib import Path

stage3_input = []

for dataset_dir in sorted(Path(".").glob("wds-*")):
    s1_file = dataset_dir / "stage1_results.jsonl"
    s2_file = dataset_dir / "stage2_results.jsonl"
    
    if not s1_file.exists():
        continue
    
    s1_records = {}
    with open(s1_file, 'r') as f:
        for line in f:
            rec = json.loads(line)
            s1_records[rec["sample_id"]] = rec
    
    s2_checked = {}
    if s2_file.exists():
        with open(s2_file, 'r') as f:
            for line in f:
                rec = json.loads(line)
                s2_checked[rec["sample_id"]] = rec
    
    for sid, s1_rec in s1_records.items():
        if s1_rec["verdict"] == "fail":
            continue
        
        if sid in s2_checked:
            s2 = s2_checked[sid].get("stage2", {})
            if s2.get("reasons"):
                continue
        
        stage3_input.append({
            "sample_id": sid,
            "group": s1_rec["group"],
            "tar_path": s1_rec["tar_path"],
            "stage1_verdict": s1_rec["verdict"],
            "stage2_checked": sid in s2_checked,
            "metrics": s1_rec["metrics"]
        })

output_file = Path("stage3_input_manifest.jsonl")
with open(output_file, 'w') as f:
    for rec in stage3_input:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Stage 3 manifest: {output_file}")
print(f"   总样本数: {len(stage3_input)}")
PYEOF
```

### 2. 统计样本数量
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/stage2_result/qc_output_new
wc -l stage3_input_manifest.jsonl
```

### 3. 验证数据路径
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
python scripts/data_loader_cmcc.py \
    /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
```

---

## 📚 完整文档

| 文档 | 说明 |
|------|------|
| **STAGE2_EXECUTION_SUMMARY.md** | Stage 2 完整执行方案（本文档详细版）|
| **STAGE2_ANALYSIS_REPORT.md** | Stage 2 完整分析报告（在 stage2_result/ 下）|
| **STAGE3_完整打通方案_2026-08-04.md** | Stage 3 执行计划 |
| **STAGE3_技术总结与发现.md** | 关键发现和性能数据 |
| **SESSION_SUMMARY_2026-08-04.md** | 完整对话总结 |

---

## 🔑 关键发现

### 1. Tar 损坏问题已解决 ✅
- **问题**：420 个样本的 tar 文件损坏（88.8%）
- **解决**：所有 tar 已解压，直接读取文件，绕过 tar 损坏
- **性能提升**：20-200x 加速

### 2. Pass 样本质量稳定 ✅
- **检查**：5% 随机采样（14,725 个样本）
- **问题率**：1.7% < 2% 阈值
- **结论**：无需全量检查 Pass 样本

### 3. DL3DV 特殊情况 ✅
- **Stage 1 Flag**：28.2%（主要是 caption 缺失）
- **Stage 2 问题率**：0%（视频质量很高）
- **处理**：Stage 3 用 Qwen 生成 caption

---

## ⏭️ 下一步

**优先级 1**（立即执行）：
```bash
# 生成 Stage 3 输入 manifest
cd stage2_result/qc_output_new
# 运行上面的 Python 脚本
```

**优先级 2**（准备测试）：
```bash
# 单样本测试
python scripts/run_stage3_single_sample.py --sample-id "xxx"

# 小批量测试（100 样本）
bash scripts/run_smoke_test_cmcc.sh
```

**优先级 3**（全量执行）：
```bash
# 48 GPU 并行（7-8 小时）
# 参考 STAGE3_完整打通方案_2026-08-04.md
```

---

**创建时间**：2026-08-07  
**状态**：✅ Stage 2 完成，可进入 Stage 3  
**下一步**：生成 manifest → 测试 → 全量执行

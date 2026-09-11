# 任务执行清单

## 任务 1: 清理废弃脚本 ✅

### 📋 完整分析报告
- 文档位置: `docs/DEPRECATED_SCRIPTS_LIST.md`
- 脚本总数: 13 个
- 待删除: 5 个

### 🗑️ 待删除文件清单（由你手动执行）

#### **确认错误（3个）**
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts

# 1. 最初版本 - /255归一化 + 简单平均
rm stage3_batch.py

# 2. 改进版本 - 仍用错误算法
rm stage3_batch_robust.py

# 3. 过时测试 - 使用错误实现
rm stage3_smoke_test.py
```

#### **重复造轮子（2个）**
```bash
# 4. 手动实现官方逻辑 - 可用官方接口替代
rm stage3_batch_fixed.py

# 5. 部分手动实现 - 节省70行代码
rm stage3_batch_official.py
```

#### **保守方案：先归档**
```bash
# 创建归档目录
mkdir -p ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)

# 移动而非删除
mv stage3_batch.py stage3_batch_robust.py stage3_smoke_test.py \
   stage3_batch_fixed.py stage3_batch_official.py \
   ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)/
```

### ✅ 保留文件（8个）
- **核心**: `stage3_batch_minimal.py` (唯一推荐)
- **CMCC**: `run_stage3_cmcc*.py` (3个)
- **测试**: `run_stage3_*.py` (3个)
- **工具**: `monitor_stage3.sh` (1个)

---

## 任务 2: 冒烟测试与分块对照实验 ✅

### 📝 测试配置

**测试视频**:
```
/mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00eb7564-d5e8-54a1-b8bd-52ab85334924.mp4
```

**对比方案**:
| 方案 | 分块时长 | 分辨率 | 脚本 |
|------|---------|--------|------|
| A | 2s (当前) | 720p 原始 | `stage3_batch_minimal.py` |
| B | 5s (论文) | 480p 降采样 | `stage3_test_5s.py` |

### 🚀 执行命令

#### **方案 A: 2s 分块（手动执行）**
```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建测试目录
TMP_DIR="/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_test"
mkdir -p "$TMP_DIR"
ln -sf /mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00eb7564-d5e8-54a1-b8bd-52ab85334924.mp4 "$TMP_DIR/test.mp4"

# 运行 2s 分块测试
python scripts/stage3_batch_minimal.py \
  --input_dir "$TMP_DIR" \
  --output /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl \
  --device cuda

# 查看结果
cat /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl
```

#### **方案 B: 5s 分块 + 降采样（手动执行）**
```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 运行 5s 分块测试
python scripts/stage3_test_5s.py \
  --video /mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001/00eb7564-d5e8-54a1-b8bd-52ab85334924.mp4 \
  --device cuda

# 查看结果
cat /mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl
```

#### **一键对比测试（推荐）**
```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_qc
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 运行完整对比实验
bash scripts/run_smoke_test_comparison.sh
```

### 📊 预期输出格式

**2s 分块结果** (`/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_2s.jsonl`):
```json
{
  "sample_id": "00eb7564-d5e8-54a1-b8bd-52ab85334924",
  "unimatch_flow": 12.345,
  "dover_tqe": 0.4567,
  "dover_aqe": 0.3456,
  "dover_fused": 0.6789,
  "verdict": "pass"
}
```

**5s 分块结果** (`/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_5s.jsonl`):
```json
{
  "sample_id": "00eb7564-d5e8-54a1-b8bd-52ab85334924",
  "config": "5s_downsampled_480p",
  "unimatch_flow": 12.456,
  "dover_tqe": 0.4321,
  "dover_aqe": 0.3210,
  "dover_fused": 0.6543,
  "verdict": "pass"
}
```

### 📈 分析维度

对比脚本会自动分析：

1. **DOVER 分数变化**
   - TQE/AQE/Fused 三个维度
   - 绝对差异 + 相对百分比

2. **UniMatch 分数变化**
   - 光流幅值差异

3. **影响因素识别**
   - 分块时长影响（2s vs 5s）
   - 降采样影响（720p vs 480p）
   - 判断哪个因素影响更大

4. **建议输出**
   - 如果 DOVER 差异 > 0.05: 建议调整
   - 如果 DOVER 差异 < 0.05: 两种方案均可

### 🎯 预期结论模板

```
影响分析:
----------------------------------------------------------------------
DOVER 分数变化:   +0.0234 (+3.5%)
UniMatch 分数变化: +0.111 (+0.9%)

结论:
✅ DOVER 受影响更大
   降采样 降低了 DOVER 分数（信息损失）

⚠️ DOVER 差异显著 (>0.05)，建议:
   - 降采样损失过大，考虑提高到540p或600p
```

---

## 🔧 创建的脚本文件

| 文件 | 用途 |
|------|------|
| `scripts/stage3_test_5s.py` | 5s分块+降采样测试脚本 |
| `scripts/smoke_test_2s.sh` | 2s分块单独测试脚本 |
| `scripts/run_smoke_test_comparison.sh` | 一键对比测试脚本 |
| `docs/DEPRECATED_SCRIPTS_LIST.md` | 废弃脚本分析报告 |

---

## ✅ 执行检查清单

### **任务 1: 清理废弃脚本**
- [ ] 阅读 `docs/DEPRECATED_SCRIPTS_LIST.md`
- [ ] 确认 5 个待删除文件
- [ ] 执行删除或归档命令
- [ ] 验证: `ls scripts/*stage3* | wc -l` 应为 8

### **任务 2: 冒烟测试**
- [ ] 执行 `bash scripts/run_smoke_test_comparison.sh`
- [ ] 等待测试完成（约 2-3 分钟）
- [ ] 查看对比分析结果
- [ ] 根据结论决定：
  - [ ] 保持 2s 分块（如果差异 < 0.05）
  - [ ] 切换 5s 分块（如果效果更好）
  - [ ] 调整降采样分辨率（如果 480p 损失过大）

---

## 📝 注意事项

1. **删除前备份**: 使用归档方案更安全
2. **测试前激活环境**: `conda activate sana_qc`
3. **结果文件位置**: `/mnt/afs/davidwang/workspace/data/spatialvid_001/tmp/stage3_smoke_*.jsonl`
4. **GPU 显存**: 5s分块已有降采样保护，不会OOM

---

## 🎯 下一步行动

**完成冒烟测试后**:
1. 根据对比结果选择最优方案
2. 如需调整降采样分辨率，修改 `stage3_test_5s.py` 中的 `480` 为 `540` 或 `600`
3. 重新验证后，启动 5000 视频全量任务

**启动全量任务命令**:
```bash
# 使用最优方案的脚本
nohup python scripts/stage3_batch_minimal.py \
  --input_dir /mnt/afs/davidwang/workspace/data/spatialvid_001/videos/SpatialVID/videos/group_0001 \
  --output /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_results_final.jsonl \
  --resume \
  > /mnt/afs/davidwang/workspace/data/spatialvid_001/stage3_final.nohup 2>&1 &
```

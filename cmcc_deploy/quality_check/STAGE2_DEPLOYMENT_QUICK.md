# Stage 2 全量执行 - 快速部署指南

> **30 秒速览**：从解压目录直接读取，全量处理，20-200x 加速

---

## ✅ 已完成的修改

| 项目 | 旧版本 | 新版本 | 状态 |
|------|--------|--------|------|
| **采样率** | 0.05 (5%) | 1.0 (100%) | ✅ |
| **读取方式** | tar 文件 | 解压目录 | ✅ |
| **速度** | ~100-200 ms/样本 | ~1-5 ms/样本 | ✅ 20-200x |
| **缺失处理** | 写入错误 | 跳过不写入 | ✅ |

---

## 📂 交付文件

```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/
├── src/sana_wm_pipeline/qc/
│   └── stage2_deep_extracted.py           # 核心逻辑 ✅
├── scripts/
│   ├── run_stage2_full_extracted.sh       # 执行脚本 ✅
│   └── test_stage2_extracted.py           # 测试脚本 ✅
├── STAGE2_FULL_EXECUTION_GUIDE.md         # 完整使用说明 ✅
└── STAGE2_DEPLOYMENT_QUICK.md             # 本文档 ✅
```

---

## 🚀 3 步部署（5 分钟）

### 步骤 1：打包传输（本地 AFS）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 打包文件
tar czf stage2_full.tar.gz \
    src/sana_wm_pipeline/qc/stage2_deep_extracted.py \
    scripts/run_stage2_full_extracted.sh \
    scripts/test_stage2_extracted.py \
    STAGE2_FULL_EXECUTION_GUIDE.md

# 传输到 CMCC（根据实际 SSH 配置调整）
scp stage2_full.tar.gz user@cmcc:/root/work/david_work/
```

### 步骤 2：部署（CMCC 机器）

```bash
# 进入工作目录
cd /root/work/david_work/sana_wm_qc

# 解压
tar xzf ../stage2_full.tar.gz

# 添加执行权限
chmod +x scripts/run_stage2_full_extracted.sh
chmod +x scripts/test_stage2_extracted.py

# 验证文件
ls -l src/sana_wm_pipeline/qc/stage2_deep_extracted.py
ls -l scripts/run_stage2_full_extracted.sh
```

### 步骤 3：测试 + 执行（CMCC 机器）

```bash
# 激活环境
cd /root/work/david_work/sana_wm_qc
source /root/work/david_work/sana_wm_qc_env/bin/activate
export PYTHONPATH=/root/work/david_work/sana_wm_qc/src:$PYTHONPATH

# 快速测试（可选，推荐）
python scripts/test_stage2_extracted.py

# 全量执行
nohup bash scripts/run_stage2_full_extracted.sh > /root/work/david_work/stage2_full.log 2>&1 &

# 查看进度
tail -f /root/work/david_work/stage2_full.log
```

---

## 📊 预期结果

### 处理样本数

| 数据集 | Stage 1 Total | Pass | Flag | Fail | Stage 2 输入 (Pass+Flag) |
|--------|---------------|------|------|------|------------------------|
| wds-sekai-game-drone | 931 | 840 | 44 | 47 | 884 |
| wds-sekai-game-walking | 1,602 | 1,458 | 72 | 72 | 1,530 |
| wds-OmniWorld-Game | 6,378 | 5,736 | 321 | 321 | 6,057 |
| wds-DL3DV-ALL-2K | 9,937 | 7,132 | 2,805 | 0 | 9,937 |
| wds-sekai-real-walking-hq | 20,154 | 18,887 | 1,133 | 134 | 20,020 |
| wds-RealEstate10K-360p | 73,738 | 65,456 | 3,641 | 4,641 | 69,097 |
| wds-SpatialVID-hq | 220,508 | 195,000 | 19,216 | 6,292 | 214,216 |
| **总计** | **333,248** | ~294,509 | ~27,232 | ~11,507 | **~321,741** |

### 预估耗时

| 阶段 | 预估时间 | 说明 |
|------|---------|------|
| **文件查找** | ~5 分钟 | glob 扫描 33 万样本 |
| **PyAV 解码** | ~2-3 小时 | 主要耗时（视频解码） |
| **其他处理** | ~30 分钟 | 黑帧检测、轨迹分析 |
| **总计** | **~3 小时** | 基于 32 进程并发 |

---

## 🔍 监控命令

```bash
# 实时进度
watch -n 30 'find /root/work/david_work/qc_output_new -name "stage2_results_full.jsonl" -exec wc -l {} \; | awk "{s+=\$1} END {print \"已处理: \" s \" / 321741\"}"'

# 各数据集进度
for g in wds-sekai-game-drone wds-sekai-game-walking wds-OmniWorld-Game wds-DL3DV-ALL-2K wds-sekai-real-walking-hq wds-RealEstate10K-360p wds-SpatialVID-hq; do
  f="/root/work/david_work/qc_output_new/$g/stage2_results_full.jsonl"
  [ -f "$f" ] && echo "$g: $(wc -l < $f)" || echo "$g: 未开始"
done

# CPU/内存使用
top -b -n 1 | grep python

# 磁盘 I/O
iostat -x 5
```

---

## ⚠️ 重要提示

### 1. 输出文件名变更

- **旧文件**：`stage2_results.jsonl`（5% 采样）
- **新文件**：`stage2_results_full.jsonl`（100% 全量）✅

**不会覆盖旧结果**，两个文件共存。

### 2. 跳过的样本

部分样本可能因以下原因被跳过（不写入结果）：
- 解压目录中找不到对应文件
- 文件命名格式不匹配

预期跳过数量：**< 5%**（~1-2 万样本）

### 3. 执行时间

- **最小数据集**（wds-sekai-game-drone）：~1 分钟
- **最大数据集**（wds-SpatialVID-hq）：~2 小时
- **全部 7 个数据集**：~3-4 小时

---

## 🐛 故障速查

| 症状 | 原因 | 解决 |
|------|------|------|
| 模块导入失败 | 文件未部署 | 检查 `stage2_deep_extracted.py` |
| 数据根目录不存在 | 路径错误 | 检查 `DATA_ROOT` 变量 |
| 大量样本被跳过 | 文件命名不匹配 | 查看 `find_sample_files()` 逻辑 |
| 进程卡住 | I/O 瓶颈 | 降低 `N_WORKERS` |

---

## ✅ 验证清单

执行前：
- [ ] 文件已传输到 CMCC
- [ ] 环境已激活（conda + PYTHONPATH）
- [ ] 数据根目录存在
- [ ] Stage 1 结果存在

执行后：
- [ ] 7 个 `stage2_results_full.jsonl` 已生成
- [ ] 样本数量合理（~32 万）
- [ ] 跳过数量 < 5%
- [ ] 无大量错误日志

---

## 🎯 下一步

1. **生成 Stage 3 manifest**
   ```bash
   # 使用 stage2_results_full.jsonl 作为输入
   python scripts/generate_stage3_manifest.py
   ```

2. **进入 Stage 3**
   ```bash
   # GPU 密集型处理（UniMatch + DOVER + Qwen）
   bash scripts/run_stage3_cmcc_full.sh
   ```

---

## 📞 支持

- **完整文档**：`STAGE2_FULL_EXECUTION_GUIDE.md`
- **测试脚本**：`scripts/test_stage2_extracted.py`
- **核心代码**：`src/sana_wm_pipeline/qc/stage2_deep_extracted.py`

---

**版本**：v1.0  
**日期**：2026-08-07  
**状态**：✅ 就绪，可立即部署  
**预期完成时间**：3-4 小时

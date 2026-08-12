# CMCC 训练数据提取 - 完整部署指南

> **文档版本**：v1.0  
> **创建日期**：2026-08-01  
> **适用环境**：CMCC 生产服务器

---

## 📋 任务概述

从 CMCC 原始 WDS tar 分片数据集中，按照人工反馈筛选结果，依次提取三个版本的训练数据。

### 三个版本

| 版本 | 样本数 | 评级标准 | 数据集 | 用途 |
|------|--------|---------|--------|------|
| **v1.0** | 1,980 | excellent + good | 3 个真实场景 | 高质量基线 |
| **v1.1** | 2,651 | + acceptable | 3 个真实场景 | 鲁棒性提升 |
| **v1.2** | 4,720 | + acceptable | 3 真实 + DL3DV | 规模扩展 |

---

## 🗂️ 目录结构说明

### CMCC 原始数据结构

```
/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/
├── final_wds-RealEstate10K-360p/
│   └── wds-RealEstate10K-360p/
│       ├── w000/
│       │   ├── shard-0000.tar
│       │   ├── shard-0001.tar
│       │   └── ...
│       ├── w001/
│       └── ...
├── final_wds-SpatialVID-hq/
│   └── wds-SpatialVID-hq/
│       └── w000/
├── final_wds-sekai-real-walking-hq/
│   └── wds-sekai-real-walking-hq/
│       └── w000/
└── final_wds-DL3DV-ALL-2K/
    └── wds-DL3DV-ALL-2K/
        └── w000/
```

### 输出目录结构

```
/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/
├── v1.0/                                    # v1.0 提取数据
│   ├── RealEstate10K-360p_train__xxx.video.mp4
│   ├── RealEstate10K-360p_train__xxx.poses_c2w.npy
│   ├── RealEstate10K-360p_train__xxx.intrinsics.npy
│   ├── RealEstate10K-360p_train__xxx.scale.npy
│   ├── RealEstate10K-360p_train__xxx.caption.txt
│   ├── ...
│   ├── extraction_v1.0.log
│   ├── extraction_report.txt
│   └── missing_samples.jsonl (如有)
├── v1.1/                                    # v1.1 提取数据
├── v1.2/                                    # v1.2 提取数据
├── v1.0_20260801_143022.tar.gz             # v1.0 归档
├── v1.0_20260801_143022.tar.gz.md5
├── v1.1_20260801_150315.tar.gz             # v1.1 归档
├── v1.1_20260801_150315.tar.gz.md5
├── v1.2_20260801_153142.tar.gz             # v1.2 归档
├── v1.2_20260801_153142.tar.gz.md5
├── extraction_pipeline.log                  # 总执行日志
└── EXTRACTION_FINAL_REPORT.txt             # 最终报告
```

---

## 🚀 快速部署（5 步完成）

### Step 1: 准备筛选列表文件

**在 AFS 开发环境**执行：

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback

# 验证三个筛选列表存在
ls -lh filtered_training_samples*.jsonl

# 输出应包含：
#   filtered_training_samples.jsonl (v1.0)
#   filtered_training_samples_v1.1_with_acceptable.jsonl
#   filtered_training_samples_v1.2_with_dl3dv.jsonl
```

### Step 2: 打包脚本和筛选列表

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建部署包
mkdir -p /tmp/cmcc_deployment
cp scripts/extract_training_data_from_filtered_corrected.py /tmp/cmcc_deployment/
cp scripts/run_all_versions_extraction.sh /tmp/cmcc_deployment/
cp sana-human-feedback/filtered_training_samples*.jsonl /tmp/cmcc_deployment/

# 打包
cd /tmp
tar -czf cmcc_deployment.tar.gz cmcc_deployment/

# 查看打包结果
ls -lh cmcc_deployment.tar.gz
```

### Step 3: 传输到 CMCC 服务器

```bash
# 方法 1：使用 scp
scp cmcc_deployment.tar.gz cmcc_server:/root/work/david_work/

# 方法 2：使用 rsync
rsync -avz --progress cmcc_deployment.tar.gz cmcc_server:/root/work/david_work/
```

### Step 4: 在 CMCC 服务器上部署

**登录 CMCC 服务器**：

```bash
ssh cmcc_server
```

**解压部署包**：

```bash
cd /root/work/david_work
tar -xzf cmcc_deployment.tar.gz
cd cmcc_deployment

# 验证文件完整性
ls -lh
# 应包含：
#   extract_training_data_from_filtered_corrected.py
#   run_all_versions_extraction.sh
#   filtered_training_samples.jsonl
#   filtered_training_samples_v1.1_with_acceptable.jsonl
#   filtered_training_samples_v1.2_with_dl3dv.jsonl
```

**修改脚本配置**（重要！）：

```bash
vim run_all_versions_extraction.sh

# 修改以下路径（约在第 25-35 行）：
FILTERED_V1_0="/root/work/david_work/cmcc_deployment/filtered_training_samples.jsonl"
FILTERED_V1_1="/root/work/david_work/cmcc_deployment/filtered_training_samples_v1.1_with_acceptable.jsonl"
FILTERED_V1_2="/root/work/david_work/cmcc_deployment/filtered_training_samples_v1.2_with_dl3dv.jsonl"

DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"
OUTPUT_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output"

EXTRACT_SCRIPT="/root/work/david_work/cmcc_deployment/extract_training_data_from_filtered_corrected.py"

# 保存退出（:wq）
```

**赋予执行权限**：

```bash
chmod +x run_all_versions_extraction.sh
chmod +x extract_training_data_from_filtered_corrected.py
```

### Step 5: 执行提取任务

#### 测试模式（推荐先运行）

```bash
# 仅测试 v1.0，不实际提取
python3 extract_training_data_from_filtered_corrected.py \
  --filtered_list filtered_training_samples.jsonl \
  --data_root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output_dir /tmp/test_output \
  --dry_run

# 查看测试结果
cat /tmp/test_output/extraction.log
```

#### 正式执行（后台运行）

```bash
# 使用 nohup 后台运行，防止 SSH 断开中断
nohup bash run_all_versions_extraction.sh > run.log 2>&1 &

# 记录进程 ID
echo $! > extraction.pid

# 查看实时日志
tail -f run.log

# 或者查看详细执行日志
tail -f /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/extraction_pipeline.log
```

---

## 📊 监控与验证

### 监控执行进度

```bash
# 查看主日志
tail -f /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/extraction_pipeline.log

# 查看当前正在执行的版本日志
tail -f /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0/extraction_v1.0.log

# 查看进程状态
ps aux | grep run_all_versions_extraction

# 查看已提取文件数
watch -n 10 'find /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0 -type f | wc -l'
```

### 验证提取结果

```bash
cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output

# 查看最终报告
cat EXTRACTION_FINAL_REPORT.txt

# 验证各版本文件数
for v in v1.0 v1.1 v1.2; do
    echo "版本 $v:"
    find $v -type f ! -name "*.log" ! -name "*.txt" ! -name "*.jsonl" | wc -l
done

# 预期输出：
# 版本 v1.0: 9900 (1980 × 5)
# 版本 v1.1: 13255 (2651 × 5)
# 版本 v1.2: 23600 (4720 × 5)

# 验证归档文件完整性
for f in *.tar.gz; do
    echo "验证 $f..."
    md5sum -c ${f}.md5
done

# 查看目录大小
du -sh v1.0 v1.1 v1.2
```

---

## ⚠️ 常见问题与排查

### 问题 1：找不到样本文件

**症状**：
```
⚠️  无法识别数据集：DL3DV-ALL-2K_xxx
missing_samples.jsonl 中有大量样本
```

**原因**：数据集路径映射不正确

**解决**：
```bash
# 检查实际目录结构
ls /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output/

# 如果目录名不匹配，修改脚本中的 DATASET_MAPPING
vim extract_training_data_from_filtered_corrected.py

# 修改约第 38-43 行的路径映射
```

### 问题 2：权限不足

**症状**：
```
Permission denied: /root/work/filestorage/...
```

**解决**：
```bash
# 检查目录权限
ls -ld /root/work/filestorage/shangaoooooo/davidwang/

# 如果需要，调整权限
chmod 755 /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output
```

### 问题 3：磁盘空间不足

**症状**：
```
No space left on device
```

**解决**：
```bash
# 检查磁盘空间
df -h /root/work/filestorage/

# 预估空间需求：
#   v1.0: ~99 GB
#   v1.1: ~133 GB
#   v1.2: ~236 GB
#   归档: ~468 GB (压缩后约 300-400 GB)
#   总计: ~900 GB

# 如果空间不足，清理临时文件或调整输出路径
```

### 问题 4：执行中断

**症状**：SSH 断开或手动 Ctrl+C

**恢复方法**：
```bash
# 查看已完成的版本
ls /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/

# 修改脚本，跳过已完成的版本
vim run_all_versions_extraction.sh

# 注释掉已完成的版本提取代码，然后重新运行
```

---

## 🔧 高级配置

### 并行提取（不推荐，仅限高性能服务器）

如果服务器 I/O 性能足够强（NVMe SSD + 多核 CPU），可以修改脚本实现并行提取：

```bash
# 修改 run_all_versions_extraction.sh，将串行改为并行
extract_version "v1.0" ... &
extract_version "v1.1" ... &
extract_version "v1.2" ... &
wait  # 等待所有任务完成
```

**注意**：并行会显著增加 I/O 压力，可能导致：
- 磁盘性能下降
- 系统负载过高
- 提取时间反而增加

### 自定义提取子集

如果只需要提取部分样本（测试用）：

```bash
# 提取前 100 个样本测试
head -n 100 filtered_training_samples.jsonl > test_100_samples.jsonl

# 使用测试列表
python3 extract_training_data_from_filtered_corrected.py \
  --filtered_list test_100_samples.jsonl \
  --data_root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output \
  --output_dir /tmp/test_100_output
```

---

## 📈 性能预估

### 提取速度

**影响因素**：
- 磁盘 I/O 速度（读取 tar + 写入提取文件）
- CPU 性能（tar 解压缩）
- 网络速度（如果跨挂载点）

**预估时间**（基于 HDD 7200rpm）：
- v1.0 (1,980 样本)：约 1-2 小时
- v1.1 (2,651 样本)：约 1.5-3 小时
- v1.2 (4,720 样本)：约 2.5-5 小时
- **总计**：约 5-10 小时

**预估时间**（基于 SSD）：
- v1.0：约 20-40 分钟
- v1.1：约 30-60 分钟
- v1.2：约 45-90 分钟
- **总计**：约 2-3 小时

### 存储空间

| 版本 | 原始文件 | 归档文件 | 总计 |
|------|---------|---------|------|
| v1.0 | ~99 GB | ~70 GB | ~169 GB |
| v1.1 | ~133 GB | ~95 GB | ~228 GB |
| v1.2 | ~236 GB | ~170 GB | ~406 GB |
| **总计** | **~468 GB** | **~335 GB** | **~803 GB** |

**建议预留**：1 TB 空间

---

## ✅ 完成检查清单

- [ ] AFS 环境准备完成（筛选列表文件存在）
- [ ] 脚本打包完成
- [ ] 传输到 CMCC 服务器完成
- [ ] 脚本配置修改完成（路径正确）
- [ ] 测试模式运行成功
- [ ] 正式提取任务启动
- [ ] v1.0 提取完成并归档
- [ ] v1.1 提取完成并归档
- [ ] v1.2 提取完成并归档
- [ ] 验证文件数量正确
- [ ] 验证归档完整性（MD5）
- [ ] 生成最终报告
- [ ] 备份到其他存储位置

---

## 📞 联系与支持

**技术负责人**：David Wang  
**文档版本**：v1.0  
**最后更新**：2026-08-01

**相关文档**：
- `FILTERING_GUIDE.md` - 数据筛选完整指南
- `TASK_SUMMARY.md` - 任务快速总结
- `FILTERING_REPORT_v1.2.md` - v1.2 详细报告

---

## 🎉 快速命令速查卡

```bash
# === 部署阶段 ===
# 1. 打包（AFS）
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
tar -czf /tmp/cmcc_deployment.tar.gz scripts/*.py scripts/*.sh sana-human-feedback/filtered_*.jsonl

# 2. 传输
scp /tmp/cmcc_deployment.tar.gz cmcc_server:/root/work/david_work/

# 3. 部署（CMCC）
cd /root/work/david_work && tar -xzf cmcc_deployment.tar.gz && cd cmcc_deployment
chmod +x *.sh *.py

# === 执行阶段 ===
# 测试
python3 extract_training_data_from_filtered_corrected.py --filtered_list filtered_training_samples.jsonl --data_root /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output --output_dir /tmp/test --dry_run

# 正式运行
nohup bash run_all_versions_extraction.sh > run.log 2>&1 &

# === 监控阶段 ===
# 查看日志
tail -f run.log
tail -f /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/extraction_pipeline.log

# 查看进度
watch -n 10 'find /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output/v1.0 -type f | wc -l'

# === 验证阶段 ===
cd /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output
cat EXTRACTION_FINAL_REPORT.txt
for f in *.tar.gz; do md5sum -c ${f}.md5; done
```

---

**祝提取顺利！** 🚀

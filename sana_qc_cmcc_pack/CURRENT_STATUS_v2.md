# SANA QC 系统状态总览

**最后更新：** 2026-07-02  
**会话主题：** QC系统性能优化与全量数据处理

---

## 🎯 核心目标

1. ✅ **整理 sana_qc_cmcc_pack 文档**（已完成）
2. 🔄 **优化 Stage 3 性能**（进行中）
3. 🔄 **执行全量数据 Stage 1+2**（进行中）

---

## 📊 项目背景

### 数据规模
- **根目录**：`/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/`
- **总样本数**：~140,000 个
- **数据量**：1.1 TB
- **Group 数量**：7 个
- **Tar 文件数**：1,178 个

### Group 分布
| Group | Tar数量 | 优先级 |
|-------|---------|--------|
| RealEstate10K-360p | 463 | 高 |
| SpatialVID-hq | 256 | 高 |
| sekai-real-walking-hq | 240 | 中 |
| DL3DV-ALL-2K | 87 | 中 |
| OmniWorld-Game | 80 | 中 |
| sekai-game-walking | 43 | 低 |
| sekai-game-drone | 9 | 低 (已验证) |

---

## 🔍 关键发现

### 发现 1：Stage 3 性能瓶颈定位 ✅

**问题：** 用户报告"8小时处理100个样本"

**调查结果：**
1. 实际情况：进程运行6.5天，输出文件0字节
2. 终端显示"200 samples done"是**误导性计数**（读取行数，非处理数量）
3. 实际处理速度：**~3.3分钟/样本**（227秒 → 167秒平均197秒）

**根本原因：**
- **Qwen VLM 推理**是主要瓶颈（27B模型，8个关键帧）
- **DOVER 被强制在CPU上运行**（H100 兼容性问题导致 coredump）
- 单样本串行处理：UniMatch (GPU) → DOVER (CPU) → Qwen (GPU)

---

### 发现 2：不必要的 VLM 调用 ✅

**问题：** 所有样本都调用 Qwen，即使某些 source 不需要 VLM 指标

**Table 6 配置分析：**
- **不需要 VLM**：DL3DV, RealEstate10K (`vlm_entity: null`, `vlm_quality: null`)
- **需要 VLM**：其他5个 source

**影响：**
- DL3DV 和 RealEstate10K 浪费 ~70% 时间在不必要的 VLM 推理上

---

### 发现 3：损坏的 Tar 文件处理 ✅

**问题：** 数据生成过程中机器故障导致部分 tar 文件损坏

**原始行为：**
- 遇到损坏 tar → 整个进程崩溃
- 即使 tar 中有 200 个好样本，也会全部丢失

**用户需求：**
- 部分损坏的 tar 文件中仍有大量可用样本
- 需要尽可能恢复所有可用数据

---

### 发现 4：DOVER 部署问题（历史问题）✅

**问题：** CMCC 机器无法访问 GitHub

**解决方案：**
- v3 脚本改为从本地路径安装：`pip install -e /root/work/david_work/sana_qc_pipeline/DOVER`
- DOVER 目录已手动传输到目标机器

---

## ✅ 已完成的优化

### 1. Stage 3 条件跳过 VLM（性能优化）

**文件：** `src/sana_wm_pipeline/qc/stage3_gpu.py`

**修改内容：**
```python
# 检查 table6 配置，判断是否需要 VLM
need_vlm = False
if cfg.table6_source is not None:
    source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
    need_vlm = (source_cfg.get("vlm_entity") is not None or
                source_cfg.get("vlm_quality") is not None or
                has_camera_words)

# 只有需要时才调用 Qwen
if need_vlm:
    # ... Qwen 推理 ...
else:
    stage3["reasons"].append("vlm_skipped: not needed for this source")
```

**效果：**
- DL3DV/RealEstate10K：从 ~200秒/样本 降到 ~70秒/样本
- **节省 70% 时间**

---

### 2. Stage 3 详细日志（调试工具）

**文件：** `scripts/run_stage3_cmcc_debug.py`

**新增功能：**
- 单样本计时（START/DONE 日志）
- 实时进度统计（处理数量、平均耗时、总时间）
- 立即刷新输出文件（`fout.flush()`）

**输出示例：**
```
[worker 0] [1] START processing sample: xxx
[worker 0] [1] DONE sample: xxx | time: 45.3s | avg: 45.3s/sample | total: 0.8min
```

---

### 3. Stage 1 损坏 Tar 部分恢复（容错优化）

**文件：** `src/sana_wm_pipeline/qc/stage1_fast.py`

**修改内容：**
- 从 `getmembers()` 改为 `next()` 逐个读取成员
- 三层容错机制：
  1. Tar 打开失败 → 返回空
  2. 读取成员时遇到 EOF → 保留已读数据
  3. 单个文件提取失败 → 跳过该文件继续

**效果：**
```
优化前：遇到损坏 tar → 损失整个 tar 的所有样本
优化后：恢复损坏前的所有完整样本

示例输出：
WARNING: Corruption in shard-000048.tar after reading 654 members: unexpected end of data
INFO: Recovered 200 samples from shard-000048.tar (read 654 members)
```

---

### 4. 文档整理（CURRENT_STATUS.md v1）

**文件：** `sana_qc_cmcc_pack/CURRENT_STATUS.md`

**内容：**
- 当前可用资源（打包文件、脚本版本）
- 已知问题与解决方案（DOVER 包混淆、初始化错误等）
- 下一步行动清单（优先级：高/中/低）
- 关键决策速查表（Q&A 格式）

**评估结论：**
- ❌ 不需要重新打包 conda 环境和代码包
- ✅ 只需传输：v3 脚本（19KB）+ DOVER 目录（347MB）

---

## 🔄 进行中的任务

### 1. Stage 1+2 全量数据处理

**当前进度：**
- ✅ sekai-game-drone（9 shards）验证成功
- 🔄 准备批量执行其余 6 个 group

**执行计划：**
1. RealEstate10K（463 shards，~2.5小时）
2. SpatialVID（256 shards，~1.5小时）
3. sekai-real-walking-hq（240 shards，~1.5小时）
4. DL3DV（87 shards，~30分钟）
5. OmniWorld（80 shards，~30分钟）
6. sekai-game-walking（43 shards，~20分钟）

**总预计时间：** ~6.5 小时

**输出：**
- 每个 group 的 `stage1_results.jsonl` 和 `stage2_results.jsonl`
- 每个 group 的 `report.html` 和 manifest

---

### 2. Stage 3 优化代码部署

**待传输文件：**
1. ✅ `src/sana_wm_pipeline/qc/stage3_gpu.py`（VLM 条件跳过）
2. ✅ `scripts/run_stage3_cmcc_debug.py`（详细日志）
3. ✅ `src/sana_wm_pipeline/qc/stage1_fast.py`（部分恢复损坏 tar）

**部署状态：**
- 文件已在源机器准备好
- 等待用户传输到 CMCC

---

## ⚠️ 待解决问题

### 问题 1：Stage 3 是否必需？

**背景：**
- Stage 1+2 已经能筛掉大部分坏样本
- Stage 3 主要提供额外的质量分数（UniMatch flow, DOVER, VLM）

**决策点：**
- 如果 Stage 1+2 的筛选已经足够严格 → 可以跳过 Stage 3
- 如果需要 Table 6 的质量评分 → 必须跑 Stage 3

**推荐：**
- 先完成 Stage 1+2 全量
- 查看报告，评估 pass 样本的质量
- 再决定是否值得投入 GPU 资源跑 Stage 3

---

### 问题 2：Stage 3 的全量预估

**基于冒烟测试数据：**
- 单样本平均耗时：197秒（优化前）→ ~70秒（优化后，DL3DV/RealEstate）
- 假设全量 7% pass rate
- 140,000 × 7% = 9,800 个样本需要 Stage 3

**预计时间（48 H100 并行）：**
- 优化前：9,800 × 197秒 / 60 / 48 ≈ **67 小时**
- 优化后（混合）：约 **40-50 小时**（取决于各 source 的 pass 比例）

**建议：**
- 分 source 逐步执行
- 优先处理不需要 VLM 的 source（DL3DV, RealEstate10K）
- 根据结果决定是否继续其他 source

---

### 问题 3：DOVER H100 兼容性

**当前状态：**
- DOVER 在 H100 上会 coredump
- 被强制使用 CPU（性能损失）

**影响：**
- 单样本 DOVER 耗时约 50 秒（CPU）
- 如果能用 GPU，预计可降到 5-10 秒

**可能方案：**
1. 升级 PyTorch/CUDA 版本
2. 降级到 A100/V100 测试
3. 寻找 DOVER 的替代模型
4. 接受 CPU 运行的现状

**当前决策：** 暂时接受 CPU 运行

---

## 📝 关键决策记录

### 决策 1：不重新打包环境 ✅

**理由：**
- conda 环境虽可能包含错误的 dover CLI 工具
- v3 脚本会从本地路径安装正确版本（`pip install -e $DOVER_DIR`）
- 完全不依赖网络，适合 CMCC 隔离环境
- 避免传输 3.7GB 环境包

---

### 决策 2：先运行 Stage 1+2 全量 ✅

**理由：**
1. 快速获得可交付的初步报告（目标 A）
2. Stage 1+2 能筛掉大部分坏样本
3. 评估是否值得跑 Stage 3（目标 B）
4. CPU 密集型，4-10 小时完成
5. 不需要等待 Stage 3 优化完成

---

### 决策 3：优化 Stage 3 VLM 调用 ✅

**理由：**
- DL3DV 和 RealEstate10K 不需要 VLM 指标
- 节省 70% 时间（从 200秒降到 70秒）
- 代码改动小，风险低
- 向后兼容（其他 source 行为不变）

---

### 决策 4：部分恢复损坏 Tar ✅

**理由：**
- 直接跳过整个 tar 浪费数据
- 部分损坏的 tar 中仍有大量可用样本
- 使用 `next()` 逐个读取可以恢复损坏前的数据
- 三层容错机制确保最大程度恢复

---

## 🚀 执行路线图

### 短期（当前）

1. ✅ **验证 Stage 1+2 流程**（sekai-game-drone，5分钟）
2. 🔄 **传输优化后的代码到 CMCC**
   - `stage1_fast.py`（损坏 tar 恢复）
   - `stage3_gpu.py`（VLM 条件跳过）
   - `run_stage3_cmcc_debug.py`（详细日志）
3. 🔄 **批量执行 Stage 1+2**（6个 group，~6.5小时）

---

### 中期（Stage 1+2 完成后）

4. 📊 **分析 Stage 1+2 报告**
   - 各 group 的 pass/fail 比例
   - flag 原因分布
   - 评估数据质量

5. 🎯 **决策：是否执行 Stage 3**
   - 如果 Stage 1+2 已足够 → 交付结果
   - 如果需要质量分数 → 继续 Stage 3

---

### 长期（如果执行 Stage 3）

6. 🔄 **分 source 执行 Stage 3**
   - 优先：DL3DV, RealEstate10K（不需要 VLM，快）
   - 次之：其他 5 个 source（需要 VLM，慢）

7. 📊 **汇总最终报告**
   - 合并所有 group 的结果
   - 生成全量质量报告

---

## 📂 需要传输到 CMCC 的文件

### 优先级：高（Stage 1+2 必需）

#### `src/sana_wm_pipeline/qc/stage1_fast.py`
**功能：** 部分恢复损坏的 tar 文件  
**源路径：** `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage1_fast.py`  
**目标路径：** `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/qc/stage1_fast.py`  
**大小：** ~5KB

---

### 优先级：中（Stage 3 优化）

#### `src/sana_wm_pipeline/qc/stage3_gpu.py`
**功能：** 条件跳过 VLM（节省70%时间）  
**源路径：** `/mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/stage3_gpu.py`  
**目标路径：** `/root/work/david_work/sana_wm_qc/src/sana_wm_pipeline/qc/stage3_gpu.py`  
**备份：** `stage3_gpu.py.backup`  
**大小：** ~15KB

#### `scripts/run_stage3_cmcc_debug.py`
**功能：** 详细日志和计时  
**源路径：** `/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_cmcc_debug.py`  
**目标路径：** `/root/work/david_work/sana_wm_qc/scripts/run_stage3_cmcc_debug.py`  
**大小：** ~4KB

---

## 📋 Stage 1+2 批量执行命令

```bash
cd /root/work/david_work/sana_wm_qc

# 1. RealEstate10K (463 shards, ~2.5h)
python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-RealEstate10K-360p/wds-RealEstate10K-360p \
  --group RealEstate10K \
  --output-dir /root/work/david_work/qc_output/full_RealEstate10K \
  --n-workers 32 \
  --read-video-frames

# 2. SpatialVID (256 shards, ~1.5h)
python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-SpatialVID-hq/wds-SpatialVID-hq \
  --group SpatialVID \
  --output-dir /root/work/david_work/qc_output/full_SpatialVID \
  --n-workers 32 \
  --read-video-frames

# 3. sekai-real-walking-hq (240 shards, ~1.5h)
python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-real-walking-hq/wds-sekai-real-walking-hq \
  --group Sekai_Walking \
  --output-dir /root/work/david_work/qc_output/full_sekai_real_walking \
  --n-workers 32 \
  --read-video-frames

# 4. DL3DV (87 shards, ~30min)
python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-DL3DV-ALL-2K/wds-DL3DV-ALL-2K \
  --group DL3DV \
  --output-dir /root/work/david_work/qc_output/full_DL3DV \
  --n-workers 32 \
  --read-video-frames

# 5. OmniWorld (80 shards, ~30min)
python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-OmniWorld-Game/wds-OmniWorld-Game \
  --group OmniWorld \
  --output-dir /root/work/david_work/qc_output/full_OmniWorld \
  --n-workers 32 \
  --read-video-frames

# 6. sekai-game-walking (43 shards, ~20min)
python scripts/run_qc.py \
  --tar-root /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-sekai-game-walking/wds-sekai-game-walking \
  --group Sekai_Game_Walking \
  --output-dir /root/work/david_work/qc_output/full_sekai_game_walking \
  --n-workers 32 \
  --read-video-frames

# 7. sekai-game-drone (9 shards, ~5min) - 已验证完成 ✅
```

---

## 🔧 技术细节速查

### Stage 1+2 特点
- **CPU 密集型**，32 进程并行
- **快速**：~10秒/shard
- **筛选项**：
  - Stage 1: 11 项检查（文件完整性、轨迹、caption等）
  - Stage 2: 4 项深度检测（黑帧、场景切换、轨迹冻结）

### Stage 3 特点
- **GPU 密集型**，单样本串行
- **慢**：
  - 优化前：~200秒/样本
  - 优化后（无VLM）：~70秒/样本
  - 优化后（有VLM）：~200秒/样本
- **评估项**：
  - UniMatch flow（光流幅度）
  - DOVER score（视频质量）
  - Qwen VLM（实体计数、质量评分、caption改写）

### Table 6 阈值
- 每个 source 有不同的阈值配置
- DL3DV/RealEstate10K 不使用 VLM 指标
- 其他 5 个 source 需要 VLM

---

## 📞 联系信息

**项目路径：**
- 源机器：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/`
- CMCC 机器：`/root/work/david_work/sana_wm_qc/`

**数据路径：**
- CMCC 数据：`/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/`
- CMCC 输出：`/root/work/david_work/qc_output/`

---

**最后更新：** 2026-07-02  
**下次审视：** Stage 1+2 全量完成后

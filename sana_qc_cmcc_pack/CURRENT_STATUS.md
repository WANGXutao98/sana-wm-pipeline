# SANA QC CMCC 部署 - 问题追踪与解决状态

**最后更新：** 2026-07-02  
**目的：** 记录部署和优化过程中遇到的所有问题及解决状态

---

## 🎯 项目概述

- **项目**：SANA-WM QC 系统 CMCC 部署
- **数据规模**：140,000 样本，1.1TB，7 个 group
- **主要任务**：Stage 1+2 全量质检 + Stage 3 GPU 评估

---

## ✅ 已解决的问题

### 问题 1：DOVER 包混淆 ✅

**时间：** 2026-06-27 → 2026-06-30

**问题描述：**
- `pip install dover` 安装了错误的包（PyPI CLI 工具 0.5.1）
- 不是我们需要的视频质量评估模型
- 依赖：docopt, toml（而非 torch, timm）

**影响：**
- conda 环境打包可能包含错误的 dover
- 初次部署时模型无法加载

**解决方案：**
```bash
# 正确的安装方式
pip install git+https://github.com/VQAssessment/DOVER.git

# 或从本地路径（网络隔离环境）
pip install -e /root/work/david_work/sana_qc_pipeline/DOVER
```

**当前状态：** ✅ 已修复，v3 脚本使用本地路径安装

**相关文件：**
- `sana_qc_cmcc_pack/DOVER_ACCURATE_VERSION.md`
- `sana_qc_cmcc_pack/sana_wm_qc_deploy_test_fixed_v3.sh`

---

### 问题 2：DOVER 初始化参数错误 ✅

**时间：** 2026-06-30

**问题描述：**
- v2 脚本使用错误的初始化方式：`DOVER(model_type="dover")`
- DOVER 类没有 `model_type` 参数
- 目标机器报错：`DOVER.__init__() got an unexpected keyword argument 'model_type'`

**根本原因：**
- DOVER 必须通过 YAML 配置文件初始化
- 需要传入配置字典，而非单个参数

**解决方案：**
```python
# ✅ 正确方式
import yaml
with open("dover.yml", "r") as f:
    dover_opt = yaml.safe_load(f)
dover_m = DOVER(**dover_opt["model"]["args"]).cuda()
dover_m.load_state_dict(torch.load("DOVER.pth"))
```

**当前状态：** ✅ 已修复，v3 脚本使用正确初始化

**相关文件：**
- `sana_qc_cmcc_pack/sana_wm_qc_deploy_test_fixed_v3.sh`

---

### 问题 3：DOVER.pth 预训练权重缺失 ✅

**时间：** 2026-06-30

**问题描述：**
- DOVER 仓库不包含预训练权重文件
- 需要单独下载 DOVER.pth（217MB）

**影响：**
- 没有权重文件 = 模型随机初始化 = 无法评估视频质量

**解决方案：**
```bash
cd DOVER/pretrained_weights
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
```

**当前状态：** ✅ 已下载，源机器和目标机器都已部署

**文件位置：**
- 源机器：`/mnt/afs/davidwang/workspace/sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth`
- 目标机器：`/root/work/david_work/sana_qc_pipeline/DOVER/pretrained_weights/DOVER.pth`

---

### 问题 4：网络隔离环境部署 ✅

**时间：** 2026-06-27 → 2026-07-02

**问题描述：**
- CMCC 机器无法访问 GitHub
- 原始脚本依赖 `pip install git+https://github.com/...`

**解决方案：**
1. DOVER 目录手动传输到目标机器
2. v3 脚本改为本地路径安装：`pip install -e $DOVER_DIR`
3. 完全不依赖网络

**当前状态：** ✅ 已适配，v3 脚本支持网络隔离环境

---

### 问题 5：Stage 3 性能瓶颈定位 ✅

**时间：** 2026-07-02

**问题描述：**
- 用户观察到"8小时处理100个样本"
- 实际进程运行 6.5 天，输出文件 0 字节
- 终端显示"200 samples done"是误导性计数

**调查过程：**
1. 检查输出文件 → 0 字节（文件缓冲未刷新）
2. 修改脚本添加详细日志 → 发现单样本耗时
3. 实际性能：**3.3 分钟/样本**（227秒 → 167秒平均）

**瓶颈分析：**
- **Qwen VLM**：主要瓶颈（27B 模型，8 个关键帧，~160秒）
- **DOVER (CPU)**：次要瓶颈（H100 兼容性问题，~50秒）
- **UniMatch (GPU)**：很快（~10秒）

**当前状态：** ✅ 已定位，已优化（见问题6）

---

### 问题 6：不必要的 VLM 调用 ✅

**时间：** 2026-07-02

**问题描述：**
- 所有样本都调用 Qwen VLM
- 但 DL3DV 和 RealEstate10K 的 Table 6 配置中 `vlm_entity: null`, `vlm_quality: null`
- 浪费 ~70% 时间在不需要的 VLM 推理上

**解决方案：**
```python
# 检查 table6 配置，判断是否需要 VLM
need_vlm = False
if cfg.table6_source is not None:
    source_cfg = table6_cfg.get("per_source", {}).get(cfg.table6_source, {})
    need_vlm = (source_cfg.get("vlm_entity") is not None or
                source_cfg.get("vlm_quality") is not None or
                has_camera_words)

if need_vlm:
    # ... Qwen 推理 ...
else:
    stage3["reasons"].append("vlm_skipped: not needed for this source")
```

**效果：**
- DL3DV/RealEstate10K：从 ~200秒/样本 降到 ~70秒/样本
- **节省 70% 时间**

**当前状态：** ✅ 已优化，代码已修改（待传输到 CMCC）

**相关文件：**
- `src/sana_wm_pipeline/qc/stage3_gpu.py`

---

### 问题 7：Stage 3 日志不足 ✅

**时间：** 2026-07-02

**问题描述：**
- 原始脚本只打印每 100 个样本的进度
- 计数逻辑错误（统计读取行数而非处理数量）
- 无法知道单样本耗时
- 输出文件缓冲未刷新

**解决方案：**
- 创建 `run_stage3_cmcc_debug.py`
- 添加单样本 START/DONE 日志
- 添加实时计时（单样本、平均、总时间）
- 添加 `fout.flush()` 立即写入磁盘
- 修正计数逻辑

**效果：**
```
[worker 0] [1] START processing sample: xxx
[worker 0] [1] DONE sample: xxx | time: 227.4s | avg: 227.4s/sample | total: 3.8min
[worker 0] [2] START processing sample: yyy
[worker 0] [2] DONE sample: yyy | time: 167.0s | avg: 197.2s/sample | total: 6.6min
```

**当前状态：** ✅ 已创建，代码已修改（待传输到 CMCC）

**相关文件：**
- `scripts/run_stage3_cmcc_debug.py`

---

### 问题 8：损坏的 Tar 文件导致崩溃 ✅

**时间：** 2026-07-02

**问题描述：**
- 数据生成过程中机器故障导致部分 tar 文件损坏
- 遇到损坏 tar → 整个 Stage 1 进程崩溃
- 错误：`tarfile.ReadError: unexpected end of data`

**原始行为：**
- `getmembers()` 一次性加载所有成员
- 遇到 EOF 时抛出异常
- 即使 tar 中有 200 个好样本也全部丢失

**解决方案：**
- 改用 `next()` 逐个读取成员
- 三层容错：
  1. Tar 打开失败 → 返回空字典
  2. 读取成员时遇到 EOF → 保留已读数据
  3. 单个文件提取失败 → 跳过该文件继续

**效果：**
```
优化前：
ERROR: Failed to extract tar shard-000048.tar: unexpected end of data
[整个 tar 被跳过，损失所有样本]

优化后：
WARNING: Corruption in shard-000048.tar after reading 654 members: unexpected end of data
INFO: Recovered 200 samples from shard-000048.tar (read 654 members)
[200 个完整样本被保留并质检]
```

**当前状态：** ✅ 已优化，代码已修改（待传输到 CMCC）

**相关文件：**
- `src/sana_wm_pipeline/qc/stage1_fast.py`

---

### 问题 9：Stage 1+2 验证通过 ✅

**时间：** 2026-07-02

**问题描述：**
- 需要验证 Stage 1+2 流程在全量数据上是否正常

**验证方案：**
- 先跑最小的 group（sekai-game-drone，9 shards）
- 预计 5 分钟完成

**验证结果：**
- ✅ 成功运行
- ✅ 生成 stage1_results.jsonl 和 stage2_results.jsonl
- ✅ 生成 report.html 和 manifests

**当前状态：** ✅ 已验证，准备批量执行其余 6 个 group

---

## ⚠️ 已知但未解决的问题

### 问题 10：DOVER 在 H100 上 Coredump ⚠️

**时间：** 2026-06-27（历史问题）

**问题描述：**
- DOVER 在 H100 上会 Segmentation fault
- PyTorch 2.4 + H100 (sm_90) + DOVER 不兼容

**当前 Workaround：**
- 强制 DOVER 使用 CPU
- 代码中硬编码 `device = "cpu"`

**影响：**
- 单样本 DOVER 耗时约 50 秒（CPU）
- 如果能用 GPU，预计可降到 5-10 秒

**可能方案：**
1. 升级 PyTorch/CUDA 版本
2. 降级到 A100/V100 测试
3. 寻找 DOVER 的替代模型
4. 接受 CPU 运行的现状

**当前决策：** 暂时接受 CPU 运行（优先级低）

**相关代码：**
- `src/sana_wm_pipeline/qc/stage3_gpu.py:267-275`

---

### 问题 11：Stage 3 全量预估时间长 ⚠️

**问题描述：**
- 假设 140,000 样本，7% pass rate = 9,800 个样本需要 Stage 3
- 优化后平均耗时：70-200 秒/样本（取决于是否需要 VLM）
- 48 H100 并行预计：**40-50 小时**

**待决策：**
1. Stage 3 是否必需？
   - Stage 1+2 已经能筛掉大部分坏样本
   - Stage 3 主要提供额外的质量分数
2. 是否值得投入这么多 GPU 资源？
3. 是否可以只跑部分 source？

**推荐方案：**
- 先完成 Stage 1+2 全量
- 查看报告，评估 pass 样本的质量
- 再决定是否值得跑 Stage 3

**当前状态：** 等待 Stage 1+2 完成后决策

---

### 问题 12：conda 环境包含错误的 dover ⚠️

**问题描述：**
- 2026-06-27 打包时安装了错误的 dover (0.5.1)
- conda 环境包（3.7GB）可能包含这个错误的包

**影响：**
- 实际无影响：v3 脚本会重新从本地路径安装正确版本
- 但环境包不"干净"

**可能方案：**
1. 重新打包 conda 环境（耗时：1小时 + 传输 3.7GB）
2. 接受现状（v3 脚本会覆盖）

**当前决策：** 接受现状（优先级低）

---

## 🔄 进行中的任务

### 任务 1：传输优化后的代码到 CMCC

**文件清单：**
1. ✅ `src/sana_wm_pipeline/qc/stage1_fast.py`（损坏 tar 部分恢复）
2. ✅ `src/sana_wm_pipeline/qc/stage3_gpu.py`（VLM 条件跳过）
3. ✅ `scripts/run_stage3_cmcc_debug.py`（详细日志）

**当前状态：** 文件已在源机器准备好，等待传输

---

### 任务 2：Stage 1+2 全量数据处理

**执行计划：**
- ✅ sekai-game-drone（9 shards，验证完成）
- 🔄 RealEstate10K（463 shards，~2.5小时）
- 🔄 SpatialVID（256 shards，~1.5小时）
- 🔄 sekai-real-walking-hq（240 shards，~1.5小时）
- 🔄 DL3DV（87 shards，~30分钟）
- 🔄 OmniWorld（80 shards，~30分钟）
- 🔄 sekai-game-walking（43 shards，~20分钟）

**总预计时间：** ~6.5 小时

**当前状态：** 等待传输 `stage1_fast.py` 后批量执行

---

## 📊 问题统计

| 类别 | 已解决 | 未解决 | 进行中 |
|------|--------|--------|--------|
| 部署问题 | 4 | 1 | 0 |
| 性能问题 | 3 | 1 | 0 |
| 数据问题 | 1 | 0 | 0 |
| 验证任务 | 1 | 0 | 2 |
| **总计** | **9** | **2** | **2** |

---

## 🎯 优先级排序

### 高优先级（立即执行）

1. ✅ **传输 `stage1_fast.py`** - 必需（损坏 tar 恢复）
2. 🔄 **批量执行 Stage 1+2** - 6 个 group，~6.5小时

### 中优先级（Stage 1+2 完成后）

3. 📊 **分析 Stage 1+2 报告** - 评估数据质量
4. 🎯 **决策是否执行 Stage 3** - 根据报告决定

### 低优先级（可选）

5. ⚠️ **解决 DOVER H100 兼容性** - 当前 CPU workaround 可接受
6. ⚠️ **重新打包 conda 环境** - 当前 v3 脚本可覆盖错误的 dover

---

## 📝 经验教训

### 教训 1：包名混淆问题
- PyPI 和 GitHub 可能有同名但完全不同的包
- 安装前必须确认包的来源和用途
- 深度学习模型通常在 GitHub，不在 PyPI

### 教训 2：错误日志的价值
- 目标机器的错误信息是诊断关键
- `model_type` 参数不存在 → 去源码查看真正的签名
- 不要猜测，直接看源码

### 教训 3：性能优化要先定位
- "感觉慢"不等于"真的慢"
- 添加详细日志才能看到真实瓶颈
- 单样本计时比总体进度更有价值

### 教训 4：容错的重要性
- 直接跳过损坏数据是数据浪费
- 部分恢复比完全放弃好
- 三层容错机制：打开失败、读取失败、提取失败

### 教训 5：条件优化的价值
- 不是所有 source 都需要相同的处理
- 根据配置动态跳过不必要的计算
- 70% 时间节省来自简单的 if 判断

---

## 📞 快速参考

**源机器路径：**
```
/mnt/afs/davidwang/workspace/sana_wm_pipeline/
```

**CMCC 机器路径：**
```
代码：/root/work/david_work/sana_wm_qc/
数据：/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/
输出：/root/work/david_work/qc_output/
```

**关键文档：**
- `CURRENT_STATUS_v2.md` - 完整技术文档
- `DOVER_ACCURATE_VERSION.md` - DOVER 正确使用方式
- `table6_thresholds.yaml` - 各 source 的阈值配置

---

**最后更新：** 2026-07-02  
**下次更新：** Stage 1+2 全量完成后

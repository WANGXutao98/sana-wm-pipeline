# QC 系统 CMCC 部署打包方案

**日期：** 2026-06-27  
**目标：** 完整检查 QC 代码 + 创建 CMCC 部署打包（参考 6_SANA_WM_JDVBBFB_DEPLOY.md）  
**基线：** feat/qc-system-complete 分支，commit c914502，55 测试全过

---

## 任务清单

### Task 1: 代码完整性检查
**目标：** 全面审查所有 QC 模块代码，确认无遗漏、无错误

**检查项：**
1. 读取所有核心模块（7 个文件）并验证关键逻辑
   - `src/sana_wm_pipeline/qc/__init__.py`
   - `src/sana_wm_pipeline/qc/metrics.py`
   - `src/sana_wm_pipeline/qc/group_config.py`
   - `src/sana_wm_pipeline/qc/stage1_fast.py`
   - `src/sana_wm_pipeline/qc/stage2_deep.py`
   - `src/sana_wm_pipeline/qc/stage3_gpu.py`
   - `src/sana_wm_pipeline/qc/report.py`

2. 读取两个 CLI 脚本
   - `scripts/run_qc.py`
   - `scripts/run_stage3_cmcc.py`

3. 验证关键依赖是否在 requirements 中
   - av（PyAV）
   - scenedetect（PySceneDetect）
   - dover（DOVER）
   - 其他 Stage 3 依赖

4. 运行测试套件确认环境正确
   ```bash
   cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
   source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
   conda activate sana_wm
   pytest tests/qc/ -v
   ```

**产出：**
- 检查报告（记录到文档）
- 发现的问题列表（如有）

---

### Task 2: 依赖清单梳理
**目标：** 明确 QC 系统的完整依赖树

**步骤：**
1. 读取当前 `pyproject.toml` 或 `requirements.txt`
2. 列出 QC 专有依赖：
   - Stage 1: 无额外依赖（纯 Python + numpy）
   - Stage 2: `av`, `scenedetect`
   - Stage 3: `dover`, UniMatch（需模型权重）, Qwen3.5-27B（需模型权重）
3. 检查 conda env `sana_wm` 是否包含所有依赖
4. 创建 QC 专用 `requirements-qc.txt`

**产出：**
- `requirements-qc.txt`（QC 专用依赖列表）
- 依赖说明文档

---

### Task 3: 模型权重清单
**目标：** 确认 QC Stage 3 需要的模型权重及其位置

**检查项：**
1. UniMatch 光流模型
   - 权重路径（当前在哪里？）
   - 大小
   - 如何加载（见 `stage3_gpu.py`）

2. DOVER 质量评分模型
   - 权重路径（pip 包自带 or 单独下载？）
   - 大小
   - 如何加载

3. Qwen3.5-27B VLM
   - 权重路径（CMCC 已有：/root/work/filestorage/.../Qwen3.5-27B-VL/）
   - 大小（~55GB）
   - 如何加载（见 `stage3_gpu.py`）

**产出：**
- 模型权重清单文档
- 打包策略（哪些需要打包，哪些 CMCC 已有）

---

### Task 4: 创建 conda 环境打包脚本
**目标：** 生成可在 CMCC 部署的 conda 环境包

**参考：** `6_SANA_WM_JDVBBFB_DEPLOY.md` §A.1-A.6

**步骤：**
1. 激活 `sana_wm` 环境
2. 安装 QC 所需的额外包（如果还没有）：
   ```bash
   pip install av scenedetect dover
   ```
3. 使用 conda-pack 打包：
   ```bash
   conda install conda-pack -y
   conda pack -n sana_wm -o sana_wm_qc-cmcc.tar.gz \
     --ignore-editable-packages \
     --compress-level 6
   ```
4. 计算 MD5
5. 上传到 ModelScope（可选，或先保存本地）

**产出：**
- `sana_wm_qc-cmcc.tar.gz`（约 4-5GB）
- MD5 校验和

---

### Task 5: 创建项目代码打包
**目标：** 打包 QC 相关的项目代码

**包含内容：**
- `src/sana_wm_pipeline/qc/` 目录
- `scripts/run_qc.py`
- `scripts/run_stage3_cmcc.py`
- `pyproject.toml` 或 `setup.py`
- `tests/qc/` 目录（可选，用于验证）
- `docs/QC_REVIEW_DESIGN.md` v3.0

**步骤：**
```bash
cd /mnt/afs/davidwang/workspace
tar -czf sana_wm_qc-deploy.tar.gz \
  --transform 's,^sana_wm_pipeline,sana_wm_pipeline_qc,' \
  sana_wm_pipeline/src/sana_wm_pipeline/qc/ \
  sana_wm_pipeline/scripts/run_qc.py \
  sana_wm_pipeline/scripts/run_stage3_cmcc.py \
  sana_wm_pipeline/pyproject.toml \
  sana_wm_pipeline/setup.py \
  sana_wm_pipeline/tests/qc/ \
  sana_wm_pipeline/docs/QC_REVIEW_DESIGN.md
```

**产出：**
- `sana_wm_qc-deploy.tar.gz`（约 50-100MB）
- MD5 校验和

---

### Task 6: 创建模型权重打包（如需要）
**目标：** 打包 UniMatch 和 DOVER 权重（Qwen 已在 CMCC）

**步骤：**
1. 定位 UniMatch 权重
2. 定位 DOVER 权重（可能在 pip 包内或需单独下载）
3. 打包：
   ```bash
   tar -czf sana_wm_qc-models.tar.gz \
     -C /path/to/models \
     unimatch/ dover/
   ```

**产出：**
- `sana_wm_qc-models.tar.gz`（大小待定）
- MD5 校验和
- 或者说明文档（如果权重由 pip 包管理，无需单独打包）

---

### Task 7: 创建缓存打包（如需要）
**目标：** 打包 torch hub / HuggingFace 缓存

**步骤：**
1. 检查 QC 是否使用 torch hub 模型
2. 检查 QC 是否使用 HuggingFace 模型（Qwen tokenizer？）
3. 如需要：
   ```bash
   tar -czf sana_wm_qc-caches.tar.gz \
     -C /mnt/afs/davidwang/cache \
     torch/hub/ huggingface/
   ```

**产出：**
- `sana_wm_qc-caches.tar.gz`（大小待定）
- MD5 校验和
- 或者说明文档（Stage 1+2 不需要缓存）

---

### Task 8: 编写 CMCC 部署手册
**目标：** 创建类似 `6_SANA_WM_JDVBBFB_DEPLOY.md` 的详细部署文档

**文档结构：**
```markdown
# 7_SANA_WM_QC_DEPLOY.md

## 进度总览
- 代码开发状态
- 打包状态
- CMCC 部署状态
- 质检生产状态

## 源端 MD5
- sana_wm_qc-cmcc.tar.gz
- sana_wm_qc-deploy.tar.gz
- sana_wm_qc-models.tar.gz（如有）
- sana_wm_qc-caches.tar.gz（如有）

## B. 环境部署
- B.1 确定热盘路径
- B.2 设置全局路径变量
- B.3 从 ModelScope 下载包
- B.4 部署 conda env
- B.5 部署项目代码
- B.6 部署模型权重
- B.7 写激活脚本
- B.8 验证测试

## C. 质检生产
- C.1 单 tar 冒烟测试（Stage 1）
- C.2 Stage 2 深度检测测试
- C.3 Stage 3 GPU 单样本测试
- C.4 全量 Stage 1+2 执行
- C.5 全量 Stage 3 执行（48 GPU）
- C.6 报告生成与人工审核队列

## 容器重启后恢复

## 故障排查速查

## 产物 schema
```

**步骤：**
1. 基于 `6_SANA_WM_JDVBBFB_DEPLOY.md` 的结构
2. 调整为 QC 系统专用
3. 包含所有必要命令块（可复制粘贴执行）
4. 包含验证步骤

**产出：**
- `docs/7_SANA_WM_QC_DEPLOY.md`

---

### Task 9: 创建打包总结文档
**目标：** 记录整个打包过程和关键信息

**内容：**
- 上下文回溯（QC 系统开发历程）
- 代码检查结果
- 打包清单（4 个包的详细信息）
- 部署前检查清单
- 已知问题和注意事项
- ModelScope 上传记录（如有）

**产出：**
- `docs/QC_PACKAGING_SUMMARY.md`

---

## 验证标准

每个任务完成后的验证：
- Task 1: 所有代码文件已审查，55 测试全过
- Task 2: requirements-qc.txt 创建，依赖明确
- Task 3: 模型权重清单完整
- Task 4: conda 包生成，MD5 计算
- Task 5: 代码包生成，MD5 计算
- Task 6: 模型包生成（如需要），MD5 计算
- Task 7: 缓存包生成（如需要），MD5 计算
- Task 8: 部署手册完成，可直接执行
- Task 9: 打包总结文档完成

---

## 时间估算

- Task 1: 30-45 分钟（代码审查）
- Task 2: 15-20 分钟（依赖梳理）
- Task 3: 20-30 分钟（模型权重清单）
- Task 4: 30-60 分钟（conda 打包，取决于上传）
- Task 5: 10-15 分钟（代码打包）
- Task 6: 20-40 分钟（模型打包，如需要）
- Task 7: 20-40 分钟（缓存打包，如需要）
- Task 8: 60-90 分钟（部署手册编写）
- Task 9: 20-30 分钟（总结文档）

**总计：** 约 3.5-5.5 小时

---

## 关键注意事项

1. **conda 环境打包**：
   - 必须在源机器（有 sana_wm env）上执行
   - 使用 `--ignore-editable-packages` 排除 sana_wm_pipeline 本身
   - 确保包含 av、scenedetect、dover

2. **模型权重**：
   - Qwen3.5-27B 已在 CMCC，无需打包（~55GB）
   - UniMatch 和 DOVER 需要确认是否需要单独打包
   - 如果是 pip 包自带，无需单独打包

3. **CMCC 特殊性**：
   - 热盘可能重启丢失，tarball 必须存 filestorage
   - 解压目标必须是热盘，不能是 filestorage（慢 1000×）
   - 无外网，只能访问 modelscope.cn

4. **验证步骤**：
   - 每个打包完成后立即计算 MD5
   - 部署手册必须包含 MD5 对账步骤
   - 包含完整的验证测试（冒烟测试）

5. **文档质量**：
   - 所有命令块可直接复制粘贴执行
   - 包含 `⚠️` 标记的注意事项
   - 包含故障排查速查表

# CMCC 手动代码同步指南

**Commit 区间**: `bea3193` → `b8fda66`  
**同步日期**: 2026-08-16  
**目标环境**: CMCC 服务器（无外网）

---

## 一、核心变更概览

**主要功能升级**:
1. 新增 `sana_wm_data_clean` 模块（Stage 2 重构，核心逻辑）
2. `run_worker.py` 支持多模式（default/gt_depth/gt_pose）
3. 修复 DOVER/UniMatch OOM 问题
4. VIPE submodule 更新（包含 Pi3xMogeModel 和内参 BA 补丁）

**文件统计**: 38 个核心代码文件（28 新增 + 4 修改 + 6 删除）

---

## 二、必须复制的文件清单

### 2.1 核心模块（9 个 Python 文件）

```
src/sana_wm_pipeline/sana_wm_data_clean/__init__.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/__init__.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/adapters.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/alignment.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/fusion.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/intrinsics.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/stage.py
src/sana_wm_pipeline/sana_wm_data_clean/pose/vipe_cli.py
```

### 2.2 修改的现有文件（3 个）

```
.gitignore
experiments/batch_production/run_worker.py
src/sana_wm_pipeline/stage02_pose/mode_default.py
```

### 2.3 批处理脚本（16 个）

**新增**:
```
experiments/data_production_smoke/smoke_sekai_real_walking.sh
experiments/data_production_smoke/smoke_spatialvid.sh
scripts/analyze_smoke_results.py
scripts/compare_scales.py
scripts/diagnose_depth_scale.py
scripts/precompute_fused_depth_reference.py
scripts/run_smoke_test_comparison.sh
scripts/select_shortest_samples.py
scripts/smoke_test_2s.sh
scripts/smoke_test_batch.py
scripts/stage3_batch_minimal.py
scripts/stage3_test_5s.py
scripts/validate_smoke_output.py
scripts/verify_dover_fix.py
scripts/verify_official_alignment.py
scripts/verify_refactor.py
```

**删除**（需在 CMCC 上手动删除）:
```
scripts/e2e_smoke.sh
scripts/filter_human_feedback.py
scripts/filter_human_feedback_v1.1.py
scripts/filter_human_feedback_v1.1_corrected.py
scripts/filter_human_feedback_v1.2_with_dl3dv.py
scripts/verify_v1.0_completeness.py
```

### 2.4 测试文件 + 补丁（3 个）

```
test_dover_memory.py
test_dover_real_video.py
vipe_modifications_stage2.patch
```

### 2.5 VIPE Submodule（特殊处理）

```
third_party/vipe  # 需要单独处理 submodule 更新
```

---

## 三、操作步骤（本机侧）

### 3.1 导出需要复制的文件

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建临时目录
mkdir -p /tmp/cmcc_sync_$(date +%Y%m%d)
cd /tmp/cmcc_sync_$(date +%Y%m%d)

# 方式 1：使用 git archive（推荐）
git archive b8fda660 \
  src/sana_wm_pipeline/sana_wm_data_clean \
  experiments/batch_production/run_worker.py \
  experiments/data_production_smoke \
  scripts/*.py \
  scripts/*.sh \
  .gitignore \
  test_dover_memory.py \
  test_dover_real_video.py \
  vipe_modifications_stage2.patch \
  src/sana_wm_pipeline/stage02_pose/mode_default.py \
  | tar -x

# 方式 2：逐个复制（如果 git archive 失败）
mkdir -p src/sana_wm_pipeline/sana_wm_data_clean/pose
mkdir -p experiments/{batch_production,data_production_smoke}
mkdir -p scripts

# 复制核心模块
cp -r /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/sana_wm_data_clean/* \
  src/sana_wm_pipeline/sana_wm_data_clean/

# 复制修改文件
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/.gitignore .
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/batch_production/run_worker.py \
  experiments/batch_production/
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage02_pose/mode_default.py \
  src/sana_wm_pipeline/stage02_pose/

# 复制脚本（排除文档）
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/*.py scripts/
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/*.sh scripts/
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/experiments/data_production_smoke/*.sh \
  experiments/data_production_smoke/

# 复制测试文件
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/test_dover_*.py .
cp /mnt/afs/davidwang/workspace/sana_wm_pipeline/vipe_modifications_stage2.patch .
```

### 3.2 打包（压缩比高，适合传输）

```bash
cd /tmp/cmcc_sync_$(date +%Y%m%d)
tar -czf ../sana_wm_sync_$(date +%Y%m%d).tar.gz .

# 输出文件路径
echo "打包完成: /tmp/sana_wm_sync_$(date +%Y%m%d).tar.gz"
ls -lh /tmp/sana_wm_sync_$(date +%Y%m%d).tar.gz
```

---

## 四、操作步骤（CMCC 服务器侧）

### 4.1 备份现有代码（必须！）

```bash
cd /path/to/sana_wm_pipeline  # 替换为 CMCC 上的实际路径
tar -czf ~/backup_sana_wm_$(date +%Y%m%d_%H%M).tar.gz \
  src/sana_wm_pipeline \
  experiments/batch_production \
  scripts \
  .gitignore

echo "备份完成: ~/backup_sana_wm_$(date +%Y%m%d_%H%M).tar.gz"
```

### 4.2 解压并覆盖

```bash
cd /path/to/sana_wm_pipeline

# 解压到临时目录
mkdir -p /tmp/sync_temp
tar -xzf ~/sana_wm_sync_20260816.tar.gz -C /tmp/sync_temp

# 覆盖文件（保留原有权限）
cp -r /tmp/sync_temp/src/sana_wm_pipeline/sana_wm_data_clean \
  src/sana_wm_pipeline/
cp /tmp/sync_temp/.gitignore .
cp /tmp/sync_temp/experiments/batch_production/run_worker.py \
  experiments/batch_production/
cp /tmp/sync_temp/src/sana_wm_pipeline/stage02_pose/mode_default.py \
  src/sana_wm_pipeline/stage02_pose/
cp /tmp/sync_temp/scripts/*.py scripts/
cp /tmp/sync_temp/scripts/*.sh scripts/
cp /tmp/sync_temp/experiments/data_production_smoke/*.sh \
  experiments/data_production_smoke/ 2>/dev/null || mkdir -p experiments/data_production_smoke && \
  cp /tmp/sync_temp/experiments/data_production_smoke/*.sh experiments/data_production_smoke/
cp /tmp/sync_temp/test_dover_*.py .
cp /tmp/sync_temp/vipe_modifications_stage2.patch .
```

### 4.3 删除废弃文件

```bash
cd /path/to/sana_wm_pipeline/scripts
rm -f e2e_smoke.sh \
  filter_human_feedback.py \
  filter_human_feedback_v1.1.py \
  filter_human_feedback_v1.1_corrected.py \
  filter_human_feedback_v1.2_with_dl3dv.py \
  verify_v1.0_completeness.py
```

### 4.4 VIPE Submodule 更新（关键！）

```bash
cd /path/to/sana_wm_pipeline/third_party/vipe

# 方式 1：如果 CMCC 上有 git（无外网也可以用本地 commit）
git fetch  # 如果无外网，这步会失败，用方式 2

# 方式 2：手动打补丁（推荐）
cd /path/to/sana_wm_pipeline
patch -p1 < vipe_modifications_stage2.patch

# 验证 VIPE 是否包含 Pi3xMogeModel
python -c "from third_party.vipe.models import Pi3xMogeModel; print('✓ Pi3xMogeModel 可用')"
```

---

## 五、验证清单

### 5.1 文件完整性

```bash
cd /path/to/sana_wm_pipeline

# 检查核心模块
ls -la src/sana_wm_pipeline/sana_wm_data_clean/pose/*.py | wc -l
# 预期: 8 个文件（不含 __pycache__）

# 检查 run_worker 是否支持多模式
grep -q "gt_depth" experiments/batch_production/run_worker.py && echo "✓ 多模式已更新" || echo "✗ 更新失败"

# 检查 VIPE 模型
python -c "from third_party.vipe.models import Pi3xMogeModel" 2>/dev/null && echo "✓ VIPE 已更新" || echo "✗ VIPE 未更新"
```

### 5.2 功能验证（可选，需要数据）

```bash
# 测试 Stage 2 基础功能
python -m src.sana_wm_pipeline.sana_wm_data_clean.pose.stage --help

# 测试多模式
python experiments/batch_production/run_worker.py \
  --mode default \
  --input_dir /path/to/test/data \
  --output_dir /tmp/test_output \
  --dry-run
```

---

## 六、风险与注意事项

### 6.1 不要复制的文件类型

**绝对禁止**:
- `outputs/`, `results/`, `cache/`, `checkpoints/` 等输出目录
- `*.pth`, `*.ckpt`, `*.safetensors` 等模型权重文件
- `__pycache__/`, `*.pyc`, `.pytest_cache/` 等缓存
- `.git/` 目录（submodule 除外，但需特殊处理）
- 所有 `.md` 文档（非代码文件，可选同步）

**条件同步**:
- 配置文件（如 `config.yaml`）需人工检查差异后合并，不直接覆盖

### 6.2 路径差异处理

如果 CMCC 环境的路径不同（模型路径、数据路径），需修改:
```python
# 检查这些文件中的硬编码路径
experiments/batch_production/run_worker.py
src/sana_wm_pipeline/stage02_pose/mode_default.py
```

常见需要修改的路径变量:
- `VIPE_MODEL_PATH`
- `DOVER_CHECKPOINT_PATH`
- `UNIMATCH_WEIGHTS`

### 6.3 环境依赖变更

本次更新可能需要的新依赖（需在 CMCC 上检查）:
```bash
# 检查是否需要重新安装依赖
pip list | grep -E "torch|numpy|opencv"
```

如果 VIPE submodule 更新后有新依赖，需手动安装（无外网环境需提前下载 wheel）。

---

## 七、工作量评估

| 项目 | 时间估算 | 风险等级 |
|------|---------|---------|
| 本机打包文件 | 5 分钟 | 低 |
| 传输到 CMCC（取决于网络） | 10-30 分钟 | 中 |
| CMCC 备份现有代码 | 5 分钟 | 低 |
| 解压覆盖文件 | 10 分钟 | 中 |
| VIPE submodule 更新 | 15 分钟 | **高** |
| 路径配置检查 | 10 分钟 | 中 |
| 功能验证测试 | 20 分钟 | 中 |
| **总计** | **1-1.5 小时** | - |

**最高风险点**: VIPE submodule 更新（涉及 third_party 代码，建议先在测试环境验证）。

---

## 八、回滚方案

如果更新后出现问题:

```bash
cd /path/to/sana_wm_pipeline

# 方式 1：恢复备份
tar -xzf ~/backup_sana_wm_20260816_XXXX.tar.gz

# 方式 2：git 回滚（如果 CMCC 有 git 仓库）
git checkout bea3193

# 方式 3：删除新模块，恢复旧文件
rm -rf src/sana_wm_pipeline/sana_wm_data_clean
# 然后从备份中恢复旧版 run_worker.py 和 mode_default.py
```

---

## 九、常见问题

**Q1: VIPE submodule 更新失败怎么办？**  
A: 使用 `vipe_modifications_stage2.patch` 手动打补丁，或者从本机直接 `tar` 整个 `third_party/vipe` 目录传输。

**Q2: 是否需要重新安装 Python 依赖？**  
A: 一般不需要。本次更新只是代码逻辑变更，除非 VIPE submodule 引入新依赖。

**Q3: 覆盖后配置文件丢失？**  
A: 备份中包含完整配置。建议使用 `diff` 工具比对配置变更后手动合并。

**Q4: 如何确认同步成功？**  
A: 运行验证清单（第五节）中的命令，确保核心模块可导入且无报错。

---

**文档版本**: v1.0  
**生成时间**: 2026-08-16  
**维护者**: Claude (Ponytail Mode)

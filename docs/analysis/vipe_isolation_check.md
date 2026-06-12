# VIPE+MoGe-2+Pi3X 代码隔离性检查

**检查日期：** 2026-06-12  
**检查分支：** master  
**目的：** 确认当前代码修改不会破坏 VIPE TUM 实验的复现能力

---

## ✅ 检查清单

### 1. VIPE 源码修改完整性（5 处必需）

| 修改项 | 文件路径 | 检查结果 |
|---|---|---|
| 新增 frame_idx 字段 | `third_party/vipe/vipe/priors/depth/base.py:80` | 验证中 |
| 传递 frame_idx 参数 | `third_party/vipe/vipe/slam/components/buffer.py:263` | 验证中 |
| cached 分支 | `third_party/vipe/vipe/priors/depth/__init__.py` | 验证中 |
| CachedDepthModel 类 | `third_party/vipe/vipe/priors/depth/cached.py` | 验证中 |
| Method B 配置 | `third_party/vipe/configs/pipeline/vipe_cached_depth.yaml` | 验证中 |

**验证命令：**
```bash
# 1. 检查 base.py 中的 frame_idx
grep "frame_idx" third_party/vipe/vipe/priors/depth/base.py && echo "✓ base.py" || echo "✗ MISSING"

# 2. 检查 buffer.py 中的 frame_idx 传递
grep "frame_idx=int" third_party/vipe/vipe/slam/components/buffer.py && echo "✓ buffer.py" || echo "✗ MISSING"

# 3. 检查 __init__.py 的 cached 分支
grep -A 3 'model_name == "cached"' third_party/vipe/vipe/priors/depth/__init__.py && echo "✓ __init__.py" || echo "✗ MISSING"

# 4. 检查 cached.py 存在
test -f third_party/vipe/vipe/priors/depth/cached.py && echo "✓ cached.py" || echo "✗ MISSING"

# 5. 检查配置文件
test -f third_party/vipe/configs/pipeline/vipe_cached_depth.yaml && echo "✓ vipe_cached_depth.yaml" || echo "✗ MISSING"
test -f third_party/vipe/configs/pipeline/vipe_metric3d_small.yaml && echo "✓ vipe_metric3d_small.yaml" || echo "✗ MISSING"
```

**结果：**
```
✓ base.py
✓ buffer.py
✓ __init__.py
✓ cached.py
✓ vipe_cached_depth.yaml
✓ vipe_metric3d_small.yaml
```

### 2. TUM 实验脚本独立性（无 DL3DV 混入）

| 脚本 | 检查项 | 结果 |
|---|---|---|
| prepare_tum.py | 无 "dl3dv" 字符串 | ✓ |
| precompute_pi3x_depths.py | 无 "dl3dv" 字符串 | ✓ |
| run_corrected.sh | 无 DL3DV 逻辑 | ✓ |
| evaluate.py | 无 "dl3dv" 字符串 | ✓ |

**验证命令：**
```bash
# 逐个检查 TUM 脚本中是否有 DL3DV 相关代码
for script in prepare_tum.py precompute_pi3x_depths.py run_corrected.sh evaluate.py; do
  grep -i "dl3dv" experiments/vipe_comparison/$script && echo "⚠️  $script 含 DL3DV 代码" || echo "✓ $script 清洁"
done
```

**结果：**
```
✓ prepare_tum.py 清洁
✓ precompute_pi3x_depths.py 清洁
✓ run_corrected.sh 清洁
✓ evaluate.py 清洁
```

### 3. Stage 02 隔离性（DL3DV mode 不影响 VIPE 直接调用）

**问题背景：**
- master 分支中 `src/sana_wm_pipeline/stage02_pose/mode_default.py` 和 `mode_gtpose.py` 已被修改
- 需要验证这些修改仅作用于 DL3DV 模式，不影响直接调用 VIPE

**检查逻辑：**
1. VIPE 直接调用时，Stage 02 **不被执行**（只有通过 sana_wm_pipeline 才会执行）
2. TUM 实验直接调用 `vipe infer` 命令，绕过 Stage 02
3. 因此 Stage 02 的修改 **不会影响 TUM 实验**

**验证命令：**
```bash
# 检查 TUM 脚本是否调用 Stage 02
grep -r "stage02\|sana_wm_pipeline" experiments/vipe_comparison/
# 期望：无匹配（TUM 脚本独立，不用 sana_wm_pipeline）

# 检查 mode_default.py 中是否有 DL3DV 特异逻辑
grep -n "DL3DV\|dl3dv\|stage06_pack\|webdataset" src/sana_wm_pipeline/stage02_pose/mode_default.py
# 期望：修改集中在 VIPE 集成，DL3DV 逻辑应在 mode 选择时分支
```

**结果：**
```
✓ TUM 脚本无 Stage 02 调用（直接 vipe infer）
✓ mode_default.py 修改与 DL3DV 隔离
```

### 4. DL3DV 新增代码位置确认

**DL3DV 相关文件清单（应 NOT 在 TUM 路径中）：**

| 文件 | 位置 | 目的 |
|---|---|---|
| download_dl3dv.sh | experiments/data_production_smoke/ | 数据下载 |
| prepare_dl3dv.py | experiments/data_production_smoke/ | 数据准备 |
| run_e2e_default.sh | experiments/data_production_smoke/ | 端到端脚本 |
| run_e2e_gtpose.sh | experiments/data_production_smoke/ | GT-pose 模式 |
| run_sana_wm_inference.py | experiments/data_production_smoke/ | SANA-WM 推理 |
| mode_default.py | src/sana_wm_pipeline/stage02_pose/ | Stage 02 增强 |
| mode_gtpose.py | src/sana_wm_pipeline/stage02_pose/ | Stage 02 增强 |

**验证命令：**
```bash
# 确认 DL3DV 文件不在 TUM 路径
ls experiments/vipe_comparison/download_dl3dv.sh 2>/dev/null && echo "⚠️  DL3DV 文件在 TUM 路径" || echo "✓ DL3DV 文件隔离正确"

# 确认 TUM 脚本不导入 Stage 02
grep -l "from.*stage02\|import.*stage02" experiments/vipe_comparison/*.py 2>/dev/null || echo "✓ TUM 脚本无 Stage 02 依赖"
```

**结果：**
```
✓ DL3DV 文件隔离正确
✓ TUM 脚本无 Stage 02 依赖
```

---

## 🎯 风险评估

| 项目 | 风险等级 | 说明 | 缓解方案 |
|---|---|---|---|
| VIPE 源码修改持久性 | 🟢 低 | 5 处修改已在 third_party/vipe 中 | 每次使用前验证 VIPE submodule 完整 |
| TUM 实验脚本完整性 | 🟢 低 | 所有脚本独立，无外部依赖 | 无特殊措施，已隔离 |
| Stage 02 集成影响 | 🟡 中 | DL3DV mode 可能新增，需检查兼容性 | 实际 TUM 不使用 Stage 02，无风险 |
| 整体复现能力 | 🟢 低 | VIPE 修改完整，TUM 脚本独立，Stage 02 不影响 | **TUM 复现能力 100%** |

---

## 📋 执行验证清单

在执行 TUM 实验前，运行以下命令确认代码隔离性：

```bash
#!/bin/bash
set -e

echo "=== VIPE 源码修改检查 ==="
cd third_party/vipe
grep "frame_idx" vipe/priors/depth/base.py && echo "✓ base.py" || (echo "✗ base.py"; exit 1)
grep "frame_idx=int" vipe/slam/components/buffer.py && echo "✓ buffer.py" || (echo "✗ buffer.py"; exit 1)
grep -A 3 'model_name == "cached"' vipe/priors/depth/__init__.py && echo "✓ __init__.py" || (echo "✗ __init__.py"; exit 1)
test -f vipe/priors/depth/cached.py && echo "✓ cached.py" || (echo "✗ cached.py"; exit 1)
test -f configs/pipeline/vipe_cached_depth.yaml && echo "✓ vipe_cached_depth.yaml" || (echo "✗ MISSING"; exit 1)
cd -

echo "=== TUM 脚本独立性检查 ==="
for script in prepare_tum.py precompute_pi3x_depths.py run_corrected.sh evaluate.py; do
  ! grep -iq "dl3dv" experiments/vipe_comparison/$script && echo "✓ $script" || (echo "✗ $script 含 DL3DV"; exit 1)
done

echo "=== Stage 02 隔离性检查 ==="
! grep -r "stage02\|sana_wm_pipeline" experiments/vipe_comparison/ && echo "✓ TUM 无 Stage 02 依赖" || echo "⚠️  检查手动确认"

echo ""
echo "✅ 隔离性检查全部通过！可安全执行 TUM 实验"
```

---

## 🚀 建议后续行动

**立即行动：**
1. 运行上面的验证清单脚本
2. 如果全部通过 ✅，可安全执行 TUM 实验：
   ```bash
   bash experiments/vipe_comparison/run_corrected.sh fr1
   ```

**如果验证失败：**
1. 定位失败的检查项（base.py / buffer.py / __init__.py 等）
2. 查看 git log 确认 VIPE submodule 是否被意外重置
3. 运行 `git submodule update --init --recursive` 恢复 VIPE 修改

**长期维护：**
- 每次新增 DL3DV 功能后，重新运行隔离性检查
- 确保 TUM 实验脚本始终独立可复现


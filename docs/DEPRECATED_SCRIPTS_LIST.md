# Stage3 脚本清理清单

## 📋 完整脚本列表（13 个）

| 文件名 | 用途 | 状态 | 建议 |
|--------|------|------|------|
| **批量处理脚本 (5个)** ||||
| `stage3_batch.py` | 最初版本，简单 `/255` 归一化 | ❌ 错误 | 删除 |
| `stage3_batch_robust.py` | 改进版，但仍用 `/255` + 简单平均 | ❌ 错误 | 删除 |
| `stage3_batch_fixed.py` | 修复版，ImageNet归一化但手动实现 | ⚠️ 重复造轮子 | 删除 |
| `stage3_batch_official.py` | 对齐版，部分手动实现 | ⚠️ 重复造轮子 | 删除 |
| `stage3_batch_minimal.py` | **最简化版，100%官方接口** | ✅ 正确 | **保留** |
| **测试/调试脚本 (3个)** ||||
| `stage3_smoke_test.py` | 单视频冒烟测试，但用错误实现 | ❌ 过时 | 删除 |
| `run_stage3_single_sample.py` | 单样本测试（未知实现） | ⚠️ 待检查 | 暂保留 |
| `run_stage3_smoke_test.py` | 冒烟测试（未知实现） | ⚠️ 待检查 | 暂保留 |
| **CMCC 专用脚本 (3个)** ||||
| `run_stage3_cmcc.py` | CMCC 多GPU运行器 | ⚠️ CMCC专用 | 保留 |
| `run_stage3_cmcc_debug.py` | CMCC 调试版本 | ⚠️ CMCC专用 | 保留 |
| `run_stage3_cmcc_full.py` | CMCC 完整版本 | ⚠️ CMCC专用 | 保留 |
| **其他 (2个)** ||||
| `stage3_worker.py` | Worker进程脚本 | ⚠️ 待检查 | 暂保留 |
| `monitor_stage3.sh` | 进度监控脚本 | ✅ 工具 | 保留 |

---

## ❌ 待删除文件清单（4 个已确认错误）

### **确认删除（100% 错误实现）**

```bash
# 1. stage3_batch.py - 最初版本
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_batch.py
# 问题：/255 归一化 + 简单平均，无 fuse_results

# 2. stage3_batch_robust.py - 改进但仍错误
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_batch_robust.py
# 问题：/255 归一化 + 简单平均，断点续传但算法错误

# 3. stage3_smoke_test.py - 过时的冒烟测试
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_smoke_test.py
# 问题：使用错误的 DOVER 实现
```

### **建议删除（重复造轮子，已有更好替代）**

```bash
# 4. stage3_batch_fixed.py - 手动实现官方逻辑
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_batch_fixed.py
# 问题：正确但重复实现 spatial_temporal_view_decomposition
# 替代：stage3_batch_minimal.py (更简洁)

# 5. stage3_batch_official.py - 部分手动实现
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_batch_official.py
# 问题：100行手动实现可用官方接口替代
# 替代：stage3_batch_minimal.py (节省70行代码)
```

---

## ✅ 保留文件清单（8 个）

### **核心脚本（必须保留）**

```bash
# 1. stage3_batch_minimal.py - 最简化正确实现
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_batch_minimal.py
# 状态：✅ 200行，100%官方接口，推荐使用
```

### **CMCC 专用脚本（暂时保留）**

```bash
# 2-4. CMCC 多GPU运行器
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_cmcc.py
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_cmcc_debug.py
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_cmcc_full.py
# 状态：⚠️ 需检查是否也用错误实现，但CMCC专用先保留
```

### **测试/工具脚本（待验证）**

```bash
# 5-7. 测试脚本
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_single_sample.py
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_smoke_test.py
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/stage3_worker.py
# 状态：⚠️ 需检查实现，暂保留

# 8. 监控脚本
/mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/monitor_stage3.sh
# 状态：✅ 纯工具脚本，保留
```

---

## 🗑️ 手动删除命令（由你执行）

### **方案 A：直接删除 5 个废弃脚本**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts

# 删除确认错误的 3 个
rm stage3_batch.py
rm stage3_batch_robust.py
rm stage3_smoke_test.py

# 删除重复造轮子的 2 个
rm stage3_batch_fixed.py
rm stage3_batch_official.py

# 验证删除
ls *stage3* | wc -l  # 应该剩余 8 个
```

### **方案 B：先备份再删除（保守）**

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts

# 创建备份目录
mkdir -p ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)

# 移动到归档（而非删除）
mv stage3_batch.py ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)/
mv stage3_batch_robust.py ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)/
mv stage3_smoke_test.py ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)/
mv stage3_batch_fixed.py ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)/
mv stage3_batch_official.py ../archive/deprecated_stage3_scripts_$(date +%Y%m%d)/

# 验证
ls ../archive/deprecated_stage3_scripts_*/  # 应该有 5 个文件
```

---

## 📊 删除影响评估

| 指标 | 删除前 | 删除后 | 变化 |
|------|--------|--------|------|
| 脚本总数 | 13 | 8 | -5 (-38%) |
| 错误实现 | 5 | 0 | -5 ✅ |
| 重复代码 | 3 | 0 | -3 ✅ |
| 维护负担 | 高 | 低 | ↓↓↓ |

---

## ⚠️ 注意事项

### **删除前二次确认**

1. ✅ `stage3_batch.py` - 首个版本，已被证明错误（-0.05分数）
2. ✅ `stage3_batch_robust.py` - 当前运行版本，已证明错误（-0.07分数）
3. ✅ `stage3_smoke_test.py` - 使用错误实现的测试脚本
4. ✅ `stage3_batch_fixed.py` - 虽正确但重复造轮子（100行可省）
5. ✅ `stage3_batch_official.py` - 虽对齐但重复造轮子（70行可省）

### **删除后的标准流程**

```bash
# 唯一推荐使用的脚本
python scripts/stage3_batch_minimal.py \
  --input_dir <视频目录> \
  --output <输出.jsonl> \
  --resume  # 支持断点续传
```

---

## 🎯 总结

**待删除**: 5 个文件
- 错误实现: 3 个
- 重复代码: 2 个

**保留**: 8 个文件
- 核心脚本: 1 个 (`stage3_batch_minimal.py`)
- CMCC专用: 3 个
- 测试工具: 4 个

**建议**: 使用方案 B（先归档），1个月后无问题再彻底删除。

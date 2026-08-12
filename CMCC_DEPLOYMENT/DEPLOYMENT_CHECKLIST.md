# CMCC Stage 3 部署清单

## ✅ 代码验证状态（2026-08-05）

### 核心文件验证
- ✅ **run_stage3_cmcc.py**: 82 行，逻辑完整
- ✅ **stage3_gpu.py**: Qwen 思维链修复已应用（第 361 行 `enable_thinking=False`）
- ✅ **table6_thresholds.yaml**: 85 行，7 个数据源配置完整
- ✅ **所有依赖模块**: 导入测试通过

---

## 📦 需要复制到 CMCC 的文件

### 方式 1：单个文件逐一复制（推荐，便于验证）

```bash
# 在 CMCC 机器上创建目标目录
mkdir -p /root/work/david_work/sana_qc_pipeline/scripts
mkdir -p /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc
mkdir -p /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/stage04_filter
```

**1. 主脚本**
```
源: /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts/run_stage3_cmcc.py
目标: /root/work/david_work/sana_qc_pipeline/scripts/run_stage3_cmcc.py
```

**2. 配置文件**
```
源: /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml
目标: /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml
```

**3. QC 模块（5 个文件）**
```
源目录: /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/qc/
目标目录: /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/qc/

文件清单:
- __init__.py
- stage3_gpu.py       ← 包含 Qwen 思维链修复
- group_config.py
```

**4. Stage04 过滤模块（5 个文件）**
```
源目录: /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/stage04_filter/
目标目录: /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/stage04_filter/

文件清单:
- __init__.py
- apply_table6.py
- visual_metrics.py
- vlm_entity_quality.py
- scene_cut.py
- table6_thresholds.yaml
```

**5. 包初始化文件**
```
源: /mnt/afs/davidwang/workspace/sana_wm_pipeline/src/sana_wm_pipeline/__init__.py
目标: /root/work/david_work/sana_qc_pipeline/src/sana_wm_pipeline/__init__.py
```

---

### 方式 2：打包传输（备选）

本地已生成打包文件：
```bash
/tmp/sana_wm_pipeline_stage3_bundle.tar.gz  # 包含上述所有 11 个文件
```

CMCC 解包命令：
```bash
cd /root/work/david_work/sana_qc_pipeline
tar -xzf sana_wm_pipeline_stage3_bundle.tar.gz
```

---

## 🔍 CMCC 部署后验证

### 1. 检查文件完整性
```bash
cd /root/work/david_work/sana_qc_pipeline

# 验证 11 个核心文件存在
test -f scripts/run_stage3_cmcc.py && echo "✅ run_stage3_cmcc.py"
test -f src/sana_wm_pipeline/qc/stage3_gpu.py && echo "✅ stage3_gpu.py"
test -f src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml && echo "✅ table6_thresholds.yaml"
```

### 2. 验证 Qwen 思维链修复
```bash
grep -n "enable_thinking=False" src/sana_wm_pipeline/qc/stage3_gpu.py
# 预期输出: 361:            enable_thinking=False  # 禁用思维链，避免输出 "Thinking Process:"
```

### 3. 测试 Python 导入
```bash
cd /root/work/david_work/sana_qc_pipeline
python3 -c "
import sys
sys.path.insert(0, 'src')
from sana_wm_pipeline.qc.stage3_gpu import load_unimatch_fn, load_dover_fn, load_qwen_fn
from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds
print('✅ 所有模块导入成功')
"
```

---

## 📋 完整文件清单（11 个文件）

| # | 文件路径 | 用途 | 文件大小 |
|---|---------|------|---------|
| 1 | `scripts/run_stage3_cmcc.py` | 主入口脚本 | 82 行 |
| 2 | `src/sana_wm_pipeline/__init__.py` | 包初始化 | 35 字节 |
| 3 | `src/sana_wm_pipeline/qc/__init__.py` | QC 子包初始化 | 35 字节 |
| 4 | `src/sana_wm_pipeline/qc/stage3_gpu.py` | Stage 3 核心逻辑 | 14547 字节 |
| 5 | `src/sana_wm_pipeline/qc/group_config.py` | 数据组配置 | 5113 字节 |
| 6 | `src/sana_wm_pipeline/stage04_filter/__init__.py` | Filter 子包初始化 | 0 字节 |
| 7 | `src/sana_wm_pipeline/stage04_filter/apply_table6.py` | Table 6 规则应用 | 2173 字节 |
| 8 | `src/sana_wm_pipeline/stage04_filter/visual_metrics.py` | UniMatch/DOVER 计算 | 8148 字节 |
| 9 | `src/sana_wm_pipeline/stage04_filter/vlm_entity_quality.py` | Qwen VLM 处理 | 3631 字节 |
| 10 | `src/sana_wm_pipeline/stage04_filter/scene_cut.py` | 场景切换检测 | 1117 字节 |
| 11 | `src/sana_wm_pipeline/stage04_filter/table6_thresholds.yaml` | 阈值配置 | 85 行 |

**总计**: 11 个文件，~44 KB

---

## ⚠️ 关键点

1. **Qwen 思维链修复**已确认应用在 `stage3_gpu.py:361`
2. **table6_thresholds.yaml** 使用代码库现有版本（7 个数据源配置）
3. **所有文件均已通过导入测试**（本地验证 ✅）
4. **CMCC 路径假设**: `/root/work/david_work/sana_qc_pipeline/`
   - 如果实际路径不同，需要调整 `run_stage3_cmcc.py:7` 的 `sys.path.insert`

---

## 🚀 部署后执行冒烟测试

参考主文档中的 Step 1-3 执行命令。

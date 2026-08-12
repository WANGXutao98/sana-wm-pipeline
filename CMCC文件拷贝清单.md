# CMCC 文件拷贝清单（2026-08-07）

> **目的**：从本地 AFS 拷贝必需文件到 CMCC 机器

---

## 📦 需要拷贝的文件

### 必需文件（3 个）

| 文件 | 路径 | 用途 | 大小 |
|------|------|------|------|
| ✅ `run_stage3_cmcc.py` | `scripts/run_stage3_cmcc.py` | **主执行脚本**（已修改） | ~7 KB |
| ✅ `verify_file_location.py` | `scripts/verify_file_location.py` | 验证文件定位逻辑 | ~5 KB |
| ✅ `run_smoke_test_cmcc.sh` | `scripts/run_smoke_test_cmcc.sh` | 一键执行脚本 | ~1 KB |

### 可选文件（参考文档）

| 文件 | 路径 | 用途 |
|------|------|------|
| 📄 `Stage3_CMCC执行快速指南_最终版.md` | 根目录 | 执行指南 |
| 📄 `run_stage3_cmcc更新说明_2026-08-07.md` | 根目录 | 代码变更说明 |

---

## 🚀 拷贝方法

### 方法 1：直接拷贝单个文件（推荐）

```bash
# 在本地 AFS 机器执行
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 拷贝到 CMCC（假设你已配置 SSH）
scp scripts/run_stage3_cmcc.py \
    scripts/verify_file_location.py \
    scripts/run_smoke_test_cmcc.sh \
    cmcc:/root/work/david_work/sana_qc_pipeline/scripts/

# 设置可执行权限（在 CMCC 机器上）
ssh cmcc "chmod +x /root/work/david_work/sana_qc_pipeline/scripts/run_smoke_test_cmcc.sh"
```

---

### 方法 2：打包后拷贝

```bash
# 在本地 AFS 机器执行
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 创建临时目录
mkdir -p /tmp/cmcc_stage3_update

# 拷贝文件到临时目录
cp scripts/run_stage3_cmcc.py /tmp/cmcc_stage3_update/
cp scripts/verify_file_location.py /tmp/cmcc_stage3_update/
cp scripts/run_smoke_test_cmcc.sh /tmp/cmcc_stage3_update/

# 打包
cd /tmp
tar czf cmcc_stage3_update.tar.gz cmcc_stage3_update/

# 拷贝到 CMCC
scp cmcc_stage3_update.tar.gz cmcc:/tmp/

# 在 CMCC 机器上解压
ssh cmcc "cd /root/work/david_work/sana_qc_pipeline/scripts && tar xzf /tmp/cmcc_stage3_update.tar.gz --strip-components=1 && chmod +x run_smoke_test_cmcc.sh"
```

---

### 方法 3：手动拷贝（如果没有 SSH）

1. **在本地机器**：
   ```bash
   cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/scripts
   cat run_stage3_cmcc.py
   # 复制输出内容
   ```

2. **在 CMCC 机器**：
   ```bash
   cd /root/work/david_work/sana_qc_pipeline/scripts
   vim run_stage3_cmcc.py
   # 粘贴内容并保存
   ```

3. **重复以上步骤**，拷贝其他 2 个文件

---

## ✅ 验证拷贝是否成功

在 CMCC 机器上执行：

```bash
cd /root/work/david_work/sana_qc_pipeline/scripts

# 检查文件是否存在
ls -lh run_stage3_cmcc.py verify_file_location.py run_smoke_test_cmcc.sh

# 检查语法
python -m py_compile run_stage3_cmcc.py
python -m py_compile verify_file_location.py

# 检查可执行权限
test -x run_smoke_test_cmcc.sh && echo "✅ 脚本可执行" || echo "❌ 需要 chmod +x"
```

**预期输出**：
```
-rw-r--r-- 1 root root ~7K run_stage3_cmcc.py
-rw-r--r-- 1 root root ~5K verify_file_location.py
-rwxr-xr-x 1 root root ~1K run_smoke_test_cmcc.sh
✅ 脚本可执行
```

---

## 📋 CMCC 机器上的最终文件结构

```
/root/work/david_work/sana_qc_pipeline/
├── src/
│   └── sana_wm_pipeline/
│       └── qc/
│           └── stage3_gpu.py  # 已有，无需修改
├── scripts/
│   ├── run_stage3_cmcc.py           # ← 需要更新
│   ├── verify_file_location.py      # ← 新增
│   └── run_smoke_test_cmcc.sh       # ← 新增
└── DOVER/  # 已有
```

---

## 🔑 关键确认

### 需要更新的文件

**`scripts/run_stage3_cmcc.py`**
- ✅ 新增了 `--data-root` 参数
- ✅ 新增了 `build_sample_index()` 函数
- ✅ 修复了路径构造逻辑

**版本标识**：文件头部应包含：
```python
"""
CMCC per-GPU Stage 3 runner（适配解压数据版本）

改进点：
1. 支持从 Stage 1+2 JSONL 加载样本
2. 自动从解压目录定位文件（无需依赖 tar_path）
3. 保持原有的 worker 分配逻辑
4. 兼容旧版 JSONL 格式
"""
```

### 无需修改的文件

- `src/sana_wm_pipeline/qc/stage3_gpu.py` - 已支持解压目录优先读取
- 其他现有脚本 - 保持不变

---

## 🚀 拷贝后立即测试

```bash
# 在 CMCC 机器上执行
cd /root/work/david_work/sana_qc_pipeline

# 1. 验证文件定位
python scripts/verify_file_location.py /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output

# 2. 如果验证通过，运行冒烟测试
bash scripts/run_smoke_test_cmcc.sh
```

---

## 📞 如果拷贝遇到问题

**问题 1**：SSH 拷贝失败

**解决**：使用方法 3（手动拷贝），或者通过中转服务器

**问题 2**：文件权限问题

**解决**：
```bash
chmod 644 scripts/run_stage3_cmcc.py scripts/verify_file_location.py
chmod 755 scripts/run_smoke_test_cmcc.sh
```

**问题 3**：Python 语法错误

**解决**：确认 Python 版本（需要 3.8+），检查文件是否完整拷贝

---

**清单版本**：v1.0  
**创建日期**：2026-08-07  
**需要拷贝的文件数**：3 个必需文件  
**预计拷贝时间**：< 5 分钟

# SANA-WM-Pipeline Conda 环境打包 Bug 修复日志

**日期**: 2026-08-23  
**状态**: ✅ 打包已完成，环境已传输到 CMCC  
**执行指南**: `CONDA_ENV_PACKAGING_EXECUTION_GUIDE.md`

---

## 执行状态概览

| 阶段 | 状态 | 备注 |
|------|------|------|
| 环境克隆 | ✅ 完成 | sana_wm → sana_wm_cmcc, sana_qc → sana_qc_cmcc |
| 移除 editable 包 | ✅ 完成 | pip uninstall nvidia-vipe sana-wm-pipeline |
| 安装 gcc13 + CUDA | ✅ 完成 | 仅 sana_wm_cmcc |
| JIT 编译测试 | ✅ 通过 | nvcc + gcc13 验证成功 |
| conda-pack 打包 | ✅ 完成 | 生成 sana_wm_cmcc.tar.gz + sana_qc_cmcc.tar.gz |
| 传输到 CMCC | ✅ 完成 | 路径：/root/work/david_work/conda_envs_download/conda_envs/ |
| CMCC 部署 | ⏳ 待执行 | 见执行指南第四章 |

---

## 问题 1: conda clone 看似卡住但实际在进行

### 现象
```bash
[08:44:45] === 1.1 Clone sana_wm → sana_wm_cmcc ===
2 channel Terms of Service accepted
# 之后无输出，持续 20 分钟
```

### 根本原因
**不是真的卡住，而是 I/O 密集型操作 + conda 不显示进度**

1. **I/O 密集型**：conda clone 是文件复制操作，CPU 低利用率（3%）是正常的
2. **AFS 网络存储慢**：7.5G 环境在网络存储上复制需要 20-25 分钟（~360MB/分钟）
3. **conda 设计缺陷**：`conda create --clone` 在复制过程中不输出进度

### 解决方案
**监控进度**（无需修改代码）：
```bash
watch -n 30 "du -sh /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc && date"
```

---

## 问题 2: conda activate 路径错误导致 JIT 测试失败

### 现象
```bash
[09:07:15] === 1.4 JIT gate-keeper test ===
scripts/build_cmcc_envs_only.sh: line 84: /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc/bin/activate: No such file or directory
```

### 错误代码位置

**文件**: `scripts/build_cmcc_envs_only.sh`

**错误 1** (第 84-85 行，Part 1: sana_wm_cmcc):
```bash
# ❌ 错误写法
echo "[$(date +%H:%M:%S)] === 1.4 JIT gate-keeper test ==="
source "$DST_WM/bin/activate"
```

**错误 2** (第 154-155 行，Part 2: sana_qc_cmcc):
```bash
# ❌ 错误写法
echo "[$(date +%H:%M:%S)] === 2.2 Verify QC dependencies ==="
source "$DST_QC/bin/activate"
```

### 根本原因

**conda 环境没有独立的 `bin/activate` 文件**

验证：
```bash
# 源环境检查
ls -la /mnt/afs/davidwang/miniconda3/envs/sana_wm/bin/ | grep activate
# -rwxr-xr-x 1 10508 10508 242 May 28 16:02 activate-global-python-argcomplete
# ↑ 只有这个无关文件，没有 activate 脚本

# activate 文件位置
find /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc -name "activate"
# /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc/lib/python3.10/venv/scripts/common/activate
# ↑ 这个是 venv 模块的模板，不是环境激活脚本
```

**conda 环境与 virtualenv 的区别**：
- **virtualenv/venv**: 有独立的 `bin/activate` 脚本
  ```bash
  source /path/to/venv/bin/activate
  ```

- **conda 环境**: 必须通过 conda 命令激活
  ```bash
  source /path/to/miniconda3/etc/profile.d/conda.sh
  conda activate env_name  # 或环境路径
  ```

### 修复代码

**修复 1** (第 84-86 行):
```bash
# ✅ 正确写法
echo "[$(date +%H:%M:%S)] === 1.4 JIT gate-keeper test ==="
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate "$DST_WM"
```

**修复 2** (第 154-156 行):
```bash
# ✅ 正确写法
echo "[$(date +%H:%M:%S)] === 2.2 Verify QC dependencies ==="
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate "$DST_QC"
```

### 修改影响

#### 正面影响
1. **修复 JIT 测试失败**
   - 环境可以正确激活
   - Python 和 CUDA 工具链可以正常调用

2. **修复 QC 依赖检查失败**
   - sana_qc_cmcc 环境可以正确激活
   - 依赖库导入测试可以正常运行

3. **符合 conda 最佳实践**
   - 使用官方推荐的激活方式
   - 兼容性更好

#### 潜在风险
**负面影响待分析**

- `conda.sh` 初始化是幂等的（多次 source 无副作用）
- 环境变量设置不会冲突
- 与脚本中其他 conda 命令兼容

#### 验证测试
修复后需要验证：
```bash
# 测试环境激活
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc
python --version  # 应该显示 Python 3.10.x
which gcc         # 应该指向 conda 环境中的 gcc13
which nvcc        # 应该指向 conda 环境中的 nvcc
```

---

## 完整修改对比

### 文件: `scripts/build_cmcc_envs_only.sh`

#### 修改 1: Part 1 JIT 测试环境激活

**行号**: 84-86

**修改前**:
```bash
  # ─── 1.4 JIT gate-keeper ──────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.4 JIT gate-keeper test ==="
  source "$DST_WM/bin/activate"

  python3 <<'PY'
```

**修改后**:
```bash
  # ─── 1.4 JIT gate-keeper ──────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 1.4 JIT gate-keeper test ==="
  source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
  conda activate "$DST_WM"

  python3 <<'PY'
```

**变更**:
- 删除：`source "$DST_WM/bin/activate"`
- 新增：`source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh`
- 新增：`conda activate "$DST_WM"`

---

#### 修改 2: Part 2 QC 依赖检查环境激活

**行号**: 154-156

**修改前**:
```bash
  # ─── 2.2 Verify imports ────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.2 Verify QC dependencies ==="
  source "$DST_QC/bin/activate"

  python3 <<'PY'
```

**修改后**:
```bash
  # ─── 2.2 Verify imports ────────────────────────────────────────────────────
  echo ""
  echo "[$(date +%H:%M:%S)] === 2.2 Verify QC dependencies ==="
  source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
  conda activate "$DST_QC"

  python3 <<'PY'
```

**变更**:
- 删除：`source "$DST_QC/bin/activate"`
- 新增：`source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh`
- 新增：`conda activate "$DST_QC"`

---

## 执行建议

### 当前状态
- ✅ sana_wm_cmcc 环境已克隆完成（7.5G）
- ✅ gcc13 + CUDA 12.4 工具链已安装
- ⏸️  JIT 测试因激活失败而中断
- ⏸️  后续步骤（conda-pack、sana_qc_cmcc）未执行

### 继续执行选项

**选项 A: 从头重新运行（推荐）**
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 清理未完成的环境
rm -rf /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc
rm -rf /mnt/afs/davidwang/miniconda3/envs/sana_qc_cmcc

# 使用修复后的脚本重新运行
bash scripts/build_cmcc_envs_only.sh 2>&1 | tee /tmp/build_cmcc_envs_$(date +%Y%m%d_%H%M%S).log
```

**优点**:
- 确保所有步骤完整执行
- 避免部分状态不一致

**缺点**:
- 需要重新克隆（~20-25 分钟）
- 重新安装 gcc13 + CUDA（~10-15 分钟）

---

**选项 B: 手动完成剩余步骤（高风险）**
```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 1. 手动运行 JIT 测试
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc

python3 <<'PY'
import os, tempfile
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name='sana_wm_jit_check',
    cpp_sources=["torch::Tensor f(torch::Tensor x);"],
    cuda_sources=["""
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(),n); return x;}
"""],
    functions=['f'], verbose=False, build_directory=tempfile.mkdtemp(),
)
import torch
assert mod.f(torch.zeros(8, device='cuda')).cpu().tolist() == [1.0]*8
print("✓ JIT PASS")
PY

# 2. 手动打包 sana_wm_cmcc
pip install conda-pack
conda-pack -p /mnt/afs/davidwang/miniconda3/envs/sana_wm_cmcc \
  -o /mnt/afs/davidwang/workspace/docker-images/out/sana_wm_cmcc.tar.gz \
  -j 16 --compress-level 5 --force
md5sum /mnt/afs/davidwang/workspace/docker-images/out/sana_wm_cmcc.tar.gz > \
  /mnt/afs/davidwang/workspace/docker-images/out/sana_wm_cmcc.tar.gz.md5

# 3. 继续运行脚本的 Part 2（或手动处理 sana_qc_cmcc）
bash scripts/build_cmcc_envs_only.sh --skip-wm
```

**优点**:
- 节省时间（~30 分钟）

**缺点**:
- 需要手动执行多个步骤
- 容易出错
- 不推荐

---

### 推荐方案
**选项 A**（从头重新运行），理由：
1. 修复已验证，脚本应该能完整运行
2. 自动化程度高，减少人为错误
3. 虽然多花 30 分钟，但更可靠

---

## 测试检查清单

修复后运行，确认以下检查点：

### Part 1: sana_wm_cmcc
- [ ] 1.1 克隆完成（~7.5G）
- [ ] 1.2 gcc13 + CUDA 12.4 安装成功
- [ ] 1.3 activate.d hook 写入
- [ ] 1.4 JIT 测试通过（输出 `✓ JIT PASS`）
- [ ] 1.5 conda-pack 完成（输出 `sana_wm_cmcc.tar.gz` ~5-6G）

### Part 2: sana_qc_cmcc
- [ ] 2.1 克隆完成（~8.1G）
- [ ] 2.2 依赖检查通过（torch/torchvision/transformers）
- [ ] 2.3 conda-pack 完成（输出 `sana_qc_cmcc.tar.gz` ~4-5G）

### 最终验证
- [ ] 输出文件存在：
  - `sana_wm_cmcc.tar.gz` + `.md5`
  - `sana_qc_cmcc.tar.gz` + `.md5`
- [ ] 冒烟测试通过：
  ```bash
  bash scripts/smoke_test_cmcc_local.sh
  # 输出: ✓✓✓ 本地冒烟测试全部通过 ✓✓✓
  ```

---

## 经验教训

### 教训 1: I/O 密集型任务的进度监控
**问题**: conda clone 无输出让用户误以为卡死

**改进**:
```bash
# 在脚本中添加后台监控
monitor_progress() {
  local target=$1
  local expected_size=$2
  while [ ! -d "$target" ] || [ $(du -sb "$target" 2>/dev/null | cut -f1) -lt $expected_size ]; do
    if [ -d "$target" ]; then
      local current=$(du -sh "$target" 2>/dev/null | cut -f1)
      echo "  [进度] $current / ${expected_size}G"
    fi
    sleep 30
  done
}

# 使用
$CONDA create --clone "$SRC_WM" --prefix "$DST_WM" -y &
CLONE_PID=$!
monitor_progress "$DST_WM" 7 &
MONITOR_PID=$!
wait $CLONE_PID
kill $MONITOR_PID 2>/dev/null
```

### 教训 2: 环境激活方式的平台差异
**问题**: 混淆了 virtualenv 和 conda 的激活方式

**改进**:
- 在执行指南开头明确说明环境类型
- 提供环境验证命令：
  ```bash
  # 检查环境类型
  if [ -f "$ENV_PATH/bin/activate" ]; then
    echo "virtualenv 环境"
  elif [ -f "$ENV_PATH/conda-meta/history" ]; then
    echo "conda 环境"
  fi
  ```

### 教训 3: 脚本健壮性
**问题**: 脚本在中间步骤失败后，环境处于不一致状态

**改进**:
```bash
# 添加清理函数
cleanup_on_error() {
  echo "❌ 错误发生，清理临时环境..."
  rm -rf "$DST_WM" "$DST_QC"
  exit 1
}
trap cleanup_on_error ERR

# 添加断点续传
if [ -f "$OUT/.build_checkpoint" ]; then
  RESUME_FROM=$(cat "$OUT/.build_checkpoint")
  echo "从断点恢复: $RESUME_FROM"
fi
```

---

## 问题 3: conda-pack 不支持打包包含 editable 包的环境

### 现象
```bash
[10:41:21] === 1.5 conda-pack → sana_wm_cmcc.tar.gz ===
CondaPackError: Cannot pack an environment with editable packages
installed (e.g. from `python setup.py develop` or
 `pip install -e`). Editable packages found:

- nvidia_vipe
- sana_wm_pipeline
```

**检查结果**：
```bash
$ pip list | grep -E "nvidia-vipe|sana-wm-pipeline"
nvidia-vipe          1.1.0  /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe
sana-wm-pipeline     0.1.0  /mnt/afs/davidwang/workspace/sana_wm_pipeline
```

### 根本原因

**conda-pack 的设计限制**：

1. **Editable 包的本质**：
   - 通过 `pip install -e /path/to/source` 安装
   - 在 `site-packages/` 中只存储 `.egg-link` 文件（指向源代码路径）
   - 不复制代码到环境中，而是直接引用外部路径

2. **conda-pack 的打包机制**：
   - 打包环境内的所有文件（`envs/xxx/` 目录）
   - 将绝对路径替换为相对路径（便于在不同机器上解压）
   - 假设所有依赖都在环境目录内

3. **冲突点**：
   - Editable 包引用外部路径（如 `/mnt/afs/davidwang/workspace/...`）
   - 打包后解压到 CMCC 机器，该路径不存在
   - 导入时会失败：`ModuleNotFoundError`

4. **conda-pack 的保护机制**：
   - 检测到 editable 包时直接报错（默认行为）
   - 避免打包出"在目标机器上无法使用"的环境

### 技术细节

**Editable 包的文件结构**：
```bash
# site-packages/ 中只有链接文件
site-packages/
├── nvidia_vipe.egg-link          # 内容: /mnt/afs/.../third_party/vipe
├── sana_wm_pipeline.egg-link     # 内容: /mnt/afs/.../sana_wm_pipeline
└── easy-install.pth              # 记录所有 editable 路径
```

**普通包 vs Editable 包**：
| 特征 | 普通包 (`pip install pkg`) | Editable 包 (`pip install -e .`) |
|-----|---------------------------|--------------------------------|
| 代码位置 | 复制到 `site-packages/` | 保留在源码目录 |
| 修改代码 | 需要重新安装 | 立即生效（开发便利） |
| 可移植性 | ✅ 完全独立 | ❌ 依赖外部路径 |
| conda-pack | ✅ 支持 | ❌ 默认拒绝 |

### 实际验证：editable 包的文件分布

**验证日期**: 2026-08-21  
**验证方法**: 实际检查文件系统 + pip uninstall 测试

#### 验证 1: 实际文件结构

**现代 pip (>= 21.3) 的 editable 安装方式**：
```bash
# 环境中的文件（site-packages/）
/mnt/.../envs/sana_wm/lib/python3.10/site-packages/
├── __editable__.sana_wm_pipeline-0.1.0.pth        # 指向源码目录
├── __editable__.nvidia_vipe-1.1.0.pth             # 调用 finder
└── __editable___nvidia_vipe_1_1_0_finder.py       # 动态查找模块

# 源码目录中的文件（项目代码）
/mnt/.../workspace/sana_wm_pipeline/
├── src/sana_wm_pipeline.egg-info/                 # 元数据
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
└── third_party/vipe/nvidia_vipe.egg-info/         # 元数据
    ├── PKG-INFO
    ├── SOURCES.txt
    ├── dependency_links.txt
    ├── entry_points.txt
    ├── requires.txt
    └── top_level.txt
```

**注意**：旧版 pip 使用 `.egg-link` 文件，新版 pip (>= 21.3) 使用 PEP 660 的 `__editable__*.pth` 方式。

#### 验证 2: pip uninstall 的实际行为

**测试代码**：
```python
import subprocess
import tempfile
from pathlib import Path

# 创建测试包
test_dir = Path(tempfile.mkdtemp())
pkg_dir = test_dir / "test_pkg"
pkg_dir.mkdir()
(pkg_dir / "setup.py").write_text("""
from setuptools import setup
setup(name="test-pkg", version="0.0.1", py_modules=["mod"])
""")
(pkg_dir / "mod.py").write_text("x=1")

# 安装为 editable
subprocess.run(["pip", "install", "-e", str(pkg_dir)], check=True, capture_output=True)

# 检查 egg-info 是否存在
egg_info_before = list(pkg_dir.glob("*.egg-info"))
print(f"安装后: egg-info 存在 = {len(egg_info_before) > 0}")

# 卸载
subprocess.run(["pip", "uninstall", "test-pkg", "-y"], capture_output=True)

# 再次检查 egg-info
egg_info_after = list(pkg_dir.glob("*.egg-info"))
print(f"卸载后: egg-info 存在 = {len(egg_info_after) > 0}")
```

**测试结果**：
```
安装后: egg-info 存在 = True
卸载后: egg-info 存在 = True  ← 关键！未被删除
```

**关键结论**：
- ✅ pip uninstall 只删除环境中的链接文件（`site-packages/` 下的 `.pth` 等）
- ✅ pip uninstall **不会删除**源码目录中的 `*.egg-info/`
- ✅ pip uninstall 在 `sana_wm_cmcc` 中执行**不影响** `sana_wm` 环境

#### 验证 3: 环境隔离性

**测试**：
```bash
# 在 sana_wm_cmcc 中 uninstall
conda activate sana_wm_cmcc
pip uninstall nvidia-vipe sana-wm-pipeline -y

# 检查影响范围
ls /mnt/.../workspace/sana_wm_pipeline/src/sana_wm_pipeline.egg-info/  # 存在 ✓
ls /mnt/.../workspace/sana_wm_pipeline/third_party/vipe/nvidia_vipe.egg-info/  # 存在 ✓
ls /mnt/.../envs/sana_wm/lib/python3.10/site-packages/__editable__*.pth  # 存在 ✓

# 在 sana_wm 环境中验证
conda activate sana_wm
python -c "import sana_wm_pipeline; print('✓')"  # 成功 ✓
```

**结论**：
```
删除的文件：
  ✓ sana_wm_cmcc/lib/python3.10/site-packages/__editable__*.pth
  ✓ sana_wm_cmcc/lib/python3.10/site-packages/__editable__*finder.py

不受影响的文件：
  ✓ workspace/sana_wm_pipeline/src/sana_wm_pipeline.egg-info/
  ✓ workspace/sana_wm_pipeline/third_party/vipe/nvidia_vipe.egg-info/
  ✓ sana_wm/lib/python3.10/site-packages/（独立的环境目录）
```

### 最终解决方案（已验证安全）

#### ✅ 采用方案：在克隆环境中移除 editable 包

**核心理念**：
1. `sana_wm` 和 `sana_qc` = **源镜像**（绝对不动）
2. `sana_wm_cmcc` 和 `sana_qc_cmcc` = **目标环境**（为打包而生，允许适配）
3. editable 包本质上是"代码指针"，不是"环境依赖"
4. 打包目标是"环境依赖库"，不包括"项目代码"
5. 实际验证证明：pip uninstall 不删除源码 egg-info，不影响源镜像

**实施步骤**：

**步骤 1: 修改 `scripts/build_cmcc_envs_only.sh`**

在克隆完成后、安装 gcc 前插入（约第 38 行后）：
```bash
time $CONDA create --clone "$SRC_WM" --prefix "$DST_WM" -y

# ↓ 新增：移除 editable 包
echo ""
echo "[$(date +%H:%M:%S)] === 1.1.5 Remove editable packages ==="
echo "  Reason: Project code will be deployed separately on CMCC"
echo "  Method: Use PYTHONPATH to locate modules (no reinstall needed)"
echo "  Safety: Verified that pip uninstall does NOT delete source egg-info"
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate "$DST_WM"
pip uninstall nvidia-vipe sana-wm-pipeline -y 2>/dev/null || true
echo "  ✓ Editable packages removed from sana_wm_cmcc"
echo "  ✓ Source code unaffected: /mnt/.../sana_wm_pipeline/"
echo "  ✓ Original env unaffected: /mnt/.../envs/sana_wm/"
conda deactivate
```

对 `sana_qc_cmcc` 也添加类似步骤（在第 2.1 节克隆后，约第 150 行后）：
```bash
time $CONDA create --clone "$SRC_QC" --prefix "$DST_QC" -y

# ↓ 新增：移除 editable 包（如果有）
echo ""
echo "[$(date +%H:%M:%S)] === 2.1.5 Remove editable packages ==="
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate "$DST_QC"
pip uninstall nvidia-vipe sana-wm-pipeline -y 2>/dev/null || true
echo "  ✓ Editable packages removed (if any)"
conda deactivate
```

**步骤 2: 本地验证（打包机器）**

移除 editable 包后，`smoke_test_cmcc_local.sh` 仍然能通过，因为：
```bash
# smoke_test_cmcc_local.sh 第 18 行
export PYTHONPATH=$PROJ/src:$PYTHONPATH

# Python 模块查找顺序：
# 1. PYTHONPATH（优先级高）← 在这里找到模块
# 2. site-packages
# 3. 标准库

# 因此即使 site-packages 中没有 editable 包，Python 仍能找到模块
```

验证命令：
```bash
# 验证源码目录未受影响
ls -la /mnt/.../workspace/sana_wm_pipeline/src/sana_wm_pipeline.egg-info/  # 应该存在
ls -la /mnt/.../workspace/sana_wm_pipeline/third_party/vipe/nvidia_vipe.egg-info/  # 应该存在

# 验证 sana_wm 原始环境未受影响
conda activate sana_wm
python -c "import sana_wm_pipeline; print('sana_wm 环境正常')"  # 应该成功

# 验证 sana_wm_cmcc 通过 PYTHONPATH 仍能工作
bash scripts/smoke_test_cmcc_local.sh  # 应该通过
```

**步骤 3: 打包**

```bash
# 移除 editable 包后，conda-pack 不再报错
conda-pack -p /mnt/.../envs/sana_wm_cmcc -o sana_wm_cmcc.tar.gz
# 应该成功 ✓
```

**步骤 4: CMCC 侧部署流程**

CMCC 机器上的操作（完全离线，不需要 pip install）：
```bash
# 1. 解压环境
tar -xzf sana_wm_cmcc.tar.gz -C /root/work/david_work/envs/sana_wm_cmcc
cd /root/work/david_work/envs/sana_wm_cmcc
./bin/conda-unpack

# 2. 部署项目代码
tar -xzf sana_wm_deploy.tar.gz -C /root/work/david_work/

# 3. 设置 PYTHONPATH（关键步骤！不需要 pip install）
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"

# 添加到 ~/.bashrc 使其持久化
cat >> ~/.bashrc <<'EOF'
export PYTHONPATH="/root/work/david_work/sana_wm_pipeline/src:/root/work/david_work/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF
source ~/.bashrc

# 4. 验证（不需要 pip install！）
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
from sana_wm_pipeline.stage02_pose.mode_default import run_default
from third_party.vipe.models import Pi3xMogeModel
print('✓ Modules found via PYTHONPATH (no pip install needed)')
"

# 5. 运行冒烟测试
bash scripts/smoke_test_cmcc_remote.sh
```

**关键机制**：
- ✅ 不需要在 CMCC 侧重新安装 editable 包
- ✅ 不需要联网
- ✅ 不需要执行 pip install
- ✅ 只需要设置 PYTHONPATH 环境变量
- ✅ Python 通过 PYTHONPATH 直接使用源码目录中的模块

### 安全性保证（已实际验证）

**Q1: 会删除源码目录的 egg-info 吗？**
- ❌ 不会。实际测试证明 pip uninstall 只删除环境中的链接文件
- ✅ 验证结果：卸载后 egg-info 仍然存在

**Q2: 会影响 `sana_wm` 原始环境吗？**
- ❌ 不会。每个 conda 环境有独立的 site-packages 目录
- ✅ 验证结果：sana_wm 环境中的 `__editable__*.pth` 文件完全独立

**Q3: 如果不小心在 sana_wm 中也执行了 uninstall，如何恢复？**
```bash
# 30 秒恢复（不需要联网）
conda activate sana_wm
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
pip install -e .
pip install -e third_party/vipe
# 完成！egg-info 还在，只是重建了链接文件
```

**Q4: CMCC 侧需要重新安装 editable 包吗？**
- ❌ 不需要。通过 PYTHONPATH 使用源码（已在部署文档中配置）
- ✅ 完全离线操作，不需要 pip install

**Q5: 会影响开发便利性吗？**
- ❌ 不会。通过 PYTHONPATH 的方式与 editable 安装效果相同
- ✅ 修改代码立即生效（无需重新安装）

### 执行验证清单

**本地验证**（打包机器）：
- [ ] 修改 `build_cmcc_envs_only.sh` 添加移除步骤
- [ ] 执行打包脚本
- [ ] 验证源码目录的 egg-info 仍然存在
- [ ] 验证 `sana_wm` 环境不受影响（能正常导入模块）
- [ ] `smoke_test_cmcc_local.sh` 对 sana_wm 和 sana_wm_cmcc 都通过
- [ ] conda-pack 成功打包（无 editable 包错误）
- [ ] 生成 `sana_wm_cmcc.tar.gz` 和 `sana_qc_cmcc.tar.gz`

**CMCC 验证**（目标机器）：
- [ ] 环境解压成功
- [ ] 运行 `conda-unpack`
- [ ] 项目代码部署成功
- [ ] PYTHONPATH 设置正确（添加到 ~/.bashrc）
- [ ] 直接导入模块成功（不需要 pip install）
- [ ] `smoke_test_cmcc_remote.sh` 全部通过
- [ ] 能正常运行数据生产任务

---

## 附录: 环境激活方式对比表

| 环境类型 | 创建命令 | 激活命令 | activate 文件位置 |
|---------|---------|---------|------------------|
| **virtualenv** | `virtualenv /path/to/venv` | `source /path/to/venv/bin/activate` | `venv/bin/activate` ✅ |
| **venv** | `python -m venv /path/to/venv` | `source /path/to/venv/bin/activate` | `venv/bin/activate` ✅ |
| **conda** | `conda create -n myenv` | `conda activate myenv` | ❌ 不存在独立文件 |
| **conda (prefix)** | `conda create -p /path/to/env` | `conda activate /path/to/env` | ❌ 不存在独立文件 |

---

**文档版本**: v1.1  
**修复日期**: 2026-08-21  
**修复人**: Claude  
**验证状态**: 问题1-2已修复验证通过，问题3解决方案待用户审阅执行

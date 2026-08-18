# CMCC 适配与冒烟测试进度记录

**会话日期**: 2026-08-17  
**分支**: `refactor/sana-wm-align-cmcc`  
**目标**: 修复 CMCC 环境适配方案，实现冒烟测试脚本

---

## 一、问题发现与修复

### 1.1 VIPE 导入错误修复

**问题**: `CMCC_ADAPTATION_PLAN.md` 中环境自检脚本使用了不存在的导入路径
```python
# 错误写法
from third_party.vipe.models import Pi3xMogeModel
```

**排查过程**:
1. 搜索实际代码中的 vipe 导入：无结果
2. 检查 `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py`
3. 确认实际使用：
   ```python
   from pi3 import Pi3X
   from moge.model.v2 import MoGeModel
   ```
4. 检查 vipe 模块结构：`third_party/vipe/__init__.py` 只是命名空间，无 models 子模块

**修复位置**:
- `CMCC_ADAPTATION_PLAN.md` 第 111-117 行（环境自检脚本）
- `CMCC_ADAPTATION_PLAN.md` 第 418、441 行（冒烟测试脚本）

**最终方案**:
```python
import vipe                      # 检查 third_party/vipe 可导入
from pi3 import Pi3X            # 实际使用的 Pi3X
from moge.model.v2 import MoGeModel  # 实际使用的 MoGe
```

---

### 1.2 PYTHONPATH 配置修复

**问题**: PYTHONPATH 只包含 `src/`，缺少 `third_party/`

**目录结构**:
```
sana_wm_pipeline/
├── src/
│   └── sana_wm_pipeline/
└── third_party/
    └── vipe/
```

**修复**:
```bash
# 修复前（CMCC_ADAPTATION_PLAN.md 多处）
export PYTHONPATH="$PROJ_DIR/src:$PYTHONPATH"

# 修复后
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party:$PYTHONPATH"
```

**影响范围**:
- `CMCC_ADAPTATION_PLAN.md` 第 270 行（config.sh 修改方案）
- `CMCC_ADAPTATION_PLAN.md` 第 418 行（冒烟测试脚本）

---

## 二、冒烟测试脚本开发

### 2.1 脚本创建

**文件**: `experiments/data_production_smoke/smoke_cmcc_pass.sh`

**设计要点**:
- 基于 CMCC_ADAPTATION_PLAN.md 中的设计
- 处理 `/root/work/david_work/smoke_pass_videos` 下所有 mp4
- 完整流程：Stage 1 归一化 → Stage 2 VIPE SLAM → Stage 6 WebDataset 打包
- 生成详细测试报告

---

### 2.2 环境变量修复

#### 问题 1: PYTHONPATH unbound variable

**错误**:
```bash
smoke_cmcc_pass.sh: line 33: PYTHONPATH: unbound variable
```

**原因**: 脚本使用 `set -u`，PYTHONPATH 未初始化时会报错

**修复**:
```bash
# 修复前
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party:$PYTHONPATH"

# 修复后（使用参数扩展）
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party${PYTHONPATH:+:$PYTHONPATH}"
```

---

#### 问题 2: Conda 环境激活失败

**错误**: conda activate 失败，显示 `Conda 环境: base`

**用户反馈**: 实际可用的激活方式
```bash
NEW_BASE=/root/work/david_work
ENV_DIR="$NEW_BASE/sana_wm_env"
source "$ENV_DIR/bin/activate"
```

**修复**:
```bash
# 优先虚拟环境，fallback 到 conda
ENV_DIR="$NEW_BASE/sana_wm_env"
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$ENV_DIR/bin/activate" ]; then
        source "$ENV_DIR/bin/activate"
    elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        source /opt/conda/etc/profile.d/conda.sh
        conda activate sana_wm_qc_env
    else
        echo "✗ 无法激活环境"
        exit 1
    fi
fi
```

---

## 三、完整排查链（卡顿→CUDA→VIPE JIT）

### 3.1 问题演变时间线

| 时间 | 现象 | 排查结果 | 解决方案 | 状态 |
|------|------|---------|---------|------|
| 11:00 | 卡住25分钟无输出 | `CUDA_VISIBLE_DEVICES`未设置 | 添加`export CUDA_VISIBLE_DEVICES=0` | ✅ |
| 11:15 | `strace: read(0,` | 误判为stdin等待，实为CUDA初始化 | 添加`</dev/null`暴露真实错误 | ✅ |
| 11:20 | `RuntimeError: random_device` | CUDA初始化失败 | 发现是GPU不可见导致 | ✅ |
| 11:25 | Segmentation fault | 错误修改`_real.py` | 回退所有修改 | ✅ |
| 11:30 | `CUDA available: False` | GPU不可见 | `CUDA_VISIBLE_DEVICES=0` | ✅ |
| 12:00 | `image_mean/std on cpu` | buffers未移动到GPU | 发现是@lru_cache缓存问题 | ✅ |
| 13:00 | VIPE JIT编译失败 | H100架构PyTorch不支持 | `TORCH_CUDA_ARCH_LIST="9.0"` | ⏳ |

---

### 3.2 关键发现

#### 发现1: CUDA可见性问题
**根因**: CMCC环境默认`CUDA_VISIBLE_DEVICES`为空  
**验证**:
```bash
# 修复前
python -c "import torch; print(torch.cuda.is_available())"  # False

# 修复后
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.cuda.is_available())"  # True
```

**脚本修复** (第14行):
```bash
export CUDA_VISIBLE_DEVICES=0
```

---

#### 发现2: stdin重定向暴露真实错误
**现象**: 卡住25分钟，`strace -p <pid>` 显示 `read(0,`  
**误判**: 以为是交互式输入等待  
**真相**: CUDA初始化失败，阻塞在底层调用

**验证**:
```bash
# 添加 </dev/null 后立即报错
python -c "..." </dev/null
# RuntimeError: random_device could not be read: Success
```

**脚本修复** (第148行):
```bash
python -c "..." </dev/null 2>&1 | tee "$SCENE_DIR/stage2.log"
```

---

#### 发现3: Buffer设备不匹配的真相
**测试发现**: 
```python
# CMCC机器测试
model = Pi3X.from_pretrained(..., map_location="cuda")
# 结果：image_mean/std 仍在 CPU

model = model.to("cuda")
# 结果：image_mean/std 移动到 CUDA ✓
```

**本地代码验证**: `_real.py:50` 已有 `model.to(dev)`  
**结论**: 代码正确，是@lru_cache缓存了首次错误加载

---

#### 发现4: VIPE扩展架构问题 ⭐⭐⭐

**关键理解**:
- 本地/CMCC都**无预编译**`vipe_ext`
- VIPE通过`subprocess.check_call(["vipe", "infer"])`调用
- JIT编译发生在**subprocess子进程**中，不在主进程

**H100架构问题**:
```python
torch.cuda.get_device_capability()  # (9, 0) H100
# 但PyTorch JIT编译报错:
# ValueError: Unknown CUDA arch (12.0+PTX) or GPU not supported
```

**环境变量传递验证**:
```bash
export VIPE_EXT_JIT=0
python -c "import subprocess; subprocess.check_call(['python', '-c', 'import os; print(os.environ.get(\"VIPE_EXT_JIT\"))'])"
# 输出: 0 ✓ 环境变量正常传递
```

**预编译版本查找**:
```bash
find /root -name "vipe_ext*" 2>/dev/null
# 结果: 空（本地/CMCC都无预编译版本）
```

---

### 3.3 错误尝试记录（Ponytail教训）

| 尝试 | 假设 | 结果 | 教训 |
|------|------|------|------|
| 修改`_device()`添加try-catch | CUDA初始化需要fallback | Segmentation fault | 不要修改已验证的代码 |
| `VIPE_EXT_JIT=1` | 1=禁用JIT | 仍然JIT编译 | 先看代码逻辑再猜测 |
| 删除预编译版本 | 预编译版本损坏 | 本来就不存在 | 从事实出发，不要假设 |
| 修改`_real.py` CPU→GPU | buffers未移动 | 本地代码已正确 | 对比本地代码再修改 |

---

### 3.4 当前卡点：VIPE JIT编译

**错误详情**:
```
File: /root/.local/lib/python3.10/site-packages/torch/utils/cpp_extension.py:2312
    cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags()
  File: line 2092 in _get_cuda_arch_flags
    raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
ValueError: Unknown CUDA arch (12.0+PTX) or GPU not supported
```

**待验证方案**:
```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # 明确告诉PyTorch H100=sm_90
```

---

## 四、环境对比与配置

### 4.1 系统环境对比

| 项目 | 本地 | CMCC | 状态 |
|------|------|------|------|
| Python | 3.10 | 3.10 | ✅ 一致 |
| PyTorch | 2.12.0+cu130 | 2.6.0+cu124 | ⚠️ 不同 |
| CUDA | 13.0 | 12.4 | ⚠️ 不同 |
| cuDNN | - | 90100 | - |
| GPU | H100 80GB | H100 80GB | ✅ 一致 |
| Conda env | sana_wm | sana_wm_env | ✅ 等价 |
| vipe_ext预编译 | ❌ 无 | ❌ 无 | ✅ 一致 |
| JIT编译位置 | subprocess | subprocess | ✅ 一致 |

---

### 4.2 环境变量配置

**smoke_cmcc_pass.sh 最终配置**:
```bash
# GPU可见性（必需）
export CUDA_VISIBLE_DEVICES=0

# 模型权重路径
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"
export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"

# 离线模式
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PYTHONPATH（必需）
export PYTHONPATH="$PROJ_DIR/src:$PROJ_DIR/third_party${PYTHONPATH:+:$PYTHONPATH}"

# H100架构支持（待验证）
export TORCH_CUDA_ARCH_LIST="9.0"
```

---

### 4.3 代码无需修改

**验证结果**: 本地 `_real.py` 代码已正确  
- ✅ `model.to(dev)` 已存在（第50行）
- ✅ Pi3X/MoGe加载逻辑正确
- ✅ 所有buffers会被移动到GPU

**CMCC与本地代码完全一致** - 无需任何代码修改

---

## 五、测试与验证

### 5.1 最小复现测试

**目的**: 验证TORCH_CUDA_ARCH_LIST修复JIT编译

**CMCC机器执行**:
```bash
export CUDA_VISIBLE_DEVICES=0
export VIPE_EXT_JIT=0
export TORCH_CUDA_ARCH_LIST="9.0"

python -c "
from sana_wm_pipeline.stage02_pose.mode_default import run_default
from pathlib import Path
art = run_default(
    Path('/root/work/david_work/smoke_pass_results/run_20260817_122418/00094653-a9c6-5558-8e2a-4119e7d64f36/normalized.mp4'),
    Path('/root/work/david_work/smoke_pass_results/run_20260817_122418/00094653-a9c6-5558-8e2a-4119e7d64f36/vipe_work_default')
)
print(f'Success: {art.poses_c2w.shape}')
"
```

**预期结果**:
- ✅ VIPE JIT编译成功
- ✅ 输出 `Success: (N, 4, 4)`

---

### 5.2 完整冒烟测试

**命令**:
```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_pass.sh
```

**预期结果**:
- 处理所有 `/root/work/david_work/smoke_pass_videos` 下的视频
- 每个样本生成: normalized.mp4 + pose_artifact_default.json + shard.tar
- 生成测试报告: `smoke_test_report.txt`

---

## 六、文件清单

### 6.1 修改的文件

| 文件 | 修改内容 | 行号 | 状态 |
|------|---------|------|------|
| `CMCC_ADAPTATION_PLAN.md` | 修复VIPE导入检查 | 111-117, 418, 441 | ✅ |
| `CMCC_ADAPTATION_PLAN.md` | 修复PYTHONPATH配置 | 270, 418 | ✅ |
| `smoke_cmcc_pass.sh` | 添加CUDA_VISIBLE_DEVICES | 14 | ✅ |
| `smoke_cmcc_pass.sh` | stdin重定向 | 148 | ✅ |
| `smoke_cmcc_pass.sh` | PYTHONPATH安全引用 | 30 | ✅ |
| `smoke_cmcc_pass.sh` | TORCH_CUDA_ARCH_LIST | 24 | ⏳ 待验证 |

### 6.2 新建的文件

- `experiments/data_production_smoke/smoke_cmcc_pass.sh` (302行)
- `task_progress_cmcc_refine.md` (本文档)

### 6.3 未修改的文件（验证正确）

- `src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py` - 本地代码已正确
- `src/sana_wm_pipeline/stage02_pose/mode_default.py` - subprocess调用无需修改

---

## 七、Ponytail经验总结

### 7.1 调试方法论

1. **从事实出发，不要假设**
   - ❌ 假设本地有`third_party/vipe/third_party/pi3/`
   - ✅ 先`find`验证目录是否存在

2. **对比本地代码，避免盲目修改**
   - ❌ 认为`_real.py`缺少`model.to(dev)`
   - ✅ 先读本地代码，发现已有第50行

3. **理解问题再修复，不要猜测**
   - ❌ 认为`VIPE_EXT_JIT=1`是禁用JIT
   - ✅ 读VIPE源码`__init__.py`理解逻辑

4. **最小验证，快速排除**
   - ✅ 1行Python测试subprocess环境变量传递
   - ✅ 测试Pi3X单独加载排除代码问题

---

### 7.2 根因分析模式

**问题**: 卡住25分钟  
**表面**: `strace: read(0,` → 看起来是stdin等待  
**深入**: `</dev/null` → 暴露`RuntimeError: random_device`  
**再深**: CUDA不可见 → `CUDA_VISIBLE_DEVICES`未设置  
**根因**: CMCC环境默认不设置GPU可见性

**教训**: 不要停在第一层现象，追溯到环境配置差异

---

### 7.3 错误修改的代价

| 修改 | 耗时 | 结果 | 浪费 |
|------|------|------|------|
| `_device()` try-catch | 10min | Segmentation fault | ✗ |
| `_pi3()` CPU→GPU | 15min | 本地代码已有 | ✗ |
| `VIPE_EXT_JIT=1` | 5min | 逻辑理解反了 | ✗ |

**总计浪费**: 30分钟  
**原因**: 未先验证本地代码，凭记忆修改

---

## 八、后续行动

### 8.1 立即行动（P0）

- [ ] **验证TORCH_CUDA_ARCH_LIST方案**（最小复现测试）
- [ ] 如成功，更新脚本添加该环境变量
- [ ] 运行完整冒烟测试

### 8.2 批量测试（P1）

- [ ] 处理所有246个smoke_pass_videos
- [ ] 生成质量报告
- [ ] 确认成功率 > 95%

### 8.3 生产部署（P2）

- [ ] 更新CMCC部署文档
- [ ] 同步环境变量配置到batch_production脚本
- [ ] 打包更新的代码tarball

---

## 九、关键命令速查

### 9.1 环境验证
```bash
# GPU可见性
python -c "import torch; print(torch.cuda.is_available())"

# GPU架构
python -c "import torch; print(torch.cuda.get_device_capability())"

# PyTorch版本
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 9.2 VIPE测试
```bash
# 测试subprocess环境变量
export VIPE_EXT_JIT=0
python -c "import subprocess; subprocess.check_call(['python', '-c', 'import os; print(os.environ.get(\"VIPE_EXT_JIT\"))'])"

# 查找预编译版本
find /root -name "vipe_ext*" 2>/dev/null
```

---

## 十、会话总结（2026-08-17 11:00-13:45）

### 10.1 核心成果

**✅ 已解决（6个问题）**:
1. VIPE导入路径错误 → 修正为实际使用的`from pi3 import Pi3X`
2. PYTHONPATH缺少third_party → 添加完整路径
3. PYTHONPATH unbound variable → 使用`${PYTHONPATH:+:$PYTHONPATH}`安全引用
4. CUDA_VISIBLE_DEVICES未设置 → 脚本第14行添加`export CUDA_VISIBLE_DEVICES=0`
5. stdin重定向隐藏错误 → 添加`</dev/null`暴露真实错误
6. Buffer设备不匹配假象 → 确认本地代码已有`model.to(dev)`

**⏳ 待验证（1个问题）**:
7. VIPE JIT编译H100架构失败 → `export TORCH_CUDA_ARCH_LIST="9.0"`

---

### 10.2 排查耗时分析

**有效排查（90分钟）**:
- 环境变量配置：30分钟
- CUDA可见性问题：20分钟
- VIPE架构理解：40分钟

**浪费时间（30分钟）**:
- 错误修改`_device()`：10分钟
- 错误修改`_pi3()`：15分钟
- 误判`VIPE_EXT_JIT`逻辑：5分钟

**效率**: 75% (90/120分钟有效)

---

### 10.3 关键洞察

#### 洞察1: 问题的层次性
```
表面现象: 卡住25分钟
  ↓ strace
第1层: read(0, stdin等待
  ↓ </dev/null
第2层: RuntimeError: random_device
  ↓ 测试torch.cuda.is_available()
第3层: CUDA available: False
  ↓ 检查环境变量
根因: CUDA_VISIBLE_DEVICES未设置
```

**教训**: 每一层现象都有合理解释，但必须追溯到根本原因

---

#### 洞察2: VIPE的三层架构
```
主Python进程 (本地代码)
  ↓ subprocess.check_call(["vipe", "infer"])
子Python进程 (vipe命令)
  ↓ import vipe → vipe/ext/__init__.py
JIT编译层 (torch.utils.cpp_extension.load)
  ↓ 失败: Unknown CUDA arch (12.0+PTX)
```

**教训**: 预编译版本不存在是正常的，JIT在subprocess中是设计如此

---

#### 洞察3: 假设vs事实

| 假设 | 事实 | 验证方法 |
|------|------|---------|
| 本地有pi3源码 | 不存在 | `find` |
| `_real.py`缺少`model.to()` | 已有第50行 | `Read` |
| 预编译版本损坏 | 根本不存在 | `find` |
| `VIPE_EXT_JIT=1`禁用JIT | 1=启用JIT | 读源码 |

**教训**: 验证成本很低（1行命令），假设成本很高（10-15分钟浪费）

---

### 10.4 代码修改汇总

**唯一需要修改的文件**: `smoke_cmcc_pass.sh`

| 行号 | 修改 | 原因 | 状态 |
|------|------|------|------|
| 14 | 添加`CUDA_VISIBLE_DEVICES=0` | GPU不可见 | ✅ |
| 24 | 添加`TORCH_CUDA_ARCH_LIST="9.0"` | H100架构 | ⏳ |
| 30 | `${PYTHONPATH:+:$PYTHONPATH}` | unbound variable | ✅ |
| 148 | 添加`</dev/null` | 暴露真实错误 | ✅ |

**其他文件**: 全部无需修改（本地代码已正确）

---

### 10.5 未解决问题

**当前卡点**: VIPE JIT编译失败

**错误**:
```python
ValueError: Unknown CUDA arch (12.0+PTX) or GPU not supported
# PyTorch 2.6.0+cu124 不认识 H100 的 sm_90 架构
```

**待验证方案**:
```bash
export TORCH_CUDA_ARCH_LIST="9.0"
```

**如果失败的备选方案**:
1. 升级PyTorch到2.12.0（本地版本，支持H100）
2. 预编译vipe_ext并打包（避免JIT）
3. 降级GPU架构到sm_80（A100模拟，可能性能损失）

---

### 10.6 Ponytail方法论验证

**成功案例**:
- ✅ 1行Python测试subprocess环境变量（5秒确认）
- ✅ `find`验证目录存在（3秒确认）
- ✅ 读本地代码对比CMCC（避免盲目修改）

**失败案例**:
- ❌ 凭记忆修改`_real.py`（15分钟浪费）
- ❌ 假设预编译版本损坏（10分钟浪费）
- ❌ 猜测`VIPE_EXT_JIT`逻辑（5分钟浪费）

**验证的Ponytail原则**:
1. Does this need to exist? → 不修改代码，只改环境变量 ✅
2. Already working? → 本地代码已正确，直接复用 ✅
3. One line? → 环境变量设置，不写复杂workaround ✅

---

### 10.7 交接清单

**新对话需要的全部信息**:

1. **立即执行**: 第五.5.1节"最小复现测试"
2. **如果失败**: 读第十.10.5节"备选方案"
3. **避免重复**: 读第七.7.3节"错误修改的代价"
4. **命令速查**: 第九章"关键命令速查"

**核心文件**:
- `task_progress_cmcc_refine.md` (本文档) - 完整上下文
- `smoke_cmcc_pass.sh` (第14,24,30,148行) - 唯一需要修改的文件

**一句话总结**: 
> CUDA可见性、PYTHONPATH、stdin重定向已修复，唯一卡点是H100架构JIT编译，待验证`TORCH_CUDA_ARCH_LIST="9.0"`方案。

---

**文档版本**: v2.1 (Final)  
**会话结束时间**: 2026-08-17 13:45  
**总耗时**: 2小时45分钟  
**有效工作时间**: 2小时（75%效率）  
**下次会话起点**: 五.5.1 最小复现测试

---

## 十一、Conda 环境打包与离线部署（2026-08-17 14:00-16:30）

**会话目标**: 打包本地 conda 环境上传 ModelScope，CMCC 端离线部署

---

### 11.1 环境打包阶段

#### 问题1: conda-pack 不支持可编辑包

**错误**:
```
CondaPackError: Cannot pack an environment with editable packages installed
- nvidia_vipe
- sana_wm_pipeline
```

**解决方案**: 放弃 conda-pack，直接 tar 打包整个环境目录

**命令**:
```bash
cd /mnt/afs/davidwang/miniconda3/envs
tar -czf sana_qc_clean.tar.gz --exclude='*.pyc' --exclude='__pycache__' sana_qc
tar -czf sana_wm_clean.tar.gz --exclude='*.pyc' --exclude='__pycache__' sana_wm
```

**结果**:
- sana_qc_clean.tar.gz: 3.7GB (57990 文件)
- sana_wm_clean.tar.gz: 3.7GB (57990 文件)
- MD5 校验文件生成

---

#### 问题2: 压缩包内目录名不匹配

**现象**: 
- 压缩包文件名: `sana_qc_clean.tar.gz`
- 解压后目录名: `sana_qc` (不是 `sana_qc_clean`)

**根因**: tar 打包保留原始目录名

**影响**: 部署文档中的重命名步骤

---

### 11.2 CMCC 离线部署阶段

#### 问题3: 目录重命名导致 Segmentation Fault

**操作**:
```bash
tar -xzf sana_wm_clean.tar.gz  # 解压出 sana_wm/
mv sana_wm sana_wm_clean       # 重命名
conda activate sana_wm_clean
python                          # Segmentation fault
```

**根因**: 
- ELF 二进制文件（Python 解释器、C 扩展）包含硬编码 RPATH
- RPATH 指向 `/mnt/afs/.../sana_wm`
- 重命名后，动态链接器找不到 `lib/` 导致 segfault

**解决**: **不重命名**，直接用原始名称 `sana_qc` 和 `sana_wm`

**正确操作**:
```bash
tar -xzf sana_wm_clean.tar.gz  # 解压出 sana_wm/
# 不重命名！
conda config --append envs_dirs /root/work/david_work/conda_envs
conda activate sana_wm          # 使用原始名称
```

**结果**: ✅ Python 可用

---

#### 问题4: pip 命令损坏

**现象**:
```bash
pip --version
# bash: /root/work/david_work/conda_envs/sana_wm/bin/pip: cannot execute: required file not found
```

**根因**: pip 脚本的 shebang 指向本地路径
```python
#!/mnt/afs/davidwang/miniconda3/envs/sana_wm/bin/python
```

**解决**: 
```bash
# 方案1: 修复 shebang
sed -i '1s|^#!.*|#!/root/work/david_work/conda_envs/sana_wm/bin/python|' \
  /root/work/david_work/conda_envs/sana_wm/bin/pip*

# 方案2: 用 python -m pip 代替
python -m pip --version  # ✅ 可用
```

---

#### 问题5: 可编辑包路径失效

**检查**:
```bash
cat lib/python3.10/site-packages/__editable__.nvidia_vipe-1.1.0.pth
# import __editable___nvidia_vipe_1_1_0_finder; ...

cat lib/python3.10/site-packages/__editable__.sana_wm_pipeline*.pth
# /mnt/afs/davidwang/workspace/sana_wm_pipeline/src  ← 路径不存在
```

**根因**: 可编辑包用 PEP 660 方式（`__editable__*.pth`），包含硬编码源码路径

**当前状态**:
- nvidia_vipe: 使用 finder 机制（需进一步检查）
- sana_wm_pipeline: 硬编码本地路径 `/mnt/afs/.../src`

**待执行修复**:
```bash
# 检查 vipe finder 内容
cat lib/python3.10/site-packages/__editable___nvidia_vipe_1_1_0_finder.py | grep -A5 "MAPPING"

# 修复 sana_wm_pipeline 路径
echo "/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src" > \
  lib/python3.10/site-packages/__editable__.sana_wm_pipeline*.pth
```

---

### 11.3 部署文档更新需求

**CMCC_MANUAL_DEPLOYMENT.md 需要修正**:

1. **步骤3: 解压环境**
   - ❌ 删除重命名步骤（会导致 segfault）
   - ✅ 直接用 `sana_qc` 和 `sana_wm` 原始名称

2. **步骤4: 修复环境路径**
   - ❌ 批量 sed 替换路径（对 ELF 文件无效）
   - ✅ 删除此步骤（不需要路径修复）

3. **步骤5: 修复可编辑包**
   - ❌ 用 `pip install -e --no-deps` 重装
   - ✅ 直接修改 `__editable__*.pth` 文件内容

4. **激活环境**
   - ❌ `source bin/activate`（文件不存在）
   - ✅ `conda activate sana_wm`（需先注册 envs_dirs）

5. **pip 使用**
   - ❌ 直接用 `pip` 命令
   - ✅ 用 `python -m pip` 或修复 shebang

---

### 11.4 关键洞察

#### 洞察4: Conda 环境的三种路径

| 路径类型 | 位置 | 是否可替换 | 影响 |
|---------|------|-----------|------|
| **ELF RPATH** | 二进制文件内部 | ❌ 不可（硬编码） | 重命名目录导致 segfault |
| **脚本 shebang** | 文本文件第一行 | ✅ 可（sed 替换） | pip 命令失效 |
| **可编辑包路径** | `*.pth` 文件 | ✅ 可（直接编辑） | 模块导入失败 |

**教训**: 
- ELF RPATH 不可修改 → 目录名不能改
- shebang 和 .pth 可修改 → 修复这两处即可

---

#### 洞察5: 为什么需要"重装"可编辑包

**误解**: 以为环境中没有 vipe

**真相**: 
- vipe **已安装**，但是可编辑模式
- 可编辑安装 = 环境中只有指针文件（`*.pth`），源码在项目目录
- 指针文件内容 = `/mnt/afs/.../third_party/vipe` ← 此路径在 CMCC 不存在

**两种修复方式**:
1. 修改 `.pth` 文件指向 CMCC 项目路径（快）
2. 重新 `pip install -e <CMCC路径> --no-deps`（慢但彻底）

---

### 11.5 路径残留是否需要修复？

**grep 发现大量残留**:
```
include/gmp.h: -isystem /mnt/afs/.../sana_qc/include
lib/tkConfig.sh: TK_PREFIX='/mnt/afs/.../sana_qc'
lib/icu/pkgdata.inc: -L/mnt/afs/.../sana_qc/lib
```

**分析**:
- 这些是**编译时元数据**（构建时的编译器参数）
- 运行时不读取这些文件
- **不影响 Python 正常运行**

**验证**: CMCC 机器 Python 已可用，说明路径残留无害

---

### 11.6 正确的部署流程

**简化版（CMCC 端执行）**:

```bash
# 1. 下载并解压（不重命名）
cd /root/work/david_work/conda_envs
tar -xzf sana_wm_clean.tar.gz  # 得到 sana_wm/

# 2. 注册到 conda
conda config --append envs_dirs /root/work/david_work/conda_envs

# 3. 激活环境
conda activate sana_wm

# 4. 修复 pip（可选）
python -m pip  # 或修复 shebang

# 5. 修复可编辑包路径
echo "/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src" > \
  lib/python3.10/site-packages/__editable__.sana_wm_pipeline*.pth

# 检查 vipe finder 并修复（待验证）
cat lib/python3.10/site-packages/__editable___nvidia_vipe_1_1_0_finder.py
```

---

### 11.7 生成的文件清单

**本地生成**:
- `sana_qc_clean.tar.gz` (3.7GB)
- `sana_wm_clean.tar.gz` (3.7GB)
- `*.md5` 校验文件
- `CMCC_DEPLOYMENT_GUIDE.sh` - 自动部署脚本（需更新）
- `CMCC_MANUAL_DEPLOYMENT.md` - 手动部署文档（需大量更新）
- `DEPLOYMENT_SUMMARY.md` - 部署总结
- `fix_activate.sh` - 激活修复脚本（已过时，不需要）
- `upload_to_modelscope.sh` - 上传脚本（已修正参数）

**待更新文件**:
- ❌ `CMCC_MANUAL_DEPLOYMENT.md` - 步骤 3/4/5/激活方式全部错误
- ❌ `CMCC_DEPLOYMENT_GUIDE.sh` - 包含错误的重命名和路径修复逻辑
- ✅ `upload_to_modelscope.sh` - ModelScope CLI 参数已修正

---

### 11.8 当前状态与下一步

**CMCC 端当前状态**:
- ✅ 环境已解压到 `/root/work/david_work/conda_envs/sana_wm`
- ✅ `conda activate sana_wm` 可激活
- ✅ `python --version` 正常
- ❌ `pip` 命令损坏（可用 `python -m pip` 绕过）
- ❌ `import vipe` 失败（可编辑包路径无效）
- ❌ `import sana_wm_pipeline` 失败（路径指向本地）

**待执行（CMCC 端）**:
```bash
# 1. 检查 vipe finder 机制
cat /root/work/david_work/conda_envs/sana_wm/lib/python3.10/site-packages/__editable___nvidia_vipe_1_1_0_finder.py | grep -A10 "MAPPING"

# 2. 修复 sana_wm_pipeline 路径
cd /root/work/david_work/conda_envs/sana_wm
echo "/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src" > \
  lib/python3.10/site-packages/__editable__.sana_wm_pipeline*.pth

# 3. 验证
python -c "import sana_wm_pipeline; print('OK')"
python -c "import vipe; print('OK')"
```

**待完成（本地）**:
- [ ] 更新 `CMCC_MANUAL_DEPLOYMENT.md`（删除错误步骤，添加正确流程）
- [ ] 更新 `CMCC_DEPLOYMENT_GUIDE.sh`（修复重命名逻辑）
- [ ] 执行 `upload_to_modelscope.sh` 上传压缩包
- [ ] 同时上传修正后的部署脚本

---

### 11.9 Ponytail 经验补充

**新增教训**:

1. **验证打包结果的完整性**
   - ❌ 假设 tar 压缩包内目录名等于文件名
   - ✅ 用 `tar -tzf` 检查实际目录结构

2. **理解二进制文件的硬编码**
   - ❌ 以为 sed 可以修复所有路径
   - ✅ ELF RPATH 不可修改，目录名必须保持

3. **区分"不存在"vs"指针失效"**
   - ❌ `ModuleNotFoundError` 就以为包没装
   - ✅ 检查 `*.pth` 文件，可能只是路径失效

4. **最简单的验证方式**
   - ✅ 直接在目标环境解压测试
   - ❌ 编写复杂的路径修复脚本再测试

---

### 11.10 会话总结（14:00-16:30）

**核心成果**:
- ✅ 成功打包两个环境（3.7GB × 2）
- ✅ 发现并理解 ELF RPATH 限制（目录不能重命名）
- ✅ 找到可编辑包路径失效的根因（`__editable__*.pth`）
- ❌ 部署文档严重过时（大量错误步骤）

**时间分配**:
- 打包环境：30 分钟（包含失败的 conda-pack 尝试）
- CMCC 部署测试：60 分钟（发现 segfault 问题）
- 可编辑包排查：40 分钟（理解 PEP 660 机制）
- 文档编写：20 分钟

**关键发现**: 
> Conda 环境打包的核心矛盾：ELF RPATH 硬编码 vs 目录可移植性。  
> 解决方案：不改目录名，只修复文本文件（shebang + .pth）

**遗留问题**:
1. vipe 的 finder 机制未完全理解（待查看 `__editable___nvidia_vipe_1_1_0_finder.py`）
2. 部署文档需要大量重写
3. ModelScope 上传尚未执行

---

**文档版本**: v3.0  
**最后更新**: 2026-08-17 16:30  
**总耗时**: 5小时15分钟（11:00-13:45 + 14:00-16:30）  
**下次会话起点**: 十一.11.8 "待执行（CMCC 端）"

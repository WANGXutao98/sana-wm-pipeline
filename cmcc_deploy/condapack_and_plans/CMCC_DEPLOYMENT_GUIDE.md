# CMCC 环境部署与验证指南

**目标环境**: CMCC 离线服务器（无外网、无 GitHub）  
**生成日期**: 2026-08-21  
**适用文件**: sana_wm_cmcc.tar.gz, sana_qc_cmcc.tar.gz

---

## 一、部署前提条件

### 1.1 文件清单

从打包机器传输以下文件到 CMCC：

| 文件名 | 大小 | MD5 | 用途 |
|--------|------|-----|------|
| `sana_wm_cmcc.tar.gz` | ~5-6GB | 见 .md5 文件 | VIPE SLAM 环境 |
| `sana_wm_cmcc.tar.gz.md5` | - | - | MD5 校验文件 |
| `sana_qc_cmcc.tar.gz` | ~4-5GB | 见 .md5 文件 | QC 环境 |
| `sana_qc_cmcc.tar.gz.md5` | - | - | MD5 校验文件 |
| `smoke_test_cmcc_remote.sh` | - | - | 冒烟测试脚本 |

### 1.2 CMCC 机器要求

- **操作系统**: Linux (CentOS/Ubuntu)
- **GPU**: NVIDIA GPU with CUDA 12.x support
- **磁盘空间**: 至少 30GB 可用空间
- **权限**: root 或 sudo 权限

---

## 二、部署步骤

### 步骤 1: 创建工作目录

```bash
# 创建标准工作目录
mkdir -p /root/work/david_work/envs
mkdir -p /root/work/david_work/models/{pi3x,moge2}
mkdir -p /root/work/david_work/cache/{torch,huggingface}
mkdir -p /root/work/david_work/tmp

# 进入工作目录
cd /root/work/david_work
```

---

### 步骤 2: 上传并校验文件

```bash
# 假设文件已通过 ModelScope/scp 传输到当前目录

# 校验 MD5（重要！）
md5sum -c sana_wm_cmcc.tar.gz.md5
# 输出: sana_wm_cmcc.tar.gz: OK

md5sum -c sana_qc_cmcc.tar.gz.md5
# 输出: sana_qc_cmcc.tar.gz: OK
```

**如果 MD5 校验失败**，说明传输损坏，需要重新传输！

---

### 步骤 3: 解压 sana_wm_cmcc 环境

```bash
# 解压到指定目录
mkdir -p /root/work/david_work/envs/sana_wm_cmcc
tar -xzf sana_wm_cmcc.tar.gz -C /root/work/david_work/envs/sana_wm_cmcc

# 进入环境目录
cd /root/work/david_work/envs/sana_wm_cmcc

# 运行 conda-unpack（修复路径引用）
./bin/conda-unpack

# 验证解压成功
./bin/python3 --version
# 输出: Python 3.10.x

./bin/python3 -c "import torch; print(torch.__version__)"
# 输出: 2.x.x+cu124 或类似版本
```

**预计时间**: 10-15 分钟（取决于磁盘 I/O）

---

### 步骤 4: 解压 sana_qc_cmcc 环境

```bash
# 解压到指定目录
mkdir -p /root/work/david_work/envs/sana_qc_cmcc
tar -xzf sana_qc_cmcc.tar.gz -C /root/work/david_work/envs/sana_qc_cmcc

# 进入环境目录
cd /root/work/david_work/envs/sana_qc_cmcc

# 运行 conda-unpack
./bin/conda-unpack

# 验证解压成功
./bin/python3 --version
# 输出: Python 3.10.x

./bin/python3 -c "import torch; print(torch.__version__)"
# 输出: 2.x.x+cu124 或类似版本
```

**预计时间**: 10-15 分钟

---

### 步骤 5: 部署项目代码（如果需要）

```bash
# 解压项目代码（假设已传输 sana_wm_deploy.tar.gz）
cd /root/work/david_work
tar -xzf sana_wm_deploy.tar.gz

# 验证项目结构
ls -la /root/work/david_work/sana_wm_pipeline/src/
```

---

### 步骤 6: 部署模型权重（如果需要）

```bash
# 解压模型权重（假设已传输 sana_wm_models.tar.gz）
cd /root/work/david_work
tar -xzf sana_wm_models.tar.gz -C /root/work/david_work/models/

# 验证模型文件
ls -la /root/work/david_work/models/pi3x/
ls -la /root/work/david_work/models/moge2/
```

---

### 步骤 7: 运行冒烟测试

```bash
# 上传冒烟测试脚本到 CMCC
# 假设已将 smoke_test_cmcc_remote.sh 上传到项目目录

cd /root/work/david_work/sana_wm_pipeline
chmod +x scripts/smoke_test_cmcc_remote.sh

# 运行冒烟测试
bash scripts/smoke_test_cmcc_remote.sh
```

**预期输出**：
```
════════════════════════════════════════════════════════════════════
  CMCC 环境冒烟测试（直接调用模式）
════════════════════════════════════════════════════════════════════
...
✓✓✓ sana_wm_cmcc 测试通过 ✓✓✓
✓✓✓ sana_qc_cmcc 测试通过 ✓✓✓
...
✓✓✓ 所有测试通过，环境部署成功 ✓✓✓
```

**如果测试失败**，查看详细错误信息，根据提示排查问题。

---

## 三、环境使用方式

部署成功后，有 **3 种方式** 使用打包的环境：

### 方式 1: 直接调用（推荐，无需 conda）

```bash
# 直接使用环境的 Python 解释器
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 your_script.py

# 或设置别名
alias python_wm='/root/work/david_work/envs/sana_wm_cmcc/bin/python3'
python_wm your_script.py
```

**优点**：
- 不依赖 conda 安装
- 启动速度快
- 适合生产环境

---

### 方式 2: 通过 PATH 环境变量

```bash
# 临时设置（当前会话）
export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH
python3 your_script.py

# 永久设置（添加到 ~/.bashrc）
echo 'export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**优点**：
- 可以直接使用 `python3` 命令
- 环境变量自动生效

---

### 方式 3: 通过 conda activate（如果 CMCC 有 conda）

```bash
# 假设 CMCC 机器上安装了 conda（路径可能不同）
source /opt/conda/etc/profile.d/conda.sh

# 激活环境（使用绝对路径）
conda activate /root/work/david_work/envs/sana_wm_cmcc

# 使用环境
python3 your_script.py

# 退出环境
conda deactivate
```

**优点**：
- 符合 conda 使用习惯
- 环境变量管理方便

---

## 四、环境变量配置

### 4.1 必需的环境变量

添加以下内容到 `~/.bashrc`：

```bash
# ─── SANA-WM-Pipeline 环境变量 ─────────────────────────────────────────
export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2
export TORCH_HOME=/root/work/david_work/cache/torch
export HF_HOME=/root/work/david_work/cache/huggingface

# 离线模式（CMCC 无外网）
export VIPE_EXT_JIT=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目路径
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:$PYTHONPATH
```

使环境变量生效：
```bash
source ~/.bashrc
```

---

## 五、常见问题排查

### 问题 1: conda-unpack 失败

**现象**：
```
bash: ./bin/conda-unpack: No such file or directory
```

**原因**：tar 解压时没有保留可执行权限

**解决**：
```bash
cd /root/work/david_work/envs/sana_wm_cmcc
chmod +x bin/conda-unpack
./bin/conda-unpack
```

---

### 问题 2: Python 导入 torch 失败

**现象**：
```python
ImportError: libcudart.so.12: cannot open shared object file
```

**原因**：CUDA 运行时库路径未设置

**解决**：
```bash
# 检查环境中的 CUDA 库
ls /root/work/david_work/envs/sana_wm_cmcc/lib/libcuda*

# 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/work/david_work/envs/sana_wm_cmcc/lib:$LD_LIBRARY_PATH

# 或直接使用环境的 Python（推荐）
/root/work/david_work/envs/sana_wm_cmcc/bin/python3
```

---

### 问题 3: JIT 编译失败

**现象**：
```
RuntimeError: Error building extension 'xxx'
```

**原因**：gcc/nvcc 路径未正确设置

**解决**：
```bash
# 方式 1：通过 conda activate（自动设置环境变量）
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/work/david_work/envs/sana_wm_cmcc

# 方式 2：手动设置编译器路径
export CC=/root/work/david_work/envs/sana_wm_cmcc/bin/x86_64-conda-linux-gnu-gcc
export CXX=/root/work/david_work/envs/sana_wm_cmcc/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=/root/work/david_work/envs/sana_wm_cmcc
export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH
```

---

### 问题 4: GPU 不可用

**现象**：
```python
torch.cuda.is_available() = False
```

**排查步骤**：

1. **检查 GPU 驱动**：
   ```bash
   nvidia-smi
   ```
   应该能看到 GPU 信息

2. **检查 CUDA 版本兼容性**：
   ```bash
   # 查看环境的 CUDA 版本
   /root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "import torch; print(torch.version.cuda)"
   
   # 查看系统 CUDA 驱动版本
   nvidia-smi | grep "CUDA Version"
   ```
   
   环境的 CUDA 版本应该 **≤** 系统驱动支持的版本

3. **检查 CUDA 库路径**：
   ```bash
   export LD_LIBRARY_PATH=/root/work/david_work/envs/sana_wm_cmcc/lib:$LD_LIBRARY_PATH
   ```

---

### 问题 5: 模块导入失败

**现象**：
```python
ModuleNotFoundError: No module named 'sana_wm_pipeline'
```

**原因**：PYTHONPATH 未设置

**解决**：
```bash
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:$PYTHONPATH
```

---

## 六、验证检查清单

部署完成后，按此清单逐项验证：

- [ ] **文件传输**
  - [ ] MD5 校验通过
  - [ ] 所有文件完整上传

- [ ] **环境解压**
  - [ ] sana_wm_cmcc 解压成功
  - [ ] sana_qc_cmcc 解压成功
  - [ ] conda-unpack 执行成功

- [ ] **基础验证**
  - [ ] Python 可执行
  - [ ] torch 可导入
  - [ ] CUDA 可用（如果有 GPU）

- [ ] **冒烟测试**
  - [ ] `smoke_test_cmcc_remote.sh` 全部通过
  - [ ] JIT 编译测试通过
  - [ ] VIPE 模块导入成功
  - [ ] QC 依赖库完整

- [ ] **环境变量**
  - [ ] 必需的环境变量已设置
  - [ ] 添加到 ~/.bashrc
  - [ ] 重新登录后仍然生效

---

## 七、下一步操作

### 7.1 运行数据生产任务

```bash
# 使用 sana_wm_cmcc 环境
cd /root/work/david_work/sana_wm_pipeline

# 方式 1: 直接调用
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 scripts/your_data_pipeline.py

# 方式 2: 通过 PATH
export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH
python3 scripts/your_data_pipeline.py

# 方式 3: 通过 conda（如果可用）
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/work/david_work/envs/sana_wm_cmcc
python3 scripts/your_data_pipeline.py
```

### 7.2 批量生产脚本适配

如果使用 `experiments/batch_production/` 中的脚本：

```bash
# 修改 config.sh 中的路径
export NEW_BASE=/root/work/david_work
export PROJ_DIR=$NEW_BASE/sana_wm_pipeline

# 激活环境的方式需要适配（根据 CMCC 实际情况）
# 如果 CMCC 有 conda:
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/work/david_work/envs/sana_wm_cmcc

# 如果 CMCC 没有 conda:
export PATH=/root/work/david_work/envs/sana_wm_cmcc/bin:$PATH
```

---

## 八、回滚方案

如果部署失败需要重新部署：

```bash
# 1. 删除失败的环境
rm -rf /root/work/david_work/envs/sana_wm_cmcc
rm -rf /root/work/david_work/envs/sana_qc_cmcc

# 2. 清理临时文件
rm -rf /root/work/david_work/tmp/*

# 3. 重新执行步骤 3-7
```

---

## 九、性能优化建议

### 9.1 多 GPU 使用

```bash
# 查看可用 GPU
nvidia-smi

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 在脚本中使用
python3 your_script.py --gpus 0,1,2,3
```

### 9.2 显存优化

```bash
# 已在环境变量中设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 如果仍然 OOM，调整 batch size 或启用混合精度
```

---

## 十、联系与支持

**技术文档**：
- 完整方案：`CONDA_ENV_PACKAGING_PLAN.md`
- 执行指南：`CONDA_ENV_PACKAGING_EXECUTION_GUIDE.md`
- Bug 修复日志：`CONDA_ENV_PACKAGING_BUGFIX_LOG.md`

**常见问题**：
- 优先查看本文档的"五、常见问题排查"部分
- 冒烟测试脚本会给出详细的错误信息

---

**文档版本**: v1.0  
**最后更新**: 2026-08-21  
**适用环境**: sana_wm_cmcc + sana_qc_cmcc (conda-pack 打包)

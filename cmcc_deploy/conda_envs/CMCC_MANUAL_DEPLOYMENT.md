# CMCC 离线部署完整指南

**环境对应关系**:
- 本地原环境: `sana_qc` → CMCC 新环境: `sana_qc_clean`
- 本地原环境: `sana_wm` → CMCC 新环境: `sana_wm_clean`

**部署目标**: 在 CMCC 无外网集群离线安装两个 conda 环境，可直接激活使用。

---

## 方案 A: 一键自动部署（推荐）

### 1. 前置条件

```bash
# 确保 modelscope CLI 已安装
pip install modelscope

# 登录 ModelScope（只需一次）
modelscope login --token YOUR_TOKEN
```

### 2. 执行部署脚本

```bash
# 下载部署脚本到 CMCC 机器
cd /root/work/david_work
wget https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data/resolve/master/conda_envs/CMCC_DEPLOYMENT_GUIDE.sh

# 或从本地传输
# scp CMCC_DEPLOYMENT_GUIDE.sh cmcc:/root/work/david_work/

# 执行一键部署
bash CMCC_DEPLOYMENT_GUIDE.sh
```

**脚本功能**:
1. 下载环境压缩包（7.4GB）
2. MD5 校验完整性
3. 解压到 `/root/work/david_work/conda_envs/`
4. 修复所有硬编码路径
5. 重新安装可编辑包（vipe、sana_wm_pipeline）
6. 验证环境可用性
7. 生成快捷激活脚本

**预计耗时**: 15-20 分钟

---

## 方案 B: 手动逐步部署

### 步骤 1: 下载环境压缩包

```bash
mkdir -p /root/work/david_work/conda_envs_download
cd /root/work/david_work/conda_envs_download

# 使用 ModelScope CLI 下载
modelscope download \
    --repo-type dataset \
    davidxwang/sana_spatialvid_smoke_data \
    --include 'conda_envs/*' \
    --local_dir .

# 移动文件到当前目录
mv conda_envs/* .
rmdir conda_envs
```

**下载文件列表**:
```
sana_qc_clean.tar.gz       3.7GB
sana_wm_clean.tar.gz       3.7GB
sana_qc_clean.tar.gz.md5   67 bytes
sana_wm_clean.tar.gz.md5   67 bytes
```

---

### 步骤 2: 校验文件完整性

```bash
cd /root/work/david_work/conda_envs_download

md5sum -c sana_qc_clean.tar.gz.md5
md5sum -c sana_wm_clean.tar.gz.md5

# 预期输出:
# sana_qc_clean.tar.gz: OK
# sana_wm_clean.tar.gz: OK
```

**如果校验失败**: 重新下载对应文件

---

### 步骤 3: 解压环境

```bash
mkdir -p /root/work/david_work/conda_envs
cd /root/work/david_work/conda_envs

# 解压 sana_qc_clean（约 5 分钟）
tar -xzf /root/work/david_work/conda_envs_download/sana_qc_clean.tar.gz

# ⚠️ 解压后目录名是 sana_qc（不是 sana_qc_clean），需要重命名
mv sana_qc sana_qc_clean

# 解压 sana_wm_clean（约 5 分钟）
tar -xzf /root/work/david_work/conda_envs_download/sana_wm_clean.tar.gz

# ⚠️ 解压后目录名是 sana_wm（不是 sana_wm_clean），需要重命名
mv sana_wm sana_wm_clean

# 验证解压结果
ls -lh
# drwxr-xr-x sana_qc_clean
# drwxr-xr-x sana_wm_clean
```

**注意**: 压缩包内保留了原始目录名（`sana_qc`/`sana_wm`），解压后必须重命名为 `sana_qc_clean`/`sana_wm_clean`。

---

### 步骤 4: 修复环境路径（关键步骤）⚠️

**问题**: 环境中硬编码了本地路径 `/mnt/afs/davidwang/miniconda3/envs/sana_qc`  
**解决**: 批量替换为 CMCC 路径 `/root/work/david_work/conda_envs/sana_qc_clean`

#### 4.1 修复 sana_qc_clean

```bash
cd /root/work/david_work/conda_envs/sana_qc_clean

# 修复 conda-meta 中的路径
find conda-meta -name "*.json" -exec sed -i \
    's|/mnt/afs/davidwang/miniconda3/envs/sana_qc|/root/work/david_work/conda_envs/sana_qc_clean|g' {} \;

# 修复 bin 目录中的 shebang（Python 脚本第一行）
find bin -type f -exec sed -i \
    's|/mnt/afs/davidwang/miniconda3/envs/sana_qc|/root/work/david_work/conda_envs/sana_qc_clean|g' {} \;

# 修复 lib 中的硬编码路径
find lib -name "*.py" -o -name "*.pth" | xargs sed -i \
    's|/mnt/afs/davidwang/miniconda3/envs/sana_qc|/root/work/david_work/conda_envs/sana_qc_clean|g' 2>/dev/null || true

echo "✓ sana_qc_clean 路径修复完成"
```

#### 4.2 修复 sana_wm_clean

```bash
cd /root/work/david_work/conda_envs/sana_wm_clean

# 修复 conda-meta
find conda-meta -name "*.json" -exec sed -i \
    's|/mnt/afs/davidwang/miniconda3/envs/sana_wm|/root/work/david_work/conda_envs/sana_wm_clean|g' {} \;

# 修复 bin
find bin -type f -exec sed -i \
    's|/mnt/afs/davidwang/miniconda3/envs/sana_wm|/root/work/david_work/conda_envs/sana_wm_clean|g' {} \;

# 修复 lib
find lib -name "*.py" -o -name "*.pth" | xargs sed -i \
    's|/mnt/afs/davidwang/miniconda3/envs/sana_wm|/root/work/david_work/conda_envs/sana_wm_clean|g' 2>/dev/null || true

echo "✓ sana_wm_clean 路径修复完成"
```

---

### 步骤 5: 修复可编辑包（vipe & sana_wm_pipeline）

**背景**: 原环境包含可编辑安装的包，需要重新链接到 CMCC 项目路径。

#### 5.1 前置条件

确保项目代码已下载到 CMCC:
```bash
ls /root/work/david_work/sana_wm_optimized/sana_wm_pipeline

# 应包含:
# - third_party/vipe/
# - pyproject.toml
# - src/sana_wm_pipeline/
```

#### 5.2 重新安装可编辑包

```bash
# 激活 sana_wm_clean 环境
source /root/work/david_work/conda_envs/sana_wm_clean/bin/activate

# 重新安装 vipe（不安装依赖，避免网络请求）
pip install -e /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe --no-deps

# 重新安装 sana_wm_pipeline
pip install -e /root/work/david_work/sana_wm_optimized/sana_wm_pipeline --no-deps

# 验证安装
python -c "import vipe; print(vipe.__file__)"
python -c "import sana_wm_pipeline; print(sana_wm_pipeline.__version__)"

conda deactivate
```

---

### 步骤 6: 验证环境

#### 6.1 验证 sana_qc_clean

```bash
source /root/work/david_work/conda_envs/sana_qc_clean/bin/activate

# 检查 Python
python --version
# 预期: Python 3.10.x

# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# 预期: PyTorch: 2.12.0+cu130

# 检查 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# 预期: CUDA available: True

conda deactivate
```

#### 6.2 验证 sana_wm_clean

```bash
source /root/work/david_work/conda_envs/sana_wm_clean/bin/activate

# 检查基础环境
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 检查可编辑包
python -c "import vipe; print('✓ vipe OK')"
python -c "from sana_wm_pipeline import __version__; print(f'✓ sana_wm_pipeline {__version__}')"

# 检查关键依赖
python -c "from pi3 import Pi3X; print('✓ Pi3X OK')"
python -c "from moge.model.v2 import MoGeModel; print('✓ MoGe OK')"

conda deactivate
```

**所有检查通过 → 部署成功 ✓**

---

### 步骤 7: 创建快捷激活脚本（可选）

```bash
# 创建 sana_qc_clean 激活脚本
cat > /root/work/david_work/conda_envs/activate_qc.sh <<'EOF'
#!/bin/bash
source /root/work/david_work/conda_envs/sana_qc_clean/bin/activate
echo "✓ 已激活 sana_qc_clean 环境"
python --version
EOF
chmod +x /root/work/david_work/conda_envs/activate_qc.sh

# 创建 sana_wm_clean 激活脚本
cat > /root/work/david_work/conda_envs/activate_wm.sh <<'EOF'
#!/bin/bash
source /root/work/david_work/conda_envs/sana_wm_clean/bin/activate
echo "✓ 已激活 sana_wm_clean 环境"
python --version
EOF
chmod +x /root/work/david_work/conda_envs/activate_wm.sh
```

**使用快捷脚本**:
```bash
source /root/work/david_work/conda_envs/activate_qc.sh
source /root/work/david_work/conda_envs/activate_wm.sh
```

---

## 日常使用

### 激活环境

```bash
# 方法 1: 直接激活
source /root/work/david_work/conda_envs/sana_wm_clean/bin/activate

# 方法 2: 使用快捷脚本
source /root/work/david_work/conda_envs/activate_wm.sh
```

### 运行冒烟测试

```bash
# 激活环境
source /root/work/david_work/conda_envs/activate_wm.sh

# 进入项目目录
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline

# 运行冒烟测试
bash experiments/data_production_smoke/smoke_cmcc_pass.sh
```

---

## 常见问题排查

### Q1: 激活环境后找不到命令

**症状**: `python: command not found`

**原因**: 路径未正确修复

**解决**:
```bash
# 检查 bin 目录的 shebang
head -1 /root/work/david_work/conda_envs/sana_wm_clean/bin/python

# 应该是:
#!/root/work/david_work/conda_envs/sana_wm_clean/bin/python

# 如果不是，重新执行步骤 4
```

---

### Q2: 导入 vipe 或 sana_wm_pipeline 失败

**症状**: `ModuleNotFoundError: No module named 'vipe'`

**原因**: 可编辑包未正确安装

**解决**:
```bash
source /root/work/david_work/conda_envs/sana_wm_clean/bin/activate

# 检查安装状态
pip show nvidia-vipe
pip show sana-wm-pipeline

# 如果显示路径错误，重新安装（步骤 5）
pip uninstall nvidia-vipe sana-wm-pipeline -y
pip install -e /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe --no-deps
pip install -e /root/work/david_work/sana_wm_optimized/sana_wm_pipeline --no-deps
```

---

### Q3: CUDA 不可用

**症状**: `torch.cuda.is_available() = False`

**解决**:
```bash
# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0

# 检查 GPU
nvidia-smi

# 检查 CUDA 版本匹配
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

---

### Q4: 磁盘空间不足

**下载 + 解压需要 ~16GB**:
- 下载目录: 7.4GB（压缩包）
- 安装目录: 8GB × 2 = 16GB（解压后）

**清理空间**:
```bash
# 解压后删除压缩包
rm /root/work/david_work/conda_envs_download/*.tar.gz

# 清理 pip 缓存
pip cache purge
```

---

## 部署检查清单

部署完成后，逐项确认：

- [ ] 文件下载完成（7.4GB）
- [ ] MD5 校验通过
- [ ] 两个环境解压成功
- [ ] 路径修复完成（步骤 4）
- [ ] 可编辑包安装成功（步骤 5）
- [ ] sana_qc_clean 可激活
- [ ] sana_wm_clean 可激活
- [ ] PyTorch CUDA 可用
- [ ] vipe 可导入
- [ ] sana_wm_pipeline 可导入
- [ ] 冒烟测试脚本可运行

**全部打钩 → 部署成功 ✓**

---

## 文件结构

部署完成后的目录结构：

```
/root/work/david_work/
├── conda_envs_download/          # 下载目录（可删除）
│   ├── sana_qc_clean.tar.gz
│   ├── sana_wm_clean.tar.gz
│   └── *.md5
├── conda_envs/                    # 环境安装目录
│   ├── sana_qc_clean/            # QC 环境（8GB）
│   ├── sana_wm_clean/            # WM 环境（7.5GB）
│   ├── activate_qc.sh            # 快捷激活脚本
│   └── activate_wm.sh
└── sana_wm_optimized/            # 项目代码
    └── sana_wm_pipeline/
        ├── third_party/vipe/     # vipe 源码
        └── src/sana_wm_pipeline/ # 管线代码
```

---

## 支持

遇到问题请查看：
- 部署脚本日志（如使用方案 A）
- 项目文档: `sana_wm_pipeline/docs/`
- 冒烟测试排查: `task_progress_cmcc_refine.md`

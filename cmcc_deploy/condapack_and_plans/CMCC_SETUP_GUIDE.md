# CMCC 环境部署操作手册

**当前状态**: 
- 环境包：`/root/work/david_work/conda_envs_download/conda_envs/`
- 项目代码：`/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/`

**目标**: 解压环境、配置 PYTHONPATH、验证可用

---

## 前置检查

```bash
# 检查环境包
cd /root/work/david_work/conda_envs_download/conda_envs/
ls -lh sana_*.tar.gz

# 检查项目代码
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src/
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe/

# 预期输出：
# sana_wm_cmcc.tar.gz  (5-6G)
# sana_qc_cmcc.tar.gz  (4-5G)
# 代码目录存在
```

---

## 步骤 1: 解压 sana_wm_cmcc

```bash
mkdir -p /root/work/david_work/envs/sana_wm_cmcc
tar -xzf /root/work/david_work/conda_envs_download/conda_envs/sana_wm_cmcc.tar.gz \
    -C /root/work/david_work/envs/sana_wm_cmcc
cd /root/work/david_work/envs/sana_wm_cmcc
source bin/activate
conda-unpack
python -c "import torch; print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
deactivate
```

**预期**: `✓ torch 2.x.x (CUDA: True)`

---

## 步骤 2: 解压 sana_qc_cmcc

```bash
mkdir -p /root/work/david_work/envs/sana_qc_cmcc
tar -xzf /root/work/david_work/conda_envs_download/conda_envs/sana_qc_cmcc.tar.gz \
    -C /root/work/david_work/envs/sana_qc_cmcc
cd /root/work/david_work/envs/sana_qc_cmcc
source bin/activate
conda-unpack
python -c "import torch; print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
deactivate
```

**预期**: `✓ torch 2.x.x (CUDA: True)`

---

## 步骤 3: 配置 PYTHONPATH

```bash
# 检查是否已配置
grep "sana_wm_pipeline" ~/.bashrc

# 如果无输出，执行配置
cat >> ~/.bashrc <<'EOF'

# SANA-WM Pipeline
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
EOF

source ~/.bashrc
echo $PYTHONPATH
```

**预期**: 输出包含 `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src`

---

## 步骤 4: 验证环境

```bash
# 测试 sana_wm_cmcc
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
from sana_wm_pipeline.stage02_pose.mode_default import run_default
from nvidia_vipe.models import Pi3xMogeModel
import torch
print(f'✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print('✓ sana_wm_pipeline 导入成功')
print('✓ nvidia_vipe 导入成功')
"

# 测试 sana_qc_cmcc
/root/work/david_work/envs/sana_qc_cmcc/bin/python3 -c "
import torch, torchvision, transformers
print(f'✓ torch {torch.__version__}')
print(f'✓ torchvision {torchvision.__version__}')
print(f'✓ transformers {transformers.__version__}')
"
```

**预期**: 所有模块导入成功，无报错

---

## 步骤 5: 运行冒烟测试

```bash
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash scripts/smoke_test_cmcc_remote.sh
```

**预期**: 
```
✓ sana_wm_cmcc PASS
✓ sana_qc_cmcc PASS
✓✓✓ CMCC 冒烟测试全部通过
```

---

## 故障排查

### 问题 1: `ModuleNotFoundError: No module named 'sana_wm_pipeline'`

```bash
# 检查 PYTHONPATH
echo $PYTHONPATH

# 检查代码是否存在
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src/

# 临时修复
export PYTHONPATH="/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/src:/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe:$PYTHONPATH"
```

### 问题 2: `conda-unpack: command not found`

```bash
# conda-unpack 已内置在环境中
cd /root/work/david_work/envs/sana_wm_cmcc
./bin/conda-unpack
```

### 问题 3: `CUDA: False`

```bash
# 检查 GPU
nvidia-smi

# 检查驱动
/root/work/david_work/envs/sana_wm_cmcc/bin/python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"
```

---

## 完成检查清单

- [ ] sana_wm_cmcc 解压并 unpack
- [ ] sana_qc_cmcc 解压并 unpack
- [ ] 项目代码在 `/root/work/david_work/sana_wm_optimized/sana_wm_pipeline/`
- [ ] PYTHONPATH 配置在 `~/.bashrc`
- [ ] 可以导入 `sana_wm_pipeline` 模块
- [ ] 可以导入 `nvidia_vipe` 模块
- [ ] `smoke_test_cmcc_remote.sh` 通过

---

## 环境路径参考

```
/root/work/david_work/
├── conda_envs_download/conda_envs/  # 下载目录
│   ├── sana_wm_cmcc.tar.gz
│   └── sana_qc_cmcc.tar.gz
├── envs/                             # 解压后环境
│   ├── sana_wm_cmcc/
│   │   └── bin/python3              # 直接使用此路径
│   └── sana_qc_cmcc/
│       └── bin/python3
└── sana_wm_optimized/
    └── sana_wm_pipeline/             # 项目代码
        ├── src/
        ├── third_party/vipe/
        └── scripts/smoke_test_cmcc_remote.sh
```

---

## 关键说明

1. **无需 pip install**: 通过 PYTHONPATH 直接使用源码
2. **完全离线**: 无需网络连接
3. **使用绝对路径**: 推荐 `/root/work/david_work/envs/xxx/bin/python3` 而非 conda activate
4. **环境隔离**: sana_wm_cmcc 和 sana_qc_cmcc 完全独立

---

## 如遇问题

提供以下信息：

```bash
# 1. 系统信息
uname -a
nvidia-smi

# 2. 目录结构
ls -la /root/work/david_work/envs/
ls -la /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/

# 3. PYTHONPATH
echo $PYTHONPATH

# 4. 错误信息（完整复制）
```

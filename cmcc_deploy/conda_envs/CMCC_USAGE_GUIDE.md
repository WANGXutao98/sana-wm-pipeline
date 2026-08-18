# CMCC 端使用指南

## 一、下载环境压缩包

### 方法 1: ModelScope CLI（推荐）

```bash
# 安装 modelscope CLI
pip install modelscope

# 下载所有 conda 环境文件
modelscope download \
    --dataset davidxwang/sana_spatialvid_smoke_data \
    --include 'conda_envs/*' \
    --local_dir /root/work/david_work/downloads
```

### 方法 2: 直接下载链接

```bash
cd /root/work/david_work/downloads
wget https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data/resolve/master/conda_envs/sana_qc_clean.tar.gz
wget https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data/resolve/master/conda_envs/sana_wm_clean.tar.gz
wget https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data/resolve/master/conda_envs/sana_qc_clean.tar.gz.md5
wget https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data/resolve/master/conda_envs/sana_wm_clean.tar.gz.md5
```

---

## 二、校验文件完整性

```bash
cd /root/work/david_work/downloads/conda_envs

# 校验 MD5
md5sum -c sana_qc_clean.tar.gz.md5
md5sum -c sana_wm_clean.tar.gz.md5

# 预期输出:
# sana_qc_clean.tar.gz: OK
# sana_wm_clean.tar.gz: OK
```

---

## 三、解压环境

### 方案 A: 解压到 conda envs 目录（推荐）

```bash
# 解压到 conda 的 envs 目录
cd /opt/conda/envs  # 或你的 miniconda3/envs 路径

tar -xzf /root/work/david_work/downloads/conda_envs/sana_qc_clean.tar.gz
tar -xzf /root/work/david_work/downloads/conda_envs/sana_wm_clean.tar.gz

# 激活环境
conda activate sana_qc_clean
python -c "import torch; print(torch.__version__)"

conda activate sana_wm_clean
python -c "import torch; print(torch.__version__)"
```

### 方案 B: 解压到自定义目录

```bash
# 解压到工作目录
mkdir -p /root/work/david_work/envs
cd /root/work/david_work/envs

tar -xzf /root/work/david_work/downloads/conda_envs/sana_qc_clean.tar.gz
tar -xzf /root/work/david_work/downloads/conda_envs/sana_wm_clean.tar.gz

# 激活环境（使用绝对路径）
source /root/work/david_work/envs/sana_qc_clean/bin/activate
python -c "import torch; print(torch.__version__)"

source /root/work/david_work/envs/sana_wm_clean/bin/activate
python -c "import torch; print(torch.__version__)"
```

---

## 四、运行冒烟测试

```bash
# 确保环境已激活
source /root/work/david_work/envs/sana_wm_clean/bin/activate

# 或
conda activate sana_wm_clean

# 运行冒烟测试脚本
cd /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
bash experiments/data_production_smoke/smoke_cmcc_pass.sh
```

---

## 五、环境说明

### sana_qc_clean
- **用途**: 数据质量检查（QC）
- **大小**: 3.7GB（压缩），~8GB（解压）
- **文件数**: 57990
- **关键包**: 
  - PyTorch 2.12.0+cu130
  - CUDA 13.0
  - dover（视频质量评估）
  - clip（图像-文本对齐）

### sana_wm_clean
- **用途**: 数据生产管线
- **大小**: 3.7GB（压缩），~7.5GB（解压）
- **文件数**: 57990
- **关键包**:
  - PyTorch 2.12.0+cu130
  - CUDA 13.0
  - vipe（相机位姿估计，可编辑安装）
  - sana_wm_pipeline（管线代码，可编辑安装）

**注意**: 两个环境都包含可编辑包（editable packages）：
- `nvidia_vipe`: 位于 `third_party/vipe`
- `sana_wm_pipeline`: 项目根目录

这些包的路径在解压后自动指向压缩包内的位置，无需额外配置。

---

## 六、常见问题

### Q1: 激活环境后找不到 vipe 或 sana_wm_pipeline？

**原因**: 可编辑包的路径可能需要调整。

**解决**:
```bash
# 检查安装位置
pip show nvidia-vipe
pip show sana-wm-pipeline

# 如果路径错误，重新安装
pip install -e /root/work/david_work/sana_wm_optimized/sana_wm_pipeline/third_party/vipe
pip install -e /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
```

### Q2: CUDA 版本不匹配？

本地环境使用 CUDA 13.0，CMCC 使用 CUDA 12.4。如果遇到问题：

```bash
# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 如果不可用，设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0
```

### Q3: 磁盘空间不足？

解压需要 ~16GB 空间（两个环境）。清理空间：

```bash
# 解压后删除压缩包
rm /root/work/david_work/downloads/conda_envs/*.tar.gz

# 清理 pip 缓存
pip cache purge

# 清理 conda 缓存
conda clean --all -y
```

---

## 七、验证清单

解压后验证环境可用性：

```bash
# 1. 激活环境
conda activate sana_wm_clean

# 2. 检查 Python
python --version  # 应为 3.10.x

# 3. 检查 PyTorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 4. 检查可编辑包
python -c "import vipe; print(vipe.__file__)"
python -c "from sana_wm_pipeline import __version__; print(__version__)"

# 5. 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 全部通过 ✓
```

---

## 八、联系方式

如有问题，请查看：
- 项目文档: `/mnt/afs/davidwang/workspace/sana_wm_pipeline/docs/`
- 排查记录: `task_progress_cmcc_refine.md`
- 冒烟测试计划: `task_plan_spatialvid_smoke.md`

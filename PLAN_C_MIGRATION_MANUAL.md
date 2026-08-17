# 方案 C：代码打包迁移实施手册

**方案特点**: 纯代码打包（不含 .git），体积最小  
**预期大小**: 压缩后 0.8-1GB  
**传输时间**: 15-25 分钟  
**执行时间**: 本机 20 分钟 + CMCC 30 分钟  
**执行日期**: 2026-08-16

---

## 阶段 1：本机准备（15-20 分钟）

### 步骤 1.1：清理非必需文件（5 分钟）

**目的**: 减小工作目录，提高后续打包速度。

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 删除实验数据
rm -rf experiments/vipe_comparison/data/

# 删除人工标注数据
rm -rf sana-qc-human-final/

# 删除测试数据
rm -rf testdata/

# 删除中间结果
rm -rf stage2_result/

# 删除 Claude 工作区
rm -rf .claude/worktrees/

# 删除冗余压缩包
rm -f models/unimatch.tar

# 验证当前大小（应显著减小）
du -sh .
```

**检查点**: `du -sh .` 应显示约 3-5GB（从 51GB 降下来）。

---

### 步骤 1.2：创建打包工作目录（1 分钟）

```bash
# 创建临时工作目录
mkdir -p /tmp/sana_wm_migration_$(date +%Y%m%d)
cd /tmp/sana_wm_migration_$(date +%Y%m%d)

# 确认目录为空
ls -la
```

**检查点**: 目录应为空。

---

### 步骤 1.3：导出主仓库代码（2 分钟）

**目的**: 使用 `git archive` 导出当前代码快照（不含 .git 历史）。

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 导出主仓库代码（不含 .git）
git archive --format=tar \
    --prefix=sana_wm_pipeline/ \
    HEAD \
    > /tmp/sana_wm_migration_$(date +%Y%m%d)/main_code.tar

# 验证导出文件
ls -lh /tmp/sana_wm_migration_$(date +%Y%m%d)/main_code.tar
```

**检查点**: `main_code.tar` 应约 50-100MB。

---

### 步骤 1.4：导出 VIPE submodule（1 分钟）

**目的**: 单独导出 `third_party/vipe`，因为 `git archive` 不含 submodule。

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline/third_party/vipe

# 导出 vipe submodule
git archive --format=tar \
    --prefix=sana_wm_pipeline/third_party/vipe/ \
    HEAD \
    > /tmp/sana_wm_migration_$(date +%Y%m%d)/vipe_submodule.tar

# 验证导出文件
ls -lh /tmp/sana_wm_migration_$(date +%Y%m%d)/vipe_submodule.tar
```

**检查点**: `vipe_submodule.tar` 应约 40-50MB。

---

### 步骤 1.5：打包模型权重（2 分钟）

**目的**: 单独打包模型权重（git archive 不含未被 git 跟踪的大文件）。

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 打包 DOVER 权重
tar -cf /tmp/sana_wm_migration_$(date +%Y%m%d)/models_dover.tar \
    models/DOVER/pretrained_weights/DOVER.pth \
    models/DOVER/dover.yml

# 打包 UniMatch 权重
tar -cf /tmp/sana_wm_migration_$(date +%Y%m%d)/models_unimatch.tar \
    models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth

# 打包 ffmpeg 工具
tar -cf /tmp/sana_wm_migration_$(date +%Y%m%d)/bin_tools.tar \
    .bin/ffmpeg \
    .bin/ffprobe

# 验证文件
ls -lh /tmp/sana_wm_migration_$(date +%Y%m%d)/*.tar
```

**检查点**: 应有 5 个 .tar 文件：
- `main_code.tar` (~50-100MB)
- `vipe_submodule.tar` (~40-50MB)
- `models_dover.tar` (~230MB)
- `models_unimatch.tar` (~30MB)
- `bin_tools.tar` (~380MB)

---

### 步骤 1.6：合并并压缩（5 分钟）

**目的**: 将所有 tar 文件合并为一个最终压缩包。

```bash
cd /tmp/sana_wm_migration_$(date +%Y%m%d)

# 创建合并目录
mkdir -p merged
cd merged

# 解压所有 tar 文件到同一目录
tar -xf ../main_code.tar
tar -xf ../vipe_submodule.tar
tar -xf ../models_dover.tar
tar -xf ../models_unimatch.tar
tar -xf ../bin_tools.tar

# 验证目录结构
ls -la sana_wm_pipeline/
ls -la sana_wm_pipeline/third_party/vipe/
ls -la sana_wm_pipeline/models/DOVER/pretrained_weights/
ls -la sana_wm_pipeline/.bin/

# 最终压缩
cd ..
tar -czf sana_wm_pipeline_code_only_$(date +%Y%m%d).tar.gz \
    -C merged \
    sana_wm_pipeline/

# 移动到工作区根目录
mv sana_wm_pipeline_code_only_$(date +%Y%m%d).tar.gz \
   /mnt/afs/davidwang/workspace/

# 清理临时文件
cd /mnt/afs/davidwang/workspace
rm -rf /tmp/sana_wm_migration_$(date +%Y%m%d)
```

**检查点**: `/mnt/afs/davidwang/workspace/sana_wm_pipeline_code_only_20260816.tar.gz` 应约 0.8-1GB。

---

### 步骤 1.7：生成校验文件（1 分钟）

**目的**: 生成 MD5 校验，确保传输完整性。

```bash
cd /mnt/afs/davidwang/workspace

# 生成 MD5
md5sum sana_wm_pipeline_code_only_$(date +%Y%m%d).tar.gz \
    > sana_wm_pipeline_code_only_$(date +%Y%m%d).md5

# 查看信息
ls -lh sana_wm_pipeline_code_only_*
cat sana_wm_pipeline_code_only_*.md5
```

**输出示例**:
```
-rw-r--r-- 1 user user 892M Aug 16 10:30 sana_wm_pipeline_code_only_20260816.tar.gz
-rw-r--r-- 1 user user   85 Aug 16 10:30 sana_wm_pipeline_code_only_20260816.md5
```

---

### 步骤 1.8：验证打包完整性（2 分钟）

**目的**: 确认关键文件都已打包。

```bash
cd /mnt/afs/davidwang/workspace

# 列出压缩包内容（不解压）
tar -tzf sana_wm_pipeline_code_only_$(date +%Y%m%d).tar.gz | head -50

# 验证关键路径
echo "检查 VIPE submodule..."
tar -tzf sana_wm_pipeline_code_only_*.tar.gz | grep "third_party/vipe/vipe/" | wc -l
# 应 >100

echo "检查 DOVER 权重..."
tar -tzf sana_wm_pipeline_code_only_*.tar.gz | grep "DOVER.pth"
# 应输出: sana_wm_pipeline/models/DOVER/pretrained_weights/DOVER.pth

echo "检查 UniMatch 权重..."
tar -tzf sana_wm_pipeline_code_only_*.tar.gz | grep "gmflow-scale2"
# 应输出: sana_wm_pipeline/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth

echo "检查 ffmpeg..."
tar -tzf sana_wm_pipeline_code_only_*.tar.gz | grep ".bin/ffmpeg"
# 应输出: sana_wm_pipeline/.bin/ffmpeg

echo "检查核心代码..."
tar -tzf sana_wm_pipeline_code_only_*.tar.gz | grep "sana_wm_data_clean/pose"
# 应输出多个 .py 文件

echo "检查 Stage 3 脚本..."
tar -tzf sana_wm_pipeline_code_only_*.tar.gz | grep "stage3_batch_minimal.py"
# 应输出: sana_wm_pipeline/scripts/stage3_batch_minimal.py
```

**检查点**: 所有关键文件都应存在。

---

### 步骤 1.9：导出 Conda 环境（可选，2 分钟）

**目的**: 导出依赖列表，供 CMCC 重建环境。

```bash
# 激活环境
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 导出环境配置
conda env export --no-builds > /mnt/afs/davidwang/workspace/environment_sana_wm.yaml

# 导出 pip 依赖
pip list --format=freeze > /mnt/afs/davidwang/workspace/requirements_sana_wm.txt

# 查看文件
ls -lh /mnt/afs/davidwang/workspace/environment_sana_wm.yaml
ls -lh /mnt/afs/davidwang/workspace/requirements_sana_wm.txt
```

**检查点**: 生成两个依赖文件。

---

### 步骤 1.10：准备路径配置文档（1 分钟）

**目的**: 将已生成的路径配置文档一起传输。

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 复制配置文档到工作区根目录
cp CMCC_PATH_CONFIGURATION_CHECKLIST.md /mnt/afs/davidwang/workspace/
cp CMCC_MIGRATION_COMPLETE_GUIDE.md /mnt/afs/davidwang/workspace/
cp DISK_USAGE_ANALYSIS.md /mnt/afs/davidwang/workspace/

# 确认文件
ls -lh /mnt/afs/davidwang/workspace/*.md
```

---

## 阶段 1 完成检查清单

在继续之前，确认以下文件已生成：

```bash
cd /mnt/afs/davidwang/workspace
ls -lh sana_wm_pipeline_code_only_* environment_sana_wm.yaml requirements_sana_wm.txt *.md
```

**应有文件**:
- ✅ `sana_wm_pipeline_code_only_20260816.tar.gz` (~0.8-1GB)
- ✅ `sana_wm_pipeline_code_only_20260816.md5`
- ✅ `environment_sana_wm.yaml`
- ✅ `requirements_sana_wm.txt`
- ✅ `CMCC_PATH_CONFIGURATION_CHECKLIST.md`
- ✅ `CMCC_MIGRATION_COMPLETE_GUIDE.md`
- ✅ `DISK_USAGE_ANALYSIS.md`

---

## 阶段 2：传输到 CMCC（人工操作）

### 步骤 2.1：传输文件

**传输方式**（根据实际情况选择）:

#### 方式 A：通过跳板机 scp
```bash
# 在本机执行
scp /mnt/afs/davidwang/workspace/sana_wm_pipeline_code_only_20260816.tar.gz \
    user@jumphost:/path/to/transfer/

scp /mnt/afs/davidwang/workspace/sana_wm_pipeline_code_only_20260816.md5 \
    user@jumphost:/path/to/transfer/

scp /mnt/afs/davidwang/workspace/environment_sana_wm.yaml \
    user@jumphost:/path/to/transfer/

scp /mnt/afs/davidwang/workspace/requirements_sana_wm.txt \
    user@jumphost:/path/to/transfer/

scp /mnt/afs/davidwang/workspace/CMCC_*.md \
    user@jumphost:/path/to/transfer/
```

#### 方式 B：通过内网文件共享
（根据 CMCC 实际系统操作）

#### 方式 C：通过 U 盘/移动硬盘
1. 将文件复制到移动存储设备
2. 物理传输到 CMCC 机房
3. 复制到 CMCC 服务器

---

### 步骤 2.2：CMCC 侧接收文件

**在 CMCC 服务器执行**:

```bash
# 确认文件已到达
ls -lh /path/to/received/files/

# 验证 MD5（关键！）
cd /path/to/received/files/
md5sum -c sana_wm_pipeline_code_only_20260816.md5
```

**预期输出**: `sana_wm_pipeline_code_only_20260816.tar.gz: OK`

**如果 MD5 不匹配**: 文件传输损坏，需重新传输。

---

## 阶段 3：CMCC 侧部署（30 分钟）

### 步骤 3.1：解压到目标目录（2 分钟）

```bash
# 切换到工作目录
cd /root/work/david_work

# 备份旧版本（如果存在）
if [ -d "sana_wm_pipeline" ]; then
    mv sana_wm_pipeline sana_wm_pipeline_backup_$(date +%Y%m%d_%H%M)
fi

# 解压新版本
tar -xzf /path/to/received/files/sana_wm_pipeline_code_only_20260816.tar.gz

# 验证解压结果
ls -la sana_wm_pipeline/
```

**检查点**: 应看到 `src/`, `models/`, `scripts/`, `third_party/` 等目录。

---

### 步骤 3.2：验证关键文件完整性（3 分钟）

```bash
cd /root/work/david_work/sana_wm_pipeline

# 验证 VIPE submodule
echo "检查 VIPE..."
ls -la third_party/vipe/vipe/
test -f third_party/vipe/vipe/models.py && echo "✓ VIPE 代码存在" || echo "✗ VIPE 缺失"

# 验证模型权重
echo "检查 DOVER 权重..."
ls -lh models/DOVER/pretrained_weights/DOVER.pth
test -f models/DOVER/pretrained_weights/DOVER.pth && echo "✓ DOVER 权重存在" || echo "✗ DOVER 缺失"

echo "检查 UniMatch 权重..."
ls -lh models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth
test -f models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth && echo "✓ UniMatch 权重存在" || echo "✗ UniMatch 缺失"

# 验证 ffmpeg
echo "检查 ffmpeg..."
ls -lh .bin/ffmpeg .bin/ffprobe
.bin/ffmpeg -version | head -1

# 验证核心代码
echo "检查核心模块..."
ls -la src/sana_wm_pipeline/sana_wm_data_clean/pose/
test -f src/sana_wm_pipeline/sana_wm_data_clean/pose/stage.py && echo "✓ 核心代码存在" || echo "✗ 核心代码缺失"
```

**检查点**: 所有关键文件都应显示 `✓`。

---

### 步骤 3.3：修改硬编码路径（5 分钟）

**创建路径修改脚本**:

```bash
cd /root/work/david_work/sana_wm_pipeline

cat << 'EOF' > /tmp/fix_paths_cmcc.sh
#!/bin/bash
set -e

PROJECT_ROOT="/root/work/david_work/sana_wm_pipeline"
cd "$PROJECT_ROOT"

echo "=== 开始修改硬编码路径 ==="

# 备份原文件
echo "1. 备份原文件..."
cp src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py \
   src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py.bak

cp scripts/stage3_batch_minimal.py scripts/stage3_batch_minimal.py.bak
cp scripts/stage3_test_5s.py scripts/stage3_test_5s.py.bak
cp configs/pipeline.yaml configs/pipeline.yaml.bak

# 修改 _real.py 模型路径默认值
echo "2. 修改 _real.py 模型路径..."
sed -i 's|/mnt/afs/davidwang/models/pi3x|/root/work/david_work/models/pi3x|g' \
  src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py

sed -i 's|/mnt/afs/davidwang/models/moge2|/root/work/david_work/models/moge2|g' \
  src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py

# 修改 Stage 3 脚本
echo "3. 修改 Stage 3 脚本路径..."
sed -i 's|/mnt/afs/davidwang/workspace/sana_wm_pipeline|/root/work/david_work/sana_wm_pipeline|g' \
  scripts/stage3_batch_minimal.py \
  scripts/stage3_test_5s.py

sed -i 's|/mnt/afs/davidwang/workspace/data/.*/tmp|/root/work/david_work/tmp|g' \
  scripts/stage3_batch_minimal.py \
  scripts/stage3_test_5s.py

# 修改配置文件
echo "4. 修改配置文件..."
sed -i 's|/mnt/afs/davidwang/workspace/data/sana_wm/|/root/work/filestorage/davidwang/sana_wm/|g' \
  configs/pipeline.yaml

# 验证
echo "5. 验证修改..."
if grep -r "/mnt/afs/davidwang" \
   src/sana_wm_pipeline/sana_wm_data_clean/pose/_real.py \
   scripts/stage3_batch_minimal.py \
   scripts/stage3_test_5s.py \
   configs/pipeline.yaml; then
    echo "⚠️ 警告：仍存在未替换的路径"
    exit 1
else
    echo "✓ 核心路径修改完成"
fi

echo "=== 修改完成 ==="
EOF

# 执行修改脚本
bash /tmp/fix_paths_cmcc.sh
```

**检查点**: 脚本应输出 `✓ 核心路径修改完成`。

---

### 步骤 3.4：设置环境变量（2 分钟）

```bash
# 添加环境变量到 ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# === SANA-WM Pipeline 环境变量 ===
export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2
export TORCH_HOME=/root/work/david_work/cache/torch
export HF_HOME=/root/work/david_work/cache/huggingface
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:$PYTHONPATH

EOF

# 立即生效
source ~/.bashrc

# 验证环境变量
echo "验证环境变量..."
echo "SANA_WM_PI3X_WEIGHTS = $SANA_WM_PI3X_WEIGHTS"
echo "SANA_WM_MOGE2_WEIGHTS = $SANA_WM_MOGE2_WEIGHTS"
echo "TORCH_HOME = $TORCH_HOME"
```

**检查点**: 所有环境变量都应正确显示。

---

### 步骤 3.5：创建必需目录（1 分钟）

```bash
# 创建模型目录（如果模型在其他位置）
mkdir -p /root/work/david_work/models/pi3x
mkdir -p /root/work/david_work/models/moge2

# 创建临时目录
mkdir -p /root/work/david_work/tmp

# 创建数据目录
mkdir -p /root/work/filestorage/davidwang/sana_wm/raw
mkdir -p /root/work/filestorage/davidwang/sana_wm/staging
mkdir -p /root/work/filestorage/davidwang/sana_wm/shards

# 创建缓存目录
mkdir -p /root/work/david_work/cache/torch/hub/checkpoints
mkdir -p /root/work/david_work/cache/huggingface

# 验证目录权限
touch /root/work/david_work/tmp/test.txt && rm /root/work/david_work/tmp/test.txt
echo "✓ 目录创建完成"
```

---

### 步骤 3.6：重建 Conda 环境（10 分钟）

**方式 A：使用 environment.yaml（推荐）**:

```bash
# 复制环境文件
cp /path/to/received/files/environment_sana_wm.yaml /tmp/

# 创建环境
conda env create -f /tmp/environment_sana_wm.yaml \
    -p /root/work/david_work/conda_envs/sana_wm

# 激活环境
conda activate /root/work/david_work/conda_envs/sana_wm

# 验证安装
python --version
pip list | grep torch
pip list | grep numpy
```

**方式 B：使用 requirements.txt（备选）**:

```bash
# 创建基础环境
conda create -p /root/work/david_work/conda_envs/sana_wm python=3.10 -y

# 激活环境
conda activate /root/work/david_work/conda_envs/sana_wm

# 安装依赖
pip install -r /path/to/received/files/requirements_sana_wm.txt

# 验证
python --version
pip list | grep torch
```

**检查点**: Python 环境应正常，主要依赖（torch, numpy, opencv）已安装。

---

### 步骤 3.7：Python 模块导入测试（3 分钟）

```bash
cd /root/work/david_work/sana_wm_pipeline

# 激活环境
conda activate /root/work/david_work/conda_envs/sana_wm

# 测试 VIPE 导入
python -c "
from third_party.vipe.models import Pi3xMogeModel
print('✓ VIPE Pi3xMogeModel 可导入')
"

# 测试核心模块导入
python -c "
import sys
sys.path.insert(0, 'src')
from sana_wm_pipeline.sana_wm_data_clean.pose import stage
print('✓ sana_wm_data_clean.pose 可导入')
"

# 测试环境变量读取
python -c "
import os
from src.sana_wm_pipeline.sana_wm_data_clean.pose._real import _PI3X_WEIGHTS, _MOGE2_WEIGHTS
print(f'Pi3x 路径: {_PI3X_WEIGHTS}')
print(f'Moge2 路径: {_MOGE2_WEIGHTS}')
assert '/root/work/david_work' in _PI3X_WEIGHTS, '路径未正确配置'
print('✓ 路径配置正确')
"
```

**检查点**: 所有测试都应输出 `✓`。

---

### 步骤 3.8：Stage 3 脚本语法检查（2 分钟）

```bash
cd /root/work/david_work/sana_wm_pipeline

# 语法检查
python -m py_compile scripts/stage3_batch_minimal.py
python -m py_compile scripts/stage3_test_5s.py

# 查看帮助（验证脚本可运行）
python scripts/stage3_batch_minimal.py --help 2>&1 | head -20

echo "✓ Stage 3 脚本语法正确"
```

---

### 步骤 3.9：ffmpeg 功能测试（2 分钟）

```bash
cd /root/work/david_work/sana_wm_pipeline

# 测试 ffmpeg
.bin/ffmpeg -version | head -5

# 测试 ffprobe
.bin/ffprobe -version | head -5

# 添加到 PATH（可选）
export PATH=/root/work/david_work/sana_wm_pipeline/.bin:$PATH

# 验证
which ffmpeg
ffmpeg -version | head -1

echo "✓ ffmpeg 工具可用"
```

---

### 步骤 3.10：创建快速启动脚本（2 分钟）

```bash
# 创建启动脚本
cat > /root/work/david_work/sana_wm_pipeline/activate_env.sh << 'EOF'
#!/bin/bash
# SANA-WM Pipeline 环境激活脚本

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
    source /opt/conda/etc/profile.d/conda.sh
conda activate /root/work/david_work/conda_envs/sana_wm

# 设置环境变量
export SANA_WM_PI3X_WEIGHTS=/root/work/david_work/models/pi3x
export SANA_WM_MOGE2_WEIGHTS=/root/work/david_work/models/moge2
export TORCH_HOME=/root/work/david_work/cache/torch
export HF_HOME=/root/work/david_work/cache/huggingface
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:$PYTHONPATH
export PATH=/root/work/david_work/sana_wm_pipeline/.bin:$PATH

# 切换到项目目录
cd /root/work/david_work/sana_wm_pipeline

echo "=== SANA-WM Pipeline 环境已激活 ==="
echo "Python: $(which python)"
echo "工作目录: $(pwd)"
echo "=================================="
EOF

# 添加执行权限
chmod +x /root/work/david_work/sana_wm_pipeline/activate_env.sh

# 测试启动脚本
source /root/work/david_work/sana_wm_pipeline/activate_env.sh
```

---

## 阶段 3 完成检查清单

在继续之前，确认以下项目：

```bash
cd /root/work/david_work/sana_wm_pipeline

# 检查清单
echo "1. 目录结构..."
ls -la src/ models/ scripts/ third_party/ .bin/

echo "2. VIPE 导入..."
python -c "from third_party.vipe.models import Pi3xMogeModel; print('✓')"

echo "3. 核心模块导入..."
python -c "import sys; sys.path.insert(0, 'src'); from sana_wm_pipeline.sana_wm_data_clean.pose import stage; print('✓')"

echo "4. 路径配置..."
python -c "from src.sana_wm_pipeline.sana_wm_data_clean.pose._real import _PI3X_WEIGHTS; assert '/root/work' in _PI3X_WEIGHTS; print('✓')"

echo "5. ffmpeg..."
.bin/ffmpeg -version | head -1

echo "6. 环境变量..."
echo $SANA_WM_PI3X_WEIGHTS

echo "=== 如果以上全部通过，部署成功 ==="
```

---

## 阶段 4：功能验证（可选，需要数据）

### 步骤 4.1：Stage 2 单样本测试

**前提**: 需要一个测试视频文件。

```bash
cd /root/work/david_work/sana_wm_pipeline

# 激活环境
source activate_env.sh

# 准备测试视频
TEST_VIDEO="/path/to/test/video.mp4"
OUTPUT_DIR="/root/work/david_work/tmp/test_stage2"

# 运行 Stage 2（假设有命令行入口）
python -m sana_wm_pipeline.stage02_pose.mode_default \
    --video "$TEST_VIDEO" \
    --output "$OUTPUT_DIR" \
    --mode default

# 检查输出
ls -la "$OUTPUT_DIR/"
```

---

### 步骤 4.2：Stage 3 脚本测试

```bash
# 运行 Stage 3 测试脚本（需要 Stage 2 输出）
python scripts/stage3_test_5s.py \
    --input /path/to/stage2/output \
    --output /root/work/david_work/tmp/test_stage3

# 检查结果
cat /root/work/david_work/tmp/test_stage3/*.jsonl
```

---

## 附录 A：回滚方案

如果部署失败，需要回滚：

```bash
cd /root/work/david_work

# 删除失败的部署
rm -rf sana_wm_pipeline

# 恢复备份（如果有）
if [ -d "sana_wm_pipeline_backup_20260816_1030" ]; then
    mv sana_wm_pipeline_backup_20260816_1030 sana_wm_pipeline
    echo "已回滚到备份版本"
else
    echo "无备份，需要重新部署"
fi
```

---

## 附录 B：常见问题排查

### B.1 Python 导入失败

**问题**: `ModuleNotFoundError: No module named 'sana_wm_pipeline'`

**解决**:
```bash
# 检查 PYTHONPATH
echo $PYTHONPATH

# 手动设置
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/src:$PYTHONPATH

# 或在脚本中
import sys
sys.path.insert(0, '/root/work/david_work/sana_wm_pipeline/src')
```

---

### B.2 VIPE 导入失败

**问题**: `ModuleNotFoundError: No module named 'vipe'`

**原因**: `third_party/vipe` 不在 Python 路径中。

**解决**:
```bash
# 方式 1：添加到 PYTHONPATH
export PYTHONPATH=/root/work/david_work/sana_wm_pipeline/third_party:$PYTHONPATH

# 方式 2：在代码中
import sys
sys.path.insert(0, '/root/work/david_work/sana_wm_pipeline/third_party')
from vipe.models import Pi3xMogeModel
```

---

### B.3 模型权重找不到

**问题**: `FileNotFoundError: /root/work/david_work/models/pi3x not found`

**原因**: 模型权重未部署到对应位置。

**解决**:
```bash
# 检查模型是否在打包中
ls -la /root/work/david_work/sana_wm_pipeline/models/

# 如果模型需要单独部署，从其他位置复制
cp -r /other/location/pi3x /root/work/david_work/models/

# 或修改环境变量指向实际位置
export SANA_WM_PI3X_WEIGHTS=/actual/path/to/pi3x
```

---

### B.4 ffmpeg 找不到

**问题**: `sh: ffmpeg: command not found`

**解决**:
```bash
# 添加 .bin 到 PATH
export PATH=/root/work/david_work/sana_wm_pipeline/.bin:$PATH

# 或使用绝对路径
/root/work/david_work/sana_wm_pipeline/.bin/ffmpeg -version
```

---

### B.5 DOVER 报错 ConvNeXt 权重缺失

**问题**: `FileNotFoundError: convnext_base_1k_224_ema.pth`

**原因**: DOVER 需要额外的 ConvNeXt 权重，需要从外网下载。

**解决**:
```bash
# 在本机下载
mkdir -p /tmp/convnext
cd /tmp/convnext
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth

# 打包并传输到 CMCC
tar -czf convnext_weights.tar.gz convnext_base_1k_224_ema.pth

# CMCC 侧解压到缓存目录
mkdir -p /root/work/david_work/cache/torch/hub/checkpoints
tar -xzf convnext_weights.tar.gz \
    -C /root/work/david_work/cache/torch/hub/checkpoints/
```

---

## 附录 C：时间与成本估算

| 阶段 | 步骤 | 预计时间 | 累计时间 |
|------|------|---------|---------|
| **阶段 1：本机准备** | | | |
| | 清理非必需文件 | 5 分钟 | 5 分钟 |
| | 导出代码 + submodule | 4 分钟 | 9 分钟 |
| | 打包模型权重 | 2 分钟 | 11 分钟 |
| | 合并压缩 | 5 分钟 | 16 分钟 |
| | 验证完整性 | 4 分钟 | 20 分钟 |
| **阶段 2：传输** | | | |
| | 文件传输（取决于网络） | 15-30 分钟 | 35-50 分钟 |
| **阶段 3：CMCC 部署** | | | |
| | 解压验证 | 5 分钟 | 40-55 分钟 |
| | 路径修改 | 5 分钟 | 45-60 分钟 |
| | 环境配置 | 2 分钟 | 47-62 分钟 |
| | Conda 环境重建 | 10 分钟 | 57-72 分钟 |
| | 功能验证 | 5 分钟 | 62-77 分钟 |
| **总计** | | **1-1.5 小时** | |

---

## 附录 D：文件清单汇总

### 本机生成文件
- `sana_wm_pipeline_code_only_20260816.tar.gz` (0.8-1GB)
- `sana_wm_pipeline_code_only_20260816.md5`
- `environment_sana_wm.yaml`
- `requirements_sana_wm.txt`
- `CMCC_PATH_CONFIGURATION_CHECKLIST.md`
- `CMCC_MIGRATION_COMPLETE_GUIDE.md`
- `DISK_USAGE_ANALYSIS.md`

### CMCC 部署后目录结构
```
/root/work/david_work/
├── sana_wm_pipeline/              # 主项目目录
│   ├── src/                       # 核心代码
│   ├── scripts/                   # 批处理脚本
│   ├── models/                    # 模型权重
│   ├── third_party/vipe/          # VIPE submodule
│   ├── .bin/ffmpeg, ffprobe       # 视频工具
│   ├── configs/                   # 配置文件
│   └── activate_env.sh            # 快速启动脚本
├── conda_envs/sana_wm/            # Conda 环境
├── models/pi3x/                   # Pi3x 模型（如需单独部署）
├── models/moge2/                  # Moge2 模型（如需单独部署）
├── tmp/                           # 临时文件
└── cache/                         # 缓存目录
    ├── torch/
    └── huggingface/
```

---

**文档版本**: v1.0  
**生成时间**: 2026-08-16  
**维护者**: Claude (Ponytail Mode)  
**预计执行时间**: 1-1.5 小时

# SANA-WM Pipeline 部署指南

> **文档状态**：当前有效  
> **最后更新**：2026-08-01  
> **维护者**：David Wang

---

## 目录

1. [部署环境概览](#一部署环境概览)
2. [AFS 开发环境部署](#二afs-开发环境部署)
3. [CMCC 生产环境部署](#三cmcc-生产环境部署)
4. [数据持久化策略](#四数据持久化策略)
5. [常见部署问题](#五常见部署问题)
6. [性能调优](#六性能调优)
7. [安全检查清单](#七安全检查清单)

---

## 一、部署环境概览

### 1.1 两种环境对比

| 环境 | 用途 | 资源 | 数据持久性 |
|------|------|------|-----------|
| **AFS 开发环境** | 开发/测试/小规模验证 | H100×1, 192核CPU, 2TB RAM | ✅ 永久（AFS 网络存储）|
| **CMCC 生产环境** | 批量生产/QC 质检 | H100×48 (6节点×8卡), 分布式调度 | ⚠️ 热盘临时，需备份到 filestorage |

### 1.2 关键路径映射

**AFS 环境路径**：
```bash
工作区：/mnt/afs/davidwang/workspace/sana_wm_pipeline
模型权重：/mnt/afs/davidwang/models/{pi3x,moge2,geocal}
Conda 环境：/mnt/afs/davidwang/miniconda3/envs/abot-physworld
缓存目录：/mnt/afs/davidwang/cache/{torch,huggingface}
```

**CMCC 环境路径**：
```bash
# 热盘（机器重启丢失）
工作目录：/root/work/david_work/sana_wm_pipeline
Conda 环境：/root/work/david_work/sana_wm_env
临时缓存：/root/work/david_work/cache/

# 共享持久盘
数据源：/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full
产出目录：/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output

# 个人持久盘
备份目录：/root/work/filestorage/shangaoooooo/davidwang/
```

---

## 二、AFS 开发环境部署

### 2.1 前置条件

- ✅ H100 GPU × 1（80GB 显存）
- ✅ 200GB+ 磁盘空间（用于模型权重和缓存）
- ✅ Conda 已安装（/mnt/afs/davidwang/miniconda3）
- ✅ Git 已配置

### 2.2 部署步骤

#### Step 1：克隆代码仓库

```bash
cd /mnt/afs/davidwang/workspace
git clone <repository-url> sana_wm_pipeline
cd sana_wm_pipeline
git submodule update --init --recursive  # 拉取 third_party 子模块
```

#### Step 2：创建 Conda 环境

```bash
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda create -n abot-physworld python=3.10 -y
conda activate abot-physworld
```

#### Step 3：安装依赖

```bash
# 安装 PyTorch（CUDA 13.0）
pip install torch==2.12.0+cu130 torchvision==0.17.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# 安装项目依赖
pip install -e . --no-deps
pip install -r requirements.txt

# 安装 VIPE（需要编译 C++ 扩展）
PYTORCH_NVCC="/usr/local/cuda-13.0/bin/nvcc" \
TORCH_CUDA_ARCH_LIST="9.0" \
CUDA_HOME="/usr/local/cuda-13.0" \
  pip install -e third_party/vipe --no-deps --no-build-isolation
```

#### Step 4：下载模型权重

```bash
# 创建模型目录
mkdir -p /mnt/afs/davidwang/models/{pi3x,moge2,geocal}

# 下载 Pi3X（5.1 GB）
huggingface-cli download pi3x/pi3x-full --local-dir /mnt/afs/davidwang/models/pi3x

# 下载 MoGe-2（1.3 GB）
huggingface-cli download moge2/moge2-weights --local-dir /mnt/afs/davidwang/models/moge2

# 下载 GeoCalib（111 MB）
wget https://geocal-models.s3.amazonaws.com/pinhole.tar -O /mnt/afs/davidwang/models/geocal/pinhole.tar
```

#### Step 5：环境变量配置

```bash
# 添加到 ~/.bashrc 或项目 activate.sh
export TORCH_HOME=/mnt/afs/davidwang/cache/torch
export HF_HOME=/mnt/afs/davidwang/cache/huggingface
export SANA_WM_MODELS_DIR=/mnt/afs/davidwang/models
export VIPE_EXT_JIT=0  # 禁用 JIT，使用预编译扩展
```

#### Step 6：单样本测试

```bash
# 使用 QUICKSTART.md 中的测试命令
python -m sana_wm_pipeline.stage02_pose.run_worker \
  --input_video testdata/sample.mp4 \
  --output_dir ./test_output \
  --mode default
```

### 2.3 验证清单

运行以下命令验证环境：

```bash
# 1. Python 环境
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 2. VIPE 扩展
python -c "import vipe; import vipe_ext; print('VIPE OK')"

# 3. 项目包
python -c "import sana_wm_pipeline; print('Package OK')"

# 4. 模型路径
ls /mnt/afs/davidwang/models/pi3x/pi3x_full.pth
ls /mnt/afs/davidwang/models/moge2/moge2_weights.pth
ls /mnt/afs/davidwang/models/geocal/pinhole.tar

# 5. GPU 可用性
nvidia-smi
```

**期望输出**：
- ✅ PyTorch 2.12.0, CUDA: True
- ✅ VIPE OK
- ✅ Package OK
- ✅ 所有模型文件存在
- ✅ nvidia-smi 显示 H100 GPU

---

## 三、CMCC 生产环境部署

### 3.1 环境特点

- ❌ **无外网访问**（需离线部署所有依赖）
- ⚠️ **热盘数据重启丢失**（`/root/work/david_work` 不持久）
- ✅ **使用 modelscope.cn 镜像**（HuggingFace 国内镜像）
- ✅ **分布式调度**（6 节点 × 8 卡 = 48 worker）

### 3.2 部署步骤（A-B-C 三段）

#### A. 源端打包（AFS 环境）

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline

# 1. 打包 Conda 环境（~8 GB）
conda pack -n abot-physworld -o sana_wm_env.tar.gz

# 2. 打包模型权重（~6.5 GB）
tar -czf sana_wm_models.tar.gz -C /mnt/afs/davidwang/models pi3x moge2 geocal

# 3. 打包代码仓库（~50 MB）
git archive --format=tar.gz --prefix=sana_wm_pipeline/ HEAD -o sana_wm_code.tar.gz
tar -czf sana_wm_third_party.tar.gz third_party/

# 4. 生成 MD5 清单
md5sum sana_wm_*.tar.gz > sana_wm_md5.txt
```

#### B. CMCC 环境部署

**Step 1：上传 tar 包**

```bash
# 在 CMCC 上执行
scp user@afs-host:/path/to/sana_wm_*.tar.gz /root/work/filestorage/shangaoooooo/davidwang/
```

**Step 2：MD5 校验**

```bash
cd /root/work/filestorage/shangaoooooo/davidwang/
md5sum -c sana_wm_md5.txt
# 所有文件必须显示 OK
```

**Step 3：解压到工作目录**

```bash
export NEW_BASE=/root/work/david_work
mkdir -p $NEW_BASE

# 解压 Conda 环境
tar -xzf sana_wm_env.tar.gz -C $NEW_BASE/
mv $NEW_BASE/sana_wm_env $NEW_BASE/sana_wm_env

# 解压模型权重
mkdir -p $NEW_BASE/models
tar -xzf sana_wm_models.tar.gz -C $NEW_BASE/models/

# 解压代码
tar -xzf sana_wm_code.tar.gz -C $NEW_BASE/
tar -xzf sana_wm_third_party.tar.gz -C $NEW_BASE/sana_wm_pipeline/
```

**Step 4：配置环境变量**

创建 `$NEW_BASE/activate_sana_wm.sh`：

```bash
#!/bin/bash
export NEW_BASE=/root/work/david_work
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
export OUT_BASE="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output"

# 激活 Conda 环境
source $ENV_DIR/bin/activate

# CUDA 环境
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH

# PyTorch 环境
export TORCH_HOME=$NEW_BASE/cache/torch
export HF_HOME=$NEW_BASE/cache/huggingface

# VIPE 配置
export VIPE_EXT_JIT=0
export PYTORCH_NVCC=$CUDA_HOME/bin/nvcc
export TORCH_CUDA_ARCH_LIST="9.0"

# 离线模式（关键！）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 修复 LD_LIBRARY_PATH 污染
export LD_LIBRARY_PATH=$ENV_DIR/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Python 路径
export PYTHONPATH=$PROJ_DIR:$PYTHONPATH
```

**Step 5：验证基础功能**

```bash
source /root/work/david_work/activate_sana_wm.sh

# 验证 Python 环境
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 验证 VIPE
python -c "import vipe; import vipe_ext; print('VIPE OK')"

# 验证项目包
python -c "import sana_wm_pipeline; print('Package OK')"

# 验证模型路径
ls $NEW_BASE/models/pi3x/pi3x_full.pth
ls $NEW_BASE/models/moge2/moge2_weights.pth

# 验证数据路径
ls $DATA_ROOT/wds-DL3DV-ALL-2K/
```

#### C. 批量生产执行

**Step 1：启动 48 GPU worker（6 节点 × 8 卡）**

创建 hostfile（`~/work/filestorage/.../hostfile`）：

```
node001 slots=8
node002 slots=8
node003 slots=8
node004 slots=8
node005 slots=8
node006 slots=8
```

**Step 2：执行批量生产**

```bash
cd $PROJ_DIR
bash experiments/batch_production/launch_all_nodes.sh ~/work/filestorage/.../hostfile
```

**Step 3：监控任务进度**

```bash
# 实时监控
bash experiments/batch_production/watch_progress.sh wds-DL3DV-ALL-2K

# 查看单个 worker 日志
tail -f $OUT_BASE/logs/node0_gpu0.log

# 查看所有 worker 状态
ps -eo pid,etime,pcpu,cmd | grep run_worker.py

# GPU 利用率
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv --loop=10
```

**Step 4：结果备份到 filestorage**

```bash
# 每 24 小时自动备份（建议用 cron）
rsync -av --progress \
  $OUT_BASE/final_wds-*/ \
  /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output_backup/

# 生成 MD5 清单
find $OUT_BASE -name "*.tar" -exec md5sum {} \; > \
  /root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output_md5.txt
```

### 3.3 关键配置文件

**`experiments/batch_production/config.sh`**：

```bash
#!/bin/bash

# 环境变量
export NEW_BASE=/root/work/david_work
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
export OUT_BASE="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output"

# 数据集优先级（BATCH1 先跑）
BATCH1_GROUPS=(
  "wds-sekai-real-walking-hq"
  "wds-DL3DV-ALL-2K"
  "wds-SpatialVID-hq"
)

ALL_GROUPS=(
  "${BATCH1_GROUPS[@]}"
  "wds-OmniWorld-Game"
  "wds-RealEstate10K-360p"
  "wds-sekai-game-drone"
  "wds-sekai-game-walking"
)
```

---

## 四、数据持久化策略

### 4.1 路径分类

| 路径类型 | 持久性 | 用途 | 备份策略 |
|---------|--------|------|---------|
| `/mnt/afs/davidwang/` | ✅ 永久 | AFS：代码/模型/环境 | AFS 自动备份 |
| `/root/work/filestorage/` | ✅ 永久 | CMCC：个人持久存储 | 手动 rsync |
| `/root/work/externalstorage/` | ✅ 共享 | CMCC：批量产出 | 定期备份到 filestorage |
| `/root/work/david_work/` | ❌ 临时 | CMCC：运行时缓存 | 机器重启丢失，需重新部署 |

### 4.2 备份策略

**关键数据三份备份**：

1. **热盘产出**（临时）：`/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/`
2. **个人持久盘**（备份1）：`/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output_backup/`
3. **AFS 归档**（备份2）：通过 SCP 传回 AFS 存储

**自动备份脚本**（建议加入 cron）：

```bash
#!/bin/bash
# /root/work/filestorage/shangaoooooo/davidwang/backup.sh

SOURCE=/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output
TARGET=/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output_backup
LOG=/root/work/filestorage/shangaoooooo/davidwang/backup.log

echo "[$(date)] 开始备份" >> $LOG
rsync -av --progress $SOURCE/ $TARGET/ >> $LOG 2>&1
echo "[$(date)] 备份完成" >> $LOG
```

**添加到 cron**：

```bash
crontab -e
# 每天凌晨 2 点备份
0 2 * * * bash /root/work/filestorage/shangaoooooo/davidwang/backup.sh
```

### 4.3 数据验证

**定期验证数据完整性**：

```bash
# 生成 MD5 清单
find /root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output -name "*.tar" \
  -exec md5sum {} \; > md5_current.txt

# 与备份比对
md5sum -c md5_backup.txt | grep -v OK
```

---

## 五、常见部署问题

### 5.1 AFS 环境问题

**Q: Conda 环境创建失败**

```
Error: PackagesNotFoundError: The following packages are not available...
```

**A:** 使用国内镜像源

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
```

**Q: 模型下载超时**

```
Error: HTTPSConnectionPool timeout
```

**A:** 使用代理或镜像站

```bash
# 方法1：使用代理
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809

# 方法2：使用 ModelScope 镜像
pip install modelscope
modelscope download --model pi3x/pi3x-full --local_dir /path/to/models/pi3x
```

**Q: CUDA OOM（显存不足）**

```
RuntimeError: CUDA out of memory
```

**A:** 参考 findings.md F-2/F-3，确保模型用后立即释放

```python
del model
torch.cuda.empty_cache()
```

### 5.2 CMCC 环境问题

**Q: 环境变量未生效**

```
ModuleNotFoundError: No module named 'sana_wm_pipeline'
```

**A:** 检查激活脚本是否执行

```bash
source /root/work/david_work/activate_sana_wm.sh
echo $PYTHONPATH  # 应包含 $PROJ_DIR
```

**Q: 模型路径找不到**

```
FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/torch/...'
```

**A:** 确保 TORCH_HOME 指向正确位置

```bash
export TORCH_HOME=$NEW_BASE/cache/torch
export HF_HOME=$NEW_BASE/cache/huggingface
```

**Q: Worker 启动失败（LD_LIBRARY_PATH 污染）**

```
ImportError: undefined symbol: _ZN2at4_ops...
```

**A:** 修复 activate.sh，使用 prepend 而非 append

```bash
# 错误（系统 Python3.12 torch 会覆盖 env torch）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV_DIR/lib

# 正确（env torch 优先）
export LD_LIBRARY_PATH=$ENV_DIR/lib:$LD_LIBRARY_PATH
```

### 5.3 分布式部署问题

**Q: SSH 免密登录失败**

```
Permission denied (publickey,password)
```

**A:** 配置 SSH 密钥

```bash
# 在 master 节点生成密钥
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa

# 复制到所有 worker 节点
for node in node001 node002 node003 node004 node005 node006; do
  ssh-copy-id $node
done
```

**Q: Worker 显存被保活进程占用**

```
CUDA OOM: worker 启动时显存已被 'gg' 进程占用 24GB
```

**A:** 先停止保活进程，运行批量任务后再启动

```bash
# 停止保活
pkill -f gg

# 运行批量任务
bash launch_all_nodes.sh

# 任务结束后重启保活（如需要）
bash keep_all_gpu.sh
```

---

## 六、性能调优

### 6.1 单机优化

**混合精度推理**（减少显存占用 40%）：

```python
# 在 mode_default.py 中使用
with torch.cuda.amp.autocast():
    depth_pi3x = model_pi3x(frames)
```

**显存管理最佳实践**：

```python
# 1. 用后立即释放
del model
torch.cuda.empty_cache()

# 2. Chunk 式处理超长视频
for chunk in video_chunks:
    out = model(chunk.to(device))
    results.append(out.cpu())  # 立即搬回 CPU

# 3. 禁用梯度计算
with torch.no_grad():
    output = model(input)
```

**异步 I/O**（减少等待时间）：

```python
from concurrent.futures import ThreadPoolExecutor

# 异步读取视频
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_video, path) for path in video_paths]
    videos = [f.result() for f in futures]
```

### 6.2 多机扩展

**Slurm 作业调度**（如果使用 Slurm）：

```bash
#!/bin/bash
#SBATCH --job-name=sana_wm_batch
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

srun bash experiments/batch_production/run_groups_sequential.sh
```

**Worker 负载均衡**：

- 使用 Round-Robin 分片分配（已实现）
- 每个 worker 独立输出目录，无锁无竞争
- 失败重试机制：`.done` 标记未生成时自动重跑

**容错与重试机制**：

```bash
# 检查失败的样本
find $OUT_BASE -name "*.tar" -size 0  # 空 tar 文件

# 重跑失败的 shard
python scripts/rerun_failed_shards.py --input $OUT_BASE --output $OUT_BASE/retry
```

---

## 七、安全检查清单

在启动批量生产前，逐项检查：

### 7.1 环境检查

- [ ] 所有节点的 Conda 环境路径一致（`$ENV_DIR`）
- [ ] 所有节点的模型权重完整（MD5 校验通过）
- [ ] 所有节点的 CUDA 版本一致（13.0）
- [ ] 所有节点的 torch 版本一致（2.12.0+cu130）

### 7.2 路径检查

- [ ] `DATA_ROOT` 在所有节点可访问
- [ ] `OUT_BASE` 在所有节点可写入
- [ ] 持久盘空间充足（至少 2TB）

### 7.3 网络检查

- [ ] Master → Worker SSH 免密通过
- [ ] Hostfile 中的节点名可 ping 通
- [ ] 共享存储延迟 < 10ms（`dd` 测试）

### 7.4 配置检查

- [ ] `VIPE_EXT_JIT=0`（必须关闭 JIT）
- [ ] `TRANSFORMERS_OFFLINE=1`（离线模式）
- [ ] `LD_LIBRARY_PATH` prepend 正确

### 7.5 备份检查

- [ ] 备份脚本已配置（`backup.sh`）
- [ ] MD5 清单已生成
- [ ] Cron 定时任务已添加

---

## 八、参考资料

**相关文档**：
- `QUICKSTART.md` — 30 分钟快速上手
- `docs/01-ARCHITECTURE.md` — 系统架构全景
- `docs/05-TROUBLESHOOTING.md` — 故障排查手册
- `task_plan.md` — CMCC 部署完整计划
- `findings.md` — 技术发现汇总（9 个关键陷阱）

**批量生产脚本**：
- `experiments/batch_production/config.sh` — 环境配置
- `experiments/batch_production/launch_all_nodes.sh` — 多节点启动
- `experiments/batch_production/watch_progress.sh` — 监控面板

**数据格式参考**：
- `docs/reference/DATASETS.md` — WebDataset Schema 详解

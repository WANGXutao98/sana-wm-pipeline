# CMCC 部署手册：SANA-WM 数据质检系统

**版本：** 2026-06-27  
**适用：** 在中移动机器（无外网，H100，能访问 modelscope.cn）上部署质检系统  
**数据：** 已在 CMCC externalstorage 就位的 SANA-WM 生产数据  
**阅读方式：** 每个命令块均可直接复制粘贴执行；⚠️ 标记的命令有额外注意事项

---

## 进度总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| 代码开发 (Task 1-7) | ✅ 已完成 | commit c914502, 55 tests pass |
| 文档编写 | ✅ 已完成 | QC_REVIEW_DESIGN.md v3.0 |
| A. 源机器打包 | ⏳ 待执行 | 本手册 §A 节 |
| **B. CMCC 环境部署** | ⏳ 待执行 | 本手册 §B 节 |
| **C. 质检生产执行** | ⏳ 待执行 | 本手册 §C 节 |

---

## 源端 MD5（下载后必须对账）

```
1a6da23f7cd6bbdbaac1a3d91d2f96da  sana_wm_qc-cmcc.tar.gz      (≈4-5G)   conda env
7f982759ae2ea6725883a98c54602b3a  sana_wm_qc-deploy.tar.gz    (≈50-100MB) 项目代码
62850efbcf6f79f8104f07c152281521  sana_wm_qc-unimatch.tar.gz  (≈200MB)   UniMatch 模型
```

ModelScope 仓库：`davidxwang/sana-wm-qc`（待创建）

---

## A. 源机器打包（在 AFS 机器执行）

### A.1 确认环境就绪

```bash
cd /mnt/afs/davidwang/workspace/sana_wm_pipeline
source /mnt/afs/davidwang/miniconda3/etc/profile.d/conda.sh
conda activate sana_wm

# 验证所有测试通过
pytest tests/test_qc_*.py -v
# 期望: 55 passed

# 确认 QC 依赖已安装
python -c "import numpy, scipy, av, scenedetect; print('Stage 1+2 OK')"
```

### A.2 安装 Stage 3 依赖（如未安装）

```bash
conda activate sana_wm

# 安装 DOVER
pip install dover

# 验证
python -c "from dover import DOVER; print('DOVER OK')"
```

### A.3 打包 conda 环境

```bash
# 安装 conda-pack
conda install -c conda-forge conda-pack -y

# 打包 sana_wm 环境（排除 editable 包）
conda pack -n sana_wm \
  -o /mnt/afs/davidwang/workspace/sana_wm_qc-cmcc.tar.gz \
  --ignore-editable-packages \
  --compress-level 6

# 计算 MD5
cd /mnt/afs/davidwang/workspace
md5sum sana_wm_qc-cmcc.tar.gz > sana_wm_qc-cmcc.tar.gz.md5
cat sana_wm_qc-cmcc.tar.gz.md5
```

### A.4 打包项目代码

```bash
cd /mnt/afs/davidwang/workspace

tar -czf sana_wm_qc-deploy.tar.gz \
  --transform 's,^sana_wm_pipeline,sana_wm_qc,' \
  sana_wm_pipeline/src/sana_wm_pipeline/qc/ \
  sana_wm_pipeline/src/sana_wm_pipeline/stage02_pose/pose_quality.py \
  sana_wm_pipeline/src/sana_wm_pipeline/stage04_filter/ \
  sana_wm_pipeline/scripts/run_qc.py \
  sana_wm_pipeline/scripts/run_stage3_cmcc.py \
  sana_wm_pipeline/pyproject.toml \
  sana_wm_pipeline/setup.py \
  sana_wm_pipeline/docs/QC_REVIEW_DESIGN.md \
  sana_wm_pipeline/docs/QC_DEPENDENCIES.md \
  sana_wm_pipeline/docs/QC_MODEL_WEIGHTS.md

md5sum sana_wm_qc-deploy.tar.gz > sana_wm_qc-deploy.tar.gz.md5
cat sana_wm_qc-deploy.tar.gz.md5
```

### A.5 打包 UniMatch 模型

```bash
# 下载 UniMatch（如未下载）
cd /tmp
git clone https://github.com/autonomousvision/unimatch.git
cd unimatch
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained_models/gmflow-scale2-regrefine6-mixdata.pth

# 打包
cd /tmp
tar -czf /mnt/afs/davidwang/workspace/sana_wm_qc-unimatch.tar.gz unimatch/

md5sum /mnt/afs/davidwang/workspace/sana_wm_qc-unimatch.tar.gz > \
  /mnt/afs/davidwang/workspace/sana_wm_qc-unimatch.tar.gz.md5
cat /mnt/afs/davidwang/workspace/sana_wm_qc-unimatch.tar.gz.md5
```

### A.6 上传到 ModelScope（可选）

```bash
# 如果需要通过 ModelScope 分发
export MODELSCOPE_API_TOKEN=ms-...

# 创建仓库（首次）
# 在 modelscope.cn 网页创建 davidxwang/sana-wm-qc 仓库

# 上传三个包
cd /mnt/afs/davidwang/workspace
for pkg in sana_wm_qc-cmcc.tar.gz sana_wm_qc-deploy.tar.gz sana_wm_qc-unimatch.tar.gz; do
  modelscope upload --model davidxwang/sana-wm-qc "$pkg"
done
```

---

## B. CMCC 环境部署

### B.1 确定热盘路径

中移动三层存储铁律：
- `/root/work/<userspace>/`（热盘）：快，**小概率重启丢失**，放 env + 工作数据
- `/root/work/filestorage/.../`（冷盘）：持久不丢，metadata 慢 1000×，**只放 tarball 归档**
- `/root/work/externalstorage/.../`（外存）：持久，顺序读 69MB/s，**只读数据集**

```bash
# 查看你的用户目录名
ls /root/work/

# ⚠️ 把 <USERSPACE> 替换为你看到的实际目录名
HOT=/root/work/<USERSPACE>
mkdir -p "$HOT"

# 速度测试
echo "=== Testing $HOT (期望 <1s) ==="
time bash -c "for i in \$(seq 1 1000); do echo x > $HOT/.t\$i; done; sync"
rm -f $HOT/.t*
```

根据测试结果：
- 热盘 <1s → `NEW_BASE=/root/work/<USERSPACE>`（推荐）
- 热盘 >10s → `NEW_BASE=/tmp`（重启后需重解压）

### B.2 设置全局路径变量

⚠️ **整个 B/C 阶段每次开新 shell 都要执行这段**

```bash
# !! 必须替换以下路径 !!
export NEW_BASE=/root/work/<USERSPACE>      # 热盘
export FS_DIR=/root/work/filestorage/shangaoooooo/davidwang  # 持久冷盘

# 以下路径固定
export ENV_DIR="$NEW_BASE/sana_wm_qc_env"
export PROJ_DIR="$NEW_BASE/sana_wm_qc"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/sana_wm_output"
export QC_OUT="$NEW_BASE/qc_output"
export QWEN_DIR="/root/work/filestorage/shangaoooooo/davidwang/Qwen3.5-27B-VL"

mkdir -p "$ENV_DIR" "$PROJ_DIR" \
          "$NEW_BASE/models" \
          "$NEW_BASE/cache/torch/hub" \
          "$FS_DIR" "$QC_OUT"

# 验证数据集已就位（SANA-WM 生产输出）
echo "数据集 group 数: $(ls $DATA_ROOT | grep -c wds-)"
# 期望: 7
```

### B.3 从 ModelScope 下载包到 filestorage

⚠️ **下载目标必须是 filestorage（持久）**

```bash
export MODELSCOPE_API_TOKEN=ms-...

# 方法 A: modelscope CLI
for PKG in sana_wm_qc-cmcc.tar.gz sana_wm_qc-deploy.tar.gz sana_wm_qc-unimatch.tar.gz; do
  echo "=== 下载 $PKG ==="
  modelscope download \
    --model davidxwang/sana-wm-qc \
    "$PKG" \
    --local_dir "$FS_DIR" \
    --token "$MODELSCOPE_API_TOKEN"
  echo "本地 MD5: $(md5sum $FS_DIR/$PKG)"
done

# MD5 对账
echo "=== 期望值 ==="
cat <<EOF
<待填入>  sana_wm_qc-cmcc.tar.gz
<待填入>  sana_wm_qc-deploy.tar.gz
<待填入>  sana_wm_qc-unimatch.tar.gz
EOF

echo "=== 实际值 ==="
md5sum "$FS_DIR/sana_wm_qc-cmcc.tar.gz" \
       "$FS_DIR/sana_wm_qc-deploy.tar.gz" \
       "$FS_DIR/sana_wm_qc-unimatch.tar.gz"
```

### B.4 部署 conda env

⚠️ **解压目标 = `ENV_DIR`（热盘），不是 filestorage**

```bash
echo "=== B.4.1 解压 conda env（约 5-10 分钟）==="
cd "$ENV_DIR"
time tar -xzf "$FS_DIR/sana_wm_qc-cmcc.tar.gz"

echo "=== B.4.2 conda-unpack 修复 shebang（约 30-60 秒）==="
time "$ENV_DIR/bin/conda-unpack"
touch "$ENV_DIR/.cmcc_unpacked"

echo "=== B.4.3 激活 env ==="
source "$ENV_DIR/bin/activate"

echo "--- 验证 ---"
echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
python -c "import numpy, scipy, av, scenedetect, dover; print('QC deps OK')"
```

### B.5 部署项目代码

```bash
echo "=== B.5.1 解压项目代码 ==="
tar -xzf "$FS_DIR/sana_wm_qc-deploy.tar.gz" -C "$NEW_BASE"
# 产出: $NEW_BASE/sana_wm_qc/

echo "=== B.5.2 安装 editable 包 ==="
source "$ENV_DIR/bin/activate"
cd "$PROJ_DIR"

# 创建 setup.py（如果 tar 里没有完整的）
cat > setup.py <<'SETUP'
from setuptools import setup, find_packages
setup(
    name="sana-wm-qc",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
SETUP

pip install -e . --no-deps --no-build-isolation

echo "--- 验证 ---"
python -c "from sana_wm_pipeline.qc import stage1_fast; print('QC package ✓')"
```

### B.6 部署 UniMatch 模型

```bash
echo "=== B.6.1 解压 UniMatch（约 1-2 分钟）==="
mkdir -p "$NEW_BASE/models"
tar -xzf "$FS_DIR/sana_wm_qc-unimatch.tar.gz" -C "$NEW_BASE/models"

echo "=== B.6.2 验证权重存在 ==="
ls -lh "$NEW_BASE/models/unimatch/gmflow-scale2-regrefine6-mixdata.pth"
# 期望: ~180MB
```

### B.7 写激活脚本

```bash
cat > "$NEW_BASE/activate_qc.sh" <<SCRIPT
#!/bin/bash
# 每次进入新 shell 后 source 这个文件
source "$ENV_DIR/bin/activate"

export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"
export UNIMATCH_DIR="$NEW_BASE/models/unimatch"
export QWEN_DIR="$QWEN_DIR"

export DATA_ROOT="$DATA_ROOT"
export QC_OUT="$QC_OUT"

echo "✓ SANA-WM QC env 已激活"
echo "  DATA_ROOT=\$DATA_ROOT"
echo "  QC_OUT=\$QC_OUT"
echo "  QWEN_DIR=\$QWEN_DIR"
SCRIPT

# 立即生效
source "$NEW_BASE/activate_qc.sh"
```

### B.8 验证 Stage 3 模型加载

```bash
source "$NEW_BASE/activate_qc.sh"

python3 - <<'PY'
import sys, torch
sys.path.insert(0, "$NEW_BASE/models/unimatch")

# UniMatch
from unimatch.unimatch import UniMatch
um = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4,
              num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
              reg_refine=True, task="flow").cuda()
print("✓ UniMatch loaded")

# DOVER
from dover import DOVER
dover_m = DOVER().cuda()
print("✓ DOVER loaded")

# Qwen
from transformers import Qwen2_5_VLForConditionalGeneration
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "$QWEN_DIR", torch_dtype=torch.bfloat16, device_map="cuda"
).eval()
print("✓ Qwen3.5-27B-VL loaded")

print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
PY
```

期望输出：
```
✓ UniMatch loaded
✓ DOVER loaded
✓ Qwen3.5-27B-VL loaded

GPU Memory: ~58 GB
```

---

## C. 质检生产执行

### C.1 单 tar 冒烟测试（Stage 1）

```bash
source "$NEW_BASE/activate_qc.sh"
cd "$PROJ_DIR"

# 选一个 group 的第一个 tar
TEST_GROUP="wds-OmniWorld-Game"
TEST_TAR=$(find "$DATA_ROOT/$TEST_GROUP" -name "*.tar" | head -1)

echo "测试 tar: $TEST_TAR"

# 跑 Stage 1（单 tar，单进程验证）
python scripts/run_qc.py \
  --tar-root "$(dirname $TEST_TAR)" \
  --group "$TEST_GROUP" \
  --output-dir "$QC_OUT/smoke_stage1" \
  --n-workers 1 \
  --skip-stage2

# 验证输出
cat "$QC_OUT/smoke_stage1/stage1_results.jsonl" | head -3
wc -l "$QC_OUT/smoke_stage1/stage1_results.jsonl"
```

### C.2 Stage 2 深度检测冒烟

```bash
# 继续 C.1 的输出
python scripts/run_qc.py \
  --tar-root "$(dirname $TEST_TAR)" \
  --group "$TEST_GROUP" \
  --output-dir "$QC_OUT/smoke_stage2" \
  --n-workers 4 \
  --sample-frac 1.0

# 验证 Stage 2 输出
cat "$QC_OUT/smoke_stage2/stage2_results.jsonl" | head -3
ls "$QC_OUT/smoke_stage2/manifests/"
```

### C.3 Stage 3 GPU 单样本测试

```bash
source "$NEW_BASE/activate_qc.sh"
cd "$PROJ_DIR"

# 从 Stage 1 结果中取第一条样本
SAMPLE_ID=$(head -1 "$QC_OUT/smoke_stage1/stage1_results.jsonl" | python -c "import json,sys; print(json.load(sys.stdin)['sample_id'])")

echo "测试样本: $SAMPLE_ID"

# 跑单样本 Stage 3
python3 - <<PYEOF
import sys, json
sys.path.insert(0, 'src')
from pathlib import Path
from sana_wm_pipeline.qc.stage3_gpu import (
    load_unimatch_fn, load_dover_fn, load_qwen_fn, process_sample_stage3
)
from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds

# 加载模型
flow_fn = load_unimatch_fn("$UNIMATCH_DIR", "cuda")
dover_fn = load_dover_fn("cuda")
vlm_call = load_qwen_fn("$QWEN_DIR", "cuda")
table6_cfg = load_thresholds(Path("$PROJ_DIR/configs/filter_thresholds.yaml"))

# 读取 Stage 1 记录
with open("$QC_OUT/smoke_stage1/stage1_results.jsonl") as f:
    rec = json.loads(f.readline())

# 处理
result = process_sample_stage3(
    rec["sample_id"], rec["tar_path"], rec.get("group", ""),
    flow_fn=flow_fn, dover_fn=dover_fn, vlm_call=vlm_call,
    table6_cfg=table6_cfg, has_camera_words=False
)

print(json.dumps(result, indent=2, ensure_ascii=False))
PYEOF
```

### C.4 全量 Stage 1+2 执行

```bash
source "$NEW_BASE/activate_qc.sh"
cd "$PROJ_DIR"

# 枚举所有 group
for group_dir in "$DATA_ROOT"/wds-*; do
  group=$(basename "$group_dir")
  echo "=== Processing $group ==="
  
  python scripts/run_qc.py \
    --tar-root "$group_dir" \
    --group "$group" \
    --output-dir "$QC_OUT/$group" \
    --n-workers 32 \
    --sample-frac 0.05
  
  echo "✓ $group Stage 1+2 完成"
  echo "  Stage 1: $(wc -l < $QC_OUT/$group/stage1_results.jsonl) samples"
  echo "  Stage 2: $(wc -l < $QC_OUT/$group/stage2_results.jsonl) samples"
done
```

### C.5 全量 Stage 3 执行（48 GPU 并行）

⚠️ **需要在所有机器上同步代码和环境，然后用 SSH launcher 派发**

#### C.5.1 生成 worker 分配表

```bash
# 假设 6 机器 × 8 GPU = 48 workers
NODES=("node1" "node2" "node3" "node4" "node5" "node6")
GPUS_PER_NODE=8
TOTAL_WORKERS=48

# 生成启动脚本
cat > "$NEW_BASE/launch_stage3.sh" <<'LAUNCH'
#!/bin/bash
set -euo pipefail

NODE_RANK=$1  # 0-5
GROUP=$2      # e.g., wds-OmniWorld-Game

source "$NEW_BASE/activate_qc.sh"
cd "$PROJ_DIR"

for gpu in $(seq 0 7); do
  worker_id=$((NODE_RANK * 8 + gpu))
  
  CUDA_VISIBLE_DEVICES=$gpu \
  python scripts/run_stage3_cmcc.py \
    --stage1-jsonl "$QC_OUT/$GROUP/stage1_results.jsonl" \
    --output-dir "$QC_OUT/$GROUP" \
    --qwen-dir "$QWEN_DIR" \
    --unimatch-dir "$UNIMATCH_DIR" \
    --worker-id $worker_id \
    --total-workers $TOTAL_WORKERS \
    --table6-cfg "$PROJ_DIR/configs/filter_thresholds.yaml" \
    --device cuda \
    > "$QC_OUT/$GROUP/worker_${worker_id}.log" 2>&1 &
done

wait
echo "Node $NODE_RANK 完成"
LAUNCH

chmod +x "$NEW_BASE/launch_stage3.sh"
```

#### C.5.2 SSH 派发执行

```bash
# 在主节点执行
for group_dir in "$DATA_ROOT"/wds-*; do
  group=$(basename "$group_dir")
  echo "=== Stage 3: $group ==="
  
  for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    ssh "$node" "bash $NEW_BASE/launch_stage3.sh $i $group" &
  done
  
  wait
  echo "✓ $group Stage 3 完成（48 workers）"
done
```

#### C.5.3 合并 Stage 3 结果

```bash
for group_dir in "$DATA_ROOT"/wds-*; do
  group=$(basename "$group_dir")
  
  # 合并 worker 输出
  cat "$QC_OUT/$group"/stage3_worker*.jsonl > "$QC_OUT/$group/stage3_results.jsonl"
  cat "$QC_OUT/$group"/caption_overrides_worker*.jsonl > "$QC_OUT/$group/caption_overrides.jsonl"
  
  echo "$group: $(wc -l < $QC_OUT/$group/stage3_results.jsonl) samples processed"
done
```

### C.6 生成最终报告

```bash
source "$NEW_BASE/activate_qc.sh"
cd "$PROJ_DIR"

for group_dir in "$DATA_ROOT"/wds-*; do
  group=$(basename "$group_dir")
  
  python scripts/run_qc.py \
    --output-dir "$QC_OUT/$group" \
    --report-only
  
  echo "✓ $group 报告生成完成"
  echo "  HTML: $QC_OUT/$group/report.html"
  echo "  Pass: $(wc -l < $QC_OUT/$group/manifests/pass.txt)"
  echo "  Reject: $(wc -l < $QC_OUT/$group/manifests/reject.txt)"
  echo "  Human review: $(wc -l < $QC_OUT/$group/manifests/human_review.txt)"
done
```

### C.7 备份产出到 filestorage

⚠️ **热盘有丢失风险，立即备份**

```bash
PERSIST_DIR="$FS_DIR/qc_results_$(date +%Y%m%d)"
mkdir -p "$PERSIST_DIR"

for group_dir in "$DATA_ROOT"/wds-*; do
  group=$(basename "$group_dir")
  mkdir -p "$PERSIST_DIR/$group"
  
  rsync -av --progress \
    "$QC_OUT/$group/" \
    "$PERSIST_DIR/$group/"
  
  echo "✓ $group 备份完成"
done

echo "=== 完整备份路径 ==="
du -sh "$PERSIST_DIR"
```

---

## 容器重启后恢复

```bash
# 每次重启后执行一次
export NEW_BASE=/root/work/<USERSPACE>
export FS_DIR=/root/work/filestorage/shangaoooooo/davidwang
export ENV_DIR="$NEW_BASE/sana_wm_qc_env"

if [ -f "$ENV_DIR/.cmcc_unpacked" ]; then
  echo "热盘 env 仍在，直接激活"
  source "$NEW_BASE/activate_qc.sh"
else
  echo "热盘丢失，从 filestorage 重建（约 10 分钟）"
  mkdir -p "$ENV_DIR" && cd "$ENV_DIR"
  time tar -xzf "$FS_DIR/sana_wm_qc-cmcc.tar.gz"
  time "$ENV_DIR/bin/conda-unpack"
  touch "$ENV_DIR/.cmcc_unpacked"
  source "$ENV_DIR/bin/activate"
  cd "$NEW_BASE/sana_wm_qc"
  pip install -e . --no-deps --no-build-isolation
  source "$NEW_BASE/activate_qc.sh"
fi
```

---

## 故障排查速查

| 症状 | 根因 | 处置 |
|------|------|------|
| `tar -xzf` 解压几小时不动 | 解压到 filestorage | Ctrl+C，改到热盘或 /tmp |
| `import av` 失败 | av 未安装 | `conda install -c conda-forge av` |
| `import dover` 失败 | dover 未安装 | `pip install dover` |
| Stage 3 OOM | 三个模型 >80GB | 确认 Qwen 用 BF16，检查其他进程 |
| Stage 3 卡住不动 | Qwen 首次下载权重 | 网络问题，预先打包 Qwen 权重 |
| `unimatch.unimatch` 找不到 | sys.path 错误 | 确认 `UNIMATCH_DIR` 正确 |
| worker .done 文件一直不生成 | 某 worker 崩溃 | 查看对应 worker log |
| report.html 打不开 | 路径错误 | 确认在 `$QC_OUT/<group>/report.html` |

---

## 质检产出物 schema

每个 group 的输出目录结构：
```
$QC_OUT/wds-OmniWorld-Game/
├── stage1_results.jsonl          # Stage 1 所有样本
├── stage2_results.jsonl          # Stage 2 深度检测
├── stage3_worker000.jsonl        # Stage 3 worker 输出（48个）
├── stage3_results.jsonl          # Stage 3 合并结果
├── caption_overrides.jsonl       # Caption 改写 sidecar
├── manifests/
│   ├── pass.txt                  # 通过样本 ID 列表
│   ├── reject.txt                # 拒绝样本 ID 列表
│   └── human_review.txt          # 人工审核队列
└── report.html                   # 可视化报告
```

---

*本手册对应 QC System v1.0 (commit c914502)。如有问题请联系 David Wang。*

# CMCC 部署修复计划：nvidia-vipe 安装失败 + VIPE 权重核查

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 CMCC 机器上 nvidia-vipe editable install 失败问题，核查 VIPE 权重完整性，完成 B.7-B.10 剩余部署步骤，跑通 C.1 单样本 smoke test

**Architecture:** 三步修复策略：(1) 用 TORCH_CUDA_ARCH_LIST=9.0 + PYTHONNOUSERSITE=1 绕过 CUDA 13.0/Blackwell 架构检测问题；(2) 单独核查 GeoCalib .tar 格式权重是否打包进去；(3) 按顺序完成 B.7-B.10 并进入 C 阶段

**Tech Stack:** conda-pack env (torch 2.6+cu124, Python 3.10), H100 sm_90, VIPE SLAM C++ extension, WebDataset

**根因说明（必读）：**

| 问题 | 根因 | 修复方法 |
|------|------|----------|
| `Unknown CUDA arch (12.0+PTX)` | CUDA 13.0 Driver 报告 Blackwell sm_120；且 build 时加载了 `/root/.local` 的系统 torch，不是 conda env 的 | `TORCH_CUDA_ARCH_LIST=9.0 PYTHONNOUSERSITE=1 $ENV_DIR/bin/pip install` |
| `VIPE 权重数=5 期望6` | GeoCalib 格式是 `.tar` 不是 `.pth`/`.pt`，验证脚本 `find -name "*.pth"` 根本找不到 | `find $NEW_BASE/cache -name "*.tar" \| grep geocalib` 单独核查 |

---

## 前置：每次开新 shell 必须先恢复环境变量

```bash
export NEW_BASE=/root/work/david_work
export FS_DIR=/root/work/filestorage/shangaoooooo/davidwang/sana-wm-pipeline
export ENV_DIR="$NEW_BASE/sana_wm_env"
export PROJ_DIR="$NEW_BASE/sana_wm_pipeline"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb-v3-full"
export OUT_BASE="$NEW_BASE/jdvbbfb_out"
```

---

### Task 1: 修复 nvidia-vipe editable install

**Files:**
- 无需修改代码，仅调整 install 命令

- [ ] **Step 1: 激活 conda env（若尚未激活）**

```bash
source "$ENV_DIR/bin/activate"
# 验证 python 来自 conda env，不是系统 python
which python   # 期望: /root/work/david_work/sana_wm_env/bin/python
```

- [ ] **Step 2: 确认问题根因 — 查看当前 torch 从哪里加载**

```bash
python -c "import torch; print(torch.__file__)"
# 若输出含 /root/.local → 确认是系统 torch 干扰，PYTHONNOUSERSITE=1 必须加
# 若输出含 sana_wm_env  → torch 正常，只需 TORCH_CUDA_ARCH_LIST=9.0
```

- [ ] **Step 3: 用正确的环境变量重新安装 nvidia-vipe**

```bash
TORCH_CUDA_ARCH_LIST="9.0" \
PYTHONNOUSERSITE=1 \
  "$ENV_DIR/bin/pip" install -e "$PROJ_DIR/third_party/vipe" \
  --no-deps --no-build-isolation
```

预期输出（末尾）：
```
Successfully installed nvidia-vipe-1.2.0
```

如果编译过程报 `cc1plus: error` 或 `CUDA_HOME not set`，执行：
```bash
source "$ENV_DIR/etc/conda/activate.d/cc_nvcc.sh" 2>/dev/null || true
echo "CC=$CC  CUDA_HOME=$CUDA_HOME"
# 然后重跑 Step 3
```

- [ ] **Step 4: 验证安装成功**

```bash
PYTHONNOUSERSITE=1 python -c "import nvidia_vipe; print('nvidia_vipe ✓')"
```

期望输出：`nvidia_vipe ✓`

**Step 3 仍失败的备用方案（跳过 C 扩展，仅安装 Python 部分）：**

```bash
# 方案 B：手动让 setup.py 跳过 C 扩展编译
# 先查看 vipe 的 setup.py 是否有 SKIP_BUILD 之类的环境变量
grep -r "SKIP\|JIT\|BUILD_EXT" "$PROJ_DIR/third_party/vipe/setup.py" 2>/dev/null | head -10

# 方案 C：直接把 vipe Python 包加入 PYTHONPATH（最后手段）
echo "$PROJ_DIR/third_party/vipe" >> "$ENV_DIR/lib/python3.10/site-packages/nvidia_vipe.pth"
python -c "import nvidia_vipe; print('nvidia_vipe ✓')"
# 注意：方案 C 跳过了 C 扩展编译，runtime 靠 VIPE_EXT_JIT=1 重新 JIT 编译
```

---

### Task 2: 核查 VIPE 权重完整性（GeoCalib .tar）

**Files:**
- 无代码修改，纯验证

- [ ] **Step 1: 核查 GeoCalib — 用正确的文件格式搜索**

```bash
echo "=== GeoCalib（.tar 格式，不是 .pth）==="
find "$NEW_BASE/cache/torch/hub" -name "*.tar" 2>/dev/null
# 期望: .../geocalib/pinhole.tar 或 geocalib-pinhole.tar

ls -lh "$NEW_BASE/cache/torch/hub/geocalib/" 2>/dev/null \
  || echo "⚠ geocalib 目录不存在"
```

- [ ] **Step 2: 核查全部 6 个 VIPE 权重**

```bash
echo "=== 全量权重核查（.pth / .pt / .tar）==="
find "$NEW_BASE/cache/torch/hub" \
  \( -name "*.pth" -o -name "*.pt" -o -name "*.tar" \) \
  | sort | while read f; do
    bname=$(basename "$f")
    for kw in droid_slam groundingdino sam_vit DeAOTL metric_depth geocalib; do
      echo "$bname" | grep -qi "$kw" && echo "  ✓ [$kw] $f"
    done
  done
```

期望 6 行全部打印。

- [ ] **Step 3（仅当 geocalib 缺失时）— 从 HuggingFace/GitHub 拿**

如果 geocalib tar 缺失，且 CMCC 能访问 modelscope.cn，检查是否已打包进 sana_wm-caches.tar.gz：

```bash
# 检查 filestorage 里的 tarball 是否包含 geocalib（不解压整个包）
tar -tzf "$FS_DIR/sana_wm-caches.tar.gz" 2>/dev/null | grep -i geocalib
```

若已包含但没解压出来：
```bash
# 单独解压 geocalib 相关文件
mkdir -p "$NEW_BASE/cache/torch/hub/geocalib"
tar -xzf "$FS_DIR/sana_wm-caches.tar.gz" -C "$NEW_BASE/cache" \
  $(tar -tzf "$FS_DIR/sana_wm-caches.tar.gz" | grep -i geocalib)
```

若 tarball 里根本没有（漏打包）且机器有限外网：
```bash
# 尝试从 modelscope 镜像 wget（CMCC 能访问）
# 没有则需要在源机器重打包并重传 sana_wm-caches.tar.gz
echo "需在源机器重新打包 — 见 Task 2 末尾说明"
```

**漏打包时的源机器修复命令（在 AFS 机器上执行）：**
```bash
# AFS 机器上确认 geocalib 位置
ls /mnt/afs/davidwang/cache/torch/hub/geocalib/

# 重打包加入 geocalib
cd /mnt/afs/davidwang/cache
# 若原来的 tarball 是 torch/ 子目录结构：
tar -czf /tmp/sana_wm-caches-v2.tar.gz torch/ huggingface/

# 重传到 ModelScope（需要 ms-... token）
modelscope upload --model davidxwang/conda-cmcc /tmp/sana_wm-caches-v2.tar.gz
```

---

### Task 3: 完成 B.7 激活脚本 + B.8 linker 软链

**Files:**
- 创建: `$NEW_BASE/activate_sana_wm.sh`

- [ ] **Step 1: 写激活脚本**

```bash
cat > "$NEW_BASE/activate_sana_wm.sh" <<SCRIPT
#!/bin/bash
source "$ENV_DIR/bin/activate"

export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"
export SANA_WM_PI3X_WEIGHTS="$NEW_BASE/models/pi3x"
export SANA_WM_MOGE2_WEIGHTS="$NEW_BASE/models/moge2"

export DISABLE_XFORMERS=1
export VIPE_EXT_JIT=1
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONNOUSERSITE=1

export JDVBBFB_LOCAL_ROOT="$DATA_ROOT"

echo "✓ sana_wm CMCC env 已激活"
echo "  TORCH_HOME=\$TORCH_HOME"
echo "  PI3X=\$SANA_WM_PI3X_WEIGHTS"
echo "  MOGE2=\$SANA_WM_MOGE2_WEIGHTS"
SCRIPT

source "$NEW_BASE/activate_sana_wm.sh"
echo "激活脚本写入 ✓"
```

注意：相比部署文档，额外加了 `PYTHONNOUSERSITE=1`（防止 /root/.local 干扰）。

- [ ] **Step 2: linker 软链补漏**

```bash
for nvdir in "$ENV_DIR/lib/python3.10/site-packages/nvidia"/*/lib; do
  for so in "$nvdir"/*.so.[0-9]*; do
    [ -f "$so" ] || continue
    libname=$(basename "$so")
    unver=$(echo "$libname" | sed -E 's/\.so\.[0-9].*$/.so/')
    [ -e "$ENV_DIR/lib/$unver" ] || ln -sf "$so" "$ENV_DIR/lib/$unver"
  done
done
echo "linker 软链完成"

rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true
echo "torch_extensions 缓存已清"
```

---

### Task 4: B.9 JIT 门控测试 + B.10 Python 导入验证

**Files:**
- 无文件修改，纯测试

- [ ] **Step 1: B.9 JIT 测试（必过才能继续）**

```bash
source "$NEW_BASE/activate_sana_wm.sh"

python3 - <<'PY'
import os, tempfile
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name='sana_wm_jit_preflight',
    cpp_sources=["torch::Tensor f(torch::Tensor x);"],
    cuda_sources=["""
#include <torch/extension.h>
__global__ void k(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
torch::Tensor f(torch::Tensor x) {
    k<<<(x.numel()+255)/256, 256>>>(x.data_ptr<float>(), x.numel());
    return x;
}
"""],
    functions=['f'], verbose=True,
    build_directory=tempfile.mkdtemp(),
)
import torch
result = mod.f(torch.zeros(8, device='cuda')).cpu().tolist()
assert result == [1.0] * 8, f"FAIL: {result}"
print("✓ JIT PASS — nvcc+gcc13 工作正常")
PY
```

期望输出最后一行：`✓ JIT PASS — nvcc+gcc13 工作正常`

**JIT 失败排查表：**

| 错误信息 | 根因 | 处置 |
|----------|------|------|
| `Unknown CUDA arch (12.0+PTX)` | TORCH_CUDA_ARCH_LIST 未生效 | 确认 `echo $TORCH_CUDA_ARCH_LIST` 输出 `9.0` |
| `unsupported GNU version` | gcc≥14 | `$CC --version` 查看；若不对则重新 source activate |
| `cannot find -lcudadevrt` | 缺静态库 | 联系重打包 |
| `CC 为空` | activate.d 未执行 | `source $ENV_DIR/etc/conda/activate.d/cc_nvcc.sh` |

- [ ] **Step 2: B.10 Python 导入验证**

```bash
python3 - <<'PY'
import os
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '9.0')

pkgs = [
    'torch', 'numpy', 'einops',
    'sana_wm_pipeline', 'nvidia_vipe',
    'static_ffmpeg', 'modelscope',
]
all_ok = True
for name in pkgs:
    try:
        m = __import__(name)
        ver = getattr(m, '__version__', 'ok')
        print(f"  ✓ {name:25s} {ver}")
    except ImportError as e:
        print(f"  ✗ {name:25s} MISSING: {e}")
        all_ok = False
print("✓ 全部通过" if all_ok else "\n⚠ 有缺失，检查 Task 1")
PY
```

全部 ✓ 才继续 Task 5。

---

### Task 5: C.1 单样本 smoke test

**Files:**
- 修改: `$PROJ_DIR/experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh`（sed 修复 AFS 路径，幂等）

- [ ] **Step 1: 激活环境 + sed 修复脚本**

```bash
source "$NEW_BASE/activate_sana_wm.sh"
cd "$PROJ_DIR"

E2E_SCRIPT="experiments/data_production_smoke/run_e2e_default_jdvbbfb.sh"

# 修复 AFS 硬编码路径（多次执行幂等，因为 sed 匹配具体模式）
sed -i \
  -e "s|^source /mnt/afs/davidwang/miniconda3/.*|source \"$ENV_DIR/bin/activate\"|" \
  -e '/^conda activate /d' \
  -e "s|export TORCH_HOME=.*|export TORCH_HOME=$NEW_BASE/cache/torch|" \
  -e "s|export HF_HOME=.*|export HF_HOME=$NEW_BASE/cache/huggingface|" \
  -e "s|export SANA_WM_PI3X_WEIGHTS=.*|export SANA_WM_PI3X_WEIGHTS=$NEW_BASE/models/pi3x|" \
  -e "s|export SANA_WM_MOGE2_WEIGHTS=.*|export SANA_WM_MOGE2_WEIGHTS=$NEW_BASE/models/moge2|" \
  "$E2E_SCRIPT"

# 验证 AFS 路径已消失
grep -n "/mnt/afs" "$E2E_SCRIPT" && echo "⚠ 还有 AFS 路径残留！" || echo "✓ 无 AFS 路径残留"
head -25 "$E2E_SCRIPT"
```

- [ ] **Step 2: 确认数据集就位**

```bash
echo "数据集子集数: $(ls $DATA_ROOT | grep -c wds-)"
# 期望: 8

echo "第一个 group 的 shard 数:"
ls "$DATA_ROOT/wds-DL3DV-ALL-2K/shards/" | wc -l
# 期望: >0（至少有几个 tar）
```

- [ ] **Step 3: 跑单样本 E2E（约 20-40 分钟）**

```bash
mkdir -p "$OUT_BASE/smoke"
bash "$E2E_SCRIPT" wds-DL3DV-ALL-2K 0 "$OUT_BASE/smoke"
```

首次运行时 VIPE JIT 编译约 2 分钟，属正常现象。

- [ ] **Step 4: 验证 smoke test 产物**

```bash
SMK="$OUT_BASE/smoke"
echo "=== Shard tar 是否存在 ==="
ls -lh "$SMK/shards_default/"*.tar 2>/dev/null || echo "⚠ 未找到 tar 产物"

echo "=== Shard 成员（期望 6 个：mp4 + 4个npy + caption + meta）==="
python3 -c "
import tarfile, sys
tars = __import__('glob').glob('$SMK/shards_default/shard-*.tar')
if not tars:
    print('⚠ 无 tar 文件')
    sys.exit(1)
tf = tarfile.open(sorted(tars)[0])
for m in tf.getmembers():
    print(' ', m.name)
"
# 期望成员:
# {scene_id}.mp4
# {scene_id}.poses_c2w.npy
# {scene_id}.intrinsics.npy
# {scene_id}.scale.npy
# {scene_id}.caption.txt
# {scene_id}.meta.json

echo "=== ATE 评估（可选）==="
cat "$SMK/shards_default/eval_output/pose_eval_summary.json" 2>/dev/null \
  || echo "(eval_output 可选，不影响生产)"
```

---

## 通过标准

- [ ] Task 1: `nvidia_vipe ✓` 打印成功
- [ ] Task 2: 6 个 VIPE 权重全部找到（含 geocalib .tar）
- [ ] Task 3: `activate_sana_wm.sh` 含 `PYTHONNOUSERSITE=1`；linker 软链完成
- [ ] Task 4: B.9 JIT PASS + B.10 全部 7 个包 ✓
- [ ] Task 5: smoke tar 含 6 个成员

全部通过后进入 C.2-C.4 全量生产。

---

## 依赖关系图

```
Task 1 (nvidia-vipe 安装)
    │
    ├─► Task 2 (权重核查，独立，可并行)
    │
    └─► Task 3 (激活脚本 + linker)
            │
            └─► Task 4 (JIT + 导入验证)
                    │
                    └─► Task 5 (smoke test)
```

Task 2 与 Task 1/3 相互独立，可在 Task 1 运行时并行核查。

#!/bin/bash
# ============================================================
# SANA-WM QC Pipeline 中移动部署测试脚本（v3 - 修复 DOVER 初始化）
# 修复：1. 添加 dover 安装  2. 修正 UniMatch 路径  3. 使用国内镜像源  4. 修复 DOVER 初始化参数
# ============================================================

set -euo pipefail

# ============================================================
# A. 环境变量定义
# ============================================================

export NEW_BASE=/root/work/david_work      # 热盘
export FS_DIR=/root/work/filestorage/shangaoooooo/davidwang/sana_wm_qc_pipeline  # 持久冷盘

# 以下路径固定
export ENV_DIR="$NEW_BASE/sana_wm_qc_env"
export PROJ_DIR="$NEW_BASE/sana_wm_qc"
export DATA_ROOT="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output"
export QC_OUT="$NEW_BASE/qc_output"
export QWEN_DIR="/root/work/david_work/models/Qwen3.5-27B"

# ⚠️ 新增：国内镜像源配置
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_TRUSTED_HOST="mirrors.aliyun.com"

echo "=== 环境变量 ==="
echo "NEW_BASE  = $NEW_BASE"
echo "FS_DIR    = $FS_DIR"
echo "ENV_DIR   = $ENV_DIR"
echo "PROJ_DIR  = $PROJ_DIR"
echo "DATA_ROOT = $DATA_ROOT"
echo "QC_OUT    = $QC_OUT"
echo "QWEN_DIR  = $QWEN_DIR"
echo "PIP_INDEX = $PIP_INDEX_URL"
echo ""

# ============================================================
# B. 创建目录结构
# ============================================================

echo "=== 创建目录结构 ==="
mkdir -p "$ENV_DIR" "$PROJ_DIR" \
         "$NEW_BASE/models" \
         "$NEW_BASE/cache/torch" \
         "$NEW_BASE/cache/huggingface" \
         "$FS_DIR" "$QC_OUT"
echo "✓ 目录创建完成"
echo ""

# ============================================================
# C. 部署 conda env
# ============================================================

if [ -f "$ENV_DIR/.cmcc_unpacked" ]; then
    echo "=== conda env 已存在，跳过解压 ==="
    echo "  标记文件: $ENV_DIR/.cmcc_unpacked"
else
    echo "=== C.1 解压 conda env（约 5-10 分钟）==="
    cd "$ENV_DIR"
    if [ ! -f "$FS_DIR/sana_wm_qc-cmcc.tar.gz" ]; then
        echo "❌ 错误: 未找到 $FS_DIR/sana_wm_qc-cmcc.tar.gz"
        exit 1
    fi
    time tar -xzf "$FS_DIR/sana_wm_qc-cmcc.tar.gz"
    echo "✓ 解压完成"
    echo ""

    echo "=== C.2 conda-unpack 修复 shebang/RPATH（约 30-60 秒）==="
    time "$ENV_DIR/bin/conda-unpack"
    touch "$ENV_DIR/.cmcc_unpacked"
    echo "✓ conda-unpack 完成"
    echo ""
fi

# ============================================================
# D. 激活 env 并验证
# ============================================================

echo "=== D.1 激活 conda env ==="
# ⚠️ 必须用 source bin/activate，不能用 conda activate
source "$ENV_DIR/bin/activate"
echo "✓ env 已激活"
echo ""

echo "=== D.2 验证基础环境 ==="
echo "Python   : $(which python)"
echo "pip      : $(which pip)"
python -c "import sys; print(f'Python {sys.version}')"
echo ""

echo "=== D.3 验证 PyTorch + CUDA ==="
python -c "import torch; print(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "✓ CUDA 可用"
else
    echo "❌ 警告: CUDA 不可用"
fi
echo ""

echo "=== D.4 验证 QC 依赖 ==="
python -c "import numpy, scipy, av, scenedetect; print('✓ numpy, scipy, av, scenedetect')"

# ⚠️ 修复：卸载错误的 dover CLI 工具，安装正确的 DOVER 视频质量评估
echo ""
echo "=== D.4.1 修复 DOVER（视频质量评估）==="
DOVER_DIR="/root/work/david_work/sana_qc_pipeline/DOVER"

if python -c "from dover import DOVER" 2>/dev/null; then
    echo "✓ DOVER 视频质量评估模型已安装"
else
    echo "⚠️  检测到错误的 dover 包（CLI 工具），正在卸载..."
    pip uninstall -y dover 2>/dev/null || true

    echo "正在从本地 DOVER 目录安装..."
    if [ ! -d "$DOVER_DIR" ]; then
        echo "❌ 错误: DOVER 目录不存在: $DOVER_DIR"
        echo ""
        echo "手动部署步骤："
        echo "  1. 确保 DOVER 目录已传输到目标机器"
        echo "  2. 路径应为: $DOVER_DIR"
        echo "  3. 目录应包含 setup.py 文件"
        exit 1
    fi

    if [ ! -f "$DOVER_DIR/setup.py" ]; then
        echo "❌ 错误: setup.py 不存在于 $DOVER_DIR"
        echo "   DOVER 目录可能不完整，请重新传输"
        exit 1
    fi

    # 从本地路径安装（网络隔离环境）
    echo "从 $DOVER_DIR 安装 DOVER 包..."
    pip install -e "$DOVER_DIR"

    if python -c "from dover import DOVER" 2>/dev/null; then
        echo "✓ DOVER 安装成功（从本地路径）"
    else
        echo "❌ DOVER 安装失败"
        echo "   请检查 $DOVER_DIR 目录是否完整"
        echo "   手动安装命令: pip install -e $DOVER_DIR"
        exit 1
    fi
fi

# 安装 PyYAML（DOVER 配置文件需要）
echo ""
echo "=== D.4.2 验证 PyYAML ==="
if python -c "import yaml" 2>/dev/null; then
    echo "✓ PyYAML 已安装"
else
    echo "正在安装 PyYAML..."
    pip install pyyaml -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
fi
echo ""

# ============================================================
# E. 部署项目代码
# ============================================================

if [ -d "$PROJ_DIR/src" ]; then
    echo "=== 项目代码已存在，跳过解压 ==="
else
    echo "=== E.1 解压项目代码 ==="
    if [ ! -f "$FS_DIR/sana_wm_qc-deploy.tar.gz" ]; then
        echo "❌ 错误: 未找到 $FS_DIR/sana_wm_qc-deploy.tar.gz"
        exit 1
    fi
    tar -xzf "$FS_DIR/sana_wm_qc-deploy.tar.gz" -C "$NEW_BASE"
    echo "✓ 项目代码解压完成: $PROJ_DIR"
    echo ""
fi

echo "=== E.2 创建 setup.py（如果不存在）==="
cd "$PROJ_DIR"
if [ ! -f setup.py ]; then
    cat > setup.py <<'SETUP_EOF'
from setuptools import setup, find_packages

setup(
    name="sana-wm-qc",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
SETUP_EOF
    echo "✓ setup.py 已创建"
else
    echo "✓ setup.py 已存在"
fi
echo ""

echo "=== E.3 安装 editable 包 ==="
pip install -e . --no-deps --no-build-isolation
echo "✓ editable 包安装完成"
echo ""

echo "=== E.4 验证 QC 包导入 ==="
if python -c "from sana_wm_pipeline.qc import stage1_fast; print('✓ stage1_fast')"; then
    echo "✓ QC 包可导入"
else
    echo "❌ QC 包导入失败"
    exit 1
fi
echo ""

# ============================================================
# F. 部署 UniMatch 模型
# ============================================================

# ⚠️ 修复：正确的 UniMatch 权重路径
UNIMATCH_WEIGHT="$NEW_BASE/models/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth"

if [ -f "$UNIMATCH_WEIGHT" ]; then
    echo "=== UniMatch 模型已存在，跳过解压 ==="
else
    echo "=== F.1 解压 UniMatch（约 1-2 分钟）==="
    if [ ! -f "$FS_DIR/sana_wm_qc-unimatch.tar.gz" ]; then
        echo "❌ 错误: 未找到 $FS_DIR/sana_wm_qc-unimatch.tar.gz"
        exit 1
    fi
    mkdir -p "$NEW_BASE/models"
    tar -xzf "$FS_DIR/sana_wm_qc-unimatch.tar.gz" -C "$NEW_BASE/models"
    echo "✓ UniMatch 解压完成"
    echo ""
fi

echo "=== F.2 验证 UniMatch 权重 ==="
if [ -f "$UNIMATCH_WEIGHT" ]; then
    ls -lh "$UNIMATCH_WEIGHT"
    echo "✓ UniMatch 权重存在（期望 ~29MB）"
else
    echo "❌ UniMatch 权重未找到: $UNIMATCH_WEIGHT"
    echo "检查解压内容:"
    find "$NEW_BASE/models/unimatch" -name "*.pth" 2>/dev/null || echo "未找到 .pth 文件"
    exit 1
fi
echo ""

# ============================================================
# F.3 验证 DOVER 配置和权重（已手动部署）
# ============================================================

echo "=== F.3 验证 DOVER 配置文件和预训练权重 ==="

# ⚠️ DOVER 仓库已由运维手动部署到目标机器
DOVER_DIR="/root/work/david_work/sana_qc_pipeline/DOVER"
DOVER_CONFIG="$DOVER_DIR/dover.yml"
DOVER_WEIGHT="$DOVER_DIR/pretrained_weights/DOVER.pth"

if [ ! -d "$DOVER_DIR" ]; then
    echo "❌ 错误: DOVER 目录不存在: $DOVER_DIR"
    echo ""
    echo "手动部署步骤："
    echo "  1. 克隆 DOVER 仓库:"
    echo "     cd /root/work/david_work"
    echo "     git clone https://github.com/VQAssessment/DOVER.git sana_qc_pipeline/DOVER"
    echo ""
    echo "  2. 下载预训练权重:"
    echo "     cd $DOVER_DIR"
    echo "     mkdir -p pretrained_weights"
    echo "     cd pretrained_weights"
    echo "     wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"
    exit 1
fi

if [ ! -f "$DOVER_CONFIG" ]; then
    echo "❌ 错误: DOVER 配置文件不存在: $DOVER_CONFIG"
    echo ""
    echo "诊断："
    echo "  DOVER 目录存在但配置文件缺失，可能是不完整的克隆"
    echo ""
    echo "修复步骤："
    echo "  cd /root/work/david_work"
    echo "  rm -rf sana_qc_pipeline/DOVER"
    echo "  git clone https://github.com/VQAssessment/DOVER.git sana_qc_pipeline/DOVER"
    exit 1
else
    echo "✓ DOVER 配置文件存在: $DOVER_CONFIG"
fi

if [ ! -f "$DOVER_WEIGHT" ]; then
    echo "❌ 错误: DOVER 预训练权重不存在: $DOVER_WEIGHT"
    echo ""
    echo "手动下载步骤："
    echo "  mkdir -p $DOVER_DIR/pretrained_weights"
    echo "  cd $DOVER_DIR/pretrained_weights"
    echo "  wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"
    echo ""
    echo "  或使用备用地址（如果 GitHub 不可达）："
    echo "  wget https://huggingface.co/teowu/DOVER/resolve/main/DOVER.pth"
    exit 1
else
    ls -lh "$DOVER_WEIGHT"
    echo "✓ DOVER 预训练权重存在（期望 ~200MB）"
fi
echo ""

# ============================================================
# G. 创建激活脚本
# ============================================================

echo "=== G.1 生成 activate_qc.sh ==="
cat > "$NEW_BASE/activate_qc.sh" <<ACTIVATE_EOF
#!/bin/bash
# 每次进入新 shell 后 source 这个文件
source "$ENV_DIR/bin/activate"

export TORCH_HOME="$NEW_BASE/cache/torch"
export HF_HOME="$NEW_BASE/cache/huggingface"
export UNIMATCH_DIR="$NEW_BASE/models/unimatch"
export DOVER_DIR="/root/work/david_work/sana_qc_pipeline/DOVER"
export QWEN_DIR="$QWEN_DIR"

export DATA_ROOT="$DATA_ROOT"
export QC_OUT="$QC_OUT"

# 国内镜像源
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export PIP_TRUSTED_HOST="mirrors.aliyun.com"

echo "✓ SANA-WM QC env 已激活"
echo "  DATA_ROOT=\$DATA_ROOT"
echo "  QC_OUT=\$QC_OUT"
echo "  QWEN_DIR=\$QWEN_DIR"
echo "  UNIMATCH_DIR=\$UNIMATCH_DIR"
echo "  DOVER_DIR=\$DOVER_DIR"
echo "  PIP_INDEX=\$PIP_INDEX_URL"
ACTIVATE_EOF

chmod +x "$NEW_BASE/activate_qc.sh"
echo "✓ activate_qc.sh 已创建: $NEW_BASE/activate_qc.sh"
echo ""

# 立即生效
source "$NEW_BASE/activate_qc.sh"

# ============================================================
# H. Stage 3 模型加载验证（可选，需 GPU）
# ============================================================

echo "=== H.1 验证 Qwen 模型路径 ==="
if [ -d "$QWEN_DIR" ]; then
    echo "✓ Qwen 目录存在: $QWEN_DIR"
    ls -lh "$QWEN_DIR" | head -5
else
    echo "❌ 警告: Qwen 目录不存在: $QWEN_DIR"
    echo "   Stage 3 测试将跳过"
fi
echo ""

echo "=== H.2 Stage 3 模型加载测试 ==="
read -p "是否运行 Stage 3 GPU 模型加载测试？(需要 GPU，约占用 58GB 显存) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "$QWEN_DIR" ]; then
        echo "❌ 跳过: Qwen 模型不存在"
    else
        echo "开始加载模型..."
        python3 <<'PYEOF'
import sys
import torch
import os

# 设置环境变量
unimatch_dir = os.environ.get("UNIMATCH_DIR", "/root/work/david_work/models/unimatch")
dover_dir = os.environ.get("DOVER_DIR", "/root/work/david_work/models/DOVER")
sys.path.insert(0, unimatch_dir)

print("=== 1. UniMatch ===")
try:
    from unimatch.unimatch import UniMatch
    um = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4,
                  num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
                  reg_refine=True, task="flow").cuda()
    print("✓ UniMatch loaded")
    del um
    torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ UniMatch failed: {e}")

print("\n=== 2. DOVER (视频质量评分) ===")
try:
    from dover import DOVER
    import yaml
    import warnings

    # ✅ 修复：使用正确的 DOVER 初始化方式
    # 1. 加载配置文件
    dover_config_path = os.path.join(dover_dir, "dover.yml")
    if not os.path.exists(dover_config_path):
        raise FileNotFoundError(f"DOVER 配置文件不存在: {dover_config_path}")

    with open(dover_config_path, "r") as f:
        dover_opt = yaml.safe_load(f)

    # ⚠️ WORKAROUND: H100 sm_90 不兼容，强制使用 CPU
    warnings.warn("DOVER 使用 CPU 模式（H100 兼容性问题）", RuntimeWarning)
    device = "cpu"

    # 2. 初始化模型（CPU 模式）
    dover_m = DOVER(**dover_opt["model"]["args"])

    # 3. 加载预训练权重（先到 CPU）
    dover_weight_path = os.path.join(dover_dir, "pretrained_weights/DOVER.pth")
    if not os.path.exists(dover_weight_path):
        raise FileNotFoundError(f"DOVER 权重文件不存在: {dover_weight_path}")

    dover_m.load_state_dict(torch.load(dover_weight_path, map_location="cpu", weights_only=False))
    dover_m = dover_m.to(device)
    dover_m.eval()

    print("✓ DOVER loaded (CPU mode)")
    print(f"  配置文件: {dover_config_path}")
    print(f"  权重文件: {dover_weight_path}")
    print("  架构: technical (Swin Tiny) + aesthetic (ConvNeXt Tiny)")
    print("  论文: Disentangled Objective Video Quality Evaluation (ICCV 2023)")
    print("  ⚠️  运行在 CPU 模式（H100 兼容性限制）")

    del dover_m
    torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ DOVER failed: {e}")
    print("\n  完整错误堆栈:")
    import traceback
    traceback.print_exc()
    print("\n  可能原因:")
    print("  1. DOVER 配置文件缺失（dover.yml）")
    print("  2. DOVER 预训练权重缺失（pretrained_weights/DOVER.pth）")
    print("  3. PyYAML 未安装")

print("\n=== 3. Qwen3.5-27B-VL ===")
try:
    qwen_dir = os.environ.get("QWEN_DIR")
    print(f"  加载路径: {qwen_dir}")
    print(f"  期望类型: Qwen3_5ForConditionalGeneration (from config.json)")

    from transformers import AutoModelForCausalLM

    qwen = AutoModelForCausalLM.from_pretrained(
        qwen_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()

    print("✓ Qwen3.5-27B-VL loaded")
    print(f"  实际类型: {type(qwen).__name__}")
    print(f"  验证 lm_head: {'有' if hasattr(qwen, 'lm_head') else '无（生成功能不可用）'}")
    print(f"  架构特性: Linear Attention (混合) + GQA")
    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    del qwen
    torch.cuda.empty_cache()

except Exception as e:
    print(f"❌ Qwen failed: {e}")
    print("\n  诊断建议:")
    print("  1. 检查模型目录是否包含 modeling_qwen3_5.py（必需）")
    print("  2. 如果缺失，执行：")
    print(f"     cd {qwen_dir or '$QWEN_DIR'}")
    print("     wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/modeling_qwen3_5.py")
    print("     wget https://huggingface.co/Qwen/Qwen3.5-27B/resolve/main/configuration_qwen3_5.py")
PYEOF
        echo "✓ Stage 3 模型加载测试完成"
    fi
else
    echo "⊘ 跳过 Stage 3 测试"
fi
echo ""

# ============================================================
# I. Stage 1 单 tar 冒烟测试
# ============================================================

echo "=== I.1 Stage 1 冒烟测试 ==="
read -p "是否运行 Stage 1 单 tar 冒烟测试？[y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # ✅ 硬编码测试路径（已验证存在）
    TEST_GROUP="wds-DL3DV-ALL-2K"
    TEST_TAR_DIR="/root/work/externalstorage/jtcvdatasets/cxy/jdvbbfb_output/final_wds-DL3DV-ALL-2K/wds-DL3DV-ALL-2K/w000"

    echo "测试 group: $TEST_GROUP"
    echo "测试 tar 目录: $TEST_TAR_DIR"
    echo ""

    # 检查目录是否存在
    if [ ! -d "$TEST_TAR_DIR" ]; then
        echo "❌ 测试目录不存在: $TEST_TAR_DIR"
    else
        # 检查 run_qc.py 是否存在
        if [ ! -f "$PROJ_DIR/scripts/run_qc.py" ]; then
            echo "❌ 未找到 scripts/run_qc.py"
        else
            mkdir -p "$QC_OUT/smoke_stage1"
            cd "$PROJ_DIR"

            # 询问是否运行完整 3 阶段测试
            read -p "是否运行完整 Stage 1+2+3 测试？(否则只运行 Stage 1) [y/N]: " -n 1 -r
            echo ""

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "开始完整 Stage 1+2+3 测试（w000 子目录，单进程）..."
                echo "⚠️  注意：Stage 3 包含 DOVER (CPU 模式) 和 Qwen，可能较慢"
                python scripts/run_qc.py \
                  --tar-root "$TEST_TAR_DIR" \
                  --group "$TEST_GROUP" \
                  --output-dir "$QC_OUT/smoke_full" \
                  --n-workers 1 || true

                OUTPUT_DIR="$QC_OUT/smoke_full"
            else
                echo "开始 Stage 1 测试（w000 子目录，单进程）..."
                python scripts/run_qc.py \
                  --tar-root "$TEST_TAR_DIR" \
                  --group "$TEST_GROUP" \
                  --output-dir "$QC_OUT/smoke_stage1" \
                  --n-workers 1 \
                  --skip-stage2 || true

                OUTPUT_DIR="$QC_OUT/smoke_stage1"
            fi

            if [ -f "$OUTPUT_DIR/stage1_results.jsonl" ]; then
                echo ""
                echo "=== Stage 1 输出 ==="
                echo "文件: $OUTPUT_DIR/stage1_results.jsonl"
                echo "样本数: $(wc -l < $OUTPUT_DIR/stage1_results.jsonl)"
                echo "前 3 行:"
                head -3 "$OUTPUT_DIR/stage1_results.jsonl"
                echo "✓ Stage 1 完成"
            else
                echo "❌ Stage 1 输出文件未生成"
            fi

            # 检查 Stage 2 输出
            if [ -f "$OUTPUT_DIR/stage2_results.jsonl" ]; then
                echo ""
                echo "=== Stage 2 输出 ==="
                echo "文件: $OUTPUT_DIR/stage2_results.jsonl"
                echo "样本数: $(wc -l < $OUTPUT_DIR/stage2_results.jsonl)"
                echo "✓ Stage 2 完成"
            fi

            # 检查 Stage 3 输出
            if [ -f "$OUTPUT_DIR/stage3_results.jsonl" ]; then
                echo ""
                echo "=== Stage 3 输出 ==="
                echo "文件: $OUTPUT_DIR/stage3_results.jsonl"
                echo "样本数: $(wc -l < $OUTPUT_DIR/stage3_results.jsonl)"
                echo "✓ Stage 3 完成"
            fi

            echo ""
            echo "✓ 冒烟测试完成"
        fi
    fi
else
    echo "⊘ 跳过 Stage 1 测试"
fi
echo ""

# ============================================================
# J. 部署完成总结
# ============================================================

echo "============================================================"
echo "部署测试完成"
echo "============================================================"
echo ""
echo "激活命令（每次新 shell）:"
echo "  source $NEW_BASE/activate_qc.sh"
echo ""
echo "关键路径:"
echo "  项目代码: $PROJ_DIR"
echo "  conda env: $ENV_DIR"
echo "  数据根目录: $DATA_ROOT"
echo "  输出目录: $QC_OUT"
echo "  Qwen 模型: $QWEN_DIR"
echo "  UniMatch : $NEW_BASE/models/unimatch"
echo "  DOVER    : /root/work/david_work/sana_qc_pipeline/DOVER"
echo ""
echo "pip 镜像源: $PIP_INDEX_URL"
echo ""
echo "下一步:"
echo "  1. 全量 Stage 1+2: bash scripts/run_all_groups_stage1_2.sh"
echo "  2. 全量 Stage 3   : bash scripts/run_all_groups_stage3.sh"
echo "  3. 生成报告      : bash scripts/generate_reports.sh"
echo ""

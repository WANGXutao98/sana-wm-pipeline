#!/bin/bash
# CMCC 离线部署脚本 - sana_qc_clean & sana_wm_clean
# 适用环境：无外网离线集群
# 功能：下载、解压、修复、激活 conda 环境

set -euo pipefail

################################################################################
# 配置区
################################################################################

# ModelScope 数据集 ID
DATASET_ID="davidxwang/sana_spatialvid_smoke_data"

# CMCC 工作目录
WORK_DIR="/root/work/david_work"
DOWNLOAD_DIR="$WORK_DIR/conda_envs_download"
INSTALL_DIR="$WORK_DIR/conda_envs"

# 环境名称
ENV_QC="sana_qc_clean"
ENV_WM="sana_wm_clean"

# 文件列表
FILES=(
    "sana_qc_clean.tar.gz"
    "sana_wm_clean.tar.gz"
    "sana_qc_clean.tar.gz.md5"
    "sana_wm_clean.tar.gz.md5"
)

################################################################################
# 步骤 1: 下载环境压缩包
################################################################################

download_files() {
    echo "========================================"
    echo "步骤 1: 下载 Conda 环境"
    echo "========================================"

    mkdir -p "$DOWNLOAD_DIR"
    cd "$DOWNLOAD_DIR"

    echo "使用 ModelScope CLI 下载..."
    modelscope download \
        --repo-type dataset \
        --dataset "$DATASET_ID" \
        --include 'conda_envs/*' \
        --local_dir .

    # 移动文件到当前目录
    if [ -d "conda_envs" ]; then
        mv conda_envs/* .
        rmdir conda_envs
    fi

    echo ""
    echo "下载完成，文件列表："
    ls -lh "${FILES[@]}"
    echo ""
}

################################################################################
# 步骤 2: 校验文件完整性
################################################################################

verify_files() {
    echo "========================================"
    echo "步骤 2: 校验文件完整性"
    echo "========================================"

    cd "$DOWNLOAD_DIR"

    for md5file in *.md5; do
        echo "校验 $md5file..."
        if md5sum -c "$md5file"; then
            echo "  ✓ 校验通过"
        else
            echo "  ✗ 校验失败"
            exit 1
        fi
    done

    echo ""
}

################################################################################
# 步骤 3: 解压环境
################################################################################

extract_envs() {
    echo "========================================"
    echo "步骤 3: 解压 Conda 环境"
    echo "========================================"

    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    # 解压 sana_qc_clean.tar.gz -> sana_qc -> 重命名为 sana_qc_clean
    echo "解压 sana_qc_clean.tar.gz..."
    tar -xzf "$DOWNLOAD_DIR/sana_qc_clean.tar.gz"
    if [ -d "sana_qc" ]; then
        mv sana_qc "$ENV_QC"
        echo "  ✓ 解压并重命名: $INSTALL_DIR/$ENV_QC"
    else
        echo "  ✗ 解压失败：未找到 sana_qc 目录"
        exit 1
    fi

    # 解压 sana_wm_clean.tar.gz -> sana_wm -> 重命名为 sana_wm_clean
    echo "解压 sana_wm_clean.tar.gz..."
    tar -xzf "$DOWNLOAD_DIR/sana_wm_clean.tar.gz"
    if [ -d "sana_wm" ]; then
        mv sana_wm "$ENV_WM"
        echo "  ✓ 解压并重命名: $INSTALL_DIR/$ENV_WM"
    else
        echo "  ✗ 解压失败：未找到 sana_wm 目录"
        exit 1
    fi

    echo ""
    echo "解压完成，环境列表："
    ls -lh "$INSTALL_DIR"
    echo ""
}

################################################################################
# 步骤 4: 修复环境路径（关键步骤）
################################################################################

fix_env_paths() {
    echo "========================================"
    echo "步骤 4: 修复环境路径"
    echo "========================================"

    # 修复 sana_qc_clean
    echo "修复 $ENV_QC..."
    cd "$INSTALL_DIR/$ENV_QC"

    # 修复 conda-meta 中的路径
    if [ -d "conda-meta" ]; then
        find conda-meta -name "*.json" -exec sed -i \
            "s|/mnt/afs/davidwang/miniconda3/envs/sana_qc|$INSTALL_DIR/$ENV_QC|g" {} \;
    fi

    # 修复 bin 目录中的 shebang
    if [ -d "bin" ]; then
        find bin -type f -exec sed -i \
            "s|/mnt/afs/davidwang/miniconda3/envs/sana_qc|$INSTALL_DIR/$ENV_QC|g" {} \;
    fi

    # 修复 lib 中的硬编码路径
    if [ -d "lib" ]; then
        find lib -name "*.py" -o -name "*.pth" | xargs sed -i \
            "s|/mnt/afs/davidwang/miniconda3/envs/sana_qc|$INSTALL_DIR/$ENV_QC|g" 2>/dev/null || true
    fi

    echo "  ✓ $ENV_QC 路径修复完成"

    # 修复 sana_wm_clean
    echo "修复 $ENV_WM..."
    cd "$INSTALL_DIR/$ENV_WM"

    if [ -d "conda-meta" ]; then
        find conda-meta -name "*.json" -exec sed -i \
            "s|/mnt/afs/davidwang/miniconda3/envs/sana_wm|$INSTALL_DIR/$ENV_WM|g" {} \;
    fi

    if [ -d "bin" ]; then
        find bin -type f -exec sed -i \
            "s|/mnt/afs/davidwang/miniconda3/envs/sana_wm|$INSTALL_DIR/$ENV_WM|g" {} \;
    fi

    if [ -d "lib" ]; then
        find lib -name "*.py" -o -name "*.pth" | xargs sed -i \
            "s|/mnt/afs/davidwang/miniconda3/envs/sana_wm|$INSTALL_DIR/$ENV_WM|g" 2>/dev/null || true
    fi

    echo "  ✓ $ENV_WM 路径修复完成"
    echo ""
}

################################################################################
# 步骤 5: 修复可编辑包（vipe & sana_wm_pipeline）
################################################################################

fix_editable_packages() {
    echo "========================================"
    echo "步骤 5: 修复可编辑包"
    echo "========================================"

    # 假设项目代码在 /root/work/david_work/sana_wm_optimized/sana_wm_pipeline
    PROJECT_DIR="$WORK_DIR/sana_wm_optimized/sana_wm_pipeline"

    if [ ! -d "$PROJECT_DIR" ]; then
        echo "⚠️  项目目录不存在: $PROJECT_DIR"
        echo "   请先下载项目代码，然后手动执行："
        echo "   source $INSTALL_DIR/$ENV_WM/bin/activate"
        echo "   pip install -e $PROJECT_DIR/third_party/vipe"
        echo "   pip install -e $PROJECT_DIR"
        echo ""
        return
    fi

    echo "激活 $ENV_WM 并重新安装可编辑包..."
    source "$INSTALL_DIR/$ENV_WM/bin/activate"

    # 安装 vipe
    if [ -d "$PROJECT_DIR/third_party/vipe" ]; then
        echo "安装 nvidia-vipe..."
        pip install -e "$PROJECT_DIR/third_party/vipe" --no-deps
        echo "  ✓ vipe 安装完成"
    else
        echo "  ⚠️  vipe 源码不存在: $PROJECT_DIR/third_party/vipe"
    fi

    # 安装 sana_wm_pipeline
    if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
        echo "安装 sana_wm_pipeline..."
        pip install -e "$PROJECT_DIR" --no-deps
        echo "  ✓ sana_wm_pipeline 安装完成"
    else
        echo "  ⚠️  项目配置不存在: $PROJECT_DIR/pyproject.toml"
    fi

    conda deactivate
    echo ""
}

################################################################################
# 步骤 6: 验证环境
################################################################################

verify_envs() {
    echo "========================================"
    echo "步骤 6: 验证环境"
    echo "========================================"

    # 验证 sana_qc_clean
    echo "验证 $ENV_QC..."
    source "$INSTALL_DIR/$ENV_QC/bin/activate"

    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

    conda deactivate
    echo "  ✓ $ENV_QC 验证通过"
    echo ""

    # 验证 sana_wm_clean
    echo "验证 $ENV_WM..."
    source "$INSTALL_DIR/$ENV_WM/bin/activate"

    python --version
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

    # 验证可编辑包
    if python -c "import vipe" 2>/dev/null; then
        echo "  ✓ vipe 可导入"
    else
        echo "  ⚠️  vipe 导入失败（需手动修复）"
    fi

    if python -c "import sana_wm_pipeline" 2>/dev/null; then
        echo "  ✓ sana_wm_pipeline 可导入"
    else
        echo "  ⚠️  sana_wm_pipeline 导入失败（需手动修复）"
    fi

    conda deactivate
    echo "  ✓ $ENV_WM 验证通过"
    echo ""
}

################################################################################
# 步骤 7: 生成激活脚本
################################################################################

generate_activate_scripts() {
    echo "========================================"
    echo "步骤 7: 生成激活脚本"
    echo "========================================"

    # sana_qc_clean 激活脚本
    cat > "$INSTALL_DIR/activate_qc.sh" <<'EOF'
#!/bin/bash
# 激活 sana_qc_clean 环境

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sana_qc_clean/bin/activate"

echo "✓ 已激活 sana_qc_clean 环境"
python --version
EOF
    chmod +x "$INSTALL_DIR/activate_qc.sh"

    # sana_wm_clean 激活脚本
    cat > "$INSTALL_DIR/activate_wm.sh" <<'EOF'
#!/bin/bash
# 激活 sana_wm_clean 环境

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sana_wm_clean/bin/activate"

echo "✓ 已激活 sana_wm_clean 环境"
python --version
EOF
    chmod +x "$INSTALL_DIR/activate_wm.sh"

    echo "生成激活脚本："
    echo "  $INSTALL_DIR/activate_qc.sh"
    echo "  $INSTALL_DIR/activate_wm.sh"
    echo ""
}

################################################################################
# 主流程
################################################################################

main() {
    echo "========================================"
    echo "CMCC Conda 环境离线部署"
    echo "========================================"
    echo "目标环境: $ENV_QC, $ENV_WM"
    echo "安装目录: $INSTALL_DIR"
    echo ""

    # 检查 modelscope CLI
    if ! command -v modelscope &>/dev/null; then
        echo "✗ modelscope CLI 未安装"
        echo "  请先安装: pip install modelscope"
        exit 1
    fi

    # 执行部署步骤
    download_files
    verify_files
    extract_envs
    fix_env_paths
    fix_editable_packages
    verify_envs
    generate_activate_scripts

    echo "========================================"
    echo "部署完成"
    echo "========================================"
    echo ""
    echo "快速激活："
    echo "  source $INSTALL_DIR/activate_qc.sh   # 激活 QC 环境"
    echo "  source $INSTALL_DIR/activate_wm.sh   # 激活 WM 环境"
    echo ""
    echo "或手动激活："
    echo "  source $INSTALL_DIR/sana_qc_clean/bin/activate"
    echo "  source $INSTALL_DIR/sana_wm_clean/bin/activate"
    echo ""
    echo "运行冒烟测试："
    echo "  source $INSTALL_DIR/activate_wm.sh"
    echo "  cd $WORK_DIR/sana_wm_optimized/sana_wm_pipeline"
    echo "  bash experiments/data_production_smoke/smoke_cmcc_pass.sh"
    echo ""
}

# 执行主流程
main "$@"

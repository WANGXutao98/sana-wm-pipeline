#!/bin/bash
# CMCC 环境激活修复脚本
# 问题：解压后重命名导致 activate 脚本失效
# 解决：创建新的激活脚本

set -euo pipefail

# 配置
INSTALL_DIR="/root/work/david_work/conda_envs"
ENV_QC="sana_qc_clean"
ENV_WM="sana_wm_clean"

echo "========================================"
echo "修复 CMCC Conda 环境激活"
echo "========================================"

# 修复 sana_qc_clean
if [ -d "$INSTALL_DIR/$ENV_QC" ]; then
    echo "修复 $ENV_QC..."

    # 创建激活脚本
    cat > "$INSTALL_DIR/$ENV_QC/bin/activate" <<'ACTIVATE_QC_EOF'
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
CONDA_EXE='/opt/conda/bin/conda'
_CONDA_ROOT='/opt/conda'
_CONDA_EXE='/opt/conda/bin/conda'
CONDA_PYTHON_EXE='/opt/conda/bin/python'

__conda_setup="$($_CONDA_EXE 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$_CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        . "$_CONDA_ROOT/etc/profile.d/conda.sh"
    else
        export PATH="$_CONDA_ROOT/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# 激活环境（设置环境变量）
CONDA_PREFIX="/root/work/david_work/conda_envs/sana_qc_clean"
export CONDA_PREFIX
export CONDA_DEFAULT_ENV="sana_qc_clean"
export CONDA_PROMPT_MODIFIER="(sana_qc_clean) "

# 更新 PATH
export PATH="$CONDA_PREFIX/bin:$PATH"

# 设置库路径
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Python 路径
export PYTHONHOME="$CONDA_PREFIX"

echo "✓ 已激活 sana_qc_clean 环境"
ACTIVATE_QC_EOF

    chmod +x "$INSTALL_DIR/$ENV_QC/bin/activate"
    echo "  ✓ $ENV_QC/bin/activate 已创建"
else
    echo "  ✗ $ENV_QC 目录不存在"
fi

echo ""

# 修复 sana_wm_clean
if [ -d "$INSTALL_DIR/$ENV_WM" ]; then
    echo "修复 $ENV_WM..."

    # 创建激活脚本
    cat > "$INSTALL_DIR/$ENV_WM/bin/activate" <<'ACTIVATE_WM_EOF'
# >>> conda initialize >>>
CONDA_EXE='/opt/conda/bin/conda'
_CONDA_ROOT='/opt/conda'
_CONDA_EXE='/opt/conda/bin/conda'
CONDA_PYTHON_EXE='/opt/conda/bin/python'

__conda_setup="$($_CONDA_EXE 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$_CONDA_ROOT/etc/profile.d/conda.sh" ]; then
        . "$_CONDA_ROOT/etc/profile.d/conda.sh"
    else
        export PATH="$_CONDA_ROOT/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# 激活环境
CONDA_PREFIX="/root/work/david_work/conda_envs/sana_wm_clean"
export CONDA_PREFIX
export CONDA_DEFAULT_ENV="sana_wm_clean"
export CONDA_PROMPT_MODIFIER="(sana_wm_clean) "

# 更新 PATH
export PATH="$CONDA_PREFIX/bin:$PATH"

# 设置库路径
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Python 路径
export PYTHONHOME="$CONDA_PREFIX"

echo "✓ 已激活 sana_wm_clean 环境"
ACTIVATE_WM_EOF

    chmod +x "$INSTALL_DIR/$ENV_WM/bin/activate"
    echo "  ✓ $ENV_WM/bin/activate 已创建"
else
    echo "  ✗ $ENV_WM 目录不存在"
fi

echo ""
echo "========================================"
echo "修复完成"
echo "========================================"
echo ""
echo "测试激活："
echo "  source $INSTALL_DIR/$ENV_QC/bin/activate"
echo "  python --version"
echo "  conda deactivate"
echo ""
echo "  source $INSTALL_DIR/$ENV_WM/bin/activate"
echo "  python --version"
echo ""

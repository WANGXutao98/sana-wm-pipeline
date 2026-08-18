#!/bin/bash
set -euo pipefail

# ModelScope 上传脚本
# 功能：上传 sana_qc_clean.tar.gz 和 sana_wm_clean.tar.gz 到 ModelScope 数据集
# 目标仓库：https://modelscope.cn/datasets/davidxwang/sana_spatialvid_smoke_data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
DATASET_ID="davidxwang/sana_spatialvid_smoke_data"
FILES=(
    "sana_qc_clean.tar.gz"
    "sana_wm_clean.tar.gz"
    "sana_qc_clean.tar.gz.md5"
    "sana_wm_clean.tar.gz.md5"
)

echo "========================================"
echo "ModelScope Conda 环境上传"
echo "========================================"
echo "数据集: $DATASET_ID"
echo "文件列表:"
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  ✓ $f ($size)"
    else
        echo "  ✗ $f (不存在)"
        exit 1
    fi
done
echo ""

# 检查 modelscope CLI
# if ! command -v modelscope &>/dev/null; then
#     echo "安装 modelscope CLI..."
#     pip install modelscope -q
# fi

# # 检查登录状态
# echo "检查 ModelScope 登录状态..."
# if ! modelscope whoami &>/dev/null; then
#     echo ""
#     echo "未登录 ModelScope，请先登录："
#     echo "  modelscope login --token YOUR_TOKEN"
#     echo ""
#     echo "获取 token: https://modelscope.cn/my/myaccesstoken"
#     exit 1
# fi

# echo "已登录: $(modelscope whoami)"
# echo ""

# 上传文件
for f in "${FILES[@]}"; do
    echo "上传 $f..."
    remote_path="conda_envs/$f"

    modelscope upload \
        --repo-type dataset \
        "$DATASET_ID" \
        "$f" \
        "$remote_path" \
        2>&1 | tee "upload_${f}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "  ✓ $f 上传成功"
    else
        echo "  ✗ $f 上传失败，查看日志: upload_${f}.log"
        exit 1
    fi
    echo ""
done

echo "========================================"
echo "上传完成"
echo "========================================"
echo "数据集地址: https://modelscope.cn/datasets/$DATASET_ID/files"
echo ""
echo "CMCC 端下载命令:"
echo "  modelscope download --dataset $DATASET_ID --include 'conda_envs/*'"
echo ""

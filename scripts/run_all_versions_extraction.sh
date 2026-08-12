#!/bin/bash
################################################################################
# SANA-WM 训练数据提取 - 三版本串行执行脚本
#
# 功能：按顺序依次提取 v1.0 → v1.1 → v1.2 三个版本的训练数据
#
# 执行逻辑：
#   1. 每个版本独立提取到专属子目录
#   2. 提取完成后自动打包归档
#   3. 生成详细日志和报告
#   4. 异常自动中断，防止数据覆盖
#
# 使用方法：
#   bash run_all_versions_extraction.sh
################################################################################

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错
set -o pipefail  # 管道命令任一失败则整体失败

################################################################################
# 配置参数（根据实际情况修改）
################################################################################

# 筛选列表文件路径（AFS 开发环境）
FILTERED_V1_0="/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples.jsonl"
FILTERED_V1_1="/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples_v1.1_with_acceptable.jsonl"
FILTERED_V1_2="/mnt/afs/davidwang/workspace/sana_wm_pipeline/sana-human-feedback/filtered_training_samples_v1.2_with_dl3dv.jsonl"

# CMCC 原始数据根目录
DATA_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_output"

# 输出根目录
OUTPUT_ROOT="/root/work/filestorage/shangaoooooo/davidwang/jdvbbfb_guiyu_filtered_output"

# 提取脚本路径
EXTRACT_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extract_training_data_from_filtered_corrected.py"

# Python 可执行文件
PYTHON_CMD="python3"

################################################################################
# 版本配置
################################################################################

declare -A VERSION_INFO=(
    ["v1.0_samples"]=1980
    ["v1.0_description"]="高质量基线数据集（excellent + good）"
    ["v1.1_samples"]=2651
    ["v1.1_description"]="扩充数据集（+ acceptable）"
    ["v1.2_samples"]=4720
    ["v1.2_description"]="大规模数据集（+ DL3DV）"
)

################################################################################
# 日志函数
################################################################################

LOG_FILE="${OUTPUT_ROOT}/extraction_pipeline.log"

# 初始化日志
init_log() {
    mkdir -p "${OUTPUT_ROOT}"
    echo "====================================================================" > "${LOG_FILE}"
    echo "SANA-WM 训练数据提取 - 串行执行日志" >> "${LOG_FILE}"
    echo "====================================================================" >> "${LOG_FILE}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
}

# 日志函数
log_info() {
    local msg="[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$msg" | tee -a "${LOG_FILE}"
}

log_success() {
    local msg="[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "\033[32m$msg\033[0m" | tee -a "${LOG_FILE}"
}

log_error() {
    local msg="[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "\033[31m$msg\033[0m" | tee -a "${LOG_FILE}"
}

log_warn() {
    local msg="[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "\033[33m$msg\033[0m" | tee -a "${LOG_FILE}"
}

################################################################################
# 环境检查
################################################################################

check_environment() {
    log_info "====== 环境检查 ======"

    # 检查 Python
    if ! command -v ${PYTHON_CMD} &> /dev/null; then
        log_error "Python 未安装或不在 PATH 中"
        exit 1
    fi
    log_info "✓ Python: $(${PYTHON_CMD} --version)"

    # 检查提取脚本
    if [ ! -f "${EXTRACT_SCRIPT}" ]; then
        log_error "提取脚本不存在: ${EXTRACT_SCRIPT}"
        exit 1
    fi
    log_info "✓ 提取脚本: ${EXTRACT_SCRIPT}"

    # 检查数据根目录
    if [ ! -d "${DATA_ROOT}" ]; then
        log_error "数据根目录不存在: ${DATA_ROOT}"
        exit 1
    fi
    log_info "✓ 数据根目录: ${DATA_ROOT}"

    # 检查筛选列表文件
    for version in "v1.0" "v1.1" "v1.2"; do
        local var_name="FILTERED_${version//./_}"
        local file_path="${!var_name}"
        if [ ! -f "${file_path}" ]; then
            log_error "${version} 筛选列表不存在: ${file_path}"
            exit 1
        fi
        log_info "✓ ${version} 筛选列表: ${file_path}"
    done

    log_success "环境检查通过"
    echo ""
}

################################################################################
# 单版本提取函数
################################################################################

extract_version() {
    local version="$1"
    local filtered_list="$2"
    local expected_samples="$3"
    local description="$4"

    log_info "====== 开始提取 ${version} ======"
    log_info "描述: ${description}"
    log_info "预期样本数: ${expected_samples}"
    log_info "筛选列表: ${filtered_list}"

    # 创建版本输出目录
    local output_dir="${OUTPUT_ROOT}/${version}"
    mkdir -p "${output_dir}"

    local start_time=$(date +%s)

    # 执行提取
    log_info "正在提取..."
    if ${PYTHON_CMD} "${EXTRACT_SCRIPT}" \
        --filtered_list "${filtered_list}" \
        --data_root "${DATA_ROOT}" \
        --output_dir "${output_dir}" \
        --log_file "extraction_${version}.log"; then

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log_success "${version} 提取完成，耗时: ${duration} 秒"

        # 统计文件数
        local file_count=$(find "${output_dir}" -type f ! -name "*.log" ! -name "*.txt" ! -name "*.jsonl" | wc -l)
        log_info "${version} 提取文件数: ${file_count} (预期: $((expected_samples * 5)))"

        return 0
    else
        log_error "${version} 提取失败"
        return 1
    fi
}

################################################################################
# 打包归档函数
################################################################################

archive_version() {
    local version="$1"
    local output_dir="${OUTPUT_ROOT}/${version}"

    log_info "====== 开始归档 ${version} ======"

    local archive_name="${version}_$(date '+%Y%m%d_%H%M%S').tar.gz"
    local archive_path="${OUTPUT_ROOT}/${archive_name}"

    log_info "归档文件: ${archive_path}"
    log_info "正在打包..."

    local start_time=$(date +%s)

    if tar -czf "${archive_path}" -C "${OUTPUT_ROOT}" "${version}/"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        local archive_size=$(du -h "${archive_path}" | cut -f1)
        log_success "${version} 归档完成"
        log_info "归档大小: ${archive_size}"
        log_info "归档耗时: ${duration} 秒"

        # 生成 MD5 校验
        log_info "生成 MD5 校验..."
        md5sum "${archive_path}" > "${archive_path}.md5"
        log_success "MD5 校验文件: ${archive_path}.md5"

        return 0
    else
        log_error "${version} 归档失败"
        return 1
    fi
}

################################################################################
# 生成最终报告
################################################################################

generate_final_report() {
    log_info "====== 生成最终报告 ======"

    local report_file="${OUTPUT_ROOT}/EXTRACTION_FINAL_REPORT.txt"

    cat > "${report_file}" << EOF
================================================================================
SANA-WM 训练数据提取 - 最终报告
================================================================================

执行时间: $(date '+%Y-%m-%d %H:%M:%S')

【一、版本统计】

EOF

    for version in "v1.0" "v1.1" "v1.2"; do
        local output_dir="${OUTPUT_ROOT}/${version}"
        local samples_key="${version}_samples"
        local desc_key="${version}_description"
        local expected_samples=${VERSION_INFO[$samples_key]}
        local description=${VERSION_INFO[$desc_key]}

        echo "版本: ${version}" >> "${report_file}"
        echo "  描述: ${description}" >> "${report_file}"
        echo "  预期样本数: ${expected_samples}" >> "${report_file}"

        if [ -d "${output_dir}" ]; then
            local file_count=$(find "${output_dir}" -type f ! -name "*.log" ! -name "*.txt" ! -name "*.jsonl" | wc -l)
            local dir_size=$(du -sh "${output_dir}" | cut -f1)
            echo "  实际文件数: ${file_count}" >> "${report_file}"
            echo "  目录大小: ${dir_size}" >> "${report_file}"
            echo "  状态: ✓ 完成" >> "${report_file}"
        else
            echo "  状态: ✗ 未完成" >> "${report_file}"
        fi
        echo "" >> "${report_file}"
    done

    cat >> "${report_file}" << EOF

【二、归档文件】

EOF

    for archive in "${OUTPUT_ROOT}"/*.tar.gz; do
        if [ -f "${archive}" ]; then
            local archive_name=$(basename "${archive}")
            local archive_size=$(du -h "${archive}" | cut -f1)
            echo "  ${archive_name} (${archive_size})" >> "${report_file}"
        fi
    done

    cat >> "${report_file}" << EOF

【三、输出目录结构】

${OUTPUT_ROOT}/
├── v1.0/                     # v1.0 提取数据
├── v1.1/                     # v1.1 提取数据
├── v1.2/                     # v1.2 提取数据
├── v1.0_YYYYMMDD_HHMMSS.tar.gz  # v1.0 归档
├── v1.1_YYYYMMDD_HHMMSS.tar.gz  # v1.1 归档
├── v1.2_YYYYMMDD_HHMMSS.tar.gz  # v1.2 归档
├── extraction_pipeline.log   # 执行日志
└── EXTRACTION_FINAL_REPORT.txt  # 本报告

【四、下一步操作】

1. 验证提取数据完整性
   cd ${OUTPUT_ROOT}
   for v in v1.0 v1.1 v1.2; do
       echo "检查 \$v..."
       find \$v -type f ! -name "*.log" ! -name "*.txt" | wc -l
   done

2. 验证归档文件完整性
   for f in *.tar.gz; do
       md5sum -c \${f}.md5
   done

3. 将归档文件传输到训练服务器
   rsync -av --progress ${OUTPUT_ROOT}/*.tar.gz training_server:/path/to/training_data/

================================================================================
提取完成！
================================================================================
EOF

    log_success "最终报告已生成: ${report_file}"
    cat "${report_file}"
}

################################################################################
# 主执行流程
################################################################################

main() {
    # 初始化日志
    init_log

    log_info "====== SANA-WM 训练数据提取 - 三版本串行执行 ======"
    log_info "输出根目录: ${OUTPUT_ROOT}"
    log_info ""

    # 环境检查
    check_environment

    local overall_start_time=$(date +%s)

    # v1.0 提取
    if extract_version "v1.0" "${FILTERED_V1_0}" 1980 "高质量基线数据集（excellent + good）"; then
        archive_version "v1.0"
        log_info ""
    else
        log_error "v1.0 提取失败，终止执行"
        exit 1
    fi

    # v1.1 提取
    if extract_version "v1.1" "${FILTERED_V1_1}" 2651 "扩充数据集（+ acceptable）"; then
        archive_version "v1.1"
        log_info ""
    else
        log_error "v1.1 提取失败，终止执行"
        exit 1
    fi

    # v1.2 提取
    if extract_version "v1.2" "${FILTERED_V1_2}" 4720 "大规模数据集（+ DL3DV）"; then
        archive_version "v1.2"
        log_info ""
    else
        log_error "v1.2 提取失败，终止执行"
        exit 1
    fi

    local overall_end_time=$(date +%s)
    local overall_duration=$((overall_end_time - overall_start_time))

    # 生成最终报告
    generate_final_report

    log_success "====== 所有版本提取完成 ======"
    log_info "总耗时: ${overall_duration} 秒 ($((overall_duration / 60)) 分钟)"
    log_info "输出目录: ${OUTPUT_ROOT}"
    log_info "详细日志: ${LOG_FILE}"
    log_info ""

    echo ""
    echo "================================================================================
    echo "✅ 三版本数据提取全部完成！"
    echo "================================================================================"
    echo ""
    echo "快速验证："
    echo "  cd ${OUTPUT_ROOT}"
    echo "  ls -lh"
    echo "  cat EXTRACTION_FINAL_REPORT.txt"
    echo ""
}

################################################################################
# 执行主函数
################################################################################

main "$@"

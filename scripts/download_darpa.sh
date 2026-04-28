#!/bin/bash
# CAAAPT: Simple download script for DARPA TC3 data from official Google Drive
# Source: https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md

# --- 配置 ---
# Google Drive 文件ID (需要从官方分享链接中提取)
# 注意: 这些ID仅为示例，您需要根据实际发布的TC3数据链接进行替换。
# 您可以从项目的 "data/" 目录下找到每个数据集（如 cadets, theia, trace）对应的Google Drive链接。
FILE_ID_EXAMPLE="YOUR_FILE_ID_HERE"
OUTPUT_FILE="darpa_tc3_data.tar.gz"
OUTPUT_DIR="../data/raw/darpa_tc3"

# --- 函数: 检查并安装 gdown ---
check_and_install_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown 未安装，正在尝试安装..."
        pip install gdown || {
            echo "错误: 无法安装 gdown。请手动运行 'pip install gdown' 后重试。"
            exit 1
        }
    fi
}

# --- 主下载逻辑 ---
download_from_gdrive() {
    local file_id=$1
    local output_path=$2

    echo "正在从 Google Drive (ID: $file_id) 下载文件到 $output_path ..."
    gdown --id "$file_id" -O "$output_path"
    if [ $? -eq 0 ]; then
        echo "下载成功: $output_path"
    else
        echo "下载失败: 请检查文件ID是否正确，或网络连接。"
        exit 1
    fi
}

# --- 解压提示 ---
extract_hint() {
    echo ""
    echo "数据已下载完成。要解压文件，请运行以下命令:"
    echo "mkdir -p $OUTPUT_DIR && tar -xzvf $OUTPUT_FILE -C $OUTPUT_DIR"
}

# --- 主程序 ---
main() {
    echo "=== DARPA TC3 数据下载助手 ==="
    echo "请根据官方 GitHub 仓库说明，获取正确的 Google Drive 文件ID。"
    echo "参考链接: https://github.com/darpa-i2o/Transparent-Computing"

    # 检查并安装 gdown
    check_and_install_gdown

    # 提示用户输入文件ID（更灵活的方式）
    read -p "请输入要下载的数据集 (如 cadets, theia, trace) 对应的 Google Drive 文件ID: " user_file_id

    if [[ -z "$user_file_id" || "$user_file_id" == "YOUR_FILE_ID_HERE" ]]; then
        echo "错误: 必须提供有效的 Google Drive 文件ID。"
        exit 1
    fi

    # 创建输出目录
    mkdir -p "$(dirname "$OUTPUT_FILE")"

    # 执行下载
    download_from_gdrive "$user_file_id" "$OUTPUT_FILE"

    # 给出解压提示
    extract_hint
}

# 运行主程序
main
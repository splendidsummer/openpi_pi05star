#!/bin/bash
# 手动下载 GitHub 依赖以避免 uv 每次更新

echo "手动下载 GitHub 依赖..."

# 创建临时目录
TMP_DIR=/tmp/git_deps
mkdir -p $TMP_DIR

# 下载 dlimp
echo "下载 dlimp..."
cd $TMP_DIR
if [ ! -d "dlimp" ]; then
    git clone https://github.com/kvablack/dlimp.git
    cd dlimp
    git checkout ad72ce3a9b414db2185bc0b38461d4101a65477a
    cd ..
fi

# 下载 lerobot
echo "下载 lerobot..."
if [ ! -d "lerobot" ]; then
    git clone --recurse-submodules https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
    cd ..
fi

echo "✓ GitHub 依赖下载完成"
echo "现在可以使用 uv run --frozen 运行脚本"

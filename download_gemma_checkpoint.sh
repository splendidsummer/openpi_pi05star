#!/bin/bash
# 下载 Gemma 3 270M 预训练检查点

CHECKPOINT_PATH="gs://gemma-data/checkpoints/gemma3-270m-pt"
LOCAL_DIR="/root/autodl-tmp/gemma-3-270m"

echo "=========================================="
echo "下载 Gemma 3 270M 预训练检查点"
echo "=========================================="
echo "源路径: $CHECKPOINT_PATH"
echo "目标路径: $LOCAL_DIR"
echo ""

# 检查 gsutil 是否安装
# 优先使用虚拟环境中的 gsutil
if [ -f "/root/autodl-tmp/openpi/.venv/bin/gsutil" ]; then
    GSUTIL="/root/autodl-tmp/openpi/.venv/bin/gsutil"
elif command -v gsutil &> /dev/null; then
    GSUTIL="gsutil"
else
    echo "错误: gsutil 未安装"
    echo ""
    echo "安装方法（使用 uv pip 和清华镜像源）:"
    echo "  cd /root/autodl-tmp/openpi"
    echo "  source .venv/bin/activate"
    echo "  uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gsutil"
    echo ""
    echo "或者使用标准 pip:"
    echo "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gsutil"
    exit 1
fi

echo "使用 gsutil: $GSUTIL"
echo "版本信息:"
$GSUTIL --version
echo ""

# 创建目标目录
mkdir -p "$LOCAL_DIR"

echo "开始下载..."
echo "注意: 这可能需要一些时间，取决于网络速度"
echo ""

# 使用 gsutil 递归下载（推荐使用 rsync，支持断点续传）
echo "使用 rsync 下载（支持断点续传）..."
$GSUTIL -m rsync -r "$CHECKPOINT_PATH" "$LOCAL_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 下载完成！"
    echo "=========================================="
    echo "检查点已保存到: $LOCAL_DIR"
    echo ""
    echo "验证下载的文件:"
    ls -lh "$LOCAL_DIR" | head -10
else
    echo ""
    echo "=========================================="
    echo "✗ 下载失败"
    echo "=========================================="
    echo "请检查:"
    echo "1. 网络连接是否正常"
    echo "2. 是否有访问 gs://gemma-data 的权限"
    echo "3. 磁盘空间是否充足"
    exit 1
fi

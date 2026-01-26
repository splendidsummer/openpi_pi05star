#!/bin/bash
# 修复 git 网络问题

echo "配置 git 使用 HTTP/1.1 以解决网络问题..."

# 配置 git 使用 HTTP/1.1
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

echo "✓ Git 配置已更新"
echo ""
echo "当前配置:"
git config --global --list | grep http

echo ""
echo "现在可以重试 uv sync 或 uv run 命令"

#!/bin/bash
# 使用 --frozen 模式运行 uv，避免更新 GitHub 依赖

# 设置环境变量确保使用冻结模式
export UV_FROZEN=1

# 运行命令
uv run --frozen "$@"




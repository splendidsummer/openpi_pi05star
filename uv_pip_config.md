# UV Pip 配置清华镜像源

## 方法 1: 临时使用（推荐）

在每次安装时指定镜像源：

```bash
cd /root/autodl-tmp/openpi
source .venv/bin/activate
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package_name>
```

## 方法 2: 配置环境变量

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install <package_name>
```

## 方法 3: 在 pyproject.toml 中配置（如果支持）

目前 uv 可能不支持在 pyproject.toml 中直接配置镜像源，建议使用方法 1。

## 安装 gsutil 示例

```bash
cd /root/autodl-tmp/openpi
source .venv/bin/activate
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gsutil
```

## 其他常用镜像源

- 清华大学: https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云: https://mirrors.aliyun.com/pypi/simple/
- 中科大: https://pypi.mirrors.ustc.edu.cn/simple/
- 豆瓣: https://pypi.douban.com/simple/



cd /root/autodl-tmp/openpi && cat << 'EOF'
# 解决方案：使用 --frozen 而不是 --offline

# --offline 会完全禁用网络，但如果依赖未缓存会失败
# --frozen 会使用锁定的版本，不会更新，但允许下载缺失的依赖

# 正确的方法：
uv run --frozen scripts/compute_norm_stats.py --config-name pi05_libero

# 或者先确保所有依赖都已同步：
uv sync --frozen
# 然后再运行（此时应该不会更新）
uv run scripts/compute_norm_stats.py --config-name pi05_libero
EOF

# 使用默认本地路径（/root/autodl-tmp/gemma-3-270m）
uv run check_gemma_model.py

# 使用自定义本地路径
GEMMA_CHECKPOINT_PATH=/path/to/other/checkpoint uv run check_gemma_model.py

# 使用在线路径
GEMMA_CHECKPOINT_PATH=online uv run check_gemma_model.py

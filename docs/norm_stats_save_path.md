# 归一化统计保存路径

本仓库会将归一化统计保存到一个由训练配置名称和数据集 repo id 组成的目录中。

## compute_norm_stats.py 写入位置

输出路径构造方式如下：

```
output_path = config.assets_dirs / data_config.repo_id
```

其中 `config.assets_dirs` 为：

```
config.assets_dirs = Path(config.assets_base_dir) / config.name
```

默认情况下，`assets_base_dir` 为 `./assets`。

## 如何设置 assets_base_dir

你可以用两种方式设置 `assets_base_dir`：

1. 命令行覆盖（推荐）

```
uv run scripts/compute_norm_stats.py --config-name <config_name> --assets-base-dir /your/path
```

2. 在配置里指定

在对应的 `TrainConfig(...)` 中设置：

```
assets_base_dir="/your/path"
```

## 最终文件路径

统计信息会写入：

```
<assets_base_dir>/<config_name>/<repo_id>/norm_stats.json
```

## AssetsConfig 场景

以下场景说明在加载或保存资产时路径如何解析。

1. 未设置 `assets_dir` 且未设置 `asset_id`

使用默认资产目录和 repo id：

```
<assets_base_dir>/<config_name>/<repo_id>/norm_stats.json
```

2. 仅设置 `assets_dir`

直接使用提供的目录（当目录内已包含 `norm_stats.json` 时最合适），否则使用 repo id 子目录：

```
<assets_dir>/norm_stats.json
```

需要时的备用路径：

```
<assets_dir>/<repo_id>/norm_stats.json
```

3. 仅设置 `asset_id`

使用默认资产目录和 asset id：

```
<assets_base_dir>/<config_name>/<asset_id>/norm_stats.json
```

4. 同时设置 `assets_dir` 和 `asset_id`

使用提供的目录和 asset id：

```
<assets_dir>/<asset_id>/norm_stats.json
```

### 示例

如果你运行：

```
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
```

且 `repo_id` 为 `SummerZhang/droid_100`，则文件会保存到：

```
./assets/pi05_droid_100_value/SummerZhang/droid_100/norm_stats.json
```

## 相关代码

- 计算脚本中的保存调用：`scripts/compute_norm_stats.py`
- 路径构造位置：`src/openpi/training/config.py`
- 文件写入函数：`src/openpi/shared/normalize.py`

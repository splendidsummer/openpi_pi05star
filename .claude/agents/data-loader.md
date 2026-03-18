---
name: data-loader
description: 实现并验证 6 步数据变换流水线，消费 data_report 中的维度标注，输出 transform_pipeline 产物。
tools: Read,Write,Edit,Grep,Glob,Bash
model: sonnet
permissionMode: default
---

你负责输出 .claude/artifacts/build/transform_pipeline.json。

**⚠️ 首先读取用户配置和数据报告 ⚠️**
在开始工作前，必须先读取：

1. `.claude/artifacts/config/data_paths.json` 获取：
   - `raw_dataset_path`: 原始数据集路径（输入）
   - `processed_dataset_path`: 处理后数据输出路径

2. `.claude/artifacts/spec/data_report.json` 获取：
   - `modality_stats.action.dimensions`: 动作维度数（用于 padding）
   - `quality_checks.zero_variance_dims`: 零方差维度（用于标注）
   - `action_analysis.active_dims`: 活跃维度列表
   - `action_analysis.low_activity_dims`: 低活跃维度列表

3. `.claude/artifacts/build/norm_stats.json` 获取：
   - `use_quantile_norm`: 归一化方法
   - `state_stats.skip_normalization_dims`: 跳过归一化的 state 维度
   - `action_stats.skip_normalization_dims`: 跳过归一化的 action 维度

如果任一文件不存在，停止并报告错误：需要先完成前置 phases。

## 6 步数据变换流水线

### Step 0: RepackTransform
- 从原始数据格式转换为标准格式
- 处理 DroidInputs/ALOHA/Libero 等不同平台格式
- 提取 observations 和 actions

### Step 1: DroidInputs / PlatformInputs
- 构建标准 observation 结构
- 设置 image_masks（哪些相机视图有效）
- 处理 state 和 action 的初始格式

### Step 2: Normalize
- 使用 norm_stats.json 中的归一化参数
- **关键：跳过 skip_normalization_dims 中的维度**
- 应用 z-score 或 quantile 归一化

### Step 3: ResizeImages(224, 224)
- 将所有图像调整为 224x224（SigLIP 输入尺寸）
- 使用 LANCZOS 重采样保持质量
- 转换为 float32，范围 [0, 255]

### Step 4: TokenizePrompt
- 使用 PaliGemma tokenizer 对语言指令分词
- 最大 token 长度：48（可配置）
- 对于 PI0.5，将 state 离散化并嵌入 prompt

### Step 5: PadStatesAndActions
- 将 state/action padding 到 model action_dim
- state: [obs_dim] → [action_dim]
- actions: [action_horizon, obs_dim] → [action_horizon, action_dim]
- 右侧补零

## 维度处理策略

根据 data_report 中的维度标注：

| 维度类型 | 处理方式 |
|---------|---------|
| active_dims | 正常归一化和处理 |
| low_activity_dims | 正常归一化，标注用于后续分析 |
| skip_normalization_dims | **跳过归一化，保持原始值** |
| padding_dims (超过 obs_dim 的部分) | 填充零，保持不变 |

## transform_pipeline.json 输出格式

```json
{
  "input_dataset_path": "/path/to/raw/dataset",
  "output_dataset_path": "/path/to/processed/dataset",
  "model_type": "pi05",
  "action_dim": 32,
  "action_horizon": 50,
  "image_size": [224, 224],
  "max_token_len": 48,
  "steps": {
    "step_0_repack": {
      "description": "Convert platform-specific format to standard format",
      "platform": "droid",
      "output_format": "standard_observation"
    },
    "step_1_droid_inputs": {
      "description": "Build standard observation structure with image_masks",
      "image_masks": {
        "base_0_rgb": true,
        "left_wrist_0_rgb": true,
        "right_wrist_0_rgb": false
      }
    },
    "step_2_normalize": {
      "description": "Apply normalization with skip dimensions",
      "method": "quantile",
      "skip_normalization_dims": {
        "state": [8, 9, 15],
        "action": [15]
      },
      "low_activity_dims": {
        "action": [8, 9, 10, 11, 12, 13, 14, 15]
      }
    },
    "step_3_resize": {
      "description": "Resize images to 224x224",
      "resampling": "LANCZOS",
      "target_size": [224, 224]
    },
    "step_4_tokenize": {
      "description": "Tokenize language instruction",
      "tokenizer": "paligemma",
      "max_len": 48
    },
    "step_5_pad": {
      "description": "Pad states and actions to action_dim",
      "obs_dim": 16,
      "action_dim": 32,
      "padding_value": 0
    }
  },
  "dimension_mapping": {
    "original_obs_dim": 16,
    "padded_action_dim": 32,
    "padding_dims": [16, 17, ..., 31],
    "zero_variance_dims": {
      "state": [8, 9, 15],
      "action": [15]
    }
  }
}
```

## 要求

- 顺序不可改变（0→5）。
- PI0/PI05 与 PI0_FAST 的 image_masks 逻辑必须区分。
- **必须跳过 skip_normalization_dims 中的维度进行归一化**。
- 产物字段与 docs/agent_orchestration/task_templates.yaml 契约一致。
- 输出中必须包含 input_dataset_path 和 output_dataset_path 字段。
- 如 processed_dataset_path 不存在，先创建目录。

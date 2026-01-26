# 多学习率优化器实现计划

## 概述

计划为 Value 模型训练中的不同参数组实现不同的学习率：
- **SigLIP（视觉编码器）**: lr = 1e-5
- **LLM（Gemma 骨干网络）**: lr = 5e-5  
- **Value head（价值头）**: lr = 1e-4

## 当前状态

- LLM 和视觉模型（SigLIP）都已解冻（可训练）
- 目前对所有参数使用单一学习率
- 需要实现参数组特定的学习率

## 实现方法

### 使用 `optax.multi_transform`

实现将使用 `optax.multi_transform` 为不同的参数组创建独立的优化器。

### 步骤 1：创建 `create_multi_lr_optimizer()` 函数

**位置**: `src/openpi/training/optimizer.py`

```python
def create_multi_lr_optimizer(
    optimizer: OptimizerConfig,
    param_lr_schedules: dict[str, LRScheduleConfig],
    param_masks: dict[str, nnx.filterlib.Filter],
    weight_decay_mask: at.PyTree | None = None,
) -> optax.GradientTransformation:
    """创建具有不同参数组学习率的优化器。
    
    参数:
        optimizer: 优化器配置（例如 AdamW）
        param_lr_schedules: 参数组名称到学习率调度器的字典映射
                           例如: {"siglip": CosineDecaySchedule(peak_lr=1e-5), ...}
        param_masks: 参数组名称到过滤器的字典映射
                    例如: {"siglip": PathRegex(".*ValueGemma/img/.*"), ...}
        weight_decay_mask: 可选的权重衰减掩码
    
    返回:
        支持多学习率的 optax.GradientTransformation
    """
    import openpi.shared.nnx_utils as nnx_utils
    
    # 为每个参数组创建独立的优化器
    transforms = {}
    labels = {}
    
    for group_name, lr_schedule in param_lr_schedules.items():
        lr = lr_schedule.create()
        tx = optimizer.create(lr, weight_decay_mask=weight_decay_mask)
        transforms[group_name] = tx
        
        # 为此参数组创建掩码
        mask = param_masks[group_name]
        labels[group_name] = mask
    
    # 使用 optax.multi_transform 组合优化器
    return optax.multi_transform(transforms, labels)
```

### 步骤 2：更新 `train.py` 中的 `init_train_state()`

**位置**: `scripts/train.py`

修改优化器创建逻辑，当提供参数组调度器时使用多学习率优化器。

### 步骤 3：更新配置

**位置**: `src/openpi/training/config.py`

配置示例：

```python
# 参数组学习率调度器
param_lr_schedules = {
    "siglip": CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-5,  # SigLIP 学习率
        decay_steps=1_000_000,
        decay_lr=1e-5,
    ),
    "llm": CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,  # LLM 学习率（SigLIP 的 5 倍）
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    "value_head": CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=1e-4,  # Value head 学习率（比 SigLIP/LLM 高 10 倍）
        decay_steps=1_000_000,
        decay_lr=1e-4,
    ),
}

# 参数组掩码
param_masks = {
    "siglip": nnx_utils.PathRegex(".*ValueGemma/img/.*"),
    "llm": nnx_utils.PathRegex(".*ValueGemma/llm/.*"),
    "value_head": nnx_utils.PathRegex(".*value_head/.*"),
}
```

## 参数路径模式

基于 Value 模型结构：
- **SigLIP**: `ValueGemma/img/...`
- **LLM**: `ValueGemma/llm/...`
- **Value head**: `value_head/...`

## 注意事项

- 所有参数都已解冻（可训练）
- 不同的学习率允许以较低的学习率微调视觉编码器，同时以较高的学习率训练 LLM
- **Value head 学习率设置**：
  - 使用 **1e-4**（比预训练模型高 10 倍）
  - Value head 是一个较小的 3 层 MLP（640 → 512 → 128 → num_bins），参数量少
  - 较高的学习率有助于快速学习价值函数映射
  - 根据常见实现（如 DQN、Rainbow、C51），value head 通常使用 1e-4 到 3e-4 的学习率
  - 如果训练不稳定，可以降低到 5e-5；如果需要更快收敛，可以提高到 3e-4

## 测试

实现后，需要验证：
1. 不同参数组接收到正确的学习率
2. 每个组的优化器状态正确初始化
3. 训练过程无错误
4. 学习率在 wandb 中正确记录

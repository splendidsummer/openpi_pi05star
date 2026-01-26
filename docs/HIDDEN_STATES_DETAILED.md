# Hidden States 详细说明

## 什么是 Hidden States？

**Hidden States（隐藏状态）** 是 Transformer 模型经过所有 transformer blocks 处理后的**最终表示**。

## 输出层确认

### 最后一层：`final_norm` (RMSNorm)

根据 `_transformer.py` 的 `_apply_attention` 方法实现：

```python
def _apply_attention(self, inputs: _Inputs, cache: _config.Cache | None):
    x = inputs.embeddings  # 初始 embeddings
    
    # 遍历所有 transformer blocks（18 个）
    for i, block in enumerate(self.blocks):
        layer_cache, x = block(
            x,
            inputs.positions,
            old_cache.get(layer_name),
            inputs.attention_mask,
        )
    
    # ⭐ 最后一层：final_norm (RMSNorm)
    x = self.final_norm(x)  # 最终归一化层
    
    return x, new_cache  # x 就是 hidden_states
```

### 完整的数据流

```
输入: prefix_embeddings
Shape: [batch_size, seq_len, embed_dim]
例如: [2, 100, 640]
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 0                │
│   - Attention Layer                 │
│   - Feed Forward Network            │
│   - Residual Connections            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 1                │
│   - Attention Layer                 │
│   - Feed Forward Network            │
│   - Residual Connections            │
└─────────────────────────────────────┘
    ↓
    ... (共 18 个 blocks)
    ↓
┌─────────────────────────────────────┐
│ Transformer Block 17               │
│   - Attention Layer                 │
│   - Feed Forward Network            │
│   - Residual Connections            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ ⭐ final_norm (RMSNorm)            │ ← 最后一层
│   - 归一化处理                       │
│   - 稳定训练                         │
└─────────────────────────────────────┘
    ↓
输出: hidden_states
Shape: [batch_size, seq_len, embed_dim]
例如: [2, 100, 640]
```

## Hidden States 的特性

### 1. 形状
- **输入 embeddings**: `[batch_size, seq_len, embed_dim]`
- **输出 hidden_states**: `[batch_size, seq_len, embed_dim]`
- **形状保持不变**，但内容已经过处理

### 2. 内容变化
- **输入**: 每个 token 的独立表示（无上下文）
- **输出**: 每个 token 的上下文感知表示（包含整个序列的信息）

### 3. 最后一层的作用

**`final_norm` (RMSNorm)** 的作用：
- **归一化**: 将 hidden states 归一化到合适的范围
- **稳定性**: 提高训练和推理的数值稳定性
- **标准化**: 确保输出在一致的分布范围内

## 在 Value 模型中的使用

### 代码流程

```python
# 1. 输入 embeddings
prefix_embeddings, inputs_mask, ar_mask = self.embed_prefix(observation)
# Shape: [batch, seq_len, 640]

# 2. 通过所有 transformer blocks + final_norm
hidden_states, _ = self.ValueGemma.llm(
    tokens=prefix_embeddings,
    positions=positions,
    attention_mask=attn_mask,
    inputs_mask=inputs_mask,
    return_hidden_states=True,
    cache=None,
)
# hidden_states Shape: [batch, seq_len, 640]
# ⭐ 这是经过 18 个 blocks + final_norm 后的输出

# 3. 提取最后一个有效位置的 hidden state
last_valid_positions = jnp.sum(inputs_mask, axis=1, dtype=jnp.int32) - 1
last_timestep_output = hidden_states[batch_indices, last_valid_positions, :]
# Shape: [batch, 640]
# ⭐ 这是序列最后一个 token 的 hidden state（经过 final_norm）

# 4. 通过 value head 预测价值
value_logits = self.value_head(last_timestep_output)
# Shape: [batch, num_value_bins]
```

## 关键点总结

| 项目 | 说明 |
|------|------|
| **定义** | 经过所有 transformer blocks 和 final_norm 后的表示 |
| **形状** | `[batch_size, seq_len, embed_dim]` |
| **最后一层** | `final_norm` (RMSNorm) |
| **处理层数** | 18 个 transformer blocks + 1 个 final_norm |
| **特点** | 上下文感知、归一化、稳定 |
| **用途** | 作为序列的聚合表示，用于下游任务 |

## 验证

可以通过以下方式验证：

1. **形状检查**: `hidden_states.shape == (batch_size, seq_len, embed_dim)`
2. **值范围**: 经过 RMSNorm 后，值应该在合理范围内
3. **上下文信息**: 不同位置的 hidden states 应该包含不同的上下文信息

## 注意事项

- **Hidden States ≠ Logits**: Hidden states 是中间表示，logits 是经过 `embedder.decode()` 后的输出
- **最后一层是 final_norm**: 不是最后一个 transformer block，而是 final_norm
- **归一化很重要**: final_norm 确保输出的数值稳定性


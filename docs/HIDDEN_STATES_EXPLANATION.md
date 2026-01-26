# Hidden States 详解

## 什么是 Hidden States？

在 Transformer 模型中，**hidden_states**（隐藏状态）是经过所有 transformer blocks 处理后的**上下文感知表示**（contextualized representations）。

## 数据流转换过程

### 1. 输入阶段：Embeddings（嵌入向量）

```python
prefix_embeddings  # Shape: [batch_size, seq_len, embed_dim]
# 例如: [2, 100, 640]
```

- **含义**：初始的 token 嵌入向量
- **特点**：每个 token 的表示是**独立的**，没有考虑上下文信息
- **来源**：
  - 图像 tokens：来自 SigLIP 视觉编码器
  - 文本 tokens：来自 Gemma 的 embedding 层

### 2. Transformer Blocks 处理：Hidden States（隐藏状态）

```python
hidden_states, _ = self.ValueGemma.llm._apply_attention(inputs, cache=None)
# Shape: [batch_size, seq_len, embed_dim]
# 例如: [2, 100, 640]
```

**处理流程**：
```
prefix_embeddings (输入)
    ↓
[Transformer Block 0]  ← 注意力机制 + 前馈网络
    ↓
[Transformer Block 1]  ← 注意力机制 + 前馈网络
    ↓
    ...
    ↓
[Transformer Block 17] ← 注意力机制 + 前馈网络
    ↓
final_norm (RMSNorm)   ← 最终归一化
    ↓
hidden_states (输出)
```

**关键区别**：
- **Embeddings**：每个 token 的表示是**静态的**，只包含 token 本身的语义信息
- **Hidden States**：每个 token 的表示是**动态的**，包含了**整个序列的上下文信息**

### 3. 输出阶段：提取最后一个有效位置的表示

```python
last_valid_positions = jnp.sum(inputs_mask, axis=1, dtype=jnp.int32) - 1
last_timestep_output = hidden_states[batch_indices, last_valid_positions, :]
# Shape: [batch_size, embed_dim]
# 例如: [2, 640]
```

- **含义**：序列中最后一个有效 token 的 hidden state
- **用途**：作为整个序列的**聚合表示**，用于下游任务（如价值预测）

## 在 Value 模型中的使用

### 完整流程

```python
# 步骤 1: 获取 embeddings（图像 + 文本）
prefix_embeddings, inputs_mask, ar_mask = self.embed_prefix(observation)
# prefix_embeddings: [batch, seq_len, 640] - 初始嵌入

# 步骤 2: 通过所有 transformer blocks
hidden_states, _ = self.ValueGemma.llm._apply_attention(inputs, cache=None)
# hidden_states: [batch, seq_len, 640] - 上下文感知的表示

# 步骤 3: 提取最后一个有效位置的 hidden state
last_timestep_output = hidden_states[batch_indices, last_valid_positions, :]
# last_timestep_output: [batch, 640] - 序列的聚合表示

# 步骤 4: 通过 value head 预测价值
value_logits = self.value_head(last_timestep_output)
# value_logits: [batch, num_value_bins] - 价值分布
```

## 为什么使用 Hidden States？

### 1. **上下文信息**
- Hidden states 包含了每个 token 与序列中所有其他 token 的交互信息
- 这对于理解整个序列的语义至关重要

### 2. **位置信息**
- 通过注意力机制，hidden states 编码了 token 在序列中的位置关系
- 这对于理解序列的顺序和结构很重要

### 3. **聚合表示**
- 最后一个位置的 hidden state 可以看作是整个序列的**聚合表示**
- 它包含了从序列开始到结束的所有信息

## 技术细节

### `_apply_attention` 方法

```python
def _apply_attention(self, inputs: _Inputs, cache: _config.Cache | None):
    x = inputs.embeddings  # 初始 embeddings
    for i, block in enumerate(self.blocks):  # 18 个 transformer blocks
        x = block(x, positions, cache, attention_mask)
    x = self.final_norm(x)  # 最终归一化
    return x, new_cache  # 返回 hidden_states
```

### Hidden States vs Logits

- **Hidden States**：经过所有 transformer blocks 后的表示，形状为 `[batch, seq_len, embed_dim]`
- **Logits**：hidden states 经过 `embedder.decode()` 后的输出，形状为 `[batch, seq_len, vocab_size]`
  - 在 Value 模型中，我们**不需要 logits**，只需要 hidden states

## 总结

| 阶段 | 名称 | 形状 | 特点 |
|------|------|------|------|
| 输入 | Embeddings | `[batch, seq_len, embed_dim]` | 静态，无上下文 |
| 处理 | Hidden States | `[batch, seq_len, embed_dim]` | 动态，有上下文 |
| 输出 | Last Hidden State | `[batch, embed_dim]` | 序列聚合表示 |

**关键点**：
- Hidden states 是 transformer 的核心输出
- 它们包含了丰富的上下文信息
- 在 Value 模型中，我们使用最后一个位置的 hidden state 来预测价值


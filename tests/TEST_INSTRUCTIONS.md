# Pi05_STAR 测试指南

本指南提供逐步说明，用于测试 Pi05_STAR 模型对 TokenizeStarPrompt 输出的适配性。

## 测试概述

测试分为三个主要部分：

1. **基础测试**：验证代码修改无语法错误
2. **配置测试**：确认 TokenizeStarPrompt 变换正确配置
3. **端到端测试**：使用小数据集测试完整训练流程

## 环境设置

### 1. 设置缓存目录（避免填满主目录）
```bash
export HF_HOME=/root/autodl-tmp/huggingface
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
```

### 2. 内存限制（GPU测试时）
```bash
# 限制 GPU 内存使用
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# 如果只想在 CPU 上测试
export CUDA_VISIBLE_DEVICES=""
```


### 3. 安装依赖
确保已按照项目 README 安装所有依赖：
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 测试步骤

### 第 1 步：基础测试（语法验证）

运行基础测试以验证代码修改无语法错误：

```bash
cd /root/autodl-tmp/openpi_pi05star

# 方法 1：直接运行 Python 脚本
python test/test_pi05_star_basic.py
# 方法 1：直接运行 Python 脚本 without gpu 
CUDA_VISIBLE_DEVICES="" python test/test_pi05_star_basic.py

# 方法 2：使用 pytest
pytest test/test_pi05_star_basic.py -v

# 方法 3：运行所有测试文件
pytest test/ -v
```

**预期输出：**
- 所有测试通过（无断言错误）
- 无导入错误或语法错误
- 打印 "✅ All basic tests passed!"

### 第 2 步：配置测试（TokenizeStarPrompt 验证）

运行配置测试以确认 TokenizeStarPrompt 变换正确配置在训练流水线中：

```bash
# 运行配置测试
python test/test_tokenize_star_prompt_config.py

# 或使用 pytest
pytest test/test_tokenize_star_prompt_config.py -v
```

**测试内容：**
- ModelTransformFactory 为 Pi05_STAR 创建正确的变换
- TokenizeStarPrompt 变换生成预期的 6 个新字段
- 变换流水线集成测试
- 转换后的数据可成功转换为 Observation 对象

**预期输出：**
- 所有测试通过
- 打印 "✅ All TokenizeStarPrompt configuration tests passed!"

### 第 3 步：端到端测试（完整数据流）

运行端到端测试以验证完整的数据流：

```bash
# 运行端到端测试（部分测试可能被跳过）
python test/test_pi05_star_end_to_end.py

# 或使用 pytest（注意：某些测试被标记为跳过）
pytest test/test_pi05_star_end_to_end.py -v
```

**注意：** 端到端测试中的某些测试可能需要实际数据集或会被跳过。测试主要验证：
- 概念性数据流水线
- Pi05_STAR 训练步骤的冒烟测试
- 向后兼容性测试
- 内存和性能考虑事项文档

## 高级测试

### 使用真实数据集测试

如果需要使用真实数据集进行更全面的测试：

1. **设置小型测试数据集**
   ```bash
   # 创建或使用现有的小型数据集
   # 例如，使用 DROID 100 数据集的子集
   ```

2. **修改测试以使用真实数据**
   - 更新 `test_pi05_star_end_to_end.py` 中的 `create_mock_lerobot_dataset` 函数
   - 或注释掉 `pytest.skip()` 语句

3. **运行完整测试**
   ```bash
   # 设置更大的 GPU 内存（如果需要）
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

   # 运行测试
   pytest test/test_pi05_star_end_to_end.py::test_pi05_star_training_step_smoke_test -v
   ```

### 内存优化测试

对于内存受限的环境：

```bash
# 使用小批量大小
export TEST_BATCH_SIZE=1

# 使用 CPU 模式
export CUDA_VISIBLE_DEVICES=""
python test/test_pi05_star_basic.py
```

## 测试文件说明

### test_pi05_star_basic.py
- **目的**：验证基础代码修改
- **包含测试**：
  - `test_paligemma_star_tokenizer_bug_fixes()`: 验证 tokenizer bug 修复
  - `test_observation_extensions()`: 验证 Observation 类扩展
  - `test_pi05_star_model_instantiation()`: 验证模型实例化
  - `test_embed_prefix_uses_star_tokens()`: 验证模型使用正确的 tokenized 输入

### test_tokenize_star_prompt_config.py
- **目的**：验证训练配置
- **包含测试**：
  - `test_model_transform_factory_for_pi05_star()`: 验证 ModelTransformFactory
  - `test_tokenize_star_prompt_transform()`: 验证 TokenizeStarPrompt 变换
  - `test_transform_pipeline_integration()`: 验证变换流水线集成
  - `test_observation_from_transformed_data()`: 验证数据转换

### test_pi05_star_end_to_end.py
- **目的**：端到端验证
- **包含测试**：
  - `test_pi05_star_training_step_smoke_test()`: 训练步骤冒烟测试
  - `test_backward_compatibility()`: 向后兼容性测试
  - 概念性测试（需要实际数据集实现）

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'openpi'
   ```
   **解决方案**：确保在项目根目录运行测试，且已安装包 (`uv pip install -e .`)

2. **内存不足**
   ```
   XLA runtime error: RESOURCE_EXHAUSTED
   ```
   **解决方案**：
   - 减少批量大小
   - 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`
   - 在 CPU 上运行测试

3. **Tokenizer 下载失败**
   ```
   Failed to download tokenizer from gs://big_vision/paligemma_tokenizer.model
   ```
   **解决方案**：
   - 检查网络连接
   - 确保有适当的 Google Cloud 访问权限（或使用匿名令牌）
   - Tokenizer 可能已缓存，检查 `/root/autodl-tmp/huggingface` 目录

4. **FAST tokenizer 路径错误**
   ```
   Could not find tokenizer at path: physical-intelligence/fast
   ```
   **解决方案**：
   - 确保 `transformers` 库可访问该 tokenizer
   - 可能需要先从 HuggingFace 下载 tokenizer

### 调试技巧

1. **详细输出**
   ```bash
   pytest test/ -v -s
   ```

2. **单个测试调试**
   ```bash
   pytest test/test_pi05_star_basic.py::test_observation_extensions -v -s
   ```

3. **使用 pdb 调试**
   ```python
   import pdb; pdb.set_trace()
   ```

## 下一步

测试通过后，可以：

1. **运行实际训练**：
   ```bash
   # 计算归一化统计
   uv run scripts/compute_norm_stats.py --config-name pi05_star_droid

   # 运行训练
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_star_droid --exp-name=test_pi05_star
   ```

2. **验证模型输出**：
   - 检查训练损失是否正常下降
   - 验证生成的动作用期形状正确
   - 测试推理功能

3. **集成到现有工作流**：
   - 更新 CI/CD 流水线包含新测试
   - 添加模型性能基准测试
   - 文档化 Pi05_STAR 使用方式

## 联系支持

如果测试遇到问题，请参考：
- 项目 README 和 CLAUDE.md
- 现有测试文件中的模式
- 相关源代码注释

或创建 issue 描述具体问题。
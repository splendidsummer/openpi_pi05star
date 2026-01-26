#!/usr/bin/env python3
"""使用 gemma_utils 的示例

展示如何使用 build_gemma_3_270m_model 函数。
"""

from gemma_utils import build_gemma_3_270m_model
import jax.numpy as jnp


def main():
    """示例：构建模型并使用"""

    # 方式 1: 使用本地检查点路径
    print("方式 1: 使用本地检查点路径")
    try:
        # build_gemma_3_270m_model now returns a single ToNNX-wrapped model
        llm = build_gemma_3_270m_model(
            model_path="/root/autodl-tmp/gemma-3-270m"
        )
        print("✓ 模型加载成功")

        # 测试前向传播 (使用 token IDs)
        input_tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        output = llm(input_tokens)
        print(f"✓ 前向传播成功，输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")

    except Exception as e:
        print(f"✗ 失败: {e}")

    print("\n" + "="*70 + "\n")

    # 方式 2: 使用在线检查点（自动下载）
    print("方式 2: 使用在线检查点")
    try:
        # build_gemma_3_270m_model now returns a single ToNNX-wrapped model
        llm = build_gemma_3_270m_model(
            model_path="online"  # 或 model_path=None
        )
        print("✓ 模型加载成功")

        # 测试前向传播 (使用 token IDs)
        input_tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        output = llm(input_tokens)
        print(f"✓ 前向传播成功，输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")

    except Exception as e:
        print(f"✗ 失败: {e}")


if __name__ == "__main__":
    main()


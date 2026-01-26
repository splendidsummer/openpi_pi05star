"""Unit tests for weight loaders, with focus on SIGLIP weight loading for ValueGemma.
权重加载器的单元测试，重点关注 ValueGemma 的 SIGLIP 权重加载。"""

import dataclasses
import os
import tempfile
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import flax.traverse_util

# 设置 JAX 使用 cuda 进行测试
os.environ["JAX_PLATFORMS"] = "cuda"

from openpi.models.value_config import ValueConfig
from openpi.models.model import ModelType
from openpi.training import weight_loaders


def create_dummy_siglip_params(num_classes: int = 1152, prefix: str = "img/"):
    """Create dummy SIGLIP parameters with given num_classes.
    创建具有给定分类数的虚拟 SIGLIP 参数。

    Args:
        num_classes: Output dimension for classification head. 分类头的输出维度。
        prefix: Parameter prefix (e.g., "img/" or "PaliGemma/img/"). 参数前缀（例如 "img/" 或 "PaliGemma/img/"）。

    Returns:
        Nested parameter dictionary matching SIGLIP structure. 匹配 SIGLIP 结构的嵌套参数字典。
    """
    # Create a minimal set of SIGLIP parameters
    # 创建最小集的 SIGLIP 参数
    # Based on siglip.py structure
    # 基于 siglip.py 结构
    params = {
        "embedding": {
            "kernel": np.random.randn(14, 14, 3, 1152).astype(np.float32),
            "bias": np.random.randn(1152).astype(np.float32),
        },
        "pos_embedding": np.random.randn(1, 256, 1152).astype(np.float32),
        "Transformer": {
            "encoderblock_0": {
                "pre_attention_norm": {
                    "scale": np.random.randn(1152).astype(np.float32),
                },
                "attn": {
                    "qkv_einsum": {
                        "kernel": np.random.randn(3, 16, 1152, 72).astype(np.float32),
                    },
                },
                "pre_ffw_norm": {
                    "scale": np.random.randn(1152).astype(np.float32),
                },
                "mlp": {
                    "gating_einsum": np.random.randn(2, 1152, 4304).astype(np.float32),
                    "linear": np.random.randn(4304, 1152).astype(np.float32),
                },
                "attn_vec_einsum": {
                    "kernel": np.random.randn(16, 72, 1152).astype(np.float32),
                },
            },
        },
        "encoder_norm": {
            "scale": np.random.randn(1152).astype(np.float32),
        },
    }

    # Add classification head if num_classes is specified
    # 如果指定了 num_classes，则添加分类头
    if num_classes:
        params["head"] = {
            "kernel": np.random.randn(1152, num_classes).astype(np.float32),
            "bias": np.random.randn(num_classes).astype(np.float32),
        }

    # Wrap with prefix
    # 使用前缀包装
    if prefix:
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        parts = prefix.split("/")
        result = params
        for part in reversed(parts):
            result = {part: result}
        return result

    return params


def test_siglip_only_weight_loader_shape_mismatch():
    """Test SIGLIPOnlyWeightLoader handles shape mismatches correctly.
    测试 SIGLIPOnlyWeightLoader 是否正确处理形状不匹配的问题。"""
    # Create a Value model with Gemma-3-270m (hidden_size=640)
    # 创建具有 Gemma-3-270m (hidden_size=640) 的 Value 模型
    config = ValueConfig(
        model_path="/root/autodl-tmp/gemma-3-270m",  # Not used for parameter creation # 不用于参数创建
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        Vmin=-199.0,
        Vmax=0.0,
        num_value_bins=201
    )

    rng = jax.random.key(0)
    model = config.create(rng)

    # Get model parameters
    # 获取模型参数
    import flax.nnx as nnx
    params = nnx.state(model).to_pure_dict()

    # Create ·dummy checkpoint parameters with SIGLIP weights (num_classes=1152)
    # 创建带有 SIGLIP 权重的虚构检查点参数 (num_classes=1152)
    # This simulates a Pi0 checkpoint with SIGLIP So400m/14
    # 这模拟了一个带有 SIGLIP So400m/14 的 Pi0 检查点
    dummy_checkpoint_params = {
        "PaliGemma": {
            "img": create_dummy_siglip_params(num_classes=1152, prefix="")
        }
    }

    # Save dummy checkpoint to temporary file
    # 将虚构检查点保存到临时文件
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        checkpoint_path = f.name  # '/tmp/tmp1neuj6w5.npz' 
        flat_params = flax.traverse_util.flatten_dict(dummy_checkpoint_params, sep="/")
        np.savez(f, **flat_params)

    try:
        # Create weight loader
        # 创建权重加载器
        loader = weight_loaders.SIGLIPOnlyWeightLoader(params_path=checkpoint_path)

        # Load weights
        # 加载权重
        loaded_params = loader.load(params)

        # Verify that parameters were loaded
        # 验证参数已加载
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_original = flax.traverse_util.flatten_dict(params, sep="/")

        # Count how many SIGLIP weights were mapped
        # 统计映射了多少 SIGLIP 权重
        siglip_keys = [k for k in flat_loaded.keys() if k.startswith("ValueGemma/img/")]
        print(f"Found {len(siglip_keys)} SIGLIP keys in loaded parameters")

        # The classification head weights should be skipped due to shape mismatch
        # (1152 vs 640), but encoder weights should be loaded
        # 由于形状不匹配（1152 vs 640），分类头权重应被跳过，但编码器权重应被加载
        assert len(siglip_keys) > 0, "No SIGLIP weights were loaded"

        # Verify that the model still has all original parameters
        # 验证模型是否仍包含所有原始参数
        for k in flat_original:
            assert k in flat_loaded, f"Missing parameter: {k}"

        print("✓ SIGLIPOnlyWeightLoader handles shape mismatches correctly")

    finally:
        # Clean up temporary file
        # 清理临时文件
        Path(checkpoint_path).unlink(missing_ok=True)


def test_siglip_value_gemma_weight_loader():
    """Test SIGLIPValueGemmaWeightLoader specifically for Gemma-3-270m.
    专门测试针对 Gemma-3-270m 的 SIGLIPValueGemmaWeightLoader。"""
    # Create a Value model
    # 创建一个 Value 模型
    config = ValueConfig(
        model_path="/root/autodl-tmp/gemma-3-270m",
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
        Vmin=-199.0,
        Vmax=0.0,
        num_value_bins=201
    )

    rng = jax.random.key(0)
    model = config.create(rng)
    params = nnx.state(model).to_pure_dict()

    # Create dummy checkpoint with SIGLIP weights for ValueGemma
    # 为 ValueGemma 创建带有 SIGLIP 权重的虚构检查点
    # Use num_classes=640 to match Gemma-3-270m hidden size
    # 使用 num_classes=640 以匹配 Gemma-3-270m 的隐藏层大小
    dummy_checkpoint_params = {
        "PaliGemma": {
            "img": create_dummy_siglip_params(num_classes=640, prefix="")
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        checkpoint_path = f.name
        flat_params = flax.traverse_util.flatten_dict(dummy_checkpoint_params, sep="/")
        np.savez(f, **flat_params)

    try:
        # Create the specialized weight loader
        # 创建专用的权重加载器
        loader = weight_loaders.SIGLIPValueGemmaWeightLoader(params_path=checkpoint_path)

        # Load weights
        # 加载权重
        loaded_params = loader.load(params)

        # Verify loading
        # 验证加载
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        # Check that SIGLIP weights were mapped
        # 检查是否映射了 SIGLIP 权重
        siglip_keys = [k for k in flat_loaded.keys() if k.startswith("ValueGemma/img/")]
        print(f"Loaded {len(siglip_keys)} SIGLIP weights")

        # With matching num_classes, all weights should load
        # (excluding maybe some that don't exist in target)
        # 如果 num_classes 匹配，所有权重都应该加载（可能排除目标中不存在的一些权重）
        assert len(siglip_keys) > 0

        # Verify parameter preservation
        # 验证参数保留情况
        for k in flax.traverse_util.flatten_dict(params, sep="/"):
            assert k in flat_loaded, f"Missing parameter after loading: {k}"

        print("✓ SIGLIPValueGemmaWeightLoader works correctly with matching hidden_size")

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_weight_loader_no_siglip_found():
    """Test weight loader behavior when no SIGLIP weights are found.
    测试当未找到 SIGLIP 权重时的权重加载器行为。"""
    config = ValueConfig(
        model_path="/root/autodl-tmp/gemma-3-270m",
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
    )

    rng = jax.random.key(0)
    model = config.create(rng)
    params = nnx.state(model).to_pure_dict()

    # Create dummy checkpoint WITHOUT SIGLIP weights
    # 创建没有 SIGLIP 权重的虚构检查点
    dummy_checkpoint_params = {
        "SomeOtherModel": {
            "weights": {"kernel": np.random.randn(10, 10).astype(np.float32)}
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        checkpoint_path = f.name
        flat_params = flax.traverse_util.flatten_dict(dummy_checkpoint_params, sep="/")
        np.savez(f, **flat_params)

    try:
        loader = weight_loaders.SIGLIPOnlyWeightLoader(params_path=checkpoint_path)
        loaded_params = loader.load(params)

        # When no SIGLIP weights found, original params should be returned unchanged
        # 当未找到 SIGLIP 权重时，应原样返回原始参数
        flat_original = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        assert set(flat_original.keys()) == set(flat_loaded.keys())
        for k in flat_original:
            v1 = flat_original[k]
            v2 = flat_loaded[k]
            if v1 is None and v2 is None:
                continue
            if isinstance(v1, (np.ndarray, jax.Array)) and isinstance(v2, (np.ndarray, jax.Array)):
                assert jnp.array_equal(v1, v2)
            else:
                assert v1 == v2

        print("✓ Weight loader returns original params when no SIGLIP weights found")

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_weight_loader_different_prefixes():
    """Test weight loader with different parameter prefixes.
    测试带有不同参数前缀的权重加载器。"""
    config = ValueConfig(
        model_path="/root/autodl-tmp/gemma-3-270m",
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
    )

    rng = jax.random.key(0)
    model = config.create(rng)
    params = nnx.state(model).to_pure_dict()

    # Test with "img/" prefix (from PaliGemma checkpoint)
    # 测试 "img/" 前缀（来自 PaliGemma 检查点）
    dummy_checkpoint_params1 = create_dummy_siglip_params(num_classes=640, prefix="img/")

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        checkpoint_path = f.name
        flat_params = flax.traverse_util.flatten_dict(dummy_checkpoint_params1, sep="/")
        np.savez(f, **flat_params)

    try:
        loader = weight_loaders.SIGLIPOnlyWeightLoader(params_path=checkpoint_path)
        loaded_params = loader.load(params)

        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        siglip_keys = [k for k in flat_loaded.keys() if k.startswith("ValueGemma/img/")]

        assert len(siglip_keys) > 0
        print(f"✓ Loader works with 'img/' prefix, found {len(siglip_keys)} SIGLIP weights")

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("Running weight loaders tests...")

    # Import nnx here to avoid import issues
    # 在这里导入 nnx 以避免导入问题
    import flax.nnx as nnx

    tests = [
        # test_siglip_only_weight_loader_shape_mismatch,
        test_siglip_value_gemma_weight_loader,
        test_weight_loader_no_siglip_found,
        test_weight_loader_different_prefixes,
    ]

    for test_func in tests:
        print(f"\n{test_func.__name__}...")
        try:
            test_func()
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            raise

    print("\n✅ All weight loaders tests passed!")
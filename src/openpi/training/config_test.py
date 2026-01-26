"""Unit tests for training configurations, with focus on pi05_droid_100_value config."""

import dataclasses
import os
import pathlib

import pytest
import jax
import jax.numpy as jnp

# Set JAX to CPU for testing
os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config
from openpi.training.data_loader import create_data_loader
from openpi.models.value_config import ValueConfig
from openpi.models.model import ModelType


@pytest.mark.manual
def test_pi05_droid_100_value_config_exists():
    """Test that pi05_droid_100_value config exists and has correct properties."""
    config = _config.get_config("pi05_droid_100_value")

    assert config.name == "pi05_droid_100_value"
    assert config.model.model_type == ModelType.VALUE
    assert isinstance(config.model, ValueConfig)

    # Check specific config values from config.py lines 1008-1033
    assert config.batch_size == 2
    assert config.num_train_steps == 20000
    assert config.weight_loader.__class__.__name__ == "NoOpWeightLoader"

    # Check model config
    assert config.model.pi05 is True
    assert config.model.Vmin == -199.0
    assert config.model.Vmax == 0.0
    assert config.model.num_value_bins == 201
    assert config.model.model_path == "/root/autodl-tmp/gemma-3-270m"

    print("✓ pi05_droid_100_value config exists with correct properties")


@pytest.mark.manual
def test_value_config_creation():
    """Test ValueConfig creation and validation."""
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

    assert config.model_type == ModelType.VALUE
    assert config.discrete_state_input is True  # Should be set from pi05=True
    assert config.max_token_len == 200

    # Test model creation
    rng = jax.random.key(0)
    model = config.create(rng)
    assert model.__class__.__name__ == "Value"
    assert model.num_value_bins == 201
    assert model.Vmin == -199.0
    assert model.Vmax == 0.0

    print("✓ ValueConfig creates model correctly")


@pytest.mark.manual
def test_config_data_loading():
    """Test that config can create data loader for pi05_droid_100_value."""
    config = _config.get_config("pi05_droid_100_value")

    # Create a temporary config with fake data for testing
    test_config = dataclasses.replace(
        config,
        data=_config.FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="test_data_loading"
    )

    # Test data loader creation
    data_loader = create_data_loader(test_config, shuffle=False, num_batches=2)

    batch_count = 0
    for batch in data_loader:
        batch_count += 1
        # Should yield (observation, actions, value_targets) tuple
        assert len(batch) == 3, f"Expected 3-tuple, got {len(batch)}-tuple"

        observation, actions, value_targets = batch

        # Check shapes
        assert actions.shape[0] == test_config.batch_size
        assert value_targets.shape[0] == test_config.batch_size
        assert value_targets.ndim == 1  # Value targets should be 1D

        if batch_count >= 2:
            break

    assert batch_count > 0, "No batches loaded"
    print(f"✓ Data loader created successfully, loaded {batch_count} batches")


@pytest.mark.manual
def test_config_model_initialization():
    """Test that config can initialize model with correct shapes."""
    config = _config.get_config("pi05_droid_100_value")

    # Use fake data config for testing
    test_config = dataclasses.replace(
        config,
        data=_config.FakeDataConfig(),
        batch_size=2
    )

    rng = jax.random.key(42)
    model = test_config.model.create(rng)

    # Get input specs
    obs_spec, actions_spec = test_config.model.inputs_spec(batch_size=2)

    # Check observation spec
    assert "images" in obs_spec._fields
    assert "state" in obs_spec._fields
    assert "tokenized_prompt" in obs_spec._fields

    # Check action spec
    assert actions_spec.shape == (2, test_config.model.action_horizon, test_config.model.action_dim)

    print("✓ Model initialization with config works correctly")


@pytest.mark.manual
def test_value_targets_range():
    """Test that value targets have expected range for DROID dataset."""
    config = _config.get_config("pi05_droid_100_value")

    # Note: This test requires actual dataset access
    # For now, just verify config has value-related settings
    assert hasattr(config.model, 'Vmin')
    assert hasattr(config.model, 'Vmax')
    assert config.model.Vmin == -199.0
    assert config.model.Vmax == 0.0

    # Check that num_value_bins is reasonable
    assert config.model.num_value_bins > 1
    assert config.model.num_value_bins == 201  # As defined in config

    print("✓ Value targets range configured correctly")


@pytest.mark.manual
def test_config_serialization():
    """Test that config can be serialized and deserialized."""
    import pickle

    config = _config.get_config("pi05_droid_100_value")

    # Test pickling
    pickled = pickle.dumps(config)
    unpickled = pickle.loads(pickled)

    assert unpickled.name == config.name
    assert unpickled.model.model_type == config.model.model_type
    assert unpickled.batch_size == config.batch_size

    print("✓ Config serialization works correctly")


@pytest.mark.manual
def test_all_configs_unique():
    """Test that all config names are unique (sanity check)."""
    names = [c.name for c in _config._CONFIGS]
    assert len(names) == len(set(names)), "Config names are not unique"

    # Specifically check our config
    assert "pi05_droid_100_value" in names

    print("✓ All config names are unique")


@pytest.mark.manual
def test_config_assets():
    """Test that config assets directory is accessible."""
    config = _config.get_config("pi05_droid_100_value")

    assets_dir = config.assets_dirs
    assert isinstance(assets_dir, pathlib.Path)

    # Check that assets_dir is a valid path (may not exist yet)
    print(f"Assets directory: {assets_dir}")

    print("✓ Config assets directory accessible")


if __name__ == "__main__":
    print("Running pi05_droid_100_value config tests...")

    tests = [
        test_pi05_droid_100_value_config_exists,
        test_value_config_creation,
        test_config_data_loading,
        test_config_model_initialization,
        test_value_targets_range,
        test_config_serialization,
        test_all_configs_unique,
        test_config_assets,
    ]

    for test_func in tests:
        print(f"\n{test_func.__name__}...")
        try:
            test_func()
            print("✓ PASSED")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            raise

    print("\n✅ All pi05_droid_100_value config tests passed!")
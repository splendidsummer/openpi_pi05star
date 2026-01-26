import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


# ============================================================================
# Value Training Specific Tests
# ============================================================================

import pytest
import jax.numpy as jnp
import numpy as np


@pytest.mark.manual
def test_value_data_loader_yields_three_tuple():
    """Test that data loader yields 3-tuple for value training"""
    config = _config.get_config("pi05_droid_100_value")
    loader = _data_loader.create_data_loader(config, shuffle=False, num_batches=1)

    batch = next(iter(loader))

    assert len(batch) == 3, f"Expected 3-tuple, got {len(batch)}-tuple"
    observation, actions, value_targets = batch

    print("✓ Data loader yields 3-tuple (observation, actions, value_targets)")
    print(f"  Observation type: {type(observation).__name__}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Value targets shape: {value_targets.shape}")


@pytest.mark.manual
def test_value_targets_shape_and_type():
    """Test that value_targets have correct shape and data type"""
    config = _config.get_config("pi05_droid_100_value")
    loader = _data_loader.create_data_loader(config, shuffle=False, num_batches=1)

    observation, actions, value_targets = next(iter(loader))

    # Get batch size
    batch_size = observation.state.shape[0]

    # Check shape: should be 1D with batch_size elements
    assert len(value_targets.shape) == 1, f"Expected 1D, got shape {value_targets.shape}"
    assert value_targets.shape[0] == batch_size, (
        f"Batch size mismatch: value_targets={value_targets.shape[0]}, state={batch_size}"
    )

    # Check data type: should be float32
    assert value_targets.dtype in [jnp.float32, np.float32], (
        f"Expected float32, got {value_targets.dtype}"
    )

    print("✓ value_targets have correct shape and type")
    print(f"  Shape: {value_targets.shape}")
    print(f"  Dtype: {value_targets.dtype}")


@pytest.mark.manual
def test_value_targets_are_finite():
    """Test that value_targets contain no NaN or Inf values"""
    config = _config.get_config("pi05_droid_100_value")
    loader = _data_loader.create_data_loader(config, shuffle=False, num_batches=10)

    all_values = []
    for observation, actions, value_targets in loader:
        all_values.extend(value_targets.tolist())

    all_values = np.array(all_values)
    assert np.all(np.isfinite(all_values)), "value_targets contain NaN or Inf"

    print("✓ value_targets are all finite")
    print(f"  Min: {np.min(all_values):.2f}")
    print(f"  Max: {np.max(all_values):.2f}")
    print(f"  Mean: {np.mean(all_values):.2f}")


@pytest.mark.manual
def test_value_batch_consistency():
    """Test that batch dimensions are consistent across observation, actions, and value_targets"""
    config = _config.get_config("pi05_droid_100_value")
    loader = _data_loader.create_data_loader(config, shuffle=False, num_batches=1)

    observation, actions, value_targets = next(iter(loader))

    # Get batch sizes from different sources
    batch_sizes = {
        "state": observation.state.shape[0],
        "actions": actions.shape[0],
        "value_targets": value_targets.shape[0],
    }

    # Check all batch sizes are the same
    unique_batch_sizes = set(batch_sizes.values())
    assert len(unique_batch_sizes) == 1, f"Inconsistent batch sizes: {batch_sizes}"

    batch_size = list(unique_batch_sizes)[0]
    print("✓ Batch dimensions are consistent")
    print(f"  Batch size: {batch_size}")
    print(f"  All fields have matching batch dimension")

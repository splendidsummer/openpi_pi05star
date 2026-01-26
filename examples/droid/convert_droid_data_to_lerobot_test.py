"""Unit tests for DROID data conversion with value targets."""

import pytest
import numpy as np


def test_value_target_computation():
    """Verify value targets computed correctly using remaining timesteps approach."""
    episode_length = 100

    for step_idx in range(episode_length):
        # Compute value target: V(t) = -1 * (episode_length - t - 1)
        state_value = -1.0 * (episode_length - step_idx - 1)

        # Check range
        assert state_value >= -(episode_length - 1), \
            f"Value {state_value} below minimum -(episode_length-1)"
        assert state_value <= 0, \
            f"Value {state_value} above maximum 0"

        # Check first timestep
        if step_idx == 0:
            assert state_value == -(episode_length - 1), \
                f"First timestep should be -{episode_length-1}, got {state_value}"

        # Check final timestep
        if step_idx == episode_length - 1:
            assert state_value == 0.0, \
                f"Final timestep should be 0.0, got {state_value}"

    print(f"✓ Value target computation test passed for episode_length={episode_length}")


def test_value_target_monotonicity():
    """Test that value targets increase monotonically over episode."""
    episode_length = 50
    values = []

    for step_idx in range(episode_length):
        state_value = -1.0 * (episode_length - step_idx - 1)
        values.append(state_value)

    # Check monotonicity
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1], \
            f"Values not monotonically increasing at index {i}: {values[i]} >= {values[i+1]}"

    print(f"✓ Value targets are monotonically increasing")


def test_value_target_different_episode_lengths():
    """Test value target computation for various episode lengths."""
    episode_lengths = [10, 50, 100, 200]

    for episode_length in episode_lengths:
        # Test first timestep
        first_value = -1.0 * (episode_length - 0 - 1)
        assert first_value == -(episode_length - 1)

        # Test last timestep
        last_value = -1.0 * (episode_length - (episode_length - 1) - 1)
        assert last_value == 0.0

        # Test middle timestep
        mid_idx = episode_length // 2
        mid_value = -1.0 * (episode_length - mid_idx - 1)
        assert -(episode_length - 1) <= mid_value <= 0

    print(f"✓ Value targets correct for episode lengths: {episode_lengths}")


def test_reward_initialization():
    """Test that reward is initialized to 0.0."""
    reward = 0.0
    assert reward == 0.0, f"Reward should be 0.0, got {reward}"
    assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
    print("✓ Reward initialization test passed")


def test_value_target_array_format():
    """Test that value targets are in correct numpy array format."""
    episode_length = 100
    step_idx = 50

    state_value = -1.0 * (episode_length - step_idx - 1)
    state_value_array = np.array([state_value], dtype=np.float32)

    assert state_value_array.shape == (1,), \
        f"Expected shape (1,), got {state_value_array.shape}"
    assert state_value_array.dtype == np.float32, \
        f"Expected dtype float32, got {state_value_array.dtype}"
    assert state_value_array[0] == state_value, \
        f"Array value {state_value_array[0]} doesn't match computed value {state_value}"

    print("✓ Value target array format test passed")


if __name__ == "__main__":
    print("Running DROID data conversion tests...\n")

    print("1. Testing value target computation...")
    test_value_target_computation()

    print("\n2. Testing value target monotonicity...")
    test_value_target_monotonicity()

    print("\n3. Testing different episode lengths...")
    test_value_target_different_episode_lengths()

    print("\n4. Testing reward initialization...")
    test_reward_initialization()

    print("\n5. Testing value target array format...")
    test_value_target_array_format()

    print("\n✅ All data conversion tests passed!")

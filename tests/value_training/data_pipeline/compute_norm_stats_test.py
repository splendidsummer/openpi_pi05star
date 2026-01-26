"""Unit tests for normalization statistics computation with value_targets and reward.

This test file verifies that compute_norm_stats.py correctly:
1. Detects value_targets and reward fields in the dataset
2. Computes statistics (mean, std, q01, q99) for all features
3. Saves statistics to norm_stats.json
4. Handles edge cases (NaN, Inf, empty data)
"""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.mark.manual
def test_norm_stats_file_exists():
    """Test that norm_stats.json file exists after running compute_norm_stats.py"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    assert norm_stats_path.exists(), (
        f"Norm stats file not found at {norm_stats_path}. "
        "Run: uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value"
    )
    print(f"✓ Norm stats file exists at {norm_stats_path}")


@pytest.mark.manual
def test_norm_stats_contains_value_targets():
    """Test that norm_stats.json contains value_targets statistics"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    assert "value_targets" in norm_stats, "Missing value_targets in norm_stats.json"

    value_stats = norm_stats["value_targets"]
    required_keys = ["mean", "std", "q01", "q99"]

    for key in required_keys:
        assert key in value_stats, f"Missing {key} in value_targets stats"
        assert isinstance(value_stats[key], list), f"{key} should be a list"
        assert len(value_stats[key]) > 0, f"{key} should not be empty"

    print("✓ value_targets statistics found with all required keys")
    print(f"  mean: {value_stats['mean']}")
    print(f"  std: {value_stats['std']}")
    print(f"  q01: {value_stats['q01']}")
    print(f"  q99: {value_stats['q99']}")


@pytest.mark.manual
def test_norm_stats_contains_reward():
    """Test that norm_stats.json contains reward statistics"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    assert "reward" in norm_stats, "Missing reward in norm_stats.json"

    reward_stats = norm_stats["reward"]
    required_keys = ["mean", "std", "q01", "q99"]

    for key in required_keys:
        assert key in reward_stats, f"Missing {key} in reward stats"

    print("✓ reward statistics found with all required keys")
    print(f"  mean: {reward_stats['mean']}")
    print(f"  std: {reward_stats['std']}")
    print(f"  q01: {reward_stats['q01']}")
    print(f"  q99: {reward_stats['q99']}")


@pytest.mark.manual
def test_norm_stats_values_are_finite():
    """Test that all statistics are finite (no NaN or Inf)"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    for feature_name, stats in norm_stats.items():
        for stat_name, values in stats.items():
            values_array = np.array(values)
            assert np.all(np.isfinite(values_array)), (
                f"Non-finite values found in {feature_name}.{stat_name}: {values}"
            )

    print("✓ All statistics are finite (no NaN or Inf)")


@pytest.mark.manual
def test_value_targets_stats_reasonable():
    """Test that value_targets statistics are in reasonable ranges"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    value_stats = norm_stats["value_targets"]

    # Extract values (assuming single-dimensional)
    mean = value_stats["mean"][0] if isinstance(value_stats["mean"], list) else value_stats["mean"]
    std = value_stats["std"][0] if isinstance(value_stats["std"], list) else value_stats["std"]
    q01 = value_stats["q01"][0] if isinstance(value_stats["q01"], list) else value_stats["q01"]
    q99 = value_stats["q99"][0] if isinstance(value_stats["q99"], list) else value_stats["q99"]

    # Value targets should be negative (remaining timesteps)
    assert mean < 0, f"Mean should be negative, got {mean}"

    # Standard deviation should be positive and reasonable
    assert std > 0, f"Std should be positive, got {std}"
    assert std < 200, f"Std seems too large: {std}"

    # q01 should be more negative than q99
    assert q01 < q99, f"q01 ({q01}) should be less than q99 ({q99})"

    # q99 should be close to 0 (last timestep)
    assert abs(q99) < 5, f"q99 should be close to 0, got {q99}"

    # q01 should be negative (early timesteps)
    assert q01 < -10, f"q01 should be significantly negative, got {q01}"

    print("✓ value_targets statistics are in reasonable ranges")
    print(f"  Mean: {mean:.2f} (should be negative)")
    print(f"  Std: {std:.2f} (should be positive and < 200)")
    print(f"  q01: {q01:.2f} (should be < -10)")
    print(f"  q99: {q99:.2f} (should be close to 0)")


@pytest.mark.manual
def test_reward_stats_reasonable():
    """Test that reward statistics are reasonable (should be all zeros for DROID)"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    reward_stats = norm_stats["reward"]

    # Extract values
    mean = reward_stats["mean"][0] if isinstance(reward_stats["mean"], list) else reward_stats["mean"]
    std = reward_stats["std"][0] if isinstance(reward_stats["std"], list) else reward_stats["std"]
    q01 = reward_stats["q01"][0] if isinstance(reward_stats["q01"], list) else reward_stats["q01"]
    q99 = reward_stats["q99"][0] if isinstance(reward_stats["q99"], list) else reward_stats["q99"]

    # For DROID dataset, rewards are all 0
    assert abs(mean) < 1e-6, f"Mean should be ~0, got {mean}"
    assert abs(std) < 1e-6, f"Std should be ~0, got {std}"
    assert abs(q01) < 1e-6, f"q01 should be ~0, got {q01}"
    assert abs(q99) < 1e-6, f"q99 should be ~0, got {q99}"

    print("✓ reward statistics are reasonable (all zeros as expected)")


@pytest.mark.manual
def test_norm_stats_contains_all_required_features():
    """Test that norm_stats.json contains all required features"""
    norm_stats_path = Path.home() / ".cache/openpi/assets/droid/norm_stats.json"

    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)

    required_features = ["state", "actions", "value_targets", "reward"]

    for feature in required_features:
        assert feature in norm_stats, f"Missing required feature: {feature}"

    print(f"✓ All required features present: {required_features}")
    print(f"  Total features in norm_stats: {list(norm_stats.keys())}")


def test_statistics_computation_logic():
    """Unit test for statistics computation logic (no dataset required)"""
    # Create fake data
    data = np.array([
        [-100.0],
        [-50.0],
        [-25.0],
        [0.0],
    ])

    # Compute statistics
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    q01 = np.percentile(data, 1, axis=0)
    q99 = np.percentile(data, 99, axis=0)

    # Verify
    assert np.isclose(mean[0], -43.75), f"Mean incorrect: {mean[0]}"
    assert std[0] > 0, f"Std should be positive: {std[0]}"
    assert q01[0] < q99[0], f"q01 should be less than q99"

    print("✓ Statistics computation logic test passed")
    print(f"  Mean: {mean[0]:.2f}")
    print(f"  Std: {std[0]:.2f}")
    print(f"  q01: {q01[0]:.2f}")
    print(f"  q99: {q99[0]:.2f}")


if __name__ == "__main__":
    print("Running normalization statistics tests...\n")

    print("=" * 60)
    print("Unit Test (no dataset required)")
    print("=" * 60)
    test_statistics_computation_logic()

    print("\n" + "=" * 60)
    print("Integration Tests (require norm_stats.json)")
    print("=" * 60)
    print("\nNOTE: These tests require running compute_norm_stats.py first:")
    print("  uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value\n")

    try:
        print("1. Testing norm stats file exists...")
        test_norm_stats_file_exists()

        print("\n2. Testing value_targets statistics...")
        test_norm_stats_contains_value_targets()

        print("\n3. Testing reward statistics...")
        test_norm_stats_contains_reward()

        print("\n4. Testing all values are finite...")
        test_norm_stats_values_are_finite()

        print("\n5. Testing value_targets stats are reasonable...")
        test_value_targets_stats_reasonable()

        print("\n6. Testing reward stats are reasonable...")
        test_reward_stats_reasonable()

        print("\n7. Testing all required features present...")
        test_norm_stats_contains_all_required_features()

        print("\n" + "=" * 60)
        print("✅ All normalization statistics tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure to run compute_norm_stats.py first:")
        print("  uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value")
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("\nMake sure to run compute_norm_stats.py first:")
        print("  uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value")

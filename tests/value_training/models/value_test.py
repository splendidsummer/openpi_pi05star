"""Unit tests for Value model."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.value import Value, DistributionalValueHeadGemma3
from openpi.models.value_config import ValueConfig
from openpi.models.model import Observation


@pytest.mark.manual
def test_value_model_initialization():
    """Test Value model initializes correctly with Gemma-3-270M."""
    config = ValueConfig(
        model_path="/root/autodl-tmp/gemma-3-270m",
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=200,
    )
    rng = jax.random.key(0)
    model = config.create(rng)

    assert isinstance(model, Value)
    assert model.num_value_bins == 201
    assert model.Vmin == -199.0
    assert model.Vmax == 0.0
    assert model.valuegemma_width == 640  # gemma_3_270m width
    print("✓ Value model initialized successfully")


@pytest.mark.manual
def test_value_forward_pass():
    """Test forward pass produces correct output shapes."""
    config = ValueConfig(model_path="/root/autodl-tmp/gemma-3-270m")
    model = config.create(jax.random.key(0))

    # Create fake observation
    obs = config.fake_obs(batch_size=2)
    value_targets = jnp.array([-50.0, -100.0])

    # Forward pass
    rng = jax.random.key(1)
    loss = model.compute_loss(rng, obs, train=True, value_targets=value_targets)

    assert loss.shape == (2,), f"Expected loss shape (2,), got {loss.shape}"
    assert jnp.all(jnp.isfinite(loss)), "Loss contains NaN or Inf values"
    print(f"✓ Forward pass successful, loss: {loss}")


@pytest.mark.manual
def test_distributional_value_head():
    """Test value head produces valid probability distributions."""
    head = DistributionalValueHeadGemma3(
        hidden_size=640,
        num_bins=201,
        v_min=-199.0,
        v_max=0.0,
        rngs=jax.random.key(0)
    )

    # Test forward pass
    x = jnp.ones((4, 640))  # batch_size=4, hidden_size=640
    value = head(x, return_expectation=True)

    assert value.shape == (4,), f"Expected value shape (4,), got {value.shape}"
    assert jnp.all(value >= -199.0), f"Values below Vmin: {jnp.min(value)}"
    assert jnp.all(value <= 0.0), f"Values above Vmax: {jnp.max(value)}"
    print(f"✓ Value head test passed, values: {value}")


@pytest.mark.manual
def test_value_head_support_atoms():
    """Test value head support atoms are correctly initialized."""
    head = DistributionalValueHeadGemma3(
        hidden_size=640,
        num_bins=201,
        v_min=-199.0,
        v_max=0.0,
        rngs=jax.random.key(0)
    )

    assert head.support.shape == (201,), f"Expected support shape (201,), got {head.support.shape}"
    assert jnp.isclose(head.support[0], -199.0), f"First support atom should be -199.0, got {head.support[0]}"
    assert jnp.isclose(head.support[-1], 0.0), f"Last support atom should be 0.0, got {head.support[-1]}"
    print(f"✓ Support atoms correctly initialized: [{head.support[0]:.2f}, ..., {head.support[-1]:.2f}]")


if __name__ == "__main__":
    print("Running Value model tests...")
    print("\n1. Testing model initialization...")
    test_value_model_initialization()

    print("\n2. Testing forward pass...")
    test_value_forward_pass()

    print("\n3. Testing distributional value head...")
    test_distributional_value_head()

    print("\n4. Testing value head support atoms...")
    test_value_head_support_atoms()

    print("\n✅ All tests passed!")

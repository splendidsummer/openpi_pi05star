#!/usr/bin/env python3
"""Debug script to inspect Gemma model state structure."""

import sys
sys.path.insert(0, '/root/autodl-tmp/openpi/src')

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.models.gemma_utils import build_gemma_3_270m_model

# Build Gemma model
llm = build_gemma_3_270m_model("/root/autodl-tmp/gemma-3-270m")

# Initialize with dummy tokens
dummy_tokens = jnp.ones((1, 10), dtype=jnp.int32)
rngs = nnx.Rngs(0)
llm.lazy_init(dummy_tokens, rngs=rngs)

# Get the state
llm_state = nnx.state(llm)

# Print the state structure
def print_state_structure(state_dict, prefix="", max_depth=3, current_depth=0):
    """Recursively print the state structure."""
    if current_depth >= max_depth:
        return

    for key, value in state_dict.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}/")
            print_state_structure(value, prefix + "  ", max_depth, current_depth + 1)
        elif hasattr(value, 'value') and hasattr(value.value, 'shape'):
            print(f"{prefix}{key}: {value.value.shape} {value.value.dtype}")
        else:
            print(f"{prefix}{key}: {type(value)}")

print("Gemma model state structure:")
print_state_structure(llm_state)

print("\n\nEmbedder state structure:")
if 'embedder' in llm_state:
    embedder_state = llm_state['embedder']
    print(f"Type: {type(embedder_state)}")
    if hasattr(embedder_state, '__dict__'):
        print(f"Attributes: {list(embedder_state.__dict__.keys())}")

    # Try to iterate over it
    try:
        for key, value in embedder_state.items():
            if hasattr(value, 'value') and hasattr(value.value, 'shape'):
                print(f"  {key}: {value.value.shape} {value.value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    except Exception as e:
        print(f"Error iterating: {e}")

    # Try to access as dict
    try:
        print(f"\nEmbedder as dict: {dict(embedder_state)}")
    except Exception as e:
        print(f"Error converting to dict: {e}")

#!/usr/bin/env python3
"""Debug value model state and trainable filter."""

import sys
sys.path.insert(0, '/root/autodl-tmp/openpi/src')

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.traverse_util as traverse_util
from openpi.training import config as _config
from openpi.models import value_config

# Load the value config
config = _config.get_config("pi05_droid_100_value")
print(f"Config name: {config.name}")
print(f"Freeze filter: {config.freeze_filter}")
print(f"Trainable filter: {config.trainable_filter}")

# Create model
rng = jax.random.PRNGKey(42)
model = config.model.create(rng)
print("Model created.")

# Get full state
full_state = nnx.state(model)
print("\nFull state structure (top-level):")
for key in full_state.keys():
    print(f"  {key}: {type(full_state[key])}")

# Get trainable params (filtered)
trainable_params = full_state.filter(config.trainable_filter)
print("\nTrainable params count:", len(trainable_params.flat_state()))
print("Trainable params keys:")
flat_trainable = traverse_util.flatten_dict(trainable_params.to_pure_dict(), sep='/')
for key in sorted(flat_trainable.keys())[:20]:
    print(f"  {key}: {flat_trainable[key].shape}")

# Check if support variable is in full state
print("\nChecking for support variable:")
def find_support(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            find_support(v, prefix + k + "/")
        elif isinstance(v, nnx.Variable):
            if 'support' in k:
                print(f"  Found support variable at {prefix}{k}: shape {v.value.shape}")
                # Check if it's a Param
                print(f"    Is Param? {isinstance(v, nnx.Param)}")
                print(f"    Is Variable? {isinstance(v, nnx.Variable)}")
                # Check if it's in trainable params
                path = prefix + k
                if path in flat_trainable:
                    print(f"    WARNING: support variable appears in trainable params!")
                else:
                    print(f"    Not in trainable params.")
        elif hasattr(v, '__dict__'):
            # maybe nested object
            pass

find_support(full_state.to_pure_dict())

# Check param groups for multi-LR optimizer
if isinstance(config.lr_schedule, _config.MultiLRScheduleConfig):
    print("\nMulti-LR schedule param masks:")
    for group_name, filter_obj in config.lr_schedule.param_masks.items():
        print(f"  {group_name}: {filter_obj}")
        # Apply filter to full state
        filtered = full_state.filter(filter_obj)
        flat = traverse_util.flatten_dict(filtered.to_pure_dict(), sep='/')
        print(f"    Matches {len(flat)} keys")
        for key in sorted(flat.keys())[:3]:
            print(f"      {key}")
        if len(flat) > 3:
            print(f"      ...")

print("\nDone.")
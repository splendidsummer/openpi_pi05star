#!/usr/bin/env python3
"""Test data loader returns 3-tuple for value training."""

import sys
sys.path.insert(0, '/root/autodl-tmp/openpi/src')

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# Load config
config = _config.get_config("pi05_droid_100_value")

# Override num_workers to 0 to avoid multiprocessing issues
import dataclasses
config = dataclasses.replace(config, num_workers=0)

# Create data loader
loader = _data_loader.create_data_loader(
    config=config,
    shuffle=False,
    num_batches=1
)

# Get first batch
batch = next(iter(loader))
print(f"Batch type: {type(batch)}")
print(f"Batch length: {len(batch)}")

if len(batch) == 3:
    observation, actions, value_targets = batch
    print(f"✓ Data loader returns 3-tuple (observation, actions, value_targets)")
    print(f"  - Observation type: {type(observation)}")
    print(f"  - Actions shape: {actions.shape}")
    print(f"  - Value targets shape: {value_targets.shape}")
    print(f"  - Value targets sample: {value_targets[:2]}")
    print("\n✓ Test 5 PASSED: Data loader returns correct 3-tuple format")
else:
    print(f"✗ Data loader returns {len(batch)}-tuple instead of 3-tuple")
    print("\n✗ Test 5 FAILED")
    sys.exit(1)

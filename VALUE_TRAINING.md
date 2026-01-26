# Value Training Guide

This guide explains how to train value models using the `pi05_droid_100_value` configuration in openpi.

## Overview

Value training is different from standard action prediction training. Instead of predicting robot actions, value models predict the expected future return (value) for a given state. This is useful for:

- Learning state value functions for reinforcement learning
- Evaluating the quality of different states
- Providing auxiliary training signals for policy learning

### Key Differences from Action Training

| Aspect | Action Training | Value Training |
|--------|----------------|----------------|
| **Model Output** | Action sequences | Scalar value predictions |
| **Loss Function** | Flow matching / autoregressive loss | MSE between predicted and target values |
| **Data Requirements** | Observations + actions | Observations + value targets |
| **Training Script** | `scripts/train.py` | `scripts/train_value.py` |

## Prerequisites

### Hardware Requirements

- **GPU Memory**: Minimum 24GB (e.g., RTX 3090, RTX 4090)
- **Batch Size**: Default is 2 (can be adjusted based on GPU memory)
- **Training Time**: Depends on dataset size and number of steps

### Software Requirements

Ensure you have completed the standard openpi installation:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Model Checkpoint

The value training configuration uses the SIGLIP vision encoder from the pi05_droid checkpoint:

- **Checkpoint Path**: `gs://openpi-assets/checkpoints/pi05_droid/params`
- **Weight Loader**: `SIGLIPOnlyWeightLoader` (loads only vision encoder, not LLM)
- **LLM Path**: `/root/autodl-tmp/gemma-3-270m` (update this path in config if needed)

## Data Preparation

### Dataset Requirements

Value training requires a dataset with the following fields:

1. **Observations**: Standard robot observations (images, joint positions, etc.)
2. **Actions**: Robot actions (included for compatibility but not used by value model)
3. **Value Targets**: The key difference - each observation needs an associated value target

### Value Target Computation

Value targets typically represent the expected future return from a given state. Common approaches:

1. **Remaining Timesteps (Cost-to-Go)**: Negative number of steps until episode end
   ```
   V(s_t) = -1 * (episode_length - t) + 1
   ```
   Example: For a 100-step episode:
   - At t=0: V = -99
   - At t=50: V = -49
   - At t=99: V = 0

2. **Monte Carlo Returns**: Sum of discounted future rewards from that state
   ```
   V(s_t) = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^n*r_{t+n}
   ```

3. **TD(λ) Returns**: Mixture of n-step returns
4. **Reward-to-Go**: Cumulative future rewards without discounting

### Dataset Format

The dataset should be in LeRobot format with an additional `state_value` field:

```python
{
    "observation/exterior_image_1_left": [...],
    "observation/wrist_image_left": [...],
    "observation/joint_position": [...],
    "observation/gripper_position": [...],
    "actions": [...],
    "state_value": [0.75],  # Value target for this state
    "reward": [0.1],        # Optional: immediate reward
    "prompt": "pick up the fork"
}
```

### Example: Computing Value Targets for DROID Dataset

#### Method 1: Remaining Timesteps (Recommended)

This computes `state_value = -1 * (episode_length - t) + 1` for each state:

```python
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load your dataset
dataset = LeRobotDataset("SummerZhang/droid_100")

# Compute value targets as negative remaining timesteps
for episode_idx in range(dataset.num_episodes):
    episode_data = dataset.get_episode(episode_idx)
    episode_length = len(episode_data["observation/exterior_image_1_left"])

    # Compute: V(t) = -1 * (episode_length - t) + 1
    # At t=0: V = -(episode_length-1)
    # At t=episode_length-1: V = 0
    value_targets = -1 * np.arange(episode_length, 0, -1) + 1

    # Add to dataset
    episode_data["state_value"] = value_targets
    # Save back to dataset...

print(f"Value range: [{value_targets.min()}, {value_targets.max()}]")
# Example output: Value range: [-99, 0] for 100-step episodes
```

#### Method 2: Reward-to-Go (Alternative)

If you prefer using actual rewards:

```python
# Compute reward-to-go as value targets
for episode_idx in range(dataset.num_episodes):
    episode_data = dataset.get_episode(episode_idx)
    rewards = episode_data["reward"]

    # Compute cumulative future rewards
    value_targets = np.cumsum(rewards[::-1])[::-1]

    episode_data["state_value"] = value_targets
```

## Configuration

### Understanding the `pi05_droid_100_value` Config

The value training configuration is defined in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_droid_100_value",
    model=ValueConfig(
        pi05=True,
        model_path="/root/autodl-tmp/gemma-3-270m",  # Update this path!
        num_value_bins=201,  # Distributional value learning with 201 bins
        Vmin=-199.0,  # For max episode length ~200: -(200-1) = -199
        Vmax=0.0,  # Maximum value at episode end (0 remaining steps)
    ),
    data=LeRobotDROIDValueDataConfig(
        repo_id="SummerZhang/droid_100",  # Your dataset repo
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            asset_id="droid",  # Reuse DROID normalization stats
        ),
    ),
    weight_loader=SIGLIPOnlyWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_droid/params"
    ),
    num_train_steps=20_000,
    batch_size=2,
)
```

### Key Configuration Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `model_path` | Path to Gemma-3-270M checkpoint | `/root/autodl-tmp/gemma-3-270m` | **Must update to your local path** |
| `num_value_bins` | Number of bins for distributional RL | 201 | More bins = finer value resolution |
| `Vmin` / `Vmax` | Value range | -199.0 / 0.0 | For remaining timesteps: Vmin = -(max_episode_length-1), Vmax = 0 |
| `repo_id` | HuggingFace dataset repo | `SummerZhang/droid_100` | Update to your dataset |
| `batch_size` | Training batch size | 2 | Increase if you have more GPU memory |
| `num_train_steps` | Total training steps | 20,000 | Adjust based on dataset size |

### Customizing the Configuration

To use your own dataset, modify the config in `src/openpi/training/config.py`:

```python
# Update these fields:
repo_id="your_username/your_dataset",  # Your dataset
model_path="/path/to/your/gemma-3-270m",  # Your Gemma checkpoint

# For remaining timesteps method:
# Set Vmin to -(max_episode_length - 1)
# Set Vmax to 0 (final timestep has 0 remaining steps)
Vmin=-199.0,  # If max episode length is ~200 steps: -(200-1) = -199
Vmax=0.0,     # Episodes end at 0

# For reward-based methods:
# Vmin=min_expected_return,  # Minimum cumulative reward
# Vmax=max_expected_return,  # Maximum cumulative reward
```

**How to determine Vmin for remaining timesteps:**

```python
# Find maximum episode length in your dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("your_dataset")
max_length = 0
for ep_idx in range(dataset.num_episodes):
    ep_data = dataset.get_episode(ep_idx)
    ep_length = len(ep_data["observation/exterior_image_1_left"])
    max_length = max(max_length, ep_length)

print(f"Set Vmin = -{max_length - 1}")  # e.g., Vmin = -149 for max_length=150
```

## Training Steps

### Step 1: Verify Dataset

First, ensure your dataset is accessible and contains the required `state_value` field:

```bash
# Test loading the dataset
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('SummerZhang/droid_100')
sample = dataset[0]
print('Dataset keys:', sample.keys())
assert 'state_value' in sample, 'Missing state_value field!'
print('✓ Dataset is ready for value training')
"
```

### Step 2: Compute Normalization Statistics

Normalization statistics must be computed before training:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
```

This will:
- Load your dataset
- Compute statistics (mean, std, quantiles) for states and actions
- Save results to `~/.cache/openpi/assets/droid/norm_stats.json`

**Expected output:**
```
Computing normalization statistics for config: pi05_droid_100_value
Loading dataset: SummerZhang/droid_100
Computing statistics for 1000 samples...
Saved normalization stats to: ~/.cache/openpi/assets/droid/norm_stats.json
```

### Step 3: Start Training

Run the value training script:

```bash
# Set JAX to use maximum GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Start training
uv run scripts/train_value.py pi05_droid_100_value \
    --exp-name=my_value_experiment \
    --overwrite
```

**Command options:**
- `--exp-name`: Name for this training run (creates `checkpoints/pi05_droid_100_value/my_value_experiment/`)
- `--overwrite`: Overwrite existing checkpoints (remove for safety)
- `--resume`: Resume from latest checkpoint (if training was interrupted)

**Expected output:**
```
[INFO] Running on: your-machine-name
[INFO] Initialized data loader:
  observation/images: (2, 224, 224, 3) float32
  observation/state: (2, 32) float32
  actions: (2, 50, 32) float32
  value_targets: (2,) float32
[INFO] Initialized train state
Step 0: loss=0.2341, grad_norm=1.2345, param_norm=45.6789
Step 100: loss=0.1823, grad_norm=0.9876, param_norm=45.6543
...
```

## Monitoring Training

### Weights & Biases (W&B)

Training metrics are automatically logged to W&B. View your training progress at:

```
https://wandb.ai/your-username/openpi/runs/your-run-id
```

**Key metrics to monitor:**

| Metric | Description | Expected Behavior |
|--------|-------------|-------------------|
| `loss` | MSE between predicted and target values | Should decrease over time |
| `grad_norm` | Gradient magnitude | Should stabilize (not explode) |
| `param_norm` | Parameter magnitude | Should remain stable |

### Checkpoints

Checkpoints are saved periodically to:
```
checkpoints/pi05_droid_100_value/my_value_experiment/
├── 1000/          # Checkpoint at step 1000
├── 2000/          # Checkpoint at step 2000
├── ...
└── wandb_id.txt   # W&B run ID for resuming
```

**Checkpoint contents:**
- `params/`: Model parameters
- `opt_state/`: Optimizer state
- `assets/`: Normalization statistics

### Training Progress

Monitor training in real-time:

```bash
# Watch training logs
tail -f checkpoints/pi05_droid_100_value/my_value_experiment/train.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### Common Issues

#### 1. Missing `state_value` field

**Error:**
```
KeyError: 'state_value'
```

**Solution:**
- Verify your dataset contains the `state_value` field
- Check the data config's repack transform includes `"state_value": "state_value"`
- Ensure value targets were computed and added to your dataset

#### 2. Model path not found

**Error:**
```
FileNotFoundError: the model path does not exist: /root/autodl-tmp/gemma-3-270m
```

**Solution:**
- Download the Gemma-3-270M checkpoint
- Update `model_path` in the config to point to your local checkpoint
- Ensure the path contains the model files (config.json, model weights, etc.)

#### 3. Out of memory (OOM)

**Error:**
```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED
```

**Solution:**
- Reduce batch size in config (try `batch_size=1`)
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to use more GPU memory
- Use gradient accumulation if available
- Consider using a smaller model or fewer value bins

#### 4. Loss not decreasing

**Symptoms:**
- Loss remains constant or increases
- Training appears stuck

**Solution:**
- Check that value targets are in the correct range (match `Vmin`/`Vmax`)
- Verify normalization statistics were computed correctly
- Ensure learning rate is appropriate (default: 3e-4)
- Check for NaN values in your dataset
- Verify the data loader is yielding value_targets correctly

#### 5. Value range mismatch

**Error:**
```
Values outside expected range [Vmin, Vmax]
```

**Solution:**
- Update `Vmin` and `Vmax` in config to match your value target range
- Normalize value targets to [0, 1] range before training
- Check value target computation (e.g., reward-to-go calculation)

### Debugging Tips

**Check data loader output:**
```python
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

config = _config.get_config("pi05_droid_100_value")
loader = _data_loader.create_data_loader(config, sharding=None, shuffle=False)
batch = next(iter(loader))

print("Batch structure:")
print(f"  Observation: {batch[0]}")
print(f"  Actions: {batch[1].shape}")
print(f"  Value targets: {batch[2].shape}")
print(f"  Value range: [{batch[2].min():.3f}, {batch[2].max():.3f}]")
```

## Quick Reference

### Complete Training Workflow

```bash
# 1. Verify dataset has state_value field
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
           ds = LeRobotDataset('SummerZhang/droid_100'); \
           assert 'state_value' in ds[0]"

# 2. Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value

# 3. Start training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_value.py pi05_droid_100_value \
    --exp-name=my_value_experiment

# 4. Resume training (if interrupted)
uv run scripts/train_value.py pi05_droid_100_value \
    --exp-name=my_value_experiment \
    --resume
```

### Key Files Modified

- **`scripts/train_value.py`**: Value-specific training script
- **`src/openpi/training/data_loader.py`**: Modified to yield value_targets
- **`src/openpi/training/config.py`**: Contains `pi05_droid_100_value` config
- **`src/openpi/models/value.py`**: Value model implementation
- **`src/openpi/models/value_config.py`**: Value model configuration

### Important Configuration Parameters

```python
# In src/openpi/training/config.py
model_path="/path/to/gemma-3-270m"  # Update this!
repo_id="your_username/your_dataset"  # Your dataset
Vmin=0.0  # Minimum value
Vmax=1.0  # Maximum value
num_value_bins=201  # Number of bins
batch_size=2  # Adjust based on GPU memory
num_train_steps=20_000  # Total training steps
```

### Next Steps

After training completes:

1. **Evaluate the model**: Use the trained value model to predict values for new states
2. **Use for RL**: Integrate the value function into your RL algorithm
3. **Fine-tune**: Continue training on additional data if needed

For questions or issues, please file an issue at: https://github.com/Physical-Intelligence/openpi/issues

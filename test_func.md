# Value Function Training - Test Documentation

This document outlines all tests required to verify the value function training implementation for DROID dataset.

## Test Categories

1. [Unit Tests](#unit-tests)
2. [Data Pipeline Tests](#data-pipeline-tests)
3. [Model Tests](#model-tests)
4. [Integration Tests](#integration-tests)
5. [End-to-End Verification](#end-to-end-verification)

---

## Unit Tests

### 1.1 Value Model Unit Tests

**File:** `src/openpi/models/value_test.py`

**Purpose:** Verify Value model initialization, forward pass, and distributional value head functionality.

**Tests Included:**
- `test_value_model_initialization()` - Verify model initializes with correct parameters
- `test_value_forward_pass()` - Test forward pass produces correct output shapes
- `test_distributional_value_head()` - Test value head produces valid probability distributions
- `test_value_head_support_atoms()` - Test support atoms are correctly initialized

**Run Command:**
```bash
# Run all value model tests
pytest src/openpi/models/value_test.py -v -s

# Run specific test
pytest src/openpi/models/value_test.py::test_value_model_initialization -v -s
```

**Expected Results:**
- ✓ Model initializes successfully with Gemma-3-270M
- ✓ num_value_bins = 201
- ✓ Vmin = -199.0, Vmax = 0.0
- ✓ valuegemma_width = 640
- ✓ Forward pass produces loss with shape (batch_size,)
- ✓ Loss values are finite (no NaN or Inf)
- ✓ Value predictions are within [Vmin, Vmax] range
- ✓ Support atoms span from -199.0 to 0.0

**How to Verify:**
```bash
# Should see output like:
# ✓ Value model initialized successfully
# ✓ Forward pass successful, loss: [...]
# ✓ Value head test passed, values: [...]
# ✓ Support atoms correctly initialized: [-199.00, ..., 0.00]
```

---

### 1.2 Data Conversion Unit Tests

**File:** `examples/droid/convert_droid_data_to_lerobot_test.py`

**Purpose:** Verify value target computation logic is correct.

**Tests Included:**
- `test_value_target_computation()` - Verify value targets computed correctly
- `test_value_target_monotonicity()` - Test values increase monotonically
- `test_value_target_different_episode_lengths()` - Test various episode lengths
- `test_reward_initialization()` - Test reward is initialized to 0.0
- `test_value_target_array_format()` - Test correct numpy array format

**Run Command:**
```bash
# Run all data conversion tests
pytest examples/droid/convert_droid_data_to_lerobot_test.py -v

# Run specific test
pytest examples/droid/convert_droid_data_to_lerobot_test.py::test_value_target_computation -v
```

**Expected Results:**
- ✓ Value targets range from -(episode_length-1) to 0
- ✓ First timestep: V(0) = -(episode_length-1)
- ✓ Last timestep: V(T-1) = 0.0
- ✓ Values increase monotonically over episode
- ✓ Correct for various episode lengths (10, 50, 100, 200)
- ✓ Reward initialized to 0.0
- ✓ Arrays have shape (1,) and dtype float32

**How to Verify:**
```bash
# Should see output like:
# ✓ Value target computation test passed for episode_length=100
# ✓ Value targets are monotonically increasing
# ✓ Value targets correct for episode lengths: [10, 50, 100, 200]
# ✓ Reward initialization test passed
# ✓ Value target array format test passed
```

---

## Data Pipeline Tests

### 2.1 Dataset Conversion Verification

**Purpose:** Verify converted DROID dataset contains required fields.

**Test Command:**
```bash
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('SummerZhang/droid_100')
sample = ds[0]
assert 'state_value' in sample, 'Missing state_value field'
assert 'reward' in sample, 'Missing reward field'
print('✓ Dataset has state_value and reward fields')
print(f'Sample state_value: {sample[\"state_value\"]}')
print(f'Sample reward: {sample[\"reward\"]}')
"
```

**Expected Results:**
- ✓ Dataset loads successfully
- ✓ state_value field exists in samples
- ✓ reward field exists in samples
- ✓ Values are in expected range

**How to Verify:**
- Check that state_value is a negative number (for early timesteps)
- Check that reward is 0.0
- Verify data types are float32

---

### 2.2 Normalization Statistics Verification

**Purpose:** Verify normalization statistics are computed for all features including value_targets.

**Test Command:**
```bash
# Check norm stats file exists
test -f ~/.cache/openpi/assets/droid/norm_stats.json && echo "✓ Norm stats file exists"

# Inspect norm stats
cat ~/.cache/openpi/assets/droid/norm_stats.json | jq '.'

# Check value_targets stats specifically
cat ~/.cache/openpi/assets/droid/norm_stats.json | jq '.value_targets'
```

**Expected Results:**
- ✓ norm_stats.json file exists
- ✓ Contains stats for: state, actions, value_targets, reward
- ✓ Each stat has: mean, std, q01, q99
- ✓ No NaN or Inf values
- ✓ value_targets mean is negative (around -100 for ~200 step episodes)
- ✓ value_targets std is reasonable (not too small or too large)

**How to Verify:**
```bash
# Should see JSON output like:
# {
#   "state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
#   "actions": {"mean": [...], "std": [...], "q01": [...], "q99": [...]},
#   "value_targets": {"mean": [-99.5], "std": [57.7], "q01": [-199.0], "q99": [0.0]},
#   "reward": {"mean": [0.0], "std": [0.0], "q01": [0.0], "q99": [0.0]}
# }
```

---

## Model Tests

### 3.1 Model Initialization Test

**Purpose:** Verify Value model initializes correctly with all components.

**Test Command:**
```bash
python -c "
from openpi.models.value_config import ValueConfig
import jax

config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))

print(f'✓ Model initialized: {type(model).__name__}')
print(f'✓ Value bins: {model.num_value_bins}')
print(f'✓ Value range: [{model.Vmin}, {model.Vmax}]')
print(f'✓ Hidden size: {model.valuegemma_width}')
"
```

**Expected Results:**
- ✓ Model type is 'Value'
- ✓ num_value_bins = 201
- ✓ Vmin = -199.0, Vmax = 0.0
- ✓ valuegemma_width = 640

---

### 3.2 Model Forward Pass Test

**Purpose:** Verify forward pass works correctly with fake data.

**Test Command:**
```bash
python -c "
from openpi.models.value_config import ValueConfig
import jax
import jax.numpy as jnp

config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))

# Create fake observation
obs = config.fake_obs(batch_size=2)
value_targets = jnp.array([-50.0, -100.0])

# Forward pass
rng = jax.random.key(1)
loss = model.compute_loss(rng, obs, train=True, value_targets=value_targets)

print(f'✓ Forward pass successful')
print(f'✓ Loss shape: {loss.shape}')
print(f'✓ Loss values: {loss}')
print(f'✓ All finite: {jnp.all(jnp.isfinite(loss))}')
"
```

**Expected Results:**
- ✓ Forward pass completes without errors
- ✓ Loss shape is (2,) for batch_size=2
- ✓ Loss values are finite (no NaN or Inf)
- ✓ Loss values are positive (MSE loss)

---

### 3.3 Weight Loading Test

**Purpose:** Verify SIGLIP weights load correctly from pi05_droid checkpoint.

**Test Command:**
```bash
python -c "
from openpi.training.weight_loaders import SIGLIPOnlyWeightLoader
from openpi.models.value_config import ValueConfig
from flax import nnx
import jax

config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))
_, state = nnx.split(model)
params = state.to_pure_dict()

loader = SIGLIPOnlyWeightLoader('gs://openpi-assets/checkpoints/pi05_droid/params')
loaded_params = loader.load(params)

assert 'ValueGemma' in loaded_params
assert 'img' in loaded_params['ValueGemma']
print('✓ SIGLIP weights loaded successfully')
print(f'✓ Loaded keys: {list(loaded_params.keys())}')
"
```

**Expected Results:**
- ✓ Weight loader completes without errors
- ✓ ValueGemma key exists in loaded params
- ✓ img key exists under ValueGemma
- ✓ SIGLIP vision encoder weights are loaded
- ✓ LLM weights are NOT loaded (should be from Gemma-3-270M)

---

## Integration Tests

### 4.1 Training Pipeline Test (Short Run)

**Purpose:** Verify training starts and runs without errors for a few steps.

**Test Command:**
```bash
# Start training for 100 steps
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=test_run \
    --overwrite &

TRAIN_PID=$!

# Wait for first checkpoint (or 5 minutes)
sleep 300

# Kill training
kill $TRAIN_PID

# Verify checkpoint saved
test -d ./checkpoints/pi05_droid_100_value/test_run/1000 && echo "✓ Checkpoint saved"
```

**Expected Results:**
- ✓ Training starts without errors
- ✓ Loss is computed and logged
- ✓ Gradients are computed successfully
- ✓ Checkpoint is saved at step 1000
- ✓ No NaN or Inf in loss/gradients

**How to Verify:**
- Check W&B dashboard for loss curves
- Check terminal output for loss values
- Verify checkpoint directory exists

---

### 4.2 Data Loader Test

**Purpose:** Verify data loader yields correct batch format with value_targets.

**Test Command:**
```bash
python -c "
from openpi.training.config import get_config
from openpi.training.data_loader import create_data_loader

config = get_config('pi05_droid_100_value')
data_loader = create_data_loader(config, shuffle=False, num_batches=1)

for batch in data_loader:
    if len(batch) == 3:
        observation, actions, value_targets = batch
        print('✓ Data loader yields 3-tuple (observation, actions, value_targets)')
        print(f'✓ Observation type: {type(observation).__name__}')
        print(f'✓ Actions shape: {actions.shape}')
        print(f'✓ Value targets shape: {value_targets.shape}')
    else:
        print('✗ Data loader should yield 3-tuple for value training')
    break
"
```

**Expected Results:**
- ✓ Data loader yields 3-tuple (observation, actions, value_targets)
- ✓ Observation is Observation dataclass
- ✓ Actions shape is (batch_size, action_horizon, action_dim)
- ✓ Value targets shape is (batch_size,) or (batch_size, 1)

---

## End-to-End Verification

### 5.1 Complete Pipeline Verification

**Purpose:** Verify the entire value training pipeline works end-to-end.

**Test Steps:**

**Step 1: Verify Data Conversion**
```bash
# Check dataset has required fields
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('SummerZhang/droid_100')
sample = ds[0]
assert 'state_value' in sample
assert 'reward' in sample
print('✓ Dataset conversion successful')
"
```

**Step 2: Verify Normalization Stats**
```bash
# Check norm stats computed
test -f ~/.cache/openpi/assets/droid/norm_stats.json && echo "✓ Norm stats exist"
cat ~/.cache/openpi/assets/droid/norm_stats.json | jq '.value_targets' && echo "✓ Value targets stats computed"
```

**Step 3: Verify Model Initialization**
```bash
# Test model loads correctly
python -c "
from openpi.models.value_config import ValueConfig
import jax
config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))
print('✓ Model initialization successful')
"
```

**Step 4: Verify Training Starts**
```bash
# Run training for 10 steps
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=verification_test \
    --overwrite

# Check logs for successful training
# Should see loss values being logged
```

**Expected Results:**
- ✓ All steps complete without errors
- ✓ Data flows correctly through pipeline
- ✓ Training produces finite loss values
- ✓ Checkpoints are saved correctly

---

## Test Summary and Checklist

### Complete Test Execution Order

Execute tests in this order for comprehensive verification:

**Phase 1: Unit Tests (Day 1)**
```bash
# 1. Run value model unit tests
pytest src/openpi/models/value_test.py -v -s

# 2. Run data conversion unit tests
pytest examples/droid/convert_droid_data_to_lerobot_test.py -v
```

**Phase 2: Data Pipeline Tests (Day 1)**
```bash
# 3. Verify dataset conversion
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('SummerZhang/droid_100'); assert 'state_value' in ds[0]; print('✓ Dataset OK')"

# 4. Verify normalization stats
test -f ~/.cache/openpi/assets/droid/norm_stats.json && cat ~/.cache/openpi/assets/droid/norm_stats.json | jq '.value_targets'
```

**Phase 3: Model Tests (Day 1-2)**
```bash
# 5. Test model initialization
python -c "from openpi.models.value_config import ValueConfig; import jax; config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m'); model = config.create(jax.random.key(0)); print('✓ Model OK')"

# 6. Test forward pass
python -c "from openpi.models.value_config import ValueConfig; import jax; import jax.numpy as jnp; config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m'); model = config.create(jax.random.key(0)); obs = config.fake_obs(batch_size=2); loss = model.compute_loss(jax.random.key(1), obs, train=True, value_targets=jnp.array([-50.0, -100.0])); print(f'✓ Forward pass OK, loss: {loss}')"

# 7. Test weight loading
python -c "from openpi.training.weight_loaders import SIGLIPOnlyWeightLoader; from openpi.models.value_config import ValueConfig; from flax import nnx; import jax; config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m'); model = config.create(jax.random.key(0)); _, state = nnx.split(model); params = state.to_pure_dict(); loader = SIGLIPOnlyWeightLoader('gs://openpi-assets/checkpoints/pi05_droid/params'); loaded_params = loader.load(params); print('✓ Weight loading OK')"
```

**Phase 4: Integration Tests (Day 2)**
```bash
# 8. Test data loader
python -c "from openpi.training.config import get_config; from openpi.training.data_loader import create_data_loader; config = get_config('pi05_droid_100_value'); data_loader = create_data_loader(config, shuffle=False, num_batches=1); batch = next(iter(data_loader)); print(f'✓ Data loader OK, batch length: {len(batch)}')"

# 9. Test training pipeline (short run)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py pi05_droid_100_value --exp-name=test_run --overwrite
```

---


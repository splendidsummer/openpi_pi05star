# Value Function Training Implementation Plan for DROID Dataset

## Executive Summary

**Good News**: ~80% of the value training infrastructure already exists in the codebase!

**What's Already Implemented:**
- ✅ `value.py` - Complete Value model with SigLIP + Gemma-3-270M
- ✅ `value_config.py` - ValueConfig with proper hyperparameters
- ✅ `train_value.py` - Full training script with value loss computation
- ✅ `SIGLIPOnlyWeightLoader` - Loads vision encoder from pi05_droid checkpoint
- ✅ `ValueGemmaTokenizer` - Gemma-3 tokenizer in tokenizer.py
- ✅ `DroidInputs` transform - Handles value_targets and reward extraction
- ✅ `compute_norm_stats.py` - Detects and computes stats for value_targets/reward
- ✅ `DataLoaderImpl` - Yields (observation, actions, value_targets) tuples
- ✅ Training config `pi05_droid_100_value` - Ready to use

**What Needs Implementation:**
1. Modify `convert_droid_data_to_lerobot.py` to add state_value and reward fields
2. Fix TODOs in `value.py` (5 items)
3. Run data conversion on raw DROID data
4. Compute normalization statistics
5. Write comprehensive unit tests
6. Execute training and validation

**User Preferences:**
- Value computation: Method 1 (remaining timesteps: V(t) = -1 * (episode_length - t) + 1)
- Loss function: MSE on expected values
- Dataset status: Need to convert raw DROID data

---

## Implementation Phases

### Phase 1: Data Conversion (CRITICAL - Must Complete First)

**Objective**: Modify the DROID-to-LeRobot conversion script to include value targets and rewards.

**File**: `/root/autodl-tmp/openpi/examples/droid/convert_droid_data_to_lerobot.py`

**Changes Required:**

1. **Add features to dataset schema** (around line 51):
```python
features={
    # ... existing features ...
    "state_value": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state_value"],
    },
    "reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["reward"],
    },
}
```

2. **Compute value targets during episode processing** (around line 106-148):
```python
for episode_path in tqdm(episode_paths, desc="Converting episodes"):
    # Load trajectory
    trajectory = load_trajectory(str(episode_path), recording_folderpath=str(recording_folderpath))
    episode_length = len(trajectory)

    # ... existing code to get language_instruction ...

    # Write frames with value targets
    for step_idx, step in enumerate(trajectory):
        # Compute value target: remaining timesteps approach
        state_value = -1.0 * (episode_length - step_idx - 1)
        # At t=0: V = -(episode_length-1)
        # At t=episode_length-1: V = 0

        # Reward (set to 0 for now, can be task-specific later)
        reward = 0.0

        dataset.add_frame({
            # ... existing fields ...
            "state_value": np.array([state_value], dtype=np.float32),
            "reward": np.array([reward], dtype=np.float32),
        })
    dataset.save_episode()
```

**Validation**:
- Verify state_value ranges from -(episode_length-1) to 0
- Check final timestep has value = 0
- Ensure all episodes have consistent computation

**Output**: LeRobot dataset with state_value and reward fields ready for training

---

### Phase 2: Model Refinements (HIGH PRIORITY)

**Objective**: Fix TODOs in value.py to ensure proper model initialization and loss computation.

**File**: `/root/autodl-tmp/openpi/src/openpi/models/value.py`

**Changes Required:**

**TODO #1 (Line 115)**: Build Gemma-3-270M model
- **Status**: Already implemented via `build_gemma_3_270m_model()` in gemma_utils.py
- **Action**: Verify function works with local checkpoint path
- **Test**: Ensure model loads from `/root/autodl-tmp/gemma-3-270m`

**TODO #2 (Line 130)**: Confirm train/init mode setting
- **Current**: `img.lazy_init(..., train=False, rngs=rngs)`
- **Issue**: Need to verify this is correct for both training and inference
- **Action**: Keep as-is (train=False during initialization is correct)
- **Note**: Training mode controlled by model.train() / model.eval() during forward pass

**TODO #3 (Line 136)**: Pass config parameters to value head
- **Current**: `self.value_head = DistributionalValueHeadGemma3()`
- **Fix**:
```python
self.value_head = DistributionalValueHeadGemma3(
    hidden_size=640,  # gemma_3_270m width
    num_bins=self.num_value_bins,  # 201 from config
    v_min=self.Vmin,  # -199.0 from config
    v_max=self.Vmax,  # 0.0 from config
    rngs=rngs
)
```
- **Also update DistributionalValueHeadGemma3.__init__** to accept these parameters

**TODO #4 (Line 162)**: Check PaliGemma embed method
- **Current**: `self.ValueGemma.llm(obs.tokenized_prompt, method="embed")`
- **Action**: Verify with gemma library documentation
- **Expected**: method="embed" returns token embeddings without forward pass
- **Test**: Print output shape to confirm

**TODO #5 (Line 173)**: Fix loss computation issues
- **Issue 1**: Line 200 - `last_timestep_output` indexing may be incorrect
  - Current: `prefix_out[:, positions, :]` (uses positions as indices)
  - Should be: `prefix_out[:, -1, :]` (last position in sequence)
- **Issue 2**: Keep MSE loss (per user preference)
- **Fix**:
```python
# Get last timestep output (final token in sequence)
last_timestep_output = prefix_out[:, -1, :]  # [batch, hidden_dim]

# Project through value head
value_expectation = self.value_head(last_timestep_output)  # [batch]

# MSE loss
loss = jnp.square(value_expectation - value_targets)  # [batch]
return loss
```

**Validation**:
- Test forward pass with fake observation
- Verify loss computation produces scalar per batch element
- Check gradient flow through all parameters

---

### Phase 3: Normalization Statistics (CRITICAL)

**Objective**: Compute normalization statistics for the converted DROID dataset.

**File**: `/root/autodl-tmp/openpi/scripts/compute_norm_stats.py`

**Current State**: Already handles value_targets and reward detection (lines 102-126)

**Execution**:
```bash
# After converting DROID data and pushing to HuggingFace
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
```

**What It Does**:
1. Loads dataset with repack + data transforms
2. Detects keys: ["state", "actions", "value_targets", "reward"]
3. Computes running statistics (mean, std, q01, q99)
4. Saves to `~/.cache/openpi/assets/droid/norm_stats.json`

**Validation**:
- Check that value_targets stats are computed
- Verify mean/std are reasonable (not NaN or inf)
- Ensure q01/q99 cover the expected range

**Output**: `norm_stats.json` with statistics for all features including value_targets

---

### Phase 4: Unit Testing (HIGH PRIORITY)

**Objective**: Create comprehensive unit tests for all value model components.

**Test Files to Create:**

**A. `/root/autodl-tmp/openpi/src/openpi/models/value_test.py`**

```python
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.value import Value
from openpi.models.value_config import ValueConfig
from openpi.models.model import Observation

@pytest.mark.manual
def test_value_model_initialization():
    """Test Value model initializes correctly with Gemma-3-270M"""
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

@pytest.mark.manual
def test_value_forward_pass():
    """Test forward pass produces correct output shapes"""
    config = ValueConfig(model_path="/root/autodl-tmp/gemma-3-270m")
    model = config.create(jax.random.key(0))

    # Create fake observation
    obs = model.fake_obs(batch_size=2)
    value_targets = jnp.array([[-50.0], [-100.0]])

    # Forward pass
    rng = jax.random.key(1)
    loss = model.compute_loss(rng, obs, train=True, value_targets=value_targets)

    assert loss.shape == (2,)  # One loss per batch element
    assert jnp.all(jnp.isfinite(loss))

@pytest.mark.manual
def test_distributional_value_head():
    """Test value head produces valid probability distributions"""
    from openpi.models.value import DistributionalValueHeadGemma3

    head = DistributionalValueHeadGemma3(
        hidden_size=640,
        num_bins=201,
        v_min=-199.0,
        v_max=0.0,
        rngs=jax.random.key(0)
    )

    # Test forward pass
    x = jnp.ones((4, 640))  # batch_size=4, hidden_size=640
    value = head(x)

    assert value.shape == (4,)
    assert jnp.all(value >= -199.0)
    assert jnp.all(value <= 0.0)
```

**B. Add tests to `/root/autodl-tmp/openpi/src/openpi/training/weight_loaders_test.py`**

```python
def test_siglip_only_weight_loader():
    """Test SIGLIP weights load correctly from pi05_droid checkpoint"""
    from openpi.training.weight_loaders import SIGLIPOnlyWeightLoader
    from openpi.models.value_config import ValueConfig

    # Create value model
    config = ValueConfig(model_path="/root/autodl-tmp/gemma-3-270m")
    model = config.create(jax.random.key(0))
    _, state = nnx.split(model)
    params = state.to_pure_dict()

    # Load SIGLIP weights
    loader = SIGLIPOnlyWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params")
    loaded_params = loader.load(params)

    # Verify SIGLIP weights were loaded
    assert "ValueGemma" in loaded_params
    assert "img" in loaded_params["ValueGemma"]
```

**C. `/root/autodl-tmp/openpi/examples/droid/convert_droid_data_to_lerobot_test.py`**

```python
def test_value_target_computation():
    """Verify value targets computed correctly"""
    episode_length = 100

    for step_idx in range(episode_length):
        state_value = -1.0 * (episode_length - step_idx - 1)

        # Check range
        assert state_value >= -(episode_length - 1)
        assert state_value <= 0

        # Check final timestep
        if step_idx == episode_length - 1:
            assert state_value == 0.0
```

**Execution**:
```bash
# Run all value model tests
pytest src/openpi/models/value_test.py -v -s

# Run weight loader tests
pytest src/openpi/training/weight_loaders_test.py::test_siglip_only_weight_loader -v
```

---

### Phase 5: Training Execution (CRITICAL)

**Objective**: Execute value model training on the converted DROID dataset.

**Prerequisites**:
- ✅ Data converted with state_value and reward fields
- ✅ Normalization statistics computed
- ✅ Model TODOs fixed
- ✅ Unit tests passing

**Training Command**:
```bash
# Set GPU memory fraction to 90% for maximum utilization
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=value_training_v1 \
    --overwrite
```

**Configuration Review** (`pi05_droid_100_value` in config.py):
- Model: ValueConfig with Gemma-3-270M
- Dataset: SummerZhang/droid_100 (your converted dataset)
- Batch size: 2 (adjust based on GPU memory)
- Learning rate: 3e-4 (default, may need tuning)
- Training steps: 20,000
- Weight loader: SIGLIPOnlyWeightLoader (loads vision encoder from pi05_droid)

**Monitoring**:
- Watch W&B dashboard for loss curves
- Check for gradient explosions (grad_norm should be < 10)
- Verify loss decreases over time
- Monitor GPU memory usage

**Expected Behavior**:
- Initial loss: ~1000-5000 (MSE on unnormalized values)
- After 1000 steps: Loss should drop to ~100-500
- After 10000 steps: Loss should stabilize around 10-50
- Training time: ~2-4 hours on single GPU

**Checkpoints**:
- Saved every 1000 steps to `./checkpoints/pi05_droid_100_value/value_training_v1/`
- Contains: params, optimizer state, training metadata

---

### Phase 6: Validation and Testing

**Objective**: Validate trained value model produces reasonable predictions.

**A. Load Trained Checkpoint**:
```python
from openpi.models.value_config import ValueConfig
from openpi.models.model import restore_params

config = ValueConfig(model_path="/root/autodl-tmp/gemma-3-270m")
params = restore_params("./checkpoints/pi05_droid_100_value/value_training_v1/10000/params")
model = config.load(params)
```

**B. Run Inference on Validation Set**:
```python
# Load validation data
data_loader = create_data_loader(config, shuffle=False, num_batches=100)

predictions = []
targets = []

for obs, actions, value_targets in data_loader:
    # Predict values
    rng = jax.random.key(0)
    loss = model.compute_loss(rng, obs, train=False, value_targets=value_targets)

    # Get predictions (need to add method to return predictions, not just loss)
    # value_pred = model.predict_value(rng, obs)

    predictions.append(value_pred)
    targets.append(value_targets)
```

**C. Compute Evaluation Metrics**:
```python
import numpy as np

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

# Mean Squared Error
mse = np.mean((predictions - targets) ** 2)

# Mean Absolute Error
mae = np.mean(np.abs(predictions - targets))

# R² Score
ss_res = np.sum((targets - predictions) ** 2)
ss_tot = np.sum((targets - np.mean(targets)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
```

**Expected Results**:
- MSE: < 100 (good), < 50 (excellent)
- MAE: < 10 (good), < 5 (excellent)
- R²: > 0.8 (good), > 0.9 (excellent)

---

## Critical Files Summary

### Files to Modify

1. **`/root/autodl-tmp/openpi/examples/droid/convert_droid_data_to_lerobot.py`**
   - Add state_value and reward to features schema
   - Compute value targets during episode processing
   - Lines to modify: ~51 (features), ~106-148 (episode loop)

2. **`/root/autodl-tmp/openpi/src/openpi/models/value.py`**
   - Fix TODO #3: Pass config params to DistributionalValueHeadGemma3 (line 136)
   - Fix TODO #5: Correct last_timestep_output indexing (line 200)
   - Update DistributionalValueHeadGemma3.__init__ to accept parameters

3. **`/root/autodl-tmp/openpi/src/openpi/training/config.py`**
   - Review pi05_droid_100_value config (lines 1007-1031)
   - Verify Vmin/Vmax match dataset episode lengths
   - Adjust learning rate if needed

### Files to Create

1. **`/root/autodl-tmp/openpi/src/openpi/models/value_test.py`**
   - Unit tests for Value model
   - Tests for DistributionalValueHeadGemma3
   - Integration tests

2. **`/root/autodl-tmp/openpi/examples/droid/convert_droid_data_to_lerobot_test.py`**
   - Tests for value target computation
   - Validation tests for converted data

### Files Already Correct (No Changes Needed)

- ✅ `/root/autodl-tmp/openpi/src/openpi/models/value_config.py`
- ✅ `/root/autodl-tmp/openpi/scripts/train_value.py`
- ✅ `/root/autodl-tmp/openpi/src/openpi/training/weight_loaders.py` (SIGLIPOnlyWeightLoader)
- ✅ `/root/autodl-tmp/openpi/src/openpi/models/tokenizer.py` (ValueGemmaTokenizer)
- ✅ `/root/autodl-tmp/openpi/src/openpi/policies/droid_policy.py` (DroidInputs)
- ✅ `/root/autodl-tmp/openpi/scripts/compute_norm_stats.py`
- ✅ `/root/autodl-tmp/openpi/src/openpi/training/data_loader.py`

---

## Step-by-Step Execution Sequence

Execute these steps in order for successful value training implementation:

### Step 1: Data Conversion (Day 1)
```bash
# 1. Modify convert_droid_data_to_lerobot.py (add state_value and reward)
# 2. Run conversion on your raw DROID data
cd /root/autodl-tmp/openpi
uv run examples/droid/convert_droid_data_to_lerobot.py \
    --data_dir /path/to/your/raw/droid/data \
    --push_to_hub

# 3. Verify output dataset
# Check that state_value and reward fields exist in the parquet files
```

### Step 2: Model Refinements (Day 1)
```bash
# 1. Fix value.py TODOs #3 and #5
# 2. Update DistributionalValueHeadGemma3.__init__ signature
# 3. Test model initialization
python -c "
from openpi.models.value_config import ValueConfig
import jax
config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))
print('Model initialized successfully!')
"
```

### Step 3: Compute Normalization Statistics (Day 1)
```bash
# After dataset is ready on HuggingFace
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value

# Verify output
cat ~/.cache/openpi/assets/droid/norm_stats.json | jq '.value_targets'
```

### Step 4: Unit Testing (Day 2)
```bash
# 1. Create value_test.py with test cases
# 2. Create convert_droid_data_to_lerobot_test.py
# 3. Run tests
pytest src/openpi/models/value_test.py -v -s
pytest examples/droid/convert_droid_data_to_lerobot_test.py -v
```

### Step 5: Training (Day 2-3)
```bash
# Start training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=value_training_v1 \
    --overwrite

# Monitor training
# - Check W&B dashboard
# - Watch loss curves
# - Verify checkpoints are saved
```

### Step 6: Validation (Day 3)
```bash
# 1. Load trained checkpoint
# 2. Run inference on validation set
# 3. Compute evaluation metrics (MSE, MAE, R²)
# 4. Analyze predictions vs targets
```

---

## Verification Steps

### End-to-End Verification Checklist

After completing all implementation phases, verify the entire pipeline:

**✓ Data Pipeline Verification:**
```bash
# 1. Check converted dataset has required fields
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('SummerZhang/droid_100')
sample = ds[0]
assert 'state_value' in sample, 'Missing state_value field'
assert 'reward' in sample, 'Missing reward field'
print('✓ Dataset has state_value and reward fields')
"

# 2. Verify normalization stats computed
test -f ~/.cache/openpi/assets/droid/norm_stats.json && echo "✓ Norm stats exist"
cat ~/.cache/openpi/assets/droid/norm_stats.json | jq '.value_targets' && echo "✓ Value targets stats computed"
```

**✓ Model Verification:**
```bash
# 1. Test model initialization
python -c "
from openpi.models.value_config import ValueConfig
import jax
config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))
print(f'✓ Model initialized: {type(model).__name__}')
print(f'✓ Value bins: {model.num_value_bins}')
print(f'✓ Value range: [{model.Vmin}, {model.Vmax}]')
"

# 2. Test forward pass
python -c "
from openpi.models.value_config import ValueConfig
import jax
import jax.numpy as jnp
config = ValueConfig(model_path='/root/autodl-tmp/gemma-3-270m')
model = config.create(jax.random.key(0))
obs = model.fake_obs(batch_size=2)
value_targets = jnp.array([[-50.0], [-100.0]])
loss = model.compute_loss(jax.random.key(1), obs, train=True, value_targets=value_targets)
print(f'✓ Forward pass successful, loss shape: {loss.shape}')
"
```

**✓ Training Verification:**
```bash
# 1. Check training starts without errors
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=test_run \
    --overwrite &
TRAIN_PID=$!

# Wait for first checkpoint
sleep 300
kill $TRAIN_PID

# 2. Verify checkpoint saved
test -d ./checkpoints/pi05_droid_100_value/test_run/1000 && echo "✓ Checkpoint saved"
```

**✓ Weight Loading Verification:**
```bash
# Test SIGLIP weight loader
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
"
```

---

## Training Pipeline Alignment with CLAUDE.md

This implementation follows the training pipeline structure from CLAUDE.md:

```
Raw DROID Data (trajectory.h5 + MP4 videos)
  ↓
[Data Conversion] convert_droid_data_to_lerobot.py
  - Add state_value = -1 * (episode_length - t - 1)
  - Add reward = 0.0
  ↓
LeRobot Dataset (parquet files with state_value and reward)
  ↓
[repack_transforms] DroidInputs
  - Extract state_value → value_targets
  - Extract reward
  - Map cameras to standard format
  ↓
[data_transforms] ResizeImages, PadStatesAndActions
  - Resize to 224×224
  - Pad state/actions to action_dim
  ↓
[Normalize] Apply norm_stats.json
  - Normalize state, actions, value_targets, reward
  ↓
[model_transforms] TokenizePrompt
  - Tokenize language instruction with ValueGemmaTokenizer
  ↓
Observation Construction
  - Create Observation(images, image_masks, state, tokenized_prompt)
  ↓
[Image Encoding] SigLIP Vision Encoder
  - 3 camera views → 768 image tokens (256 per camera)
  ↓
[Language Encoding] Gemma-3-270M
  - Tokenized prompt → language embeddings
  ↓
[Feature Fusion] Concatenate prefix tokens
  - Images + Language → bidirectional attention
  ↓
[LLM Forward Pass] Gemma-3-270M
  - Process fused features
  - Extract last timestep output
  ↓
[Value Head] DistributionalValueHeadGemma3
  - 3-layer MLP: 640 → 512 → 128 → 201 bins
  - Softmax → expected value
  ↓
[Loss Computation] MSE Loss
  - loss = (predicted_value - value_target)²
  ↓
[Backpropagation] Update parameters
  - Optimizer: AdamW with learning rate 3e-4
  - Freeze: LLM parameters (fine-tune vision encoder + value head)
```

---

## Summary

**Implementation Complexity**: LOW (80% already implemented)

**Time Estimate**: 2-3 days
- Day 1: Data conversion + model fixes + norm stats
- Day 2: Unit tests + training start
- Day 3: Training completion + validation

**Key Success Factors**:
1. Correct value target computation in data conversion
2. Proper TODO fixes in value.py (especially line 200 indexing)
3. Normalization statistics computed correctly
4. Training loss decreases steadily

**Risk Mitigation**:
- Start with small test run (100 steps) to verify pipeline
- Monitor loss curves closely for divergence
- Keep checkpoints every 1000 steps for recovery
- Test weight loading before full training

**Next Steps After Plan Approval**:
1. Implement data conversion modifications
2. Fix value.py TODOs
3. Run normalization stats computation
4. Create unit tests
5. Execute training
6. Validate results


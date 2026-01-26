# Value Training Test Workflow for pi05_droid_100_value

## Overview
This document outlines the complete workflow for testing the value training implementation using the `pi05_droid_100_value` configuration.

## Configuration Details

From `src/openpi/training/config.py` (lines 1008-1033):
- **Model**: ValueConfig with Gemma-3-270m backbone
- **Dataset**: SummerZhang/droid_100 (LeRobot format)
- **Value Range**: Vmin=-199.0, Vmax=0.0
- **Batch Size**: 2
- **Training Steps**: 20,000
- **Weight Loader**: NoOpWeightLoader (training from scratch for testing)

## Test Workflow Phases

### Phase 1: Unit Tests ✓
**Objective**: Verify core model and data conversion functionality

#### 1.1 Value Model Unit Tests
```bash
pytest src/openpi/models/value_test.py -v -s
```
**Expected Results**:
- ✓ Model initialization with Gemma-3-270M
- ✓ num_value_bins = 201
- ✓ Vmin = -199.0, Vmax = 0.0
- ✓ Forward pass produces finite loss values

#### 1.2 Data Conversion Unit Tests
```bash
pytest examples/droid/convert_droid_data_to_lerobot_test.py -v
```
**Expected Results**:
- ✓ Value targets range from -(episode_length-1) to 0
- ✓ Values increase monotonically over episode
- ✓ Reward initialized to 0.0

---

### Phase 2: Data Pipeline Tests ✓
**Objective**: Verify dataset has required fields and normalization stats

#### 2.1 Dataset Verification
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

#### 2.2 Normalization Statistics
```bash
# Check if norm stats exist
test -f ~/.cache/openpi/assets/droid/norm_stats.json && echo "✓ Norm stats exist" || echo "✗ Need to compute norm stats"

# If norm stats don't exist, compute them
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
```

---

### Phase 3: Model Tests ✓
**Objective**: Verify model initialization and forward pass

#### 3.1 Value Model Initialization Test
```bash
python test_value_gemma.py
```
**Expected Results**:
- ✓ Model type: Value
- ✓ num_value_bins = 201
- ✓ Value range: [-199.0, 0.0]
- ✓ valuegemma_width = 640
- ✓ Prefix embedding successful
- ✓ Forward pass successful with compute_loss
- ✓ Loss values are finite

---

### Phase 4: Integration Tests ✓
**Objective**: Verify data loader and training pipeline integration

#### 4.1 Data Loader Test
```bash
python -c "
from openpi.training.config import get_config
from openpi.training.data_loader import create_data_loader

config = get_config('pi05_droid_100_value')
print(f'Config loaded: {config.name}')
print(f'Model type: {config.model.model_type}')
print(f'Batch size: {config.batch_size}')

# Test data loader
data_loader = create_data_loader(config, shuffle=False, num_batches=1)
print('✓ Data loader created')

for batch in data_loader:
    print(f'✓ Batch loaded, length: {len(batch)}')
    if len(batch) == 3:
        observation, actions, value_targets = batch
        print('✓ Data loader yields 3-tuple (observation, actions, value_targets)')
        print(f'✓ Observation type: {type(observation).__name__}')
        print(f'✓ Actions shape: {actions.shape}')
        print(f'✓ Value targets shape: {value_targets.shape}')
    else:
        print(f'✗ Expected 3-tuple, got {len(batch)}-tuple')
    break
"
```

---

### Phase 5: Training Execution 🚀
**Objective**: Run actual training with monitoring

#### 5.1 Short Test Run (100 steps)
```bash
# Quick smoke test
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=test_workflow_short \
    --overwrite
```
**Monitoring**:
- Watch for loss values in terminal
- Check W&B dashboard (if enabled)
- Verify no NaN/Inf values

#### 5.2 Full Training Run (20,000 steps)
```bash
# Full training run
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value.py \
    pi05_droid_100_value \
    --exp-name=value_training_full \
    --overwrite
```

**Expected Checkpoints**:
- Saved at: `./checkpoints/pi05_droid_100_value/value_training_full/`
- Frequency: Every 1000 steps
- Keep: Every 5000 steps permanently

---

## Success Criteria

### Unit Tests
- [ ] All value_test.py tests pass
- [ ] All convert_droid_data_to_lerobot_test.py tests pass

### Data Pipeline
- [ ] Dataset contains state_value and reward fields
- [ ] Normalization stats computed and saved
- [ ] value_targets stats are reasonable (mean ≈ -100, std ≈ 58)

### Model Tests
- [ ] test_value_gemma.py completes successfully
- [ ] Model initializes with correct parameters
- [ ] Forward pass produces finite loss values

### Integration
- [ ] Data loader yields correct 3-tuple format
- [ ] Observation, actions, and value_targets have correct shapes

### Training
- [ ] Training starts without errors
- [ ] Loss values are logged and finite
- [ ] Checkpoints are saved at intervals
- [ ] No memory errors or crashes

---

## Troubleshooting

### Common Issues

1. **Missing norm stats**
   ```bash
   uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
   ```

2. **CUDA out of memory**
   - Reduce batch_size in config
   - Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`

3. **Dataset not found**
   ```bash
   # Verify dataset exists
   python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('SummerZhang/droid_100'); print(f'Dataset loaded: {len(ds)} samples')"
   ```

4. **Model path not found**
   - Ensure `/root/autodl-tmp/gemma-3-270m` exists
   - Or download: `bash download_gemma_checkpoint.sh`

---

## Execution Checklist

Use this checklist to track progress:

- [ ] Phase 1.1: Run value model unit tests
- [ ] Phase 1.2: Run data conversion unit tests
- [ ] Phase 2.1: Verify dataset fields
- [ ] Phase 2.2: Compute/verify normalization stats
- [ ] Phase 3.1: Run value model initialization test
- [ ] Phase 4.1: Test data loader integration
- [ ] Phase 5.1: Short test run (100 steps)
- [ ] Phase 5.2: Full training run (20,000 steps)
- [ ] Verify final checkpoints saved
- [ ] Review training metrics on W&B

---

## Expected Timeline

- **Phase 1-2**: ~10 minutes (unit tests + data verification)
- **Phase 3**: ~5 minutes (model tests)
- **Phase 4**: ~5 minutes (integration tests)
- **Phase 5.1**: ~10-15 minutes (short test run)
- **Phase 5.2**: ~8-12 hours (full training run)

**Total Time**: ~10-15 hours for complete workflow

---

## Output Artifacts

1. **Test Results**: Console output from pytest and python tests
2. **Norm Stats**: `~/.cache/openpi/assets/droid/norm_stats.json`
3. **Checkpoints**: `./checkpoints/pi05_droid_100_value/value_training_full/`
4. **Training Logs**: W&B dashboard or terminal logs
5. **This Document**: Progress tracking and results

---

## Next Steps After Training

1. Evaluate model on validation set
2. Analyze value predictions vs ground truth
3. Visualize value distributions
4. Compare with baseline models
5. Fine-tune hyperparameters if needed


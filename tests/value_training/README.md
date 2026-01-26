# Value Function Training - Test Suite

This directory contains all unit and integration tests for the value function training implementation on the DROID dataset.

## Test Structure

```
tests/value_training/
├── data_pipeline/          # Data conversion and preprocessing tests
│   ├── compute_norm_stats_test.py
│   └── convert_droid_data_to_lerobot_test.py
├── models/                 # Model architecture tests
│   └── value_test.py
└── training/               # Training pipeline tests
    └── data_loader_test.py
```

## Test Categories

### 1. Data Pipeline Tests (`data_pipeline/`)

**`convert_droid_data_to_lerobot_test.py`**
- Tests value target computation logic
- Validates value computation formula: V(t) = -1 * (episode_length - t - 1)
- Tests monotonicity property (values increase over episode)
- Validates data types and array shapes

**Tests included:**
- `test_value_target_computation()` - Verify value targets computed correctly
- `test_value_target_monotonicity()` - Test values increase monotonically
- `test_value_target_different_episode_lengths()` - Test various episode lengths
- `test_reward_initialization()` - Test reward initialization
- `test_value_target_array_format()` - Test numpy array format

**Run command:**
```bash
pytest tests/value_training/data_pipeline/convert_droid_data_to_lerobot_test.py -v
```

---

**`compute_norm_stats_test.py`**
- Tests normalization statistics computation
- Verifies value_targets and reward statistics are computed correctly
- Validates statistics are finite and in reasonable ranges

**Tests included:**
- `test_norm_stats_file_exists()` - Verify norm_stats.json exists
- `test_norm_stats_contains_value_targets()` - Check value_targets stats present
- `test_norm_stats_contains_reward()` - Check reward stats present
- `test_norm_stats_values_are_finite()` - Verify no NaN/Inf values
- `test_value_targets_stats_reasonable()` - Validate value ranges
- `test_reward_stats_reasonable()` - Validate reward stats (zeros for DROID)
- `test_norm_stats_contains_all_required_features()` - Check all features present
- `test_statistics_computation_logic()` - Unit test (no dataset required)

**Run command:**
```bash
# Run all tests
pytest tests/value_training/data_pipeline/compute_norm_stats_test.py -v -s

# Or run directly
python tests/value_training/data_pipeline/compute_norm_stats_test.py
```

**Prerequisites:** Must run `compute_norm_stats.py` first:
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
```

### 2. Model Tests (`models/`)

**`value_test.py`**
- Tests Value model initialization with Gemma-3-270M
- Tests forward pass and loss computation
- Tests distributional value head functionality

**Tests included:**
- `test_value_model_initialization()` - Verify model initializes correctly
- `test_value_forward_pass()` - Test forward pass produces correct output shapes
- `test_distributional_value_head()` - Test value head produces valid distributions
- `test_value_head_support_atoms()` - Test support atoms correctly initialized

**Run command:**
```bash
pytest tests/value_training/models/value_test.py -v -s
```

**Prerequisites:** Requires Gemma-3-270M checkpoint at `/root/autodl-tmp/gemma-3-270m`

### 3. Training Pipeline Tests (`training/`)

**`data_loader_test.py`**
- Tests data loader functionality for both policy and value training
- Verifies correct batch format with value_targets
- Validates batch consistency across all fields

**Value training specific tests:**
- `test_value_data_loader_yields_three_tuple()` - Verify 3-tuple output
- `test_value_targets_shape_and_type()` - Check shape and dtype
- `test_value_targets_are_finite()` - Verify no NaN/Inf
- `test_value_batch_consistency()` - Check batch dimension consistency

**Run command:**
```bash
# Run all data loader tests
pytest tests/value_training/training/data_loader_test.py -v -s

# Run only value training tests
pytest tests/value_training/training/data_loader_test.py::test_value_data_loader_yields_three_tuple -v -s
pytest tests/value_training/training/data_loader_test.py::test_value_targets_shape_and_type -v -s
pytest tests/value_training/training/data_loader_test.py::test_value_targets_are_finite -v -s
pytest tests/value_training/training/data_loader_test.py::test_value_batch_consistency -v -s
```

**Prerequisites:** Requires converted DROID dataset and norm_stats.json

## Running All Tests

### Run all value training tests:
```bash
pytest tests/value_training/ -v -s
```

### Run tests by category:
```bash
# Data pipeline tests
pytest tests/value_training/data_pipeline/ -v -s

# Model tests
pytest tests/value_training/models/ -v -s

# Training tests
pytest tests/value_training/training/ -v -s
```

### Run specific test:
```bash
pytest tests/value_training/models/value_test.py::test_value_model_initialization -v -s
```

## Test Execution Order

For comprehensive verification, execute tests in this order:

**Phase 1: Unit Tests (No Dataset Required)**
```bash
# 1. Data conversion logic tests
pytest tests/value_training/data_pipeline/convert_droid_data_to_lerobot_test.py -v

# 2. Statistics computation logic test
pytest tests/value_training/data_pipeline/compute_norm_stats_test.py::test_statistics_computation_logic -v
```

**Phase 2: Data Pipeline Tests (Requires Dataset)**
```bash
# 3. Normalization statistics tests (requires norm_stats.json)
pytest tests/value_training/data_pipeline/compute_norm_stats_test.py -v -s

# 4. Data loader tests (requires dataset + norm_stats.json)
pytest tests/value_training/training/data_loader_test.py -v -s
```

**Phase 3: Model Tests (Requires Gemma Checkpoint)**
```bash
# 5. Value model tests
pytest tests/value_training/models/value_test.py -v -s
```

## Prerequisites

Before running tests, ensure you have:

1. **Converted DROID Dataset**:
   - Dataset with state_value and reward fields
   - Available on HuggingFace (e.g., SummerZhang/droid_100)

2. **Normalization Statistics**:
   ```bash
   uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
   ```
   - Creates `~/.cache/openpi/assets/droid/norm_stats.json`

3. **Gemma-3-270M Checkpoint**:
   - Downloaded to `/root/autodl-tmp/gemma-3-270m`

4. **Environment Setup**:
   ```bash
   uv sync
   uv pip install -e .
   ```

## Test Markers

Tests use pytest markers to indicate requirements:

- `@pytest.mark.manual` - Tests requiring manual execution (dataset, checkpoints, etc.)
- No marker - Unit tests that can run without external dependencies

## Expected Test Results

### Data Pipeline Tests
- ✓ Value targets range from -(episode_length-1) to 0
- ✓ Values increase monotonically over episode
- ✓ Normalization stats contain all required features
- ✓ All statistics are finite (no NaN/Inf)

### Model Tests
- ✓ Model initializes with num_value_bins=201, Vmin=-199.0, Vmax=0.0
- ✓ Forward pass produces loss with shape (batch_size,)
- ✓ Loss values are finite
- ✓ Value predictions within [Vmin, Vmax] range

### Training Tests
- ✓ Data loader yields 3-tuple (observation, actions, value_targets)
- ✓ Value targets have shape (batch_size,) and dtype float32
- ✓ All batch dimensions are consistent
- ✓ No NaN or Inf values in value_targets

## Troubleshooting

**Issue: "Missing norm_stats.json"**
```bash
# Solution: Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name pi05_droid_100_value
```

**Issue: "Dataset not found"**
```bash
# Solution: Verify dataset is available on HuggingFace
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('SummerZhang/droid_100'); print('Dataset OK')"
```

**Issue: "Gemma checkpoint not found"**
```bash
# Solution: Download Gemma-3-270M checkpoint
# Follow instructions in project documentation
```

## Related Documentation

- **Implementation Plan**: `/root/autodl-tmp/openpi/.claude/plans/ticklish-meandering-aho.md`
- **Test Documentation**: `/root/autodl-tmp/openpi/test_func.md`
- **Code Modifications**: `/root/autodl-tmp/openpi/CODE_MODIFICATIONS_SUMMARY.md`
- **Value Training Guide**: `/root/autodl-tmp/openpi/VALUE_TRAINING.md`

## Contact

For issues or questions about these tests, refer to the project documentation or create an issue in the repository.

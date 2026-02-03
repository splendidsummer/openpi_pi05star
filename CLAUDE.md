# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

openpi is Physical Intelligence's open-source repository for vision-language-action (VLA) models for robotics. It contains three model families:
- **π₀**: Flow-based VLA model
- **π₀-FAST**: Autoregressive VLA with FAST action tokenizer
- **π₀.₅**: Upgraded π₀ with better generalization via knowledge insulation

The repository uses JAX for all model implementations, with fine-tuning capabilities for custom robot platforms.

## Development Rules

**IMPORTANT: When working with this codebase, you MUST follow these rules:**

1. **JAX-Only Implementation**: All model modifications and implementations must use JAX. Do NOT create or modify PyTorch implementations. Focus exclusively on the JAX models in `src/openpi/models/`.

2. **Mandatory Unit Testing**: For ALL main functionality and model modifications, you MUST:
   - Write comprehensive unit tests using pytest
   - Co-locate test files with source files (e.g., `model_test.py` next to `model.py`)
   - Test critical functionality including forward passes, shape validation, and edge cases
   - Run tests before considering any implementation complete
   - Use appropriate pytest markers (e.g., `@pytest.mark.manual` for tests requiring manual execution)

## Development Commands

### Environment Setup
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
git submodule update --init --recursive

# Install dependencies with uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Cache Configuration:**
To avoid filling the home directory cache, set the following environment variables before running any commands:
```bash
export HF_LEROBOT_HOME=/root/autodl-tmp/huggingface
export HF_HOME=/root/autodl-tmp/huggingface
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
```

- `HF_LEROBOT_HOME=/root/autodl-tmp/huggingface` – LeRobot datasets cache here
- `HF_HOME=/root/autodl-tmp/huggingface` – HuggingFace general cache
- `OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache` – openpi assets cache

### Training

**JAX Training:**
```bash
# Compute normalization statistics (required before training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Run training (use XLA_PYTHON_CLIENT_MEM_FRACTION for max GPU memory)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name>

# Resume training
uv run scripts/train.py <config_name> --exp-name=<experiment_name> --resume

# Overwrite existing checkpoints
uv run scripts/train.py <config_name> --exp-name=<experiment_name> --overwrite
```

**PyTorch Training:**
```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Resume training
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --resume
```

## Architecture

### Core Components

**Models (`src/openpi/models/`):**
- `model.py`: Base model abstractions (`BaseModel`, `BaseModelConfig`, `Observation`, `Actions`)
- `pi0.py`: Flow-based π₀ model implementation
- `pi0_fast.py`: Autoregressive π₀-FAST model
- `gemma.py` / `gemma_fast.py`: Gemma backbone implementations
- `siglip.py`: Vision encoder
- `tokenizer.py`: Text tokenization
- `lora.py`: LoRA (Low-Rank Adaptation) support

**Policies (`src/openpi/policies/`):**
- `policy.py`: Base `Policy` class that wraps models for inference
- `droid_policy.py`: DROID robot platform-specific I/O mappings
- `aloha_policy.py`: ALOHA robot platform-specific I/O mappings
- `libero_policy.py`: LIBERO benchmark-specific I/O mappings
- `policy_config.py`: Policy creation utilities

**Training (`src/openpi/training/`):**
- `config.py`: Training configurations for all models and datasets (see `_CONFIGS` dict)
- `data_loader.py`: LeRobot dataset loading and batching
- `droid_rlds_dataset.py`: DROID RLDS dataset handling
- `weight_loaders.py`: Checkpoint loading from base models
- `optimizer.py`: Optimizer configuration
- `checkpoints.py`: Checkpoint saving/loading

**Transforms (`src/openpi/transforms.py`):**
Data transformation pipeline with three stages:
1. `repack_transforms`: Dataset-specific format conversion
2. `data_transforms`: Robot-specific transformations (applied before normalization)
3. `model_transforms`: Model-specific transformations (applied after normalization)

**PyTorch Models (`src/openpi/models_pytorch/`):**
- PyTorch implementations of π₀ and π₀.₅
- `transformers_replace/`: Patches for HuggingFace transformers library (AdaRMS, precision control, KV cache)

### Data Flow

1. **Training Pipeline (DROID Dataset via LeRobot):**

   ```
   LeRobot DROID Dataset (e.g., droid_100)
   ↓
   Load from HuggingFace with action chunking
   ↓
   repack_transforms (DroidInputs → standard format)
   ↓
   data_transforms (resize images, pad states/actions)
   ↓
   normalize (apply norm_stats.json)
   ↓
   model_transforms (tokenize prompt)
   ↓
   Observation/Actions Construction
   ↓
   Image Processing (SigLIP Vision Encoder)
   ↓
   Prompt Tokenization (PaliGemma Tokenizer)
   ↓
   State Input Processing (normalization + padding)
   ↓
   Feature Fusion (π₀.₅: AdaRMS conditioning)
   ↓
   Model Forward Pass (Flow Matching Loss)
   ```

   **Detailed Steps:**

   a. **LeRobot DROID Dataset Loading** (`data_loader.py` lines 130-151):
      - Load from HuggingFace repository (e.g., "SummerZhang/droid_100")
      - Uses `lerobot_dataset.LeRobotDataset` with action chunking
      - Action horizon: configurable (default 50 for π₀.₅)
      - Delta timestamps: `[0/fps, 1/fps, ..., (action_horizon-1)/fps]`
      - Dataset format per sample:
        * `observation/exterior_image_1_left`: exterior camera [H, W, 3]
        * `observation/wrist_image_left`: wrist camera [H, W, 3]
        * `observation/joint_position`: 7-DOF joint positions
        * `observation/gripper_position`: 1-DOF gripper position
        * `actions`: action sequence [action_horizon, action_dim]
        * `prompt`: language instruction string
      - Optional: Convert task names to prompts via `PromptFromLeRobotTask`

   b. **repack_transforms** (`droid_policy.py` lines 35-101):
      - Convert DROID LeRobot format to standard observation format
      - Map camera names:
        * `observation/exterior_image_1_left` → `image.base_0_rgb`
        * `observation/wrist_image_left` → `image.left_wrist_0_rgb`
        * Add padding image for `image.right_wrist_0_rgb` (masked out)
      - State composition: concatenate [joint_position(7), gripper_position(1)] → [8]
      - Actions: extract from `actions` field → [action_horizon, 8]
      - Create image_masks: track which images are valid (base and left_wrist are True, right_wrist is False)
      - Extract prompt string from `prompt` field

   c. **data_transforms** (`transforms.py`):
      - Robot-specific preprocessing applied before normalization
      - `ResizeImages`: resize all images to 224×224 (SigLIP input size)
        * Uses PIL.Image.LANCZOS resampling for high quality
        * Converts to float32 in range [0, 255]
      - `PadStatesAndActions`: pad state and actions to model action_dim
        * State: [8] → [action_dim] (e.g., [8] → [32] for π₀.₅)
        * Actions: [action_horizon, 8] → [action_horizon, action_dim]
        * Padding with zeros on the right

   d. **normalize** (`transforms.py` lines 200-250):
      - Apply normalization statistics from `norm_stats.json`
      - Computed via `scripts/compute_norm_stats.py` before training
      - Two normalization modes:

        **Z-score normalization** (default):
        ```
        normalized = (value - mean) / (std + 1e-6)
        ```

        **Quantile normalization** (use_quantile_norm=True):
        ```
        normalized = (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        # Maps to [-1, 1] range using 1st and 99th percentiles
        ```

      - Applied to:
        * State: [action_dim] → normalized state
        * Actions: [action_horizon, action_dim] → normalized actions
      - Ensures stable training dynamics and prevents gradient explosion

   e. **model_transforms** (`transforms.py`):
      - Model-specific transformations applied after normalization
      - `TokenizePrompt`: tokenize language instruction
        * Uses PaliGemma tokenizer (SentencePiece)
        * Max token length: 48 (configurable)
        * Output: `tokenized_prompt` [max_len], `tokenized_prompt_mask` [max_len]
        * For π₀.₅ with discrete state: prepends state to prompt
      - Image normalization: convert [0, 255] → [-1, 1] range
      - No action chunking needed (already done by LeRobot dataset loader)

   f. **Observation/Actions Construction** (`model.py`):
      - Create structured `Observation` dataclass from transformed data:
        * `images`: Dict[str, Array] - {"base_0_rgb": [batch, 224, 224, 3], "left_wrist_0_rgb": [...], ...}
        * `image_masks`: Dict[str, Array] - {"base_0_rgb": [batch], ...} (True if valid)
        * `state`: Array [batch, action_dim] - normalized and padded state
        * `tokenized_prompt`: Array [batch, max_token_len] - token IDs
        * `tokenized_prompt_mask`: Array [batch, max_token_len] - validity mask
      - Create `Actions` array: [batch, action_horizon, action_dim]
      - All arrays are float32 or int32, ready for model input

   g. **Image Processing via SigLIP Vision Encoder** (`siglip.py` lines 208-290):
      - SigLIP variant: "So400m/14" (400M parameters, 14-pixel patch size)
      - Input: images [batch, 224, 224, 3] in range [-1, 1]

      **Processing steps:**
      1. **Patch Extraction**:
         - Conv2D with kernel_size=14×14, stride=14
         - Reshape to [batch, num_patches, 1152] where num_patches = (224/14)² = 256

      2. **Positional Embeddings**:
         - Add learned positional embeddings [1, 256, 1152]
         - Broadcast across batch dimension

      3. **Transformer Encoding**:
         - 27 encoder blocks with Multi-Head Self-Attention (MHSA) + MLP
         - 16 attention heads, 1152 hidden dimensions
         - Layer normalization, dropout, residual connections
         - Scan mode enabled for memory efficiency

      4. **Output**:
         - Image tokens: [batch, 256, 1152] per image
         - Each 14×14 patch becomes one token with 1152-dim features
         - Multiple camera views processed independently, then concatenated
         - For DROID: base_0_rgb (256 tokens) + left_wrist_0_rgb (256 tokens) = 512 image tokens total

   h. **Prompt Tokenization via PaliGemma Tokenizer** (`tokenizer.py` lines 27-61):
      - Uses SentencePiece tokenizer from PaliGemma
      - Downloads from `gs://big_vision/paligemma_tokenizer.model`
      - Max token length: 48 (configurable)

      **Tokenization process:**
      1. **Text Preprocessing**:
         - Strip whitespace
         - Replace underscores with spaces
         - Remove newlines

      2. **Encoding**:
         - Encode with SentencePiece (add_bos=True for beginning-of-sequence token)
         - Append "\n" token (start-of-answer marker)

      3. **Padding/Truncation**:
         - Pad to max_len with padding tokens
         - Truncate if exceeds max_len

      4. **Output**:
         - `tokens`: int32[max_len] - token IDs
         - `token_mask`: bool[max_len] - True for valid tokens, False for padding

      **For π₀.₅ with discrete state input**:
      - Discretize state: `np.digitize(state, bins=linspace(-1, 1, 257))` → 256 bins
      - Format: `"Task: {prompt}, State: {state_str};\nAction: "`
      - Encodes both task description and current state as text

   i. **Feature Fusion via Attention Masking** (`pi0.py` lines 106-186):

      The model fuses different feature types using attention masking controlled by `ar_mask` (autoregressive mask).

      **ar_mask Assignment (from actual implementation):**

      ```python
      # embed_prefix (lines 106-137):
      # Images: ar_mask = False (bidirectional)
      ar_mask += [False] * num_image_tokens  # Line 125

      # Language: ar_mask = False (bidirectional)
      ar_mask += [False] * num_language_tokens  # Line 133
      # Comment: "full attention between image and language inputs"

      # embed_suffix (lines 140-186):
      # State (π₀ only): ar_mask = True (causal)
      ar_mask += [True]  # Line 157

      # Actions: ar_mask = [True, False, False, ...]
      ar_mask += [True] + ([False] * (action_horizon - 1))  # Line 182
      ```

      **What ar_mask means:**
      - `ar_mask=False`: Bidirectional attention (token can attend to all other bidirectional tokens)
      - `ar_mask=True`: Causal attention (token can only attend to previous tokens + all bidirectional tokens)

      **Concrete Example - Token Sequence for π₀:**

      ```
      Sequence: [Img(512 tokens), Lang(48 tokens), State(1 token), Action(50 tokens)]

      ar_mask:  [False×512,        False×48,        True×1,         True, False×49]

      cumsum:   [0×512,            0×48,            1×1,            2, 2×49]
      ```

      **Attention Mask Construction** (`pi0.py` lines 19-44):
      ```python
      cumsum = cumsum(ar_mask)  # Cumulative sum of ar_mask
      # Token i can attend to token j if cumsum[i] >= cumsum[j]
      attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
      ```

      **Resulting Attention Pattern:**

      ```
      Query\Key        Images(512)  Language(48)  State(1)  Actions(50)
      Images(512)         ✓             ✓            ✗          ✗
      Language(48)        ✓             ✓            ✗          ✗
      State(1)            ✓             ✓            ✓          ✗
      Actions(50)         ✓             ✓            ✓          ✓
      ```

      - **Images & Language** (cumsum=0): Full bidirectional attention to each other
      - **State** (cumsum=1): Attends to images, language, and itself (causal)
      - **Actions** (cumsum=2): Attend to images, language, state, and each other (causal within actions)

      **Key Insight:**
      - Prefix tokens (images + language) form a bidirectional context
      - Suffix tokens (state + actions) can see the prefix but not vice versa
      - This prevents visual/linguistic features from being contaminated by action predictions

   j. **State Input Processing**:

      **For π₀ models (continuous state token):**
      - State: [batch, action_dim] normalized float32
      - Project to embedding: `state_proj(state)` → [batch, 1, 1152]
      - Added as single token to suffix sequence
      - ar_mask=True (causal attention)

      **For π₀.₅ models (discrete state in prompt):**
      - State discretized to 256 bins and encoded as text
      - Integrated into prompt: "Task: {task}, State: {state};\nAction: "
      - No separate state token in model input
      - Enables better knowledge insulation and generalization

   k. **Model Forward Pass** (`pi0.py` lines 189-214):

      1. **Preprocess observation**: Apply data augmentation if training
      2. **Sample noise and timestep**: For flow matching training
         - Noise: `ε ~ N(0, I)` with shape [batch, action_horizon, action_dim]
         - Time: `t ~ Beta(1.5, 1)` scaled to [0.001, 0.999]
         - Noisy actions: `x_t = t·ε + (1-t)·actions`
      3. **Embed prefix**: Images + language → [batch, 560, 1152]
      4. **Embed suffix**: State + actions → [batch, 51, 1152] (π₀) or [batch, 50, 1152] (π₀.₅)
      5. **Construct attention mask**: Based on ar_mask values
      6. **Forward through PaliGemma LLM**:
         - Input: concatenated prefix + suffix tokens
         - Attention mask: controls information flow
         - AdaRMS conditioning (π₀.₅ only): time embedding modulates layer norms
      7. **Extract action predictions**: From suffix output tokens
      8. **Compute flow matching loss**: MSE between predicted and actual velocity
         - Loss: `mean((v_t - u_t)²)` where `u_t = actions - ε`
      9. **Backpropagation**: Compute gradients and update parameters

      **π₀ vs π₀.₅ Differences:**
      - **π₀**: Continuous state token + time-action MLP fusion
      - **π₀.₅**: Discrete state in prompt + AdaRMS time conditioning
      - **π₀.₅** achieves better generalization through knowledge insulation

2. **Inference Pipeline:**
   ```
   Raw Observation → input_transforms → Observation → Model.sample_actions → output_transforms → Actions
   ```

### Key Abstractions

**Observation:** Structured input to models containing:
- `images`: Dict of camera views (e.g., "base_0_rgb", "left_wrist_0_rgb")
- `image_masks`: Validity masks for images
- `state`: Low-dimensional robot state
- `tokenized_prompt`: Optional language instruction
- Model-specific fields (e.g., `token_ar_mask` for π₀-FAST)

**Actions:** Float array with shape `[batch, action_horizon, action_dim]`

**Policy:** Wraps a model with transforms for robot-specific inference. Handles:
- Input/output transformations
- Batching/unbatching
- JAX/PyTorch model dispatch
- Timing metrics

### Configuration System

Training configs are defined in `src/openpi/training/config.py` in the `_CONFIGS` dictionary. Each config specifies:
- `model`: Model architecture and hyperparameters
- `data`: Dataset configuration and transforms
- `optimizer`: Learning rate, weight decay, etc.
- `weight_loader`: How to initialize from base model checkpoints

Available configs include: `pi0_base`, `pi0_fast_base`, `pi05_base`, `pi0_droid`, `pi0_fast_droid`, `pi05_droid`, `pi05_libero`, `pi0_aloha_sim`, etc.

## Important Patterns

### Adding Support for a New Robot Platform

1. Create policy class in `src/openpi/policies/` (e.g., `my_robot_policy.py`):
   - Define `MyRobotInputs` and `MyRobotOutputs` dataclasses
   - Implement `repack_transforms` to map robot observations to standard format
   - Implement `data_transforms` for robot-specific preprocessing

2. Create training config in `src/openpi/training/config.py`:
   - Define `DataConfig` with your LeRobot dataset repo_id
   - Specify transforms from your policy class
   - Configure weight loader to initialize from base model

3. Compute normalization stats:
   ```bash
   uv run scripts/compute_norm_stats.py --config-name my_robot_config
   ```

4. Run training:
   ```bash
   uv run scripts/train.py my_robot_config --exp-name=my_experiment
   ```

### PyTorch Setup Requirements

When using PyTorch models, you must apply transformers library patches:
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

This modifies the transformers library to support AdaRMS normalization, precision control, and KV cache behavior needed by openpi models.

### Normalization Statistics

Normalization stats are computed per-dataset and stored in `norm_stats.json` within checkpoint directories. They can be reloaded from base model checkpoints during fine-tuning using `AssetsConfig` in the training config.

### Memory Management

- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use 90% of GPU memory (default is 75%)
- Use `--fsdp-devices <n>` for fully-sharded data parallelism to reduce per-GPU memory
- Disable EMA if running out of memory during training

## Common Issues

- **Dependency conflicts during `uv sync`**: Remove `.venv` and run `uv sync` again
- **Missing norm stats**: Run `scripts/compute_norm_stats.py` before training
- **CUDA errors**: System CUDA libraries can conflict with uv-installed CUDA; try uninstalling system CUDA
- **Diverging training loss**: Check `norm_stats.json` for dimensions with very small `q01`, `q99`, or `std` values
- **Import errors**: Ensure `GIT_LFS_SKIP_SMUDGE=1` was set during installation (needed for LeRobot dependency)

## Checkpoints

Base and fine-tuned model checkpoints are stored in Google Cloud Storage at `gs://openpi-assets/checkpoints/`. They are automatically downloaded and cached in `~/.cache/openpi` (override with `OPENPI_DATA_HOME` environment variable).

## Testing Philosophy

- Tests use pytest with markers (e.g., `@pytest.mark.manual` for tests requiring manual execution)
- Test files are co-located with source files (e.g., `model_test.py` next to `model.py`)
- Test paths configured in `pyproject.toml`: `["src", "scripts", "packages"]`

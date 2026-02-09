import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.value import Value


@dataclasses.dataclass(frozen=True)
class ValueConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_3_270m"
    # TODO: tune this value accordingly
    num_value_bins: int = 201  
    model_path: str = "path/to/gemma3_270m/model"
    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = True
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    
    Vmin: float = -199.0
    Vmax: float = 0.0

    @override
    @property
    def model_type(self) -> _model.ModelType:
        """The model type."""
        return _model.ModelType.VALUE

    @override
    def create(self, rng: at.KeyArrayLike) -> "Value":
        from openpi.models.value import Value

        return Value(self, rngs=nnx.Rngs(rng))
    
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        # Check if model_path exists (skip if empty string or None)
        if self.model_path and self.model_path.strip():
            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"the model path does not exist: {self.model_path}. Please check the path is correct."
                )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(self, unfreeze_last_n_layers: int = 3, use_lora: bool = False) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.
        
        For value model training:
        - SigLIP vision encoder: Freeze most layers, unfreeze last N layers OR use LoRA
        - LLM: Fully trainable (not frozen) - lr = 1e-5
        - Value head: Fully trainable (not frozen)
        
        Args:
            unfreeze_last_n_layers: Number of last SigLIP encoder layers to unfreeze (default: 3)
                                    Set to 0 to freeze all SigLIP layers
                                    IMPORTANT: Since SigLIP uses scan=True (parameter sharing),
                                    all layers share the same parameters. Therefore, if 
                                    unfreeze_last_n_layers > 0, we unfreeze ALL SigLIP parameters.
                                    To truly freeze first layers and unfreeze last layers only,
                                    you would need to set scan=False in the model config.
            use_lora: If True, freeze SigLIP base params but allow LoRA params to train
                     (LoRA implementation would need to be added separately)
        
        Returns:
            Filter that freezes SigLIP parameters (except last N layers or LoRA params),
            while keeping LLM and value_head trainable.
        """
        # Handle SigLIP vision encoder freezing
        if use_lora:
            # Freeze base SigLIP params, but allow LoRA params to train
            siglip_base_params = nnx_utils.PathRegex(".*ValueGemma/img/.*")
            lora_params = nnx_utils.PathRegex(".*ValueGemma/img/.*lora.*")
            # Freeze SigLIP params that are NOT LoRA params
            return nnx.All(siglip_base_params, nnx.Not(lora_params))
        elif unfreeze_last_n_layers <= 0:
            # Freeze all SigLIP layers
            return nnx_utils.PathRegex(".*ValueGemma/img/.*")
        else:
            # Unfreeze SigLIP (either all layers or last N layers)
            # Note: Due to scan=True, all encoder layers share parameters, so we cannot
            # selectively unfreeze specific layers. Setting unfreeze_last_n_layers > 0
            # means we unfreeze ALL SigLIP parameters.
            # To freeze first layers and only unfreeze last N layers, you would need:
            # 1. Set scan=False in Value model config, OR
            # 2. Implement LoRA for SigLIP attention layers
            
            # For now, if unfreeze_last_n_layers > 0, don't freeze SigLIP
            # (all parameters will be trainable)
            return nnx.Nothing


@dataclasses.dataclass(frozen=True)
class ValueInferenceConfig(ValueConfig):
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "Value":
        from openpi.models.value import Value

        return Value(self, rngs=nnx.Rngs(rng))
    
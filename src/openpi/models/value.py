import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.value_config as value_config
# import openpi.models.gemma as _gemma  # Not using openpi gemma model per user request
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
# Ensure we import Google's gemma library, not local gemma.py
import sys
if 'gemma' in sys.modules:
    # Check if it's the local gemma module
    module = sys.modules['gemma']
    if hasattr(module, '__file__') and 'src/openpi/models' in module.__file__:
        del sys.modules['gemma']
import gemma.gm as gm
from openpi.models.gemma_utils import build_gemma_3_270m_model, get_token_embeddings

logger = logging.getLogger("openpi")


from flax import nnx
import jax.numpy as jnp


class DistributionalValueHeadGemma3(nnx.Module):
    """
    针对 Gemma 3 优化的分布式价值头 (Flax NNX 版)
    - B = 201 bins
    - 使用 SiLU 激活函数
    - 使用默认初始化（避免 pytree 结构不匹配）
    """
    def __init__(self, hidden_size: int = 640, 
                 # TODO: setting num_bins according to value range of state_value 
                 num_bins: int = 321,
                 v_min: float = -319.0, v_max: float = 0.0,
                 *, rngs: nnx.Rngs):
        self.num_bins = num_bins

        # 定义支持向量 (Support Vector)
        # Wrap as nnx.Variable to prevent NNX from treating it as a trainable parameter
        # This is a constant that doesn't need gradients
        self.support = nnx.Variable(jnp.linspace(v_min, v_max, num_bins))

        # 定义层 - 使用默认初始化器以避免 pytree 结构不匹配
        # NNX 的默认初始化器在 JIT 编译时保持一致的 pytree 结构
        self.head_l1 = nnx.Linear(hidden_size, 512, rngs=rngs)         # bias=True
        self.head_l2 = nnx.Linear(512, 128, rngs=rngs)                 # bias=True
        # 输出层，从 128 映射到 201 bins
        # Keep bias=True (default). If checkpoint doesn't have bias, it will be skipped during loading
        # and the bias will use its initialized values.
        self.head_l3 = nnx.Linear(128, num_bins, rngs=rngs)

    def __call__(self, x: jax.Array, return_expectation: bool = True):
        # Forward pass
        x = nnx.silu(self.head_l1(x))
        x = nnx.silu(self.head_l2(x))

        # Get logits
        logits = self.head_l3(x)

        # Compute probabilities for each bin (Softmax)
        probs = jax.nn.softmax(logits, axis=-1)

        if return_expectation:
            # Formula: V = sum(p * v(b))
            # Use dot product to compute expected value
            expectation = jnp.sum(probs * self.support.value, axis=-1)
            return expectation

        return logits


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class Value(_model.BaseModel):
    def __init__(self, config: value_config.ValueConfig, rngs: nnx.Rngs):
        super().__init__(
            config.action_dim, 
            config.action_horizon,
            config.max_token_len
            )
        self.pi05 = config.pi05
        self.model_path = config.model_path 
        self.num_value_bins = config.num_value_bins 
        # Hardcode config for gemma_3_270m (width=640) from Google DeepMind Gemma model
        self.valuegemma_width = 640  # gemma_3_270m hidden size
        self.Vmin = config.Vmin
        self.Vmax = config.Vmax
        
        # Build Gemma LLM (already wrapped with ToNNX by build_gemma_3_270m_model)
        llm = build_gemma_3_270m_model(self.model_path)

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=self.valuegemma_width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )

        # Initialize SigLIP with dummy image
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.ValueGemma = nnx.Dict(llm=llm, img=img)

        # Build value head: 3-layer MLP to project last timestep output to 201 bins
        self.value_head = DistributionalValueHeadGemma3(
            hidden_size=self.valuegemma_width,
            num_bins=self.num_value_bins,
            v_min=self.Vmin,
            v_max=self.Vmax,
            rngs=rngs
        )

    def _embed_tokens(self, token_ids: jax.Array) -> jax.Array:
        """Manually embed token IDs using the Gemma embedding table.

        Args:
            token_ids: Token IDs to embed, shape [batch, seq_len]

        Returns:
            Embeddings, shape [batch, seq_len, embed_dim]
        """
        # Access the Gemma model's state to get the embedding table
        # The embedding table is located at embedder/input_embedding in model parameters
        # Based on debug output: shape (262144, 640) bfloat16

        try:
            embeddings = get_token_embeddings(self.ValueGemma.llm, token_ids)
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Could not tokenizing the token_ids in Gemma model: {e}") 

        embed_dim = embeddings.shape[-1]
        embeddings = embeddings * jnp.sqrt(embed_dim).astype(embeddings.dtype)

        return embeddings

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.ValueGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            # Manually embed the tokenized prompt using the Gemma embedding table
            tokenized_inputs = self._embed_tokens(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    # TODO: review the loss computation according to distributional RL!! 
    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation,  *, train: bool = False,  value_targets: at.Float[at.Array, "b"] | None = None, 
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute value loss.
        
        For value model, we only use prefix (images + state), no actions needed.
        The actions parameter is kept for interface compatibility but not used.
        """
        preprocess_rng = jax.random.split(rng, 1)[0]
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Only use prefix tokens (images + language), no suffix needed
        # embed_prefix returns embeddings (3D), not token IDs (2D)
        prefix_embeddings, inputs_mask, ar_mask = self.embed_prefix(observation)
        # TODO: confirm the last input token can attend to all previous tokens 
        attn_mask = make_attn_mask(inputs_mask, ar_mask)
        positions = jnp.cumsum(inputs_mask, axis=1) - 1
        
        # Since prefix_embeddings is 3D embeddings (not token IDs), we need to call
        # the underlying model's _apply_attention method directly.
        from gemma.gm.nn._transformer import _Inputs

        # Get the underlying Flax Linen model
        underlying_model = self.ValueGemma.llm.module

        # Get model parameters from NNX state
        graphdef, state = nnx.split(self.ValueGemma.llm)
        # state = state.unfreeze()
        params_dict = state.to_pure_dict()
        variables = {'params': params_dict}

        # Create inputs for _apply_attention
        inputs = _Inputs(
            embeddings=prefix_embeddings,
            positions=positions,
            attention_mask=attn_mask,
            inputs_mask=inputs_mask,
        )

        # Call _apply_attention
        hidden_states, _ = underlying_model.apply(
            variables,
            inputs,
            None,  # cache
            method=underlying_model._apply_attention,
        )
        
        # Extract the output at the last valid position for each sequence
        last_valid_positions = jnp.sum(inputs_mask, axis=1, dtype=jnp.int32) - 1  # Shape: (batch_size,)
        batch_indices = jnp.arange(hidden_states.shape[0])
        
        last_timestep_output = hidden_states[batch_indices, last_valid_positions, :]  # Shape: (batch_size, hidden_dim)
        
        # Project through value head to get num_value_bins bins prediction
        value_logits = self.value_head(last_timestep_output, return_expectation=False)  # Shape: (batch_size, num_value_bins)
        value_probs = jax.nn.softmax(value_logits, axis=-1)  # Shape: (batch_size, num_value_bins)
        
        # Build value support atoms (bins)
        value_bins, _ = self.build_z()  # Shape: (num_value_bins,) 
        expected_value = jnp.sum(value_bins[None, :] * value_probs, axis=-1)  # Shape: (batch_size,)
        
        # Compute value loss (MSE between predicted and target values)
        loss = jnp.square(expected_value - value_targets)

        return loss

    @at.typecheck
    def build_z(self) -> tuple[at.Float[at.Array, "nb_atoms"], float]:
        """Build value support atoms for distributional value learning.

        Args:
            Vmin: Minimum value in the support
            Vmax: Maximum value in the support
            nb_atoms: Number of atoms (bins) in the support

        Returns:
            z: Array of value atoms [nb_atoms]
            dz: Step size between atoms
        """
        dz = (self.Vmax - self.Vmin) / (self.num_value_bins - 1)
        z = jnp.arange(self.Vmin, self.Vmax + dz / 2, dz, dtype=jnp.float32)
        return z, dz

    def compute_value(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, train: bool = False, value_targets: at.Float[at.Array, "b"] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute value prediction for the given observation.

        For value model, we only use prefix (images + state), no actions needed.
        The actions parameter is kept for interface compatibility but not used.
        """
        preprocess_rng = jax.random.split(rng, 1)[0]
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Only use prefix tokens (images + language), no suffix needed
        # embed_prefix returns embeddings (3D), not token IDs (2D)
        prefix_embeddings, inputs_mask, ar_mask = self.embed_prefix(observation)
        # TODO: confirm the last input token can attend to all previous tokens
        attn_mask = make_attn_mask(inputs_mask, ar_mask)
        positions = jnp.cumsum(inputs_mask, axis=1) - 1

        # Since prefix_embeddings is 3D embeddings (not token IDs), we need to call
        # the underlying model's _apply_attention method directly.
        from gemma.gm.nn._transformer import _Inputs

        # Get the underlying Flax Linen model
        underlying_model = self.ValueGemma.llm.module

        # Get model parameters from NNX state
        graphdef, state = nnx.split(self.ValueGemma.llm)
        # state = state.unfreeze()
        params_dict = state.to_pure_dict()
        variables = {'params': params_dict}

        # Create inputs for _apply_attention
        inputs = _Inputs(
            embeddings=prefix_embeddings,
            positions=positions,
            attention_mask=attn_mask,
            inputs_mask=inputs_mask,
        )

        # Call _apply_attention
        hidden_states, _ = underlying_model.apply(
            variables,
            inputs,
            None,  # cache
            method=underlying_model._apply_attention,
        )

        # Extract the output at the last valid position for each sequence
        last_valid_positions = jnp.sum(inputs_mask, axis=1, dtype=jnp.int32) - 1  # Shape: (batch_size,)
        batch_indices = jnp.arange(hidden_states.shape[0])

        last_timestep_output = hidden_states[batch_indices, last_valid_positions, :]  # Shape: (batch_size, hidden_dim)

        # Project through value head to get num_value_bins bins prediction
        value_logits = self.value_head(last_timestep_output, return_expectation=False)  # Shape: (batch_size, num_value_bins)
        value_probs = jax.nn.softmax(value_logits, axis=-1)  # Shape: (batch_size, num_value_bins)

        # Build value support atoms (bins)
        value_bins, _ = self.build_z()  # Shape: (num_value_bins,)
        expected_values = jnp.sum(value_bins[None, :] * value_probs, axis=-1)  # Shape: (batch_size,)

        return expected_values

    @override
    @at.typecheck
    def sample_actions(
        self, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        **kwargs
    ) -> _model.Actions:
        """Sample actions from the value model.
        
        For value models, this is a placeholder that returns zero actions.
        Value models predict values, not actions, so this method does nothing.
        It exists only to satisfy the abstract base class requirement.
        
        Args:
            rng: Random number generator key (unused)
            observation: Observation data (unused)
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            Zero actions with shape (*batch_dims, action_horizon, action_dim)
        """
        batch_size = observation.state.shape[0]
        return jnp.zeros((batch_size, self.action_horizon, self.action_dim), dtype=jnp.float32)


class SimpleValue(Value):
    def __init__(self, config: value_config.ValueConfig, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.model_path = config.model_path 
        self.num_value_bins = config.num_value_bins 
        # Hardcode config for gemma_3_270m (width=640) from Google DeepMind Gemma model
        self.valuegemma_width = 640  # gemma_3_270m hidden size
        self.Vmin = config.Vmin
        self.Vmax = config.Vmax
        
        # Load Gemma model weights and return as ToNNX.
        llm = build_gemma_3_270m_model(self.model_path)

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=self.valuegemma_width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )

        # Initialize SigLIP with dummy image
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.ValueGemma = nnx.Dict(llm=llm, img=img)

        # Build value head: 3-layer MLP to project last timestep output to 201 bins
        self.value_head = DistributionalValueHeadGemma3(
            hidden_size=self.valuegemma_width,
            num_bins=self.num_value_bins,
            v_min=self.Vmin,
            v_max=self.Vmax,
            rngs=rngs
        )
        
    def compute_value(
        self, rng: at.KeyArrayLike, observation: _model.Observation,  *, train: bool = False,  value_targets: at.Float[at.Array, "b"] | None = None, 
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute value.
        
        For value model, we only use prefix (images + state), no actions needed.
        The actions parameter is kept for interface compatibility but not used.
        """
        preprocess_rng = jax.random.split(rng, 1)[0]
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # Only use prefix tokens (images + language), no suffix needed
        # embed_prefix returns embeddings (3D), not token IDs (2D)
        prefix_embeddings, inputs_mask, ar_mask = self.embed_prefix(observation)
        # TODO: confirm the last input token can attend to all previous tokens 
        attn_mask = make_attn_mask(inputs_mask, ar_mask)
        positions = jnp.cumsum(inputs_mask, axis=1) - 1
        
        # Since prefix_embeddings is 3D embeddings (not token IDs), we need to call
        # the underlying model's _apply_attention method directly.
        from gemma.gm.nn._transformer import _Inputs

        # Get the underlying Flax Linen model
        underlying_model = self.ValueGemma.llm.module

        # Get model parameters from NNX state
        graphdef, state = nnx.split(self.ValueGemma.llm)
        # state = state.unfreeze()
        params_dict = state.to_pure_dict()
        variables = {'params': params_dict}

        # Create inputs for _apply_attention
        inputs = _Inputs(
            embeddings=prefix_embeddings,
            positions=positions,
            attention_mask=attn_mask,
            inputs_mask=inputs_mask,
        )

        # Call _apply_attention
        hidden_states, _ = underlying_model.apply(
            variables,
            inputs,
            None,  # cache
            method=underlying_model._apply_attention,
        )
        
        # Extract the output at the last valid position for each sequence
        last_valid_positions = jnp.sum(inputs_mask, axis=1, dtype=jnp.int32) - 1  # Shape: (batch_size,)
        batch_indices = jnp.arange(hidden_states.shape[0])
        
        last_timestep_output = hidden_states[batch_indices, last_valid_positions, :]  # Shape: (batch_size, hidden_dim)
        
        # Project through value head to get num_value_bins bins prediction
        value_logits = self.value_head(last_timestep_output, return_expectation=False)  # Shape: (batch_size, num_value_bins)
        value_probs = jax.nn.softmax(value_logits, axis=-1)  # Shape: (batch_size, num_value_bins)
        
        # Build value support atoms (bins)
        value_bins, _ = self.build_z()  # Shape: (num_value_bins,) 
        expected_values = jnp.sum(value_bins[None, :] * value_probs, axis=-1)  # Shape: (batch_size,)

        return expected_values
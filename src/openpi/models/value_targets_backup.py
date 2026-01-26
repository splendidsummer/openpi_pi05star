"""Value function model combining SigLIP and Gemma for VLM-based value estimation."""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

NUM_ATOMS = 201
V_MIN = -1.0
V_MAX = 0.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)


def get_supports():
    """Get the support values for C51 distributional RL."""
    return jnp.linspace(V_MIN, V_MAX, NUM_ATOMS)


def make_attn_mask(input_mask, mask_ar):
    """Creates attention mask from input mask and autoregressive mask."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def target_to_twohot(target_values: jnp.ndarray) -> jnp.ndarray:
    """Convert target values to two-hot distribution using linear interpolation.

    Args:
        target_values: Target values in range [V_MIN, V_MAX], shape [batch].

    Returns:
        Two-hot target distribution, shape [batch, NUM_ATOMS].
    """
    target_values = jnp.clip(target_values, V_MIN, V_MAX)

    b_left = jnp.floor((target_values - V_MIN) / DELTA_Z).astype(jnp.int32)
    b_left = jnp.clip(b_left, 0, NUM_ATOMS - 2)
    b_right = b_left + 1

    z_left = V_MIN + b_left.astype(jnp.float32) * DELTA_Z

    weight_right = (target_values - z_left) / DELTA_Z
    weight_left = 1.0 - weight_right

    batch_size = target_values.shape[0]
    target_dist = jnp.zeros((batch_size, NUM_ATOMS))

    batch_idx = jnp.arange(batch_size)
    target_dist = target_dist.at[batch_idx, b_left].set(weight_left)
    return target_dist.at[batch_idx, b_right].set(weight_right)



def dist_to_value(logits: jnp.ndarray) -> jnp.ndarray:
    """Convert distribution logits to expected value.

    Args:
        logits: Distribution logits, shape [batch, NUM_ATOMS].

    Returns:
        Expected value, shape [batch].
    """
    supports = get_supports()
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * supports, axis=-1)


class ValueHead(nnx.Module):
    """Two-layer MLP value head with LayerNorm and GELU."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(input_dim, rngs=rngs)
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = nnx.gelu(x)
        # 简化：训练时不使用 dropout，避免状态管理复杂性
        # if train and self.dropout_rate > 0:
        #     x = nnx.dropout(x, rate=self.dropout_rate, deterministic=False)
        return self.linear2(x)


class ValueModel(nnx.Module):
    """Value function model using SigLIP (400M) + Gemma (270M) with C51-style distributional output."""

    def __init__(self, config: "ValueModelConfig", rngs: nnx.Rngs):
        self.config = config
        gemma_config = _gemma.get_config(config.gemma_variant)

        self.img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=1152,  # SigLIP So400m 的实际输出维度
                variant=config.siglip_variant,
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )

        # 添加投影层将SigLIP的1152维投影到Gemma的640维
        self.img_projection = nnx.Linear(1152, gemma_config.width, rngs=rngs)

        self.llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[gemma_config],
                embed_dtype=config.dtype,
                adarms=False,
            )
        )

        fake_image = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
        self.img.lazy_init(fake_image, train=False, rngs=rngs)
        self.llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])

        hidden_dim = gemma_config.width // 2
        self.value_head = ValueHead(
            input_dim=gemma_config.width,
            hidden_dim=hidden_dim,
            output_dim=NUM_ATOMS,
            dropout_rate=0.1,
            rngs=rngs,
        )

        self.deterministic = True

    @at.typecheck
    def embed_tokens(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed images and text into a single token sequence."""
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name], train=False)
            # 投影SigLIP特征到Gemma维度
            image_tokens = self.img_projection(image_tokens)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            text_tokens = self.llm(obs.tokenized_prompt, method="embed")
            tokens.append(text_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * text_tokens.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    def __call__(self, observation: _model.Observation, *, train: bool = False) -> at.Float[at.Array, "b num_atoms"]:
        """Forward pass to compute value distribution logits.

        Args:
            observation: Input observation containing images and optional text.
            train: Whether in training mode.

        Returns:
            Value distribution logits of shape [batch, NUM_ATOMS].
        """
        tokens, input_mask, ar_mask = self.embed_tokens(observation)

        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        out, _ = self.llm([tokens], mask=attn_mask, positions=positions, adarms_cond=[None])

        last_token_idx = jnp.sum(input_mask, axis=-1) - 1
        batch_idx = jnp.arange(out[0].shape[0])
        pooled = out[0][batch_idx, last_token_idx]

        return self.value_head(pooled, train=train)

    def compute_value(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        """Compute expected value from distribution.

        Args:
            rng: Random key for preprocessing.
            observation: Input observation.
            train: Whether in training mode.

        Returns:
            Expected value of shape [batch].
        """
        # Value model 使用数据中实际可用的图像键
        available_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(rng, observation, train=train, image_keys=available_keys)
        logits = self(observation, train=train)
        return dist_to_value(logits)

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        target: at.Float[at.Array, "..."],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, ""]:
        """Compute cross-entropy loss between predicted distribution and two-hot target.

        Args:
            rng: Random key for preprocessing.
            observation: Input observation.
            target: Either scalar values [batch] or two-hot distribution [batch, 201].
            train: Whether in training mode.

        Returns:
            Scalar cross-entropy loss.
        """
        # Value model 使用数据中实际可用的图像键
        available_keys = list(observation.images.keys())
        observation = _model.preprocess_observation(rng, observation, train=train, image_keys=available_keys)
        logits = self(observation, train=train)

        target_dist = target_to_twohot(target) if target.ndim == 1 else target

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target_dist * log_probs, axis=-1)

        return jnp.mean(loss)


class ValueHeadGemma3(nn. Module):
    """Value head optimized for Gemma 3 backbone"""
    def __init__(self, hidden_size):
        super().__init__()
        self.head_l1 = nn.Linear(hidden_size, 512)
        self.head_act1 = nn.SiLU()  # ✅ Matches Gemma 3
        self.head_l2 = nn.Linear(512, 128)
        self.head_act2 = nn.SiLU()  # ✅ Matches Gemma 3
        self.head_l3 = nn.Linear(128, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        # SiLU is similar to ReLU family for initialization
        nn.init.kaiming_normal_(self.head_l1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.head_l1.bias)
        nn.init.kaiming_normal_(self.head_l2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.head_l2.bias)
        nn.init.normal_(self.head_l3.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.head_act1(self.head_l1(x))
        x = self.head_act2(self.head_l2(x))
        x = self.head_l3(x)
        return x
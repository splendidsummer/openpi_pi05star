import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi05_star_config
import openpi.models.gemma as _gemma 
from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


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


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi05_STAR(_model.BaseModel):
    def __init__(self, config: pi05_star_config.Pi05_STAR_Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.config = config
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
        
    @at.typecheck
    def embed_fm_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

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
        # Prefer star_tokenized_prompt for STAR expert, fall back to tokenized_prompt for backward compatibility
        tokenized_prompt = obs.star_tokenized_prompt 
        tokenized_prompt_mask = obs.star_tokenized_prompt_mask

        if tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_fm_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
 
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_fm_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_fm_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_fm_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @at.typecheck
    def embed_fast_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # 处理图片
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)  # 图像编码

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # 图片token之间可互相关注，AR mask为0
            ar_mask.append(0 * input_mask[-1])

        # 处理文本token
        assert obs.fast_tokenized_prompt is not None, "Fast tokenized prompt is required"
        assert obs.fast_tokenized_prompt_mask is not None, "Fast tokenized prompt mask is required"
        assert obs.fast_token_ar_mask is not None, "Fast token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.fast_tokenized_prompt, method="embed")  # 文本编码
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.fast_tokenized_prompt_mask)
        ar_mask.append(obs.fast_token_ar_mask)

        # 拼接所有embedding、mask
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_fast_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 预处理观测
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # 前向：一次性处理prefix+suffix
        input_token_embeddings, input_mask, ar_mask = self.embed_fast_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # 构造one-hot目标，预测下一个token（右移一位）
        targets = jax.nn.one_hot(
            # shifting one token afterwards to get the auto-regressive target
            observation.fast_tokenized_prompt[:, 1:],
                PALIGEMMA_VOCAB_SIZE,
        )

        # 输入去掉最后一个token
        seq_len = input_token_embeddings.shape[1] - 1
        batch_size = input_token_embeddings.shape[0]
        positions = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch_size, seq_len))

        logits, _ = self.PaliGemma.llm(
            input_token_embeddings[:, :-1],
            positions,
            attn_mask[:, :-1, :-1],
            return_logits=True,
            method="forward_first_expert",
        )

        # Only take logits for text tokens (exclude image tokens)
        # logits shape: [batch, total_seq_len-1, vocab_size]
        # We need the last (fast_tokenized_prompt_len - 1) logits which correspond to text predictions
        text_token_len = observation.fast_tokenized_prompt.shape[1]
        text_logits = logits[:, -(text_token_len - 1):]

        logp = jax.nn.log_softmax(text_logits, axis=-1)

        # 计算交叉熵损失
        assert observation.fast_token_loss_mask is not None, "Fast token loss mask is required"
        loss_mask = observation.fast_token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    def compute_loss(
        self,
        rng,
        observation,
        actions,
        alpha: float = 1.0,
        *,
        train = False):

        rng_fm, rng_fast = jax.random.split(rng)

        # 1. Flow Matching Loss
        # Returns [batch, action_horizon], representing loss per timestep
        fm_loss = self.compute_fm_loss(rng_fm, observation, actions, train=train)
        # Reduce to [batch] by averaging over the horizon
        fm_loss = jnp.mean(fm_loss, axis=-1)

        # 2. Fast Tokenization Loss
        # Returns [batch], representing average cross-entropy loss per sequence
        fast_loss = self.compute_fast_loss(rng_fast, observation, actions, train=train)

        # Combine losses weighted by alpha
        return alpha * fm_loss + fast_loss
    
    # TODO: sample_actions method needs to be completely modified for STAR design
    # TODO: 1. This threshold allows us to control the optimality indicator, and minimizes the need for finding an attenuation factor β to sharpen the improvement conditioned distribution after training.2   
            # 2. Prior work [4] instead uniformly chose ϵ = 0 and tuned β at test time, as
            # in classifier-free guidance (CFG). However, high CFG weights can drive the
            # action distribution to the corners of its support (leading to aggressive behavior)
            # and would not affect the autoregressive part of the model. We found it easier to
            # obtain good results by instead using the threshold ϵℓ to trade off regularization
            # and optimality.  
            # references: [4] J. Ho and N. Salimans. Classifier-free diffusion guidance.  

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        pass 
        # observation = _model.preprocess_observation(None, observation, train=False)
        # # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        # dt = -1.0 / num_steps
        # batch_size = observation.state.shape[0]
        # if noise is None:
        #     noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # # first fill KV cache with a forward pass of the prefix
        # prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # def step(carry):
        #     x_t, time = carry
        #     suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        #         observation, x_t, jnp.broadcast_to(time, batch_size)
        #     )
        #     # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        #     # other
        #     suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        #     # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        #     # prefix tokens
        #     prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        #     # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        #     # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        #     full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        #     assert full_attn_mask.shape == (
        #         batch_size,
        #         suffix_tokens.shape[1],
        #         prefix_tokens.shape[1] + suffix_tokens.shape[1],
        #     )
        #     # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        #     positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        #     (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        #         [None, suffix_tokens],
        #         mask=full_attn_mask,
        #         positions=positions,
        #         kv_cache=kv_cache,
        #         adarms_cond=[None, adarms_cond],
        #     )
        #     assert prefix_out is None
        #     v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        #     return x_t + dt * v_t, time + dt

        # def cond(carry):
        #     x_t, time = carry
        #     # robust to floating-point error
        #     return time >= -dt / 2

        # x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        # return x_0

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# 初始化日志格式
def init_logging():
    """自定义日志格式以提高可读性。"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)

# 初始化wandb日志
def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    # 如果是恢复训练
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新的训练运行
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

# 加载并验证权重
def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """加载并验证权重。返回加载的权重子集。"""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 从加载的参数中移除jax.ShapeDtypeStruct，确保只返回加载的参数
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )

# 初始化训练状态
@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    # 创建优化器
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # 初始化模型及其参数
        model = config.model.create(model_rng)

        # 将部分参数合并到模型中
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # 如果部分参数不是状态的子集会报错
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # 将冻结的参数转换为bfloat16
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 计算训练状态的形状（不进行实际计算）
    train_state_shape = jax.eval_shape(init, init_rng)
    # 定义FSDP分片策略
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # 加载并验证权重
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    
    # 计算部分参数的分片，避免初始化期间跨设备复制
    params_shape = train_state_shape.params.to_pure_dict()
    params_sharding = sharding.fsdp_sharding(params_shape, mesh, log=False)
    
    # 提取与partial_params结构匹配的分片规范
    flat_partial = traverse_util.flatten_dict(partial_params, sep="/")
    flat_sharding = traverse_util.flatten_dict(params_sharding, sep="/")
    
    # 在jit编译之前将numpy数组转换为分片的JAX数组，避免复制到所有GPU
    sharded_partial_params = {}
    for key, arr in flat_partial.items():
        if key in flat_sharding and isinstance(arr, np.ndarray):
            shard_spec = flat_sharding[key]
            if isinstance(shard_spec, jax.sharding.NamedSharding):
                # 将numpy数组转换为指定分片的JAX数组
                sharded_partial_params[key] = jax.device_put(arr, shard_spec)
            else:
                sharded_partial_params[key] = jnp.array(arr)
        else:
            sharded_partial_params[key] = jnp.array(arr) if isinstance(arr, np.ndarray) else arr
    
    partial_params = traverse_util.unflatten_dict(sharded_partial_params, sep="/")
    
    # 构建匹配partial_params结构的分片树，用于in_shardings
    matching_sharding = {
        k: flat_sharding[k] for k in flat_partial.keys() if k in flat_sharding
    }
    partial_params_sharding = traverse_util.unflatten_dict(matching_sharding, sep="/")

    # 初始化训练状态并混合部分参数，使用分片输入分片以避免复制
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 释放partial params的buffer
        in_shardings=(None, partial_params_sharding),  # rng为None（复制），params为分片
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding

# 单步训练
@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, at.Float[at.Array, "b"]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    # 合并模型定义和参数
    model = nnx.merge(state.model_def, state.params)
    # 设置模型为训练模式
    model.train()

    # 检查是否为 Value 模型
    is_value_model = config.model.model_type == _model.ModelType.VALUE

    # 根据模型类型选择不同的损失函数
    if is_value_model:
        # Value 模型的损失函数
        @at.typecheck
        def loss_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: _model.Observation,
            value_targets: at.Float[at.Array, "b"]
        ):
            # Value 模型使用 value_targets 计算损失
            loss_per_sample = model.compute_loss(
                rng,
                observation,
                train=True,
                value_targets=value_targets
            )
            return jnp.mean(loss_per_sample)
        # TODO: 检查batch的类型 FROM DATA_LOADER!!!`
        # 解包 batch（Value 模型有3个元素）
        observation, actions, value_targets = batch  # type: ignore
        train_rng = jax.random.fold_in(rng, state.step)
        
        # 过滤掉冻结的参数
        diff_state = nnx.DiffState(0, config.trainable_filter)
        # 计算损失和梯度（使用 value_targets）
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, value_targets)
    else:
        # 标准模型的损失函数
        @at.typecheck
        def loss_fn(
            model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
        ):
            # 计算损失
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        # 生成训练随机数
        train_rng = jax.random.fold_in(rng, state.step)
        observation, actions = batch  # type: ignore

        # 过滤掉冻结的参数
        diff_state = nnx.DiffState(0, config.trainable_filter)
        # 计算损失和梯度
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 获取可训练参数
    params = state.params.filter(config.trainable_filter)
    # 计算更新和新的优化器状态
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    # 应用更新到参数
    # 检查是否使用了 _MultiLROptimizerWrapper（它会在内部应用更新）
    # 通过检查优化器是否有 apply_updates 方法来判断
    if updates is None:
        # 如果没有更新（所有参数都被排除），保持原参数
        new_params = params
    elif hasattr(state.tx, '_inner_optimizer'):
        # _MultiLROptimizerWrapper 已经应用了更新，直接使用
        new_params = updates
    elif isinstance(updates, nnx.State):
        # 如果 updates 是 nnx.State，说明已经应用了更新
        new_params = updates
    else:
        # 标准 optax 优化器，需要应用更新
        # optax.apply_updates 会自动处理 None 值，无需过滤
        new_params = optax.apply_updates(params, updates)

    # 更新模型并返回新的完整状态
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # 创建新的训练状态
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    
    # 如果设置了EMA衰减，更新EMA参数
    if state.ema_decay is not None:
        def _ema_update(old, new):
            if isinstance(old, jax.Array) and isinstance(new, jax.Array):
                try:
                    if jnp.issubdtype(old.dtype, jnp.inexact) and jnp.issubdtype(new.dtype, jnp.inexact):
                        return state.ema_decay * old + (1 - state.ema_decay) * new
                except TypeError:
                    pass
            return new

        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                _ema_update,
                state.ema_params,
                new_params,
            ),
        )

    # 过滤掉非核参数（用于统计）
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # 收集训练信息
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

# 主训练流程
def main(config: _config.TrainConfig):
    # 初始化日志记录
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 检查批量大小是否可被设备数量整除
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 设置JAX编译缓存目录
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 初始化随机数种子
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 创建网格和分片规范
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化检查点管理器
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    # 初始化wandb
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建数据加载器
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    # 获取第一个批次数据
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 记录第一个批次的图像进行健全性检查
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果是从检查点恢复，则恢复状态
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 编译训练步骤函数
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # 设置训练进度条
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    # 训练循环
    for step in pbar:
        # 在指定网格上下文中运行训练步骤
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # 定期记录日志
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        
        # 获取下一个批次数据
        batch = next(data_iter)

        # 定期保存检查点
        # if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
        if (step % config.save_interval == 0) or step == config.num_train_steps - 1:  
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

# 程序入口
if __name__ == "__main__":
    main(_config.cli())

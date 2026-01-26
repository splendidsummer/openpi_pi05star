import dataclasses
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import flax.nnx as nnx
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from typing import TypeAlias
    Filter: TypeAlias = nnx.filterlib.Filter
else:
    # 使用字符串形式避免 tyro 解析问题
    Filter = "nnx.filterlib.Filter"


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@dataclasses.dataclass(frozen=True)
class MultiLRScheduleConfig(LRScheduleConfig):
    """多学习率调度配置，为不同的参数组设置不同的学习率。
    
    这个配置类实现了 LRScheduleConfig 协议，但实际上 create() 方法不会被直接调用。
    相反，create_optimizer() 函数会检测 MultiLRScheduleConfig 并使用 optax.multi_transform
    来创建多学习率优化器。
    """
    
    param_lr_schedules: dict[str, LRScheduleConfig]
    """参数组名称到学习率调度器的映射，例如：
    {
        "siglip": CosineDecaySchedule(peak_lr=1e-5, ...),
        "llm": CosineDecaySchedule(peak_lr=5e-5, ...),
        "value_head": CosineDecaySchedule(peak_lr=1e-4, ...),
    }
    """
    
    param_masks: dict[str, Any]  # 使用 Any 避免 tyro 解析 Filter 类型的问题
    """参数组名称到过滤器的映射，例如：
    {
        "siglip": nnx_utils.PathRegex(".*ValueGemma/img/.*"),
        "llm": nnx_utils.PathRegex(".*ValueGemma/llm/.*"),
        "value_head": nnx_utils.PathRegex(".*value_head/.*"),
    }
    """
    
    def create(self) -> optax.Schedule:
        """这个方法实际上不会被调用。
        
        当使用 MultiLRScheduleConfig 时，create_optimizer() 会检测它并直接创建
        多学习率优化器，而不是调用这个方法。
        """
        # 返回一个默认的学习率调度器（不会被使用）
        return optax.constant_schedule(1e-5)


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    """创建优化器，支持单学习率和多学习率配置。
    
    如果 lr_schedule 是 MultiLRScheduleConfig，则返回一个特殊的优化器包装器，
    它会在初始化时根据参数树创建多学习率优化器。否则，使用标准的单学习率优化器。
    """
    # 检测是否为多学习率配置
    if isinstance(lr_schedule, MultiLRScheduleConfig):
        # 创建多学习率优化器包装器
        return _MultiLROptimizerWrapper(optimizer, lr_schedule, weight_decay_mask)
    else:
        # 标准的单学习率优化器
        lr = lr_schedule.create()
        return optimizer.create(lr, weight_decay_mask=weight_decay_mask)


class _MultiLROptimizerWrapper:
    """多学习率优化器包装器。
    
    这个类包装了多学习率优化器的创建逻辑。在 init() 时，它会根据参数树
    创建 labels PyTree，然后使用 optax.multi_transform 创建真正的优化器。
    """
    
    def __init__(
        self,
        optimizer: OptimizerConfig,
        lr_schedule: MultiLRScheduleConfig,
        weight_decay_mask: at.PyTree | None = None,
    ):
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.weight_decay_mask = weight_decay_mask
        self._inner_optimizer: optax.GradientTransformation | None = None
    
    def init(self, params: at.PyTree) -> optax.OptState:
        """初始化优化器状态，同时创建多学习率优化器。"""
        if self._inner_optimizer is None:
            # 将 params 转换为纯字典格式（如果是 nnx.State）
            if isinstance(params, nnx.State):
                params_dict = params.to_pure_dict()
            else:
                params_dict = params
            
            # 创建 transforms 字典
            transforms = {}
            for group_name, group_lr_schedule in self.lr_schedule.param_lr_schedules.items():
                lr = group_lr_schedule.create()
                tx = self.optimizer.create(lr, weight_decay_mask=self.weight_decay_mask)
                transforms[group_name] = tx
            
            # 根据参数树创建 labels PyTree
            # labels 是一个与 params 结构相同的 PyTree，每个叶子节点是 transform 名称
            labels = self._create_labels_tree(params_dict)
            
            # 创建多学习率优化器
            self._inner_optimizer = optax.multi_transform(transforms, labels)
        
        # 确保传入纯字典格式的参数
        if isinstance(params, nnx.State):
            params_for_init = params.to_pure_dict()
        else:
            params_for_init = params
        
        return self._inner_optimizer.init(params_for_init)
    
    def update(self, updates: at.PyTree, state: optax.OptState, params: at.PyTree | None = None) -> tuple[at.PyTree, optax.OptState]:
        """更新参数。"""
        if self._inner_optimizer is None:
            raise RuntimeError("Optimizer not initialized. Call init() first.")
        
        # 记录原始类型，以便后续转换
        updates_is_state = isinstance(updates, nnx.State)
        params_is_state = isinstance(params, nnx.State) if params is not None else False
        
        # 将 updates 和 params 转换为纯字典格式（如果是 nnx.State）
        if updates_is_state:
            updates_dict = updates.to_pure_dict()
        else:
            updates_dict = updates
        
        if params is not None:
            if params_is_state:
                params_dict = params.to_pure_dict()
            else:
                params_dict = params
        else:
            params_dict = None
        
        # 调用 optax.multi_transform 的 update 方法
        updated_dict, new_state = self._inner_optimizer.update(updates_dict, state, params_dict)
        
        # 如果原始输入是 nnx.State，需要将结果转换回 nnx.State
        # optax.multi_transform.update 返回的是更新量（updates），不是更新后的参数
        # 所以我们需要先应用更新，然后转换回 nnx.State
        if params_is_state:
            # 将 params 转换为字典
            params_dict = params.to_pure_dict()
            
            # 使用 optax.apply_updates 应用更新（两个都是字典）
            updated_params_dict = optax.apply_updates(params_dict, updated_dict)
            
            # 将更新后的字典转换回 nnx.State
            # 使用 replace_by_pure_dict，它会自动处理结构匹配
            updated_state = params.replace_by_pure_dict(updated_params_dict)
            
            return updated_state, new_state
        else:
            # 如果输入不是 nnx.State，直接返回字典
            return updated_dict, new_state
    
    def _create_labels_tree(self, params: at.PyTree) -> at.PyTree:
        """根据参数树和过滤器创建 labels PyTree。
        
        labels 是一个与 params 结构相同的 PyTree，每个叶子节点是字符串，
        表示应该使用哪个 transform。
        
        改进：使用更高效的实现，支持 nnx.State 类型。
        """
        import jax.tree_util as tree_util
        
        # 如果 params 是 nnx.State，先转换为纯字典
        if isinstance(params, nnx.State):
            params_dict = params.to_pure_dict()
        else:
            params_dict = params
        
        def get_label(path: tuple, value: Any) -> str:
            """根据路径确定应该使用哪个 transform。"""
            # 检查每个参数组的过滤器（按顺序检查，第一个匹配的返回）
            for group_name, filter_obj in self.lr_schedule.param_masks.items():
                # 将路径转换为 nnx.filterlib.PathParts 格式
                path_parts = tuple(path)
                # 检查过滤器是否匹配
                try:
                    if filter_obj(path_parts, value):
                        return group_name
                except Exception:
                    # 如果过滤器检查失败，跳过这个过滤器
                    continue
            
            # 如果没有匹配，使用第一个 transform（作为默认值）
            # 通常这应该是 "llm"，因为它是主要的参数组
            default_group = list(self.lr_schedule.param_lr_schedules.keys())[0]
            return default_group
        
        # 使用 jax.tree_util.tree_map_with_path 创建 labels 树
        labels = tree_util.tree_map_with_path(
            lambda path, value: get_label(path, value),
            params_dict
        )
        
        return labels

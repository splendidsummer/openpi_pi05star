import numpy as np

import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class SIGLIPOnlyWeightLoader(WeightLoader):
    """只加载 SIGLIP 图像编码器权重的权重加载器。

    该类将 PaliGemma.img 权重映射到 ValueGemma.img，实现从 Pi0 检查点到 Value 模型的图像编码器迁移。

    兼容：
      训练检查点：
        例如: "./checkpoints/<config>/<exp>/<step>/params"
      发布检查点：
        例如: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str  # 检查点路径

    def load(self, params: at.Params) -> at.Params:
        # 加载检查点参数
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        # 将参数字典展平
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")  # 参考参数
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")  # 加载参数
        
        # 只提取 SIGLIP 图像编码器权重并映射到 ValueGemma.img
        result = {}
        target_prefix = "ValueGemma/img/"  # 目标前缀
        
        # 判断检查点中使用的前缀
        # 情况1: 检查点有 PaliGemma/img/ 前缀（如 pi05_droid 训练检查点）
        # 情况2: 检查点有 img/ 前缀（如 PaliGemma 检查点或扁平结构）
        source_prefix = None
        for k in flat_loaded.keys():
            if k.startswith("PaliGemma/img/"):
                source_prefix = "PaliGemma/img/"  # 来源前缀为 PaliGemma/img/
                break
            elif k.startswith("img/") and source_prefix is None:
                source_prefix = "img/"  # 来源前缀为 img/
       
        
        if source_prefix is None:
            logger.warning("未在检查点中找到 SIGLIP 图像编码器权重。可用的 key: %s", list(flat_loaded.keys())[:10])
            # 返回原始参数的深拷贝，确保结构和内容一致
            import copy
            return copy.deepcopy(params)
        
        # 将 SIGLIP 权重从 source_prefix 映射到 target_prefix
        mapped_count = 0  # 映射计数
        loaded_keys = []  # 记录已加载的 key
        skipped_keys = []  # 记录被跳过的 key
        for k, v in flat_loaded.items():
            if k.startswith(source_prefix):
                # 将 source_prefix 映射为 ValueGemma.img/*
                target_key = k.replace(source_prefix, target_prefix, 1)
                if target_key in flat_ref:
                    # 跳过分类头（通常以 head/ 开头）
                    if '/head/' in k or target_key.endswith('/head/kernel') or target_key.endswith('/head/bias'):
                        skipped_keys.append(target_key)
                        continue
                    # 如有需要，转换 dtype
                    result[target_key] = v.astype(flat_ref[target_key].dtype) if v.dtype != flat_ref[target_key].dtype else v
                    mapped_count += 1
                    loaded_keys.append(target_key)
                    if mapped_count <= 5:  # 只记录前5个映射
                        logger.info(f"映射 {k} -> {target_key}")

        # 统计所有 ValueGemma/img/ 下的参数数量和 head 参数数量
        total_img_params = 0
        head_img_params = 0
        for k in flat_ref.keys():
            if k.startswith(target_prefix):
                total_img_params += 1
                if '/head/' in k or k.endswith('/head/kernel') or k.endswith('/head/bias'):
                    head_img_params += 1

        expected_loaded = total_img_params - head_img_params

        print(f"[SIGLIPOnlyWeightLoader] Loaded {mapped_count} params (excluding classification head). First 10 loaded keys:")
        for key in loaded_keys[:10]:
            print(f"  {key}")
        if skipped_keys:
            print(f"[SIGLIPOnlyWeightLoader] Skipped {len(skipped_keys)} classification head params:")
            for key in skipped_keys[:5]:
                print(f"  {key}")

        # 检查所有非 head 的 img 参数都被加载
        missing_keys = []
        for k in flat_loaded.keys():
            if k.startswith(source_prefix):
                target_key = k.replace(source_prefix, target_prefix, 1)
                if '/head/' not in k and not target_key.endswith('/head/kernel') and not target_key.endswith('/head/bias'):
                    if target_key in flat_ref and target_key not in loaded_keys:
                        missing_keys.append(target_key)
        if missing_keys:
            print(f"[SIGLIPOnlyWeightLoader][WARNING] {len(missing_keys)} non-head params were not loaded:")
            for key in missing_keys[:5]:
                print(f"  {key}")
        else:
            print("[SIGLIPOnlyWeightLoader] All non-classification-head params loaded successfully.")

        logger.info(f"成功将 {mapped_count} 个 SIGLIP 图像编码器权重从 {source_prefix} 映射到 {target_prefix}")

        # 检查映射数量是否等于期望数量
        if mapped_count != expected_loaded:
            raise RuntimeError(f"[SIGLIPOnlyWeightLoader][ERROR] Loaded param count ({mapped_count}) does not match expected ({expected_loaded}) (total: {total_img_params}, head: {head_img_params})")

        # 保留参考参数中的其他权重
        for k, v in flat_ref.items():
            if k not in result:
                result[k] = v

        # 递归地将所有数组包装为 nnx.Param，防止后续模型合并时报错
        import flax.nnx as nnx
        import jax.numpy as jnp
        def wrap_arrays_as_param(tree):
            if isinstance(tree, dict):
                return {k: wrap_arrays_as_param(v) for k, v in tree.items()}
            elif isinstance(tree, (np.ndarray, jnp.ndarray)):
                return nnx.Param(tree)
            else:
                return tree

        result = {k: v for k, v in result.items()}
        params_tree = flax.traverse_util.unflatten_dict(result, sep="/")
        params_tree = wrap_arrays_as_param(params_tree)
        return params_tree


@dataclasses.dataclass(frozen=True)
class SIGLIPValueGemmaWeightLoader(WeightLoader):
    """Loads SIGLIP image encoder weights for ValueGemma with hidden_size=640.

    This is specifically designed for ValueGemma models using Gemma-3-270m backbone
    where the SIGLIP output dimension (num_classes) must be 640 to match Gemma's
    hidden size. It handles shape mismatches by skipping incompatible parameters
    (e.g., classification head weights) and only loading compatible encoder weights.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load the checkpoint parameters
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # Flatten both parameter dictionaries
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        # Extract only SIGLIP image encoder weights and map them to ValueGemma.img
        result = {}
        target_prefix = "ValueGemma/img/"

        # Determine which prefix to use based on what's in the checkpoint
        # Case 1: Checkpoint has PaliGemma/img/ prefix (from training checkpoint like pi05_droid)
        # Case 2: Checkpoint has img/ prefix (from PaliGemma checkpoint or flattened structure)
        source_prefix = None
        for k in flat_loaded.keys():
            if k.startswith("PaliGemma/img/"):
                source_prefix = "PaliGemma/img/"
                break
            elif k.startswith("img/") and source_prefix is None:
                source_prefix = "img/"

        if source_prefix is None:
            logger.warning("No SIGLIP image encoder weights found in checkpoint. Available keys: %s", list(flat_loaded.keys())[:10])
            # Return reference params unchanged
            return params

        # Map SIGLIP weights from source to target, handling shape mismatches
        mapped_count = 0
        skipped_shape_mismatch = []
        for k, v in flat_loaded.items():
            if k.startswith(source_prefix):
                # Map source prefix to ValueGemma.img/*
                target_key = k.replace(source_prefix, target_prefix, 1)
                if target_key in flat_ref:
                    ref_shape = flat_ref[target_key].shape
                    if v.shape == ref_shape:
                        # Convert dtype if needed
                        result[target_key] = v.astype(flat_ref[target_key].dtype) if v.dtype != flat_ref[target_key].dtype else v
                        mapped_count += 1
                        if mapped_count <= 5:  # Log first few mappings
                            logger.info(f"Mapped {k} -> {target_key}")
                    else:
                        # Shape mismatch, skip this parameter
                        skipped_shape_mismatch.append((k, v.shape, ref_shape))
                        logger.debug(f"Skipping {k} due to shape mismatch: {v.shape} vs {ref_shape}")

        if skipped_shape_mismatch:
            logger.warning(
                f"Skipped {len(skipped_shape_mismatch)} SIGLIP weights due to shape mismatches. "
                "This is expected for classification head weights when num_classes differs."
            )
            for k, src_shape, tgt_shape in skipped_shape_mismatch[:5]:
                logger.warning(f"  {k}: {src_shape} -> {tgt_shape}")

        logger.info(f"Successfully mapped {mapped_count} SIGLIP image encoder weights from {source_prefix} to {target_prefix}")

        # Keep all other weights from the reference parameters
        for k, v in flat_ref.items():
            if k not in result:
                result[k] = v

        return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

#!/usr/bin/env python3
"""Gemma 模型工具函数
提供构建和加载 Gemma-3-270M 模型的工具函数。
"""

# Ensure we import Google's gemma library, not local gemma.py
import sys
import gc
if 'gemma' in sys.modules:
    # Check if it's the local gemma module
    module = sys.modules['gemma']
    if hasattr(module, '__file__') and 'src/openpi/models' in module.__file__:
        del sys.modules['gemma']
import gemma.gm as gm
from gemma.gm.ckpts import load_params
from gemma.gm.ckpts import _paths
import jax
import jax.numpy as jnp
import pathlib
from orbax import checkpoint as ocp
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from collections.abc import Sequence
from typing import Any
from flax import traverse_util


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title: str):
    """打印子节标题"""
    print(f"\n【{title}】")
    print("-" * 70)


# def test_load_checkpoint(model, checkpoint_path=None, use_online=True):
#     """测试从检查点加载参数
    
#     Args:
#         model: Gemma 模型实例
#         checkpoint_path: 检查点路径（本地路径或 gs:// 路径）
#         use_online: 如果为 True，使用在线检查点路径（默认）
#     """
#     print_subsection("5. 检查点参数加载测试")
#     try:
#         # 如果使用在线加载，使用 gemma 库提供的检查点路径
#         if use_online or checkpoint_path is None:
#             print("  使用在线检查点路径...")
#             # 使用 Gemma 3 270M 预训练检查点
#             checkpoint_path = _paths.CheckpointPath.GEMMA3_270M_PT
#             print(f"  检查点路径: {checkpoint_path}")
#             print(f"  类型: 预训练模型 (PT)")
#             print(f"  注意: 首次使用时会自动下载检查点，可能需要一些时间...")
#         else:
#             checkpoint_path = pathlib.Path(checkpoint_path)
            
#             # 检查本地路径是否存在
#             if isinstance(checkpoint_path, pathlib.Path) and not checkpoint_path.exists():
#                 print(f"⚠ 检查点路径不存在: {checkpoint_path}")
#                 print("  尝试使用在线检查点路径...")
#                 checkpoint_path = _paths.CheckpointPath.GEMMA3_270M_PT
#                 print(f"  检查点路径: {checkpoint_path}")
#             else:
#                 print(f"  检查点路径: {checkpoint_path}")
#                 if isinstance(checkpoint_path, pathlib.Path):
#                     print(f"  路径存在: ✓")
                    
#                     # 检查关键文件
#                     key_files = ["manifest.ocdbt", "_METADATA", "descriptor"]
#                     missing_files = []
#                     for key_file in key_files:
#                         file_path = checkpoint_path / key_file
#                         if file_path.exists():
#                             print(f"  {key_file}: ✓")
#                         else:
#                             print(f"  {key_file}: ✗ (缺失)")
#                             missing_files.append(key_file)

#                     if missing_files:
#                         raise FileNotFoundError(f"检查点目录缺少关键文件: {missing_files}")
        
#         # 初始化模型参数（用于验证结构）
#         print("\n  初始化模型参数结构...")
#         key = jax.random.PRNGKey(42)
#         dummy_tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
#         variables = model.init(key, dummy_tokens)
#         init_params = variables['params']
        
#         print("  使用 orbax 直接加载检查点...")
#         print("  注意: 如果使用在线路径，首次下载可能需要几分钟...")
        
#         # 尝试使用 gemma 的 load_params（适用于在线路径）
#         loaded_params = None
#         if isinstance(checkpoint_path, str) and checkpoint_path.startswith('gs://'):
#             try:
#                 print("  尝试使用 gemma.load_params 加载在线检查点...")
#                 loaded_params = load_params(
#                     checkpoint_path,
#                     params=init_params,
#                     donate=False,
#                     text_only=False,
#                 )
#                 print("✓ 使用 gemma.load_params 加载成功")
#             except Exception as e:
#                 print(f"  ⚠ gemma.load_params 失败: {e}")
#                 print("  尝试使用 orbax 直接加载...")
        
#         # 如果 gemma.load_params 失败或使用本地路径，使用 orbax 直接加载
#         if loaded_params is None:
#             try:
#                 checkpoint_path_obj = pathlib.Path(checkpoint_path) if isinstance(checkpoint_path, (str, pathlib.Path)) else checkpoint_path
                
#                 # 使用 orbax 直接加载
#                 with ocp.PyTreeCheckpointer() as ckptr:
#                     # 直接恢复整个检查点（不指定 item，让 orbax 自动检测）
#                     print("  恢复整个检查点...")
#                     restored = ckptr.restore(checkpoint_path_obj)
                    
#                     # 检查点结构：键是扁平化的 transformer/embedder 格式
#                     # 需要转换为嵌套的 {embedder: {...}} 格式
#                     if isinstance(restored, dict):
#                         # 检查是否有 transformer/ 前缀的键
#                         transformer_keys = [k for k in restored.keys() if isinstance(k, str) and k.startswith('transformer/')]
                        
#                         if transformer_keys:
#                             print(f"  找到 {len(transformer_keys)} 个 transformer/ 前缀的键")
#                             print(f"  示例键: {transformer_keys[:3]}")
                            
#                             # 将扁平化的键转换为嵌套结构
#                             # transformer/embedder/input_embedding -> {embedder: {input_embedding: ...}}
#                             # 注意：只有 attn 下的 *_einsum 需要保留 {'w': array} 结构，其他需要提取出 array
#                             loaded_params = {}
                            
#                             for key, value in restored.items():
#                                 if isinstance(key, str) and key.startswith('transformer/'):
#                                     # 移除 transformer/ 前缀
#                                     path = key.replace('transformer/', '').split('/')
                                    
#                                     # 检查是否需要保留 w 结构
#                                     # 只有 attn 下的 *_einsum 需要保留 {'w': array} 结构
#                                     full_path = '/'.join(path)
#                                     should_keep_w = (
#                                         'attn/' in full_path and 
#                                         ('attn_vec_einsum' in full_path or 'kv_einsum' in full_path or 'q_einsum' in full_path)
#                                     )
                                    
#                                     # 如果值是字典且只有一个 'w' 键
#                                     if isinstance(value, dict) and len(value) == 1 and 'w' in value:
#                                         if should_keep_w:
#                                             # 保留 {'w': array} 结构
#                                             value = value
#                                         else:
#                                             # 提取出 array
#                                             value = value['w']
                                    
#                                     # 构建嵌套字典
#                                     current = loaded_params
#                                     for i, part in enumerate(path):
#                                         if i == len(path) - 1:
#                                             # 最后一个部分是值
#                                             current[part] = value
#                                         else:
#                                             # 中间部分是字典键
#                                             if part not in current:
#                                                 current[part] = {}
#                                             current = current[part]
#                                 else:
#                                     # 非 transformer/ 前缀的键直接添加
#                                     if isinstance(value, dict) and len(value) == 1 and 'w' in value:
#                                         # 检查是否需要保留 w 结构
#                                         should_keep_w = (
#                                             'attn/' in key and 
#                                             ('attn_vec_einsum' in key or 'kv_einsum' in key or 'q_einsum' in key)
#                                         )
#                                         if should_keep_w:
#                                             loaded_params[key] = value
#                                         else:
#                                             loaded_params[key] = value['w']
#                                     else:
#                                         loaded_params[key] = value
                            
#                             print("✓ 参数结构转换成功（扁平化 -> 嵌套，智能提取 'w' 键）")
#                         elif 'transformer' in restored:
#                             # 参数已经在 transformer 键下
#                             loaded_params = restored['transformer']
#                             print("✓ 从 transformer 键提取参数成功")
#                         else:
#                             # 直接使用
#                             loaded_params = restored
#                             print("✓ 直接使用恢复的参数")
#                     else:
#                         loaded_params = restored
#                         print("✓ 直接使用恢复的参数")
                        
#             except Exception as e:
#                 print(f"  ✗ orbax 加载失败: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 return None
        
#         if loaded_params is None:
#             print("✗ 所有加载方法都失败了")
#             return None
        
#         print("✓ 检查点参数加载成功")
        
#         # 比较加载的参数和初始参数的结构
#         def count_params(pytree):
#             count = 0
#             for leaf in jax.tree_util.tree_leaves(pytree):
#                 if hasattr(leaf, 'size'):
#                     count += leaf.size
#             return count
        
#         init_param_count = count_params(init_params)
#         loaded_param_count = count_params(loaded_params)
        
#         print(f"  初始参数数量: {init_param_count:,}")
#         print(f"  加载参数数量: {loaded_param_count:,}")
        
#         # 检查参数结构是否匹配（使用展平的键进行比较）
#         init_flat = traverse_util.flatten_dict(init_params, sep='/')
#         loaded_flat = traverse_util.flatten_dict(loaded_params, sep='/')
        
#         init_keys = set(init_flat.keys())
#         loaded_keys = set(loaded_flat.keys())
        
#         missing_keys = init_keys - loaded_keys
#         extra_keys = loaded_keys - init_keys
        
#         if len(missing_keys) == 0 and len(extra_keys) == 0:
#             # 进一步验证形状
#             shape_mismatches = []
#             for key in list(init_keys)[:20]:  # 检查前20个
#                 def get_shape(val):
#                     if isinstance(val, dict) and 'w' in val:
#                         return val['w'].shape if hasattr(val['w'], 'shape') else None
#                     return val.shape if hasattr(val, 'shape') else None
                
#                 init_shape = get_shape(init_flat[key])
#                 loaded_shape = get_shape(loaded_flat[key])
#                 if init_shape != loaded_shape:
#                     shape_mismatches.append((key, init_shape, loaded_shape))
            
#             if shape_mismatches:
#                 print(f"  参数结构匹配: ⚠ (键匹配但形状不匹配: {len(shape_mismatches)})")
#             else:
#                 print(f"  参数结构匹配: ✓✓✓ (完全匹配)")
#         else:
#             print(f"  参数结构匹配: ✗ (缺失 {len(missing_keys)} 个键，多余 {len(extra_keys)} 个键)")
#             if missing_keys:
#                 print(f"    缺失键示例: {list(missing_keys)[:3]}")
#             if extra_keys:
#                 print(f"    多余键示例: {list(extra_keys)[:3]}")
        
#         return loaded_params
        
#     except Exception as e:
#         print(f"✗ 检查点加载失败: {e}")
#         import traceback
#         traceback.print_exc()
#         print("\n  提示:")
#         print("    - 如果使用在线路径，请确保网络连接正常")
#         print("    - 如果使用 gs:// 路径，请确保已安装 gcsfs 或配置了 Google Cloud 认证")
#         print("    - 可以尝试使用本地路径: /root/autodl-tmp/gemma-3-270m")
#         return None


def build_gemma_3_270m_model(model_path=None, rng_key=None):
    """构建 Gemma-3-270M 模型。如果提供了 model_path，则加载权重。

    Args:
        model_path: 模型检查点路径。可以是本地路径、'gs://' 路径或 'online'。
        rng_key: 用于初始化的 JAX 随机密钥。

    Returns:
        nnx_bridge.ToNNX 包装的 Gemma-3-270M 模型。如果提供了 model_path，模型将被初始化并加载权重。
        否则，返回未初始化的模型。
    """
    # 创建 Gemma 模型并使用 ToNNX 包装（类似 SigLIP 的处理方式）
    model = gm.nn.Gemma3_270M()
    
    if rng_key is None:
        rng_key = jax.random.key(0)
    
    wrapped_model = nnx_bridge.ToNNX(model, rngs=nnx.Rngs(rng_key))

    print(f"正在加载 Gemma 模型权重: {model_path}...")
        
    # 使用现有的 test_load_checkpoint 函数加载参数, 这个函数会进行完整性检查、参数数量验证等
    loaded_params = load_checkpoint(model, checkpoint_path=model_path, use_online=model_path.startswith("gs://"))

    dummy_tokens = jnp.zeros((1, 8), dtype=jnp.int32)
    print("将加载的参数合并到模型中...")

    # 初始化 ToNNX 模型以便创建变量
    wrapped_model.lazy_init(dummy_tokens)

    # 将加载的参数合并到 NNX 状态中
    graphdef, state = nnx.split(wrapped_model)
    # loaded_params 对应于 variables['params']，结构应与 ToNNX 状态匹配
    # 使用 replace_by_pure_dict 进行合并
    state.replace_by_pure_dict(loaded_params)
    wrapped_model = nnx.merge(graphdef, state)

    print("Gemma 模型构建并加载完成。")

    del graphdef, state
    gc.collect()
        
    # 返回 ToNNX 包装的模型
    return wrapped_model


def get_token_embeddings(
    nnx_model,
    token_ids,
    min_vocab_threshold=1000):
    """Return token embeddings for given token_ids.

    nnx_model: an NNX model containing parameters.
    token_ids: array-like of ints (batch, seq) or (seq,).
    The function heuristically finds a 2D embedding matrix in model parameters
    (one dimension should be large, >= min_vocab_threshold).
    """
    import jax.numpy as jnp

    token_ids = jnp.asarray(token_ids)

    # Extract parameters from nnx model
    params = nnx.state(nnx_model).to_pure_dict()

    # DFS to collect 2D arrays as candidates
    candidates = []

    def _visit(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                _visit(v, path + "/" + k)
        else:
            try:
                shape = getattr(x, "shape", None)
                if shape is not None:
                    # Debug print
                    # print(f"DEBUG: Found array at {path} with shape {shape}")
                    
                    # Handle 2D arrays
                    if len(shape) == 2:
                        s0, s1 = int(shape[0]), int(shape[1])
                        if max(s0, s1) >= min_vocab_threshold:
                            candidates.append((x, (s0, s1)))
                    # Handle 3D arrays that might be (1, vocab, dim)
                    elif len(shape) == 3 and shape[0] == 1:
                        s0, s1 = int(shape[1]), int(shape[2])
                        if max(s0, s1) >= min_vocab_threshold:
                            candidates.append((x, (s0, s1)))
            except Exception:
                pass

    _visit(params)

    if not candidates:
        import jax
        print("DEBUG: Dumping all shapes found in params:")
        def print_shape(p, x):
             if hasattr(x, 'shape'):
                 print(f"  {p}: {x.shape}")
        jax.tree_util.tree_map_with_path(print_shape, params)
        raise ValueError("No 2D embedding-like matrix found in params.")

    # choose the most likely embedding (largest max dimension)
    emb_param, (s0, s1) = max(candidates, key=lambda t: max(t[1]))
    
    # ensure it is 2D
    emb_array = jnp.asarray(emb_param) 
    if emb_array.ndim == 3 and emb_array.shape[0] == 1:
        emb_array = jnp.squeeze(emb_array, axis=0)
        s0, s1 = emb_array.shape

    # determine which axis is the vocab axis (the larger one)
    if s0 >= s1:
        emb_matrix = jnp.asarray(emb_array)  # vocab x dim
    else:
        emb_matrix = jnp.asarray(emb_array).T  # transpose to vocab x dim

    # gather embeddings by indexing along vocab axis
    
    return emb_matrix[token_ids]


def load_checkpoint(model, checkpoint_path: str = None, use_online=True):
    """测试从检查点加载参数
    
    Args:
        model: Gemma 模型实例
        checkpoint_path: 检查点路径（本地路径或 gs:// 路径）
        use_online: 如果为 True，使用在线检查点路径（默认）
    """
    
    # 比较加载的参数和初始参数的结构
    def count_params(pytree):
        count = 0
        for leaf in jax.tree_util.tree_leaves(pytree):
            if hasattr(leaf, 'size'):
                count += leaf.size
        return count

    # 初始化模型参数（用于验证结构）
    print("\n  初始化模型参数结构...")
    key = jax.random.PRNGKey(42)
    dummy_tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
    variables = model.init(key, dummy_tokens)
    init_params = variables['params']
    loaded_params = None
    init_param_count = count_params(init_params)
    
    if use_online:
        # 使用 Gemma 3 270M 预训练检查点
        checkpoint_path = _paths.CheckpointPath.GEMMA3_270M_PT
        print(f"  使用在线检查点路径: {checkpoint_path}...")
        print(f"  类型: 预训练模型 (PT)")
        print(f"  注意: 首次使用时会自动下载检查点，可能需要一些时间...")
        print("  尝试使用 gemma.load_params 加载在线检查点...")
        loaded_params = load_params(
            checkpoint_path,
            params=init_params,
            donate=False,
            text_only=False,
        )
        assert loaded_params is not None, "Failed to 使用 gemma.load_params 加载成功"
        
        print("✓ 使用 gemma.load_params 加载成功")   
        

        del init_params, variables
        gc.collect()

    else:
        checkpoint_path = pathlib.Path(checkpoint_path)
    
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")
        
        print(f"  检查点路径: {checkpoint_path}")
                    
        # 检查关键文件
        key_files = ["manifest.ocdbt", "_METADATA", "descriptor"]
        missing_files = []
        for key_file in key_files:
            file_path = checkpoint_path / key_file
            if not file_path.exists():
                raise FileNotFoundError(f"检查点目录缺少关键文件: {key_file}")

        # 如果使用本地路径，使用 orbax 直接加载
        with ocp.PyTreeCheckpointer() as ckptr:
            # 直接恢复整个检查点（不指定 item，让 orbax 自动检测）
            print("  恢复整个检查点...")
            restored = ckptr.restore(checkpoint_path)
            
            # 检查点结构：键是扁平化的 transformer/embedder 格式
            # 需要转换为嵌套的 {embedder: {...}} 格式
            if isinstance(restored, dict):
                # 检查是否有 transformer/ 前缀的键
                transformer_keys = [k for k in restored.keys() if isinstance(k, str) and k.startswith('transformer/')]
                
                if transformer_keys:
                    print(f"  找到 {len(transformer_keys)} 个 transformer/ 前缀的键")
                    print(f"  示例键: {transformer_keys[:3]}")
                    
                    # 将扁平化的键转换为嵌套结构
                    # transformer/embedder/input_embedding -> {embedder: {input_embedding: ...}}
                    # 注意：只有 attn 下的 *_einsum 需要保留 {'w': array} 结构，其他需要提取出 array
                    loaded_params = {}
                    
                    for key, value in restored.items():
                        if isinstance(key, str) and key.startswith('transformer/'):
                            # 移除 transformer/ 前缀
                            path = key.replace('transformer/', '').split('/')
                            
                            # 检查是否需要保留 w 结构
                            # 只有 attn 下的 *_einsum 需要保留 {'w': array} 结构
                            full_path = '/'.join(path)
                            should_keep_w = (
                                'attn/' in full_path and 
                                ('attn_vec_einsum' in full_path or 'kv_einsum' in full_path or 'q_einsum' in full_path)
                            )
                            
                            # 如果值是字典且只有一个 'w' 键
                            if isinstance(value, dict) and len(value) == 1 and 'w' in value:
                                if should_keep_w:
                                    # 保留 {'w': array} 结构
                                    value = value
                                else:
                                    # 提取出 array
                                    value = value['w']
                            
                            # 构建嵌套字典
                            current = loaded_params
                            for i, part in enumerate(path):
                                if i == len(path) - 1:
                                    # 最后一个部分是值
                                    current[part] = value
                                else:
                                    # 中间部分是字典键
                                    if part not in current:
                                        current[part] = {}
                                    current = current[part]
                        else:
                            # 非 transformer/ 前缀的键直接添加
                            if isinstance(value, dict) and len(value) == 1 and 'w' in value:
                                # 检查是否需要保留 w 结构
                                should_keep_w = (
                                    'attn/' in key and 
                                    ('attn_vec_einsum' in key or 'kv_einsum' in key or 'q_einsum' in key)
                                )
                                if should_keep_w:
                                    loaded_params[key] = value
                                else:
                                    loaded_params[key] = value['w']
                            else:
                                loaded_params[key] = value
                    
                    print("✓ 参数结构转换成功（扁平化 -> 嵌套，智能提取 'w' 键）")
                elif 'transformer' in restored:
                    # 参数已经在 transformer 键下
                    loaded_params = restored['transformer']
                    print("✓ 从 transformer 键提取参数成功")
                else:
                    # 直接使用
                    loaded_params = restored
                    print("✓ 直接使用恢复的参数")
            else:
                loaded_params = restored
                print("✓ 直接使用恢复的参数")
                
    if loaded_params is None:
        raise RuntimeError("所有加载方法都失败了")
    
    print("✓ 检查点参数加载成功")
        
    loaded_param_count = count_params(loaded_params)
    assert init_param_count == loaded_param_count 
        
    # 检查参数结构是否匹配（使用展平的键进行比较）
    # init_flat = traverse_util.flatten_dict(init_params, sep='/')
    # loaded_flat = traverse_util.flatten_dict(loaded_params, sep='/')
    
    # init_keys = set(init_flat.keys())
    # loaded_keys = set(loaded_flat.keys())
    
    # missing_keys = init_keys - loaded_keys
    # extra_keys = loaded_keys - init_keys
    
    # if len(missing_keys) == 0 and len(extra_keys) == 0:
    #     # 进一步验证形状
    #     shape_mismatches = []
    #     for key in list(init_keys):  # 检查前20个
    #         def get_shape(val):
    #             if isinstance(val, dict) and 'w' in val:
    #                 return val['w'].shape if hasattr(val['w'], 'shape') else None
    #             return val.shape if hasattr(val, 'shape') else None
            
    #         init_shape = get_shape(init_flat[key])
    #         loaded_shape = get_shape(loaded_flat[key])
    #         if init_shape != loaded_shape:
    #             shape_mismatches.append((key, init_shape, loaded_shape))
        
    #     if shape_mismatches:
    #         raise RuntimeError(f"  参数结构匹配: ⚠ (键匹配但形状不匹配: {len(shape_mismatches)})")
    #     print(f"  参数结构匹配: ✓✓✓ (完全匹配)")

    return loaded_params


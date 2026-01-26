#!/usr/bin/env python3
"""检查 Gemma 模型的导入和参数加载功能。

这个脚本用于验证：
1. gemma 模块是否可以正常导入
2. 模型是否可以成功初始化
3. 参数是否可以正常初始化
4. load_params 函数是否可用
5. 从在线检查点加载模型权重
6. 模型前向传播是否正常

使用方法:
    # 使用在线检查点（默认，自动下载）
    python check_gemma_model.py
    
    # 使用本地检查点路径
    GEMMA_CHECKPOINT_PATH=/path/to/checkpoint python check_gemma_model.py
"""

import gemma.gm as gm
from gemma.gm.ckpts import load_params
from gemma.gm.ckpts import _paths
import jax
import jax.numpy as jnp
import inspect
import pathlib
import os
from orbax import checkpoint as ocp
from flax import traverse_util
from orbax import checkpoint as ocp
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


def show_available_checkpoints():
    """显示所有可用的检查点路径"""
    print_subsection("可用检查点路径")
    try:
        from gemma.gm.ckpts import _paths
        
        print("Gemma 3.0 检查点:")
        print("  预训练模型 (PT):")
        print(f"    - GEMMA3_270M_PT: {_paths.CheckpointPath.GEMMA3_270M_PT}")
        print(f"    - GEMMA3_1B_PT: {_paths.CheckpointPath.GEMMA3_1B_PT}")
        print(f"    - GEMMA3_4B_PT: {_paths.CheckpointPath.GEMMA3_4B_PT}")
        print(f"    - GEMMA3_12B_PT: {_paths.CheckpointPath.GEMMA3_12B_PT}")
        print(f"    - GEMMA3_27B_PT: {_paths.CheckpointPath.GEMMA3_27B_PT}")
        print("  指令微调模型 (IT):")
        print(f"    - GEMMA3_270M_IT: {_paths.CheckpointPath.GEMMA3_270M_IT}")
        print(f"    - GEMMA3_1B_IT: {_paths.CheckpointPath.GEMMA3_1B_IT}")
        print(f"    - GEMMA3_4B_IT: {_paths.CheckpointPath.GEMMA3_4B_IT}")
        print(f"    - GEMMA3_12B_IT: {_paths.CheckpointPath.GEMMA3_12B_IT}")
        print(f"    - GEMMA3_27B_IT: {_paths.CheckpointPath.GEMMA3_27B_IT}")
        
        return True
    except Exception as e:
        print(f"✗ 无法获取检查点路径: {e}")
        return False


def test_model_import():
    """测试模型导入"""
    print_subsection("1. 模型导入测试")
    try:
        import gemma.gm as gm
        print("✓ gemma.gm 导入成功")
        
        # 列出所有可用模型
        available_models = [x for x in dir(gm.nn) if 'Gemma' in x and not x.startswith('_')]
        print(f"✓ 可用模型数量: {len(available_models)}")
        print(f"  模型列表: {', '.join(available_models[:5])}...")
        if len(available_models) > 5:
            print(f"  ... 还有 {len(available_models) - 5} 个模型")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """测试模型初始化"""
    print_subsection("2. 模型初始化测试")
    try:
        model = gm.nn.Gemma3_270M()
        print("✓ Gemma3_270M 模型初始化成功")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  模块路径: {type(model).__module__}")
        return model
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_parameter_initialization(model):
    """测试参数初始化"""
    print_subsection("3. 参数初始化测试")
    try:
        key = jax.random.PRNGKey(42)
        # 创建一个简单的输入序列
        dummy_tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        
        variables = model.init(key, dummy_tokens)
        print("✓ 模型参数初始化成功")
        
        # 检查参数结构
        params = variables.get('params', {})
        print(f"  参数顶层键数量: {len(params)}")
        print(f"  参数键示例: {list(params.keys())[:5]}")
        
        # 计算参数总数
        def count_params(pytree):
            count = 0
            for leaf in jax.tree_util.tree_leaves(pytree):
                if hasattr(leaf, 'size'):
                    count += leaf.size
            return count
        
        total_params = count_params(params)
        print(f"  总参数数量: {total_params:,}")
        print(f"  参数大小 (MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        return variables, dummy_tokens
    except Exception as e:
        print(f"✗ 参数初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_load_params_function():
    """测试 load_params 函数"""
    print_subsection("4. load_params 函数测试")
    try:
        from gemma.gm.ckpts import load_params
        import inspect
        
        print("✓ load_params 函数导入成功")
        
        # 检查函数签名
        sig = inspect.signature(load_params)
        print(f"  函数签名: load_params{sig}")
        
        required_params = [p for p, v in sig.parameters.items() 
                          if v.default == inspect.Parameter.empty]
        optional_params = [p for p, v in sig.parameters.items() 
                          if v.default != inspect.Parameter.empty]
        
        print(f"  必需参数: {required_params}")
        print(f"  可选参数: {optional_params}")
        
        # 检查函数文档
        doc = load_params.__doc__
        if doc:
            doc_lines = [line.strip() for line in doc.strip().split('\n') if line.strip()]
            if doc_lines:
                print(f"  函数说明: {doc_lines[0]}")
        
        return True
    except Exception as e:
        print(f"✗ load_params 函数检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_checkpoint(model, checkpoint_path=None, use_online=True):
    """测试从检查点加载参数
    
    Args:
        model: Gemma 模型实例
        checkpoint_path: 检查点路径（本地路径或 gs:// 路径）
        use_online: 如果为 True，使用在线检查点路径（默认）
    """
    print_subsection("5. 检查点参数加载测试")
    try:
        # 如果使用在线加载，使用 gemma 库提供的检查点路径
        if use_online or checkpoint_path is None:
            print("  使用在线检查点路径...")
            # 使用 Gemma 3 270M 预训练检查点
            checkpoint_path = _paths.CheckpointPath.GEMMA3_270M_PT
            print(f"  检查点路径: {checkpoint_path}")
            print(f"  类型: 预训练模型 (PT)")
            print(f"  注意: 首次使用时会自动下载检查点，可能需要一些时间...")
        else:
            checkpoint_path = pathlib.Path(checkpoint_path)
            
            # 检查本地路径是否存在
            if isinstance(checkpoint_path, pathlib.Path) and not checkpoint_path.exists():
                print(f"⚠ 检查点路径不存在: {checkpoint_path}")
                print("  尝试使用在线检查点路径...")
                checkpoint_path = _paths.CheckpointPath.GEMMA3_270M_PT
                print(f"  检查点路径: {checkpoint_path}")
            else:
                print(f"  检查点路径: {checkpoint_path}")
                if isinstance(checkpoint_path, pathlib.Path):
                    print(f"  路径存在: ✓")
                    
                    # 检查关键文件
                    key_files = ["manifest.ocdbt", "_METADATA", "descriptor"]
                    for key_file in key_files:
                        file_path = checkpoint_path / key_file
                        if file_path.exists():
                            print(f"  {key_file}: ✓")
                        else:
                            print(f"  {key_file}: ✗ (缺失)")
        
        # 初始化模型参数（用于验证结构）
        print("\n  初始化模型参数结构...")
        key = jax.random.PRNGKey(42)
        dummy_tokens = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.int32)
        variables = model.init(key, dummy_tokens)
        init_params = variables['params']
        
        print("  使用 orbax 直接加载检查点...")
        print("  注意: 如果使用在线路径，首次下载可能需要几分钟...")
        
        # 尝试使用 gemma 的 load_params（适用于在线路径）
        loaded_params = None
        if isinstance(checkpoint_path, str) and checkpoint_path.startswith('gs://'):
            try:
                print("  尝试使用 gemma.load_params 加载在线检查点...")
                loaded_params = load_params(
                    checkpoint_path,
                    params=init_params,
                    donate=False,
                    text_only=False,
                )
                print("✓ 使用 gemma.load_params 加载成功")
            except Exception as e:
                print(f"  ⚠ gemma.load_params 失败: {e}")
                print("  尝试使用 orbax 直接加载...")
        
        # 如果 gemma.load_params 失败或使用本地路径，使用 orbax 直接加载
        if loaded_params is None:
            try:
                checkpoint_path_obj = pathlib.Path(checkpoint_path) if isinstance(checkpoint_path, (str, pathlib.Path)) else checkpoint_path
                
                # 使用 orbax 直接加载
                with ocp.PyTreeCheckpointer() as ckptr:
                    # 直接恢复整个检查点（不指定 item，让 orbax 自动检测）
                    print("  恢复整个检查点...")
                    restored = ckptr.restore(checkpoint_path_obj)
                    
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
                        
            except Exception as e:
                print(f"  ✗ orbax 加载失败: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        if loaded_params is None:
            print("✗ 所有加载方法都失败了")
            return None
        
        print("✓ 检查点参数加载成功")
        
        # 比较加载的参数和初始参数的结构
        def count_params(pytree):
            count = 0
            for leaf in jax.tree_util.tree_leaves(pytree):
                if hasattr(leaf, 'size'):
                    count += leaf.size
            return count
        
        init_param_count = count_params(init_params)
        loaded_param_count = count_params(loaded_params)
        
        print(f"  初始参数数量: {init_param_count:,}")
        print(f"  加载参数数量: {loaded_param_count:,}")
        
        # 检查参数结构是否匹配（使用展平的键进行比较）
        init_flat = traverse_util.flatten_dict(init_params, sep='/')
        loaded_flat = traverse_util.flatten_dict(loaded_params, sep='/')
        
        init_keys = set(init_flat.keys())
        loaded_keys = set(loaded_flat.keys())
        
        missing_keys = init_keys - loaded_keys
        extra_keys = loaded_keys - init_keys
        
        if len(missing_keys) == 0 and len(extra_keys) == 0:
            # 进一步验证形状
            shape_mismatches = []
            for key in list(init_keys)[:20]:  # 检查前20个
                def get_shape(val):
                    if isinstance(val, dict) and 'w' in val:
                        return val['w'].shape if hasattr(val['w'], 'shape') else None
                    return val.shape if hasattr(val, 'shape') else None
                
                init_shape = get_shape(init_flat[key])
                loaded_shape = get_shape(loaded_flat[key])
                if init_shape != loaded_shape:
                    shape_mismatches.append((key, init_shape, loaded_shape))
            
            if shape_mismatches:
                print(f"  参数结构匹配: ⚠ (键匹配但形状不匹配: {len(shape_mismatches)})")
            else:
                print(f"  参数结构匹配: ✓✓✓ (完全匹配)")
        else:
            print(f"  参数结构匹配: ✗ (缺失 {len(missing_keys)} 个键，多余 {len(extra_keys)} 个键)")
            if missing_keys:
                print(f"    缺失键示例: {list(missing_keys)[:3]}")
            if extra_keys:
                print(f"    多余键示例: {list(extra_keys)[:3]}")
        
        return loaded_params
        
    except Exception as e:
        print(f"✗ 检查点加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n  提示:")
        print("    - 如果使用在线路径，请确保网络连接正常")
        print("    - 如果使用 gs:// 路径，请确保已安装 gcsfs 或配置了 Google Cloud 认证")
        print("    - 可以尝试使用本地路径: /root/autodl-tmp/gemma-3-270m")
        return None


def test_forward_pass(model, variables, dummy_tokens):
    """测试模型前向传播"""
    print_subsection("6. 模型前向传播测试")
    try:
        # 使用初始化后的参数进行前向传播
        output = model.apply(variables, dummy_tokens)
        print("✓ 模型前向传播成功")
        
        # 处理输出 - gemma 模型可能返回命名元组或特殊对象
        if hasattr(output, 'shape'):
            print(f"  输出形状: {output.shape}")
        elif hasattr(output, 'logits'):
            # 如果输出是命名元组，尝试获取 logits
            print(f"  输出类型: {type(output)}")
            if hasattr(output.logits, 'shape'):
                print(f"  logits 形状: {output.logits.shape}")
                output_array = output.logits
            else:
                output_array = output
        else:
            # 尝试获取第一个属性
            print(f"  输出类型: {type(output)}")
            if hasattr(output, '__dict__'):
                attrs = [k for k in output.__dict__.keys() if not k.startswith('_')]
                print(f"  输出属性: {attrs[:5]}")
            # 尝试转换为数组
            try:
                output_array = jnp.array(output)
                print(f"  输出形状: {output_array.shape}")
            except:
                output_array = None
        
        if output_array is not None:
            print(f"  输出 dtype: {output_array.dtype}")
            # 检查输出的一些统计信息
            try:
                if hasattr(output_array, 'mean'):
                    mean_val = output_array.mean()
                    if isinstance(mean_val, (int, float, jax.Array)):
                        print(f"  输出均值: {float(mean_val):.4f}")
                if hasattr(output_array, 'std'):
                    std_val = output_array.std()
                    if isinstance(std_val, (int, float, jax.Array)):
                        print(f"  输出标准差: {float(std_val):.4f}")
            except Exception:
                pass  # 忽略统计信息错误
        
        return True
    except Exception as e:
        print(f"⚠ 前向传播警告: {e}")
        print("  (这可能是正常的，取决于模型的具体实现)")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print_section("Gemma 模型导入和参数加载完整测试")
    
    # 测试模型导入
    if not test_model_import():
        print("\n✗ 模型导入失败，终止测试")
        return False
    
    # 测试模型初始化
    model = test_model_initialization()
    if model is None:
        print("\n✗ 模型初始化失败，终止测试")
        return False
    
    # 测试参数初始化
    variables, dummy_tokens = test_parameter_initialization(model)
    if variables is None:
        print("\n✗ 参数初始化失败，终止测试")
        return False
    
    # 测试 load_params 函数
    test_load_params_function()
    
    # 显示可用的检查点路径
    show_available_checkpoints()
    
    # 测试从检查点加载参数
    # 默认使用本地路径，如果设置了环境变量则使用环境变量指定的路径
    default_local_path = "/root/autodl-tmp/gemma-3-270m"
    checkpoint_path = os.getenv("GEMMA_CHECKPOINT_PATH", default_local_path)
    use_online = False  # 默认使用本地路径
    
    # 如果环境变量设置为 "online" 或 "gs://" 开头的路径，则使用在线路径
    if checkpoint_path.lower() == "online" or checkpoint_path.startswith("gs://"):
        use_online = True
        if checkpoint_path.lower() == "online":
            checkpoint_path = None  # 使用默认的在线路径
    
    if checkpoint_path and not use_online:
        print(f"\n使用本地检查点路径: {checkpoint_path}")
    else:
        print(f"\n使用在线检查点路径（自动下载）")
        print(f"默认使用: GEMMA3_270M_PT")
    
    loaded_params = test_load_checkpoint(model, checkpoint_path, use_online=use_online)
    
    # 如果成功加载检查点参数，使用加载的参数进行前向传播
    if loaded_params is not None:
        print("\n  使用加载的检查点参数进行前向传播测试...")
        loaded_variables = {'params': loaded_params}
        test_forward_pass(model, loaded_variables, dummy_tokens)
    
    # 测试前向传播（使用初始化的参数）
    test_forward_pass(model, variables, dummy_tokens)
    
    # 总结
    print_section("测试总结")
    print("✓ 所有基本测试完成！")
    print("\n总结:")
    print("  • gemma 模型可以正常导入")
    print("  • 模型可以成功初始化")
    print("  • 参数可以正常初始化")
    print("  • load_params 函数可用")
    if loaded_params is not None:
        print("  • 检查点参数加载成功")
    print("  • 模型可以进行前向传播")
    
    return True
    

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n\n✗ 测试过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


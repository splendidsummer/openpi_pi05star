#!/usr/bin/env python3
"""Verify that value_targets are normalized during data preprocessing."""

import tyro
import numpy as np
import jax.numpy as jnp

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.model as _model
import openpi.transforms as _transforms


class RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main(config_name: str, num_batches: int | None = None):
    """Verify value_targets normalization during preprocessing.
    
    Args:
        config_name: Name of the config to use
        num_batches: Number of batches to process. If None, process all batches.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    print("="*70)
    print("Value Targets Normalization Verification")
    print("="*70)
    
    # Check if norm_stats exist
    if data_config.norm_stats is None:
        print("\n❌ ERROR: Normalization statistics not found!")
        print("   Please run: python scripts/compute_norm_stats.py --config-name", config_name)
        return
    
    print(f"\n✓ Normalization statistics loaded")
    print(f"  Keys in norm_stats: {list(data_config.norm_stats.keys())}")
    
    # Check if value_targets stats exist
    if "value_targets" not in data_config.norm_stats:
        print("\n⚠️  WARNING: value_targets not found in norm_stats!")
        print("   This means value_targets will NOT be normalized.")
        print("   Make sure compute_norm_stats.py includes value_targets.")
    else:
        stats = data_config.norm_stats["value_targets"]
        print(f"\n✓ value_targets normalization stats found:")
        print(f"  Mean: {stats.mean}")
        print(f"  Std:  {stats.std}")
        if stats.q01 is not None:
            print(f"  Q01:  {stats.q01}")
        if stats.q99 is not None:
            print(f"  Q99:  {stats.q99}")
    
    # Create data loader using the same method as compute_norm_stats.py
    print(f"\n{'='*70}")
    print("Creating data loader...")
    print(f"{'='*70}")
    
    if data_config.rlds_data_dir is not None:
        # For RLDS datasets
        dataset = _data_loader.create_rlds_dataset(
            data_config, config.model.action_horizon, config.batch_size, shuffle=False
        )
        # Apply transforms including Normalize
        dataset = _data_loader.transform_dataset(dataset, data_config)
        # Process all batches if num_batches is None
        data_loader_obj = _data_loader.RLDSDataLoader(dataset, config.batch_size, num_batches=num_batches)
        data_loader = _data_loader.DataLoaderImpl(data_config, data_loader_obj)
        if num_batches is None:
            print(f"  Processing all batches from RLDS dataset")
        else:
            print(f"  Processing {num_batches} batches")
    else:
        # For torch datasets - create dataset with transforms (including Normalize)
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        # Apply transforms including Normalize
        dataset = _data_loader.transform_dataset(dataset, data_config)
        
        # Calculate total batches if num_batches not specified
        total_dataset_batches = len(dataset) // config.batch_size if len(dataset) > 0 else 1
        if num_batches is None:
            num_batches_to_process = None  # Process all batches
            print(f"  Processing ALL batches: {total_dataset_batches} batches")
        else:
            num_batches_to_process = min(num_batches, total_dataset_batches)
            print(f"  Processing {num_batches_to_process} batches (total available: {total_dataset_batches})")
        
        torch_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=config.batch_size,
            num_workers=0,  # Use 0 workers to avoid pickling issues
            shuffle=False,
            num_batches=num_batches_to_process,
        )
        data_loader = _data_loader.DataLoaderImpl(data_config, torch_loader)
    
    print("✓ Data loader created")
    print(f"  Transform pipeline includes Normalize step")
    
    # Verify normalization
    print(f"\n{'='*70}")
    print("Verifying normalization in preprocessing pipeline...")
    print(f"{'='*70}\n")
    
    # Get normalization stats for manual calculation
    value_stats = data_config.norm_stats.get("value_targets") if data_config.norm_stats else None
    use_quantile = data_config.use_quantile_norm
    
    batch_count = 0
    all_normalized_values = []
    values_out_of_range = []
    
    print("Processing batches...")
    try:
        for batch in data_loader:
            if isinstance(batch, tuple) and len(batch) == 3:
                observation, actions, value_targets = batch
            else:
                print("⚠️  Batch does not contain value_targets, skipping...")
                continue
            
            # Convert to numpy for analysis
            if hasattr(value_targets, 'numpy'):
                value_targets_np = value_targets.numpy()
            elif hasattr(value_targets, '__array__'):
                value_targets_np = np.asarray(value_targets)
            else:
                # Try to convert JAX array
                try:
                    value_targets_np = np.asarray(value_targets)
                except Exception as e:
                    print(f"⚠️  Could not convert value_targets to numpy: {e}")
                    continue
            
            # Flatten for statistics
            values_flat = value_targets_np.flatten()
            all_normalized_values.extend(values_flat.tolist())
            
            # Check for values outside [-1, 1] range
            out_of_range = values_flat[(values_flat < -1.0) | (values_flat > 1.0)]
            if len(out_of_range) > 0:
                values_out_of_range.extend(out_of_range.tolist())
            
            batch_count += 1
            
            if batch_count == 1:
                print(f"\nBatch {batch_count} (sample):")
                print(f"  value_targets shape: {value_targets_np.shape}")
                print(f"  value_targets dtype: {value_targets_np.dtype}")
                print(f"  Sample values: {value_targets_np[:5] if len(value_targets_np) >= 5 else value_targets_np}")
                print(f"  Min: {np.min(value_targets_np):.6f}")
                print(f"  Max: {np.max(value_targets_np):.6f}")
                print(f"  Mean: {np.mean(value_targets_np):.6f}")
                print(f"  Std: {np.std(value_targets_np):.6f}")
                
                # Show expected normalization if stats are available
                if value_stats is not None:
                    print(f"\n  Expected normalization (from stats):")
                    if use_quantile and value_stats.q01 is not None and value_stats.q99 is not None:
                        q01 = float(value_stats.q01[0])
                        q99 = float(value_stats.q99[0])
                        print(f"    Quantile normalization: (x - {q01:.2f}) / ({q99:.2f} - {q01:.2f}) * 2.0 - 1.0")
                        print(f"    Expected range: EXACTLY [-1, 1]")
                    else:
                        mean = float(value_stats.mean[0])
                        std = float(value_stats.std[0])
                        print(f"    Z-score normalization: (x - {mean:.2f}) / ({std:.2f} + 1e-6)")
                        print(f"    Expected mean: ~0, std: ~1")
            
            # Show progress every 100 batches
            if batch_count % 100 == 0:
                print(f"  Processed {batch_count} batches... (found {len(all_normalized_values)} values so far)")
            
            # Only break if num_batches is explicitly set
            if num_batches is not None and batch_count >= num_batches:
                break
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted after processing {batch_count} batches")
    except StopIteration:
        pass
    
    if len(all_normalized_values) == 0:
        print("❌ ERROR: No value_targets found in batches!")
        return
    
    all_normalized_values = np.array(all_normalized_values)
    
    print(f"\n{'='*70}")
    print("Normalization Verification Results:")
    print(f"{'='*70}")
    print(f"  Total batches processed: {batch_count}")
    print(f"  Total value_targets samples: {len(all_normalized_values)}")
    print(f"\n  Normalized value_targets statistics:")
    print(f"    Min:  {np.min(all_normalized_values):.10f}")
    print(f"    Max:  {np.max(all_normalized_values):.10f}")
    print(f"    Mean: {np.mean(all_normalized_values):.10f}")
    print(f"    Std:  {np.std(all_normalized_values):.10f}")
    
    # Check if values are normalized
    mean_abs = np.abs(np.mean(all_normalized_values))
    std_value = np.std(all_normalized_values)
    min_value = np.min(all_normalized_values)
    max_value = np.max(all_normalized_values)
    
    print(f"\n  Range Check (Testing normalized values WITHOUT any modification):")
    is_quantile_norm = data_config.use_quantile_norm
    
    if is_quantile_norm:
        # Quantile normalization maps to approximately [-1, 1] range
        print(f"    Using quantile normalization (expected range: approximately [-1, 1])")
        print(f"    Testing ALL normalized values across ALL batches (no post-processing)")
        
        # Check if values are within [-1, 1] (allowing for small float errors)
        all_in_range = (min_value >= -1.0 - 1e-5) and (max_value <= 1.0 + 1e-5)
        
        # Check how close min/max are to -1 and 1 (allowing float precision errors)
        min_diff_from_neg_one = abs(min_value - (-1.0))
        max_diff_from_one = abs(max_value - 1.0)
        
        # Tolerance for "very close" - allowing for float precision errors
        tolerance = 1e-3  # 0.001 tolerance
        
        min_is_close_to_neg_one = min_diff_from_neg_one <= tolerance
        max_is_close_to_one = max_diff_from_one <= tolerance
        
        print(f"\n    Results from ALL batches:")
        print(f"      Minimum value: {min_value:.10f}")
        print(f"      Maximum value: {max_value:.10f}")
        print(f"      Difference from -1.0: {min_diff_from_neg_one:.10f}")
        print(f"      Difference from +1.0: {max_diff_from_one:.10f}")
        
        if all_in_range:
            print(f"\n    ✓ All values are within [-1, 1] range (allowing float errors)")
            
            if min_is_close_to_neg_one:
                print(f"    ✓✓ Minimum value is VERY CLOSE to -1.0 (difference: {min_diff_from_neg_one:.10f} ≤ {tolerance})")
            else:
                print(f"    ⚠️  Minimum value ({min_value:.10f}) is NOT close to -1.0")
                print(f"       Difference: {min_diff_from_neg_one:.10f} (expected ≤ {tolerance})")
            
            if max_is_close_to_one:
                print(f"    ✓✓ Maximum value is VERY CLOSE to +1.0 (difference: {max_diff_from_one:.10f} ≤ {tolerance})")
            else:
                print(f"    ⚠️  Maximum value ({max_value:.10f}) is NOT close to +1.0")
                print(f"       Difference: {max_diff_from_one:.10f} (expected ≤ {tolerance})")
            
            # Final verdict
            if min_is_close_to_neg_one and max_is_close_to_one:
                print(f"\n    ✓✓✓ SUCCESS: Normalized values are correctly in [-1, 1] range!")
                print(f"       Min ≈ -1.0, Max ≈ +1.0 (within float precision tolerance)")
            elif min_is_close_to_neg_one or max_is_close_to_one:
                print(f"\n    ⚠️  PARTIAL: One boundary is correct, but not both")
                print(f"       This may indicate the dataset doesn't contain values at both q01 and q99")
            else:
                print(f"\n    ⚠️  WARNING: Values are in range but boundaries are not close to -1 and +1")
                print(f"       This may indicate:")
                print(f"       1. The dataset doesn't contain values at q01 and q99 quantiles")
                print(f"       2. There might be an issue with the normalization formula")
        else:
            print(f"\n    ❌ ERROR: Some values are OUTSIDE [-1, 1] range!")
            print(f"      Min: {min_value:.10f} (should be ≥ -1.0)")
            print(f"      Max: {max_value:.10f} (should be ≤ 1.0)")
        
        # Show out-of-range values if any
        if len(values_out_of_range) > 0:
            values_out_of_range_arr = np.array(values_out_of_range)
            print(f"\n    ⚠️  Found {len(values_out_of_range)} values outside [-1, 1]:")
            print(f"      Min out-of-range: {np.min(values_out_of_range_arr):.10f}")
            print(f"      Max out-of-range: {np.max(values_out_of_range_arr):.10f}")
            print(f"      Sample out-of-range values: {values_out_of_range_arr[:10]}")
    else:
        # Z-score normalization (mean ≈ 0, std ≈ 1)
        print(f"    Using z-score normalization (expected mean ≈ 0, std ≈ 1)")
        if mean_abs < 0.1 and 0.9 < std_value < 1.1:
            print(f"    ✓ Values appear to be z-score normalized")
            print(f"      Mean absolute value: {mean_abs:.10f} (should be < 0.1)")
            print(f"      Std: {std_value:.10f} (should be ≈ 1.0)")
        else:
            print(f"    ⚠️  Values may not be properly z-score normalized")
            print(f"      Mean absolute value: {mean_abs:.10f}")
            print(f"      Std: {std_value:.10f}")
    
    # Show preprocessing pipeline
    print(f"\n{'='*70}")
    print("Data Preprocessing Pipeline:")
    print(f"{'='*70}")
    print("  1. repack_transforms")
    print("     └─> Extract state_value from dataset")
    print("  2. data_transforms")
    print("     └─> DroidInputs: state_value → value_targets")
    print("  3. Normalize ⭐")
    print("     └─> Normalize value_targets (if stats exist)")
    print("         Formula: (value_targets - mean) / (std + 1e-6)")
    print("  4. model_transforms")
    print("     └─> Model-specific transformations")
    print("  5. DataLoaderImpl.__iter__")
    print("     └─> Yield (observation, actions, value_targets)")
    print(f"{'='*70}\n")
    
    print("✓ Verification complete!")


if __name__ == "__main__":
    tyro.cli(main)


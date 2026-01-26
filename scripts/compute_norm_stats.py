"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    print(data_config)
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    # Add value_targets/reward if they are defined in the repack_transforms structure
    # Note: state_value from repack_transforms is renamed to value_targets by data_transforms (DroidInputs)
    potential_keys = []
    for transform in data_config.repack_transforms.inputs:
        if hasattr(transform, "structure"):
            # Check top-level keys directly in the structure dict
            if isinstance(transform.structure, dict):
                if "state_value" in transform.structure:
                    # state_value is renamed to value_targets by DroidInputs transform
                    potential_keys.append("value_targets")
                if "reward" in transform.structure:
                    potential_keys.append("reward")
    
    # Get the first batch to check which keys actually exist
    first_batch = None
    for batch in data_loader:
        first_batch = batch
        break
    
    # Only add keys that actually exist in the batch
    if first_batch is not None:
        for key in potential_keys:
            if key in first_batch and key not in keys:
                keys.append(key)
    
    stats = {key: normalize.RunningStats() for key in keys}

    # Reset data loader to start from beginning
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                value = np.asarray(batch[key])
                # RunningStats expects arrays where the last dimension is the feature dimension
                # For scalar values (like value_targets and reward), reshape to (batch_size, 1)
                if key in ["value_targets", "reward"]:
                    # Reshape scalar/1D arrays to (batch_size, 1)
                    if value.ndim == 0:
                        # Scalar: convert to (1, 1)
                        value = value[np.newaxis, np.newaxis]
                    elif value.ndim == 1:
                        # 1D array: add feature dimension (batch_size,) -> (batch_size, 1)
                        value = value[:, np.newaxis]
                    elif value.ndim == 2 and value.shape[-1] > 1:
                        # 2D array with last dim > 1: take first element
                        value = value[:, 0:1]
                stats[key].update(value)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)

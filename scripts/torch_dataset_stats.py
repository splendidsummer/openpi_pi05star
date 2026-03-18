import numpy as np
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

config = _config.get_config('pi05_droid_100_value')
data_config = config.data.create(config.assets_dirs, config.model)

dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
dataset = _data_loader.transform_dataset(dataset, data_config)

num_batches = len(dataset) // config.batch_size
loader = _data_loader.TorchDataLoader(
    dataset,
    local_batch_size=config.batch_size,
    num_workers=0,
    shuffle=False,
    num_batches=num_batches,
    framework='pytorch',
)

min_v = float('inf')
max_v = float('-inf')
count = 0
for batch in loader:
    if 'value_targets' not in batch:
        continue
    arr = batch['value_targets'].detach().cpu().numpy().reshape(-1)
    if arr.size == 0:
        continue
    min_v = min(min_v, float(arr.min()))
    max_v = max(max_v, float(arr.max()))
    count += int(arr.size)

print({'batches': num_batches, 'num_values': count, 'min': min_v, 'max': max_v})
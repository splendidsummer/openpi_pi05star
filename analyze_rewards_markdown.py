import pandas as pd
import os
import numpy as np
from pathlib import Path
import json

def main():
    data_dir = "/root/autodl-tmp/huggingface/lerobot/local_pkl_dataset_rewards/data/chunk-000/"
    output_dir = "reward_analysis_output"
    Path(output_dir).mkdir(exist_ok=True)

    # Get all parquet files
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    print(f"Found {len(files)} parquet files")

    all_rewards = []
    all_state_values = []
    episode_lengths = []

    # Process each file
    for i, filename in enumerate(files):
        filepath = os.path.join(data_dir, filename)
        if i % 20 == 0:
            print(f"Processing file {i+1}/{len(files)}: {filename}")

        try:
            df = pd.read_parquet(filepath)

            # Extract reward and state_value columns
            if 'reward' in df.columns:
                rewards = df['reward'].values.tolist()
                all_rewards.extend(rewards)

            if 'state_value' in df.columns:
                state_values = df['state_value'].values.tolist()
                all_state_values.extend(state_values)

            episode_lengths.append(len(df))

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    print(f"\nTotal reward values collected: {len(all_rewards)}")
    print(f"Total state_value values collected: {len(all_state_values)}")

    # Convert to numpy arrays for calculations
    rewards_array = np.array(all_rewards, dtype=np.float32)
    state_values_array = np.array(all_state_values, dtype=np.float32)

    # Calculate statistics
    reward_stats = {
        'min': float(np.min(rewards_array)),
        'max': float(np.max(rewards_array)),
        'mean': float(np.mean(rewards_array)),
        'std': float(np.std(rewards_array)),
        'median': float(np.median(rewards_array)),
        'q1': float(np.percentile(rewards_array, 25)),
        'q3': float(np.percentile(rewards_array, 75)),
        'total_count': len(rewards_array)
    }

    state_value_stats = {
        'min': float(np.min(state_values_array)),
        'max': float(np.max(state_values_array)),
        'mean': float(np.mean(state_values_array)),
        'std': float(np.std(state_values_array)),
        'median': float(np.median(state_values_array)),
        'q1': float(np.percentile(state_values_array, 25)),
        'q3': float(np.percentile(state_values_array, 75)),
        'total_count': len(state_values_array)
    }

    # Save all values to CSV files
    pd.DataFrame({'reward': all_rewards}).to_csv(f'{output_dir}/all_rewards.csv', index=False)
    pd.DataFrame({'state_value': all_state_values}).to_csv(f'{output_dir}/all_state_values.csv', index=False)

    # Save statistics to JSON
    with open(f'{output_dir}/statistics.json', 'w') as f:
        json.dump({
            'reward': reward_stats,
            'state_value': state_value_stats,
            'episode_info': {
                'total_episodes': len(files),
                'total_frames': len(all_rewards),
                'avg_episode_length': np.mean(episode_lengths),
                'min_episode_length': min(episode_lengths),
                'max_episode_length': max(episode_lengths)
            }
        }, f, indent=2)

    # Generate markdown report
    markdown_content = f"""# Reward and State Value Analysis Report

## Dataset Information
- **Dataset path**: `{data_dir}`
- **Total episodes**: {len(files)}
- **Total frames**: {len(all_rewards)}
- **Average episode length**: {np.mean(episode_lengths):.1f} frames
- **Min episode length**: {min(episode_lengths)} frames
- **Max episode length**: {max(episode_lengths)} frames

## Reward Statistics
| Metric | Value |
|--------|-------|
| Minimum | {reward_stats['min']} |
| Maximum | {reward_stats['max']} |
| Mean | {reward_stats['mean']:.6f} |
| Standard Deviation | {reward_stats['std']:.6f} |
| Median | {reward_stats['median']:.6f} |
| Q1 (25th percentile) | {reward_stats['q1']:.6f} |
| Q3 (75th percentile) | {reward_stats['q3']:.6f} |
| Total count | {reward_stats['total_count']} |

## State Value Statistics
| Metric | Value |
|--------|-------|
| Minimum | {state_value_stats['min']} |
| Maximum | {state_value_stats['max']} |
| Mean | {state_value_stats['mean']:.6f} |
| Standard Deviation | {state_value_stats['std']:.6f} |
| Median | {state_value_stats['median']:.6f} |
| Q1 (25th percentile) | {state_value_stats['q1']:.6f} |
| Q3 (75th percentile) | {state_value_stats['q3']:.6f} |
| Total count | {state_value_stats['total_count']} |

## Value Samples

### First 20 Reward Values
```
{all_rewards[:20]}
```

### First 20 State Value Values
```
{all_state_values[:20]}
```

## Files Generated
- `{output_dir}/all_rewards.csv`: All reward values
- `{output_dir}/all_state_values.csv`: All state value values
- `{output_dir}/statistics.json`: Detailed statistics in JSON format

## Notes
- Analysis completed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- The dataset contains `state_value` field (assumed to be the value target)
- Reward values are typically -1.0 for most steps (as observed in samples)
- State values appear to be decreasing (negative values increasing in magnitude)
"""

    # Write markdown report
    with open(f'{output_dir}/analysis_report.md', 'w') as f:
        f.write(markdown_content)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Report saved to: {output_dir}/analysis_report.md")
    print(f"All reward values saved to: {output_dir}/all_rewards.csv")
    print(f"All state values saved to: {output_dir}/all_state_values.csv")
    print(f"Statistics saved to: {output_dir}/statistics.json")

    # Also print key statistics to console
    print(f"\nKey Statistics:")
    print(f"Reward - Min: {reward_stats['min']}, Max: {reward_stats['max']}")
    print(f"State Value - Min: {state_value_stats['min']}, Max: {state_value_stats['max']}")

if __name__ == "__main__":
    main()
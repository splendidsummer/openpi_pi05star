# Reward and State Value Analysis Report

## Dataset Information
- **Dataset path**: `/root/autodl-tmp/huggingface/lerobot/local_pkl_dataset_rewards/data/chunk-000/`
- **Total episodes**: 100
- **Total frames**: 33865
- **Average episode length**: 338.6 frames
- **Min episode length**: 211 frames
- **Max episode length**: 541 frames

## Reward Statistics
| Metric | Value |
|--------|-------|
| Minimum | -1000.0 |
| Maximum | 0.0 |
| Mean | -1.882918 |
| Standard Deviation | 29.720732 |
| Median | -1.000000 |
| Q1 (25th percentile) | -1.000000 |
| Q3 (75th percentile) | -1.000000 |
| Total count | 33865 |

## State Value Statistics
| Metric | Value |
|--------|-------|
| Minimum | -999.0 |
| Maximum | -0.0 |
| Mean | -177.103592 |
| Standard Deviation | 113.265572 |
| Median | -169.000000 |
| Q1 (25th percentile) | -255.000000 |
| Q3 (75th percentile) | -84.000000 |
| Total count | 33865 |

## Value Samples

### First 20 Reward Values
```
[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
```

### First 20 State Value Values
```
[-362.0, -361.0, -360.0, -359.0, -358.0, -357.0, -356.0, -355.0, -354.0, -353.0, -352.0, -351.0, -350.0, -349.0, -348.0, -347.0, -346.0, -345.0, -344.0, -343.0]
```

## Files Generated
- `reward_analysis_output/all_rewards.csv`: All reward values
- `reward_analysis_output/all_state_values.csv`: All state value values
- `reward_analysis_output/statistics.json`: Detailed statistics in JSON format

## Notes
- Analysis completed on 2026-02-13 14:21:57
- The dataset contains `state_value` field (assumed to be the value target)
- Reward values are typically -1.0 for most steps (as observed in samples)
- State values appear to be decreasing (negative values increasing in magnitude)

""" 
TODO: 1. Obviously the current code that has a bottle like to use the GPU memory 
      and the total models loading and training shouldn't ooccupy that large memories.
      2. The next step it should be to the include to analyze the each line of codes
        w.r.t loading model, params, and training procedure of GPU memory usage.
      
      

Returns:
    _type_: _description_
"""

import dataclasses
import sys
import os
import logging
import re 

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import tqdm
import jax

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openpi import transforms
from openpi.models import model as _model
from openpi.models import value_config
from openpi.policies import policy as _policy_lib
from openpi.policies import droid_policy 
from openpi.training import config as _config
import openpi.shared.download as download
from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints

import io
import json
from PIL import Image
import os

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logger.warning("pyarrow not found. Will use dummy data and cannot save to parquet.")

import argparse

# Configuration placeholders
DEFAULT_PARQUET_PATH = "/root/autodl-tmp/huggingface/lerobot/SummerZhang/droid_100"
DEFAULT_CHECKPOINT_DIR = "/root/autodl-tmp/openpi_pi05star/checkpoints/pi05_droid_100_value/experiment-20260208_135947/301"

GAMMA = 1.00 # since there isn't any gamma in the advantage formula in pi_star paper 

parser = argparse.ArgumentParser(description="Droid Inference Final")
parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Path to checkpoint directory")
parser.add_argument("--data_dir", type=str, default=DEFAULT_PARQUET_PATH, help="Path to data directory")
args = parser.parse_args()



def test_episode_data():
    
    data_parquet_paths =  get_filtered_file_paths(args.data_dir + '/data')
    if data_parquet_paths:
         logger.info(f"Found {len(data_parquet_paths)} parquet files. Sample path: {data_parquet_paths[10]}")
    else:
         logger.warning("No parquet files found in the specified directory.")
         
    sample_episode_path = data_parquet_paths[10] if data_parquet_paths else None
    
    sample_episode_index = re.search(r'(\d+)\.parquet', sample_episode_path).group(1) if sample_episode_path else None
    sample_episode_index = int(sample_episode_index)
    
    logger.info(f"Sample episode index extracted from path: {sample_episode_index}")
    
    episode_task = load_episode_metadata(args.data_dir, episode_index=sample_episode_index)
    logger.info(f"Sample episode task prompt: {episode_task}")     
    
    
def main():
    
    PARQUET_FOLDER = args.data_dir + '/data' 
    
    # 1. Create the policy by assigning a TrainConfig
    logger.info("1. Creating TrainConfig and Policy...")

    # Using existing config pi05_droid_value (defined in config.py)
    config = _config.get_config("pi05_droid_value")

    # 修复：强制 seed 为 uint32 的 Python int，避免 PRNG key dtype 错误
    seed = getattr(config, "seed", 0)
    if isinstance(seed, (np.ndarray, jax.Array)):
        seed = np.asarray(seed).astype(np.uint32).item()
    else:
        seed = int(seed)
    if dataclasses.is_dataclass(config):
        config = dataclasses.replace(config, seed=seed)
    else:
        setattr(config, "seed", seed)

    logger.info("Loading trained policy from checkpoint...")
    try:
        # Manually load norm stats if available in config assets
        norm_stats = None
        if config.data.assets.assets_dir and config.data.assets.asset_id:
             try:
                 norm_stats = _checkpoints.load_norm_stats(config.data.assets.assets_dir, config.data.assets.asset_id)
                 logger.info(f"Manually loaded norm stats from {config.data.assets.assets_dir}/{config.data.assets.asset_id}")
             except Exception as e:
                 logger.warning(f"Could not manually load norm stats: {e}")

        # Use CHECKPOINT_DIR global variable here since the argument was removed from main signature
        policy = _policy_config.create_trained_policy(config, args.checkpoint_dir, norm_stats=norm_stats, default_prompt="perform the task")
 
    except Exception as e:
        logger.error(f"Error creating policy: {e}")
        return
    
    # 2. Load the episode and extract the task prompt from a Parquet file.
    for episode_path in get_filtered_file_paths(PARQUET_FOLDER):
        logger.info("Starting Advantage Inference Process")
        
        if episode_path.endswith('.parquet') and os.path.isfile(episode_path):
        
            # Extract episode index from PARQUET_PATH
            # Path format: .../chunk-000/episode_000000.parquet
            episode_index = re.search(r'(\d+)\.parquet', episode_path).group(1) if episode_path else None
            episode_index = int(episode_index)
    
            logger.info(f"Extracted episode index: {episode_index} from path: {episode_path}")

            # Load task prompt from episodes.jsonl
            data_dir = "/root/autodl-tmp/droid_100"
            task_prompt = load_episode_metadata(data_dir, episode_index=episode_index)
            
            logger.info(f"Loaded task prompt: {task_prompt}")
            
            if task_prompt is None:
                raise ValueError(f"Task prompt is None for episode index {episode_index}. Check episodes.jsonl for this episode.")
            
            if task_prompt == "":
                task_prompt = "No task prompt found for this episode. Defaulting to empty string."


            logger.info(f"1. Loading episode from {episode_path}...")

            table = None
            episode = None

            if os.path.exists(episode_path) and HAS_PYARROW:
                try:
                    table = pq.read_table(episode_path)
                    episode = table.to_pydict()
                    logger.info("Loaded parquet episode successfully.")
                except Exception as e:
                    logger.error(f"Failed to read parquet: {e}")

            num_steps = len(episode["index"])
            logger.info(f"Episode length: {num_steps}")

            # 3. After loading the data, simply set the advantage at first timestep to a zero value.
            logger.info("2. Setting initial advantage to 0.0")
            advantages = [0.0]

            def get_step_data(idx):
                return {k: v[idx] for k, v in episode.items()}

            # 4. Infer the state_value for the first timestep.
            logger.info("3. Inferring Step 0...")
            step0_data = get_step_data(0)
            
            # Apply manual repack and decoding
            step0_data = manual_repack(step0_data)
            step0_data["prompt"] = task_prompt
            
            result0 = policy.infer_value(step0_data)
            
            # Extract value - handled flexibly for Value model
            if "state_value" in result0:
                val = result0["state_value"]
            elif "value" in result0:
                val = result0["value"]
            else:
                # If model output is unexpected (e.g. random weights not outputting correctly without inputs)
                val = 0.0
            
            if hasattr(val, "item"):
                prev_state_value = val.item()
            else:
                prev_state_value = float(val)

            logger.info(f"Step 0 State Value: {prev_state_value}")

            # 4. Loop and calculate advantages
            # Formula: A = r + gamma * V(s') - V(s)
            
            logger.info("4. Starting inference loop...")
            
            for t in tqdm.trange(1, num_steps):
                # Infer V(s_t) which is "next state value" relative to previous step
                step_data = get_step_data(t)
                step_data = manual_repack(step_data)
                step_data["prompt"] = task_prompt
                result = policy.infer_value(step_data)
                
                if "state_value" in result:
                    curr_val = result["state_value"]
                elif "value" in result:
                    curr_val = result["value"]
                else:
                    curr_val = 0.0

                if hasattr(curr_val, "item"):
                    curr_state_value = curr_val.item()
                else:
                    curr_state_value = float(curr_val)
                
                # Calculate Advantage
                # Formula: A = Reward + Discount * V_next - V_curr
                # Interpreting "next state value" as curr_state_value (at t) 
                # and "current state value" as prev_state_value (at t-1)
                
                # Get Reward (assuming reward at t-1 corresponds to transition to t)
                if "reward" in episode:
                    r = episode["reward"]
                    # If reward is a list/array with length equal to num_steps, assume r[t] is the reward for transition t-1 -> t
                    # or simply take r[t-1] if we align by "step index = time index"
                    if hasattr(r, "__getitem__"):
                        reward = r[t-1]  # Using r[t-1] for consistency with prev logic
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
                    
                if hasattr(reward, "item"):
                    reward = reward.item()
                
                advantage = reward + GAMMA * curr_state_value - prev_state_value
                advantages.append(advantage)
                
                # Update for next iteration
                prev_state_value = curr_state_value

            logger.info("Inference complete.")
            logger.info(f"Total advantages calculated: {len(advantages)}")
            
            # 6. Append value and Overwrite parquet
            if table is not None and HAS_PYARROW:
                logger.info("6. Appending 'advantage' column and overwriting file...")
                try:
                    # Check if column exists and remove it if so (to allow overwrite)
                    if "advantage" in table.column_names:
                        table = table.drop(["advantage"])
                    
                    # Create array
                    adv_array = pa.array(advantages)
                    
                    # Append column
                    new_table = table.append_column("advantage", adv_array)
                    
                    # Write back
                    pq.write_table(new_table, episode_path)
                    logger.info(f"Successfully overwrote {episode_path} with new advantage column.")
                    
                except Exception as e:
                    logger.error(f"Failed to save parquet file: {e}")
            else:
                logger.warning("Skipping file save (No original table or pyarrow missing).")

    
def get_filtered_file_paths(folder_path, pattern='.parquet'):
    """
    遍历文件夹（包括子文件夹），仅返回以特定 pattern 结尾的文件完整路径。
    
    :param folder_path: 目标文件夹路径
    :param pattern: 过滤模式（例如 '.txt', '_backup.py', '.jpg'）
    :return: 匹配的文件路径列表
    """
    matched_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件名是否以指定的模式结尾
            if file.endswith(pattern):
                full_path = os.path.join(root, file)
                matched_files.append(full_path)
                
    return matched_files


def decode_image(entry):
    # Decode image from bytes dictionary (LeRobot format)
    if isinstance(entry, dict) and "bytes" in entry:
        return np.array(Image.open(io.BytesIO(entry["bytes"])))
    return entry


def load_episode_metadata(data_dir, episode_index):
    """Load task prompt for a given episode from episodes.jsonl"""
    episodes_file = os.path.join(data_dir, "meta", "episodes.jsonl")
    with open(episodes_file, 'r') as f:
        for line in f:
            episode_data = json.loads(line)
            logger.info(f"Checking episode index: {episode_data.get('episode_index')} against target: {episode_index}") 
            if episode_data["episode_index"] == episode_index:
                # Return the first task (tasks is a list)
                logger.info(f"The loaded episode data  : {episode_data}. Extracting task prompt...")
                tasks = episode_data.get("tasks", [])
                return tasks[0] if tasks else ""
    return ""


def manual_repack(data):
    new_data = {}
    # Map raw keys to observation/ keys and decode images
    img_keys = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    for key in img_keys:
        if key in data:
            new_data[f"observation/{key}"] = decode_image(data[key])
            
    if "joint_position" in data:
        new_data["observation/joint_position"] = np.asarray(data["joint_position"])
    if "gripper_position" in data:
        new_data["observation/gripper_position"] = np.asarray(data["gripper_position"])
        
    if "actions" in data:
        new_data["actions"] = np.asarray(data["actions"])
        
    if "prompt" in data:
        new_data["prompt"] = data["prompt"]
        
    return new_data


if __name__ == "__main__":
    
    main()
    
    # test_episode_data()

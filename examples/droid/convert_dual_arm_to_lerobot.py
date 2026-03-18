"""
Convert PKL episodes to a LeRobot dataset, saving per-episode frames.

For each episode in the given folders (e.g., 20260102_success/, 20260107_failure/,
20260107_success/), this script loads episode_data.pkl, reads images referenced
in each frame's image_paths, and stores only the first 8 dims from observation.state:
- dims 0-6: joint positions (7D)
- dim 7: gripper (1D)

We control a single arm; remaining dims are ignored to keep the other arm stable.

1. 目标价值的数学定义 (Target Definition)论文指出，价值函数对应于距离成功完成的（负）步数。对于成功的回合 (Success)：在时间步 $t$，剩余步数是 $T - t$。未归一化的价值 $V_{raw}(s_t) = -(T - t)$。例如：总长 50 步，第 10 步时，价值是 -40。对于失败的回合 (Failure)：由于最后一步奖励是 $-C_{fail}$，根据累计回报公式，在时间步 $t$：$V_{raw}(s_t) = -(T - t) - C_{fail}$。由于 $C_{fail}$ 是一个很大的常数，这会让失败轨迹的所有时刻都具有极大的负值。
The T_MAX is mainly determined by the maximum episode length, which is extract from "dataset_distribution_analysis_executed.ipynb"
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from termcolor import colored

FAIL_PENALTY = -1000.0  
T_MAX = 540.0 


def normalize_state_value(arr: np.ndarray) -> np.ndarray:  
    """
    为了在不同长度的任务中保持一致性，并适应 Transformer (Gemma 3) 的数值敏感性，价值被归一化到 $(-1, 0)$。
    计算公式: 
        $$V_{norm} = \text{Clip} \left( \frac{V_{raw}}{T_{max}}, \min=-1, \max=0 \right)$$
        $T_{max}$ (Task-specific)：该任务在数据集中的最大步数（通常取 95th 百分位数）。
        $C_{fail}$：设定为一个大常数，确保 $\frac{-(T-t) - C_{fail}}{T_{max}} \le -1$。
    """

    return (arr ) / (-FAIL_PENALTY)  # Shift to [-1000, 0], then scale to [-1, 1]   


def resize_image_array(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    # size is (width, height) for PIL
    img = Image.fromarray(arr)
    return np.array(img.resize(size, resample=Image.BICUBIC))


def find_episode_dirs(base_dir: Path, folder_names: List[str]) -> List[Path]:
    episode_dirs: List[Path] = []
    for name in folder_names:
        root = base_dir / name
        if not root.exists():
            continue
        for ep in sorted(root.glob("episode_*/")):
            episode_dirs.append(ep)
    return episode_dirs


def load_episode_pkl(episode_dir: Path) -> Dict[str, Any]:
    pkl_path = episode_dir / "episode_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing PKL file: {pkl_path}")
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if "frames" not in data:
        raise ValueError(f"Unexpected PKL structure in {pkl_path}; 'frames' key not found")
    return data


def read_image(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    return np.array(img)


def get_image_paths(frame: Dict[str, Any]) -> Dict[str, str]:
    return frame.get("image_paths", {})


def extract_state_first8(frame: Dict[str, Any]) -> Tuple[List[float], float]:
    obs = frame.get("observation", {})
    state = obs.get("state")
    if state is None:
        raise ValueError("Frame missing observation.state")
    if len(state) < 8:
        raise ValueError("observation.state must have at least 8 dimensions")
    joint_pos = [float(x) for x in state[:7]]
    gripper = float(state[7])
    return joint_pos, gripper


def build_features_fixed() -> Dict[str, Dict[str, Any]]:
    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is "image"
    return {
        # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (180, 320, 3),  # This is the resolution used in the DROID RLDS dataset
            "names": ["height", "width", "channel"],
        },
        "exterior_image_2_left": {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (180, 320, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),  # We will use joint velocity actions here (7D) + gripper position (1D)
            "names": ["actions"],
        },
        "state_value": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["state_value"],
        },
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
    }


def get_task_string(episode_dir: Path) -> str:
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return ""
    try:
        with meta_path.open("r") as f:
            md = json.load(f)
        task = md.get("task")
        if isinstance(task, str) and task:
            return task
        return ""
    except Exception:
        return ""


def convert_to_lerobot(repo_id: str, fps: int, episode_dirs: List[Path]) -> None:
    features = build_features_fixed()

    # Clean up any existing dataset in the output directory to rebuild
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Prepare a blank image for missing views
    blank_img = np.zeros((180, 320, 3), dtype=np.uint8)

    for ep_dir in tqdm(episode_dirs, desc="Writing episodes"):
        pkl = load_episode_pkl(ep_dir)
        frames = pkl["frames"]
        task_str = get_task_string(ep_dir)
        episode_length = len(frames)
        for step_idx, fr in enumerate(frames):
            img_paths = get_image_paths(fr)

            # Load and resize images, map to requested keys
            ext1_path = img_paths.get("exterior_1")
            wrist_left_path = img_paths.get("wrist_left")
            wrist_right_path = img_paths.get("wrist_right")

            ext1_img = resize_image_array(read_image(ep_dir / ext1_path), (320, 180)) if ext1_path else blank_img
            wrist_left_img = (
                resize_image_array(read_image(ep_dir / wrist_left_path), (320, 180))
                if wrist_left_path
                else (resize_image_array(read_image(ep_dir / wrist_right_path), (320, 180)) if wrist_right_path else blank_img)
            )

            frame_payload: Dict[str, Any] = {
                "exterior_image_1_left": ext1_img,
                # No second exterior camera in PKL; duplicate ext1 to fill the feature
                "exterior_image_2_left": ext1_img,
                "wrist_image_left": wrist_left_img,
            }

            joint_pos, gripper = extract_state_first8(fr)
            frame_payload["joint_position"] = np.asarray(joint_pos, dtype=np.float32)
            frame_payload["gripper_position"] = np.asarray([gripper], dtype=np.float32)

            # Actions: first 7 dims are joint velocities (from qvel), 8th is gripper position
            qvel = fr.get("observation", {}).get("qvel")
            if qvel is not None and len(qvel) >= 7:
                act = np.asarray(list(qvel[:7]) + [gripper], dtype=np.float32)
            else:
                act = np.asarray([0.0] * 7 + [gripper], dtype=np.float32)
            frame_payload["actions"] = act

            # Optionally include task string (not part of features but tolerated by LeRobotDataset)
            if isinstance(task_str, str) and task_str:
                frame_payload["task"] = task_str

            # Value target and reward
            # State value: remaining timesteps scheme
            # V(t) = -1 * (episode_length - t - 1)
            is_failure_episode = any("success" in part for part in ep_dir.parts) 
            if is_failure_episode:
                state_value = normalize_state_value(-1.0 * (episode_length - step_idx - 1) + FAIL_PENALTY)
            else:
                state_value = normalize_state_value(-1.0 * (episode_length - step_idx - 1))
            
            # # Colorful print of state_value
            # try:
            #     sv_str = colored(f"{state_value:.3f}", "cyan", attrs=["bold"])
            # except ImportError:
            #     sv_str = f"{state_value:.3f}"
            # print(f"  Step {step_idx+1}/{episode_length} | state_value: {sv_str}")

            # Reward:
            # - success episodes: reward = -1 for all steps, last step = 0
            # - failure episodes: reward = -1 for all steps, last step = FAIL_PENALTY
            is_success_episode = any("success" in part for part in ep_dir.parts)
            reward = -1.0
            if step_idx == episode_length - 1:
                if is_success_episode:
                    reward = 0.0
                elif is_failure_episode:
                    reward = FAIL_PENALTY
            frame_payload["state_value"] = np.asarray([state_value], dtype=np.float32)
            frame_payload["reward"] = np.asarray([reward], dtype=np.float32)

            dataset.add_frame(frame_payload)
        dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PKL episodes to LeRobotDataset (first 8 state dims)")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=str(Path.cwd()),
        help="Base directory containing episode folders (e.g., /root/autodl-tmp/data_lerobot)",
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="*",
        default=["20260102_success", "20260107_failure", "20260107_success"],
        help="Folder names to scan for episodes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second to store in the LeRobot dataset (default 15 like DROID)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="local_pkl_dataset",
        help="Output LeRobot dataset repo id (saved under $LEROBOT_HOME)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    episode_dirs = find_episode_dirs(base_dir, args.folders)
    if not episode_dirs:
        print("No episode directories found. Check --base_dir and --folders.")
        return

    convert_to_lerobot(args.repo_id, args.fps, episode_dirs)
    print("Done. LeRobot episodes saved.")


if __name__ == "__main__":
    main()

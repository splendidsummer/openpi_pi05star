
"""
Minimal example script for converting the DROID dataset (RLDS format) to LeRobot format.

Usage:
uv run convert_droid_data_to_lerobot.py --data_dir /path/to/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run convert_droid_data_to_lerobot.py --data_dir /path/to/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.

This script loads the DROID dataset using TensorFlow Datasets from the specified data_dir.
"""

from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "SummerZhang/droid_100"  # Name of the output dataset, also used for the Hugging Face Hub


def main(data_dir: str = "/root/autodl-tmp", *, push_to_hub: bool = False):
    
    data_dir = Path(data_dir)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
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
                "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
            # extend the simple feature into code block: the "adv_indicator" should be inferred from value function
            "adv_indicator": {
                "dtype": "bool", 
                "shape": (1, ), 
                "names": ["adv_indicator"]
            }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load DROID dataset using TensorFlow Datasets
    ds = tfds.load("droid_100", data_dir=str(data_dir), split="train")

    # Loop over episodes in the dataset
    for episode in tqdm(ds, desc="Converting episodes"):
        # Get language instruction from the first step
        language_instruction = "Do something"
        for step in episode["steps"].take(1):
            if "language_instruction" in step:
                language_instruction = step["language_instruction"].numpy().decode("utf-8")
        print(f"Converting episode with language instruction: {language_instruction}")

        # Write to LeRobot dataset
        for step in episode["steps"]:
            dataset.add_frame(
                {
                    "exterior_image_1_left": step["observation"]["exterior_image_1_left"].numpy(),
                    "exterior_image_2_left": step["observation"]["exterior_image_2_left"].numpy(),
                    "wrist_image_left": step["observation"]["wrist_image_left"].numpy(),
                    "joint_position": step["observation"]["joint_position"].numpy().astype(np.float32),
                    "gripper_position": step["observation"]["gripper_position"].numpy().astype(np.float32),
                    # Use joint velocity actions + gripper velocity
                    "actions": np.concatenate(
                        [
                            step["action_dict"]["joint_velocity"].numpy(),
                            step["action_dict"]["gripper_velocity"].numpy(),
                        ],
                        axis=0,
                    ).astype(np.float32),
                    "task": language_instruction, 
                    "adv_indicator": np.array([1], dtype=bool), 
                }
            )
        dataset.save_episode()

    # # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)

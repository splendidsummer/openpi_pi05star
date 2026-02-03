import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image_1 = _parse_image(data["observation/exterior_image_1_left"])
        base_image_2 = _parse_image(data["observation/exterior_image_2_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.VALUE:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI05_STAR:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image_1, base_image_1,  wrist_image)
                image_masks = (np.True_, np.True_,  np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]


        # For VALUE model: handle state_value as value_targets
        if self.model_type == _model.ModelType.VALUE and "state_value" in data:
            state_value = np.asarray(data["state_value"])
            if state_value.ndim == 0:
                state_value = state_value[np.newaxis]
            elif state_value.ndim >= 2:
                if state_value.shape[-1] == 1:
                    state_value = state_value.squeeze(-1)
                else:
                    state_value = state_value.flatten()
            inputs["value_targets"] = state_value

        # For VALUE model: handle reward
        if self.model_type == _model.ModelType.VALUE and "reward" in data:
            reward = np.asarray(data["reward"])
            if reward.ndim == 0:
                reward = reward[np.newaxis]
            elif reward.ndim >= 2:
                reward = reward.flatten()
            inputs["reward"] = reward
            
        # For PI05_STAR model: handle adv_indicator as a bool
        if self.model_type == _model.ModelType.PI05_STAR and "adv_indicator" in data:
            adv_indicator = data["adv_indicator"]
            
            # --- 新增兼容性逻辑 ---
            # 如果是 np.array([1], dtype=bool) 或对应的 Tensor
            # 使用 .item() 获取其中的标量，并用 bool() 强制转换为 Python 原生布尔类型
            if hasattr(adv_indicator, "item"):
                adv_indicator = bool(adv_indicator.item())
            elif isinstance(adv_indicator, (list, np.ndarray)) and len(adv_indicator) > 0:
                adv_indicator = bool(adv_indicator[0])

            # 此时 isinstance(adv_indicator, bool) 检查将会通过
            if not isinstance(adv_indicator, bool):
                raise ValueError(f"adv_indicator must be a boolean value, but got {type(adv_indicator)}")        
     
        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.

        return {"actions": np.asarray(data["actions"][:, :8])}

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

class SmolVLAInterface:
    def __init__(self, model_id="bendca61/smolvla-mujoco-so101-cube_on_tray", task="Put the block on the tray"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.task = task

        config = PreTrainedConfig.from_pretrained(model_id)
        config.compile_model = False

        self.policy = SmolVLAPolicy.from_pretrained(model_id, config=config).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def predict(self, realsense_img, wrist_img, state, task=None):
        def to_tensor_img(frame):
            t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0).to(self.device)

        obs = {
            "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device),
            "observation.images.realsense": to_tensor_img(realsense_img),
            "observation.images.wrist_cam": to_tensor_img(wrist_img),
            "task": task or self.task,
        }
        obs = self.preprocessor(obs)
        with torch.no_grad():
            action = self.policy.select_action(obs)
        action = self.postprocessor(action)
        action = action.squeeze(0).cpu().numpy()
        return np.deg2rad(action)

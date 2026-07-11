import numpy as np
import torch
import mujoco
from smolvla_interface import SmolVLAInterface

MODEL_ID = "bendca61/smolvla-mujoco-so101-cube_on_tray"
vla = SmolVLAInterface(model_id=MODEL_ID)

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/task_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=480, width=640)

def get_imgs():
    imgs = {}
    for cam in ["realsense", "wrist_cam"]:
        renderer.update_scene(data, camera=cam)
        imgs[cam] = renderer.render()
    return imgs

# --- the optimized data-path version, same as optimize_datapath.py ---
def to_tensor_img_fast(frame, device):
    t = torch.from_numpy(frame).to(device, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    return t

def predict_fast(realsense_img, wrist_img, state):
    obs = {
        "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(vla.device, non_blocking=True),
        "observation.images.realsense": to_tensor_img_fast(realsense_img, vla.device),
        "observation.images.wrist_cam": to_tensor_img_fast(wrist_img, vla.device),
        "task": vla.task,
    }
    obs = vla.preprocessor(obs)
    with torch.no_grad():
        action = vla.policy.select_action(obs)
    action = vla.postprocessor(action)
    return np.deg2rad(action.squeeze(0).cpu().numpy())

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# --- compare across several distinct observations, not just one ---
n_checks = 8
cos_sims = []
l2_dists = []

for i in range(n_checks):
    # step sim forward a bit each time so observations actually differ
    for _ in range(5):
        mujoco.mj_step(model, data)

    imgs = get_imgs()
    state = data.qpos[:6]

    action_baseline = vla.predict(imgs["realsense"], imgs["wrist_cam"], state)
    action_optimized = predict_fast(imgs["realsense"], imgs["wrist_cam"], state)

    cos = cosine_sim(action_baseline, action_optimized)
    l2 = float(np.linalg.norm(action_baseline - action_optimized))
    cos_sims.append(cos)
    l2_dists.append(l2)
    print(f"check {i}: cosine_sim={cos:.6f}, l2_dist={l2:.6f}")

print(f"\nMean cosine similarity: {np.mean(cos_sims):.6f}")
print(f"Mean L2 distance: {np.mean(l2_dists):.6f}")
print(f"Max L2 distance: {np.max(l2_dists):.6f}")

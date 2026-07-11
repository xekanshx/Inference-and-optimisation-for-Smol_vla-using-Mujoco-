import time, json
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

def to_tensor_img_batch(frame, batch_size, device):
    t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255.0)  # C,H,W
    return t.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)  # B,C,H,W

def predict_batched(realsense_img, wrist_img, state, batch_size):
    obs = {
        "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(vla.device),
        "observation.images.realsense": to_tensor_img_batch(realsense_img, batch_size, vla.device),
        "observation.images.wrist_cam": to_tensor_img_batch(wrist_img, batch_size, vla.device),
        "task": [vla.task] * batch_size,
    }
    obs = vla.preprocessor(obs)
    with torch.no_grad():
        action = vla.policy.predict_action_chunk(obs)  # batched call, not the queue-based select_action
    return action

results = {}
imgs = get_imgs()

for batch_size in [1, 2, 4, 8, 16]:
    # warmup
    for _ in range(2):
        predict_batched(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6], batch_size)

    n_calls = 10
    t0 = time.perf_counter()
    for _ in range(n_calls):
        predict_batched(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6], batch_size)
    elapsed = time.perf_counter() - t0

    per_call_ms = (elapsed / n_calls) * 1000
    total_obs_processed = n_calls * batch_size
    throughput = total_obs_processed / elapsed

    results[f"batch_{batch_size}"] = {
        "per_call_latency_ms": round(per_call_ms, 2),
        "throughput_obs_per_sec": round(throughput, 2),
    }
    print(f"batch={batch_size}: latency={per_call_ms:.2f}ms, throughput={throughput:.2f} obs/sec")

print(json.dumps(results, indent=2))
with open("batching_results.json", "w") as f:
    json.dump(results, f, indent=2)

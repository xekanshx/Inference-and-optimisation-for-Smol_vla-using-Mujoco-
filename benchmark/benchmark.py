import copy
import time
import json
import numpy as np
import torch
import mujoco
from torch.profiler import profile, ProfilerActivity
from smolvla_interface import SmolVLAInterface

MODEL_ID = "bendca61/smolvla-mujoco-so101-cube_on_tray"
N_PROFILE = 10
N_BENCH = 30

results = {}

# --- load baseline (fp32) interface ---
vla = SmolVLAInterface(model_id=MODEL_ID)

# --- sim setup for realistic observations ---
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

def time_calls(predict_fn, n, warmup=3):
    for _ in range(warmup):
        imgs = get_imgs()
        predict_fn(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6])
    lat = []
    for _ in range(n):
        imgs = get_imgs()
        t0 = time.perf_counter()
        action = predict_fn(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6])
        lat.append(time.perf_counter() - t0)
    return np.array(lat), action

# =========================================================
# STEP 1: Profile the FP32 baseline
# =========================================================
print("Profiling FP32 baseline...")
imgs = get_imgs()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(N_PROFILE):
        vla.predict(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6])
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
prof.export_chrome_trace("baseline_trace.json")

# =========================================================
# STEP 2: Baseline (FP32) latency
# =========================================================
print("\nBenchmarking FP32 baseline...")
lat_fp32, action_fp32 = time_calls(vla.predict, N_BENCH)
results["fp32_p50_ms"] = round(np.percentile(lat_fp32, 50) * 1000, 3)
results["fp32_p95_ms"] = round(np.percentile(lat_fp32, 95) * 1000, 3)

# =========================================================
# STEP 3: Optimization A — FP16 casting
# =========================================================
print("\nCasting policy to FP16...")
vla.policy = vla.policy.half()

def predict_fp16(realsense_img, wrist_img, state):
    def to_tensor_img(frame):
        t = torch.from_numpy(frame.copy()).permute(2, 0, 1).half() / 255.0
        return t.unsqueeze(0).to(vla.device)
    obs = {
        "observation.state": torch.tensor(state, dtype=torch.float16).unsqueeze(0).to(vla.device),
        "observation.images.realsense": to_tensor_img(realsense_img),
        "observation.images.wrist_cam": to_tensor_img(wrist_img),
        "task": vla.task,
    }
    obs = vla.preprocessor(obs)
    with torch.no_grad():
        action = vla.policy.select_action(obs)
    action = vla.postprocessor(action)
    return np.deg2rad(action.squeeze(0).float().cpu().numpy())

print("Benchmarking FP16...")
lat_fp16, action_fp16 = time_calls(predict_fp16, N_BENCH)
results["fp16_p50_ms"] = round(np.percentile(lat_fp16, 50) * 1000, 3)
results["fp16_p95_ms"] = round(np.percentile(lat_fp16, 95) * 1000, 3)
results["fp16_vs_fp32_speedup"] = round(results["fp32_p50_ms"] / results["fp16_p50_ms"], 3)
results["fp16_action_diff_l2"] = round(float(np.linalg.norm(action_fp32 - action_fp16)), 6)

# cast back to fp32 for the next test
vla.policy = vla.policy.float()

# =========================================================
# STEP 4: Optimization B — torch.compile (default mode, NOT max-autotune)
# =========================================================
print("\nCompiling policy.select_action (default mode)...")
compiled_select_action = torch.compile(vla.policy.select_action, mode="default")

def predict_compiled(realsense_img, wrist_img, state):
    def to_tensor_img(frame):
        t = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(vla.device)
    obs = {
        "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(vla.device),
        "observation.images.realsense": to_tensor_img(realsense_img),
        "observation.images.wrist_cam": to_tensor_img(wrist_img),
        "task": vla.task,
    }
    obs = vla.preprocessor(obs)
    with torch.no_grad():
        action = compiled_select_action(obs)
    action = vla.postprocessor(action)
    return np.deg2rad(action.squeeze(0).cpu().numpy())

# first call = compilation cost
imgs = get_imgs()
t0 = time.perf_counter()
first_action = predict_compiled(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6])
results["compile_first_call_s"] = round(time.perf_counter() - t0, 3)

print("Benchmarking compiled steady-state...")
lat_compiled, action_compiled = time_calls(predict_compiled, N_BENCH, warmup=2)
results["compiled_p50_ms"] = round(np.percentile(lat_compiled, 50) * 1000, 3)
results["compiled_p95_ms"] = round(np.percentile(lat_compiled, 95) * 1000, 3)
results["compiled_vs_fp32_speedup"] = round(results["fp32_p50_ms"] / results["compiled_p50_ms"], 3)

print("\n=== FINAL RESULTS ===")
print(json.dumps(results, indent=2))
with open("optimization_results.json", "w") as f:
    json.dump(results, f, indent=2)

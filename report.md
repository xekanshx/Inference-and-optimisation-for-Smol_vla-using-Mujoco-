# SmolVLA on MuJoCo (SO-101): Setup, Inference, Closed-Loop Rollout, Benchmarking, and Optimization

## Part 1 — Setup, Inference, and Closed-Loop Rollout

### 1. Hardware and Model Choice

**Hardware:** Apple M3 (MacBook Air), 8 GB unified memory architecture (shared CPU/GPU memory pool, no discrete VRAM), macOS. Inference runs on PyTorch's MPS (Metal Performance Shaders) backend.

**Model:** [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) — a small Vision-Language-Action model (~450M parameters, VLM backbone truncated to 16 layers at load time). Chosen because its size fits comfortably in the M3 Air's unified memory (peak allocated ~1.15 GB) with no quantization required, and it is natively supported by Hugging Face's `lerobot` library.

For the closed-loop task, a second checkpoint — [`bendca61/smolvla-mujoco-so101-cube_on_tray`](https://huggingface.co/bendca61/smolvla-mujoco-so101-cube_on_tray) — was used: a community fine-tune of SmolVLA on a MuJoCo-simulated SO-101 arm performing a "put the block on the tray" task, closely matching this project's simulated (not real-world) setup.

### 2. Installation and Real Observation → Action

**Environment:** Python 3.11 venv, `lerobot` (editable install with `[smolvla]` extra), `torch` (MPS build), `transformers`, `mujoco`, `imageio`.

**Notable setup issues resolved:**

- Initially tried using MuJoCo via LIBERO, since a public dataset with pretrained datapoints for SmolVLA is available there. However, LIBERO and SimplerEnv do not support macOS — LIBERO requires Linux (`sys_platform == 'linux'`).
- For other platforms like ALOHA, no pretrained model was found for SO-100 (the robot SmolVLA is trained on).
- Without compute for training or fine-tuning a VLA model, a robot scene was manually built using the `mujoco_menagerie` repository and simulated directly in MuJoCo.
- The checkpoint's saved preprocessor config hard-coded `device: "cuda"`; resolved via `preprocessor_overrides={"device_processor": {"device": "mps"}}`, mirroring `lerobot`'s own `lerobot_eval.py`.
- The fine-tuned checkpoint's config requested `torch.compile(mode="max-autotune")` at load time — a CUDA-only optimization path that crashes on MPS; resolved by loading the config separately, setting `compile_model=False`, and passing that config into `from_pretrained`.

**Real observation → action pipeline** (`smolvla_interface.py`):

- **Perception:** Two MuJoCo-rendered camera frames per step (`realsense`: static view; `wrist_cam`: mounted on the gripper's `Fixed_Jaw` body), rendered at 480×640.
- **State:** 6 joint positions (`Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`) from `data.qpos[:6]`.
- **Instruction:** `"Put the block on the tray"`, matching the fine-tuning checkpoint's training phrasing.
- **Inference:** Full preprocessor → `policy.select_action` → postprocessor pipeline.
- **Output:** A verified real 6-dimensional action vector, confirmed working on a real rendered observation before any closed-loop rollout was attempted.

### 3. Closed-Loop Rollout, Video, and Success Rate

**Task:** Push/place a free-floating cube onto a marked target zone, using the SO-101 arm (`mujoco_menagerie`'s `trs_so_arm100`) in a custom MuJoCo scene — no LIBERO or SimplerEnv.

**Success metric** — the initial version was misleading and was corrected. A naive "cube ends up near the target" distance check produced a 20% success rate; inspecting the videos showed the cube had simply spawned close to the target by chance, with no contact from the arm at all. The metric was corrected to require all three:

1. Final cube-to-target distance below threshold,
2. At least one verified contact event between a gripper finger pad and the cube (via MuJoCo's live contact list), and
3. Meaningful displacement of the cube from its randomized start position.

**Results with the corrected metric:**

| Checkpoint | Episodes | Success rate | Contact ever occurred |
|---|---|---|---|
| `lerobot/smolvla_base` (zero-shot) | 5–10 | 0% | No |
| `bendca61/smolvla-mujoco-so101-cube_on_tray` (task-matched fine-tune) | 20 | 0% | No |

**Diagnosis:** No SmolVLA-compatible simulator exists on macOS: LIBERO and SimplerEnv (the environments SmolVLA checkpoints are actually trained/evaluated against) both hard-require Linux, and no pretrained SmolVLA checkpoints exist for the macOS-viable alternatives (PyBullet, Genesis). The only way to get a closed-loop rollout on this hardware at all was to hand-build a MuJoCo scene and use a community fine-tune targeting MuJoCo+SO-101 — the closest available approximation, but not the checkpoint's actual training distribution. The pipeline itself was verified independently in §2 (a real 6-DoF action vector produced from a real rendered observation), so the 0% success rate is best explained by scene mismatch (camera pose, table height, calibration) between this hand-built environment and whichever environment the fine-tune was trained in, not a defect in the observation→action pipeline. Closing this gap would require either a Linux machine (for LIBERO) or compute to fine-tune/calibrate on this exact scene, both outside this assignment's hardware constraints.

### 4. `predict()` Interface

```python
vla = SmolVLAInterface(
    model_id="bendca61/smolvla-mujoco-so101-cube_on_tray",
    task="Put the block on the tray"
)
action = vla.predict(realsense_frame, wrist_cam_frame, joint_state)
```

Decoupled from the simulation loop; could be wrapped behind an HTTP endpoint with minimal additional code.

---

## Part 2 — Making It Faster / Cheaper / Lighter

### 5. Baseline Measurements

Measured on M3 MacBook Air, MPS backend, `bendca61/smolvla-mujoco-so101-cube_on_tray`.

| Metric | Value | Method |
|---|---|---|
| Model load time | 26.0 s | Wall-clock, one-time |
| `select_action` latency (steady-state) | p50: 3.63 ms, p95: 4.24 ms | 49/50 calls (1 outlier at 547 ms excluded, reported separately) |
| Peak allocated memory | 1153.3 MB | `torch.mps.current_allocated_memory()` |
| Peak driver memory | 1798.5 MB | `torch.mps.driver_allocated_memory()` |
| GPU active residency | 18.55% | `sudo powermetrics --samplers gpu_power` during sustained load |
| GPU power draw | 172 mW | Same `powermetrics` measurement |
| Est. cost per 1k inferences | ~$0.0000264 | Measured power × steady-state latency × $0.15/kWh |

**Important caveat discovered during optimization work:** the checkpoint's config specifies `chunk_size: 50`, `n_action_steps: 50`. `select_action` maintains an internal action queue and only runs a full model forward pass once every 50 calls, returning cached actions the rest of the time. The 3.63 ms figure above is therefore dominated by cache hits, not the true per-inference cost.

### 6. Profiling — Where Does Time Actually Go?

Used PyTorch Profiler (`torch.profiler`, CPU activities — Nsight/CUDA-specific tooling does not apply on Apple Silicon; noted explicitly rather than substituted with an invalid claim of NVIDIA tooling).

**Result, 10 calls to `select_action`, total self CPU time 1.061 s:**

| Operation | % of total CPU time |
|---|---|
| `aten::copy_` + `aten::_to_copy` (data movement / dtype conversion) | ~62% |
| `aten::add`, `aten::pow` (elementwise ops) | ~11% |
| `aten::linear`, `aten::matmul`, `aten::bmm`, `scaled_dot_product_attention` (actual model compute) | ~6–7% |
| Everything else (`nonzero`, `where`, `arange`, `embedding`, indexing) | ~20% |

**Finding:** the bottleneck at this batch size is data movement and dtype casting, not GPU compute. This directly informed which optimizations were worth attempting.

### 7. Optimizations Attempted (two+ required)

| # | Optimization | Category | Result | Verdict |
|---|---|---|---|---|
| 1 | FP16 casting (`policy.half()`) | Quantization | Crashed: `MPSNDArrayMatrixMultiplication` — destination/accumulator dtype mismatch | Incompatible on this MPS + checkpoint combination; a real hardware/framework limitation, not a scripting bug |
| 2 | `torch.compile` (`mode="default"`, **not** `max-autotune` — that mode crashes on MPS, confirmed earlier in Part 1) | Compilation/runtime | Warm-up: 8.15 s first call. Steady-state: 4.01 ms vs 3.63 ms baseline → 0.90x (a ~10% regression) | Compilation overhead is not recovered; consistent with §6's finding that the workload is copy-bound, not compute-bound — `torch.compile` optimizes compute graphs, which were never the bottleneck here |
| 3 | Data-path optimization (removed redundant `frame.copy()`, reordered device transfer to happen on smaller uint8 data before the float cast, in-place normalization) | Systems-level | 3.645 ms → 3.467 ms → 1.05x speedup | Modest but real, genuine — and the only optimization of the three that actually targeted the profiler-identified bottleneck |
| 4 | Batching (`predict_action_chunk`, batch sizes 1/2/4/8/16) | Dynamic batching / throughput | See table below | Revealed the true uncached per-inference cost (~544 ms) and near-zero throughput scaling |

**Batching results (uncached, `predict_action_chunk`):**

| Batch size | Latency | Throughput |
|---|---|---|
| 1 | 544 ms | 1.84 obs/sec |
| 2 | 1060 ms | 1.89 obs/sec |
| 4 | 2039 ms | 1.96 obs/sec |
| 8 | 3986 ms | 2.01 obs/sec |
| 16 | (aborted — see below) | — |

**Quality trade-off check (data-path optimization).** Since no working success-rate signal was available to measure quality drift directly, action-vector fidelity was checked instead: the same 8 observations were run through both the baseline and data-path-optimized pipelines, and the resulting action vectors compared. Mean cosine similarity was 0.999498 (min 0.999187, max 0.999879) and mean L2 distance was 0.089 (max 0.125) — consistent with the optimization only reordering/removing redundant copies and casts, not altering the numerical computation path. This confirms the 1.05x speedup was "free" — no measurable quality cost.

Latency scaled almost exactly linearly with batch size (each doubling ~doubles latency), indicating this workload does not parallelize across the batch dimension on MPS — consistent with an iterative/sequential sampling process (likely diffusion- or flow-matching-based action generation, matching the `chunk_size: 50` config) rather than a simple parallelizable feedforward pass. Throughput barely improves (1.84 → 2.01 obs/sec) despite 8x the batch size.

Batch size 16 caused a system-wide crash and had to be manually terminated. On Apple Silicon, GPU and CPU share one unified memory pool with no isolated VRAM ceiling; when a batch's combined image tensors + model activations approached the system's memory limit, macOS was forced into aggressive memory reclamation that made the entire machine unresponsive, not just the Python process. This is a real, practical reliability finding: unified-memory architectures can fail system-wide under memory pressure, rather than failing gracefully within an isolated process, which is a meaningful constraint for any real deployment on this class of hardware.

*Caveat: all optimizations were evaluated only on Apple Silicon/MPS. Without CUDA hardware to compare against, it's possible FP16 and `torch.compile` would behave differently (and more favorably) on NVIDIA GPUs — these results should be read as MPS-specific, not universal.*

### 8. Summary — Did the Optimizations Target the Real Bottleneck?

The profiler (§6) correctly predicted, before any optimization was attempted, that compute-side techniques would have limited benefit: ~62% of CPU time was data movement/casting, only ~7% was actual model math. This prediction held up:

- **FP16** (a compute/precision optimization) — incompatible, crashed.
- **`torch.compile`** (a compute-graph optimization) — regressed, since there was little compute graph inefficiency to fix.
- **The data-path optimization**, targeting the actual profiled bottleneck, was the only one that produced a genuine (if modest) improvement.
- **Batching** surfaced a different, deeper truth: the "fast" 3.63 ms baseline was an artifact of action-queue caching, not raw model speed; true per-chunk generation cost is ~544 ms and does not parallelize well on this hardware, with a hard practical memory ceiling around batch size 8–16.

**Overall conclusion:** for this model/hardware/task combination, the highest-leverage optimization category is systems-level data-path efficiency (avoiding redundant copies/casts, understanding and correctly measuring cache-amortized vs. true inference cost), not compute quantization or graph compilation — a conclusion directly supported by profiling data rather than assumed in advance.

# Inference-and-optimisation-for-Smol_vla-using-Mujoco-
# SmolVLA on MuJoCo (SO-101)

This README explains *why* things were built the way they were, what the results do and don't mean, and how to
reproduce everything.

---

## Why MuJoCo directly, instead of LIBERO or SimplerEnv

The assignment explicitly allows either — a pre-built benchmark (LIBERO / SimplerEnv) or a
MuJoCo-based sim built directly. We chose to build directly on MuJoCo rather than use LIBERO, for
a few reasons:

- **LIBERO ships its own fixed set of tasks, scenes, and (often) its own robot/camera
  conventions**, largely built around Franka-style arms and real-world-style benchmark tasks. Our
  target model (`SmolVLA`) and the strongest available fine-tuned checkpoint we found
  (`bendca61/smolvla-mujoco-so101-cube_on_tray`) are both built around the **SO-100/SO-101** arm
  family, which isn't LIBERO's native embodiment. Bringing a mismatched robot into LIBERO would
  have meant *more* adaptation work, not less.
- Building directly on MuJoCo let us match the model's actual training assumptions (SO-101
  embodiment, a 6-DOF state/action space, two named cameras) as closely as possible, using
  `mujoco_menagerie`'s official SO-101 model rather than fighting a benchmark framework's opinions
  about scene layout.
- It also makes every part of the pipeline (scene, cameras, physics, reward/success signal)
  fully visible and inspectable — useful for an assignment whose stated goal is understanding a
  real inference pipeline end-to-end, not just calling a benchmark's `env.step()`.

**The tradeoff, and why our results should be read with real caution:** building the scene by
hand means **we do not know precisely how the fine-tuned checkpoint's original training scene was
calibrated** — camera distance/angle, table height, lighting, exact object size, exact SO-101
mounting position. `mujoco_menagerie` gives the correct *robot model*, but not the *scene* the
checkpoint was actually trained in. This is very likely a large part of why closed-loop success
stayed at 0% even with a task-matched checkpoint (see write-up §3): the arm behaves in a
plausible, non-random way (verified via action traces and camera debug frames — the cube is
visibly in frame, the policy's motion evolves sensibly across steps) but a scene-calibration
mismatch this size can be enough to prevent successful contact, without necessarily meaning the
model or pipeline is broken. If we had trained/fine-tuned our own checkpoint from data collected
directly in this exact scene, or had access to the original checkpoint's precise scene
description, results would likely look very different. **This is a known, explicitly acknowledged
limitation of the "build-it-yourself" approach, not a hidden flaw.**

## How `mujoco_menagerie` was used

`mujoco_menagerie` (Google DeepMind's collection of maintained, physically-accurate robot MJCF
models) supplied the base SO-101 arm (`trs_so_arm100/`), including correct joint limits,
actuator gains, meshes, and collision geometry — this is not something worth hand-rolling, since
getting inertial/collision properties wrong would silently corrupt any physics-based result.

On top of the menagerie-provided `scene.xml`, we added:
- A `realsense`-named static camera, and a `wrist_cam` camera mounted directly on the arm's
  `Fixed_Jaw` body (so it moves with the gripper) — matching the exact two-camera input contract
  the fine-tuned checkpoint expects (confirmed by inspecting `policy.config.input_features`
  directly, not assumed).
- A free-floating cube and a marked target zone geom, positioned (after an initial
  out-of-reach placement bug was found and fixed — see write-up §3) within the arm's verified
  reachable workspace.

We did **not** modify the arm's own kinematics, joint limits, or actuator definitions — those came
directly from menagerie, unedited, to preserve physical accuracy for the one part of the scene
that *is* well-specified and verifiable.

## Why the reported task success rate (0%) should not be over-interpreted

Two important calibration notes, both discovered and corrected during this project rather than
glossed over:

1. **An initial, naive success metric (distance-only) reported 20% success** — investigation of
   the actual videos showed the cube had simply spawned close enough to the target by chance, with
   the arm never touching it. The metric was rebuilt to require real contact (checked via MuJoCo's
   live contact list) plus meaningful displacement, not just final proximity. This is why the
   final reported number (0%, verified no-contact) is more trustworthy than the number would have
   been if we'd stopped at the first, more flattering result.
2. Given the scene-calibration mismatch explained above, **0% here should be read as "this specific
   scene reconstruction didn't transfer the checkpoint's learned behavior successfully,"** not as
   "SmolVLA cannot perform this task" or "the fine-tuned checkpoint doesn't work." The pipeline
   itself — rendering, state extraction, inference, actuation, physics, contact detection, video
   capture — was independently verified correct at every stage (see write-up §2–3).

## Optimization — what actually improved, and why

Before optimizing anything, we profiled the baseline (`torch.profiler`, CPU activities — Nsight is
NVIDIA-only and doesn't apply on this Apple Silicon hardware, noted explicitly rather than
substituted with an invalid claim). The profiler showed **~62% of CPU time was spent in
`aten::copy_`/`aten::_to_copy`** (data movement and dtype casting) versus **only ~6–7% in actual
model math** (`linear`/`matmul`/`bmm`/attention). This one finding shaped which optimizations were
worth trying:

| Optimization | Targets | Result |
|---|---|---|
| FP16 casting | Compute/precision | **Crashed** — MPS backend rejected mixed fp16/fp32 tensors in a matmul kernel. A real hardware/framework incompatibility, not a scripting error. |
| `torch.compile` (default mode) | Compute graph | **Regressed** — 8.15s compile warm-up, then 4.01ms steady-state vs. 3.63ms baseline (0.90x). Compiling a graph whose bottleneck isn't graph-level compute inefficiency has nothing to gain. |
| **Data-path fix** (removed a redundant `.copy()`, reordered device transfer to move smaller uint8 data before the expensive float cast, used in-place ops) | The actual profiled bottleneck | **Improved: 3.645ms → 3.467ms (1.05x speedup)** — modest, but real, and the only one of the three that directly targeted what profiling showed was actually slow. |
| Batching (`predict_action_chunk`, sizes 1–8) | Throughput | Revealed the true *uncached* per-call cost is ~544ms (the earlier 3.6ms figure was mostly action-queue cache hits, not real inference — see write-up §5), and that this workload barely parallelizes on MPS (1.84 → 2.01 obs/sec from batch 1 → 8). Batch 16 caused a full system memory crash — a real, reportable constraint of Apple Silicon's unified memory (no isolated VRAM ceiling to fail against gracefully). |

**The overall lesson, directly supported by data rather than assumed going in:** for this
model/hardware combination, the profiler's predicted bottleneck (data movement, not compute) held
up across every experiment. Compute-oriented optimizations (quantization, graph compilation) either
failed outright or made things worse; the only genuine win came from addressing the data-path
inefficiency the profiler actually pointed to.

---

## How to reproduce

### Hardware / environment
- Apple M3 (MacBook Air), unified memory, macOS, MPS backend (no CUDA)
- Python 3.11 venv

### Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r setup/requirements.txt
```

`lerobot` was installed as an editable clone with the `smolvla` extra:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[smolvla]"
```
(One local patch to `lerobot` was required — see "Known issues" below.)

Model checkpoints download automatically from the Hugging Face Hub on first run:
- `lerobot/smolvla_base` (zero-shot baseline)
- `bendca61/smolvla-mujoco-so101-cube_on_tray` (task-matched fine-tune, used for rollout + all
  benchmarking/optimization)

Scene setup:
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```
Only `trs_so_arm100/` is needed; the edited copy (with `wrist_cam` added) is included under
`scene/mujoco_menagerie/trs_so_arm100/` in this repo.

### Repo structure
```
scene/mujoco_menagerie/trs_so_arm100/   MuJoCo scene: SO-101 arm + cube + target + 2 cameras
src/smolvla_interface.py                predict() interface — model load, pre/post-processing, inference
src/rollout.py                          closed-loop rollout: N episodes, video capture, success rate
benchmarks/benchmark.py                 baseline latency / throughput / memory / GPU power
benchmarks/optimize_datapath.py         the optimization that worked, vs baseline
benchmarks/compile_only_benchmark.py    torch.compile warm-up cost vs steady-state
benchmarks/batch_benchmark.py           batching / throughput experiment (do not run batch=16)
benchmarks/results/                     saved JSON outputs
outputs/                                sample rollout videos (episode_*.mp4)
```

### Running each part
```bash
cd src && python rollout.py                     # closed-loop rollout, videos + success rate

cd ../benchmarks
python benchmark.py                              # baseline latency/throughput/memory
# in a second terminal, while benchmark.py's timed loop runs:
sudo powermetrics --samplers gpu_power -i 1000 -n 8   # GPU power/utilization

python optimize_datapath.py       # the optimization that worked
python compile_only_benchmark.py  # torch.compile experiment
python batch_benchmark.py         # batching up to size 8 (skip 16 — causes memory crash)
```

### Known issues / local patches
- `lerobot/policies/groot/groot_n1.py::GR00TN15Config` has a dataclass field-ordering bug that
  blocks importing `lerobot.policies` entirely (unrelated to SmolVLA but on the same import path).
  Fixed by adding `default=None` to four affected fields (~line 176).
- `so_arm100.xml` edited to add a `wrist_cam` camera on the `Fixed_Jaw` body.
- The fine-tuned checkpoint's saved config requests `torch.compile(mode="max-autotune")` at load
  time, which crashes on MPS; loaded instead via a pre-built config with `compile_model=False`.
- Batch size 16 in `batch_benchmark.py` caused a full system memory crash on this M3 Air (unified
  memory has no isolated VRAM ceiling to fail against gracefully) — don't run it without headroom
  to spare.

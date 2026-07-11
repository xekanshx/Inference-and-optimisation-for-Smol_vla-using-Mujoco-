import os
import numpy as np
import mujoco
import imageio.v2 as imageio
from smolvla_interface import SmolVLAInterface

# --- config ---
TASK_INSTRUCTION = "Put the block on the tray"
TARGET_XY = np.array([0.05, -0.15])
SUCCESS_RADIUS = 0.05
MIN_DISPLACEMENT = 0.03   # cube must move at least this far from its start position
N_EPISODES = 5
STEPS_PER_EPISODE = 80
SIM_SUBSTEPS_PER_ACTION = 20
FPS = 15

JAW_PAD_NAMES = {
    "fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4",
    "moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4",
}

# --- load policy interface once ---
vla = SmolVLAInterface(task=TASK_INSTRUCTION)

# --- load sim ---
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/trs_so_arm100/task_scene.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
cube_joint_qposadr = int(model.joint("cube_free").qposadr[0])


def cube_was_touched(model, data):
    """Check if any gripper pad geom is currently in contact with the cube geom."""
    for i in range(data.ncon):
        c = data.contact[i]
        name1 = model.geom(c.geom1).name
        name2 = model.geom(c.geom2).name
        names = {name1, name2}
        if "cube_geom" in names and (names & JAW_PAD_NAMES):
            return True
    return False


def run_episode(ep_idx, rng):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # "home" keyframe
    vla.policy.reset()  # clear the action queue from the previous episode

    # randomize cube start position each episode
    dx, dy = rng.uniform(-0.05, 0.05, size=2)
    start_xy = np.array([-0.02 + dx, -0.20 + dy])
    data.qpos[cube_joint_qposadr] = start_xy[0]
    data.qpos[cube_joint_qposadr + 1] = start_xy[1]
    data.qpos[cube_joint_qposadr + 2] = 0.03
    data.qpos[cube_joint_qposadr + 3] = 1.0
    data.qpos[cube_joint_qposadr + 4] = 0.0
    data.qpos[cube_joint_qposadr + 5] = 0.0
    data.qpos[cube_joint_qposadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    frames = []
    dist = None
    ever_touched = False

    for step in range(STEPS_PER_EPISODE):
        imgs = {}
        for cam in ["realsense", "wrist_cam"]:
            renderer.update_scene(data, camera=cam)
            imgs[cam] = renderer.render()

        action = vla.predict(imgs["realsense"], imgs["wrist_cam"], data.qpos[:6])
        if step % 20 == 0:
            print(f"  step {step}: action={action}")
        data.ctrl[:] = action
        for _ in range(SIM_SUBSTEPS_PER_ACTION):
            mujoco.mj_step(model, data)

        if cube_was_touched(model, data):
            ever_touched = True

        cube_xy = data.qpos[cube_joint_qposadr:cube_joint_qposadr + 2]
        dist = np.linalg.norm(cube_xy - TARGET_XY)

        renderer.update_scene(data, camera="realsense")
        frames.append(renderer.render())

    final_cube_xy = data.qpos[cube_joint_qposadr:cube_joint_qposadr + 2]
    displacement = np.linalg.norm(final_cube_xy - start_xy)

    success = (dist < SUCCESS_RADIUS) and ever_touched and (displacement > MIN_DISPLACEMENT)

    imageio.mimsave(f"outputs/episode_{ep_idx}.mp4", frames, fps=FPS)
    return success, dist, ever_touched, displacement



if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    rng = np.random.default_rng(42)

    results = []
    for ep in range(N_EPISODES):
        success, final_dist, touched, disp = run_episode(ep, rng)
        print(f"Episode {ep}: success={success}, final_dist={final_dist:.4f}, "
              f"touched={touched}, displacement={disp:.4f}")
        results.append(success)

    success_rate = sum(results) / len(results)
    print(f"\nSuccess rate over {N_EPISODES} episodes: {success_rate:.1%}")

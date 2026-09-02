import mujoco
import mujoco.viewer
import time

MODEL_PATH = "universal_robots_ur5e/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")

    for i in range(model.njnt):
        print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

    for i in range(model.nu):
        print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

    # zero qpos looks weird for this arm, use the home keyframe instead
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print("home qpos:", data.qpos)

    for _ in range(100):
        mujoco.mj_step(model, data)

    print("qpos after 100 steps:", data.qpos)
    print("qvel:", data.qvel)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 15:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

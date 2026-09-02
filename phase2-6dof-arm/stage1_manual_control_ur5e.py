import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "universal_robots_ur5e/scene.xml"

# these actuators take a target angle directly, not torque
AMPLITUDE = 0.3
FREQ_HZ = 0.2


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    home = data.qpos.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            for i in range(model.nu):
                data.ctrl[i] = home[i] + AMPLITUDE * np.sin(2 * np.pi * FREQ_HZ * t)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("final qpos:", data.qpos)


if __name__ == "__main__":
    main()

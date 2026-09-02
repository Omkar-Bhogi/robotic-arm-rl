import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("arm_model.xml")
data = mujoco.MjData(model)

print(f"joints: {model.njnt}, actuators: {model.nu}")
print([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)])

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 15:
        t = time.time() - start
        data.ctrl[0] = 0.5 * np.sin(t)
        data.ctrl[1] = 0.5 * np.sin(t * 1.5)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)

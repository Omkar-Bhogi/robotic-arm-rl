import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("arm_model.xml")
data = mujoco.MjData(model)

end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
target_pos = np.array([0.4, 0.3, 0.15])
last_print_time = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 15:
        t = time.time() - start
        data.ctrl[0] = 0.5 * np.sin(t)
        data.ctrl[1] = 0.5 * np.sin(t * 1.5)
        mujoco.mj_step(model, data)

        ee_pos = data.xpos[end_effector_id].copy()
        error = np.linalg.norm(ee_pos - target_pos)

        if t - last_print_time >= 0.5:
            print(f"t={t:.1f}s joints={data.qpos} ee={ee_pos} err={error:.3f}")
            last_print_time = t

        viewer.sync()
        time.sleep(0.005)

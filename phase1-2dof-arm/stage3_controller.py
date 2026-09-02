import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path("arm_model.xml")
data = mujoco.MjData(model)

end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")

L1 = 0.3
L2 = 0.3
target_pos = np.array([0.4, 0.3, 0.15])
target_x, target_y = target_pos[0], target_pos[1]


def inverse_kinematics(x, y, L1, L2):
    dist = min(np.sqrt(x**2 + y**2), L1 + L2)
    cos_theta2 = np.clip((dist**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return theta1, theta2


desired_theta1, desired_theta2 = inverse_kinematics(target_x, target_y, L1, L2)
print(f"IK: theta1={np.degrees(desired_theta1):.1f} deg, theta2={np.degrees(desired_theta2):.1f} deg")

Kp = 20.0
Kd = 2.0
last_print_time = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 15:
        t = time.time() - start

        theta1, theta2 = data.qpos[0], data.qpos[1]
        vel1, vel2 = data.qvel[0], data.qvel[1]

        torque1 = Kp * (desired_theta1 - theta1) - Kd * vel1
        torque2 = Kp * (desired_theta2 - theta2) - Kd * vel2

        data.ctrl[0] = np.clip(torque1, -1, 1)
        data.ctrl[1] = np.clip(torque2, -1, 1)

        mujoco.mj_step(model, data)

        ee_pos = data.xpos[end_effector_id].copy()
        error = np.linalg.norm(ee_pos - target_pos)

        if t - last_print_time >= 0.5:
            print(f"t={t:.1f}s ee={ee_pos[:2]} err={error:.4f}")
            last_print_time = t

        viewer.sync()
        time.sleep(0.005)

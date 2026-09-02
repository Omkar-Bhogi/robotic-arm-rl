import mujoco
import numpy as np

MODEL_PATH = "universal_robots_ur5e/scene.xml"
EE_BODY = "wrist_3_link"  # no gripper on this model, so last link = end effector


def show(model, data, label):
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
    print(f"\n{label}")
    print("qpos:", np.round(data.qpos, 4))
    print("ee pos:", np.round(data.xpos[ee_id], 4))
    print("ee quat:", np.round(data.xquat[ee_id], 4))


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_kinematics(model, data)
    show(model, data, "home")

    data.qpos[:] = [0.0, -1.2, 1.2, -1.5, -1.5708, 0.0]
    mujoco.mj_kinematics(model, data)
    show(model, data, "config A")

    data.qpos[:] = [1.0, -0.8, 0.9, -2.0, -1.0, 0.5]
    mujoco.mj_kinematics(model, data)
    show(model, data, "config B")


if __name__ == "__main__":
    main()

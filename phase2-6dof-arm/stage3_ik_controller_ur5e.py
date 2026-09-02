import mujoco
import numpy as np

MODEL_PATH = "universal_robots_ur5e/scene.xml"
EE_BODY = "wrist_3_link"
N_JOINTS = 6

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
ee_id = model.body(EE_BODY).id


def ee_pos():
    return data.xpos[ee_id].copy()


def solve_ik(target_pos, q_init, max_iters=200, tol=1e-4, damping=0.05):
    q = q_init.copy()
    err_norm = np.inf
    for i in range(max_iters):
        data.qpos[:N_JOINTS] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # needed for mj_jac; mj_kinematics alone leaves cdof unset
        err = target_pos - ee_pos()
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            return q, i, err_norm

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, ee_pos(), ee_id)
        J = jacp[:, :N_JOINTS]

        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(3), err)
        q = q + dq
        q = np.clip(q, model.jnt_range[:N_JOINTS, 0], model.jnt_range[:N_JOINTS, 1])

    return q, max_iters, err_norm


def run_to_target(target_pos, settle_steps=500):
    q_start = data.qpos[:N_JOINTS].copy()
    q_sol, iters, err = solve_ik(target_pos, q_start)
    print(f"IK: {iters} iters, residual {err:.5f}")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:N_JOINTS] = q_sol
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    reached = ee_pos()
    dist = np.linalg.norm(target_pos - reached)
    print(f"target {target_pos}, reached {reached}, dist {dist:.5f}")
    return q_sol, dist


if __name__ == "__main__":
    mujoco.mj_resetDataKeyframe(model, data, 0)

    targets = [
        np.array([0.3, 0.3, 0.5]),
        np.array([-0.3, 0.4, 0.4]),
        np.array([0.0, 0.5, 0.3]),
    ]

    for t in targets:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        run_to_target(t)

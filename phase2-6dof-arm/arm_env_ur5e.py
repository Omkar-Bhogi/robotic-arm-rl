import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

MODEL_PATH = "universal_robots_ur5e/scene.xml"
EE_BODY = "wrist_3_link"
N_JOINTS = 6

SUCCESS_THRESHOLD = 0.02
VEL_PENALTY_RADIUS = 0.1
VEL_PENALTY_WEIGHT = 0.05
MAX_EPISODE_STEPS = 500

WORKSPACE_MIN_RADIUS = 0.3
WORKSPACE_MAX_RADIUS = 0.7


class ArmReachEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.ee_id = self.model.body(EE_BODY).id

        self.ctrl_low = self.model.actuator_ctrlrange[:N_JOINTS, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:N_JOINTS, 1].copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)
        obs_high = np.full(N_JOINTS * 2 + 3, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.target_pos = np.zeros(3)
        self.step_count = 0

    def _ee_pos(self):
        return self.data.xpos[self.ee_id].copy()

    def _sample_target(self):
        while True:
            p = self.np_random.uniform(-1.0, 1.0, size=3)
            p[2] = abs(p[2])
            r = np.linalg.norm(p)
            if WORKSPACE_MIN_RADIUS < r < WORKSPACE_MAX_RADIUS:
                return p

    def _get_obs(self):
        qpos = self.data.qpos[:N_JOINTS].copy()
        qvel = self.data.qvel[:N_JOINTS].copy()
        return np.concatenate([qpos, qvel, self.target_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.target_pos = self._sample_target()
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)
        self.data.ctrl[:N_JOINTS] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        distance = np.linalg.norm(self._ee_pos() - self.target_pos)

        reward = -distance
        if distance < VEL_PENALTY_RADIUS:
            reward -= VEL_PENALTY_WEIGHT * np.linalg.norm(self.data.qvel[:N_JOINTS])

        terminated = bool(distance < SUCCESS_THRESHOLD)
        truncated = self.step_count >= MAX_EPISODE_STEPS

        return self._get_obs(), reward, terminated, truncated, {"distance": distance}

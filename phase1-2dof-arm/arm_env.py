import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


class ArmReachEnv(gym.Env):
    def __init__(self, xml_path="arm_model.xml", max_steps=200, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        self.target_pos = np.array([0.4, 0.3, 0.15])

        self.max_steps = max_steps
        self.current_step = 0
        self.success_threshold = 0.02

        obs_high = np.array([np.pi, np.pi, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        theta1, theta2 = self.data.qpos[0], self.data.qpos[1]
        vel1, vel2 = self.data.qvel[0], self.data.qvel[1]
        return np.array([theta1, theta2, vel1, vel2,
                          self.target_pos[0], self.target_pos[1]], dtype=np.float32)

    def _get_ee_pos(self):
        return self.data.xpos[self.end_effector_id].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = self.np_random.uniform(-0.3, 0.3)
        self.data.qpos[1] = self.np_random.uniform(-0.3, 0.3)
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]

        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        ee_pos = self._get_ee_pos()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = -distance

        # without this the policy just drifts off once it gets close
        if distance < 0.1:
            velocity_penalty = 0.01 * (abs(self.data.qvel[0]) + abs(self.data.qvel[1]))
            reward -= velocity_penalty

        terminated = bool(distance < self.success_threshold)
        truncated = bool(self.current_step >= self.max_steps)

        if terminated:
            reward += 10.0

        obs = self._get_obs()
        info = {"distance": distance, "ee_pos": ee_pos}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

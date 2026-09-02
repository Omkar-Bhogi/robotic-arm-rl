from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from arm_env import ArmReachEnv

env = ArmReachEnv()
check_env(env, warn=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

TOTAL_TIMESTEPS = 300_000
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
model.save("ppo_arm_reach")
print("saved to ppo_arm_reach.zip")

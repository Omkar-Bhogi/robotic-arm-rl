from stable_baselines3 import PPO
from arm_env import ArmReachEnv
import time

model = PPO.load("ppo_arm_reach")
env = ArmReachEnv(render_mode="human")
obs, info = env.reset()

total_reward = 0
distances = []
print("Running trained policy...")

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    distances.append(info["distance"])

    if step % 20 == 0:
        print(f"step={step} | reward={reward:.3f} | distance={info['distance']:.4f}")

    time.sleep(0.02)
    if terminated or truncated:
        print(f"Episode ended at step {step}. Total reward: {total_reward:.2f}")
        if terminated:
            print("SUCCESS: target reached!")
        break

env.close()
print(f"Final distance: {distances[-1]:.4f}")
print(f"Min distance: {min(distances):.4f}")

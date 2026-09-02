from arm_env import ArmReachEnv
import time

env = ArmReachEnv(render_mode="human")
obs, info = env.reset()
print(f"obs space: {env.observation_space}")
print(f"action space: {env.action_space}")

total_reward = 0
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 20 == 0:
        print(f"step={step} reward={reward:.3f} dist={info['distance']:.3f} done={terminated}")

    time.sleep(0.02)
    if terminated or truncated:
        print(f"ended at step {step}, total reward {total_reward:.2f}")
        break

env.close()

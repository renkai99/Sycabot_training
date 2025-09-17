from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv
import time

# Create environment
env = SycaBotEnv(render_mode="human")

# Load model (optional, just to demonstrate saving/loading)
model = PPO.load("ppo_sycabot")

# Run animation
obs, _ = env.reset()
done = False

for _ in range(300):
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    env.render()
    time.sleep(0.1)
    if done:
        obs, _ = env.reset()

env.close()

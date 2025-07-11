import gymnasium as gym
from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv

env = SycaBotEnv(render_mode=None)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

# Visualize
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

env.close()

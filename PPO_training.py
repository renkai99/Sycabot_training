from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv
import time

# Create environment
env = SycaBotEnv(render_mode="human")

# # Train policy
# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./ppo_sycabot_tensorboard/",
#     device="cuda" if hasattr(env, "device") and env.device == "cuda" else "cpu"
# )

# model.learn(
#     total_timesteps=2e6
# )

# # Save model
# model.save("ppo_sycabot")

# Load model (optional, just to demonstrate saving/loading)
model = PPO.load("ppo_sycabot")

# Run animation
obs, _ = env.reset()
done = False

for _ in range(300):
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    env.render()
    time.sleep(env.dt)
    if done:
        obs, _ = env.reset()

env.close()

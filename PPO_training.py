from stable_baselines3 import PPO
from sycabot_env import SycaBotEnv

# Create environment
env = SycaBotEnv(render_mode="human")

# Train policy
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_sycabot_tensorboard/",
    device="cuda" if hasattr(env, "device") and env.device == "cuda" else "cpu"
)

model.learn(
    total_timesteps=1e5,
    progress_bar=True
)

# Save model
model.save("ppo_sycabot")

# tensorboard --logdir ./ppo_sycabot_tensorboard/
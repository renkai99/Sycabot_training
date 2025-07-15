import torch
from stable_baselines3 import PPO

# Path to the saved PPO model
ppo_model_path = "ppo_sycabot.zip"

# Load the PPO model
model = PPO.load(ppo_model_path)

# Extract the PyTorch model
policy = model.policy

# Save the PyTorch model as a .pt file
torch.save(policy.state_dict(), "ppo_model.pt")

print("Model has been successfully converted to ppo_model.pt")

# 1.23.1
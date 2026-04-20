# Sycabot Training

This repository contains a small reinforcement learning setup for training a PPO controller on a custom `gymnasium` environment that simulates a SycaBot-style differential-drive robot. The project includes:

- `sycabot_env.py`: the custom environment definition
- `PPO_training.py`: the training script
- `sycabot_test.py`: a simple rollout script for visual testing
- `ppo_sycabot.zip`: a previously saved PPO model checkpoint

## What The Environment Does

The environment models:

- A 2D robot state: position `(x, y)` and heading `theta`
- Linear and angular velocity actions: `[v, omega]`
- Static wall obstacles
- Multiple goal locations
- A reward signal based on progress toward the nearest goal, smooth control, and collision/boundary penalties

Rendering is handled with `pygame`, so both training and testing can show the robot moving in the map when `render_mode="human"` is used.

## Requirements

This project is set up for `python3` and pip.

Install the main dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you also want the extra monitoring and notebook tooling:

```bash
pip install -r requirements-dev.txt
```

## Repository Structure

```text
.
├── PPO_training.py
├── sycabot_env.py
├── sycabot_test.py
├── ppo_sycabot.zip
├── requirements.txt
└── requirements-dev.txt
```

## Training

Run training with:

```bash
python3 PPO_training.py
```

What this script currently does:

- Creates `SycaBotEnv(render_mode="human")`
- Trains a PPO policy with `MlpPolicy`
- Logs TensorBoard data into `./ppo_sycabot_tensorboard/`
- Saves the trained model as `ppo_sycabot.zip`

Important implementation details:

- The script is currently configured for `total_timesteps=5e5`
- Rendering is enabled during training, which is useful for debugging but slower than headless training
- The device selection falls back to CPU unless your environment object exposes a `device` attribute set to `"cuda"`

If you want faster training, a reasonable first change is to edit [PPO_training.py](/home/ryankey/Git repos/Sycabot_training/PPO_training.py:1) and switch:

- `render_mode="human"` to `render_mode=None`

## Testing A Trained Model

Run the evaluation / animation script with:

```bash
python3 sycabot_test.py
```

This script:

- Loads the saved PPO model from `ppo_sycabot.zip`
- Resets the environment
- Predicts actions from the policy
- Renders 300 environment steps with a short delay between frames

Before running the test script, make sure the trained model file exists. `PPO.load("ppo_sycabot")` will load the `ppo_sycabot.zip` checkpoint automatically.

## TensorBoard

Training logs are written to:

```text
./ppo_sycabot_tensorboard/
```

To inspect training metrics:

```bash
tensorboard --logdir ./ppo_sycabot_tensorboard/
```

Then open the local URL shown by TensorBoard in your browser, typically:

```text
http://localhost:6006/
```

If the log directory does not exist yet, run training first so TensorBoard event files are created.

## How To Modify The Setup

Typical places to change behavior:

- [sycabot_env.py](/home/ryankey/Git repos/Sycabot_training/sycabot_env.py:1): reward function, obstacles, goals, action space, observation space, boundary checks
- [PPO_training.py](/home/ryankey/Git repos/Sycabot_training/PPO_training.py:1): PPO hyperparameters, total timesteps, logging path, rendering mode
- [sycabot_test.py](/home/ryankey/Git repos/Sycabot_training/sycabot_test.py:1): rollout length, render speed, checkpoint path

Examples:

- Change `total_timesteps` to train longer or shorter
- Tune the reward shaping in `step()`
- Add or remove obstacles in `_add_obstacles()`
- Adjust goal positions in `_add_goals()`
- Slow down or speed up test playback by changing `time.sleep(0.1)`

## Notes

- The training and testing scripts are plain scripts, not command-line tools, so configuration is currently done by editing the Python files directly.
- `ppo_sycabot.zip` is a saved Stable-Baselines3 model file.
- `pygame` rendering requires a graphical environment. If you are running on a headless server, disable rendering in the scripts.

## Troubleshooting

If `python3 PPO_training.py` fails because a package is missing:

```bash
pip install -r requirements.txt
```

If TensorBoard is not found:

```bash
pip install -r requirements-dev.txt
```

If the test script cannot find a model file:

- Train first with `python3 PPO_training.py`, or
- Update the path in `sycabot_test.py` to point to the correct checkpoint

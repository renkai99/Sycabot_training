# Sycabot Training

This repository contains a reinforcement learning setup for training a PPO controller on a custom `gymnasium` environment for a SycaBot-style differential-drive robot.

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

## Training

Run training with:

```bash
python3 PPO_training.py
```

What this script currently does:

- Creates `SycaBotEnv(render_mode=None)`
- Trains a PPO policy with `MlpPolicy`
- Logs TensorBoard data into `./ppo_sycabot_tensorboard/`
- Saves the trained model as `ppo_sycabot.zip`

Important implementation details:

- The script is currently configured for `total_timesteps=1e5`
- The device selection falls back to CPU unless your environment object exposes a `device` attribute set to `"cuda"`

## Testing A Trained Model

Run the evaluation / animation script with:

```bash
python3 sycabot_test.py
```

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

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class SycaBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.1  # Time step

        # Robot parameters
        self.R = 0.05  # wheel radius [m]
        self.L = 0.15  # distance between wheels [m]

        # Deadzone parameters and gains
        self.d_r_minus, self.d_r_plus = -0.2, 0.2
        self.d_l_minus, self.d_l_plus = -0.2, 0.2
        self.alpha_r_minus, self.alpha_r_plus = 10.0, 10.0
        self.alpha_l_minus, self.alpha_l_plus = 10.0, 10.0

        # Action space: [u_r, u_l]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), 
                                       high=np.array([1.0, 1.0]), dtype=np.float32)

        # Observation space: [x, y, theta, x_goal, y_goal]
        obs_high = np.array([5.0, 5.0, np.pi, 5.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initial state
        self.state = np.zeros(3)  # [x, y, theta]
        self.goal = np.zeros(2)

        # Visualization
        self.window = None
        self.clock = None
        self.screen_size = 500
        self.scale = 100  # 1 m = 100 px

    def step(self, action):
        u_r, u_l = action

        omega_r = self.deadzone_response(u_r, self.d_r_minus, self.d_r_plus, self.alpha_r_minus, self.alpha_r_plus)
        omega_l = self.deadzone_response(u_l, self.d_l_minus, self.d_l_plus, self.alpha_l_minus, self.alpha_l_plus)

        v = (self.R / 2) * (omega_r + omega_l)
        omega = (self.R / self.L) * (omega_r - omega_l)

        x, y, theta = self.state
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += omega * self.dt
        theta = self.wrap_angle(theta)

        self.state = np.array([x, y, theta])

        obs = np.concatenate([self.state, self.goal])
        done = np.linalg.norm(self.state[:2] - self.goal) < 0.1
        reward = -np.linalg.norm(self.state[:2] - self.goal)

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=[0.5, 0.5, -np.pi], high=[1.0, 1.0, np.pi])
        self.goal = self.np_random.uniform(low=[2.0, 2.0], high=[4.0, 4.0])
        obs = np.concatenate([self.state, self.goal])
        return obs, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Draw goal
        goal_pos = (int(self.goal[0] * self.scale), int(self.goal[1] * self.scale))
        pygame.draw.circle(self.window, (0, 255, 0), goal_pos, 10)

        # Draw robot
        x, y, theta = self.state
        robot_pos = (int(x * self.scale), int(y * self.scale))
        heading = (int(robot_pos[0] + 15 * np.cos(theta)), int(robot_pos[1] + 15 * np.sin(theta)))
        pygame.draw.circle(self.window, (0, 0, 255), robot_pos, 10)
        pygame.draw.line(self.window, (255, 0, 0), robot_pos, heading, 2)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    def deadzone_response(self, u, d_minus, d_plus, alpha_minus, alpha_plus):
        if u < d_minus:
            return alpha_minus * (u - d_minus)
        elif u > d_plus:
            return alpha_plus * (u - d_plus)
        else:
            return 0.0

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

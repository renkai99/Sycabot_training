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
        self.R = 0.032  # wheel radius [m] (wheel_diameter / 2)
        self.L = 0.12   # distance between wheels [m] (wheel_separation)

        # Deadzone parameters and gains
        self.d_r_minus, self.d_r_plus = -0.09, 0.09
        self.d_l_minus, self.d_l_plus = -0.09, 0.09
        self.alpha_r_minus, self.alpha_r_plus = 50.0, 50.0
        self.alpha_l_minus, self.alpha_l_plus = 50.0, 50.0

        # Action space: [v, w]
        self.action_space = spaces.Box(low=np.array([-0.6, -np.pi]), 
                                       high=np.array([0.6, np.pi]), dtype=np.float32)

        # # Observation space
        obs_high = np.array([1.6, 3.5, np.pi, 1.6, 3.5, np.pi, 3.0, np.pi, 3.0, np.pi, 3.0, np.pi, 3.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Initial state
        self.last_state = np.zeros(3)  # [x, y, theta]
        self.state = np.zeros(3)  # [x, y, theta]

        # Obstacles and goals
        self.obstacles = self._add_obstacles()
        self.goals = self._add_goals()
        self.step_count = 0
        self.reward = 0

        # Visualization
        self.window = None
        self.clock = None
        self.screen_size = 700
        self.scale = 100  # 1 m = 100 px

    def _add_obstacles(self):
        return [
            [[-1.498, 3.001], [0.001, 3.000]],
            [[1.051, 3.001], [1.494, 3.000]],
            [[1.494, 3.000], [1.493, 0.430]],
            [[1.494, -0.374], [1.497, -2.998]],
            [[0.002, -2.999], [1.497, -2.998]],
            [[-1.498, -2.999], [-1.047, -2.999]],
            [[-1.496, -0.500], [-1.498, -2.999]],
            [[-1.496, 0.750], [-1.495, 0.299]],
            [[-1.498, 2.998], [-1.498, 1.553]],
            [[-0.481, 2.382], [0.879, 1.356]],
            [[-1.498, 1.553], [-0.700, 1.551]],
            [[1.018, 0.429], [1.493, 0.430]],
            [[0.141, 1.040], [-0.269, 0.524]],
            [[-1.496, 0.526], [-0.269, 0.524]],
            [[-0.269, 0.524], [-0.261, -0.008]],
            [[-0.261, -0.008], [0.480, -0.008]],
            [[0.011, -0.008], [0.011, -0.486]],
            [[-1.496, -0.859], [-0.492, -0.860]],
            [[0.922, -0.613], [0.924, -2.093]],
            [[0.260, -1.084], [0.260, -2.093]],
            [[-0.665, -2.094], [0.924, -2.093]],
            [[-0.685, -2.103], [-0.931, -2.414]],
        ]

    def _min_distance_to_obstacles(self):
        min_distances = []
        min_orientations = []
        robot_pos = self.state[:2]
        robot_theta = self.state[2]

        for obstacle in self.obstacles:
            start, end = np.array(obstacle[0]), np.array(obstacle[1])
            obstacle_vec = end - start
            robot_vec = robot_pos - start
            proj_length = np.dot(robot_vec, obstacle_vec) / np.linalg.norm(obstacle_vec)**2
            if proj_length < 0:
                closest_point = start
            elif proj_length > 1:
                closest_point = end
            else:
                closest_point = start + proj_length * obstacle_vec
            
            distance = np.linalg.norm(robot_pos - closest_point)
            orientation = np.arctan2(closest_point[1] - robot_pos[1], closest_point[0] - robot_pos[0]) - robot_theta
            min_distances.append(distance)
            min_orientations.append(orientation)

        # Get indices of the 3 closest obstacles
        closest_indices = np.argsort(min_distances)[:3]
        result = []
        for i in closest_indices:
            result.append(min_distances[i])
            result.append(min_orientations[i])
        return result

    def _distance_to_goals(self, robot_pos):
        distance = []
        orientations = []
        for goal in self.goals:
            goal_point = np.array(goal)
            dist = np.linalg.norm(robot_pos - goal_point)
            orient = np.arctan2(goal_point[1] - robot_pos[1], goal_point[0] - robot_pos[0])
            distance.append(dist)
            orientations.append(orient)
        min_idx = np.argmin(distance)
        return [distance[min_idx], orientations[min_idx]]

    def _add_goals(self):
        return [
            [-1.497, 1.1515],  # Centroid of [[-1.498, 1.553], [-1.496, 0.750]]
            [-1.4945, -0.1005],  # Centroid of [[-1.495, 0.299], [-1.494, -0.500]]
            [-0.524, -2.999],  # Centroid of [[-1.050, -2.999], [0.002, -2.999]]
            [1.4955, 0.028],  # Centroid of [[1.494, -0.374], [1.497, 0.430]]
            [0.526, 3.001],  # Centroid of [[0.001, 3.001], [1.051, 3.001]]
        ]
    
    def is_out_of_boundary(self, state):
        x, y, _ = state
        return not (-1.2 <= x <= 1.2 and -3.5 <= y <= 3.5)

    def step(self, action):
        v, omega = action

        # omega_r = self.deadzone_response(u_r, self.d_r_minus, self.d_r_plus, self.alpha_r_minus, self.alpha_r_plus)
        # omega_l = self.deadzone_response(u_l, self.d_l_minus, self.d_l_plus, self.alpha_l_minus, self.alpha_l_plus)

        # v = (self.R / 2) * (omega_r + omega_l)
        # omega = (self.R / self.L) * (omega_r - omega_l)

        self.last_state = self.state.copy()
        x, y, theta = self.state
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += omega * self.dt
        theta = self.wrap_angle(theta)

        self.state = np.array([x, y, theta])

        min_obs_distance = self._min_distance_to_obstacles()
        curr_distance = self._distance_to_goals(self.state[:2])
        last_distance = self._distance_to_goals(self.last_state[:2])

        obs = np.concatenate([self.state, self.last_state, min_obs_distance, curr_distance])
        self.step_count += 1

        if curr_distance[0] < 0.2:
            done = True
            self.step_count = 0
            self.reward += 1000
        elif min_obs_distance[0] < 0.1:
            done = True
            self.reward = -100  # Negative reward for hitting an obstacle
        elif self.is_out_of_boundary(self.state):
            done = True
            self.reward = -100  # Negative reward for going out of boundary
        else:
            done = False

        step_decay = max(0, 1 - 1e-3 * self.step_count)
        distance_reward = 100 * (last_distance[0] - curr_distance[0]) * step_decay
        
            
        self.reward = distance_reward # - 0.001 * np.abs(omega)  - 0.001 *np.linalg.norm(theta - desired_orientation) 

        return obs, self.reward, done, False, {}

    def reset(self, seed=None, options=None):
        # super().reset(seed=seed)
        # Sample a feasible state: not too close to obstacles

        while True:
            self.state = self.np_random.uniform(low=[-1.0, -3.0, -np.pi], high=[1.0, 3.0, np.pi])
            min_obs_distance = self._min_distance_to_obstacles()
            if min_obs_distance[0] > 0.1:
                break

        self.last_state = self.state.copy()
        self.step_count = 0
        self.reward = 0
        min_obs_distance = self._min_distance_to_obstacles()
        curr_distance = self._distance_to_goals(self.state[:2])

        obs = np.concatenate([self.state, self.last_state, min_obs_distance, curr_distance])
        return obs, {}

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
    
    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))

        # Draw obstacles
        for obstacle in self.obstacles:
            start = (int(obstacle[0][0] * self.scale + self.screen_size / 2),
                     int(self.screen_size / 2 - obstacle[0][1] * self.scale))
            end = (int(obstacle[1][0] * self.scale + self.screen_size / 2),
                   int(self.screen_size / 2 - obstacle[1][1] * self.scale))
            pygame.draw.line(self.window, (0, 0, 0), start, end, 6)

        # Draw goals
        for goal in self.goals:
            goal_pos = (int(goal[0] * self.scale + self.screen_size / 2),
                int(self.screen_size / 2 - goal[1] * self.scale))
            pygame.draw.circle(self.window, (0, 255, 0), goal_pos, 10)

        # Draw robot
        x, y, theta = self.state
        robot_pos = (int(x * self.scale + self.screen_size / 2),
                     int(self.screen_size / 2 - y * self.scale))
        heading = (int(robot_pos[0] + 15 * np.cos(theta)),
                   int(robot_pos[1] - 15 * np.sin(theta)))
        pygame.draw.circle(self.window, (0, 0, 255), robot_pos, 10)
        pygame.draw.line(self.window, (255, 0, 0), robot_pos, heading, 2)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        return (theta + np.pi) % (2 * np.pi) - np.pi

    def deadzone_response(self, u, d_minus, d_plus, alpha_minus, alpha_plus):
        if u < d_minus:
            return alpha_minus * (u - d_minus)
        elif u > d_plus:
            return alpha_plus * (u - d_plus)
        else:
            return 0.0

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi


# rl/environment.py

import gym
from gym import spaces
import numpy as np

class NetworkEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data, labels):
        super(NetworkEnv, self).__init__()
        self.data = data.toarray() if hasattr(data, "toarray") else data
        self.labels = labels.values

        # Observation space: vector of processed features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )

        # Action space: binary classification (0 = benign, 1 = malicious)
        self.action_space = spaces.Discrete(2)

        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        correct_action = self.labels[self.current_step]
        reward = 1 if action == correct_action else -1

        self.current_step += 1
        done = self.current_step >= len(self.data)

        return self.data[self.current_step], reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
# rl/environment.py

import gym
from gym import spaces
import numpy as np


class NetworkEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, labels):
        super(NetworkEnv, self).__init__()
        self.data = data.toarray() if hasattr(data, "toarray") else data
        self.labels = labels.values
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # Benigno/Malevolo

    def reset(self):
        self.current_step = 0
        obs = self.data[self.current_step].astype(np.float32).flatten()
        return obs

    def step(self, action):
        correct_action = self.labels[self.current_step]

        # Reward shaping avanzato
        if action == correct_action:
            reward = 1
        else:
            # Penalità diversa per falsi positivi/negativi
            if correct_action == 0 and action == 1:
                reward = -1  # Falso positivo
            elif correct_action == 1 and action == 0:
                reward = -2  # Falso negativo → penalità maggiore
            else:
                reward = -1

        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = (
            self.data[self.current_step].astype(np.float32).flatten()
            if not done
            else None
        )
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

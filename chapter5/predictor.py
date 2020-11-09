from typing import Callable
from typing import Tuple

import numpy as np
from tqdm import trange

from env import Enviroment
from trajectory import Trajectory


class MonteCarloPredictor:
    """Monte Carlo State Value Predictor"""

    def __init__(self, env: Enviroment):
        self.env = env
        self._obs_space = env.obs_space()
        self._act_space = env.act_space()
        self.reset()

    def reset(self):
        shape = [
            _max - _min + 1
            for _max, _min in zip(self._obs_space.max, self._obs_space.min)
        ]
        self.state_value = np.zeros(shape=shape, dtype=np.float32)
        self.n_state = np.zeros_like(self.state_value, dtype=np.int32)

    def idx_to_state(self, idx: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x + y for x, y in zip(idx, self._obs_space.min))

    def state_to_idx(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x - y for x, y in zip(state, self._obs_space.min))

    @staticmethod
    def generate_episode(env, policy: Callable):
        trajectory = Trajectory()
        env.reset()
        current_state = env.state
        finished = False
        reward = 0
        while not finished:
            action = policy(current_state)
            trajectory.add_step(finished, current_state, action, reward)
            finished, reward, new_state = env.step(action)
            current_state = new_state
        trajectory.add_step(finished, current_state, action, reward)
        return trajectory

    def predict(self, policy: Callable, n_iters: int = 1):
        for _ in trange(n_iters):
            trajectory = self.generate_episode(self.env, policy)
            g = 0
            for i in range(len(trajectory) - 1, 0, -1):
                g += trajectory[i].reward
                previous_states = set(x.state for x in trajectory[0 : i - 1])
                if trajectory[i - 1].state not in previous_states:
                    index = self.state_to_idx(trajectory[i - 1].state)
                    self.n_state[index] += 1
                    self.state_value[index] += (
                        g - self.state_value[index]
                    ) / self.n_state[index]

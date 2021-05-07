import numpy as np

from chapter5.env import Enviroment, Space
from chapter5.types import Observation, State


class RandomWalk(Enviroment):
    def __init__(self, n: int = 5):
        assert n % 2, "n must be odd"
        self.n = n
        self._obs_space = Space(dims=1, min=0, max=self.n)
        self._act_space = Space(dims=1, min=0, max=1)
        self.reset()

    def reset(self):
        self.state = (self.n // 2) + 1

    def step(self, action: int) -> Observation:
        assert 0 <= action <= 1
        self.state += action
        if self.state > self.n:
            done = True
            reward = 1
        elif self.state < 0:
            done = True
            reward = 0
        else:
            done = False
            reward = 0
        return Observation(done, self.state, reward)

    def legal_actions(self, state: State) -> np.ndarray:
        return np.array([0, 1], dtype=np.int32)

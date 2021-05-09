import numpy as np

from chapter5.env import Enviroment, DiscreteDim, DiscreteSpace
from chapter5.types import Observation, State


class RandomWalk(Enviroment):
    def __init__(self, n: int = 5):
        assert n % 2, "n must be odd"
        self.n = n
        self._act_space = DiscreteDim(1, minimum=0)
        self._obs_space = DiscreteSpace(DiscreteDim(self.n))
        self.reset()

    def reset(self):
        self.state = (self.n // 2) + 1

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), "Invalid action"
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

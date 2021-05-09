import numpy as np

from chapter5.env import Enviroment, DiscreteDim, DiscreteSpace
from chapter5.types import Observation, State


class RandomWalk(Enviroment):
    def __init__(self, n: int = 5):
        assert n % 2, "n must be odd"
        self.limit = n // 2
        self._act_space = DiscreteDim(1, minimum=0)
        self._obs_space = DiscreteSpace(DiscreteDim(n, minimum=-self.limit))
        self.reset()

    def reset(self):
        self.s = 0

    @property
    def state(self) -> State:
        return (self.s,)

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), "Invalid action"
        self.s += action if action else -1
        reward = 1 if self.s > self.limit else 0
        done = np.abs(self.s) > self.limit
        return Observation(done, self.state, reward)

import numpy as np
from typing import List, Tuple

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


class WindyGridWorld(Enviroment):

    _map_action: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, n: int, m: int, winds: List[int], reward_pos: Tuple[int, int]):
        assert (l := len(n)) == m, f"winds length ({l}) greater than dimension m ({m})"
        self.n = n
        self.m = m
        self.winds = winds
        self.reward_pos = reward_pos
        self._act_space = DiscreteDim(4, minimum=0)
        self._obs_space = DiscreteSpace(DiscreteDim(self.n), DiscreteDim(self.m))
        self.reset()

    def reset(self):
        self._ns = self._n // 2 + 1
        self._ms = 0

    @property
    def state(self) -> State:
        return (self._ns, self._ms)

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), "Invalid action"

        self._ns += self.winds[self._nm] + self._map_action[action][0]
        self._ms += self._map_action[action][1]
        reward = -1
        if (
            0 > self._ns
            or self._ns >= self.n
            or 0 > self._ms
            or self._ms >= self.m  # Out
        ) or (
            self._ns == self.reward_pos[0] and self._ms == self.reward_pos[1]  # Win
        ):
            done = True

        return Observation(done, self.state, reward)

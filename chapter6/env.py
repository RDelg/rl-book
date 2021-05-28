import numpy as np
from typing import List, Optional, Tuple

from chapter5.env import Enviroment, DiscreteDim, DiscreteSpace
from chapter5.types import Observation, State


class RandomWalk(Enviroment):
    def __init__(self, n: int = 5):
        assert n % 2, "n must be odd"
        self.n = n
        self._act_space = DiscreteDim(2)
        self._obs_space = DiscreteSpace(DiscreteDim(n))
        self.reset()

    def reset(self):
        self.s = self.n // 2 + 1

    @property
    def state(self) -> State:
        return (self.s,)

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), "Invalid action"
        self.s += action if action else -1
        reward = 1 if self.s >= self.n else 0
        done = self.s >= self.n or self.s < 0
        return Observation(done, self.state, reward)


class WindyGridWorld(Enviroment):

    _map_action: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, n: int, m: int, winds: List[int], reward_pos: Tuple[int, int]):
        assert (
            l := len(winds)
        ) == m, f"winds length ({l}) greater than dimension m ({m})"
        self.n = n
        self.m = m
        self.winds = winds
        self.reward_pos = reward_pos
        self._act_space = DiscreteDim(4)
        self._obs_space = DiscreteSpace(DiscreteDim(self.n), DiscreteDim(self.m))
        self.reset()

    def reset(self):
        self._ns = self.n // 2
        self._ms = 0

    @property
    def state(self) -> State:
        return (self._ns, self._ms)

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), f"Invalid action {action}"
        clip = lambda x, maximum: 0 if x < 0 else maximum - 1 if x >= maximum else x

        self._ns += self.winds[self._ms] + self._map_action[action][0]
        self._ms += self._map_action[action][1]
        self._ns = clip(self._ns, self.n)
        self._ms = clip(self._ms, self.m)
        reward = -1

        done = self._ns == self.reward_pos[0] and self._ms == self.reward_pos[1]
        return Observation(done, self.state, reward)


class CliffGridWorld(Enviroment):

    _map_action: List[Tuple[int, int]] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, n: int, m: int):
        assert m > 2, f"m ({2}) must be greater than 2"
        self.n = n
        self.m = m
        self._act_space = DiscreteDim(4)
        self._obs_space = DiscreteSpace(DiscreteDim(self.n), DiscreteDim(self.m))
        self.reset()

    def reset(self):
        self._ns = 0
        self._ms = 0

    @property
    def state(self) -> State:
        return (self._ns, self._ms)

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), f"Invalid action {action}"
        clip = lambda x, maximum: 0 if x < 0 else maximum - 1 if x >= maximum else x
        self._ns += self._map_action[action][0]
        self._ms += self._map_action[action][1]
        self._ns = clip(self._ns, self.n)
        self._ms = clip(self._ms, self.m)
        reward = -1
        cliff = 0 < self._ms < self.m - 1 and self._ns == 0
        if cliff:
            self._ns, self._ms = 0, 0
            reward = -100
        done = self._ns == 0 and self._ms == (self.m - 1)
        return Observation(done, self.state, reward)


class DoubleState(Enviroment):
    def __init__(self, n: Optional[int] = 10):
        assert n >= 10, "n must be greater than 10"
        self.n = n
        self._act_space = DiscreteDim(self.n)
        self._obs_space = DiscreteSpace(DiscreteDim(2))
        self._a_legal_actions = [0, 1]
        self._b_legal_actions = list(range(1, n))
        self.reset()

    def reset(self):
        self._s = 0

    @property
    def state(self) -> State:
        return (self._s,)

    def legal_actions(self, state: State) -> List[int]:
        return self._b_legal_actions if state[0] else self._a_legal_actions

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), f"Invalid action {action}"
        if self._s == 0:
            if action == 0:
                reward = 0
                done = True
            else:
                reward = 0
                done = False
                self._s = 1
        else:
            done = True
            reward = np.random.randn() - 0.1
        return Observation(done, self.state, reward)

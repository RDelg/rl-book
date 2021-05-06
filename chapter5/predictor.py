from abc import ABCMeta, abstractmethod
from typing import Callable
from typing import Optional

import numpy as np
from tqdm import trange

from .env import Enviroment
from .types import State, StateIndex, Trajectory


class Predictor(metaclass=ABCMeta):
    def __init__(self, env: Enviroment, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self._obs_space = env.obs_space()
        self._act_space = env.act_space()
        self.reset()

    def reset(self):
        shape = [
            _max - _min + 1
            for _max, _min in zip(self._obs_space.max, self._obs_space.min)
        ]
        self.V = np.zeros(shape=shape, dtype=np.float32)

    def idx_to_state(self, idx: StateIndex) -> State:
        return tuple(x + y for x, y in zip(idx, self._obs_space.min))

    def state_to_idx(self, state: StateIndex) -> State:
        return tuple(x - y for x, y in zip(state, self._obs_space.min))

    @abstractmethod
    def predict(self) -> None:
        raise NotImplementedError


class MonteCarloPredictor(Predictor):
    """Monte Carlo State Value Predictor"""

    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(MonteCarloPredictor, self).__init__(env, gamma)
        self.N = np.zeros_like(self.V, dtype=np.int32)

    def generate_episode(
        self,
        policy: Callable[[np.ndarray], int],
        init_state: Optional[State] = None,
    ) -> Trajectory:
        if init_state is None:
            self.env.reset()
        else:
            self.env.state = init_state
        trajectory = Trajectory()
        current_state = self.env.state
        finished = False
        reward = 0
        while not finished:
            action = policy(current_state)
            trajectory.add_step(finished, current_state, reward, action)
            finished, new_state, reward = self.env.step(action)
            current_state = new_state
        trajectory.add_step(finished, current_state, reward, None)
        return trajectory

    def predict(
        self,
        policy: Callable[[np.ndarray], int],
        n_iters: int = 1,
        init_state: Optional[State] = None,
    ):
        for _ in trange(n_iters):
            trajectory = self.generate_episode(policy, init_state=init_state)
            G = 0
            previous_states = [x.state + (x.action,) for x in trajectory[0:-1]]
            for i in range(len(trajectory) - 2, -1, -1):
                G += self.gamma * trajectory[i + 1].reward
                previous_states.pop()
                s = trajectory[i].state
                if s not in previous_states:
                    s_idx = self.state_to_idx(s)
                    self.N[s_idx] += 1
                    self.V[s_idx] += (G - self.V[s_idx]) / self.N[s_idx]


class TDPredictor(Predictor):
    """TD State Value Predictor"""

    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(TDPredictor, self).__init__(env, gamma)

    def predict(
        self,
        policy: Callable[[np.ndarray], int],
        alpha: float = 0.01,
        n_iters: int = 1,
        init_state: Optional[State] = None,
    ):
        for _ in trange(n_iters):
            if init_state is None:
                self.env.reset()
            else:
                self.env.state = init_state
            current_state = self.env.state
            while not finished:
                action = policy(current_state)
                finished, new_state, reward = self.env.step(action)
                c_s_idx = self.state_to_idx(current_state)
                n_s_idx = self.state_to_idx(new_state)
                self.V[c_s_idx] += alpha * (
                    reward + self.gamma * self.V[n_s_idx] - self.V[c_s_idx]
                )
                current_state = new_state

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
        self.reset()

    def reset(self):
        shape = [dim.n for dim in self.env.obs_space]
        self._obs_space_mins = [x.minimum for x in self.env.obs_space.dims]
        self.V = np.zeros(shape=shape, dtype=np.float32)

    def idx_to_state(self, idx: StateIndex) -> State:
        return tuple(x + _min for x, _min in zip(idx, self._obs_space_mins))

    def state_to_idx(self, state: StateIndex) -> State:
        return tuple(x - _min for x, _min in zip(state, self._obs_space_mins))

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
        alpha: float = 0.01,
        n_iters: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
    ):
        for _ in trange(n_iters, desc="Value prediction iter", disable=disable_tqdm):
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
                    self.V[s_idx] += alpha * (G - self.V[s_idx]) / self.N[s_idx]

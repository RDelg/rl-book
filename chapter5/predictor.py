from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, List

import numpy as np
from tqdm import trange, tqdm

from .env import Enviroment
from .types import State, Trajectory


class Predictor(metaclass=ABCMeta):
    def __init__(self, env: Enviroment, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.reset()

    def reset(self):
        shape = [dim.n for dim in self.env.obs_space]
        self.V = np.zeros(shape=shape, dtype=np.float32)

    @abstractmethod
    def predict(self) -> None:
        raise NotImplementedError


class MonteCarloPredictor(Predictor):
    """Monte Carlo State Value Predictor"""

    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(MonteCarloPredictor, self).__init__(env, gamma)
        self.N = np.zeros_like(self.V, dtype=np.int32)
        self.history: List[Trajectory] = []

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
        n_episodes: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
        batch: Optional[bool] = False,
    ):
        for _ in trange(n_episodes, desc="Value prediction iter", disable=disable_tqdm):
            if batch:
                self._batch_update(alpha)
            trajectory = self.generate_episode(policy, init_state=init_state)
            self._update_V(trajectory, alpha)
            self.history.append(trajectory)

    def _batch_update(self, alpha: float, disable_tqdm: Optional[bool] = True):
        for trajectory in tqdm(self.history, desc="Batch update", disable=disable_tqdm):
            self._update_V(trajectory, alpha)

    def _update_V(self, trajectory: Trajectory, alpha: float):
        G = 0
        previous_states = [x.state + (x.action,) for x in trajectory[0:-1]]
        for i in range(len(trajectory) - 2, -1, -1):
            G += self.gamma * trajectory[i + 1].reward
            previous_states.pop()
            state = trajectory[i].state
            if state not in previous_states:
                self.N[state] += 1
                self.V[state] += alpha * (G - self.V[state]) / self.N[state]

    def reset(self, init_value: float = 0.0):
        super(MonteCarloPredictor, self).reset()
        self.V[...] = init_value

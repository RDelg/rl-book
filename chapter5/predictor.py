from abc import ABCMeta
from typing import Callable
from typing import Optional

import numpy as np
from tqdm import trange

from .env import Enviroment
from .types import State, StateIndex, Trajectory


class Predictor(metaclass=ABCMeta):
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
        self.V = np.zeros(shape=shape, dtype=np.float32)
        self.N = np.zeros_like(self.V, dtype=np.int32)

    def idx_to_state(self, idx: StateIndex) -> State:
        return tuple(x + y for x, y in zip(idx, self._obs_space.min))

    def state_to_idx(self, state: StateIndex) -> State:
        return tuple(x - y for x, y in zip(state, self._obs_space.min))


class MonteCarloPredictor(Predictor):
    """Monte Carlo State Value Predictor"""

    def __init__(self, env: Enviroment):
        super(MonteCarloPredictor, self).__init__(env)

    def generate_episode(
        self, policy: Callable[[np.ndarray], int], init_state: Optional[State] = None,
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

    def predict_on_policy(
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
                G += trajectory[i + 1].reward
                previous_states.pop()
                s = trajectory[i].state
                if s not in previous_states:
                    s_idx = self.state_to_idx(s)
                    self.N[s_idx] += 1
                    self.V[s_idx] += (G - self.V[s_idx]) / self.N[s_idx]


class TDPredictor(Predictor):
    """TD State Value Predictor"""

    def __init__(self, env: Enviroment):
        super(TDPredictor, self).__init__(env)


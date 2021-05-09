from typing import Callable, Optional

import numpy as np
from tqdm import trange

from chapter5.types import State
from chapter5.predictor import Predictor
from chapter5.env import Enviroment


class TDPredictor(Predictor):
    """TD State Value Predictor"""

    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(TDPredictor, self).__init__(env, gamma)

    def reset(self, init_value: float = 0.0):
        super(TDPredictor, self).reset()
        self.V[...] = init_value

    def predict(
        self,
        policy: Callable[[np.ndarray], int],
        alpha: float = 0.01,
        n_iters: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
    ):
        for _ in trange(n_iters, desc="Value prediction iter", disable=disable_tqdm):
            if init_state is None:
                self.env.reset()
            else:
                self.env.state = init_state
            finished = False

            while not finished:
                current_state = self.env.state
                action = policy(current_state)
                finished, new_state, reward = self.env.step(action)
                c_s_idx = self.state_to_idx(current_state)
                n_s_idx = self.state_to_idx(new_state)
                next_v = 0 if finished else self.V[n_s_idx]
                self.V[c_s_idx] += alpha * (
                    reward + self.gamma * next_v - self.V[c_s_idx]
                )
                current_state = new_state

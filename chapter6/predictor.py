from typing import Callable, Optional, List

import numpy as np
from tqdm import trange, tqdm

from chapter5.types import State, Trajectory
from chapter5.predictor import Predictor
from chapter5.env import Enviroment


class TDPredictor(Predictor):
    """TD State Value Predictor"""

    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(TDPredictor, self).__init__(env, gamma)
        self.history: List[Trajectory] = []

    def reset(self, init_value: float = 0.0):
        super(TDPredictor, self).reset()
        self.V[...] = init_value

    def predict(
        self,
        policy: Callable[[np.ndarray], int],
        alpha: float = 0.01,
        n_episodes: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
        batch: Optional[bool] = False,
    ):
        for _ in trange(
            n_episodes,
            desc=f"Value prediction iter {'(batched)' if batch else ''}",
            disable=disable_tqdm,
        ):
            if batch:
                self._batch_update(alpha)
            if init_state is None:
                self.env.reset()
            else:
                self.env.state = init_state
            done = False
            trajectory = Trajectory()
            while not done:
                current_state = self.env.state
                action = policy(current_state)
                done, new_state, reward = self.env.step(action)
                self._update_V(
                    alpha,
                    current_state,
                    new_state,
                    done,
                    reward,
                )
                current_state = new_state
                trajectory.add_step(done, current_state, reward, action)
            trajectory.add_step(done, current_state, reward, None)
            self.history.append(trajectory)

    def _batch_update(self, alpha: float, disable_tqdm: Optional[bool] = True):
        for trajectory in tqdm(self.history, desc="Batch update", disable=disable_tqdm):
            for i in range(len(trajectory) - 1):
                if not trajectory[i + 1].done:
                    self._update_V(
                        alpha,
                        trajectory[i].state,
                        trajectory[i].state,
                        trajectory[i].done,
                        trajectory[i].reward,
                    )

    def _update_V(
        self,
        alpha: float,
        current_state: State,
        new_state: State,
        done: bool,
        reward: float,
    ) -> None:
        c_s_idx = self.state_to_idx(current_state)
        n_s_idx = self.state_to_idx(new_state)
        next_v = 0 if done else self.V[n_s_idx]
        self.V[c_s_idx] += alpha * (reward + self.gamma * next_v - self.V[c_s_idx])

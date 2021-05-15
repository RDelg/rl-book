import numpy as np
from typing import Callable, List, Optional

from tqdm import trange

from chapter5.env import Enviroment
from chapter5.types import State
from chapter5.controller import DiscreteController


class SARSAController(DiscreteController):
    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(SARSAController, self).__init__(env)
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.posible_actions = self.env.act_space.to_list()
        shape = [dim.n for dim in self.env.obs_space] + [self._n_actions]
        self.Q = np.zeros(shape=shape, dtype=np.float32)

    def predict(
        self,
        policy: Callable[[DiscreteController, np.ndarray], int],
        alpha: float = 0.01,
        n_episodes: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
        max_iters: Optional[int] = None,
    ):
        done_history = []
        dones = 0
        total_iters = 0
        for _ in trange(
            n_episodes,
            desc=f"Value prediction iter",
            disable=disable_tqdm,
        ):
            if init_state is None:
                self.env.reset()
            else:
                self.env.state = init_state
            done = False

            current_state = self.env.state
            action = policy(self, current_state)
            while not done and (max_iters is None or total_iters < max_iters):
                done, new_state, reward = self.env.step(action)
                dones += int(done)
                total_iters += 1
                done_history.append(dones)
                if not done:
                    next_action = policy(self, new_state)
                else:
                    next_action = None
                self._update_Q(
                    alpha,
                    current_state,
                    action,
                    new_state,
                    next_action,
                    done,
                    reward,
                )
                current_state = new_state
                action = next_action
        return done_history

    def _update_Q(
        self,
        alpha: float,
        current_state: State,
        action: int,
        new_state: State,
        next_action,
        done: bool,
        reward: float,
    ) -> None:
        c_sa_idx = self.state_action_to_idx(current_state, action)
        n_sa_idx = self.state_action_to_idx(new_state, next_action)
        next_q = 0 if done else self.Q[n_sa_idx]
        self.Q[c_sa_idx] += alpha * (reward + self.gamma * next_q - self.Q[c_sa_idx])


def e_greedy_policy(epsilon: float) -> Callable[[DiscreteController, np.ndarray], int]:
    def policy(controller: DiscreteController, state: State) -> int:
        if np.random.uniform() < epsilon:
            action = controller.env.act_space.sample()
        else:
            state_idx = controller.state_to_idx(state)
            action = int(controller.Q[state_idx].argmax(-1))
        return action

    return policy

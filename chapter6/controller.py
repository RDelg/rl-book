from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import numpy.ma as ma
from tqdm import trange

from chapter5.env import Enviroment
from chapter5.types import State, StateIndex
from chapter5.controller import DiscreteController


class Policy(metaclass=ABCMeta):
    def __init__(self, controller: DiscreteController, *args: Any, **kwds: Any):
        pass

    @abstractmethod
    def update(self, *args: Any, **kwds: Any):
        pass

    @abstractmethod
    def V(self, *args: Any, **kwds: Any) -> float:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> int:
        pass


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
        policy: Policy,
        target_policy: Optional[Policy] = None,
        alpha: float = 0.01,
        n_episodes: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
        max_iters: Optional[int] = None,
        expected: Optional[bool] = False,
    ):
        target_policy = target_policy or policy

        history = {"dones_iter": [], "sum_reward": []}
        total_iters = 0

        for _ in trange(
            n_episodes,
            desc="Value prediction iter",
            disable=disable_tqdm,
        ):
            if init_state is None:
                self.env.reset()
            else:
                self.env.state = init_state
            done = False
            reward = 0
            current_state = self.env.state
            sum_reward = 0
            while not done and (max_iters is None or total_iters < max_iters):
                action = policy(current_state)
                done, new_state, reward = self.env.step(action)
                sum_reward += reward
                total_iters += 1
                c_sa_idx = self.state_action_to_idx(current_state, action)
                if done:
                    history["dones_iter"].append(total_iters)
                    history["sum_reward"].append(sum_reward)
                    next_action = None
                    next_q = 0
                elif expected:
                    next_q = policy.V(new_state)
                else:
                    next_action = target_policy(new_state)
                    n_sa_idx = self.state_action_to_idx(new_state, next_action)
                    next_q = self.Q[n_sa_idx]
                self.Q[c_sa_idx] += alpha * (
                    (reward + self.gamma * next_q) - self.Q[c_sa_idx]
                )
                current_state = new_state

        return history


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        controller: DiscreteController,
        epsilon: float,
        freeze: Optional[bool] = False,
    ):
        self.controller = controller
        self.epsilon = epsilon
        self.freeze = freeze
        self.update()

    def update(self):
        if self.freeze:
            self.Q = np.copy(self.controller.Q)
        else:
            self.Q = self.controller.Q

    def _get_masket_Q(
        self, state_idx: StateIndex, legal_actions: Union[List[int], None]
    ) -> np.ndarray:
        Q = self.Q[state_idx]
        if legal_actions is not None:
            mask = np.zeros_like(Q)
            mask[legal_actions] = 0
            Q = ma.masked_array(Q, mask)
        return Q

    def V(self, state: State) -> float:
        state_idx = self.controller.state_to_idx(state)
        legal_actions = self.controller.env.legal_actions(state)
        Q = self._get_masket_Q(state_idx, legal_actions)

        if legal_actions is not None:
            base_prob = self.epsilon * 1.0 / len(legal_actions)
            probs = np.ones(self.controller.env.act_space.n) * base_prob
        else:
            base_prob = self.epsilon * 1.0 / self.controller.env.act_space.n
            probs = np.ones(self.controller.env.act_space.n) * base_prob

        greedy_action = Q.argmax(-1)

        probs[greedy_action] += 1.0 - self.epsilon
        return (Q * probs).sum(-1)

    def __call__(self, state: State) -> int:
        legal_actions = self.controller.env.legal_actions(state)
        if np.random.uniform() < self.epsilon:
            if legal_actions is not None:
                action = np.random.choice(legal_actions)
            else:
                action = self.controller.env.act_space.sample()
        else:
            state_idx = self.controller.state_to_idx(state)
            action = self._get_masket_Q(state_idx, legal_actions).argmax(-1)
        return action


class GreedyPolicy(Policy):
    def __init__(self, controller: DiscreteController, freeze: Optional[bool] = False):
        self.controller = controller
        self.freeze = freeze
        self.update()

    def update(self):
        if self.freeze:
            self.Q = np.copy(self.controller.Q)
        else:
            self.Q = self.controller.Q

    def _get_masket_Q(
        self, state_idx: StateIndex, legal_actions: Union[List[int], None]
    ) -> np.ndarray:
        Q = self.Q[state_idx]
        if legal_actions is not None:
            mask = np.zeros_like(Q)
            mask[legal_actions] = 0
            Q = ma.masked_array(Q, mask)
        return Q

    def V(self, state: State) -> float:
        state_action_idx = self.controller.state_action_to_idx(
            state, self.greedy_action(state)
        )
        return self.Q[state_action_idx]

    def __call__(self, state: State) -> int:
        state_idx = self.controller.state_to_idx(state)
        legal_actions = self.controller.env.legal_actions(state)
        return int(self._get_masket_Q(state_idx, legal_actions).argmax(-1))

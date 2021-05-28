from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import numpy.ma as ma
from tqdm import trange

from chapter5.env import Enviroment
from chapter5.types import State
from chapter5.controller import DiscreteController


class Policy(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any):
        self.controller = None
        self.freeze = None

    def _get_masked_Q(
        self, state: State, legal_actions: Union[List[int], None]
    ) -> np.ndarray:
        Q = self.Q[state]
        if legal_actions is not None:
            mask = np.ones_like(Q)
            mask[legal_actions] = 0
            Q = ma.masked_array(Q, mask)
        return Q

    def update(self, Q: np.ndarray):
        self.Q = Q

    @abstractmethod
    def V(self, *args: Any, **kwargs: Any) -> float:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> int:
        pass


class SARSAController(DiscreteController):
    def __init__(
        self, env: Enviroment, gamma: float = 1.0, double: Optional[bool] = False
    ):
        super(SARSAController, self).__init__(env)
        self.gamma = gamma
        self.double_q = double
        self.reset()

    def reset(self):
        shape = [dim.n for dim in self.env.obs_space] + [self._n_actions]
        self.Q = np.zeros(shape=shape, dtype=np.float32)
        if self.double_q:
            self.Q_2 = np.zeros_like(self.Q)

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

        history = {"dones_iter": [], "sum_reward": [], "actions_per_episode": []}
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
            actions = []
            while not done and (max_iters is None or total_iters < max_iters):

                if self.double_q:
                    if np.random.rand() < 0.5:
                        Q = self.Q
                        target_policy.update(self.Q_2)
                    else:
                        Q = self.Q_2
                        target_policy.update(self.Q)
                else:
                    Q = self.Q
                action = policy(current_state)
                actions.append(action)
                done, new_state, reward = self.env.step(action)
                sum_reward += reward
                total_iters += 1
                c_sa_idx = current_state + (action,)
                if done:
                    history["actions_per_episode"].append(actions)
                    history["dones_iter"].append(total_iters)
                    history["sum_reward"].append(sum_reward)
                    next_action = None
                    next_q = 0
                elif expected:
                    next_q = policy.V(new_state)
                else:
                    next_action = target_policy(new_state)
                    n_sa_idx = new_state + (next_action,)
                    next_q = Q[n_sa_idx]
                Q[c_sa_idx] += alpha * ((reward + self.gamma * next_q) - Q[c_sa_idx])
                current_state = new_state
        return history


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        controller: DiscreteController,
        epsilon: float,
    ):
        self.controller = controller
        self.epsilon = epsilon
        self.update(self.controller.Q)

    def V(self, state: State) -> float:
        legal_actions = self.controller.env.legal_actions(state)
        Q = self._get_masked_Q(state, legal_actions)

        if legal_actions is not None:
            base_prob = self.epsilon / len(legal_actions)
        else:
            base_prob = self.epsilon / self.controller.env.act_space.n

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
            Q = self._get_masked_Q(state, legal_actions)
            action = np.random.choice(np.where(Q == Q.max())[0])
        return action


class GreedyPolicy(Policy):
    def __init__(self, controller: DiscreteController):
        self.controller = controller
        self.update(self.controller.Q)

    def V(self, state: State) -> float:
        return self.Q[state + (self(state),)]

    def __call__(self, state: State) -> int:
        legal_actions = self.controller.env.legal_actions(state)
        Q = self._get_masked_Q(state, legal_actions)
        return int(np.random.choice(np.where(Q == Q.max())[0]))

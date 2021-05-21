from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any, Optional, Tuple

from tqdm import trange

from chapter5.env import Enviroment
from chapter5.types import State, StateIndex, Trajectory
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
    def greedy_action(self, *args: Any, **kwds: Any) -> int:
        pass

    @abstractmethod
    def prob(self, *args: Any, **kwds: Any) -> float:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
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
                action, _ = policy(current_state)
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
                    next_action, _ = target_policy(new_state)
                    n_sa_idx = self.state_action_to_idx(new_state, next_action)
                    next_q = self.Q[n_sa_idx]
                self.Q[c_sa_idx] += alpha * (
                    (reward + self.gamma * next_q) - self.Q[c_sa_idx]
                )
                current_state = new_state

        return history


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self, controller: DiscreteController, epsilon: float, freeze: bool = False
    ):
        self.controller = controller
        self.epsilon = epsilon
        self.freeze = freeze
        self._base_prob = self.epsilon * 1.0 / self.controller.env.act_space.n
        self.update()

    def update(self):
        if self.freeze:
            self.Q = np.copy(self.controller.Q)
        else:
            self.Q = self.controller.Q

    def greedy_action(self, state: State) -> int:
        state_idx = self.controller.state_to_idx(state)
        return self._get_greedy_action(state=state_idx)

    def _get_greedy_action(self, state: StateIndex) -> int:
        return int(self.Q[state].argmax(-1))

    def V(self, state: State) -> float:
        state_idx = self.controller.state_to_idx(state)
        greedy = self._get_greedy_action(state_idx)
        probs = np.ones(self.controller.env.act_space.n) * self._base_prob
        probs[greedy] += 1.0 - self.epsilon
        return (self.Q[state_idx] * probs).sum(-1)

    def prob(self, state: State, action: int) -> float:
        greedy_action = self.greedy_action(state)
        prob = self._base_prob
        if action == greedy_action:
            prob += 1.0 - self.epsilon
        return prob

    def __call__(self, state: State) -> Tuple[int, float]:
        greedy_action = self.greedy_action(state)
        prob = self._base_prob
        if np.random.uniform() < self.epsilon:
            action = self.controller.env.act_space.sample()
        else:
            action = greedy_action

        if action == greedy_action:
            prob += 1.0 - self.epsilon
            action = greedy_action

        return action, prob


class GreedyPolicy(Policy):
    def __init__(self, controller: DiscreteController, freeze: bool = False):
        self.controller = controller
        self.freeze = freeze
        self.update()

    def update(self):
        if self.freeze:
            self.Q = np.copy(self.controller.Q)
        else:
            self.Q = self.controller.Q

    def greedy_action(self, state: State) -> int:
        state_idx = self.controller.state_to_idx(state)
        return int(self.Q[state_idx].argmax(-1))

    def V(self, state: State) -> float:
        state_idx = self.controller.state_action_to_idx(
            state, self.greedy_action(state)
        )
        return self.Q[state_idx]

    def prob(self, state: State, action: int) -> float:
        return float(action == self.greedy_action(state))

    def __call__(self, state: State) -> Tuple[int, float]:
        return self.greedy_action(state), 1.0

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .env import KArmedEnviroment


@dataclass
class Observation:
    """
    Observation class for the KBandit enviroment.
    """

    action: int
    reward: np.ndarray


class BaseLearner(metaclass=ABCMeta):
    """
    Base learner class.
    """

    def __init__(self, env: KArmedEnviroment):
        self.env = env
        self.reset()

    def reset(self):
        self.Q = np.zeros(self.env.k)
        self.env.reset()

    @abstractmethod
    def play_one(self, policy: Callable[[np.ndarray], int]) -> Observation:
        return


class AvgLearner(BaseLearner):
    """
    Learner that uses the mean of the rewards as the estimate of the reward.
    """

    def __init__(self, env: KArmedEnviroment, **kwargs):
        super().__init__(env)

    def play_one(self, policy: Callable[[np.ndarray], int]) -> Observation:
        action = policy(self.Q)
        reward = self.env.step(action)
        self.actions_count[action] += 1

        self.Q[action] += 1 / self.actions_count[action] * (reward - self.Q[action])
        self.steps += 1
        return Observation(action=action, reward=reward)

    def reset(self):
        self.actions_count = np.zeros(self.env.k)
        self.steps = 0
        super().reset()


class UCBLearner(BaseLearner):
    """
    Upper Confidence Bound (UCB) algorithm.

    Parameters
    ----------
    c : float, default=0.0
        Uncertainty variable used in UCB algorithm.

    initial_Q : float, default=0.0
        Initial value for the Q-table.
    """

    def __init__(
        self, env: KArmedEnviroment, initial_Q: float = 0.0, c: float = 0.0, **kwargs
    ):
        self.initial_Q = initial_Q
        self.c = c
        super().__init__(env)

    def play_one(self, policy: Callable[[np.ndarray], int]) -> Observation:
        action = policy(
            self.Q
            + self.c
            * np.sqrt(np.divide(np.log(self.steps + 1), self.actions_count + 1e-5))
        )
        reward = self.env.step(action)
        self.actions_count[action] += 1
        self.Q[action] += 1 / self.actions_count[action] * (reward - self.Q[action])
        self.steps += 1
        return Observation(action=action, reward=reward)

    def reset(self):
        self.actions_count = np.zeros(self.env.k)
        self.steps = 0
        super().reset()
        self.Q += self.initial_Q


class GradientLearner(BaseLearner):
    """
    Gradient base learner
    """

    def __init__(
        self, env: KArmedEnviroment, delta: float, gradient_base: bool = True, **kwargs
    ):
        self.delta = delta
        self.gradient_base = gradient_base
        super().__init__(env)

    def play_one(
        self, policy: Callable[[np.ndarray], Tuple[int, np.ndarray]]
    ) -> Observation:
        action, self.probs = policy(self.Q)
        reward = self.env.step(action)
        self.steps += 1

        self.one_hot[:] = 0.0
        self.one_hot[action] = 1.0
        if self.gradient_base:
            self.baseline += (reward - self.baseline) / (self.steps)
        self.Q += self.delta * (reward - self.baseline) * (self.one_hot - self.probs)
        return Observation(action=action, reward=reward)

    def reset(self):
        self.one_hot = np.zeros(self.env.k)
        self.probs = np.ones(self.env.k) / self.env.k
        self.steps = 0
        self.baseline = 0.0
        super().reset()


def random_policy(Q: np.ndarray) -> int:
    """
    Random policy.
    """
    return np.random.randint(0, Q.size)


def e_greedy_policy(epsilon: float) -> Callable[[np.ndarray], int]:
    """
    Epsilon-greedy policy.
    """
    eps = epsilon

    def policy(Q: np.ndarray) -> int:
        if np.random.random() < eps:
            return np.random.randint(0, Q.size)
        else:
            return np.argmax(Q)

    return policy


def greedy_policy(Q: np.ndarray) -> int:
    """
    Greedy policy.
    """
    return np.argmax(Q)


def gradient_policy(Q: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Gradient policy.
    """

    exp_est = np.exp(Q)
    action_prob = exp_est / np.sum(exp_est)
    action = (np.cumsum(action_prob) >= np.random.uniform()).argmax(axis=0)
    return action, action_prob

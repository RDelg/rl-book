from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union, List

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


class MeanLearner(BaseLearner):
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


# class KBanditLearner:
#     """
#     K-arms bandit learner.

#     Parameters
#     ----------
#     env : Enviroment
#         Instanciated enviroment that the learner will use.

#     k : int
#         Number of "arms" in the bandit enviroment.

#     eps : float, default=0.01
#         Epsilon variable in eps-greedy algorithm.

#     delta : float, default=0.01
#         Step size to use in grandient base updates and constant value updates.

#     initial : float, list of floats, default=0.01
#         Constant or list of contants where the stimated reward starts at time 0.

#     c : float, default=0.0
#         Uncertainty variable used in UCB algorithm.

#     eps : float, default=0.01
#         Epsilon variable in eps-greedy algorithm.

#     gradient : bool, default=False
#         If True, use gradient based bandit algorithm

#     gradient_base : bool, default=True
#         If True, use average reward as baseline for gradient based bandit algorithm.

#     """

#     def __init__(
#         self,
#         env: Enviroment,
#         k: int,
#         eps: float = 0.01,
#         delta: float = None,
#         initial: Union[float, List[float]] = 0.0,
#         c: float = 0.0,
#         gradient: bool = False,
#         gradient_base: bool = True,
#     ):
#         assert issubclass(type(env), Enviroment), "env must be an Enviroment subclass"
#         self.env = env
#         self.k = k
#         self.eps = eps
#         self.delta = delta
#         self.estimated_reward = np.zeros(self.k) + initial
#         self.actions_count = np.zeros(self.k)
#         self.c = c
#         self.gradient = gradient
#         self.gradient_base = gradient_base

#     def play(self, step_size):
#         # Inicialize
#         self.actions_taken = np.zeros(step_size)
#         self.obtained_reward = np.zeros(step_size)
#         baseline = 0
#         one_hot = np.zeros(self.k)
#         # Run env
#         reward = self.env.reward(step_size)
#         # Play
#         random_choise = np.random.choice(a=np.arange(self.k), size=(step_size))
#         is_greedy = np.random.uniform(size=(step_size)) > self.eps

#         for i in range(step_size):
#             # Taking the action
#             if self.gradient:
#                 exp_est = np.exp(self.estimated_reward)
#                 self.action_prob = exp_est / np.sum(exp_est)
#                 action = (np.cumsum(self.action_prob) >= np.random.uniform()).argmax(
#                     axis=0
#                 )
#             elif self.c > 0:
#                 action = (
#                     np.argmax(
#                         self.estimated_reward
#                         + self.c
#                         * np.sqrt(np.divide(np.log(i + 1), self.actions_count + 1e-5))
#                     )
#                     if is_greedy[i]
#                     else random_choise[i]
#                 )
#             else:
#                 action = (
#                     np.argmax(self.estimated_reward)
#                     if is_greedy[i]
#                     else random_choise[i]
#                 )
#             self.actions_count[action] += 1
#             self.actions_taken[i] = action

#             # Update estimation
#             if self.gradient:
#                 one_hot[:] = 0
#                 one_hot[action] = 1
#                 if self.gradient_base:
#                     baseline += (reward[i, action] - baseline) / (i + 1)
#                 else:
#                     baseline = 0
#                 self.estimated_reward += (
#                     self.delta
#                     * (reward[i, action] - baseline)
#                     * (one_hot - self.action_prob)
#                 )
#             elif self.delta is None:
#                 self.estimated_reward[action] += (
#                     1
#                     / self.actions_count[action]
#                     * (reward[i, action] - self.estimated_reward[action])
#                 )
#             else:
#                 self.estimated_reward[action] += self.delta * (
#                     reward[i, action] - self.estimated_reward[action]
#                 )

#             # Saving rewards
#             self.obtained_reward[i] = reward[i, action]

#     def reset(self):
#         self.env.reset()
#         self.estimated_reward = np.zeros(self.k)
#         self.actions_count = np.zeros(self.k)

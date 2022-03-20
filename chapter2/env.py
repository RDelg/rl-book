from abc import ABCMeta, abstractmethod

import numpy as np
from typing import Union, List

from pkg_resources import Environment


class Enviroment(metaclass=ABCMeta):
    @abstractmethod
    def step(self, action: int) -> float:
        raise NotImplementedError


class KArmedEnviroment(Environment):
    def __init__(self, k: int):
        self.k = k
        self.reset()

    def reset(self):
        pass

    def step(self, action: int) -> float:
        raise NotImplementedError


class NormalKBandit(KArmedEnviroment):
    """
    K-arms bandit enviroment where the reward_k for the k-th arm is a normal distribution with mean=mean_k and std=1.

    Parameters
    ----------
    k : int
        Number of "arms" in the bandit enviroment.

    offset : float, list of floats, default=0.0
        Offset value(s) to sum to arm rewards.

    stationary : bool, default=True
        If not True, then the reward "moves" each step using a
        normal distribution with mean 'walk_size' and std of 1.

    walk_size : float, default=0.1
        Mean value of normal distribution to use to calculate
        the reward walk.

    """

    def __init__(
        self,
        k: int,
        offset: Union[float, List[float]] = 0.0,
        stationary: bool = True,
        walk_size: float = 0.1,
    ):

        self.offset = offset
        self.stationary = stationary
        self.walk_size = walk_size
        super().__init__(k)

    def reset(self):
        self.mean = np.random.randn((self.k)) + self.offset
        self.best_action = np.argmax(self.mean)

    def _walk(self):
        self.mean += np.random.normal(scale=self.walk_size, size=(self.k))
        self.best_action = np.argmax(self.mean)

    def step(self, action: int) -> float:
        if not self.stationary:
            self._walk()
        return np.random.normal(loc=self.mean[action], scale=1.0)

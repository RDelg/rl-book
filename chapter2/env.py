from abc import ABCMeta, abstractmethod

import numpy as np
from typing import Union, List


class Enviroment(metaclass=ABCMeta):
    @abstractmethod
    def reward(self, *args, **kwargs):
        raise NotImplementedError


class KBandit(Enviroment):
    """
    K-arms bandit enviroment.

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
        self.k = k
        self.offset = offset
        self.stationary = stationary
        self.walk_size = walk_size
        self.init_or_reset()

    def init_or_reset(self):
        self.mean = np.random.randn((self.k)) + self.offset
        self.best_action = np.argmax(self.mean)

    def reset(self):
        self.init_or_reset()

    def reward(self, n_steps):
        reward = np.random.normal(self.mean, 1, (n_steps, self.k))
        if not self.stationary:  # walk
            reward += np.cumsum(
                np.random.normal(scale=self.walk_size, size=(n_steps, self.k)), axis=0
            )
        return reward

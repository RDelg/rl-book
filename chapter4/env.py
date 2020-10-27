from typing import Tuple, Type, List
from functools import lru_cache
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import poisson


class Space:
    def __init__(self, dims: List[int], min: float, max: float, dtype: Type[int]):
        self._dims = dims
        self._min = dtype(min)
        self._max = dtype(max)

    @property
    def dims(self):
        return self._dims

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max


class Enviroment(metaclass=ABCMeta):
    @abstractmethod
    def dynamics(self, *args, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def legal_actions(self, *args, **kwargs) -> np.array:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def obs_space() -> Space:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def act_space() -> Space:
        raise NotImplementedError


class RentalCarEnv(Enviroment):
    # How many values of the Poisson distributtion use to calculate the dynamics
    _poisson_range = 11
    # Rental car lambdas
    _lam_in_a = 4
    _lam_out_a = 2
    _lam_in_b = 3
    _lam_out_b = 3
    # Reward values
    _move_reward = -2
    _rental_reward = 10

    def __init__(self, value: np.array, gamma=0.9, fixed_return=True):
        self.gamma = gamma
        self.fixed_return = fixed_return
        self.value = value
        # PMFs
        self._return_a = poisson.pmf(range(self._poisson_range), self._lam_in_a)
        self._request_a = poisson.pmf(range(self._poisson_range), self._lam_out_a)
        self._return_b = poisson.pmf(range(self._poisson_range), self._lam_in_b)
        self._request_b = poisson.pmf(range(self._poisson_range), self._lam_out_b)
        # Instantiate spaces to reduce allocations
        self._obs_space = self.obs_space()
        self._act_space = self.act_space()

    @staticmethod
    def obs_space() -> Space:
        return Space([2], 0, 20, int)

    @staticmethod
    def act_space() -> Space:
        return Space([1], -5, 5, int)

    @staticmethod
    @lru_cache(maxsize=None)
    def legal_actions(state: Tuple[int, int]) -> np.array:
        return np.arange(
            np.max([-state[1], RentalCarEnv.act_space().min]),
            np.min([state[0], RentalCarEnv.act_space().max]) + 1,
            1,
            dtype=np.int32,
        )

    def dynamics(self, state: Tuple[int, int], action: int) -> float:
        value = 0.0
        reward_base = np.abs(action) * np.float32(self._move_reward)
        state = list(state)
        state[0] -= action
        state[1] += action

        for request_a in range(self._poisson_range):
            request_a_prob = self._request_a[request_a]
            for request_b in range(self._poisson_range):
                request_b_prob = self._request_b[request_b]

                real_request_a = np.min([request_a, state[0]])
                real_request_b = np.min([request_b, state[1]])

                reward = reward_base + np.float32(self._rental_reward) * (
                    real_request_a + real_request_b
                )

                prob = request_a_prob * request_b_prob

                if self.fixed_return:
                    return_a = self._lam_in_a
                    return_b = self._lam_in_b
                    new_state = (
                        np.min(
                            [state[0] - real_request_a + return_a, self._obs_space.max]
                        ),
                        np.min(
                            [state[1] - real_request_b + return_b, self._obs_space.max]
                        ),
                    )
                    value += prob * (
                        reward + self.gamma * self.value[new_state[0], new_state[1]]
                    )
                else:
                    for return_a in range(self._poisson_range):
                        return_a_prob = self._return_a[return_a]
                        for return_b in range(self._poisson_range):
                            return_b_prob = self._return_b[return_b]
                            _prob = prob * return_a_prob * return_b_prob
                            new_state = (
                                np.min(
                                    [
                                        state[0] - real_request_a + return_a,
                                        self._obs_space.max,
                                    ]
                                ),
                                np.min(
                                    [
                                        state[1] - real_request_b + return_b,
                                        self._obs_space.max,
                                    ]
                                ),
                            )
                            value += _prob * (
                                reward
                                + self.gamma * self.value[new_state[0], new_state[1]]
                            )
        return value

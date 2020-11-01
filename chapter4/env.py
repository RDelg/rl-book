from typing import Tuple, Type, List
from functools import lru_cache
from abc import ABCMeta, abstractmethod

import numpy as np
from numba import jit
from scipy.stats import poisson


class Space:
    """Bounded space to represent the action and
    observation spaces of an enviroment.

    Parameters
    ----------

    dims : list of ints
        Dimentions that the space have.

    min : float
        minimum value that the space could have.

    max : float
        maximum value that the space could have.

    dtype: type
        data type of the space."""

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
    """Abstract class that represents an enviroment to use in a dynamic
    programming learner."""

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

    @abstractmethod
    def idx_to_state(self, *args, **kwargs) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def state_to_idx(self, *args, **kwargs) -> Tuple:
        raise NotImplementedError


class RentalCarEnv(Enviroment):
    """Rental car enviroment.

    Parameters
    ----------

    gamma : float, default=0.9
        Gamma value to use when calculating the returns.

    fixed_return : bool, default=True
        If True, then always the returned cars are always equal to the lambda values.
        If False, then the dynamics are calculated on the returns also.

    """

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

    def __init__(self, gamma: float = 0.9, fixed_return: bool = True):
        self.gamma = gamma
        self.fixed_return = fixed_return
        # PMFs
        self._return_a_pmf = poisson.pmf(range(self._poisson_range), self._lam_in_a)
        self._request_a_pmf = poisson.pmf(range(self._poisson_range), self._lam_out_a)
        self._return_b_pmf = poisson.pmf(range(self._poisson_range), self._lam_in_b)
        self._request_b_pmf = poisson.pmf(range(self._poisson_range), self._lam_out_b)
        # Instantiate spaces to reduce allocations
        self._obs_space = self.obs_space()
        self._act_space = self.act_space()

    def idx_to_state(self, idx: Tuple[int, int]) -> Tuple[int, int]:
        return idx

    def state_to_idx(self, state: Tuple[int, int]) -> Tuple[int, int]:
        return state

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

    @staticmethod
    @jit(nopython=True)
    def _dynamics(
        estimated_value: np.array,
        state: Tuple[int, int],
        action: int,
        move_reward: int,
        rental_reward: int,
        poisson_range: int,
        request_a_pmf: np.array,
        request_b_pmf: np.array,
        return_a_pmf: np.array,
        return_b_pmf: np.array,
        lam_in_a: int,
        lam_in_b: int,
        obs_space_max: int,
        fixed_return: bool,
        gamma: float,
    ) -> float:
        value = 0.0
        reward_base = np.abs(action) * move_reward
        state_a = np.int32(state[0] - action)
        state_b = np.int32(state[1] + action)

        for request_a in range(poisson_range):
            request_a_prob = request_a_pmf[request_a]
            for request_b in range(poisson_range):
                request_b_prob = request_b_pmf[request_b]

                real_request_a = np.array([request_a, state_a]).min()
                real_request_b = np.array([request_b, state_b]).min()

                reward = reward_base + np.float32(rental_reward) * (
                    real_request_a + real_request_b
                )

                prob = request_a_prob * request_b_prob

                if fixed_return:
                    return_a = lam_in_a
                    return_b = lam_in_b
                    new_state = (
                        np.array(
                            [state_a - real_request_a + return_a, obs_space_max]
                        ).min(),
                        np.array(
                            [state_b - real_request_b + return_b, obs_space_max]
                        ).min(),
                    )
                    value += prob * (
                        reward + gamma * estimated_value[new_state[0], new_state[1]]
                    )
                else:
                    for return_a in range(poisson_range):
                        return_a_prob = return_a_pmf[return_a]
                        new_state_a = np.array(
                            [
                                state_a - real_request_a + return_a,
                                obs_space_max,
                            ]
                        ).min()
                        for return_b in range(poisson_range):
                            return_b_prob = return_b_pmf[return_b]
                            _prob = prob * return_a_prob * return_b_prob
                            new_state = (
                                new_state_a,
                                np.array(
                                    [
                                        state_b - real_request_b + return_b,
                                        obs_space_max,
                                    ]
                                ).min(),
                            )
                            value += _prob * (
                                reward
                                + gamma * estimated_value[new_state[0], new_state[1]]
                            )
        return value

    def dynamics(
        self, estimated_value: np.array, state: Tuple[int, int], action: int
    ) -> float:
        return self._dynamics(
            estimated_value,
            state,
            action,
            self._move_reward,
            self._rental_reward,
            self._poisson_range,
            self._request_a_pmf,
            self._request_b_pmf,
            self._return_a_pmf,
            self._return_b_pmf,
            self._lam_in_a,
            self._lam_in_b,
            self._obs_space.max,
            self.fixed_return,
            self.gamma,
        )


class GamblerEnv(Enviroment):
    """Gambler enviroment.

    Parameters
    ----------

    gamma : float, default=1.0
        Gamma value to use when calculating the returns.

    """

    _coin_head_prob = 0.4
    # Rewards
    _win_reward = 1.0

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        # Instantiate spaces to reduce allocations
        self._obs_space = self.obs_space()
        self._act_space = self.act_space()

    @staticmethod
    def obs_space() -> Space:
        return Space([1], 1, 99, int)

    @staticmethod
    def act_space() -> Space:
        return Space([1], 1, 99, int)

    def idx_to_state(self, idx: Tuple[int]) -> Tuple[int]:
        return (idx[0] + self._obs_space.min,)

    def state_to_idx(self, state: Tuple[int]) -> Tuple[int]:
        return (state[0] - self._obs_space.min,)

    def dynamics(
        self, estimated_value: np.array, state: Tuple[int], action: int
    ) -> float:
        # Tail case
        new_head_idx = self.state_to_idx((state[0] - action,))
        if new_head_idx[0] < 0:
            ret = 0.0
        else:
            ret = (1.0 - self._coin_head_prob) * (
                self.gamma * estimated_value[new_head_idx]
            )
        # Head case
        new_head_state = (state[0] + action,)
        if new_head_state[0] == 100:
            ret += self._coin_head_prob * self._win_reward
        else:
            ret += self._coin_head_prob * (
                self.gamma * estimated_value[self.state_to_idx(new_head_state)]
            )
        return ret

    @staticmethod
    @lru_cache(maxsize=None)
    def legal_actions(state: Tuple[int]) -> np.array:
        return np.arange(1, min(state[0], 100 - state[0]) + 1)
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
    programming learner.

    Parameters
    ----------

    gamma : float
        Gamma value to use when calculating the returns.

    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        # Instantiate spaces to reduce allocations
        self._obs_space = self.obs_space()
        self._act_space = self.act_space()

    @abstractmethod
    def dynamics(self, *args, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def legal_actions(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def obs_space() -> Space:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def act_space() -> Space:
        raise NotImplementedError

    def idx_to_state(self, idx: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x + self._obs_space.min for x in idx)

    def state_to_idx(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x - self._obs_space.min for x in state)


class RentalCarEnv(Enviroment):
    """Rental car enviroment.

    Parameters
    ----------

    gamma : float, default=0.9
        Gamma value to use when calculating the returns.

    fixed_return : bool, default=True
        If True, then the returned cars are always equal to the lambda values.
        If False, then the dynamics are calculated on the returns pmf.

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
        super().__init__(gamma)
        self.fixed_return = fixed_return
        # PMFs
        self._return_a_pmf = poisson.pmf(range(self._poisson_range), self._lam_in_a)
        self._request_a_pmf = poisson.pmf(range(self._poisson_range), self._lam_out_a)
        self._return_b_pmf = poisson.pmf(range(self._poisson_range), self._lam_in_b)
        self._request_b_pmf = poisson.pmf(range(self._poisson_range), self._lam_out_b)

    @staticmethod
    def obs_space() -> Space:
        return Space([2], 0, 20, int)

    @staticmethod
    def act_space() -> Space:
        return Space([1], -5, 5, int)

    @staticmethod
    @lru_cache(maxsize=None)
    def legal_actions(state: Tuple[int, int]) -> np.ndarray:
        return np.arange(
            np.max([-state[1], RentalCarEnv.act_space().min]),
            np.min([state[0], RentalCarEnv.act_space().max]) + 1,
            1,
            dtype=np.int32,
        )

    @staticmethod
    @jit(nopython=True)
    def _dynamics(
        estimated_value: np.ndarray,
        state: Tuple[int, int],
        action: int,
        move_reward: int,
        rental_reward: int,
        poisson_range: int,
        request_a_pmf: np.ndarray,
        request_b_pmf: np.ndarray,
        return_a_pmf: np.ndarray,
        return_b_pmf: np.ndarray,
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
        self, estimated_value: np.ndarray, state: Tuple[int, int], action: int
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
        super().__init__(gamma)

    @staticmethod
    def obs_space() -> Space:
        return Space([1], 1, 99, int)

    @staticmethod
    def act_space() -> Space:
        return Space([1], 1, 99, int)

    def dynamics(
        self, estimated_value: np.ndarray, state: Tuple[int], action: int
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
    def legal_actions(state: Tuple[int]) -> np.ndarray:
        return np.arange(1, min(state[0], 100 - state[0]) + 1)


class GridEnv(Enviroment):
    """Grid 2D enviroment.

    Parameters
    ----------

    gamma : float, default=1.0
        Gamma value to use when calculating the returns.

    """

    _shape = (4, 4)
    _actions = [
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([0, -1]),
        np.array([0, 1]),
    ]
    _terminal_states = [(0, 0), (3, 3)]
    # Rewards
    _step_reward = -1.0

    def __init__(self, gamma: float = 1.0):
        super().__init__(gamma)

    @staticmethod
    def obs_space() -> Space:
        return Space([2], 0, 3, int)

    @staticmethod
    def act_space() -> Space:
        return Space([1], 0, 3, int)

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        for terminal_state in self._terminal_states:
            if terminal_state == state:
                return True
        return False

    def step(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        new_state = tuple(np.array(state) + self._actions[action])
        if (
            new_state[0] < 0
            or new_state[0] >= self._shape[0]
            or new_state[1] < 0
            or new_state[1] >= self._shape[1]
        ):
            new_state = state

        return new_state

    def dynamics(
        self, estimated_value: np.ndarray, state: Tuple[int, int], action: int
    ) -> float:
        if self.is_terminal(state):
            return 0

        new_state = self.step(state, action)

        ret = (
            self._step_reward
            + self.gamma * estimated_value[self.state_to_idx(new_state)]
        )

        return ret

    @staticmethod
    @lru_cache(maxsize=None)
    def legal_actions(state: Tuple[int, int]) -> np.ndarray:
        return np.arange(GridEnv.act_space().max + 1, dtype=np.int32)

from abc import ABCMeta, abstractmethod
from typing import List, Union, Tuple
from dataclasses import dataclass

import numpy as np

from .types import Observation, State


class Discrete:
    def __init__(self, n: int, minimum: int = 0):
        self.n = n
        self.minimum = minimum

    def contains(self, x: Union[int, np.ndarray]) -> bool:
        if isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            x = int(x)
        if not isinstance(x, int):
            return False
        else:
            return x - self.minimum <= self.n


class Space:
    """Class represent the action and observation spaces
    of an enviroment.

    Parameters
    ----------

    dims : int
        Dimentions that the space has.

    min : int, list of ints
        minimum value that the space could have.

    max : int, list of ints
        maximum value that the space could have.
    """

    def __init__(
        self, dims: int, min: Union[int, List[int]], max: Union[int, List[int]]
    ):
        if isinstance(min, list) and len(min) != dims:
            raise ValueError(f"min length ({len(min)}) doesn't match dims ({dims})")
        if isinstance(max, list) and len(max) != dims:
            raise ValueError(f"max length ({len(max)}) doesn't match dims ({dims})")
        self.dims = dims
        to_ndarray = lambda x: np.array([x for _ in range(dims)], dtype=np.int32)
        self.min = to_ndarray(min) if isinstance(min, int) else min
        self.max = to_ndarray(max) if isinstance(max, int) else max

    def contains(self, x: List[int]) -> bool:
        if len(x) != self.dims:
            return False
        else:
            for v, _min, _max in zip(x, self.min, self.max):
                if x < _min or x > _max:
                    return False
        return True


class Enviroment(metaclass=ABCMeta):
    def __init__(self):
        self._obs_space = None
        self._act_space = None

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, state: State) -> np.ndarray:
        raise NotImplementedError

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def act_space(self) -> Space:
        return self._act_space


class BlackJack(Enviroment):
    _DEALER_THRESHOLD = 17

    @dataclass
    class Card:
        letter: str
        value: int

    _CARDS = [
        Card("A", 1),
        Card("2", 2),
        Card("3", 3),
        Card("4", 4),
        Card("5", 5),
        Card("6", 6),
        Card("7", 7),
        Card("8", 8),
        Card("9", 9),
        Card("10", 10),
        Card("J", 10),
        Card("Q", 10),
        Card("K", 10),
    ]

    def __init__(self):
        self._act_spac = Space(dims=1, min=0, max=1)
        self._obs_spac = Space(dims=3, min=[12, 0, 1], max=[21, 1, 10])
        self.reset()

    def reset(self):
        # Dealer
        self.dealer_card = self._get_card()
        self._dealer_usable_ace = self.dealer_card.letter == "A"
        # Player
        self.player_sum = 0
        self.usable_ace = False
        self.player_sum, self.usable_ace = self._hit(self.player_sum, self.usable_ace)

    def _get_card(self) -> "BlackJack.Card":
        # pylint: disable=invalid-sequence-index
        return self._CARDS[np.random.randint(0, 12)]

    @property
    def state(self) -> State:
        return (
            self.player_sum,
            int(self.usable_ace),
            self.dealer_card.value,
        )

    def legal_actions(self, _: State) -> np.ndarray:
        return self._ACTIONS

    @state.setter
    def state(self, state: Tuple[int, int, int]):
        self.player_sum = state[0]
        self.usable_ace = bool(state[1])
        self.dealer_card = [c for c in self._CARDS if c.value == state[2]][0]
        self._dealer_usable_ace = self.dealer_card.letter == "A"

    def _hit(self, current_sum: int, usable_ace: bool) -> Tuple[int, bool]:
        card = self._get_card()
        if card.letter == "A" and not usable_ace and current_sum < 11:
            current_sum += 11
            usable_ace = True
        else:
            current_sum += card.value
            if current_sum > 21 and usable_ace:
                current_sum -= 10
                usable_ace = False
        if current_sum < 12:
            return self._hit(current_sum, usable_ace)
        else:
            return current_sum, usable_ace

    def _stick(self) -> int:
        dealer_usable_ace = self._dealer_usable_ace
        dealer_sum = 11 if dealer_usable_ace else self.dealer_card.value
        while dealer_sum < self._DEALER_THRESHOLD:
            dealer_sum, dealer_usable_ace = self._hit(dealer_sum, dealer_usable_ace)
        return dealer_sum

    def step(self, action: int) -> Observation:
        assert (
            isinstance(action, int)
            or isinstance(action, (np.generic, np.ndarray))
            and (action.dtype.char in np.typecodes["AllInteger"] and action.shape == ())
        )
        assert 0 <= action <= 1
        if action:  # Hit
            self.player_sum, self.usable_ace = self._hit(
                self.player_sum, self.usable_ace
            )
            if self.player_sum > 21:
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # Stick
            dealer_sum = self._stick()
            done = True
            if dealer_sum > 21 or dealer_sum < self.player_sum:
                reward = 1
            else:
                reward = -1
        return Observation(done, self.state, reward)


class SingleState(metaclass=ABCMeta):
    _RIGHT_ACTION = 1
    _LEFT_ACTION = 0

    def __init__(self):
        self._obs_spac = Space(dims=1, min=0, max=0)
        self._act_spac = Space(dims=1, min=0, max=1)
        self.reset()

    def reset(self):
        self.state = (0,)

    def step(self, action: int) -> Observation:
        if action == self._RIGHT_ACTION:
            done = True
            reward = 0
        elif action == self._LEFT_ACTION:
            if np.random.uniform() < 0.1:
                done = True
                reward = 1
            else:
                done = False
                reward = 0

        return Observation(done, self.state, reward)

    def legal_actions(self, state: State) -> np.ndarray:
        return self._ACTIONS

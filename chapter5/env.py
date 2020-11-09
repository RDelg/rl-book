from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import lru_cache
from typing import List, Union, Tuple
from dataclasses import dataclass


import numpy as np


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
        self.min = min
        self.max = max


class Enviroment(metaclass=ABCMeta):
    def __init__(self):
        # Instantiate spaces to reduce allocations
        self._obs_space = self.obs_space()
        self._act_space = self.act_space()

    @abstractmethod
    def step(self, action: int) -> Tuple[bool, float, Tuple[int, ...]]:
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, state: Tuple[int, ...]) -> np.ndarray:
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
        if isinstance(self._obs_space.min, list):
            return tuple(x + y for x, y in zip(idx, self._obs_space.min))
        else:
            return tuple(x + self._obs_space.min for x in idx)

    def state_to_idx(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(self._obs_space.min, list):
            return tuple(x - y for x, y in zip(state, self._obs_space.min))
        else:
            return tuple(x - self._obs_space.min for x in state)


class BlackJack(Enviroment):
    @dataclass
    class Card:
        letter: str
        value: int

    _CARDS = [
        Card("A", 1),
        Card("1", 1),
        Card("2", 2),
        Card("3", 3),
        Card("4", 4),
        Card("5", 5),
        Card("6", 6),
        Card("7", 7),
        Card("8", 8),
        Card("9", 9),
        Card("J", 10),
        Card("Q", 10),
        Card("K", 10),
    ]

    _ACTIONS = np.array([0, 1], dtype=np.int32)

    def __init__(self):
        self.reset()

    def reset(self):
        # Dealer
        self.dealer_card = self._get_card()
        self._dealer_usable_ace = self.dealer_card.letter == "A"
        # Player
        self.player_sum = 0
        self.usable_ace = False
        self.player_sum, self.usable_ace = self._hit(self.player_sum, self.usable_ace)

    def _get_card(self):
        # pylint: disable=invalid-sequence-index
        return self._CARDS[np.random.randint(0, 13)]

    @staticmethod
    def act_space() -> Space:
        return Space(dims=1, min=0, max=1)

    @staticmethod
    def obs_space() -> Space:
        return Space(dims=3, min=[12, 0, 1], max=[21, 1, 11])

    @property
    def state(self):
        return (
            self.player_sum,
            self.usable_ace,
            11 if self._dealer_usable_ace else self.dealer_card.value,
        )

    def legal_actions(self, state: Tuple[int, int, int]) -> np.ndarray:
        return self._ACTIONS

    @state.setter
    def state(self, state: Tuple[int, int, int]):
        self.player_sum = state[0]
        self.usable_ace = state[1]
        self.dealer_card = self._CARDS[0] if state[2] == 11 else self._CARDS[state[2]]
        self._dealer_usable_ace = self.dealer_card.letter == "A"

    def _hit(self, current_sum: int, usable_ace: bool):
        card = self._get_card()
        if card == "_A" and current_sum < 11:
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

    def _stick(self):
        dealer_sum = 11 if self._dealer_usable_ace else self.dealer_card.value
        while dealer_sum < 17:
            dealer_sum, self._dealer_usable_ace = self._hit(
                dealer_sum, self._dealer_usable_ace
            )
        return dealer_sum

    def step(self, action: int) -> Tuple[bool, float, Tuple[int, int, int]]:
        if action == 0:  # Sticks
            dealer_sum = self._stick()
            if dealer_sum > 21:  # Dealer bust
                return True, 1, self.state
            elif dealer_sum == self.player_sum:
                return True, 0, self.state
            elif dealer_sum < self.player_sum:
                return True, 1, self.state
            else:
                return True, -1, self.state
        elif action == 1:  # Hits
            self.player_sum, self.usable_ace = self._hit(
                self.player_sum, self.usable_ace
            )
            if self.player_sum > 21:  # Bust
                return True, -1, self.state
            else:
                return False, 0, self.state
        else:
            raise Exception(f"Invalid action: {action}")

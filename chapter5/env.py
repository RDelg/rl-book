from abc import ABCMeta, abstractmethod
from typing import List, Union, Tuple, Iterator, Iterable
from dataclasses import dataclass

import numpy as np

from .types import Observation, State


class DiscreteDim:
    def __init__(self, n: int, minimum: int = 0):
        assert n > 1, "n must be greater than 1"
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

    def to_list(self) -> List[str]:
        return list(range(self.minimum, self.minimum + self.n))

    def sample(self) -> int:
        return np.random.randint(self.minimum, self.minimum + self.n)


class DiscreteSpace:
    def __init__(self, *dims: DiscreteDim):
        self.n = len(dims)
        self.dims = dims

    def contains(self, x: Iterable[int]) -> bool:
        if len(x) != self.n:
            return False
        else:
            for v, dim in zip(x, self.dims):
                if not dim.contains(v):
                    return False
        return True

    def __iter__(self) -> Iterator[DiscreteDim]:
        for dim in self.dims:
            yield dim


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

    @property
    def obs_space(self) -> DiscreteSpace:
        return self._obs_space

    @property
    def act_space(self) -> DiscreteSpace:
        return self._act_space

    def legal_actions(self, state: State) -> Union[List[int], None]:
        return None


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
        self._act_space = DiscreteDim(2)
        self._obs_space = DiscreteSpace(
            DiscreteDim(10, minimum=12), DiscreteDim(2), DiscreteDim(10, minimum=1)
        )
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
        assert self.act_space.contains(action), "Invalid action"
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


class SingleState(Enviroment):
    _RIGHT_ACTION = 1
    _LEFT_ACTION = 0

    def __init__(self):
        self._act_space = DiscreteDim(2, minimum=0)
        self._obs_space = DiscreteSpace(DiscreteDim(2))
        self.reset()

    def reset(self):
        self.state = (0,)

    def step(self, action: int) -> Observation:
        assert self.act_space.contains(action), "Invalid action"
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

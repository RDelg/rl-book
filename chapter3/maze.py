from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np


class Enviroment(metaclass=ABCMeta):
    @abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def set_state(self, *args, **kwargs):
        raise NotImplementedError


class Maze(object):
    """
    It represents a 2D grid world enviroment where some cells have rewards
    and other hops to other states.

    Parameters
    ----------
    shape : 2D int tuple
        Shape that the world will have.

    init_state : 2D int tuple, default=(0,0)
        Initial state.

    """

    actions = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]

    def __init__(self, shape: Tuple[int, int], init_state: Tuple[int, int] = (0, 0)):
        assert len(shape) == 2, "shape must be 2D"
        assert len(init_state) == 2, "init_state shape must be 2D"
        assert init_state[0] >= 0 and init_state[1] >= 0, "init_state must be positive"
        assert (
            init_state[0] < shape[0] and init_state[1] < shape[1]
        ), "init_state must be inside shape"
        self.rewards = np.zeros(shape=shape)
        self.hops = []
        self.state = np.array(init_state)
        self.shape = shape

    def add_reward(self, state: np.ndarray, reward: float):
        """Adds a reward to the state"""
        assert len(state) == 2, "state shape must be 2D"
        self.rewards[state[0], state[1]] = reward

    def add_hop(self, state: np.ndarray, new_state: np.ndarray):
        """Adds a hop from state to new_state"""
        assert len(state) == 2, "state shape must be 2D"
        assert len(new_state) == 2, "state shape must be 2D"
        self.hops.append({"old_state": state, "new_state": new_state})

    def set_state(self, state: Tuple[int, int]):
        """Sets the current state"""
        assert len(state) == 2, "state shape must be 2D"
        self.state = np.array(state)

    def step(self, action: np.ndarray):
        """Does a step performing the given action.
        Then return the new state and its reward"""
        assert len(action) == 2, "action shape must be 2D"
        current_state = self.state
        for hop in self.hops:
            if (self.state == hop["old_state"]).all():
                self.state = hop["new_state"]
                return current_state, self.rewards[current_state[0], current_state[1]]
        new_state = self.state + action
        if (
            new_state[0] < 0
            or new_state[0] >= self.shape[0]
            or new_state[1] < 0
            or new_state[1] >= self.shape[1]
        ):
            new_state = current_state
            reward = -1.0
        else:
            reward = self.rewards[current_state[0], current_state[1]]
        self.state = new_state
        return self.state, reward
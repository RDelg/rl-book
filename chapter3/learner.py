import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

from maze import Enviroment


class ValueLearner(object):
    """Value learner for a 2D grid world.

    Parameters
    ----------
    maze : Enviroment
        Maze enviroment to work with.

    discount : float
        Discount factor to calculate the return of the state.

    random : bool, default=True
        If True, it updates the value of each state simulating taking the action
        and then updating the value directly.
        If False, it updates the value performing all actions from the state and
        then taking the max return between all of them.

    """

    def __init__(self, maze: Enviroment, discount: float, random: bool = True):
        self.maze = maze
        self.random = random
        self.value = np.zeros(shape=maze.shape)
        self.discount = discount

    def update_value(self):
        new_value = np.zeros_like(self.value)
        values = np.zeros(shape=len(self.maze.actions))
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                for z, action in enumerate(self.maze.actions):
                    self.maze.set_state(np.array([i, j]))
                    new_state, reward = self.maze.step(action)
                    if self.random:
                        new_value[i, j] += (
                            1.0
                            / len(self.maze.actions)
                            * (
                                reward
                                + self.discount * self.value[new_state[0], new_state[1]]
                            )
                        )
                    else:
                        values[z] = (
                            reward
                            + self.discount * self.value[new_state[0], new_state[1]]
                        )
                if not self.random:
                    new_value[i, j] = np.max(values)
        self.value = new_value

    def plot_value(self, ax: plt.Axes = None):
        if ax is None:
            _, ax = plt.subplots()
        rounded_value = np.round(self.value, decimals=2)
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = rounded_value.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        for (i, j), val in np.ndenumerate(rounded_value):
            tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")

        for i in range(len(rounded_value)):
            tb.add_cell(
                i,
                -1,
                width,
                height,
                text=i + 1,
                loc="right",
                edgecolor="none",
                facecolor="none",
            )
            tb.add_cell(
                -1,
                i,
                width,
                height / 2,
                text=i + 1,
                loc="center",
                edgecolor="none",
                facecolor="none",
            )

        ax.add_table(tb)
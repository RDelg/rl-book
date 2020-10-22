import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from maze import Maze
from learner import ValueLearner


def figure_3_2(maze: Maze, discount: float):
    print("Running figure_3_2")
    learner = ValueLearner(maze=maze, discount=discount, random=True)
    for i in tqdm(range(1000)):
        value = learner.value.copy()
        learner.update_value()
        new_value = learner.value
        if np.sum(np.abs(value - new_value)) < 1e-4:
            print(f"Converge at iteration {i}")
            break

    fig, ax = plt.subplots()
    learner.plot_value(ax)
    fig.savefig("figure_3_2.png", dpi=100)


def figure_3_5(maze: Maze, discount: float):
    print("Running figure_3_5")
    learner = ValueLearner(maze=maze, discount=discount, random=False)
    for i in tqdm(range(1000)):
        value = learner.value.copy()
        learner.update_value()
        new_value = learner.value
        if np.sum(np.abs(value - new_value)) < 1e-4:
            print(f"Converge at iteration {i}")
            break
    fig, ax = plt.subplots()
    learner.plot_value(ax)
    fig.savefig("figure_3_5.png", dpi=100)


def main():
    SHAPE = (5, 5)
    A = np.array([0, 1])
    A_PRIME = np.array([4, 1])
    B = np.array([0, 3])
    B_PRIME = np.array([2, 3])
    DISCOUNT = 0.9

    m = Maze(shape=SHAPE)
    m.add_reward(A, 10)
    m.add_reward(B, 5)
    m.add_hop(A, A_PRIME)
    m.add_hop(B, B_PRIME)

    figure_3_2(maze=m, discount=DISCOUNT)
    figure_3_5(maze=m, discount=DISCOUNT)


if __name__ == "__main__":
    main()
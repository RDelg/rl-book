import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from learner import DynamicPolicyLearner
from env import RentalCarEnv


def figure_4_2():
    learner = DynamicPolicyLearner(RentalCarEnv, fixed_return=True)

    fig = plt.figure(figsize=(12, 6))
    # Plot policies
    for i in range(4):
        ax = fig.add_subplot(2, 3, i + 1)
        if i == 3:
            ax.set_xlabel("#Cars at second location")
            ax.set_ylabel("#Cars at first location")
        learner.plot_policy(ax, title=f"$\\pi_{i}$")
        learner.policy_evaluation()
        learner.policy_improvement()
    learner.plot_policy(fig.add_subplot(235), title="$\\pi_4$")

    # Plot value
    ax = fig.add_subplot(236, projection="3d")
    learner.plot_value(ax, title="$v_{\\pi_4}$")
    ax.set_ylabel("#Cars at second location")
    ax.set_xlabel("#Cars at first location")

    fig.savefig("figure_4_2.png", dpi=100)


if __name__ == "__main__":
    figure_4_2()
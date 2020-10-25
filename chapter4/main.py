import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from learner import DynamicPolicyLearner
from env import RentalCarEnv


def figure_4_2():
    learner = DynamicPolicyLearner(RentalCarEnv, fixed_return=True)
    # Plot
    fig = plt.figure(figsize=(12, 6))

    for i in range(4):
        learner.plot_policy(fig.add_subplot(2, 3, i + 1), title=f"$\\pi_{i}$")
        learner.policy_evaluation()
        learner.policy_improvement()

    learner.plot_policy(fig.add_subplot(235), title="$\\pi_4$")
    learner.plot_value(fig.add_subplot(236, projection="3d"), title="$v_{\\pi_4}$")

    fig.savefig("figure_4_2.png", dpi=100)


if __name__ == "__main__":
    figure_4_2()
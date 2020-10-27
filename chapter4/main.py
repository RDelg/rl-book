import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from learner import DynamicPolicyLearner
from env import RentalCarEnv


def figure_4_2():
    learner = DynamicPolicyLearner(RentalCarEnv, fixed_return=True)

    def plot_policy(
        ax: plt.Axes = None,
        title: str = "Value",
    ):
        img = np.flipud(learner.policy)
        ax.imshow(
            img,
            cmap=plt.get_cmap("Spectral"),
            vmin=RentalCarEnv.act_space().min,
            vmax=RentalCarEnv.act_space().max,
        )

        # We don't want to show all ticks...
        ticks_range = np.arange(learner.obs_space_range)
        ticks_plot = [ticks_range[0], ticks_range[-1]]

        ax.set_xticks(ticks_plot)
        ax.set_yticks(np.flip(ticks_plot))

        ax.set_xticklabels(ticks_plot)
        ax.set_yticklabels(ticks_plot)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(learner.obs_space_range):
            for j in range(learner.obs_space_range):
                ax.text(
                    j, i, img[i, j], ha="center", va="center", color="b", fontsize=8
                )

        ax.set_title(title)

    def plot_value(
        ax: plt.Axes = None,
        title: str = "Value",
    ):
        x = np.arange(learner.obs_space_range)
        y = np.arange(learner.obs_space_range)
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, np.fliplr(learner.value), cmap=plt.get_cmap("Spectral"))

        # We don't want to show all ticks...
        ticks_range = np.arange(learner.obs_space_range)
        ticks_plot = [ticks_range[0], ticks_range[-1]]

        ax.set_xticks(ticks_plot)
        ax.set_yticks(ticks_plot)

        ax.set_xticklabels(np.flip(ticks_plot))
        ax.set_yticklabels(ticks_plot)

        ax.set_title(title)
        # Rotate
        ax.view_init(60, 60)

    fig = plt.figure(figsize=(12, 6))
    # Plot policies
    for i in range(4):
        ax = fig.add_subplot(2, 3, i + 1)
        if i == 3:
            ax.set_xlabel("#Cars at second location")
            ax.set_ylabel("#Cars at first location")
        plot_policy(ax, title=f"$\\pi_{i}$")
        learner.policy_evaluation()
        learner.policy_improvement()
    plot_policy(fig.add_subplot(235), title="$\\pi_4$")

    # Plot value
    ax = fig.add_subplot(236, projection="3d")
    plot_value(ax, title="$v_{\\pi_4}$")
    ax.set_ylabel("#Cars at second location")
    ax.set_xlabel("#Cars at first location")

    fig.savefig("figure_4_2.png", dpi=100)


if __name__ == "__main__":
    figure_4_2()
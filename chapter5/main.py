import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from env import BlackJack
from predictor import MonteCarloPredictor
from controller import MonteCarloController


def plot_state_value(value, ax=None, title="Value"):
    x_range = value.shape[1]
    y_range = value.shape[0]
    X, Y = np.meshgrid(np.arange(x_range), np.arange(y_range))
    ax.plot_surface(X, Y, value, cmap=plt.get_cmap("coolwarm"))
    ax.set_xticks(np.arange(x_range))
    ax.set_yticks(np.arange(y_range))
    ax.set_xlabel("Dealer sum")
    ax.set_ylabel("Player sum")
    ax.set_xticklabels([x + 1 for x in range(x_range)])
    ax.set_yticklabels(np.arange(y_range) + 12)
    ax.set_title(title)


def plot_policy(policy, ax=None, title="Policy"):
    x_range = policy.shape[1]
    y_range = policy.shape[0]
    cmap = sns.cubehelix_palette(start=2.8, rot=0.1, light=0.9, n_colors=2)
    sns.heatmap(policy, cmap=ListedColormap(cmap), center=0.5, ax=ax)
    ax.set_xticks(np.arange(x_range) + 0.5)
    ax.set_yticks(np.arange(y_range) + 0.5)
    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xticklabels([x + 1 for x in range(x_range)])
    ax.set_yticklabels(np.arange(y_range) + 12)
    # Set the colorbar labels
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(["0", "1"])


def figure_5_1(figsize=(12, 12)):
    fig = plt.figure(figsize=figsize)
    env = BlackJack()
    policy = lambda state: state[0] < 20
    for i, it in enumerate([10_000, 500_000]):
        mc = MonteCarloPredictor(env)
        mc.predict(policy, n_iters=it)
        plot_state_value(
            mc.state_value[:, 1, :],
            fig.add_subplot(2, 2, 1 + i, projection="3d"),
            f"Usable ace {it} iterations",
        )
        plot_state_value(
            mc.state_value[:, 0, :],
            fig.add_subplot(2, 2, 3 + i, projection="3d"),
            f"No usable ace {it} iterations",
        )

    fig.savefig("figure_5_1.png", dpi=100)


def figure_5_2(figsize=(12, 12)):
    env = BlackJack()
    mc = MonteCarloController(env)

    eps = 0.3
    iterations = 5_000_000

    policy = mc.generate_soft_policy(mc.greedy_policy, eps, 2)
    mc.on_policy_improvement(policy, iters=iterations)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(222, projection="3d")
    plot_state_value(
        mc.state_value[:, 1, :], ax=ax, title=f"Value usable ace {iterations} iters"
    )
    ax = fig.add_subplot(224, projection="3d")
    plot_state_value(
        mc.state_value[:, 0, :], ax=ax, title=f"value no usable ace {iterations} iters"
    )

    ax = fig.add_subplot(221)
    plot_policy(
        mc.greedy_policy[:, 1, :], ax=ax, title=f"Policy usable ace {iterations} iters"
    )
    ax = fig.add_subplot(223)
    plot_policy(
        mc.greedy_policy[:, 0, :],
        ax=ax,
        title=f"Policy no usable ace {iterations} iters",
    )

    fig.savefig("figure_5_2.png", dpi=100)


if __name__ == "__main__":
    figure_5_1()
    figure_5_2()
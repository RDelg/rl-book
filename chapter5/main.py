import numpy as np
import matplotlib.pyplot as plt

from env import BlackJack
from predictor import MonteCarloPredictor


def figure_5_1(figsize=(12, 12)):
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


if __name__ == "__main__":
    figure_5_1()
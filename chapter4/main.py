import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from learner import DynamicPolicyLearner
from env import GridEnv, RentalCarEnv, GamblerEnv


def figure_4_1():

    env = GridEnv()
    learner = DynamicPolicyLearner(env)

    def plot_value(ax: plt.Axes):
        rounded_value = np.round(learner.value, decimals=1)
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = rounded_value.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        for (i, j), val in np.ndenumerate(rounded_value):
            tb.add_cell(i, j, width, height, text=val, loc="center", facecolor="white")

        ax.add_table(tb)

    def add_arrow(ax, x, y, direction, shape):
        x_step, y_step = 1 / shape[0], 1 / shape[1]
        new_x, new_y = x_step * x + x_step / 2, 1 - y_step * y - y_step / 2

        head_length = 0.05
        if direction == 0:
            dx, dy = -(x_step / 2 - head_length), 0.0
        elif direction == 1:
            dx, dy = x_step / 2 - head_length, 0.0
        elif direction == 2:
            dx, dy = 0.0, y_step / 2 - head_length
        elif direction == 3:
            dx, dy = 0.0, -(y_step / 2 - head_length)
        else:
            raise ValueError("Invalid direction")

        ax.arrow(
            x=new_x,
            y=new_y,
            dx=dx,
            dy=dy,
            head_width=0.02,
            head_length=head_length,
            width=0.003,
            fc="k",
            ec="k",
        )

    def plot_policy(ax: plt.Axes):
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = learner.value.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        for (i, j) in np.ndindex(learner.value.shape):
            tb.add_cell(i, j, width, height)

        ax.add_table(tb)

        for idx in np.ndindex(learner.value.shape):
            if env.is_terminal(idx):
                continue
            actions = GridEnv.legal_actions(idx)
            returns = np.zeros_like(actions, dtype=np.float32)
            for i, action in np.ndenumerate(actions):
                new_state = env.step(idx, action)
                returns[i] = learner.value[new_state]

            for a in np.where(returns >= returns.max())[0]:
                add_arrow(ax, idx[0], idx[1], a, learner.value.shape)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Plot
    fig = plt.figure(figsize=(12, 36))

    def add_k_text(ax, text):
        ax.text(-0.2, 0.5, text, fontsize=12)

    ax = fig.add_subplot(621)
    plot_value(ax)
    ax.set_title("$v_{k}$ for the random policy")
    add_k_text(ax, "$k=0$")

    ax = fig.add_subplot(622)
    plot_policy(ax)
    ax.set_title("greedy policy w.r.t $v_{k}$")

    # Plot value
    plot_ks = [1, 2, 3, 10]
    plot_i = 2
    for k in range(11):
        learner.value_iteration(max_iters=1)
        if k in plot_ks:
            ax = fig.add_subplot(6, 2, plot_i + 1)
            plot_value(ax)
            add_k_text(ax, f"$k={k}$")
            plot_policy(fig.add_subplot(6, 2, plot_i + 2))
            plot_i += 2

    learner.value_iteration(max_iters=100)

    ax = fig.add_subplot(6, 2, plot_i + 1)
    plot_value(ax)
    add_k_text(ax, "$k=\\infty$")

    plot_policy(fig.add_subplot(6, 2, plot_i + 2))

    fig.savefig("figure_4_1.png", dpi=100)


def figure_4_2():
    env = RentalCarEnv(fixed_return=False)
    learner = DynamicPolicyLearner(env)

    def plot_policy(
        ax: plt.Axes,
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
        ax: plt.Axes,
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


def figure_4_3():
    # Aggresive policy
    initial_policy = np.array(
        [GamblerEnv.legal_actions((i + 1,)).max() for i in range(99)]
    )

    env = GamblerEnv()
    learner = DynamicPolicyLearner(env, eps=1e-12, initial_policy=initial_policy)

    def plot_value(ax: plt.Axes):
        ax.plot(np.arange(1, 100), learner.value)
        xticks = [1, 25, 50, 75, 99]
        yticks = np.round(np.linspace(0, 1, 6), 1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel("Capital")
        ax.set_ylabel("Value estimates")
        ax.legend(
            [f"sweep {s+1}" for s in range(max_plot_iter)] + ["Final function value"]
        )

    def plot_policy(ax: plt.Axes):
        ax.step(np.arange(1, 100), learner.policy, where="mid")
        xticks = [1, 25, 50, 75, 99]
        yticks = [1] + [x * 10 for x in range(1, 6)]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel("Capital")
        ax.set_ylabel("Final policy (stake)")

    fig = plt.figure(figsize=(12, 6))

    # Plot value
    ax = fig.add_subplot(211)
    max_plot_iter = 3
    for i in range(30):
        learner.policy_evaluation(max_iters=1)
        if i < max_plot_iter:
            plot_value(ax)
    plot_value(ax)

    # Plot policy
    learner.policy_improvement()
    ax = fig.add_subplot(212)
    plot_policy(ax)

    fig.savefig("figure_4_3.png", dpi=100)


if __name__ == "__main__":
    figure_4_1()
    figure_4_2()
    figure_4_3()
import multiprocessing
import concurrent.futures


import numpy as np
import seaborn as sns
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from chapter5.env import BlackJack, SingleState
from chapter5.predictor import MonteCarloPredictor
from chapter5.controller import MonteCarloController


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
    policy = lambda state: state[0] < 20 - env._min_sum
    for i, it in enumerate([10_000, 500_000]):
        mc = MonteCarloPredictor(env)
        mc.predict(policy, alpha=1.0, n_episodes=it)
        plot_state_value(
            mc.V[:, 1, :],
            fig.add_subplot(2, 2, 1 + i, projection="3d"),
            f"Usable ace {it} iterations",
        )
        plot_state_value(
            mc.V[:, 0, :],
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
    plot_state_value(mc.V[:, 1, :], ax=ax, title=f"Value usable ace {iterations} iters")
    ax = fig.add_subplot(224, projection="3d")
    plot_state_value(
        mc.V[:, 0, :], ax=ax, title=f"value no usable ace {iterations} iters"
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


def figure_5_3(figsize=(12, 12)):

    true_value = -0.27726
    init_state = (13, 1, 2)
    idx_state = (1, 1, 1)

    target_policy = np.ones((10, 2, 10), dtype=np.int32)
    target_policy[-2:, :, :] = 0

    env = BlackJack()
    iters_base = [10 ** x for x in range(5)]
    iters_arr = np.unique(
        np.concatenate(
            [
                np.arange(iters_base[i - 1], iters_base[i] + 1, step=iters_base[i - 1])
                for i in range(1, len(iters_base))
            ]
        )
    )
    iters_run = [iters_arr[0]] + np.diff(iters_arr).tolist()
    runs = 100
    run_w_Vs = []
    run_o_Vs = []
    for _ in trange(runs):
        iter_w_Vs = []
        iter_o_Vs = []
        mc_w = MonteCarloController(env)
        mc_o = MonteCarloController(env)
        for iterations in tqdm(iters_run):
            mc_w.weighted_predict(
                target_policy,
                epsilon=1.0,
                iters=iterations,
                init_state=init_state,
                disable_tqdm=True,
            )
            mc_o.ordinary_predict(
                target_policy,
                epsilon=1.0,
                iters=iterations,
                init_state=init_state,
                disable_tqdm=True,
            )
            iter_w_Vs.append(mc_w.Q[idx_state].max(-1))
            iter_o_Vs.append(mc_o.Q[idx_state].max(-1))
        run_w_Vs.append(iter_w_Vs)
        run_o_Vs.append(iter_o_Vs)

    err_w = np.power(true_value - np.array(run_w_Vs), 2).mean(0)
    err_o = np.power(true_value - np.array(run_o_Vs), 2).mean(0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(iters_arr, err_w, color="red", label="Weighted importance sampling")
    ax.plot(iters_arr, err_o, color="green", label="Ordinary importance sampling")
    ax.set_xscale("log")
    ax.set_ylim([-0.1, 5.0])
    ax.legend(fontsize=20)
    ax.set_ylabel("Mean square error (average over 100 runs)", size=20)
    ax.set_xlabel("Episodes (log scale)", size=20)
    ax.tick_params(labelsize="large")
    fig.savefig("figure_5_3.png", dpi=100)


def run_prediction(target_policy, iters_arr, seed):
    np.random.seed(seed)
    env = SingleState()
    controller = MonteCarloController(env)
    Vs = []
    for iters in iters_arr:
        controller.ordinary_predict(
            target_policy,
            epsilon=1.0,
            iters=iters,
            disable_tqdm=False,
        )
        Vs.append(controller.V[0])
    return Vs


def figure_5_4(figsize=(12, 12)):
    target_policy = np.zeros((1,), dtype=np.int32)
    iters_base = [10 ** x for x in range(8)]
    iters_arr = np.unique(
        np.concatenate(
            [
                np.arange(iters_base[i - 1], iters_base[i] + 1, step=iters_base[i - 1])
                for i in range(1, len(iters_base))
            ]
        )
    )
    iters_run = [iters_arr[0]] + np.diff(iters_arr).tolist()
    runs = 10

    run_Vs = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count() * 2
    ) as executor:
        future_to_V = [
            executor.submit(run_prediction, target_policy, iters_run, i)
            for i in range(runs)
        ]
    for future in tqdm(
        concurrent.futures.as_completed(future_to_V), total=runs, disable=False
    ):
        run_Vs.append(future.result())

    run_Vs = np.array(run_Vs).T

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.plot(iters_arr, run_Vs)
    ax.set_xscale("log")
    ax.set_ylabel(
        "Monte-Carlo estimate of $v_\\pi(s)$ with ordinary importance sampling (ten runs)",
        size=20,
    )
    ax.set_xlabel("Episodes (log scale)", size=20)
    ax.set_ylim([-0.1, 4.0])
    fig.savefig("figure_5_4.png", dpi=100)


if __name__ == "__main__":
    figure_5_1()
    # figure_5_2()
    # figure_5_3()
    # figure_5_4()

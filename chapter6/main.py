import multiprocessing
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from chapter6.env import RandomWalk
from chapter6.predictor import TDPredictor
from chapter5.predictor import MonteCarloPredictor


def _figure_6_2_left(n_states, init_value, ax):
    env = RandomWalk(n_states)
    policy = lambda _: int(np.random.uniform() < 0.5)
    predictor = TDPredictor(env)
    # Generating data
    values_history = []
    iters = [0] + [10 ** x for x in range(3)]
    for n in iters:
        predictor.reset(init_value)
        predictor.predict(policy=policy, alpha=0.1, n_iters=n)
        values_history.append(predictor.V)

    for n, v in zip(iters, values_history):
        ax.plot(np.arange(n_states), v, marker="o", label=f"{n} iters")
    ax.legend(fontsize=12)
    ax.set_title("Estimated value", size=20)
    ax.set_xlabel("State", size=20)
    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels([chr(ord("A") + i) for i in range(n_states)])


def _evaluate(predictor, real_value, n_episodes, alpha):
    loss = lambda prediction: np.sum((real_value - prediction) ** 2)
    policy = lambda _: int(np.random.uniform() < 0.5)
    error_history = [loss(predictor.V)]
    for _ in range(n_episodes):
        predictor.predict(policy=policy, alpha=alpha, n_iters=1, disable_tqdm=True)
        error_history.append(loss(predictor.V))
    return error_history


def _parallel_evaluation(n_runs: int, *args):
    run_errors = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count() * 2
    ) as executor:
        future_error = [executor.submit(_evaluate, *args) for _ in range(n_runs)]
    for future in tqdm(
        concurrent.futures.as_completed(future_error), total=n_runs, disable=False
    ):
        run_errors.append(future.result())

    return run_errors


def _figure_6_2_right(n_states, init_value, ax):
    real_value = np.arange(1, n_states + 1) / (n_states + 1)
    init_state = 0.5

    env = RandomWalk(n_states)
    td_predictor = TDPredictor(env)
    mc_predictor = MonteCarloPredictor(env)

    td_predictor.reset(init_state)
    mc_predictor.reset(init_state)

    n_episodes = 100
    n_runs = 100
    # TD
    td_alphas = [0.05, 0.1, 0.15]
    td_avg_error = []
    for a in tqdm(td_alphas, desc="TD runs"):
        td_avg_error.append(
            _parallel_evaluation(n_runs, td_predictor, real_value, n_episodes, a)
        )

    # MC
    mc_alphas = [0.05, 0.1, 0.15]
    mc_avg_error = []
    for a in tqdm(mc_alphas, desc="MC runs"):
        mc_avg_error.append(
            _parallel_evaluation(n_runs, mc_predictor, real_value, n_episodes, a)
        )

    # Plot
    linestyles = ["--", "-.", ":"]
    for alpha, avg_error, style in zip(td_alphas, td_avg_error, linestyles):
        ax.plot(
            np.array(avg_error).mean(0),
            label=f"td alpha {alpha}",
            color="b",
            linestyle=style,
        )

    for alpha, avg_error, style in zip(mc_alphas, mc_avg_error, linestyles):
        ax.plot(
            np.array(avg_error).mean(0),
            label=f"mc alpha {alpha}",
            color="r",
            linestyle=style,
        )

    ax.legend(fontsize=12)
    ax.set_title("Empirical RMS errors\naveraged over states", size=20)
    ax.set_xlabel("Walks / Episodes", size=20)


def example_6_2(figsize=(12, 6)):
    n_states = 5
    init_value = 0.5
    # Plot
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(1, 2)
    _figure_6_2_left(n_states, init_value, axs[0])
    _figure_6_2_right(n_states, init_value, axs[1])

    fig.savefig("example_6_2.png", dpi=100)


if __name__ == "__main__":
    example_6_2()

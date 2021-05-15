from typing import List
import multiprocessing
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from chapter5.predictor import MonteCarloPredictor, Predictor
from chapter6.env import RandomWalk, WindyGridWorld
from chapter6.predictor import TDPredictor
from chapter6.controller import SARSAController, e_greedy_policy


def _figure_6_2_left(ax: plt.Axes, n_states: int, init_value: float):
    env = RandomWalk(n_states)
    policy = lambda _: int(np.random.uniform() < 0.5)
    predictor = TDPredictor(env)
    # Generating data
    values_history = []
    iters = [0] + [10 ** x for x in range(3)]
    for n in iters:
        predictor.reset(init_value)
        predictor.predict(policy=policy, alpha=0.1, n_episodes=n)
        values_history.append(predictor.V)

    for n, v in zip(iters, values_history):
        ax.plot(np.arange(n_states), v, marker="o", label=f"{n} iters")
    ax.legend(fontsize=12)
    ax.set_title("Estimated value", size=20)
    ax.set_xlabel("State", size=20)
    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels([chr(ord("A") + i) for i in range(n_states)])


def _evaluate_with_random_policy(
    predictor: Predictor,
    real_value: np.ndarray,
    n_episodes: int,
    alpha: float,
    batch: bool = False,
) -> List[float]:
    loss = lambda prediction: np.sum((real_value - prediction) ** 2)
    policy = lambda _: int(np.random.uniform() < 0.5)
    error_history = [loss(predictor.V)]
    for _ in range(n_episodes):
        predictor.predict(
            policy=policy, alpha=alpha, n_episodes=1, disable_tqdm=True, batch=batch
        )
        error_history.append(loss(predictor.V))
    return error_history


def _parallel_evaluation(n_runs: int, *args) -> np.ndarray:
    run_errors = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count() * 2
    ) as executor:
        future_error = [
            executor.submit(_evaluate_with_random_policy, *args) for _ in range(n_runs)
        ]
    for future in tqdm(
        concurrent.futures.as_completed(future_error), total=n_runs, disable=False
    ):
        run_errors.append(future.result())

    return np.array(run_errors).mean(0)


def _figure_6_2_right(ax: plt.Axes, n_states: int, init_value: float):
    real_value = np.arange(1, n_states + 1) / (n_states + 1)

    env = RandomWalk(n_states)
    td_predictor = TDPredictor(env)
    mc_predictor = MonteCarloPredictor(env)

    td_predictor.reset(init_value)
    mc_predictor.reset(init_value)

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
            avg_error,
            label=f"td alpha {alpha}",
            color="b",
            linestyle=style,
        )

    for alpha, avg_error, style in zip(mc_alphas, mc_avg_error, linestyles):
        ax.plot(
            avg_error,
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
    _figure_6_2_left(axs[0], n_states, init_value)
    _figure_6_2_right(axs[1], n_states, init_value)
    fig.savefig("example_6_2.png", dpi=100)


def figure_6_2(figsize=(12, 6)):
    n_states = 5
    init_value = 0.5

    env = RandomWalk(n_states)
    td_predictor = TDPredictor(env)
    mc_predictor = MonteCarloPredictor(env)

    td_predictor.reset(init_value)
    mc_predictor.reset(init_value)

    real_value = np.arange(1, n_states + 1) / (n_states + 1)

    td_batch_errors = _parallel_evaluation(
        100, td_predictor, real_value, 100, 0.05, True
    )

    mc_batch_errors = _parallel_evaluation(
        100, mc_predictor, real_value, 100, 0.15, True
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)
    ax.plot(td_batch_errors, color="b", label="TD")
    ax.plot(mc_batch_errors, color="r", label="MC")
    ax.legend(fontsize=12)
    ax.set_title("Batch training", size=12)
    ax.set_ylabel("RMS errors\naveraged over states", size=12)
    ax.set_xlabel("Walks / Episodes", size=12)

    fig.savefig("figure_6_2.png", dpi=100)


def example_6_5(figsize=(8, 6)):
    env = WindyGridWorld(7, 10, winds=[0, 0, 0, 1, 1, 1, 1, 2, 2, 0], reward_pos=[3, 7])
    policy = e_greedy_policy(0.1)
    controller = SARSAController(env)

    dones = controller.predict(policy, alpha=0.5, n_episodes=1_000, max_iters=8000)

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)
    ax.plot(dones, color="r")
    ax.set_ylabel("Episodes", size=12)
    ax.set_xlabel("Time steps", size=12)
    fig.savefig("example_6_5.png", dpi=100)


if __name__ == "__main__":
    example_6_2()
    figure_6_2()
    example_6_5()

from typing import Any, Callable, List, Optional, Type
import multiprocessing
import concurrent.futures
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from chapter5.predictor import MonteCarloPredictor, Predictor
from chapter6.env import CliffGridWorld, RandomWalk, WindyGridWorld
from chapter6.predictor import TDPredictor
from chapter6.controller import (
    Policy,
    SARSAController,
    EpsilonGreedyPolicy,
    GreedyPolicy,
)


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


def _parallel_evaluation(func: Callable, n_runs: int, *args, **kwargs) -> np.ndarray:
    run_errors = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=multiprocessing.cpu_count() * 2
    ) as executor:
        future_error = [executor.submit(func, *args, **kwargs) for _ in range(n_runs)]
    for future in tqdm(
        concurrent.futures.as_completed(future_error), total=n_runs, disable=False
    ):
        run_errors.append(future.result())

    return np.array(run_errors).mean(0)


def _figure_6_2_right(
    ax: plt.Axes,
    n_states: int,
    init_value: float,
    plot_batch_versions: Optional[bool] = False,
):
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
    parallel_eval = partial(
        _parallel_evaluation,
        _evaluate_with_random_policy,
        n_runs=n_runs,
        real_value=real_value,
        n_episodes=n_episodes,
    )

    # TD
    td_avg_error = [
        parallel_eval(predictor=td_predictor, alpha=a)
        for a in tqdm(td_alphas, desc="TD runs")
    ]

    # MC
    mc_alphas = [0.05, 0.1, 0.15]
    mc_avg_error = [
        parallel_eval(predictor=mc_predictor, alpha=a)
        for a in tqdm(td_alphas, desc="MC runs")
    ]

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

    if plot_batch_versions:
        # TD Batch
        td_batch_avg_error = [
            parallel_eval(predictor=td_predictor, alpha=a, batch=True)
            for a in tqdm(td_alphas, desc="TD batch runs")
        ]
        # MC batch
        mc_batch_avg_error = [
            parallel_eval(predictor=mc_predictor, alpha=a, batch=True)
            for a in tqdm(td_alphas, desc="MC batch runs")
        ]
        for alpha, avg_error, style in zip(td_alphas, td_batch_avg_error, linestyles):
            ax.plot(
                avg_error,
                label=f"td alpha {alpha}",
                color="c",
                linestyle=style,
            )
        for alpha, avg_error, style in zip(mc_alphas, mc_batch_avg_error, linestyles):
            ax.plot(
                avg_error,
                label=f"mc alpha {alpha}",
                color="g",
                linestyle=style,
            )

    ax.legend(fontsize=12)
    ax.set_title("Empirical RMS errors\naveraged over states", size=20)
    ax.set_xlabel("Walks / Episodes", size=20)


def example_6_2(figsize=(12, 6)):
    print("Example 6.2")
    n_states = 5
    init_value = 0.5
    # Plot
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(1, 2)
    _figure_6_2_left(axs[0], n_states, init_value)
    _figure_6_2_right(axs[1], n_states, init_value)
    fig.savefig("example_6_2.png", dpi=100)


def figure_6_2(figsize=(12, 6)):
    print("Figure 6.2")
    n_states = 5
    init_value = 0.5

    env = RandomWalk(n_states)
    td_predictor = TDPredictor(env)
    mc_predictor = MonteCarloPredictor(env)

    td_predictor.reset(init_value)
    mc_predictor.reset(init_value)

    real_value = np.arange(1, n_states + 1) / (n_states + 1)

    td_batch_errors = _parallel_evaluation(
        _evaluate_with_random_policy, 100, td_predictor, real_value, 100, 0.05, True
    )

    mc_batch_errors = _parallel_evaluation(
        _evaluate_with_random_policy, 100, mc_predictor, real_value, 100, 0.15, True
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


def example_6_5(figsize=(8, 6), plot_q_learning: Optional[bool] = False):
    print("Example 6.5")
    env = WindyGridWorld(7, 10, winds=[0, 0, 0, 1, 1, 1, 1, 2, 2, 0], reward_pos=[3, 7])
    controller = SARSAController(env)
    policy = EpsilonGreedyPolicy(controller, 0.1)

    history_s = controller.predict(
        policy,
        alpha=0.5,
        n_episodes=8_000,
        max_iters=8_000,
    )

    controller.reset()

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)

    ax.plot(
        history_s["dones_iter"],
        list(range(1, len(history_s["dones_iter"]) + 1)),
        color="r",
        label="sarsa",
    )

    if plot_q_learning:
        target_policy = GreedyPolicy(controller)

        history_q = controller.predict(
            policy,
            target_policy=target_policy,
            alpha=0.5,
            n_episodes=8_000,
            max_iters=8_000,
        )

        ax.plot(
            history_q["dones_iter"],
            list(range(1, len(history_q["dones_iter"]) + 1)),
            color="b",
            label="q-learning",
        )
        ax.legend()

    ax.set_ylabel("Episodes", size=12)
    ax.set_xlabel("Time steps", size=12)
    fig.savefig("example_6_5.png", dpi=100)


def _predict(
    controller: Predictor,
    policy: Type[Policy],
    target_policy: Type[Policy],
    n_episodes: int,
    alpha: float,
    *args: Any,
    **kwds: Any,
) -> List[float]:
    controller.reset()
    target_policy = target_policy or policy
    history = controller.predict(
        policy(controller),
        target_policy=target_policy(controller),
        alpha=alpha,
        n_episodes=n_episodes,
        disable_tqdm=True,
        *args,
        **kwds,
    )

    return history["sum_reward"]


def example_6_6(figsize=(8, 6), plot_expected_sarsa: Optional[bool] = False):
    print("Example 6.6")
    env = CliffGridWorld(4, 12)
    controller = SARSAController(env)
    policy = partial(EpsilonGreedyPolicy, epsilon=0.1)
    target_policy = GreedyPolicy

    sarsa_sum_rewards = _parallel_evaluation(
        _predict, 100, controller, policy, None, 500, 0.5
    )

    q_learning_sum_rewards = _parallel_evaluation(
        _predict, 100, controller, policy, target_policy, 500, 0.5
    )

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)
    ax.plot(
        sarsa_sum_rewards,
        color="r",
        label="sarsa",
    )
    ax.plot(
        q_learning_sum_rewards,
        color="b",
        label="q-learning",
    )
    if plot_expected_sarsa:
        sarsa_expected_sum_rewards = _parallel_evaluation(
            _predict, 100, controller, policy, None, 500, 0.5, expected=True
        )
        ax.plot(
            sarsa_expected_sum_rewards,
            color="g",
            label="expected-sarsa",
        )
    ax.legend()
    ax.set_ylim(-100, 1)
    ax.set_ylabel("Sum of rewards during episodes", size=12)
    ax.set_xlabel("Episodes", size=12)
    fig.savefig("example_6_6.png", dpi=100)


def figure_6_3(figsize=(8, 6)):
    print("Figure 6.3")
    env = CliffGridWorld(4, 12)
    controller = SARSAController(env)
    policy = partial(EpsilonGreedyPolicy, epsilon=0.1)
    target_policy = GreedyPolicy

    interim_runs = 100
    interim_episodes = 100

    asymtotic_runs = 10
    asymtotic_episodes = 100_000

    alphas = np.arange(0.1, 1.05, 0.05)

    print("Interim")

    sarsa_interim = [
        _parallel_evaluation(
            _predict, interim_runs, controller, policy, None, interim_episodes, alpha
        ).mean(-1)
        for alpha in alphas
    ]

    q_learning_interim = [
        _parallel_evaluation(
            _predict,
            interim_runs,
            controller,
            policy,
            target_policy,
            interim_episodes,
            alpha,
        ).mean(-1)
        for alpha in alphas
    ]

    expected_sarsa_interim = [
        _parallel_evaluation(
            _predict,
            interim_runs,
            controller,
            policy,
            None,
            interim_episodes,
            alpha,
            expected=True,
        ).mean(-1)
        for alpha in alphas
    ]

    print("Asymtotic")

    sarsa_asymtotic = [
        _parallel_evaluation(
            _predict,
            asymtotic_runs,
            controller,
            policy,
            None,
            asymtotic_episodes,
            alpha,
        ).mean(-1)
        for alpha in alphas
    ]

    q_learning_asymtotic = [
        _parallel_evaluation(
            _predict,
            asymtotic_runs,
            controller,
            policy,
            target_policy,
            asymtotic_episodes,
            alpha,
        ).mean(-1)
        for alpha in alphas
    ]

    expected_sarsa_asymtotic = [
        _parallel_evaluation(
            _predict,
            asymtotic_runs,
            controller,
            policy,
            None,
            asymtotic_episodes,
            alpha,
            expected=True,
        ).mean(-1)
        for alpha in alphas
    ]

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, 1)
    ax.plot(
        alphas,
        sarsa_interim,
        color="b",
        marker="^",
        label="sarsa",
    )
    ax.plot(
        alphas,
        q_learning_interim,
        color="k",
        marker="s",
        label="q-learning",
    )
    ax.plot(
        alphas,
        expected_sarsa_interim,
        marker="x",
        color="r",
        label="expected sarsa",
    )

    ax.plot(
        alphas,
        sarsa_asymtotic,
        color="b",
        marker="^",
        label="sarsa",
    )
    ax.plot(
        alphas,
        q_learning_asymtotic,
        color="k",
        marker="s",
        label="q-learning",
    )
    ax.plot(
        alphas,
        expected_sarsa_asymtotic,
        marker="x",
        color="r",
        label="expected sarsa",
    )
    ax.legend()
    ax.set_ylim(-140, 1)
    ax.set_xlim(0.1, 1)
    ax.set_ylabel("Sum of rewards per episodes", size=12)
    ax.set_xlabel("alpha", size=12)
    fig.savefig("figure_6_3.png", dpi=100)


if __name__ == "__main__":
    example_6_2()
    figure_6_2()
    example_6_5()
    example_6_6()
    # figure_6_3()

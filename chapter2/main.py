from typing import Union, List

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from chapter2.env import KBandit
from chapter2.learner import KBanditLearner


class Experiment:
    """
    Helper class to run K-Armed Bandit experiments.

    Parameters
    ----------
    k : int
        Number of "arms" in the bandit enviroment.

    t : int
        Timesteps to run each learner

    runs : int
        Number of learners to run per each config

    config : dict of list of dicts
        Dictionary containing the Enviroment and Learner parameters.

    """

    def __init__(self, k: int, t: int, runs: int, configs: Union[dict, List[dict]]):
        self.k = k
        self.t = t
        self.runs = runs
        self.configs = [configs] if not isinstance(configs, list) else configs
        self.n_configs = len(self.configs)
        self.hist_R = np.zeros(shape=(self.t, self.runs, self.n_configs))
        self.hist_A = np.zeros(shape=(self.t, self.runs, self.n_configs))
        self.hist_best_action = np.zeros(shape=(self.runs, self.n_configs))
        self.pct_correct = None

    def run(self):
        for i, c in enumerate(tqdm(self.configs)):
            for n in tqdm(range(self.runs)):
                env_config = c.get("env", {})
                learner_config = c["learner"]
                env = KBandit(k=self.k, **env_config)
                learner = KBanditLearner(env, self.k, **learner_config)
                learner.play(self.t)
                self.hist_R[:, n, i] = learner.obtained_reward
                self.hist_A[:, n, i] = learner.actions_taken
                self.hist_best_action[n, i] = env.best_action

    def pct_correct_over_time(self):
        self.pct_correct = np.ones(shape=(self.t, len(self.configs)))
        for i in range(self.t):
            self.pct_correct[i, :] = np.mean(
                self.hist_A[i, :, :] == self.hist_best_action, axis=0
            )

    def plot_pct_correct(self, ax):
        if not self.pct_correct:
            self.pct_correct_over_time()
        ax.plot(self.pct_correct, alpha=0.5)
        ax.legend([str(x) for x in self.configs])
        ax.set_xlabel("Steps", size=20)
        ax.set_ylabel("% optimal action ", size=20)

    def plot_mean_reward(self, ax):
        ax.plot(np.mean(self.hist_R, axis=1), alpha=0.5)
        ax.legend([str(x) for x in self.configs])
        ax.set_xlabel("Steps", size=20)
        ax.set_ylabel("Average reward", size=20)


def figure_2_1():
    print(f"Running figure_2_1")
    k = 10
    t = 10000
    env = KBandit(k=k)
    reward = env.reward(t)
    # plot
    f = plt.figure(figsize=(12, 8))
    ax = f.subplots(1, 1)
    ax.violinplot(reward)
    ax.set_xlabel("Action", size=20)
    ax.set_ylabel("Reward\n distribution", size=20)
    f.savefig("figure_2_1.png", dpi=200)


def figure_2_2():
    print(f"Running figure_2_2")
    k = 10
    t = 1000
    runs = 2000
    configs = [
        {"learner": {"eps": 0.0}},
        {"learner": {"eps": 0.1}},
        {"learner": {"eps": 0.01}},
    ]

    exp = Experiment(k, t, runs, configs)
    exp.run()

    # plot
    f = plt.figure(figsize=(16, 8))
    axs = f.subplots(2, 1)
    exp.plot_mean_reward(axs[0])
    exp.plot_pct_correct(axs[1])
    f.savefig("figure_2_2.png", dpi=200)


def figure_2_3():
    print(f"Running figure_2_3")
    k = 10
    t = 1000
    runs = 2000

    configs = [
        {"learner": {"eps": 0.0, "initial": 5}},
        {"learner": {"eps": 0.1}},
    ]

    exp = Experiment(k, t, runs, configs)
    exp.run()

    # plot
    f = plt.figure(figsize=(16, 8))
    axs = f.subplots(2, 1)
    exp.plot_mean_reward(axs[0])
    exp.plot_pct_correct(axs[1])
    f.savefig("figure_2_3.png", dpi=200)


def figure_2_4():
    print(f"Running figure_2_4")
    k = 10
    t = 1000
    runs = 1000

    configs = [{"learner": {"eps": 0.0, "c": 2}}, {"learner": {"eps": 0.1}}]

    exp = Experiment(k, t, runs, configs)
    exp.run()

    # plot
    f = plt.figure(figsize=(16, 8))
    axs = f.subplots(2, 1)
    exp.plot_mean_reward(axs[0])
    exp.plot_pct_correct(axs[1])
    f.savefig("figure_2_4.png", dpi=200)


def figure_2_5():
    print(f"Running figure_2_5")
    k = 10
    t = 1000
    runs = 2000

    configs = [
        {
            "learner": {"delta": 0.4, "gradient": True, "gradient_base": True},
            "env": {"offset": 4},
        },
        {
            "learner": {"delta": 0.1, "gradient": True, "gradient_base": True},
            "env": {"offset": 4},
        },
        {
            "learner": {"delta": 0.4, "gradient": True, "gradient_base": False},
            "env": {"offset": 4},
        },
        {
            "learner": {"delta": 0.1, "gradient": True, "gradient_base": False},
            "env": {"offset": 4},
        },
    ]

    exp = Experiment(k, t, runs, configs)
    exp.run()

    # plot
    f = plt.figure(figsize=(16, 8))
    axs = f.subplots(2, 1)
    exp.plot_mean_reward(axs[0])
    exp.plot_pct_correct(axs[1])
    f.savefig("figure_2_5.png", dpi=200)


def figure_2_6():
    print(f"Running figure_2_6")
    k = 10
    t = 1000
    runs = 2000
    n = 10

    eps_greedy = np.logspace(-7, -2, num=n, base=2)
    delta_gradient = np.logspace(-5, 2, num=n, base=2)
    c_UCB = np.logspace(-4, 2, num=n, base=2)
    q_optimistic = np.logspace(-2, 3, num=n, base=2)

    cfg_eps_greedy = [{"learner": {"eps": x}} for x in eps_greedy]
    cfg_delta_gradient = [
        {"learner": {"delta": x, "gradient": True}} for x in delta_gradient
    ]
    cfg_c_UCB = [{"learner": {"c": x}} for x in c_UCB]
    cfg_q_optimistic = [
        {"learner": {"eps": 0, "delta": 0.1, "initial": x}} for x in q_optimistic
    ]

    xs = [eps_greedy, delta_gradient, c_UCB, q_optimistic]
    ys = np.zeros((len(xs), n))

    for i, configs in enumerate(
        [cfg_eps_greedy, cfg_delta_gradient, cfg_c_UCB, cfg_q_optimistic]
    ):
        exp = Experiment(k, t, runs, configs)
        exp.run()
        ys[i, :] = np.mean(exp.hist_R, axis=(0, 1))

    f = plt.figure(figsize=(16, 8))
    ax = f.subplots(1, 1)
    for x, y in zip(xs, ys):
        ax.semilogx(x, y, basex=2)
    ax.legend(["eps-greedy", "delta_gradient", "c_UCB", "q_optimistic delta=0.1"])
    ax.set_xlabel("Action", size=20)
    ax.set_ylabel("Average\n reward\n over first\n 1000 steps", size=20)
    f.savefig("figure_2_6.png", dpi=200)


if __name__ == "__main__":
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()

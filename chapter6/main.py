import numpy as np

import matplotlib.pyplot as plt

from chapter6.env import RandomWalk
from chapter6.predictor import TDPredictor


def example_6_2(figsize=(12, 6)):
    n_states = 5
    init_value = 0.5

    env = RandomWalk()
    policy = lambda _: int(np.random.uniform() < 0.5)
    predictor = TDPredictor(env)
    # Generating data
    values_history = []
    iters = [0] + [10 ** x for x in range(3)]
    for n in iters:
        predictor.reset(init_value)
        predictor.predict(policy=policy, alpha=0.1, n_iters=n)
        values_history.append(predictor.V)

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(121)
    for n, v in zip(iters, values_history):
        ax.plot(np.arange(n_states), v, marker="o", label=f"{n} iters")
    ax.legend(fontsize=12)
    ax.set_ylabel("Estimated value", size=20)
    ax.set_xlabel("State", size=20)
    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels([chr(ord("A") + i) for i in range(n_states)])
    fig.savefig("example_6_2.png", dpi=100)


if __name__ == "__main__":
    example_6_2()

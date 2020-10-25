from typing import Type

import numpy as np
from tqdm import tqdm

from env import Enviroment


class Learner:
    def __init__(self, env: Type[Enviroment], eps: float = 0.001, **env_args):
        assert (
            len(env.obs_space().dims) == 1
        ), "Enviroment observation space dims must be 1 dimentional"
        assert (
            len(env.obs_space().dims) == 1
        ), "Enviroment action space dims must be 1 dimentional"

        self.obs_space_range = env.obs_space().max - env.obs_space().min + 1
        self.action_space_range = env.act_space().max - env.act_space().min + 1
        self.action_probability = 1.0 / self.action_space_range
        self.obs_space_shape = [
            self.obs_space_range for _ in range(env.obs_space().dims[0])
        ]
        self.eps = eps
        self.reset()
        self.env = env(self.value, **env_args)

    def reset(self):
        self.value = np.zeros(shape=self.obs_space_shape)
        self.policy = np.zeros(shape=self.obs_space_shape, dtype=np.int32)

    def policy_improvement(self):
        stable = True
        with np.nditer(
            [self.policy], flags=["multi_index"], op_flags=[["readwrite"]]
        ) as it:
            for pol in tqdm(it, desc="Policy improvement"):
                old_action = np.int32(pol)
                actions = self.env.legal_actions(it.multi_index)
                q = np.zeros_like(actions, dtype=np.float32)
                for a, action in enumerate(actions):
                    q[a] += self.env.dynamics(it.multi_index, action)
                pol[...] = actions[np.argmax(q)]
                if np.int32(pol) != old_action:
                    stable = False
        return stable

    def policy_evaluation(self):
        max_iters = 100
        for _ in tqdm(range(max_iters), desc="Policy evaluation"):
            old_value = self.value.copy()
            with np.nditer(
                [self.value, self.policy],
                flags=["multi_index"],
                op_flags=[["readwrite"], ["readonly"]],
            ) as it:
                for val, pol in tqdm(it, desc="Policy evaluation pass"):
                    val[...] = self.env.dynamics(it.multi_index, pol)
            max_diff = np.max(np.abs(self.value - old_value))
            print(f"Max diff: {max_diff}")
            if max_diff < self.eps:
                print("Converged")
                break

    # def plot_policy(self, ax=None, figsize=(6, 6)):
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)
    #     img = np.flipud(self.policy)
    #     im = ax.imshow(img)
    #     # We want to show all ticks...
    #     ax.set_xticks(np.arange(self.obs_space_range))
    #     ax.set_yticks(np.flip(np.arange(self.obs_space_range)))

    #     ax.set_xticklabels(np.arange(self.obs_space_range))
    #     ax.set_yticklabels(np.arange(self.obs_space_range))

    #     # Rotate the tick labels and set their alignment.
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    #     # Loop over data dimensions and create text annotations.
    #     for i in range(self.obs_space_range):
    #         for j in range(self.obs_space_range):
    #             text = ax.text(j, i, img[i, j], ha="center", va="center", color="w")

    #     ax.set_title("Policy")

    # def plot_value(self, ax=None, figsize=(6, 6)):
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)
    #         ax = fig.add_subplot(111, projection="3d")

    #     x = np.arange(self.obs_space_range)
    #     y = np.arange(self.obs_space_range)
    #     X, Y = np.meshgrid(x, y)
    #     ax.plot_surface(X, Y, self.value, cmap=cm.coolwarm)
    #     ax.set_xticks(np.arange(self.obs_space_range))
    #     ax.set_yticks(np.flip(np.arange(self.obs_space_range)))

    #     ax.set_xticklabels(np.arange(self.obs_space_range))
    #     ax.set_yticklabels(np.arange(self.obs_space_range))
    #     ax.set_title("Value")

    # def plot(self, figsize=(12, 6)):
    #     fig = plt.figure(figsize=figsize)
    #     self.plot_policy(fig.add_subplot(121))
    #     self.plot_value(fig.add_subplot(122, projection="3d"))

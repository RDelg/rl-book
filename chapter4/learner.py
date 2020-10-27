from typing import Type, Tuple, Optional

import numpy as np
from tqdm import tqdm

from env import Enviroment


class DynamicPolicyLearner:
    def __init__(
        self,
        env: Type[Enviroment],
        eps: float = 0.001,
        initial_policy: Optional[np.array] = None,
        **env_kwargs,
    ):
        assert (
            len(env.obs_space().dims) == 1
        ), "Enviroment observation space dims must be 1 dimentional"
        assert (
            len(env.obs_space().dims) == 1
        ), "Enviroment action space dims must be 1 dimentional"

        self.obs_space_range = env.obs_space().max - env.obs_space().min + 1
        self.action_space_range = env.act_space().max - env.act_space().min + 1
        self.obs_space_shape = [
            self.obs_space_range for _ in range(env.obs_space().dims[0])
        ]
        self._eps = eps
        self._env_cls = env
        self.reset(initial_policy, **env_kwargs)

    def reset(self, initial_policy, **env_kwargs):
        self.value = np.zeros(shape=self.obs_space_shape)
        if initial_policy is not None:
            assert all(
                [(a == b) for a, b in zip(initial_policy.shape, self.obs_space_shape)]
            ), f"initial policy must have shape {self.obs_space_shape}"
            self.policy = initial_policy.copy()
        else:
            self.policy = np.zeros(shape=self.obs_space_shape, dtype=np.int32)
        self.env = self._env_cls(self.value, **env_kwargs)

    def policy_improvement(self):
        stable = True
        with np.nditer(
            [self.policy], flags=["multi_index"], op_flags=[["readwrite"]]
        ) as it:
            for pol in tqdm(it, desc="Policy improvement", total=self.policy.size):
                old_action = np.int32(pol)
                state = self.env.idx_to_state(it.multi_index)
                actions = self.env.legal_actions(state)
                q = np.zeros_like(actions, dtype=np.float32)
                for a, action in enumerate(actions):
                    q[a] += self.env.dynamics(state, action)
                pol[...] = actions[np.argmax(q)]
                if np.int32(pol) != old_action:
                    stable = False
        return stable

    def policy_evaluation(self, max_iters: int = 100):
        max_diff = np.inf
        for it in tqdm(range(max_iters), desc="Policy evaluation"):
            old_value = self.value.copy()
            with np.nditer(
                [self.value, self.policy],
                flags=["multi_index"],
                op_flags=[["readwrite"], ["readonly"]],
            ) as it:
                for val, pol in tqdm(
                    it,
                    total=self.value.size,
                    desc=f"Policy evaluation iter {it}. Last max diff: {max_diff}",
                    leave=False,
                ):
                    val[...] = self.env.dynamics(
                        self.env.idx_to_state(it.multi_index), pol
                    )
            max_diff = np.max(np.abs(self.value - old_value))
            if max_diff < self._eps:
                break
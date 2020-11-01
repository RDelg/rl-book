from typing import Type, Tuple, Optional

import numpy as np
from tqdm import tqdm

from env import Enviroment


class DynamicPolicyLearner:
    """
    Dynamic programming learner.

    Parameters
    ----------

    env : Enviroment
        Enviroment from where the value and policy will be estimated.

    eps : float, default=1e-3
        Epsilon value to use to check the improvement in policy_evaluation
        and value_iteration methods.

    initial_policy: Optional numpy array, default=None
        Initial policy that the leraner will use. If none a zero initializated
        policy will be used.

    """

    def __init__(
        self,
        env: Enviroment,
        eps: float = 1e-3,
        initial_policy: Optional[np.array] = None,
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
        self._env = env
        self.reset(initial_policy)

    def reset(self, initial_policy: Optional[np.array] = None, **env_kwargs):
        """
        Resets the learner value and policy matrixes.
        """
        self.value = np.zeros(shape=self.obs_space_shape)
        if initial_policy is not None:
            assert all(
                [(a == b) for a, b in zip(initial_policy.shape, self.obs_space_shape)]
            ), f"initial policy must have shape {self.obs_space_shape}"
            self.policy = initial_policy.copy()
        else:
            self.policy = np.zeros(shape=self.obs_space_shape, dtype=np.int32)

    def policy_improvement(self):
        """Performs the policy improvement algorithm"""
        stable = True
        with np.nditer(
            [self.policy], flags=["multi_index"], op_flags=[["readwrite"]]
        ) as it:
            for pol in tqdm(it, desc="Policy improvement", total=self.policy.size):
                old_action = np.int32(pol)
                state = self._env.idx_to_state(it.multi_index)
                actions = self._env.legal_actions(state)
                q = np.zeros_like(actions, dtype=np.float32)
                for a, action in enumerate(actions):
                    q[a] = self._env.dynamics(self.value, state, action)
                pol[...] = actions[np.argmax(q)]
                if np.int32(pol) != old_action:
                    stable = False
        return stable

    def policy_evaluation(self, max_iters: int = 100):
        """Performs a policy evaluation algorithm sweep until it converge or reach
        the max_iter iterations.
        """
        max_diff = np.inf
        for i in tqdm(range(max_iters), desc="Policy evaluation"):
            old_value = self.value.copy()
            with np.nditer(
                [self.value, self.policy],
                flags=["multi_index"],
                op_flags=[["readwrite"], ["readonly"]],
            ) as it:
                for val, pol in tqdm(
                    it,
                    total=self.value.size,
                    desc=f"Policy evaluation iter {i}. Last max diff: {max_diff}",
                    leave=False,
                ):
                    val[...] = self._env.dynamics(
                        self.value, self._env.idx_to_state(it.multi_index), pol
                    )
            max_diff = np.max(np.abs(self.value - old_value))
            if max_diff < self._eps:
                break

    def value_iteration(self, max_iters: int = 100):
        """Performs a value iteration algorithm sweep until it converge or reach
        the max_iter iterations.
        """
        last_delta = np.inf
        for i in tqdm(range(max_iters), desc="Value iteration"):
            delta = 0
            with np.nditer(
                [self.value],
                flags=["multi_index"],
                op_flags=[["readwrite"]],
            ) as it:
                for val in tqdm(
                    it,
                    total=self.value.size,
                    desc=f"Value iteration iter {i}. Last delta {last_delta}",
                    leave=False,
                ):
                    state = self._env.idx_to_state(it.multi_index)
                    actions = self._env.legal_actions(state)
                    q = np.zeros_like(actions, dtype=np.float32)
                    for a, action in enumerate(actions):
                        q[a] += self._env.dynamics(self.value, state, action)
                    new_value = np.max(q)
                    delta += np.abs(val[...] - new_value)
                    val[...] = new_value
            last_delta = delta
            if last_delta < self._eps:
                break
from typing import Tuple

import numpy as np
from tqdm import trange

from env import Enviroment
from trajectory import Trajectory


class MonteCarloController:
    """Monte Carlo State Action Value Controller"""

    def __init__(self, env: Enviroment):
        self.env = env
        self._obs_space = env.obs_space()
        self._act_space = env.act_space()
        if self._act_space.dims > 1:
            raise ValueError(
                "MonteCarloController doesn't support more than one dimention in the action space"
            )
        self._n_actions = self._act_space.max - self._act_space.min + 1
        self.reset()

    def reset(self):
        get_shape = lambda x: [_max - _min + 1 for _max, _min in zip(x.max, x.min)]
        shape = get_shape(self._obs_space) + [self._n_actions]
        self.state_action_value = np.zeros(shape=shape, dtype=np.float32)

    def state_to_idx(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(x - y for x, y in zip(state, self._obs_space.min))

    def state_action_to_idx(
        self, state: Tuple[int, ...], action: int
    ) -> Tuple[int, ...]:
        return tuple(x - y for x, y in zip(state, self._obs_space.min)) + (action,)

    def generate_episode(
        self, policy: np.ndarray, init_state: Tuple[int, ...] = None
    ) -> Trajectory:
        if init_state is not None:
            self.env.state = init_state
        else:
            self.env.reset()
        current_state = self.env.state
        finished, reward = False, 0
        trajectory = Trajectory()
        while not finished:
            action_prob = policy[self.state_to_idx(current_state)]
            action = np.random.choice(
                self.env.legal_actions(current_state),
                p=action_prob,
            )
            trajectory.add_step(finished, current_state, action, reward)
            finished, reward, new_state = self.env.step(action)
            current_state = new_state
        trajectory.add_step(finished, current_state, None, reward)
        return trajectory

    @staticmethod
    def generate_soft_policy(
        greedy_policy: np.ndarray, epsilon: float, n_actions: int
    ) -> np.ndarray:
        soft_policy = (
            np.ones(greedy_policy.shape + (n_actions,), dtype=np.float32)
            * epsilon
            / n_actions
        )
        with np.nditer(
            [greedy_policy], flags=["multi_index"], op_flags=[["readonly"]]
        ) as it:
            for pol in it:
                soft_policy[it.multi_index + (int(pol),)] += 1.0 - epsilon
        return soft_policy

    def on_policy_improvement(
        self,
        policy,
        iters=1,
        epsilon=0.3,
        init_state=None,
        improve_policy=True,
        use_tqdm=False,
    ):
        n = np.zeros_like(self.state_action_value, dtype=np.int32)
        for _ in trange(iters, disable=use_tqdm):
            trajectory = self.generate_episode(policy, init_state=init_state)
            g = 0
            for i in range(len(trajectory) - 2, 0, -1):
                g += trajectory[i + 1].reward
                previous_states = set(x.state + (x.action,) for x in trajectory[0:i])
                if trajectory[i].state + (trajectory[i].action,) not in previous_states:
                    index = self.state_action_to_idx(
                        trajectory[i].state, trajectory[i].action
                    )
                    n[index] += 1
                    self.state_action_value[index] += (
                        g - self.state_action_value[index]
                    ) / n[index]
                    if improve_policy:
                        policy[index[:-1]] = self.generate_soft_policy(
                            self.state_action_value[index[:-1]].argmax(-1),
                            epsilon=epsilon,
                            n_actions=self._n_actions,
                        )

    def off_policy_improvement(
        self,
        target_policy,
        policy=None,
        iters=1,
        epsilon=0.3,
        init_state=None,
        improve_policy=True,
        weighted=True,
        use_tqdm=False,
    ):
        c = np.zeros_like(self.state_action_value, dtype=np.int32)
        if policy is None:
            policy = self.generate_soft_policy(
                target_policy, epsilon=epsilon, n_actions=self._n_actions
            )
        for _ in trange(iters, disable=use_tqdm):
            trajectory = self.generate_episode(policy, init_state=init_state)
            g = 0
            w = 1
            for i in range(len(trajectory) - 2, 0, -1):
                g += trajectory[i + 1].reward
                index = self.state_action_to_idx(
                    trajectory[i].state, trajectory[i].action
                )
                c[index] += w
                self.state_action_value[index] += (
                    g - self.state_action_value[index]
                ) * (w / c[index])
                if improve_policy:
                    target_policy[index[:-1]] = self.state_action_value[
                        index[:-1]
                    ].argmax(-1)
                if target_policy[index[:-1]] != trajectory[i].action:
                    break
                w /= policy[index]
            if improve_policy:
                policy = self.generate_soft_policy(
                    target_policy, epsilon=epsilon, n_actions=self._n_actions
                )

    @property
    def greedy_policy(self):
        return self.state_action_value.argmax(-1)

    @property
    def state_value(self):
        return self.state_action_value.max(-1)

    def get_policy_value(self, greedy_policy: np.ndarray) -> np.ndarray:
        value = np.zeros_like(greedy_policy, np.float32)
        with np.nditer(
            [value, greedy_policy],
            flags=["multi_index"],
            op_flags=[["writeonly"], ["readonly"]],
        ) as it:
            for val, pol in it:
                val[...] = self.state_action_value[it.multi_index + (int(pol),)]
        return value

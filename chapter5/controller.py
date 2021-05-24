from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
from tqdm import trange

from .env import Enviroment
from .types import StateIndex, Trajectory, State, StateActionIndex


class DiscreteController(metaclass=ABCMeta):
    def __init__(self, env: Enviroment):
        self.env = env
        self._n_actions = self.env.act_space.n

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def state_to_idx(self, state: State) -> StateIndex:
        return tuple(x - dim.minimum for x, dim in zip(state, self.env.obs_space))

    def state_action_to_idx(self, state: State, action: int) -> StateActionIndex:
        return self.state_to_idx(state) + (action,)


class MonteCarloController(DiscreteController):
    """Monte Carlo State Action Value Controller"""

    def __init__(self, env: Enviroment):
        super(MonteCarloController, self).__init__(env)
        self.reset()

    def reset(self):
        self.posible_actions = self.env.act_space.to_list()
        shape = [dim.n for dim in self.env.obs_space] + [self._n_actions]
        self.Q = np.zeros(shape=shape, dtype=np.float32)
        self.WG = np.zeros(shape=shape, dtype=np.float32)
        self.C = np.zeros_like(self.Q, dtype=np.float32)
        self.N = np.zeros_like(self.Q, dtype=np.int32)
        self.cum_iters = 0

    def generate_episode(
        self, policy: np.ndarray, init_state: Optional[State] = None
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
                self.posible_actions,
                p=action_prob,
            )
            trajectory.add_step(finished, current_state, reward, action)
            finished, new_state, reward = self.env.step(action)
            current_state = new_state
        trajectory.add_step(finished, current_state, reward, None)
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
        policy: np.ndarray,
        iters: int = 1,
        epsilon: float = 0.3,
        init_state: Optional[State] = None,
        disable_tqdm: bool = False,
    ):
        for _ in trange(iters, disable=disable_tqdm):
            trajectory = self.generate_episode(policy, init_state=init_state)
            G = 0
            previous_states = [x.state + (x.action,) for x in trajectory[0:-1]]
            for i in range(len(trajectory) - 2, -1, -1):
                G += trajectory[i + 1].reward
                previous_states.pop()
                _, s, _, a = trajectory[i]
                # First visit
                if s + (a,) not in previous_states:
                    s_a_idx = self.state_action_to_idx(s, a)
                    self.N[s_a_idx] += 1
                    self.Q[s_a_idx] += (G - self.Q[s_a_idx]) / self.N[s_a_idx]
                    policy[s_a_idx[:-1]] = self.generate_soft_policy(
                        self.Q[s_a_idx[:-1]].argmax(-1),
                        epsilon=epsilon,
                        n_actions=self._n_actions,
                    )

    def weighted_predict(
        self,
        target_policy: np.ndarray,
        iters: int = 1,
        epsilon: float = 0.3,
        init_state: Optional[State] = None,
        disable_tqdm: bool = False,
    ):
        b_policy = self.generate_soft_policy(
            target_policy, epsilon=epsilon, n_actions=self._n_actions
        )
        for _ in trange(iters, disable=disable_tqdm):
            trajectory = self.generate_episode(b_policy, init_state=init_state)
            G = 0.0
            W = 1.0
            for i in range(len(trajectory) - 2, -1, -1):
                G += trajectory[i + 1].reward
                _, s, _, a = trajectory[i]
                s_a_idx = self.state_action_to_idx(s, a)
                self.C[s_a_idx] += W
                self.Q[s_a_idx] += (G - self.Q[s_a_idx]) * (W / self.C[s_a_idx])
                W *= float(target_policy[s_a_idx[:-1]] == a) / b_policy[s_a_idx]
                if W == 0.0:
                    break

    def ordinary_predict(
        self,
        target_policy: np.ndarray,
        iters: int = 1,
        epsilon: float = 0.3,
        init_state: Optional[State] = None,
        disable_tqdm: bool = False,
    ):
        b_policy = self.generate_soft_policy(
            target_policy, epsilon=epsilon, n_actions=self._n_actions
        )
        for _ in trange(iters, disable=disable_tqdm):
            trajectory = self.generate_episode(b_policy, init_state=init_state)
            G = 0.0
            W = 1.0
            for i in range(len(trajectory) - 2, -1, -1):
                G += trajectory[i + 1].reward
                _, s, _, a = trajectory[i]
                s_a_idx = self.state_action_to_idx(s, a)
                self.WG[s_a_idx] += W * G
                self.N[s_a_idx] += 1
                W *= float(target_policy[s_a_idx[:-1]] == a) / b_policy[s_a_idx]

        with np.nditer(
            [self.Q, self.WG, self.N],
            flags=["multi_index"],
            op_flags=[["writeonly"], ["readonly"], ["readonly"]],
        ) as it:
            for q, wg, n in it:
                n = np.float32(n)
                if n != 0.0:
                    q[...] = np.float32(wg) / np.float32(n)

    def off_policy_improvement(
        self,
        target_policy: np.ndarray,
        iters: int = 1,
        epsilon: float = 0.3,
        init_state: Optional[State] = None,
        disable_tqdm: bool = False,
    ):
        for _ in trange(iters, disable=disable_tqdm):
            b_policy = self.generate_soft_policy(
                target_policy, epsilon=epsilon, n_actions=self._n_actions
            )
            trajectory = self.generate_episode(b_policy, init_state=init_state)
            G = 0
            W = 1
            for i in range(len(trajectory) - 2, -1, -1):
                G += trajectory[i + 1].reward
                _, s, _, a = trajectory[i]
                s_a_idx = self.state_action_to_idx(s, a)
                self.C[s_a_idx] += W
                self.Q[s_a_idx] += (G - self.Q[s_a_idx]) * (W / self.C[s_a_idx])
                W *= float(target_policy[s_a_idx[:-1]] == a) / b_policy[s_a_idx]
                if W == 0.0:
                    break
            target_policy[...] = self.Q.argmax(-1)

    @property
    def greedy_policy(self) -> np.ndarray:
        return self.Q.argmax(-1)

    @property
    def V(self) -> np.ndarray:
        return self.Q.max(-1)

    def get_policy_value(self, greedy_policy: np.ndarray) -> np.ndarray:
        value = np.zeros_like(greedy_policy, np.float32)
        with np.nditer(
            [value, greedy_policy],
            flags=["multi_index"],
            op_flags=[["writeonly"], ["readonly"]],
        ) as it:
            for val, pol in it:
                val[...] = self.Q[it.multi_index + (int(pol),)]
        return value

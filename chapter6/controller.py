from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any, Optional, Tuple

from tqdm import trange

from chapter5.env import Enviroment
from chapter5.types import State, Trajectory
from chapter5.controller import DiscreteController


class Policy(metaclass=ABCMeta):
    def __init__(self, controller: DiscreteController, *args: Any, **kwds: Any):
        pass

    @abstractmethod
    def update(self, *args: Any, **kwds: Any):
        pass

    @abstractmethod
    def greedy_action(self, *args: Any, **kwds: Any) -> int:
        pass

    @abstractmethod
    def prob(self, *args: Any, **kwds: Any) -> float:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class SARSAController(DiscreteController):
    def __init__(self, env: Enviroment, gamma: float = 1.0):
        super(SARSAController, self).__init__(env)
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.posible_actions = self.env.act_space.to_list()
        shape = [dim.n for dim in self.env.obs_space] + [self._n_actions]
        self.Q = np.zeros(shape=shape, dtype=np.float32)

    def predict(
        self,
        policy: Policy,
        target_policy: Optional[Policy] = None,
        alpha: float = 0.01,
        n_episodes: int = 1,
        init_state: Optional[State] = None,
        disable_tqdm: Optional[bool] = False,
        max_iters: Optional[int] = None,
    ):
        target_policy = target_policy or policy

        history = {"dones_iter": [], "sum_reward": []}
        total_iters = 0

        for _ in trange(
            n_episodes,
            desc=f"Value prediction iter",
            disable=disable_tqdm,
        ):
            if init_state is None:
                self.env.reset()
            else:
                self.env.state = init_state
            done = False
            reward = 0

            current_state = self.env.state

            trajectory = Trajectory()
            sum_reward = 0
            while not done and (max_iters is None or total_iters < max_iters):
                action, _ = policy(current_state)
                trajectory.add_step(done, current_state, reward, action)
                done, new_state, reward = self.env.step(action)
                sum_reward += reward
                total_iters += 1
                if done:
                    history["dones_iter"].append(total_iters)
                    history["sum_reward"].append(sum_reward)
                    next_action = None
                else:
                    next_action, _ = target_policy(new_state)
                    self._update_Q(
                        alpha,
                        current_state,
                        action,
                        new_state,
                        next_action,
                        done,
                        reward,
                    )
                current_state = new_state
            trajectory.add_step(done, current_state, reward, None)

        return history

    def _update_Q(
        self,
        alpha: float,
        current_state: State,
        action: int,
        new_state: State,
        next_action: int,
        done: bool,
        reward: float,
    ) -> None:
        c_sa_idx = self.state_action_to_idx(current_state, action)
        n_sa_idx = self.state_action_to_idx(new_state, next_action)
        next_q = 0 if done else self.Q[n_sa_idx]
        self.Q[c_sa_idx] += alpha * ((reward + self.gamma * next_q) - self.Q[c_sa_idx])


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self, controller: DiscreteController, epsilon: float, freeze: bool = False
    ):
        self.controller = controller
        self.epsilon = epsilon
        self.freeze = freeze
        self.update()

    def update(self):
        if self.freeze:
            self.Q = np.copy(self.controller.Q)
        else:
            self.Q = self.controller.Q

    def greedy_action(self, state: State) -> int:
        state_idx = self.controller.state_to_idx(state)
        return int(self.Q[state_idx].argmax(-1))

    def prob(self, state: State, action: int) -> float:

        greedy_action = self.greedy_action(state)
        prob = self.epsilon * 1.0 / self.controller.env.act_space.n

        prob = self.epsilon * 1.0 / self.controller.env.act_space.n

        if action == greedy_action:
            prob += 1.0 - self.epsilon

        return prob

    def __call__(self, state: State) -> Tuple[int, float]:
        greedy_action = self.greedy_action(state)

        prob = self.epsilon * 1.0 / self.controller.env.act_space.n
        if np.random.uniform() < self.epsilon:
            action = self.controller.env.act_space.sample()
        else:
            action = greedy_action

        if action == greedy_action:
            prob += 1.0 - self.epsilon

        return action, prob


class GreedyPolicy(Policy):
    def __init__(self, controller: DiscreteController, freeze: bool = False):
        self.controller = controller
        self.freeze = freeze
        self.update()

    def update(self):
        if self.freeze:
            self.Q = np.copy(self.controller.Q)
        else:
            self.Q = self.controller.Q

    def greedy_action(self, state: State) -> int:
        state_idx = self.controller.state_to_idx(state)
        return int(self.Q[state_idx].argmax(-1))

    def prob(self, state: State, action: int) -> float:
        return float(action == self.greedy_action(state))

    def __call__(self, state: State) -> Tuple[int, float]:
        return self.greedy_action(state), 1.0

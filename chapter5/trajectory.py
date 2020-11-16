from dataclasses import dataclass
from typing import Tuple


@dataclass
class Step:
    is_final: bool
    state: Tuple[int, ...]
    action: int
    reward: float

    def __iter__(self):
        return iter((self.is_final, self.state, self.action, self.reward))


class Trajectory:
    def __init__(self):
        self.steps = []

    def add_step(
        self, is_final: bool, state: Tuple[int, ...], action: int, reward: float
    ):
        self.steps.append(Step(is_final, state, action, reward))

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index: int):
        return self.steps[index]

    def __repr__(self):
        return "\n".join(
            [f"t={i}: {step.__dict__}" for i, step in enumerate(self.steps)]
        )

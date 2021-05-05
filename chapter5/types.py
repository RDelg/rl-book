from typing import List, Tuple, NamedTuple
from dataclasses import dataclass, astuple


State = Tuple[int, ...]
StateIndex = Tuple[int, ...]


class StateAction(NamedTuple):
    state: State
    action: int


class StateActionIndex(NamedTuple):
    state_idx: StateIndex
    action_idx: int


@dataclass
class Observation:
    is_final: bool
    state: State
    reward: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class Step(Observation):
    action: int

    def __iter__(self):
        return iter(astuple(self))


class Trajectory:
    def __init__(self):
        self.steps: List[Step] = []

    def add_step(self, is_final: bool, state: State, reward: float, action: int):
        self.steps.append(Step(is_final, state, reward, action))

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index: int):
        return self.steps[index]

    def __repr__(self):
        return "\n".join(
            [f"t={i}: {step.__dict__}" for i, step in enumerate(self.steps)]
        )

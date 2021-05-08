from typing import List, Tuple, NamedTuple


State = Tuple[int, ...]
StateIndex = Tuple[int, ...]
StateActionIndex = Tuple[int, ...]


class Observation(NamedTuple):
    is_final: bool
    state: State
    reward: float


class Step(NamedTuple):
    is_final: bool
    state: State
    reward: float
    action: int


class Trajectory:
    def __init__(self):
        self.steps: List[Step] = []

    def add_step(self, is_final: bool, state: State, reward: float, action: int):
        self.steps.append(Step(is_final, state, reward, action))

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> Step:
        return self.steps[index]

    def __repr__(self) -> str:
        return "\n".join(
            [f"t={i}: {step.__dict__}" for i, step in enumerate(self.steps)]
        )

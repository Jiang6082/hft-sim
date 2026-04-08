from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass
class DeterministicRNG:
    seed: int

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def uniform_int(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def normal(self, mean: float, stddev: float) -> float:
        return self._rng.gauss(mean, stddev)

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("values must not be empty")
        return self._rng.choice(values)

    def random(self) -> float:
        return self._rng.random()

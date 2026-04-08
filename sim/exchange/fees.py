from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    per_share: float = 0.0

    def compute(self, qty: int) -> float:
        if qty < 0:
            raise ValueError("qty must be non-negative")
        return self.per_share * qty

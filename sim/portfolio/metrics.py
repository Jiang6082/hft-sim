from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from sim.portfolio.account import Account


@dataclass(frozen=True)
class PerformanceMetrics:
    total_pnl: float
    sharpe: float
    max_drawdown: float
    num_fills: int
    ending_position: int


def compute_metrics(account: Account) -> PerformanceMetrics:
    equities = [point.equity for point in account.equity_curve]
    pnl_changes = [curr - prev for prev, curr in zip(equities, equities[1:])]

    total_pnl = account.last_equity() or 0.0
    sharpe = _sharpe(pnl_changes)
    max_drawdown = _max_drawdown(equities)

    return PerformanceMetrics(
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        num_fills=len(account.fill_history),
        ending_position=account.position,
    )


def _sharpe(pnl_changes: list[float]) -> float:
    if len(pnl_changes) < 2:
        return 0.0

    mean = sum(pnl_changes) / len(pnl_changes)
    variance = sum((change - mean) ** 2 for change in pnl_changes) / (len(pnl_changes) - 1)
    if variance == 0:
        return 0.0

    return mean / sqrt(variance) * sqrt(len(pnl_changes))


def _max_drawdown(equities: list[float]) -> float:
    if not equities:
        return 0.0

    peak = equities[0]
    max_drawdown = 0.0

    for equity in equities:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return max_drawdown

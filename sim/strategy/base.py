from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sim.exchange.order import OrderType, Side
from sim.portfolio.account import FillRecord


@dataclass(frozen=True)
class MarketSnapshot:
    ts: int
    best_bid: int | None
    best_ask: int | None
    position: int


@dataclass(frozen=True)
class PlaceOrder:
    oid: int
    owner: str
    side: Side
    type: OrderType
    price: int | None
    qty: int


@dataclass(frozen=True)
class CancelOrder:
    oid: int


StrategyAction = PlaceOrder | CancelOrder


class Strategy(Protocol):
    owner: str

    def on_book(self, snapshot: MarketSnapshot) -> list[StrategyAction]:
        ...

    def on_fill(self, fill: FillRecord) -> list[StrategyAction]:
        ...

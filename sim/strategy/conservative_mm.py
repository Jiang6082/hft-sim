from __future__ import annotations

from dataclasses import dataclass, field

from sim.exchange.order import OrderType, Side
from sim.portfolio.account import FillRecord
from sim.strategy.base import MarketSnapshot, PlaceOrder, StrategyAction


@dataclass
class ConservativeMarketMaker:
    owner: str = "mm"
    quote_size: int = 1
    entry_spread: int = 1
    profit_target: int = 2
    max_position: int = 2
    _next_oid: int = field(default=1, init=False)
    _last_entry_price: int | None = field(default=None, init=False)

    def on_book(self, snapshot: MarketSnapshot) -> list[StrategyAction]:
        if snapshot.best_bid is None or snapshot.best_ask is None:
            return []

        spread = snapshot.best_ask - snapshot.best_bid
        if snapshot.position == 0 and spread <= self.entry_spread:
            return [self._place(side=Side.BUY, price=snapshot.best_ask, qty=self.quote_size)]

        if snapshot.position > 0 and self._last_entry_price is not None:
            if snapshot.best_bid >= self._last_entry_price + self.profit_target:
                return [self._place(side=Side.SELL, price=snapshot.best_bid, qty=snapshot.position)]

        return []

    def on_fill(self, fill: FillRecord) -> list[StrategyAction]:
        if fill.side is Side.BUY:
            self._last_entry_price = fill.price
        elif fill.side is Side.SELL:
            self._last_entry_price = None
        return []

    def _place(self, side: Side, price: int, qty: int) -> PlaceOrder:
        oid = self._next_oid
        self._next_oid += 1
        return PlaceOrder(
            oid=oid,
            owner=self.owner,
            side=side,
            type=OrderType.LIMIT,
            price=price,
            qty=qty,
        )

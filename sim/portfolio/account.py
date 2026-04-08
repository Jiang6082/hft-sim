from __future__ import annotations

from dataclasses import dataclass, field

from sim.core.event import EventType
from sim.exchange.exchange import ExchangeEvent
from sim.exchange.fees import FeeModel
from sim.exchange.order import Side


@dataclass(frozen=True)
class FillRecord:
    oid: int
    owner: str
    side: Side
    price: int
    qty: int
    fee: float


@dataclass(frozen=True)
class EquityPoint:
    label: str
    equity: float


@dataclass
class Account:
    owner: str
    fee_model: FeeModel = field(default_factory=FeeModel)
    position: int = 0
    cash: float = 0.0
    fees_paid: float = 0.0
    fill_history: list[FillRecord] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    def apply_fill(self, oid: int, side: Side, price: int, qty: int, owner: str | None = None) -> FillRecord:
        if qty <= 0:
            raise ValueError("fill qty must be positive")
        if price <= 0:
            raise ValueError("fill price must be positive")

        fill_owner = self.owner if owner is None else owner
        if fill_owner != self.owner:
            raise ValueError(f"fill owner {fill_owner!r} does not match account owner {self.owner!r}")

        signed_qty = qty if side is Side.BUY else -qty
        gross_cash = price * qty
        fee = self.fee_model.compute(qty)

        self.position += signed_qty
        if side is Side.BUY:
            self.cash -= gross_cash
        else:
            self.cash += gross_cash
        self.cash -= fee
        self.fees_paid += fee

        record = FillRecord(
            oid=oid,
            owner=self.owner,
            side=side,
            price=price,
            qty=qty,
            fee=fee,
        )
        self.fill_history.append(record)
        return record

    def consume_exchange_event(self, event: ExchangeEvent) -> FillRecord | None:
        if event.type is not EventType.FILL:
            return None

        payload = event.payload
        return self.apply_fill(
            oid=payload["oid"],
            owner=payload["owner"],
            side=payload["side"],
            price=payload["price"],
            qty=payload["qty"],
        )

    def mark_to_mid(self, best_bid: int | None, best_ask: int | None, label: str) -> float:
        mid = self._mid_price(best_bid, best_ask)
        equity = self.cash + self.position * mid
        self.equity_curve.append(EquityPoint(label=label, equity=equity))
        return equity

    def last_equity(self) -> float | None:
        if not self.equity_curve:
            return None
        return self.equity_curve[-1].equity

    @staticmethod
    def _mid_price(best_bid: int | None, best_ask: int | None) -> float:
        if best_bid is None and best_ask is None:
            raise ValueError("cannot mark account without a reference price")
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)
        return (best_bid + best_ask) / 2.0

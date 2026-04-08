from __future__ import annotations

from dataclasses import dataclass

from sim.core.event import EventType
from sim.exchange.book import LimitOrderBook
from sim.exchange.matching import Fill, MatchingEngine
from sim.exchange.order import Order, OrderStatus, OrderType


@dataclass(frozen=True)
class ExchangeEvent:
    type: EventType
    payload: dict


class Exchange:
    def __init__(self) -> None:
        self.book = LimitOrderBook()
        self.matching = MatchingEngine()
        self.live_orders: dict[int, Order] = {}

    def submit_order(self, order: Order) -> list[ExchangeEvent]:
        events = [ExchangeEvent(type=EventType.ACK, payload={"oid": order.oid, "status": "accepted"})]
        fills = self.matching.execute(self.book, order)

        if order.type is OrderType.LIMIT and order.status is OrderStatus.LIVE:
            self.live_orders[order.oid] = order

        events.extend(self._fill_events(fills))
        return events

    def cancel_order(self, oid: int) -> list[ExchangeEvent]:
        order = self.live_orders.pop(oid, None)
        if order is None:
            return [
                ExchangeEvent(
                    type=EventType.CANCEL_ACK,
                    payload={"oid": oid, "canceled_qty": 0, "status": "missing"},
                )
            ]

        canceled_qty = self.book.cancel_limit(order.side, order.price, order.remaining)
        order.status = OrderStatus.CANCELED
        return [
            ExchangeEvent(
                type=EventType.CANCEL_ACK,
                payload={"oid": oid, "canceled_qty": canceled_qty, "status": "canceled"},
            )
        ]

    def top_of_book(self) -> tuple[int | None, int | None]:
        return self.book.top_of_book()

    @staticmethod
    def _fill_events(fills: list[Fill]) -> list[ExchangeEvent]:
        return [
            ExchangeEvent(
                type=EventType.FILL,
                payload={
                    "oid": fill.oid,
                    "owner": fill.owner,
                    "side": fill.side,
                    "price": fill.price,
                    "qty": fill.qty,
                },
            )
            for fill in fills
        ]

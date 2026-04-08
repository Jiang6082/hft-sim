from __future__ import annotations

from dataclasses import dataclass

from sim.exchange.book import LimitOrderBook
from sim.exchange.order import Order, OrderStatus, OrderType, Side


@dataclass(frozen=True)
class Fill:
    oid: int
    owner: str
    side: Side
    price: int
    qty: int


class MatchingEngine:
    def execute(self, book: LimitOrderBook, order: Order) -> list[Fill]:
        if order.qty <= 0:
            raise ValueError("order qty must be positive")

        fills: list[Fill] = []

        if order.type is OrderType.MARKET:
            fills = self._market_fills(book, order)
        elif self._is_crossing(book, order):
            fills = self._market_fills(book, order)

        order.filled += sum(fill.qty for fill in fills)

        if order.filled == 0:
            order.status = OrderStatus.LIVE
            if order.type is OrderType.LIMIT:
                book.add_limit(order.side, order.price, order.remaining)
        elif order.remaining == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL
            if order.type is OrderType.LIMIT:
                book.add_limit(order.side, order.price, order.remaining)
                order.status = OrderStatus.LIVE

        return fills

    def _market_fills(self, book: LimitOrderBook, order: Order) -> list[Fill]:
        book_fills = book.execute_market(order.side, order.remaining)
        return [
            Fill(
                oid=order.oid,
                owner=order.owner,
                side=order.side,
                price=fill.price,
                qty=fill.qty,
            )
            for fill in book_fills
        ]

    @staticmethod
    def _is_crossing(book: LimitOrderBook, order: Order) -> bool:
        if order.price is None:
            return False

        if order.side is Side.BUY:
            best_ask = book.best_ask()
            return best_ask is not None and order.price >= best_ask

        best_bid = book.best_bid()
        return best_bid is not None and order.price <= best_bid

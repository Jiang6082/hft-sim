from __future__ import annotations

from dataclasses import dataclass

from sim.exchange.order import Side


@dataclass(frozen=True)
class BookFill:
    price: int
    qty: int


class LimitOrderBook:
    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}

    def best_bid(self) -> int | None:
        if not self.bids:
            return None
        return max(self.bids)

    def best_ask(self) -> int | None:
        if not self.asks:
            return None
        return min(self.asks)

    def top_of_book(self) -> tuple[int | None, int | None]:
        return self.best_bid(), self.best_ask()

    def depth(self, side: Side, price: int) -> int:
        levels = self._levels(side)
        return levels.get(price, 0)

    def add_limit(self, side: Side, price: int, qty: int) -> None:
        self._validate_price(price)
        self._validate_qty(qty)

        levels = self._levels(side)
        levels[price] = levels.get(price, 0) + qty

    def cancel_limit(self, side: Side, price: int, qty: int) -> int:
        self._validate_price(price)
        self._validate_qty(qty)

        levels = self._levels(side)
        available = levels.get(price, 0)
        canceled = min(available, qty)
        remaining = available - canceled

        if remaining > 0:
            levels[price] = remaining
        elif price in levels:
            del levels[price]

        return canceled

    def execute_market(self, side: Side, qty: int) -> list[BookFill]:
        self._validate_qty(qty)

        fills: list[BookFill] = []
        remaining = qty
        resting = self.asks if side is Side.BUY else self.bids
        price_order = sorted(resting) if side is Side.BUY else sorted(resting, reverse=True)

        for price in price_order:
            if remaining == 0:
                break

            available = resting[price]
            fill_qty = min(available, remaining)
            fills.append(BookFill(price=price, qty=fill_qty))

            leftover = available - fill_qty
            if leftover > 0:
                resting[price] = leftover
            else:
                del resting[price]

            remaining -= fill_qty

        return fills

    def _levels(self, side: Side) -> dict[int, int]:
        if side is Side.BUY:
            return self.bids
        return self.asks

    @staticmethod
    def _validate_price(price: int) -> None:
        if price <= 0:
            raise ValueError("price must be positive")

    @staticmethod
    def _validate_qty(qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")

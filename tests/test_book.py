import pytest

from sim.exchange.book import LimitOrderBook
from sim.exchange.order import Side


def test_top_of_book_updates_after_limit_adds() -> None:
    book = LimitOrderBook()

    book.add_limit(Side.BUY, price=100, qty=5)
    book.add_limit(Side.BUY, price=101, qty=3)
    book.add_limit(Side.SELL, price=104, qty=2)
    book.add_limit(Side.SELL, price=103, qty=4)

    assert book.best_bid() == 101
    assert book.best_ask() == 103
    assert book.top_of_book() == (101, 103)


def test_add_limit_aggregates_depth_at_same_price() -> None:
    book = LimitOrderBook()

    book.add_limit(Side.BUY, price=100, qty=5)
    book.add_limit(Side.BUY, price=100, qty=7)

    assert book.depth(Side.BUY, 100) == 12


def test_cancel_limit_reduces_depth_and_removes_empty_level() -> None:
    book = LimitOrderBook()
    book.add_limit(Side.SELL, price=105, qty=10)

    canceled = book.cancel_limit(Side.SELL, price=105, qty=4)

    assert canceled == 4
    assert book.depth(Side.SELL, 105) == 6

    canceled = book.cancel_limit(Side.SELL, price=105, qty=10)

    assert canceled == 6
    assert book.depth(Side.SELL, 105) == 0
    assert book.best_ask() is None


def test_buy_market_order_sweeps_asks_in_price_order() -> None:
    book = LimitOrderBook()
    book.add_limit(Side.SELL, price=101, qty=2)
    book.add_limit(Side.SELL, price=102, qty=4)
    book.add_limit(Side.SELL, price=103, qty=3)

    fills = book.execute_market(Side.BUY, qty=5)

    assert [(fill.price, fill.qty) for fill in fills] == [(101, 2), (102, 3)]
    assert book.depth(Side.SELL, 101) == 0
    assert book.depth(Side.SELL, 102) == 1
    assert book.best_ask() == 102


def test_sell_market_order_sweeps_bids_in_descending_price_order() -> None:
    book = LimitOrderBook()
    book.add_limit(Side.BUY, price=100, qty=2)
    book.add_limit(Side.BUY, price=99, qty=4)
    book.add_limit(Side.BUY, price=98, qty=3)

    fills = book.execute_market(Side.SELL, qty=6)

    assert [(fill.price, fill.qty) for fill in fills] == [(100, 2), (99, 4)]
    assert book.depth(Side.BUY, 100) == 0
    assert book.depth(Side.BUY, 99) == 0
    assert book.best_bid() == 98


def test_invalid_price_and_qty_raise_value_error() -> None:
    book = LimitOrderBook()

    with pytest.raises(ValueError):
        book.add_limit(Side.BUY, price=0, qty=1)

    with pytest.raises(ValueError):
        book.execute_market(Side.SELL, qty=0)

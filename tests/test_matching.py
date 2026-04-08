from sim.exchange.book import LimitOrderBook
from sim.exchange.matching import MatchingEngine
from sim.exchange.order import Order, OrderStatus, OrderType, Side


def test_non_crossing_limit_order_rests_on_book() -> None:
    book = LimitOrderBook()
    engine = MatchingEngine()
    order = Order(
        oid=1,
        owner="mm",
        side=Side.BUY,
        type=OrderType.LIMIT,
        price=100,
        qty=5,
    )

    fills = engine.execute(book, order)

    assert fills == []
    assert order.status is OrderStatus.LIVE
    assert order.remaining == 5
    assert book.depth(Side.BUY, 100) == 5


def test_crossing_limit_order_executes_then_rests_remainder() -> None:
    book = LimitOrderBook()
    book.add_limit(Side.SELL, 101, 3)
    engine = MatchingEngine()
    order = Order(
        oid=2,
        owner="mm",
        side=Side.BUY,
        type=OrderType.LIMIT,
        price=101,
        qty=5,
    )

    fills = engine.execute(book, order)

    assert [(fill.price, fill.qty) for fill in fills] == [(101, 3)]
    assert order.filled == 3
    assert order.remaining == 2
    assert order.status is OrderStatus.LIVE
    assert book.depth(Side.BUY, 101) == 2
    assert book.best_ask() is None


def test_market_order_fills_without_resting() -> None:
    book = LimitOrderBook()
    book.add_limit(Side.BUY, 100, 4)
    engine = MatchingEngine()
    order = Order(
        oid=3,
        owner="taker",
        side=Side.SELL,
        type=OrderType.MARKET,
        price=None,
        qty=3,
    )

    fills = engine.execute(book, order)

    assert [(fill.price, fill.qty) for fill in fills] == [(100, 3)]
    assert order.status is OrderStatus.FILLED
    assert order.remaining == 0
    assert book.depth(Side.BUY, 100) == 1

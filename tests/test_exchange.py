from sim.core.event import EventType
from sim.exchange.exchange import Exchange
from sim.exchange.order import Order, OrderStatus, OrderType, Side


def test_submit_resting_limit_order_acknowledges_and_tracks_live_order() -> None:
    exchange = Exchange()
    order = Order(
        oid=10,
        owner="mm",
        side=Side.BUY,
        type=OrderType.LIMIT,
        price=100,
        qty=5,
    )

    events = exchange.submit_order(order)

    assert [event.type for event in events] == [EventType.ACK]
    assert exchange.live_orders[10] is order
    assert exchange.book.depth(Side.BUY, 100) == 5


def test_submit_crossing_order_emits_ack_and_fill_events() -> None:
    exchange = Exchange()
    exchange.book.add_limit(Side.SELL, 101, 4)
    order = Order(
        oid=11,
        owner="mm",
        side=Side.BUY,
        type=OrderType.LIMIT,
        price=101,
        qty=3,
    )

    events = exchange.submit_order(order)

    assert [event.type for event in events] == [EventType.ACK, EventType.FILL]
    assert events[1].payload["price"] == 101
    assert events[1].payload["qty"] == 3
    assert 11 not in exchange.live_orders
    assert order.status is OrderStatus.FILLED


def test_cancel_live_order_removes_book_depth() -> None:
    exchange = Exchange()
    order = Order(
        oid=12,
        owner="mm",
        side=Side.SELL,
        type=OrderType.LIMIT,
        price=105,
        qty=7,
    )
    exchange.submit_order(order)

    events = exchange.cancel_order(12)

    assert [event.type for event in events] == [EventType.CANCEL_ACK]
    assert events[0].payload["canceled_qty"] == 7
    assert exchange.book.depth(Side.SELL, 105) == 0
    assert order.status is OrderStatus.CANCELED


def test_cancel_missing_order_returns_missing_ack() -> None:
    exchange = Exchange()

    events = exchange.cancel_order(999)

    assert [event.type for event in events] == [EventType.CANCEL_ACK]
    assert events[0].payload["status"] == "missing"
    assert events[0].payload["canceled_qty"] == 0

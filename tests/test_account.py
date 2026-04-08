import pytest

from sim.exchange.exchange import Exchange
from sim.exchange.fees import FeeModel
from sim.exchange.order import Order, OrderType, Side
from sim.portfolio.account import Account


def test_apply_buy_and_sell_fills_updates_position_cash_and_fees() -> None:
    account = Account(owner="mm", fee_model=FeeModel(per_share=0.1))

    account.apply_fill(oid=1, owner="mm", side=Side.BUY, price=100, qty=2)
    account.apply_fill(oid=2, owner="mm", side=Side.SELL, price=103, qty=1)

    assert account.position == 1
    assert account.cash == pytest.approx(-97.3)
    assert account.fees_paid == pytest.approx(0.3)
    assert len(account.fill_history) == 2


def test_mark_to_mid_records_equity_curve() -> None:
    account = Account(owner="mm", fee_model=FeeModel(per_share=0.0))
    account.apply_fill(oid=1, owner="mm", side=Side.BUY, price=100, qty=2)

    equity = account.mark_to_mid(best_bid=101, best_ask=103, label="after-fill")

    assert equity == 4.0
    assert account.last_equity() == 4.0
    assert account.equity_curve[-1].label == "after-fill"


def test_consume_exchange_fill_event_updates_account() -> None:
    exchange = Exchange()
    account = Account(owner="mm", fee_model=FeeModel(per_share=0.05))
    exchange.book.add_limit(Side.SELL, 101, 4)
    order = Order(
        oid=7,
        owner="mm",
        side=Side.BUY,
        type=OrderType.LIMIT,
        price=101,
        qty=3,
    )

    events = exchange.submit_order(order)
    for event in events:
        account.consume_exchange_event(event)

    assert account.position == 3
    assert account.cash == pytest.approx(-(3 * 101) - 0.15)
    assert account.fees_paid == pytest.approx(0.15)
    assert len(account.fill_history) == 1


def test_mark_to_mid_uses_one_sided_reference_when_needed() -> None:
    account = Account(owner="mm")
    account.apply_fill(oid=1, owner="mm", side=Side.SELL, price=100, qty=2)

    equity = account.mark_to_mid(best_bid=99, best_ask=None, label="one-sided")

    assert equity == 2.0

import pytest

from sim.exchange.fees import FeeModel
from sim.exchange.order import Side
from sim.portfolio.account import Account
from sim.portfolio.metrics import compute_metrics


def test_compute_metrics_from_equity_curve() -> None:
    account = Account(owner="mm", fee_model=FeeModel(per_share=0.0))
    account.apply_fill(oid=1, owner="mm", side=Side.BUY, price=100, qty=1)
    account.mark_to_mid(best_bid=99, best_ask=101, label="t0")
    account.mark_to_mid(best_bid=101, best_ask=103, label="t1")
    account.mark_to_mid(best_bid=100, best_ask=102, label="t2")
    account.mark_to_mid(best_bid=102, best_ask=104, label="t3")

    metrics = compute_metrics(account)

    assert metrics.total_pnl == pytest.approx(3.0)
    assert metrics.max_drawdown == pytest.approx(1.0)
    assert metrics.num_fills == 1
    assert metrics.ending_position == 1
    assert metrics.sharpe != 0.0

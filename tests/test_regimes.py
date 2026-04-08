from sim.experiments.config import default_app_config
from sim.experiments.regimes import build_regime_events, regime_library
from sim.experiments.runner import run_regime_strategy_matrix


def test_regime_library_contains_expected_slugs() -> None:
    regimes = regime_library()

    assert set(regimes) >= {"baseline", "trend", "whipsaw"}
    assert regimes["baseline"].title == "Baseline Mean Reversion"


def test_build_regime_events_returns_deterministic_event_count() -> None:
    baseline_events = build_regime_events("baseline", seed=11, event_count=25)
    trend_events = build_regime_events("trend", seed=11, event_count=25)
    baseline_events_again = build_regime_events("baseline", seed=11, event_count=25)

    assert len(baseline_events) == 33
    assert len(trend_events) == 33
    assert baseline_events[0].ts == 0
    assert baseline_events == baseline_events_again
    assert baseline_events != trend_events


def test_run_regime_strategy_matrix_compares_strategies() -> None:
    matrix = run_regime_strategy_matrix(["baseline"], ["aggressive", "patient"], seed=9, event_count=40)

    assert set(matrix["baseline"]) == {"aggressive", "patient"}
    assert matrix["baseline"]["aggressive"].metrics.num_fills >= matrix["baseline"]["patient"].metrics.num_fills


def test_default_app_config_exposes_dashboard_defaults() -> None:
    config = default_app_config()

    assert config.dashboard.default_regime == "baseline"
    assert config.dashboard.default_strategy_a == "aggressive"

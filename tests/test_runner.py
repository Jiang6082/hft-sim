import pytest
from pathlib import Path

from sim.experiments.runner import export_artifacts, run_demo, run_mvp, run_regime_strategy_matrix


def test_run_demo_emits_human_readable_trace() -> None:
    trace = run_demo()
    joined = "\n".join(trace)

    assert trace[0] == "starting deterministic market-making replay"
    assert trace[-1] == "replay complete"
    assert "EXTERNAL_ADD" in joined
    assert "EXTERNAL_MKT" in joined
    assert "EXTERNAL_CANCEL" in joined
    assert "strategy fill buy" in joined
    assert "equity_chart" in joined


def test_run_mvp_is_deterministic_for_seeded_events() -> None:
    result_a = run_mvp()
    result_b = run_mvp()

    assert result_a.metrics.total_pnl == pytest.approx(result_b.metrics.total_pnl)
    assert result_a.metrics.num_fills == result_b.metrics.num_fills
    assert result_a.metrics.ending_position == result_b.metrics.ending_position
    assert len(result_a.rows) == 508
    assert result_a.rows[0] == result_b.rows[0]
    assert result_a.rows[-1] == result_b.rows[-1]


def test_export_artifacts_writes_csv_summary_and_svg(tmp_path: Path) -> None:
    scenario_results = run_regime_strategy_matrix(["baseline", "trend"], ["aggressive", "patient"])
    result = scenario_results["baseline"]["aggressive"]

    artifacts = export_artifacts(result, tmp_path, scenario_results=scenario_results)

    timeline_text = artifacts["timeline_csv"].read_text(encoding="utf-8")
    summary_text = artifacts["summary_txt"].read_text(encoding="utf-8")
    svg_text = artifacts["chart_svg"].read_text(encoding="utf-8")
    html_text = artifacts["report_html"].read_text(encoding="utf-8")

    assert "ts,event_type,best_bid,best_ask,position,cash,equity" in timeline_text
    assert "EXTERNAL_CANCEL" in timeline_text
    assert "total_pnl=" in summary_text
    assert "<svg" in svg_text
    assert "HFT MVP Equity / Position" in svg_text
    assert "<title>HFT Simulator MVP Dashboard</title>" in html_text
    assert "Replay Trace" in html_text
    assert "regime-select" in html_text
    assert "strategy-a-select" in html_text
    assert "strategy-b-select" in html_text
    assert "seed-input" in html_text
    assert "event-count-input" in html_text
    assert "Baseline Mean Reversion" in html_text
    assert "Trend Pressure" in html_text
    assert "Aggressive Market Maker" in html_text
    assert "Patient Market Maker" in html_text
    assert "const dashboard =" in html_text
    assert "resize-handle" in html_text
    assert "enableResize(panel);" in html_text
    assert "snapPanelRect" in html_text

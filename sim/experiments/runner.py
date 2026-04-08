from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sim.core.event import Event, EventType
from sim.core.queue import EventQueue
from sim.exchange.exchange import Exchange, ExchangeEvent
from sim.exchange.fees import FeeModel
from sim.exchange.order import Order, Side
from sim.experiments.config import default_app_config
from sim.experiments.regimes import build_regime_events, regime_library
from sim.portfolio.account import Account
from sim.portfolio.metrics import PerformanceMetrics, compute_metrics
from sim.strategy.base import MarketSnapshot, PlaceOrder
from sim.strategy.conservative_mm import ConservativeMarketMaker
from sim.strategy.dumb_mm import DumbMarketMaker


@dataclass(frozen=True)
class SimulationResult:
    trace: list[str]
    metrics: PerformanceMetrics
    rows: list["SimulationRow"]


@dataclass(frozen=True)
class SimulationRow:
    ts: int
    event_type: str
    best_bid: int | None
    best_ask: int | None
    position: int
    cash: float
    equity: float


@dataclass(frozen=True)
class ScenarioSpec:
    slug: str
    title: str
    description: str
    events: list[Event]


@dataclass(frozen=True)
class StrategySpec:
    slug: str
    title: str
    description: str
    factory: object


def demo_events() -> list[Event]:
    app_config = default_app_config()
    return build_regime_events(
        "baseline",
        seed=app_config.dashboard.default_seed,
        event_count=app_config.dashboard.default_event_count,
    )


def scenario_library(seed: int = 7, event_count: int = 500) -> dict[str, ScenarioSpec]:
    return {
        slug: ScenarioSpec(
            slug=slug,
            title=regime.title,
            description=regime.description,
            events=build_regime_events(slug, seed=seed, event_count=event_count),
        )
        for slug, regime in regime_library().items()
    }


def strategy_library() -> dict[str, StrategySpec]:
    return {
        "aggressive": StrategySpec(
            slug="aggressive",
            title="Aggressive Market Maker",
            description="Buys as soon as the spread tightens and exits on the first favorable bid improvement.",
            factory=partial(DumbMarketMaker, owner="mm"),
        ),
        "patient": StrategySpec(
            slug="patient",
            title="Patient Market Maker",
            description="Trades smaller size, demands tighter entry, and waits for a larger profit target before exiting.",
            factory=partial(ConservativeMarketMaker, owner="mm"),
        ),
    }


def run_scenario(scenario: ScenarioSpec, strategy_slug: str = "aggressive") -> SimulationResult:
    return run_mvp(scenario.events, strategy_slug=strategy_slug)


def run_all_scenarios(
    strategy_slug: str = "aggressive",
    seed: int = 7,
    event_count: int = 500,
) -> dict[str, SimulationResult]:
    return {
        slug: run_scenario(spec, strategy_slug=strategy_slug)
        for slug, spec in scenario_library(seed=seed, event_count=event_count).items()
    }


def run_regime_strategy_matrix(
    regime_slugs: Iterable[str] | None = None,
    strategy_slugs: Iterable[str] | None = None,
    seed: int = 7,
    event_count: int = 500,
) -> dict[str, dict[str, SimulationResult]]:
    scenarios = scenario_library(seed=seed, event_count=event_count)
    strategies = strategy_library()
    selected_regimes = list(regime_slugs or scenarios.keys())
    selected_strategies = list(strategy_slugs or strategies.keys())
    return {
        regime_slug: {
            strategy_slug: run_mvp(scenarios[regime_slug].events, strategy_slug=strategy_slug)
            for strategy_slug in selected_strategies
        }
        for regime_slug in selected_regimes
    }


def run_mvp(
    events: Iterable[Event] | None = None,
    strategy_slug: str = "aggressive",
) -> SimulationResult:
    queue = EventQueue()
    exchange = Exchange()
    account = Account(owner="mm", fee_model=FeeModel(per_share=0.0))
    strategy = build_strategy(strategy_slug)
    trace: list[str] = []
    rows: list[SimulationRow] = []

    for event in events or demo_events():
        queue.push(event)

    trace.append("starting deterministic market-making replay")

    while len(queue) > 0:
        event = queue.pop()
        trace.append(render_event(event))

        if event.type is EventType.EXTERNAL_ADD:
            exchange.book.add_limit(
                side=event.payload["side"],
                price=event.payload["price"],
                qty=event.payload["qty"],
            )
            trace.append(
                f"  added resting {side_name(event.payload['side'])} liquidity "
                f"qty={event.payload['qty']} @ {event.payload['price']}"
            )
        elif event.type is EventType.EXTERNAL_CANCEL:
            canceled = exchange.book.cancel_limit(
                side=event.payload["side"],
                price=event.payload["price"],
                qty=event.payload["qty"],
            )
            trace.append(
                f"  canceled {canceled} from {side_name(event.payload['side'])} "
                f"level @ {event.payload['price']}"
            )
        elif event.type is EventType.EXTERNAL_MKT:
            fills = exchange.book.execute_market(
                side=event.payload["side"],
                qty=event.payload["qty"],
            )
            trace.extend(render_external_fills(event.payload["side"], event.payload["qty"], fills))
        else:
            trace.append("  event type not yet handled")

        trace.extend(execute_strategy(exchange, account, strategy, event.ts))
        equity = account.mark_to_mid(*exchange.top_of_book(), label=f"t={event.ts}")
        rows.append(
            SimulationRow(
                ts=event.ts,
                event_type=event.type.name,
                best_bid=exchange.book.best_bid(),
                best_ask=exchange.book.best_ask(),
                position=account.position,
                cash=account.cash,
                equity=equity,
            )
        )
        trace.append(
            f"  account position={account.position} cash={account.cash:.2f} equity={equity:.2f}"
        )
        trace.append(f"  book {render_book(exchange)}")

    metrics = compute_metrics(account)
    trace.append("summary")
    trace.append(
        f"  total_pnl={metrics.total_pnl:.2f} sharpe={metrics.sharpe:.2f} "
        f"max_drawdown={metrics.max_drawdown:.2f} fills={metrics.num_fills} "
        f"ending_position={metrics.ending_position}"
    )
    trace.append(f"  equity_chart {render_equity_chart(account)}")
    trace.append("replay complete")
    return SimulationResult(trace=trace, metrics=metrics, rows=rows)


def run_demo(events: Iterable[Event] | None = None) -> list[str]:
    return run_mvp(events).trace


def build_strategy(strategy_slug: str):
    try:
        return strategy_library()[strategy_slug].factory()
    except KeyError as exc:
        raise ValueError(f"unknown strategy {strategy_slug!r}") from exc


def render_event(event: Event) -> str:
    return f"t={event.ts:>4} seq={event.seq:>2} {event.type.name}"


def render_external_fills(
    aggressor_side: Side,
    requested_qty: int,
    fills: list,
) -> list[str]:
    if not fills:
        return [
            f"  external market {side_name(aggressor_side)} qty={requested_qty} found no liquidity"
        ]

    filled_qty = sum(fill.qty for fill in fills)
    parts = ", ".join(f"{fill.qty} @ {fill.price}" for fill in fills)
    return [
        f"  external market {side_name(aggressor_side)} qty={requested_qty} filled={filled_qty}",
        f"  external fills {parts}",
    ]


def execute_strategy(exchange: Exchange, account: Account, strategy, ts: int) -> list[str]:
    snapshot = MarketSnapshot(
        ts=ts,
        best_bid=exchange.book.best_bid(),
        best_ask=exchange.book.best_ask(),
        position=account.position,
    )
    actions = strategy.on_book(snapshot)
    trace: list[str] = []

    for action in actions:
        if isinstance(action, PlaceOrder):
            trace.append(
                f"  strategy submit {side_name(action.side)} qty={action.qty} @ {action.price}"
            )
            order = Order(
                oid=action.oid,
                owner=action.owner,
                side=action.side,
                type=action.type,
                price=action.price,
                qty=action.qty,
            )
            exchange_events = exchange.submit_order(order)
            trace.extend(render_exchange_events(exchange_events, account, strategy))

    return trace


def render_exchange_events(
    exchange_events: list[ExchangeEvent],
    account: Account,
    strategy,
) -> list[str]:
    trace: list[str] = []
    for exchange_event in exchange_events:
        if exchange_event.type is EventType.ACK:
            trace.append(f"  exchange ack oid={exchange_event.payload['oid']}")
        elif exchange_event.type is EventType.FILL:
            fill = account.consume_exchange_event(exchange_event)
            if fill is not None:
                strategy.on_fill(fill)
                trace.append(
                    f"  strategy fill {side_name(fill.side)} qty={fill.qty} @ {fill.price}"
                )
    return trace


def render_book(exchange: Exchange) -> str:
    bid = exchange.book.best_bid()
    ask = exchange.book.best_ask()
    spread = "NA" if bid is None or ask is None else str(ask - bid)
    return (
        f"best_bid={format_level(bid, exchange.book.bids)} "
        f"best_ask={format_level(ask, exchange.book.asks)} "
        f"spread={spread}"
    )


def format_level(price: int | None, levels: dict[int, int]) -> str:
    if price is None:
        return "None"
    return f"{price}x{levels[price]}"


def side_name(side: Side) -> str:
    return "buy" if side is Side.BUY else "sell"


def render_equity_chart(account: Account) -> str:
    points = [point.equity for point in account.equity_curve]
    if not points:
        return "no-equity-points"

    chars = " .:-=+*#%@"
    lo = min(points)
    hi = max(points)
    if hi == lo:
        return "-" * len(points)

    rendered = []
    for point in points:
        idx = round((point - lo) / (hi - lo) * (len(chars) - 1))
        rendered.append(chars[idx])
    return "".join(rendered)


def export_artifacts(
    result: SimulationResult,
    output_dir: Path,
    scenario_results: dict[str, SimulationResult] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "timeline.csv"
    summary_path = output_dir / "summary.txt"
    chart_path = output_dir / "equity_position.svg"
    report_path = output_dir / "report.html"

    write_timeline_csv(result.rows, csv_path)
    write_summary(result, summary_path)
    write_svg_chart(result.rows, chart_path)
    write_html_report(
        result=result,
        path=report_path,
        chart_filename=chart_path.name,
        scenario_results=scenario_results,
    )

    return {
        "timeline_csv": csv_path,
        "summary_txt": summary_path,
        "chart_svg": chart_path,
        "report_html": report_path,
    }


def write_timeline_csv(rows: list[SimulationRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "event_type", "best_bid", "best_ask", "position", "cash", "equity"])
        for row in rows:
            writer.writerow(
                [
                    row.ts,
                    row.event_type,
                    row.best_bid,
                    row.best_ask,
                    row.position,
                    f"{row.cash:.2f}",
                    f"{row.equity:.2f}",
                ]
            )


def write_summary(result: SimulationResult, path: Path) -> None:
    lines = [
        "MVP simulation summary",
        f"total_pnl={result.metrics.total_pnl:.2f}",
        f"sharpe={result.metrics.sharpe:.2f}",
        f"max_drawdown={result.metrics.max_drawdown:.2f}",
        f"num_fills={result.metrics.num_fills}",
        f"ending_position={result.metrics.ending_position}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_chart(rows: list[SimulationRow], path: Path) -> None:
    width = 900
    height = 420
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    times = [row.ts for row in rows]
    equities = [row.equity for row in rows]
    positions = [float(row.position) for row in rows]

    min_value = min(equities + positions)
    max_value = max(equities + positions)
    equity_points = polyline_points(
        times, equities, min_value, max_value, margin_left, margin_top, plot_width, plot_height
    )
    position_points = polyline_points(
        times, positions, min_value, max_value, margin_left, margin_top, plot_width, plot_height
    )
    y_ticks = axis_ticks(min_value, max_value, count=5)

    grid_lines = []
    labels = []
    for tick in y_ticks:
        _, y = scale_point(
            x=times[0],
            y=tick,
            min_x=min(times),
            max_x=max(times),
            min_y=min_value,
            max_y=max_value,
            left=margin_left,
            top=margin_top,
            width=plot_width,
            height=plot_height,
        )
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="#d9dee5" stroke-width="1" />'
        )
        labels.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="12" fill="#43536b">{tick:.1f}</text>'.format(tick=tick)
        )

    x_labels = []
    for row in rows:
        x, _ = scale_point(
            x=row.ts,
            y=min_value,
            min_x=min(times),
            max_x=max(times),
            min_y=min_value,
            max_y=max_value,
            left=margin_left,
            top=margin_top,
            width=plot_width,
            height=plot_height,
        )
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 22}" text-anchor="middle" '
            f'font-size="12" fill="#43536b">{row.ts}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#f7f8fb" />
  <text x="{margin_left}" y="20" font-size="18" font-family="Menlo, Monaco, monospace" fill="#172033">HFT MVP Equity / Position</text>
  <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cfd6e0" />
  {''.join(grid_lines)}
  {''.join(labels)}
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#5b677a" stroke-width="1.5" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#5b677a" stroke-width="1.5" />
  {''.join(x_labels)}
  <polyline fill="none" stroke="#1177cc" stroke-width="3" points="{equity_points}" />
  <polyline fill="none" stroke="#d25b2a" stroke-width="3" points="{position_points}" />
  <circle cx="{last_x(rows):.1f}" cy="{last_y(rows, 'equity', min_value, max_value, margin_left, margin_top, plot_width, plot_height):.1f}" r="4" fill="#1177cc" />
  <circle cx="{last_x(rows):.1f}" cy="{last_y(rows, 'position', min_value, max_value, margin_left, margin_top, plot_width, plot_height):.1f}" r="4" fill="#d25b2a" />
  <text x="{width - 210}" y="28" font-size="12" fill="#1177cc">equity</text>
  <line x1="{width - 250}" y1="24" x2="{width - 220}" y2="24" stroke="#1177cc" stroke-width="3" />
  <text x="{width - 110}" y="28" font-size="12" fill="#d25b2a">position</text>
  <line x1="{width - 160}" y1="24" x2="{width - 130}" y2="24" stroke="#d25b2a" stroke-width="3" />
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_html_report(
    result: SimulationResult,
    path: Path,
    chart_filename: str,
    scenario_results: dict[str, dict[str, SimulationResult]] | None = None,
) -> None:
    app_config = default_app_config()
    default_regime = app_config.dashboard.default_regime
    default_strategy_a = app_config.dashboard.default_strategy_a
    default_strategy_b = app_config.dashboard.default_strategy_b
    default_seed = app_config.dashboard.default_seed
    default_event_count = app_config.dashboard.default_event_count
    regimes = regime_library()
    strategies = strategy_library()
    matrix = scenario_results or {
        default_regime: {
            default_strategy_a: result,
        }
    }
    dashboard_payload = build_dashboard_payload(
        matrix,
        seed=default_seed,
        event_count=default_event_count,
    )
    dashboard_json = json.dumps(dashboard_payload)
    regime_options = "".join(
        f'<option value="{slug}"{" selected" if slug == default_regime else ""}>{regime.title}</option>'
        for slug, regime in regimes.items()
    )
    strategy_options_a = "".join(
        f'<option value="{slug}"{" selected" if slug == default_strategy_a else ""}>{spec.title}</option>'
        for slug, spec in strategies.items()
    )
    strategy_options_b = "".join(
        f'<option value="{slug}"{" selected" if slug == default_strategy_b else ""}>{spec.title}</option>'
        for slug, spec in strategies.items()
    )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>HFT Simulator MVP Dashboard</title>
    <style>
      :root {{
        --bg: #060709;
        --panel: #0d0f12;
        --panel-2: #12151a;
        --panel-3: #171b20;
        --ink: #d6d9df;
        --muted: #6d7786;
        --line: #232830;
        --line-strong: #2e3540;
        --green: #41d392;
        --red: #eb5a52;
        --cyan: #8fb9ff;
        --amber: #d1b278;
        --shadow: none;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        color: var(--ink);
        background:
          linear-gradient(180deg, rgba(128, 151, 110, 0.12), transparent 4px),
          radial-gradient(circle at top center, rgba(111, 130, 164, 0.08), transparent 22%),
          var(--bg);
      }}
      .page {{
        width: calc(100vw - 28px);
        max-width: none;
        margin: 0;
        padding: 18px 14px 26px;
      }}
      .terminal-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        padding: 10px 8px 16px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 14px;
      }}
      .brand {{
        display: flex;
        align-items: center;
        gap: 14px;
      }}
      .brand-mark {{
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #f5f7fb;
      }}
      .brand-copy {{
        display: grid;
        gap: 4px;
      }}
      .brand-title {{
        color: #f0f3f8;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }}
      .brand-subtitle {{
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }}
      .status-strip {{
        display: flex;
        gap: 24px;
        padding-top: 6px;
        flex-wrap: wrap;
      }}
      .status-item {{
        display: grid;
        gap: 4px;
        text-align: right;
      }}
      .status-label {{
        font-size: 11px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }}
      .status-value {{
        font-size: 16px;
        color: #b5becd;
      }}
      .control-bar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px 12px;
        padding: 12px 8px 16px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 14px;
      }}
      .control-tab {{
        padding: 8px 0;
        margin-right: 14px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        border-bottom: 2px solid transparent;
      }}
      .control-tab.is-active {{
        color: #f4f6fb;
        border-color: #f4f6fb;
      }}
      .hero-controls {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-left: auto;
      }}
      .hero-controls label {{
        font-size: 11px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .hero-controls select,
      .hero-controls input {{
        appearance: none;
        border: 1px solid var(--line);
        background: var(--panel-2);
        color: var(--ink);
        border-radius: 0;
        padding: 9px 10px;
        font-size: 13px;
        min-width: 96px;
      }}
      .hero-controls button {{
        border: 1px solid var(--line-strong);
        background: #f0f2f6;
        color: #0a0d11;
        border-radius: 0;
        padding: 9px 14px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        cursor: pointer;
      }}
      .workspace {{
        position: relative;
        height: 1240px;
        margin-bottom: 14px;
        border: 1px solid var(--line);
        background:
          linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px),
          #05070a;
        background-size: 28px 28px, 28px 28px, auto;
        overflow: hidden;
      }}
      .panel {{
        position: absolute;
        background: linear-gradient(180deg, rgba(18, 22, 27, 0.98), rgba(10, 12, 15, 0.98));
        border: 1px solid var(--line);
        border-radius: 0;
        box-shadow: var(--shadow);
        overflow: hidden;
        min-width: 260px;
        min-height: 180px;
      }}
      .resize-handle {{
        position: absolute;
        z-index: 5;
      }}
      .resize-handle[data-dir="n"],
      .resize-handle[data-dir="s"] {{
        left: 10px;
        right: 10px;
        height: 10px;
      }}
      .resize-handle[data-dir="n"] {{
        top: -5px;
        cursor: ns-resize;
      }}
      .resize-handle[data-dir="s"] {{
        bottom: -5px;
        cursor: ns-resize;
      }}
      .resize-handle[data-dir="e"],
      .resize-handle[data-dir="w"] {{
        top: 10px;
        bottom: 10px;
        width: 10px;
      }}
      .resize-handle[data-dir="e"] {{
        right: -5px;
        cursor: ew-resize;
      }}
      .resize-handle[data-dir="w"] {{
        left: -5px;
        cursor: ew-resize;
      }}
      .resize-handle[data-dir="ne"],
      .resize-handle[data-dir="nw"],
      .resize-handle[data-dir="se"],
      .resize-handle[data-dir="sw"] {{
        width: 14px;
        height: 14px;
      }}
      .resize-handle[data-dir="ne"] {{
        top: -7px;
        right: -7px;
        cursor: nesw-resize;
      }}
      .resize-handle[data-dir="nw"] {{
        top: -7px;
        left: -7px;
        cursor: nwse-resize;
      }}
      .resize-handle[data-dir="se"] {{
        right: -7px;
        bottom: -7px;
        cursor: nwse-resize;
      }}
      .resize-handle[data-dir="sw"] {{
        left: -7px;
        bottom: -7px;
        cursor: nesw-resize;
      }}
      .panel-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        border-bottom: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.03);
        cursor: move;
        user-select: none;
      }}
      .panel-title {{
        margin: 0;
        color: #f1f4f9;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .panel-kicker {{
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .panel-body {{
        padding: 12px 14px 14px;
        min-height: 0;
      }}
      .scenario-copy {{
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
        margin: 0;
      }}
      .compare-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        align-items: stretch;
      }}
      .metric-card {{
        padding: 12px;
        background: var(--panel-2);
        border: 1px solid var(--line);
        min-height: 146px;
      }}
      .metric-card.is-secondary {{
        background: #10141a;
      }}
      .metric-label {{
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
        min-height: 32px;
      }}
      .metric-value {{
        font-size: clamp(28px, 2.2vw, 42px);
        line-height: 1;
        color: var(--green);
        margin-bottom: 10px;
        white-space: nowrap;
      }}
      .metric-value.is-negative {{
        color: var(--red);
      }}
      .chart-stack {{
        display: grid;
        gap: 14px;
      }}
      .chart-frame {{
        overflow: hidden;
        border: 1px solid var(--line);
        background: #090b0e;
      }}
      .chart-frame canvas {{
        display: block;
        width: 100%;
        height: auto;
      }}
      .mini-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }}
      th, td {{
        padding: 8px 8px;
        border-bottom: 1px solid var(--line);
        text-align: left;
      }}
      th {{
        color: var(--muted);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }}
      .table-wrap {{
        height: 560px;
        overflow: auto;
        border: 1px solid var(--line);
        background: #090b0e;
      }}
      tbody tr:nth-child(odd) {{
        background: rgba(255, 255, 255, 0.015);
      }}
      pre {{
        margin: 0;
        padding: 12px 14px;
        overflow: auto;
        border: 1px solid var(--line);
        background: #090b0e;
        font-family: inherit;
        font-size: 12px;
        line-height: 1.55;
        height: 560px;
      }}
      .legend {{
        display: flex;
        gap: 14px;
        margin-bottom: 10px;
        flex-wrap: wrap;
      }}
      .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .legend-swatch {{
        width: 18px;
        height: 3px;
      }}
      .caption {{
        color: var(--muted);
        font-size: 11px;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        line-height: 1.45;
      }}
      .live-badge {{
        color: var(--green);
      }}
      .warn-badge {{
        color: var(--amber);
      }}
      .pos {{
        color: var(--green);
      }}
      .neg {{
        color: var(--red);
      }}
      .neu {{
        color: var(--cyan);
      }}
      .panel-body.fill {{
        height: calc(100% - 48px);
      }}
      .panel-body.scroll {{
        overflow: auto;
      }}
      @media (max-width: 900px) {{
        .workspace {{
          height: 1600px;
        }}
        .mini-grid {{
          grid-template-columns: 1fr;
        }}
        .compare-grid {{
          grid-template-columns: 1fr;
        }}
        .hero-controls {{
          margin-left: 0;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="terminal-header">
        <div class="brand">
          <div class="brand-mark">UTC</div>
          <div class="brand-copy">
            <div class="brand-title">Trading Terminal</div>
            <div class="brand-subtitle">Event-Driven HFT Research Console</div>
          </div>
        </div>
        <div class="status-strip">
          <div class="status-item">
            <div class="status-label">Status</div>
            <div class="status-value live-badge" id="terminal-status">Streaming Playback</div>
          </div>
          <div class="status-item">
            <div class="status-label">Seed</div>
            <div class="status-value" id="terminal-seed">{default_seed}</div>
          </div>
          <div class="status-item">
            <div class="status-label">Events</div>
            <div class="status-value" id="terminal-events">{default_event_count}</div>
          </div>
          <div class="status-item">
            <div class="status-label">Latency</div>
            <div class="status-value warn-badge" id="terminal-latency">120ms</div>
          </div>
          <div class="status-item">
            <div class="status-label">Cursor</div>
            <div class="status-value" id="terminal-cursor">0 / 0</div>
          </div>
        </div>
      </section>

      <section class="control-bar">
        <div class="control-tab is-active">Equities/ETFs</div>
        <div class="control-tab">Calls</div>
        <div class="control-tab">Puts</div>
        <div class="control-tab">Prediction Markets</div>
        <div class="control-tab">Trade Events</div>
        <div class="control-tab">Market Events</div>
        <div class="hero-controls">
          <label for="regime-select">Regime</label>
          <select id="regime-select">{regime_options}</select>
          <label for="strategy-a-select">Strategy A</label>
          <select id="strategy-a-select">{strategy_options_a}</select>
          <label for="strategy-b-select">Strategy B</label>
          <select id="strategy-b-select">{strategy_options_b}</select>
          <label for="seed-input">Seed</label>
          <input id="seed-input" type="number" value="{default_seed}" min="0" step="1" />
          <label for="event-count-input">Events</label>
          <input id="event-count-input" type="number" value="{default_event_count}" min="1" step="1" />
          <button id="run-button" type="button">Run Selected</button>
        </div>
      </section>

      <section class="workspace" id="workspace">
        <section class="panel" data-panel-id="overview">
          <div class="panel-head">
            <div class="panel-title" id="scenario-title">Scenario</div>
            <div class="panel-kicker">Regime Overview</div>
          </div>
          <div class="panel-body">
            <p class="scenario-copy" id="scenario-description"></p>
            <div class="caption" id="chart-caption"></div>
          </div>
        </section>

        <section class="panel" data-panel-id="strategy">
          <div class="panel-head">
            <div class="panel-title">Strategy Comparison</div>
            <div class="panel-kicker">P&L / Risk</div>
          </div>
          <div class="panel-body">
            <div class="compare-grid" id="metric-grid"></div>
          </div>
        </section>

        <section class="panel" data-panel-id="micro">
          <div class="panel-head">
            <div class="panel-title">Spread / Cash</div>
            <div class="panel-kicker">Microstructure State</div>
          </div>
          <div class="panel-body">
            <div class="mini-grid">
              <div>
                <div class="legend">
                  <div class="legend-item"><span class="legend-swatch" style="background: var(--amber);"></span>Spread</div>
                </div>
                <div class="chart-frame">
                  <canvas id="spread-canvas" width="430" height="220"></canvas>
                </div>
              </div>
              <div>
                <div class="legend">
                  <div class="legend-item"><span class="legend-swatch" style="background: #8b79d8;"></span>Strategy A Cash</div>
                  <div class="legend-item"><span class="legend-swatch" style="background: #5b9bd5;"></span>Strategy B Cash</div>
                </div>
                <div class="chart-frame">
                  <canvas id="cash-canvas" width="430" height="220"></canvas>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="panel" data-panel-id="trace">
          <div class="panel-head">
            <div class="panel-title">Replay Trace</div>
            <div class="panel-kicker">Trace</div>
          </div>
          <div class="panel-body fill">
            <pre id="trace-panel"></pre>
          </div>
        </section>

        <section class="panel" data-panel-id="equity">
          <div class="panel-head">
            <div class="panel-title">Equity / Position</div>
            <div class="panel-kicker">Strategy Curves</div>
          </div>
          <div class="panel-body">
            <div>
              <div class="legend">
                <div class="legend-item"><span class="legend-swatch" style="background: var(--green);"></span>Strategy A Equity</div>
                <div class="legend-item"><span class="legend-swatch" style="background: #8affd1;"></span>Strategy B Equity</div>
                <div class="legend-item"><span class="legend-swatch" style="background: var(--red);"></span>Strategy A Position</div>
                <div class="legend-item"><span class="legend-swatch" style="background: #ff998d;"></span>Strategy B Position</div>
              </div>
              <div class="chart-frame">
                <canvas id="equity-position-canvas" width="880" height="360"></canvas>
              </div>
            </div>
          </div>
        </section>

        <section class="panel" data-panel-id="timeline">
          <div class="panel-head">
            <div class="panel-title">Market Timeline</div>
            <div class="panel-kicker">Book States</div>
          </div>
          <div class="panel-body fill">
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>t</th>
                    <th>event</th>
                    <th>bid</th>
                    <th>ask</th>
                    <th>pos</th>
                    <th>cash</th>
                    <th>equity</th>
                  </tr>
                </thead>
                  <tbody></tbody>
              </table>
            </div>
          </div>
        </section>
      </section>
    </main>
        <script id="scenario-data" type="application/json">{dashboard_json}</script>
        <script>
      const dashboard = JSON.parse(document.getElementById("scenario-data").textContent);
      const regimeSelect = document.getElementById("regime-select");
      const strategyASelect = document.getElementById("strategy-a-select");
      const strategyBSelect = document.getElementById("strategy-b-select");
      const seedInput = document.getElementById("seed-input");
      const eventCountInput = document.getElementById("event-count-input");
      const runButton = document.getElementById("run-button");
      const metricGrid = document.getElementById("metric-grid");
      const scenarioTitle = document.getElementById("scenario-title");
      const scenarioDescription = document.getElementById("scenario-description");
      const tracePanel = document.getElementById("trace-panel");
      const timelineBody = document.querySelector("tbody");
      const chartCaption = document.getElementById("chart-caption");
      const terminalSeed = document.getElementById("terminal-seed");
      const terminalEvents = document.getElementById("terminal-events");
      const terminalCursor = document.getElementById("terminal-cursor");
      const terminalStatus = document.getElementById("terminal-status");
      const workspace = document.getElementById("workspace");
      const PLAYBACK_INTERVAL_MS = 120;
      const CHART_WINDOW = 90;
      const TABLE_WINDOW = 22;
      const TRACE_WINDOW = 34;
      const SNAP_THRESHOLD = 14;
      const GRID_SIZE = 14;
      const MIN_PANEL_WIDTH = 260;
      const MIN_PANEL_HEIGHT = 180;
      let playbackTimer = null;
      let playbackState = null;
      const layoutStorageKey = "hft-terminal-layout-v1";
      const defaultPanelLayout = {{
        overview: {{ x: 14, y: 14, w: 280, h: 320, z: 1 }},
        strategy: {{ x: 310, y: 354, w: 420, h: 330, z: 2 }},
        micro: {{ x: 750, y: 354, w: 300, h: 330, z: 3 }},
        trace: {{ x: 1070, y: 14, w: 470, h: 670, z: 4 }},
        equity: {{ x: 14, y: 704, w: 760, h: 500, z: 5 }},
        timeline: {{ x: 794, y: 704, w: 746, h: 500, z: 6 }},
      }};
      let highestZ = 6;

      function metricCard(title, metrics, secondary) {{
        const klass = secondary ? "metric-card is-secondary" : "metric-card";
        const pnlClass = metrics.total_pnl >= 0 ? "metric-value" : "metric-value is-negative";
        return `
          <section class="${{klass}}">
            <div class="metric-label">${{title}}</div>
            <div class="${{pnlClass}}">${{metrics.total_pnl.toFixed(2)}}</div>
            <div class="caption">Sharpe ${{metrics.sharpe.toFixed(2)}} | MDD ${{metrics.max_drawdown.toFixed(2)}} | Fills ${{metrics.num_fills}} | End Pos ${{metrics.ending_position}}</div>
          </section>
        `;
      }}

      function currentSelection() {{
        return {{
          regime: regimeSelect.value,
          strategyA: strategyASelect.value,
          strategyB: strategyBSelect.value,
          seed: Number(seedInput.value || dashboard.seed || 7),
          eventCount: Number(eventCountInput.value || dashboard.event_count || 500),
        }};
      }}

      function initializeWorkspace() {{
        const saved = loadLayout();
        const resizeObserver = new ResizeObserver(() => saveLayout());
        document.querySelectorAll(".panel[data-panel-id]").forEach((panel) => {{
          const panelId = panel.dataset.panelId;
          const config = saved[panelId] || defaultPanelLayout[panelId];
          applyPanelLayout(panel, config);
          ensureResizeHandles(panel);
          panel.addEventListener("mousedown", () => bringToFront(panel));
          panel.addEventListener("mouseup", () => {{
            normalizePanelSize(panel);
            placePanel(panel, parseFloat(panel.style.left || "0"), parseFloat(panel.style.top || "0"));
            saveLayout();
          }});
          enableDrag(panel);
          enableResize(panel);
          resizeObserver.observe(panel);
        }});
        window.addEventListener("resize", clampAllPanels);
      }}

      function ensureResizeHandles(panel) {{
        if (panel.querySelector(".resize-handle")) return;
        ["n", "s", "e", "w", "ne", "nw", "se", "sw"].forEach((dir) => {{
          const handle = document.createElement("div");
          handle.className = "resize-handle";
          handle.dataset.dir = dir;
          panel.appendChild(handle);
        }});
      }}

      function enableDrag(panel) {{
        const handle = panel.querySelector(".panel-head");
        let active = false;
        let offsetX = 0;
        let offsetY = 0;

        handle.addEventListener("mousedown", (event) => {{
          if (event.target.closest("select, input, button")) return;
          active = true;
          bringToFront(panel);
          const rect = panel.getBoundingClientRect();
          const workspaceRect = workspace.getBoundingClientRect();
          offsetX = event.clientX - rect.left;
          offsetY = event.clientY - rect.top;
          event.preventDefault();
          const onMove = (moveEvent) => {{
            if (!active) return;
            const x = moveEvent.clientX - workspaceRect.left - offsetX;
            const y = moveEvent.clientY - workspaceRect.top - offsetY;
            placePanel(panel, x, y);
          }};
          const onUp = () => {{
            active = false;
            saveLayout();
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
          }};
          window.addEventListener("mousemove", onMove);
          window.addEventListener("mouseup", onUp);
        }});
      }}

      function enableResize(panel) {{
        panel.querySelectorAll(".resize-handle").forEach((handle) => {{
          handle.addEventListener("mousedown", (event) => {{
            const direction = handle.dataset.dir;
            if (!direction) return;
            const startRect = panelRect(panel);
            const startMouseX = event.clientX;
            const startMouseY = event.clientY;
            let active = true;
            bringToFront(panel);
            event.preventDefault();
            event.stopPropagation();

            const onMove = (moveEvent) => {{
              if (!active) return;
              const dx = moveEvent.clientX - startMouseX;
              const dy = moveEvent.clientY - startMouseY;
              const nextRect = resizeRectFromDirection(startRect, direction, dx, dy);
              applyPanelRect(panel, snapPanelRect(panel, nextRect, direction));
            }};
            const onUp = () => {{
              active = false;
              normalizePanelSize(panel);
              placePanel(panel, parseFloat(panel.style.left || "0"), parseFloat(panel.style.top || "0"));
              saveLayout();
              window.removeEventListener("mousemove", onMove);
              window.removeEventListener("mouseup", onUp);
            }};
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup", onUp);
          }});
        }});
      }}

      function placePanel(panel, x, y) {{
        const maxX = workspace.clientWidth - panel.offsetWidth;
        const maxY = workspace.clientHeight - panel.offsetHeight;
        const snapped = snapPanelPosition(
          panel,
          Math.max(0, Math.min(x, maxX)),
          Math.max(0, Math.min(y, maxY)),
        );
        panel.style.left = `${{snapped.x}}px`;
        panel.style.top = `${{snapped.y}}px`;
      }}

      function applyPanelLayout(panel, config) {{
        panel.style.left = `${{config.x}}px`;
        panel.style.top = `${{config.y}}px`;
        panel.style.width = `${{config.w}}px`;
        panel.style.height = `${{config.h}}px`;
        panel.style.zIndex = String(config.z);
        highestZ = Math.max(highestZ, config.z);
      }}

      function panelRect(panel) {{
        return {{
          x: parseFloat(panel.style.left || "0"),
          y: parseFloat(panel.style.top || "0"),
          w: panel.offsetWidth,
          h: panel.offsetHeight,
        }};
      }}

      function applyPanelRect(panel, rect) {{
        panel.style.left = `${{rect.x}}px`;
        panel.style.top = `${{rect.y}}px`;
        panel.style.width = `${{rect.w}}px`;
        panel.style.height = `${{rect.h}}px`;
      }}

      function bringToFront(panel) {{
        highestZ += 1;
        panel.style.zIndex = String(highestZ);
      }}

      function saveLayout() {{
        const layout = {{}};
        document.querySelectorAll(".panel[data-panel-id]").forEach((panel) => {{
          const rect = panel.getBoundingClientRect();
          const workspaceRect = workspace.getBoundingClientRect();
          layout[panel.dataset.panelId] = {{
            x: rect.left - workspaceRect.left,
            y: rect.top - workspaceRect.top,
            w: rect.width,
            h: rect.height,
            z: Number(panel.style.zIndex || 1),
          }};
        }});
        window.localStorage.setItem(layoutStorageKey, JSON.stringify(layout));
      }}

      function loadLayout() {{
        try {{
          const raw = window.localStorage.getItem(layoutStorageKey);
          return raw ? JSON.parse(raw) : defaultPanelLayout;
        }} catch (_error) {{
          return defaultPanelLayout;
        }}
      }}

      function clampAllPanels() {{
        document.querySelectorAll(".panel[data-panel-id]").forEach((panel) => {{
          normalizePanelSize(panel);
          placePanel(panel, parseFloat(panel.style.left || "0"), parseFloat(panel.style.top || "0"));
        }});
      }}

      function snapPanelPosition(panel, x, y) {{
        let snappedX = snapToGrid(x);
        let snappedY = snapToGrid(y);
        const panelWidth = panel.offsetWidth;
        const panelHeight = panel.offsetHeight;
        const workspaceWidth = workspace.clientWidth;
        const workspaceHeight = workspace.clientHeight;

        const workspaceEdgesX = [0, workspaceWidth - panelWidth];
        const workspaceEdgesY = [0, workspaceHeight - panelHeight];
        workspaceEdgesX.forEach((edge) => {{
          if (Math.abs(x - edge) <= SNAP_THRESHOLD) snappedX = edge;
        }});
        workspaceEdgesY.forEach((edge) => {{
          if (Math.abs(y - edge) <= SNAP_THRESHOLD) snappedY = edge;
        }});

        document.querySelectorAll(".panel[data-panel-id]").forEach((other) => {{
          if (other === panel) return;
          const ox = parseFloat(other.style.left || "0");
          const oy = parseFloat(other.style.top || "0");
          const ow = other.offsetWidth;
          const oh = other.offsetHeight;
          const candidateXs = [ox, ox + ow, ox - panelWidth, ox + ow - panelWidth];
          const candidateYs = [oy, oy + oh, oy - panelHeight, oy + oh - panelHeight];

          candidateXs.forEach((candidate) => {{
            if (Math.abs(x - candidate) <= SNAP_THRESHOLD) snappedX = candidate;
          }});
          candidateYs.forEach((candidate) => {{
            if (Math.abs(y - candidate) <= SNAP_THRESHOLD) snappedY = candidate;
          }});
        }});

        snappedX = Math.max(0, Math.min(snappedX, workspaceWidth - panelWidth));
        snappedY = Math.max(0, Math.min(snappedY, workspaceHeight - panelHeight));
        return {{ x: snappedX, y: snappedY }};
      }}

      function snapToGrid(value) {{
        return Math.round(value / GRID_SIZE) * GRID_SIZE;
      }}

      function resizeRectFromDirection(startRect, direction, dx, dy) {{
        let left = startRect.x;
        let top = startRect.y;
        let right = startRect.x + startRect.w;
        let bottom = startRect.y + startRect.h;

        if (direction.includes("e")) right += dx;
        if (direction.includes("w")) left += dx;
        if (direction.includes("s")) bottom += dy;
        if (direction.includes("n")) top += dy;

        if (right - left < MIN_PANEL_WIDTH) {{
          if (direction.includes("w")) {{
            left = right - MIN_PANEL_WIDTH;
          }} else {{
            right = left + MIN_PANEL_WIDTH;
          }}
        }}

        if (bottom - top < MIN_PANEL_HEIGHT) {{
          if (direction.includes("n")) {{
            top = bottom - MIN_PANEL_HEIGHT;
          }} else {{
            bottom = top + MIN_PANEL_HEIGHT;
          }}
        }}

        return clampPanelRect({{
          x: left,
          y: top,
          w: right - left,
          h: bottom - top,
        }});
      }}

      function clampPanelRect(rect) {{
        let x = rect.x;
        let y = rect.y;
        let w = rect.w;
        let h = rect.h;
        const maxWidth = workspace.clientWidth;
        const maxHeight = workspace.clientHeight;

        w = Math.max(MIN_PANEL_WIDTH, Math.min(w, maxWidth));
        h = Math.max(MIN_PANEL_HEIGHT, Math.min(h, maxHeight));
        x = Math.max(0, Math.min(x, workspace.clientWidth - w));
        y = Math.max(0, Math.min(y, workspace.clientHeight - h));
        if (x + w > workspace.clientWidth) x = workspace.clientWidth - w;
        if (y + h > workspace.clientHeight) y = workspace.clientHeight - h;

        return {{ x, y, w, h }};
      }}

      function snapValue(value, candidates) {{
        let best = snapToGrid(value);
        let bestDistance = Math.abs(value - best);
        candidates.forEach((candidate) => {{
          const distance = Math.abs(value - candidate);
          if (distance <= SNAP_THRESHOLD && distance <= bestDistance) {{
            best = candidate;
            bestDistance = distance;
          }}
        }});
        return best;
      }}

      function snapPanelRect(panel, rect, direction) {{
        let next = clampPanelRect(rect);
        let left = next.x;
        let top = next.y;
        let right = next.x + next.w;
        let bottom = next.y + next.h;

        const otherPanels = Array.from(document.querySelectorAll(".panel[data-panel-id]")).filter(
          (other) => other !== panel,
        );
        const leftCandidates = [0];
        const rightCandidates = [workspace.clientWidth];
        const topCandidates = [0];
        const bottomCandidates = [workspace.clientHeight];
        const widthCandidates = [];
        const heightCandidates = [];

        otherPanels.forEach((other) => {{
          const rect = panelRect(other);
          leftCandidates.push(rect.x, rect.x + rect.w);
          rightCandidates.push(rect.x, rect.x + rect.w);
          topCandidates.push(rect.y, rect.y + rect.h);
          bottomCandidates.push(rect.y, rect.y + rect.h);
          widthCandidates.push(rect.w);
          heightCandidates.push(rect.h);
        }});

        if (direction.includes("e")) {{
          right = snapValue(right, rightCandidates);
          let width = Math.max(MIN_PANEL_WIDTH, right - left);
          width = snapValue(width, widthCandidates);
          right = Math.min(workspace.clientWidth, left + width);
        }}
        if (direction.includes("w")) {{
          left = snapValue(left, leftCandidates);
          let width = Math.max(MIN_PANEL_WIDTH, right - left);
          width = snapValue(width, widthCandidates);
          left = Math.max(0, right - width);
        }}
        if (direction.includes("s")) {{
          bottom = snapValue(bottom, bottomCandidates);
          let height = Math.max(MIN_PANEL_HEIGHT, bottom - top);
          height = snapValue(height, heightCandidates);
          bottom = Math.min(workspace.clientHeight, top + height);
        }}
        if (direction.includes("n")) {{
          top = snapValue(top, topCandidates);
          let height = Math.max(MIN_PANEL_HEIGHT, bottom - top);
          height = snapValue(height, heightCandidates);
          top = Math.max(0, bottom - height);
        }}

        return clampPanelRect({{
          x: left,
          y: top,
          w: right - left,
          h: bottom - top,
        }});
      }}

      function normalizePanelSize(panel) {{
        const maxWidth = workspace.clientWidth;
        const maxHeight = workspace.clientHeight;
        const snappedWidth = Math.max(MIN_PANEL_WIDTH, Math.min(snapToGrid(panel.offsetWidth), maxWidth));
        const snappedHeight = Math.max(MIN_PANEL_HEIGHT, Math.min(snapToGrid(panel.offsetHeight), maxHeight));
        panel.style.width = `${{snappedWidth}}px`;
        panel.style.height = `${{snappedHeight}}px`;
      }}

      function renderSelection(selection) {{
        const regime = dashboard.regimes[selection.regime];
        const resultA = regime.strategies[selection.strategyA];
        const resultB = regime.strategies[selection.strategyB];
        scenarioTitle.textContent = regime.title;
        scenarioDescription.textContent = regime.description;
        terminalSeed.textContent = String(selection.seed);
        terminalEvents.textContent = String(selection.eventCount);
        metricGrid.innerHTML = [
          metricCard(resultA.strategy_title, resultA.metrics, false),
          metricCard(resultB.strategy_title, resultB.metrics, true),
        ].join("");
        playbackState = {{
          selection,
          regime,
          resultA,
          resultB,
          cursor: 0,
        }};
        renderPlaybackFrame();
        startPlayback();
      }}

      function startPlayback() {{
        if (playbackTimer) {{
          clearInterval(playbackTimer);
        }}
        playbackTimer = window.setInterval(() => {{
          if (!playbackState) return;
          const maxRows = Math.max(playbackState.resultA.rows.length, playbackState.resultB.rows.length);
          playbackState.cursor = (playbackState.cursor + 1) % Math.max(maxRows, 1);
          renderPlaybackFrame();
        }}, PLAYBACK_INTERVAL_MS);
      }}

      function renderPlaybackFrame() {{
        if (!playbackState) return;
        const {{ selection, regime, resultA, resultB, cursor }} = playbackState;
        const rowsA = rollingRows(resultA.rows, cursor, CHART_WINDOW);
        const rowsB = rollingRows(resultB.rows, cursor, CHART_WINDOW);
        const liveIndexA = Math.min(cursor, resultA.rows.length - 1);
        const liveRowA = resultA.rows[liveIndexA];
        terminalCursor.textContent = `${{liveIndexA + 1}} / ${{resultA.rows.length}}`;
        terminalStatus.textContent = "Streaming Playback";
        timelineBody.innerHTML = rollingRows(resultA.rows, cursor, TABLE_WINDOW).map((row) => `
          <tr>
            <td>${{row.ts}}</td>
            <td>${{row.event_type}}</td>
            <td class="${{toneClass((row.best_bid ?? 0) - (row.best_ask ?? row.best_bid ?? 0))}}">${{row.best_bid ?? ""}}</td>
            <td>${{row.best_ask ?? ""}}</td>
            <td class="${{toneClass(row.position)}}">${{row.position}}</td>
            <td class="${{toneClass(row.cash)}}">${{row.cash.toFixed(2)}}</td>
            <td class="${{toneClass(row.equity)}}">${{row.equity.toFixed(2)}}</td>
          </tr>
        `).join("");
        tracePanel.textContent = buildRollingTrace(
          resultA.strategy_title,
          resultA.trace,
          resultB.strategy_title,
          resultB.trace,
          cursor,
          resultA.rows.length,
          resultB.rows.length,
        );
        chartCaption.textContent = `${{resultA.strategy_title}} vs ${{resultB.strategy_title}} | ${{regime.title}} | seed ${{selection.seed}} | live t=${{liveRowA.ts}} | row ${{liveIndexA + 1}}`;
        drawChart("equity-position-canvas", [
          {{ rows: rowsA, key: "equity", color: "#41d392" }},
          {{ rows: rowsB, key: "equity", color: "#8affd1" }},
          {{ rows: rowsA, key: "position", color: "#eb5a52" }},
          {{ rows: rowsB, key: "position", color: "#ff998d" }},
        ]);
        drawChart("spread-canvas", [
          {{ rows: rowsA, key: "spread", color: "#d1b278" }},
        ]);
        drawChart("cash-canvas", [
          {{ rows: rowsA, key: "cash", color: "#8b79d8" }},
          {{ rows: rowsB, key: "cash", color: "#5b9bd5" }},
        ]);
      }}

      function rollingRows(rows, cursor, size) {{
        const end = Math.min(cursor + 1, rows.length);
        const start = Math.max(0, end - size);
        return rows.slice(start, end);
      }}

      function buildRollingTrace(titleA, traceA, titleB, traceB, cursor, rowCountA, rowCountB) {{
        const endA = Math.max(1, Math.min(traceA.length, Math.floor(((cursor + 1) / Math.max(rowCountA, 1)) * traceA.length)));
        const endB = Math.max(1, Math.min(traceB.length, Math.floor(((cursor + 1) / Math.max(rowCountB, 1)) * traceB.length)));
        const tailA = traceA.slice(Math.max(0, endA - TRACE_WINDOW), endA);
        const tailB = traceB.slice(Math.max(0, endB - TRACE_WINDOW), endB);
        return [
          `=== ${{titleA}} ===`,
          ...tailA,
          "",
          `=== ${{titleB}} ===`,
          ...tailB,
        ].join("\\n");
      }}

      function toneClass(value) {{
        if (value > 0) return "pos";
        if (value < 0) return "neg";
        return "neu";
      }}

      function drawChart(canvasId, series) {{
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext("2d");
        const width = canvas.width;
        const height = canvas.height;
        const margin = {{ left: 44, right: 18, top: 20, bottom: 28 }};
        const xs = series.flatMap((item) => item.rows.map((row) => row.ts));
        const values = series.flatMap((item) =>
          item.rows.map((row) => row[item.key]).filter((value) => value !== null)
        );
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        let minY = Math.min(...values);
        let maxY = Math.max(...values);
        if (minY === maxY) {{
          minY -= 1;
          maxY += 1;
        }}

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#090b0e";
        ctx.fillRect(0, 0, width, height);
        ctx.strokeStyle = "#232830";
        ctx.strokeRect(margin.left, margin.top, width - margin.left - margin.right, height - margin.top - margin.bottom);

        ctx.font = "11px SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "#6d7786";
        for (let i = 0; i < 5; i += 1) {{
          const ratio = i / 4;
          const y = margin.top + ratio * (height - margin.top - margin.bottom);
          const value = maxY - ratio * (maxY - minY);
          ctx.strokeStyle = "#171b20";
          ctx.beginPath();
          ctx.moveTo(margin.left, y);
          ctx.lineTo(width - margin.right, y);
          ctx.stroke();
          ctx.fillText(value.toFixed(1), 6, y + 4);
        }}

        [...new Set(xs)].forEach((x) => {{
          const px = scale(x, minX, maxX, margin.left, width - margin.right);
          ctx.fillStyle = "#6d7786";
          ctx.fillText(String(x), px - 6, height - 8);
        }});

        series.forEach((item) => {{
          ctx.strokeStyle = item.color;
          ctx.lineWidth = 3;
          ctx.beginPath();
          let started = false;
          item.rows.forEach((row) => {{
            const value = row[item.key];
            if (value === null) {{
              return;
            }}
            const px = scale(row.ts, minX, maxX, margin.left, width - margin.right);
            const py = scale(value, minY, maxY, height - margin.bottom, margin.top);
            if (!started) {{
              ctx.moveTo(px, py);
              started = true;
            }} else {{
              ctx.lineTo(px, py);
            }}
          }});
          ctx.stroke();
        }});
      }}

      function scale(value, min, max, start, end) {{
        if (min === max) {{
          return (start + end) / 2;
        }}
        return start + ((value - min) / (max - min)) * (end - start);
      }}

      async function requestFreshRun() {{
        if (!window.location.protocol.startsWith("http")) {{
          renderSelection(currentSelection());
          return;
        }}
        const selection = currentSelection();
        const params = new URLSearchParams({{
          regime: selection.regime,
          strategy_a: selection.strategyA,
          strategy_b: selection.strategyB,
          seed: String(selection.seed),
          event_count: String(selection.eventCount),
        }});
        const response = await fetch(`/api/run?${{params.toString()}}`);
        if (!response.ok) {{
          renderSelection(selection);
          return;
        }}
        const payload = await response.json();
        dashboard.regimes[selection.regime] = payload.regime;
        dashboard.seed = selection.seed;
        dashboard.event_count = selection.eventCount;
        renderSelection(selection);
      }}

      runButton.addEventListener("click", () => requestFreshRun());
      regimeSelect.addEventListener("change", () => renderSelection(currentSelection()));
      strategyASelect.addEventListener("change", () => renderSelection(currentSelection()));
      strategyBSelect.addEventListener("change", () => renderSelection(currentSelection()));
      initializeWorkspace();
      renderSelection(currentSelection());
    </script>
  </body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def polyline_points(
    times: list[int],
    values: list[float],
    min_y: float,
    max_y: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> str:
    min_x = min(times)
    max_x = max(times)
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    return " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (
            scale_point(time, value, min_x, max_x, min_y, max_y, left, top, width, height)
            for time, value in zip(times, values)
        )
    )


def scale_point(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[float, float]:
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    scaled_x = left + (x - min_x) / x_span * width
    scaled_y = top + height - (y - min_y) / y_span * height
    return scaled_x, scaled_y


def axis_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if min_value == max_value:
        return [min_value]
    step = (max_value - min_value) / (count - 1)
    return [min_value + idx * step for idx in range(count)]


def last_x(rows: list[SimulationRow]) -> float:
    times = [row.ts for row in rows]
    values = [row.equity for row in rows]
    x, _ = scale_point(
        times[-1],
        values[-1],
        min(times),
        max(times),
        min(values),
        max(values) if max(values) != min(values) else max(values) + 1.0,
        70,
        30,
        800,
        330,
    )
    return x


def last_y(
    rows: list[SimulationRow],
    series: str,
    min_value: float,
    max_value: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> float:
    times = [row.ts for row in rows]
    value = rows[-1].equity if series == "equity" else float(rows[-1].position)
    _, y = scale_point(
        times[-1],
        value,
        min(times),
        max(times),
        min_value,
        max_value,
        left,
        top,
        width,
        height,
    )
    return y


def blank_if_none(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def print_trace_excerpt(trace: list[str], head: int = 80, tail: int = 20) -> None:
    if len(trace) <= head + tail + 1:
        for line in trace:
            print(line)
        return

    for line in trace[:head]:
        print(line)
    omitted = len(trace) - head - tail
    print(f"... omitted {omitted} lines from the middle of the replay ...")
    for line in trace[-tail:]:
        print(line)


def build_dashboard_payload(
    matrix: dict[str, dict[str, SimulationResult]],
    seed: int = 7,
    event_count: int = 500,
) -> dict:
    regimes = regime_library()
    strategies = strategy_library()
    payload = {"regimes": {}, "strategies": {}, "seed": seed, "event_count": event_count}
    payload["strategies"] = {
        slug: {
            "title": spec.title,
            "description": spec.description,
        }
        for slug, spec in strategies.items()
    }
    for regime_slug, strategy_results in matrix.items():
        regime_spec = regimes[regime_slug]
        payload["regimes"][regime_slug] = {
            "title": regime_spec.title,
            "description": regime_spec.description,
            "strategies": {
                strategy_slug: {
                    "strategy_title": strategies[strategy_slug].title,
                    "strategy_description": strategies[strategy_slug].description,
                    "metrics": {
                        "total_pnl": sim_result.metrics.total_pnl,
                        "sharpe": sim_result.metrics.sharpe,
                        "max_drawdown": sim_result.metrics.max_drawdown,
                        "num_fills": sim_result.metrics.num_fills,
                        "ending_position": sim_result.metrics.ending_position,
                    },
                    "rows": [
                        {
                            "ts": row.ts,
                            "event_type": row.event_type,
                            "best_bid": row.best_bid,
                            "best_ask": row.best_ask,
                            "position": row.position,
                            "cash": row.cash,
                            "equity": row.equity,
                            "spread": None if row.best_bid is None or row.best_ask is None else row.best_ask - row.best_bid,
                        }
                        for row in sim_result.rows
                    ],
                    "trace": sim_result.trace,
                }
                for strategy_slug, sim_result in strategy_results.items()
            },
        }
    return payload


def api_run_payload(
    regime_slug: str,
    strategy_slugs: list[str],
    seed: int,
    event_count: int,
) -> dict:
    matrix = run_regime_strategy_matrix(
        [regime_slug],
        strategy_slugs,
        seed=seed,
        event_count=event_count,
    )
    return {
        "regime": build_dashboard_payload(
            matrix,
            seed=seed,
            event_count=event_count,
        )["regimes"][regime_slug],
    }


def serve_dashboard(output_dir: Path, host: str, port: int) -> None:
    output_dir = output_dir.resolve()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/run":
                app_config = default_app_config()
                params = parse_qs(parsed.query)
                regime_slug = params.get("regime", [app_config.dashboard.default_regime])[0]
                strategy_a = params.get("strategy_a", [app_config.dashboard.default_strategy_a])[0]
                strategy_b = params.get("strategy_b", [app_config.dashboard.default_strategy_b])[0]
                seed = int(params.get("seed", [str(app_config.dashboard.default_seed)])[0])
                event_count = int(params.get("event_count", [str(app_config.dashboard.default_event_count)])[0])
                payload = api_run_payload(regime_slug, [strategy_a, strategy_b], seed, event_count)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            target = parsed.path.lstrip("/") or "report.html"
            file_path = output_dir / target
            if not file_path.exists() or not file_path.is_file():
                self.send_response(404)
                self.end_headers()
                return

            content_type = "text/plain; charset=utf-8"
            if file_path.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif file_path.suffix == ".svg":
                content_type = "image/svg+xml"
            elif file_path.suffix == ".csv":
                content_type = "text/csv; charset=utf-8"

            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"server_url=http://{host}:{port}/report.html")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    app_config = default_app_config()
    parser = argparse.ArgumentParser(description="Run a deterministic MVP HFT simulation replay.")
    parser.add_argument(
        "--output-dir",
        default=app_config.dashboard.output_dir,
        help="Directory for exported CSV, summary, and SVG chart artifacts.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the generated dashboard locally and enable fresh-run API requests.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print the full replay trace instead of an excerpt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=app_config.dashboard.default_seed,
        help="Deterministic seed for generated market events.",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=app_config.dashboard.default_event_count,
        help="Number of generated market events per regime, excluding initial book bootstrap events.",
    )
    parser.add_argument(
        "--host",
        default=app_config.server.host,
        help="Host for the local dashboard server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=app_config.server.port,
        help="Port for the local dashboard server.",
    )
    args = parser.parse_args()

    scenario_results = run_regime_strategy_matrix(
        app_config.dashboard.enabled_regimes,
        app_config.dashboard.enabled_strategies,
        seed=args.seed,
        event_count=args.event_count,
    )
    result = scenario_results[app_config.dashboard.default_regime][app_config.dashboard.default_strategy_a]
    artifacts = export_artifacts(result, Path(args.output_dir), scenario_results=scenario_results)

    if args.print_all:
        for line in result.trace:
            print(line)
    else:
        print_trace_excerpt(result.trace)
    print(f"artifacts timeline_csv={artifacts['timeline_csv']}")
    print(f"artifacts summary_txt={artifacts['summary_txt']}")
    print(f"artifacts chart_svg={artifacts['chart_svg']}")
    print(f"artifacts report_html={artifacts['report_html']}")
    if args.serve:
        serve_dashboard(Path(args.output_dir), args.host, args.port)


if __name__ == "__main__":
    main()

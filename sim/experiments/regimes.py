from __future__ import annotations

from dataclasses import dataclass

from sim.core.event import Event
from sim.experiments.market_gen import MarketGenConfig, generate_market_events


@dataclass(frozen=True)
class RegimeSpec:
    slug: str
    title: str
    description: str


def regime_library() -> dict[str, RegimeSpec]:
    return {
        "baseline": RegimeSpec(
            slug="baseline",
            title="Baseline Mean Reversion",
            description="The spread widens, passive inventory gets lifted, and the bid later recovers enough for a profitable exit.",
        ),
        "trend": RegimeSpec(
            slug="trend",
            title="Trend Pressure",
            description="Aggressive buy pressure repeatedly lifts the offer while higher bids gradually ratchet upward.",
        ),
        "whipsaw": RegimeSpec(
            slug="whipsaw",
            title="Whipsaw Spread Shock",
            description="The spread snaps between tight and wide states, creating noisier marks and less stable exits.",
        ),
    }


def build_regime_events(slug: str, seed: int = 7, event_count: int = 500) -> list[Event]:
    if slug not in regime_library():
        raise ValueError(f"unknown regime {slug!r}")
    return generate_market_events(MarketGenConfig(regime=slug, seed=seed, event_count=event_count))

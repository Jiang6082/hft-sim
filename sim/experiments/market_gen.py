from __future__ import annotations

from dataclasses import dataclass

from sim.core.event import Event, EventType
from sim.core.rng import DeterministicRNG
from sim.exchange.order import Side


@dataclass(frozen=True)
class MarketGenConfig:
    regime: str
    seed: int
    event_count: int
    start_mid: int = 100


def generate_market_events(config: MarketGenConfig) -> list[Event]:
    if config.event_count <= 0:
        raise ValueError("event_count must be positive")

    rng = DeterministicRNG(config.seed)
    mid = config.start_mid
    spread = 2
    ts = 0
    seq = 0
    events: list[Event] = bootstrap_book(mid=mid, spread=spread)
    seq = len(events)
    ts = events[-1].ts if events else 0

    for _ in range(config.event_count):
        drift = regime_drift(config.regime, rng)
        shock = int(round(rng.normal(0.0, regime_vol(config.regime))))
        mid = max(90, min(120, mid + drift + shock))
        spread = next_spread(config.regime, rng, spread)

        ts += rng.uniform_int(1, 3)
        event_type = next_event_type(config.regime, rng)
        payload = build_payload(config.regime, rng, event_type, mid, spread)
        events.append(Event(ts=ts, seq=seq, type=event_type, payload=payload))
        seq += 1

    return events


def bootstrap_book(mid: int, spread: int) -> list[Event]:
    bid = mid - spread // 2
    ask = bid + spread
    events: list[Event] = []
    seq = 0
    for offset in range(4):
        events.append(
            Event(
                ts=offset,
                seq=seq,
                type=EventType.EXTERNAL_ADD,
                payload={"side": Side.BUY, "price": bid - offset, "qty": 14 + 2 * offset},
            )
        )
        seq += 1
        events.append(
            Event(
                ts=offset,
                seq=seq,
                type=EventType.EXTERNAL_ADD,
                payload={"side": Side.SELL, "price": ask + offset, "qty": 14 + 2 * offset},
            )
        )
        seq += 1
    return events


def regime_drift(regime: str, rng: DeterministicRNG) -> int:
    if regime == "trend":
        return 1 if rng.random() < 0.65 else 0
    if regime == "whipsaw":
        return rng.choice([-2, -1, 1, 2])
    if regime == "baseline":
        return rng.choice([-1, 0, 1])
    raise ValueError(f"unknown regime {regime!r}")


def regime_vol(regime: str) -> float:
    if regime == "trend":
        return 0.45
    if regime == "whipsaw":
        return 1.1
    if regime == "baseline":
        return 0.35
    raise ValueError(f"unknown regime {regime!r}")


def next_spread(regime: str, rng: DeterministicRNG, current_spread: int) -> int:
    if regime == "trend":
        candidate = current_spread + rng.choice([-1, 0, 0, 1])
    elif regime == "whipsaw":
        candidate = current_spread + rng.choice([-2, -1, 2, 3])
    else:
        candidate = current_spread + rng.choice([-1, 0, 1])
    return max(1, min(6, candidate))


def next_event_type(regime: str, rng: DeterministicRNG) -> EventType:
    roll = rng.random()
    if regime == "trend":
        if roll < 0.52:
            return EventType.EXTERNAL_ADD
        if roll < 0.78:
            return EventType.EXTERNAL_MKT
        return EventType.EXTERNAL_CANCEL
    if regime == "whipsaw":
        if roll < 0.42:
            return EventType.EXTERNAL_ADD
        if roll < 0.77:
            return EventType.EXTERNAL_MKT
        return EventType.EXTERNAL_CANCEL
    if roll < 0.50:
        return EventType.EXTERNAL_ADD
    if roll < 0.74:
        return EventType.EXTERNAL_MKT
    return EventType.EXTERNAL_CANCEL


def build_payload(
    regime: str,
    rng: DeterministicRNG,
    event_type: EventType,
    mid: int,
    spread: int,
) -> dict:
    best_bid = mid - spread // 2
    best_ask = best_bid + spread
    if event_type is EventType.EXTERNAL_ADD:
        side = biased_side(regime, rng, prefer_buy=False)
        level_offset = rng.choice([0, 0, 1, 1, 2])
        price = best_bid - level_offset if side is Side.BUY else best_ask + level_offset
        return {
            "side": side,
            "price": price,
            "qty": rng.uniform_int(6, 22),
        }
    if event_type is EventType.EXTERNAL_CANCEL:
        side = biased_side(regime, rng, prefer_buy=(regime == "trend"))
        level_offset = rng.choice([0, 0, 1, 2])
        price = best_bid - level_offset if side is Side.BUY else best_ask + level_offset
        return {
            "side": side,
            "price": price,
            "qty": rng.uniform_int(3, 14),
        }

    side = biased_side(regime, rng, prefer_buy=(regime == "trend"))
    return {
        "side": side,
        "qty": rng.uniform_int(6, 24),
    }


def biased_side(regime: str, rng: DeterministicRNG, prefer_buy: bool) -> Side:
    if regime == "trend":
        buy_threshold = 0.68 if prefer_buy else 0.58
    elif regime == "whipsaw":
        buy_threshold = 0.50
    else:
        buy_threshold = 0.52
    return Side.BUY if rng.random() < buy_threshold else Side.SELL

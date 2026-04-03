from __future__ import annotations

import argparse
from collections.abc import Iterable

from sim.core.event import Event, EventType
from sim.core.queue import EventQueue
from sim.exchange.book import BookFill, LimitOrderBook
from sim.exchange.order import Side


def demo_events() -> list[Event]:
    return [
        Event(
            ts=0,
            seq=0,
            type=EventType.EXTERNAL_ADD,
            payload={"side": Side.BUY, "price": 99, "qty": 10},
        ),
        Event(
            ts=0,
            seq=1,
            type=EventType.EXTERNAL_ADD,
            payload={"side": Side.SELL, "price": 101, "qty": 8},
        ),
        Event(
            ts=10,
            seq=2,
            type=EventType.EXTERNAL_ADD,
            payload={"side": Side.BUY, "price": 100, "qty": 5},
        ),
        Event(
            ts=20,
            seq=3,
            type=EventType.EXTERNAL_ADD,
            payload={"side": Side.SELL, "price": 102, "qty": 6},
        ),
        Event(
            ts=30,
            seq=4,
            type=EventType.EXTERNAL_MKT,
            payload={"side": Side.BUY, "qty": 9},
        ),
        Event(
            ts=40,
            seq=5,
            type=EventType.EXTERNAL_CANCEL,
            payload={"side": Side.BUY, "price": 99, "qty": 4},
        ),
        Event(
            ts=50,
            seq=6,
            type=EventType.EXTERNAL_MKT,
            payload={"side": Side.SELL, "qty": 3},
        ),
    ]


def run_demo(events: Iterable[Event] | None = None) -> list[str]:
    queue = EventQueue()
    book = LimitOrderBook()
    trace: list[str] = []

    for event in events or demo_events():
        queue.push(event)

    trace.append("starting deterministic market replay")

    while len(queue) > 0:
        event = queue.pop()
        trace.append(render_event(event))

        if event.type is EventType.EXTERNAL_ADD:
            book.add_limit(
                side=event.payload["side"],
                price=event.payload["price"],
                qty=event.payload["qty"],
            )
            trace.append(
                f"  added resting {side_name(event.payload['side'])} liquidity "
                f"qty={event.payload['qty']} @ {event.payload['price']}"
            )
        elif event.type is EventType.EXTERNAL_CANCEL:
            canceled = book.cancel_limit(
                side=event.payload["side"],
                price=event.payload["price"],
                qty=event.payload["qty"],
            )
            trace.append(
                f"  canceled {canceled} from {side_name(event.payload['side'])} "
                f"level @ {event.payload['price']}"
            )
        elif event.type is EventType.EXTERNAL_MKT:
            fills = book.execute_market(
                side=event.payload["side"],
                qty=event.payload["qty"],
            )
            trace.extend(render_fills(event.payload["side"], event.payload["qty"], fills))
        else:
            trace.append("  event type not yet handled")

        trace.append(f"  book {render_book(book)}")

    trace.append("replay complete")
    return trace


def render_event(event: Event) -> str:
    return f"t={event.ts:>4} seq={event.seq:>2} {event.type.name}"


def render_fills(aggressor_side: Side, requested_qty: int, fills: list[BookFill]) -> list[str]:
    if not fills:
        return [
            f"  market {side_name(aggressor_side)} qty={requested_qty} found no liquidity"
        ]

    filled_qty = sum(fill.qty for fill in fills)
    parts = ", ".join(f"{fill.qty} @ {fill.price}" for fill in fills)
    return [
        f"  market {side_name(aggressor_side)} qty={requested_qty} filled={filled_qty}",
        f"  fills {parts}",
    ]


def render_book(book: LimitOrderBook) -> str:
    bid = book.best_bid()
    ask = book.best_ask()
    spread = "NA" if bid is None or ask is None else str(ask - bid)
    return (
        f"best_bid={format_level(bid, book.bids)} "
        f"best_ask={format_level(ask, book.asks)} "
        f"spread={spread}"
    )


def format_level(price: int | None, levels: dict[int, int]) -> str:
    if price is None:
        return "None"
    return f"{price}x{levels[price]}"


def side_name(side: Side) -> str:
    return "buy" if side is Side.BUY else "sell"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic text replay of the HFT simulator.")
    parser.parse_args()

    for line in run_demo():
        print(line)


if __name__ == "__main__":
    main()

from sim.experiments.runner import run_demo


def test_run_demo_emits_human_readable_trace() -> None:
    trace = run_demo()
    joined = "\n".join(trace)

    assert trace[0] == "starting deterministic market replay"
    assert trace[-1] == "replay complete"
    assert "EXTERNAL_ADD" in joined
    assert "EXTERNAL_MKT" in joined
    assert "fills 8 @ 101, 1 @ 102" in joined
    assert "book best_bid=100x2 best_ask=102x5 spread=2" in joined

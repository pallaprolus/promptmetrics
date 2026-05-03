from __future__ import annotations

from datetime import datetime, timedelta, timezone

from promptdrift.models import Baseline
from tests.conftest import make_trace


def test_insert_and_fetch_trace(storage):
    t = make_trace(prompt_id="p1", latency_ms=123.4)
    tid = storage.insert_trace(t)
    assert tid > 0
    rows = storage.get_traces("p1")
    assert len(rows) == 1
    assert rows[0].latency_ms == 123.4
    assert rows[0].prompt_id == "p1"


def test_get_traces_respects_time_window(storage):
    now = datetime.now(timezone.utc)
    storage.insert_trace(make_trace(timestamp=now - timedelta(hours=2)))
    storage.insert_trace(make_trace(timestamp=now - timedelta(minutes=5)))
    recent = storage.get_traces("p1", since=now - timedelta(hours=1))
    assert len(recent) == 1


def test_save_baseline_marks_previous_inactive(storage):
    samples = [100.0, 110.0, 120.0]
    b1 = Baseline(
        prompt_id="p1",
        sample_size=3,
        latency_mean=110,
        latency_p50=110,
        latency_p95=120,
        latency_p99=120,
        input_tokens_mean=10,
        output_tokens_mean=10,
        total_tokens_mean=20,
        latency_samples=samples,
    )
    storage.save_baseline(b1)
    b2 = Baseline(
        prompt_id="p1",
        sample_size=4,
        latency_mean=200,
        latency_p50=200,
        latency_p95=210,
        latency_p99=220,
        input_tokens_mean=20,
        output_tokens_mean=20,
        total_tokens_mean=40,
        latency_samples=samples + [200.0],
    )
    storage.save_baseline(b2)
    active = storage.get_active_baseline("p1")
    assert active is not None
    assert active.sample_size == 4
    assert active.latency_samples[-1] == 200.0


def test_list_prompt_ids(storage):
    storage.insert_trace(make_trace(prompt_id="a"))
    storage.insert_trace(make_trace(prompt_id="b"))
    storage.insert_trace(make_trace(prompt_id="a"))
    assert storage.list_prompt_ids() == ["a", "b"]

from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import patch

import promptmetrics.decorator as decorator
from promptmetrics import track
from promptmetrics.decorator import set_storage


def test_track_records_trace(storage):
    set_storage(storage)

    @track("hello")
    def call(prompt: str) -> str:
        return prompt + "!"

    out = call("hi")
    assert out == "hi!"
    rows = storage.get_traces("hello")
    assert len(rows) == 1
    assert rows[0].input == "hi"
    assert rows[0].output == "hi!"
    assert rows[0].latency_ms >= 0


def test_track_extracts_openai_style_usage(storage):
    set_storage(storage)

    @dataclass
    class Usage:
        prompt_tokens: int
        completion_tokens: int

    @dataclass
    class Resp:
        text: str
        usage: Usage

        def __str__(self) -> str:
            return self.text

    @track("hello", model="gpt-x")
    def call(prompt: str) -> Resp:
        return Resp(text="ok", usage=Usage(prompt_tokens=12, completion_tokens=7))

    call("hi")
    rows = storage.get_traces("hello")
    assert rows[0].input_tokens == 12
    assert rows[0].output_tokens == 7
    assert rows[0].model == "gpt-x"


def test_track_does_not_swallow_function_errors(storage):
    set_storage(storage)

    @track("hello")
    def boom() -> str:
        raise ValueError("nope")

    try:
        boom()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError to propagate")

    # The wrapped function raised, so no trace should have been recorded.
    assert storage.get_traces("hello") == []


def test_get_storage_is_thread_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(decorator, "_storage", None)

    construction_count = 0
    real_storage_cls = decorator.Storage

    def counting_storage(*args, **kwargs):
        nonlocal construction_count
        construction_count += 1
        return real_storage_cls(tmp_path / "thread.db")

    with patch.object(decorator, "Storage", side_effect=counting_storage):
        instances = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            instances.append(decorator._get_storage())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert construction_count == 1
    assert all(i is instances[0] for i in instances)


def test_track_records_traces_under_concurrency(storage):
    set_storage(storage)

    @track("hello")
    def call(prompt: str) -> str:
        return prompt + "!"

    threads = [
        threading.Thread(target=call, args=(f"msg-{i}",)) for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(storage.get_traces("hello")) == 20


def test_track_swallows_extractor_errors_by_default(storage, caplog):
    set_storage(storage)

    @track("hello", extract_tokens=lambda r: 1 / 0)
    def call() -> str:
        return "ok"

    with caplog.at_level("ERROR"):
        out = call()

    assert out == "ok"
    rows = storage.get_traces("hello")
    assert len(rows) == 1
    assert rows[0].input_tokens == 0
    assert rows[0].output_tokens == 0
    assert any("token extractor raised" in r.message for r in caplog.records)


def test_track_raises_on_extractor_error_when_strict(storage):
    set_storage(storage)

    @track("hello", extract_tokens=lambda r: 1 / 0, raise_on_error=True)
    def call() -> str:
        return "ok"

    try:
        call()
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected ZeroDivisionError to propagate")

    assert storage.get_traces("hello") == []


def test_track_uses_custom_extractors(storage):
    set_storage(storage)

    @track(
        "hello",
        extract_output=lambda r: r["text"],
        extract_tokens=lambda r: (r["in"], r["out"]),
    )
    def call() -> dict:
        return {"text": "x", "in": 3, "out": 5}

    call()
    rows = storage.get_traces("hello")
    assert rows[0].output == "x"
    assert rows[0].input_tokens == 3
    assert rows[0].output_tokens == 5

from __future__ import annotations

from dataclasses import dataclass

from promptdrift import track
from promptdrift.decorator import set_storage


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

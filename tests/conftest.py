from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from promptdrift.core import PromptDrift
from promptdrift.models import Trace
from promptdrift.storage import Storage


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def storage(db_path):
    s = Storage(db_path)
    yield s
    s.close()


@pytest.fixture
def pd_instance(db_path):
    p = PromptDrift(db_path)
    yield p
    p.close()


def make_trace(
    prompt_id: str = "p1",
    latency_ms: float = 100.0,
    input_tokens: int = 50,
    output_tokens: int = 50,
    timestamp: datetime | None = None,
) -> Trace:
    return Trace(
        prompt_id=prompt_id,
        input="x",
        output="y",
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


def seed_normal_traces(
    storage: Storage,
    prompt_id: str = "p1",
    n: int = 100,
    latency_mean: float = 200.0,
    latency_sd: float = 30.0,
    tokens_mean: int = 100,
    age_minutes_back: int = 0,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc) - timedelta(minutes=age_minutes_back)
    for i in range(n):
        latency = max(1.0, float(rng.normal(latency_mean, latency_sd)))
        tokens = max(1, int(rng.normal(tokens_mean, tokens_mean * 0.1)))
        ts = now - timedelta(seconds=n - i)
        storage.insert_trace(
            make_trace(
                prompt_id=prompt_id,
                latency_ms=latency,
                input_tokens=tokens // 2,
                output_tokens=tokens - tokens // 2,
                timestamp=ts,
            )
        )

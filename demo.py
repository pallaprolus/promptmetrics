"""Simulate a drifted prompt so reviewers can try the CLI without an LLM.

Run:
    python demo.py
    promptdrift baseline demo --db ./demo.db --window 24
    promptdrift check demo --db ./demo.db --window 1
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from promptdrift.models import Trace
from promptdrift.storage import Storage

DB_PATH = Path("./demo.db")
PROMPT_ID = "demo"


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    storage = Storage(DB_PATH)
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)

    # Healthy history: 300 traces over the past 24h.
    for i in range(300):
        latency = max(1.0, float(rng.normal(220, 25)))
        in_tok = max(1, int(rng.normal(60, 6)))
        out_tok = max(1, int(rng.normal(40, 5)))
        ts = now - timedelta(hours=24) + timedelta(minutes=i * (24 * 60 / 300))
        storage.insert_trace(
            Trace(
                prompt_id=PROMPT_ID,
                input=f"healthy request {i}",
                output="ok",
                latency_ms=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
                model="demo-model",
                timestamp=ts,
            )
        )

    # Drifted recent window: latency and tokens both up.
    for i in range(60):
        latency = max(1.0, float(rng.normal(520, 90)))
        in_tok = max(1, int(rng.normal(110, 12)))
        out_tok = max(1, int(rng.normal(95, 10)))
        ts = now - timedelta(minutes=60 - i)
        storage.insert_trace(
            Trace(
                prompt_id=PROMPT_ID,
                input=f"recent request {i}",
                output="ok",
                latency_ms=latency,
                input_tokens=in_tok,
                output_tokens=out_tok,
                model="demo-model",
                timestamp=ts,
            )
        )

    storage.close()
    print(f"seeded {DB_PATH} with 300 healthy + 60 drifted traces for prompt_id={PROMPT_ID!r}")
    print("\nnext steps:")
    print(f"  promptdrift baseline {PROMPT_ID} --db {DB_PATH} --window 24 --min-samples 100")
    print(f"  promptdrift check    {PROMPT_ID} --db {DB_PATH} --window 1")


if __name__ == "__main__":
    main()

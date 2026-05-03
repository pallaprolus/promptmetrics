from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from promptdrift.models import Baseline, Trace

DEFAULT_DB_PATH = Path.home() / ".promptdrift" / "promptdrift.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    model TEXT,
    timestamp TEXT NOT NULL,
    loop_id TEXT,
    step_index INTEGER
);

CREATE INDEX IF NOT EXISTS idx_traces_prompt_ts
    ON traces (prompt_id, timestamp);

CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id TEXT NOT NULL,
    sample_size INTEGER NOT NULL,
    latency_mean REAL NOT NULL,
    latency_p50 REAL NOT NULL,
    latency_p95 REAL NOT NULL,
    latency_p99 REAL NOT NULL,
    input_tokens_mean REAL NOT NULL,
    output_tokens_mean REAL NOT NULL,
    total_tokens_mean REAL NOT NULL,
    latency_samples TEXT NOT NULL,
    window_hours INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_baselines_prompt_active
    ON baselines (prompt_id, is_active);
"""


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


class Storage:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- traces ---------------------------------------------------------

    def insert_trace(self, trace: Trace) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO traces (
                prompt_id, input, output, latency_ms,
                input_tokens, output_tokens, model, timestamp,
                loop_id, step_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.prompt_id,
                trace.input,
                trace.output,
                trace.latency_ms,
                trace.input_tokens,
                trace.output_tokens,
                trace.model,
                _to_iso(trace.timestamp),
                trace.loop_id,
                trace.step_index,
            ),
        )
        self._conn.commit()
        trace.id = cur.lastrowid
        return cur.lastrowid

    def get_traces(
        self,
        prompt_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[Trace]:
        query = "SELECT * FROM traces WHERE prompt_id = ?"
        params: list[object] = [prompt_id]
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(_to_iso(since))
        if until is not None:
            query += " AND timestamp <= ?"
            params.append(_to_iso(until))
        query += " ORDER BY timestamp ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_trace(r) for r in rows]

    def list_prompt_ids(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT prompt_id FROM traces ORDER BY prompt_id"
        ).fetchall()
        return [r["prompt_id"] for r in rows]

    # --- baselines ------------------------------------------------------

    def save_baseline(self, baseline: Baseline) -> int:
        with self._conn:
            self._conn.execute(
                "UPDATE baselines SET is_active = 0 WHERE prompt_id = ?",
                (baseline.prompt_id,),
            )
            cur = self._conn.execute(
                """
                INSERT INTO baselines (
                    prompt_id, sample_size,
                    latency_mean, latency_p50, latency_p95, latency_p99,
                    input_tokens_mean, output_tokens_mean, total_tokens_mean,
                    latency_samples, window_hours, created_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    baseline.prompt_id,
                    baseline.sample_size,
                    baseline.latency_mean,
                    baseline.latency_p50,
                    baseline.latency_p95,
                    baseline.latency_p99,
                    baseline.input_tokens_mean,
                    baseline.output_tokens_mean,
                    baseline.total_tokens_mean,
                    json.dumps(baseline.latency_samples),
                    baseline.window_hours,
                    _to_iso(baseline.created_at),
                ),
            )
        baseline.id = cur.lastrowid
        baseline.is_active = True
        return cur.lastrowid

    def get_active_baseline(self, prompt_id: str) -> Baseline | None:
        row = self._conn.execute(
            """
            SELECT * FROM baselines
            WHERE prompt_id = ? AND is_active = 1
            ORDER BY created_at DESC LIMIT 1
            """,
            (prompt_id,),
        ).fetchone()
        return _row_to_baseline(row) if row else None


def _row_to_trace(row: sqlite3.Row) -> Trace:
    return Trace(
        id=row["id"],
        prompt_id=row["prompt_id"],
        input=row["input"],
        output=row["output"],
        latency_ms=row["latency_ms"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        model=row["model"],
        timestamp=_from_iso(row["timestamp"]),
        loop_id=row["loop_id"],
        step_index=row["step_index"],
    )


def _row_to_baseline(row: sqlite3.Row) -> Baseline:
    return Baseline(
        id=row["id"],
        prompt_id=row["prompt_id"],
        sample_size=row["sample_size"],
        latency_mean=row["latency_mean"],
        latency_p50=row["latency_p50"],
        latency_p95=row["latency_p95"],
        latency_p99=row["latency_p99"],
        input_tokens_mean=row["input_tokens_mean"],
        output_tokens_mean=row["output_tokens_mean"],
        total_tokens_mean=row["total_tokens_mean"],
        latency_samples=json.loads(row["latency_samples"]),
        window_hours=row["window_hours"],
        created_at=_from_iso(row["created_at"]),
        is_active=bool(row["is_active"]),
    )


def window_since(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)

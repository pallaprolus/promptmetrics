from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from promptmetrics.detectors import detect_cost_drift, detect_latency_drift
from promptmetrics.models import Baseline, DriftReport, Trace
from promptmetrics.storage import Storage, window_since


class InsufficientDataError(RuntimeError):
    pass


class StaleBaselineError(RuntimeError):
    pass


DEFAULT_MAX_BASELINE_AGE_DAYS = 30


class PromptMetrics:
    """High-level facade combining storage + detectors."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.storage = Storage(db_path)

    def close(self) -> None:
        self.storage.close()

    def __enter__(self) -> "PromptMetrics":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- baselines ------------------------------------------------------

    def capture_baseline(
        self,
        prompt_id: str,
        window_hours: int = 168,
        min_samples: int = 30,
    ) -> Baseline:
        traces = self.storage.get_traces(
            prompt_id, since=window_since(window_hours)
        )
        if len(traces) < min_samples:
            raise InsufficientDataError(
                f"need >= {min_samples} traces in last {window_hours}h, "
                f"got {len(traces)}"
            )
        baseline = _baseline_from_traces(prompt_id, traces, window_hours)
        self.storage.save_baseline(baseline)
        return baseline

    def get_baseline(self, prompt_id: str) -> Baseline | None:
        return self.storage.get_active_baseline(prompt_id)

    # --- drift checks ---------------------------------------------------

    def check_drift(
        self,
        prompt_id: str,
        window_hours: int = 24,
        max_baseline_age_days: int | None = DEFAULT_MAX_BASELINE_AGE_DAYS,
    ) -> DriftReport:
        baseline = self.storage.get_active_baseline(prompt_id)
        if baseline is None:
            raise InsufficientDataError(
                f"no active baseline for prompt_id={prompt_id!r}; "
                f"run capture_baseline first"
            )
        if max_baseline_age_days is not None:
            age_days = (
                datetime.now(timezone.utc) - baseline.created_at
            ).total_seconds() / 86400
            if age_days > max_baseline_age_days:
                raise StaleBaselineError(
                    f"baseline for {prompt_id!r} is {age_days:.1f} days old "
                    f"(> max_baseline_age_days={max_baseline_age_days}); "
                    f"re-run capture_baseline, or pass "
                    f"max_baseline_age_days=None to disable the check"
                )
        traces = self.storage.get_traces(
            prompt_id, since=window_since(window_hours)
        )
        latencies = [t.latency_ms for t in traces]
        totals = [t.total_tokens for t in traces]
        results = [
            detect_latency_drift(latencies, baseline),
            detect_cost_drift(totals, baseline),
        ]
        return DriftReport(
            prompt_id=prompt_id,
            window_hours=window_hours,
            sample_size=len(traces),
            baseline_id=baseline.id,
            results=results,
        )

    # --- introspection --------------------------------------------------

    def list_prompts(self) -> list[str]:
        return self.storage.list_prompt_ids()


def _baseline_from_traces(
    prompt_id: str, traces: list[Trace], window_hours: int
) -> Baseline:
    latencies = np.asarray([t.latency_ms for t in traces], dtype=float)
    in_tokens = np.asarray([t.input_tokens for t in traces], dtype=float)
    out_tokens = np.asarray([t.output_tokens for t in traces], dtype=float)
    return Baseline(
        prompt_id=prompt_id,
        sample_size=len(traces),
        latency_mean=float(latencies.mean()),
        latency_p50=float(np.percentile(latencies, 50)),
        latency_p95=float(np.percentile(latencies, 95)),
        latency_p99=float(np.percentile(latencies, 99)),
        input_tokens_mean=float(in_tokens.mean()),
        output_tokens_mean=float(out_tokens.mean()),
        total_tokens_mean=float((in_tokens + out_tokens).mean()),
        latency_samples=latencies.tolist(),
        window_hours=window_hours,
    )

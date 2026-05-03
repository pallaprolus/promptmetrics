from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Severity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    DRIFTED = "drifted"


class DriftType(str, Enum):
    LATENCY = "latency"
    COST = "cost"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Trace:
    prompt_id: str
    input: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    id: int | None = None
    # v0.2 hooks: agent-loop drift. Unused in v0.1 but stored so the
    # schema doesn't need a migration when LoopTrace lands.
    loop_id: str | None = None
    step_index: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Baseline:
    prompt_id: str
    sample_size: int
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    input_tokens_mean: float
    output_tokens_mean: float
    total_tokens_mean: float
    # Raw latency samples kept so KS tests can run against the baseline
    # distribution, not just summary stats.
    latency_samples: list[float] = field(default_factory=list)
    window_hours: int = 24
    created_at: datetime = field(default_factory=_utcnow)
    id: int | None = None
    is_active: bool = True


@dataclass
class DriftResult:
    drift_type: DriftType
    severity: Severity
    score: float
    threshold: float
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DriftReport:
    prompt_id: str
    window_hours: int
    sample_size: int
    baseline_id: int | None
    results: list[DriftResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)

    @property
    def severity(self) -> Severity:
        if any(r.severity == Severity.DRIFTED for r in self.results):
            return Severity.DRIFTED
        if any(r.severity == Severity.WARNING for r in self.results):
            return Severity.WARNING
        return Severity.OK

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from llmradar.models import Baseline, DriftResult, DriftType, Severity

# Two thresholds give us OK / WARNING / DRIFTED. The defaults match
# the spec: 30% drift on p95 latency.
DEFAULT_P95_RATIO_WARNING = 1.15
DEFAULT_P95_RATIO_DRIFTED = 1.30
# KS p-values: lower means the two distributions look more different.
DEFAULT_KS_P_WARNING = 0.05
DEFAULT_KS_P_DRIFTED = 0.01

MIN_SAMPLE_SIZE = 5


def detect_latency_drift(
    recent_latencies: Sequence[float],
    baseline: Baseline,
    *,
    p95_ratio_warning: float = DEFAULT_P95_RATIO_WARNING,
    p95_ratio_drifted: float = DEFAULT_P95_RATIO_DRIFTED,
    ks_p_warning: float = DEFAULT_KS_P_WARNING,
    ks_p_drifted: float = DEFAULT_KS_P_DRIFTED,
) -> DriftResult:
    recent = np.asarray(list(recent_latencies), dtype=float)
    if recent.size < MIN_SAMPLE_SIZE or len(baseline.latency_samples) < MIN_SAMPLE_SIZE:
        return DriftResult(
            drift_type=DriftType.LATENCY,
            severity=Severity.OK,
            score=0.0,
            threshold=p95_ratio_drifted,
            detail=(
                f"insufficient samples (recent={recent.size}, "
                f"baseline={len(baseline.latency_samples)}); "
                f"need >= {MIN_SAMPLE_SIZE}"
            ),
            metrics={
                "recent_n": float(recent.size),
                "baseline_n": float(len(baseline.latency_samples)),
            },
        )

    recent_p95 = float(np.percentile(recent, 95))
    p95_ratio = (
        recent_p95 / baseline.latency_p95 if baseline.latency_p95 > 0 else 1.0
    )

    ks_stat, ks_p = stats.ks_2samp(recent, np.asarray(baseline.latency_samples))
    ks_p = float(ks_p)
    ks_stat = float(ks_stat)

    # Severity is the worst of the two checks. The KS test only counts
    # when the recent window is *worse* — a faster system shouldn't
    # trigger an alert.
    severity = Severity.OK
    direction_worse = recent_p95 >= baseline.latency_p95
    if p95_ratio >= p95_ratio_drifted or (
        direction_worse and ks_p < ks_p_drifted
    ):
        severity = Severity.DRIFTED
    elif p95_ratio >= p95_ratio_warning or (
        direction_worse and ks_p < ks_p_warning
    ):
        severity = Severity.WARNING

    detail = (
        f"recent p95={recent_p95:.1f}ms vs baseline p95="
        f"{baseline.latency_p95:.1f}ms (ratio={p95_ratio:.2f}); "
        f"KS stat={ks_stat:.3f} p={ks_p:.4f}"
    )

    return DriftResult(
        drift_type=DriftType.LATENCY,
        severity=severity,
        score=p95_ratio,
        threshold=p95_ratio_drifted,
        detail=detail,
        metrics={
            "recent_p95_ms": recent_p95,
            "baseline_p95_ms": baseline.latency_p95,
            "p95_ratio": p95_ratio,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
            "recent_n": float(recent.size),
        },
    )

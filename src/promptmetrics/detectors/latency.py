from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from promptmetrics.models import Baseline, DriftResult, DriftType, Severity

# Two thresholds give us OK / WARNING / DRIFTED. The defaults match
# the spec: 30% drift on p95 latency.
DEFAULT_P95_RATIO_WARNING = 1.15
DEFAULT_P95_RATIO_DRIFTED = 1.30
# KS p-values: lower means the two distributions look more different.
# 0.05 / 0.01 are conventional alpha levels — at p<0.01 we're 99%
# confident the distributions differ; at p<0.05, 95%. Tighten via the
# ks_p_* kwargs if false alarms are expensive in your workflow.
DEFAULT_KS_P_WARNING = 0.05
DEFAULT_KS_P_DRIFTED = 0.01

# A KS test with only a few samples on either side has very low power.
# The recent floor is intentionally low (5) so a sparse production
# window still gets *some* signal; the baseline floor matches the
# 30-sample default of capture_baseline so you never compare against a
# baseline summarised from a handful of points.
MIN_RECENT_SAMPLES = 5
MIN_BASELINE_SAMPLES = 30


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
    baseline_n = len(baseline.latency_samples)
    if recent.size < MIN_RECENT_SAMPLES or baseline_n < MIN_BASELINE_SAMPLES:
        return DriftResult(
            drift_type=DriftType.LATENCY,
            severity=Severity.OK,
            score=0.0,
            threshold=p95_ratio_drifted,
            detail=(
                f"insufficient samples (recent={recent.size}, "
                f"baseline={baseline_n}); need recent >= "
                f"{MIN_RECENT_SAMPLES} and baseline >= {MIN_BASELINE_SAMPLES}"
            ),
            metrics={
                "recent_n": float(recent.size),
                "baseline_n": float(baseline_n),
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

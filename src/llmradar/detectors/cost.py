from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from llmradar.models import Baseline, DriftResult, DriftType, Severity

DEFAULT_TOKEN_RATIO_WARNING = 1.15
DEFAULT_TOKEN_RATIO_DRIFTED = 1.30
MIN_SAMPLE_SIZE = 5


def detect_cost_drift(
    recent_total_tokens: Sequence[int],
    baseline: Baseline,
    *,
    ratio_warning: float = DEFAULT_TOKEN_RATIO_WARNING,
    ratio_drifted: float = DEFAULT_TOKEN_RATIO_DRIFTED,
) -> DriftResult:
    recent = np.asarray(list(recent_total_tokens), dtype=float)
    if recent.size < MIN_SAMPLE_SIZE:
        return DriftResult(
            drift_type=DriftType.COST,
            severity=Severity.OK,
            score=0.0,
            threshold=ratio_drifted,
            detail=f"insufficient samples (recent={recent.size}); need >= {MIN_SAMPLE_SIZE}",
            metrics={"recent_n": float(recent.size)},
        )

    recent_mean = float(recent.mean())
    baseline_mean = baseline.total_tokens_mean
    if baseline_mean <= 0:
        return DriftResult(
            drift_type=DriftType.COST,
            severity=Severity.OK,
            score=0.0,
            threshold=ratio_drifted,
            detail="baseline mean tokens is zero; cannot compute ratio",
            metrics={"recent_mean_tokens": recent_mean},
        )

    ratio = recent_mean / baseline_mean
    severity = Severity.OK
    if ratio >= ratio_drifted:
        severity = Severity.DRIFTED
    elif ratio >= ratio_warning:
        severity = Severity.WARNING

    detail = (
        f"recent mean tokens={recent_mean:.1f} vs baseline="
        f"{baseline_mean:.1f} (ratio={ratio:.2f})"
    )

    return DriftResult(
        drift_type=DriftType.COST,
        severity=severity,
        score=ratio,
        threshold=ratio_drifted,
        detail=detail,
        metrics={
            "recent_mean_tokens": recent_mean,
            "baseline_mean_tokens": baseline_mean,
            "ratio": ratio,
            "recent_n": float(recent.size),
        },
    )

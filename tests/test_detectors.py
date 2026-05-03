from __future__ import annotations

import numpy as np

from promptmetrics.detectors import detect_cost_drift, detect_latency_drift
from promptmetrics.models import Baseline, Severity


def _baseline(samples: list[float], total_tokens_mean: float = 100.0) -> Baseline:
    arr = np.asarray(samples)
    return Baseline(
        prompt_id="p1",
        sample_size=len(samples),
        latency_mean=float(arr.mean()),
        latency_p50=float(np.percentile(arr, 50)),
        latency_p95=float(np.percentile(arr, 95)),
        latency_p99=float(np.percentile(arr, 99)),
        input_tokens_mean=total_tokens_mean / 2,
        output_tokens_mean=total_tokens_mean / 2,
        total_tokens_mean=total_tokens_mean,
        latency_samples=samples,
    )


def test_latency_no_drift_when_distribution_matches():
    # KS at p<0.05 is a calibrated 5% false-alarm rate; the contract for
    # this detector is "no DRIFTED on matching distributions," not "no
    # WARNING ever." A WARNING from a noisy KS sample is acceptable.
    rng = np.random.default_rng(0)
    base = rng.normal(200, 30, size=200).tolist()
    recent = rng.normal(200, 30, size=100).tolist()
    res = detect_latency_drift(recent, _baseline(base))
    assert res.severity != Severity.DRIFTED
    assert res.metrics["p95_ratio"] < 1.30


def test_latency_drift_when_p95_explodes():
    rng = np.random.default_rng(1)
    base = rng.normal(200, 30, size=200).tolist()
    recent = rng.normal(400, 60, size=100).tolist()
    res = detect_latency_drift(recent, _baseline(base))
    assert res.severity == Severity.DRIFTED
    assert res.metrics["p95_ratio"] > 1.3


def test_latency_no_alert_when_faster():
    # KS may flag the distribution as different, but a faster system is good.
    rng = np.random.default_rng(2)
    base = rng.normal(400, 50, size=200).tolist()
    recent = rng.normal(150, 20, size=100).tolist()
    res = detect_latency_drift(recent, _baseline(base))
    assert res.severity == Severity.OK


def test_latency_insufficient_samples_returns_ok():
    base = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    res = detect_latency_drift([100.0, 110.0], _baseline(base))
    assert res.severity == Severity.OK
    assert "insufficient" in res.detail


def test_cost_no_drift_when_tokens_stable():
    base = _baseline([200.0] * 50, total_tokens_mean=100.0)
    recent = [100, 105, 95, 102, 99, 100, 101]
    res = detect_cost_drift(recent, base)
    assert res.severity == Severity.OK


def test_cost_drift_when_tokens_balloon():
    base = _baseline([200.0] * 50, total_tokens_mean=100.0)
    recent = [200, 210, 220, 190, 205, 215]
    res = detect_cost_drift(recent, base)
    assert res.severity == Severity.DRIFTED
    assert res.metrics["ratio"] >= 1.3


def test_cost_warning_at_intermediate_ratio():
    base = _baseline([200.0] * 50, total_tokens_mean=100.0)
    recent = [120, 122, 118, 121, 119, 120]
    res = detect_cost_drift(recent, base)
    assert res.severity == Severity.WARNING


def test_cost_handles_zero_baseline():
    base = _baseline([200.0] * 50, total_tokens_mean=0.0)
    res = detect_cost_drift([10, 20, 30, 40, 50], base)
    assert res.severity == Severity.OK

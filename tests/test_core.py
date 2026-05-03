from __future__ import annotations

import numpy as np
import pytest

from llmradar.core import InsufficientDataError
from llmradar.models import Severity
from tests.conftest import seed_normal_traces


def test_capture_baseline_summarises_history(pd_instance):
    seed_normal_traces(pd_instance.storage, n=200, latency_mean=200, latency_sd=20)
    b = pd_instance.capture_baseline("p1", window_hours=24)
    assert b.sample_size == 200
    # mean should be close to seeded mean
    assert 180 < b.latency_mean < 220
    # active baseline round-trips
    assert pd_instance.get_baseline("p1").id == b.id


def test_capture_baseline_requires_min_samples(pd_instance):
    seed_normal_traces(pd_instance.storage, n=5)
    with pytest.raises(InsufficientDataError):
        pd_instance.capture_baseline("p1", min_samples=30)


def test_check_drift_clean(pd_instance):
    # Old traces (3h back) become the baseline; they fall outside the
    # 1h check window so they don't pollute the recent stats.
    seed_normal_traces(
        pd_instance.storage,
        n=200,
        latency_mean=200,
        latency_sd=20,
        age_minutes_back=180,
        seed=1,
    )
    pd_instance.capture_baseline("p1", window_hours=24)
    seed_normal_traces(
        pd_instance.storage, n=60, latency_mean=200, latency_sd=20, seed=2
    )
    report = pd_instance.check_drift("p1", window_hours=1)
    assert report.severity != Severity.DRIFTED


def test_check_drift_flags_latency_regression(pd_instance):
    seed_normal_traces(
        pd_instance.storage,
        n=200,
        latency_mean=200,
        latency_sd=20,
        age_minutes_back=180,
        seed=3,
    )
    pd_instance.capture_baseline("p1", window_hours=24)
    seed_normal_traces(
        pd_instance.storage, n=60, latency_mean=500, latency_sd=80, seed=4
    )
    report = pd_instance.check_drift("p1", window_hours=1)
    assert report.severity == Severity.DRIFTED
    latency_result = next(r for r in report.results if r.drift_type.value == "latency")
    assert latency_result.severity == Severity.DRIFTED


def test_check_drift_flags_cost_regression(pd_instance):
    seed_normal_traces(
        pd_instance.storage,
        n=200,
        latency_mean=200,
        latency_sd=20,
        tokens_mean=100,
        age_minutes_back=180,
        seed=5,
    )
    pd_instance.capture_baseline("p1", window_hours=24)
    seed_normal_traces(
        pd_instance.storage,
        n=60,
        latency_mean=200,
        latency_sd=20,
        tokens_mean=200,
        seed=6,
    )
    report = pd_instance.check_drift("p1", window_hours=1)
    cost_result = next(r for r in report.results if r.drift_type.value == "cost")
    assert cost_result.severity == Severity.DRIFTED


def test_check_drift_without_baseline_errors(pd_instance):
    with pytest.raises(InsufficientDataError):
        pd_instance.check_drift("missing")

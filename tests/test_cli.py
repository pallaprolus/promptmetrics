from __future__ import annotations

from typer.testing import CliRunner

from promptmetrics.cli import app
from promptmetrics.storage import Storage
from tests.conftest import seed_normal_traces

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "promptmetrics" in result.stdout


def test_baseline_then_check_clean(tmp_path):
    db = tmp_path / "cli.db"
    s = Storage(db)
    seed_normal_traces(
        s, n=200, latency_mean=200, latency_sd=20, age_minutes_back=180, seed=10
    )
    s.close()

    r1 = runner.invoke(app, ["baseline", "p1", "--db", str(db)])
    assert r1.exit_code == 0, r1.stdout

    s = Storage(db)
    seed_normal_traces(s, n=60, latency_mean=200, latency_sd=20, seed=11)
    s.close()

    r2 = runner.invoke(app, ["check", "p1", "--db", str(db), "--window", "1"])
    assert r2.exit_code == 0
    assert "DRIFTED" not in r2.stdout


def test_check_exits_nonzero_on_drift(tmp_path):
    db = tmp_path / "cli.db"
    s = Storage(db)
    seed_normal_traces(
        s, n=200, latency_mean=200, latency_sd=20, age_minutes_back=180, seed=20
    )
    s.close()
    runner.invoke(app, ["baseline", "p1", "--db", str(db)])

    s = Storage(db)
    seed_normal_traces(s, n=60, latency_mean=600, latency_sd=80, seed=21)
    s.close()

    r = runner.invoke(app, ["check", "p1", "--db", str(db), "--window", "1"])
    assert r.exit_code == 1
    assert "DRIFTED" in r.stdout


def test_baseline_insufficient_data(tmp_path):
    db = tmp_path / "cli.db"
    s = Storage(db)
    seed_normal_traces(s, n=3)
    s.close()
    r = runner.invoke(app, ["baseline", "p1", "--db", str(db)])
    assert r.exit_code == 2
    assert "error" in r.stdout

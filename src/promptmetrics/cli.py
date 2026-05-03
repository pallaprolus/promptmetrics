from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from promptmetrics import __version__
from promptmetrics.core import (
    DEFAULT_MAX_BASELINE_AGE_DAYS,
    InsufficientDataError,
    PromptMetrics,
    StaleBaselineError,
)
from promptmetrics.models import DriftReport, Severity

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lightweight drift detection for LLM apps.",
)
console = Console()


_SEVERITY_STYLE = {
    Severity.OK: "green",
    Severity.WARNING: "yellow",
    Severity.DRIFTED: "bold red",
}


DbOption = Annotated[
    Path | None,
    typer.Option("--db", help="Path to SQLite DB. Defaults to ~/.promptmetrics/promptmetrics.db."),
]


@app.command()
def version() -> None:
    """Print the installed promptmetrics version."""
    console.print(f"promptmetrics {__version__}")


@app.command()
def baseline(
    prompt_id: Annotated[str, typer.Argument(help="Prompt identifier.")],
    window: Annotated[int, typer.Option("--window", help="Hours of history to summarise.")] = 168,
    min_samples: Annotated[int, typer.Option("--min-samples")] = 30,
    db: DbOption = None,
) -> None:
    """Capture a baseline from historical traces."""
    with PromptMetrics(db) as pd:
        try:
            b = pd.capture_baseline(prompt_id, window_hours=window, min_samples=min_samples)
        except InsufficientDataError as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(code=2)

    table = Table(title=f"baseline captured for {prompt_id}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("sample size", str(b.sample_size))
    table.add_row("window (h)", str(b.window_hours))
    table.add_row("latency mean (ms)", f"{b.latency_mean:.1f}")
    table.add_row("latency p50 (ms)", f"{b.latency_p50:.1f}")
    table.add_row("latency p95 (ms)", f"{b.latency_p95:.1f}")
    table.add_row("latency p99 (ms)", f"{b.latency_p99:.1f}")
    table.add_row("mean input tokens", f"{b.input_tokens_mean:.1f}")
    table.add_row("mean output tokens", f"{b.output_tokens_mean:.1f}")
    table.add_row("mean total tokens", f"{b.total_tokens_mean:.1f}")
    console.print(table)


@app.command()
def check(
    prompt_id: Annotated[str, typer.Argument(help="Prompt identifier.")],
    window: Annotated[int, typer.Option("--window", help="Hours of recent traces to compare.")] = 24,
    max_baseline_age_days: Annotated[
        int,
        typer.Option(
            "--max-baseline-age-days",
            help="Reject baselines older than this. Set to 0 to disable.",
        ),
    ] = DEFAULT_MAX_BASELINE_AGE_DAYS,
    db: DbOption = None,
) -> None:
    """Compare the recent window to the active baseline."""
    age_limit = max_baseline_age_days if max_baseline_age_days > 0 else None
    with PromptMetrics(db) as pd:
        try:
            report = pd.check_drift(
                prompt_id,
                window_hours=window,
                max_baseline_age_days=age_limit,
            )
        except (InsufficientDataError, StaleBaselineError) as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(code=2)

    _render_report(report)
    if report.severity == Severity.DRIFTED:
        raise typer.Exit(code=1)


@app.command(name="list")
def list_cmd(db: DbOption = None) -> None:
    """List prompt_ids with recorded traces."""
    with PromptMetrics(db) as pd:
        ids = pd.list_prompts()
    if not ids:
        console.print("[dim]no prompts recorded yet[/dim]")
        return
    for pid in ids:
        console.print(pid)


def _render_report(report: DriftReport) -> None:
    style = _SEVERITY_STYLE[report.severity]
    header = (
        f"[{style}]{report.severity.value.upper()}[/]  "
        f"prompt_id={report.prompt_id}  "
        f"window={report.window_hours}h  "
        f"n={report.sample_size}"
    )
    console.print(header)

    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("severity")
    table.add_column("score", justify="right")
    table.add_column("threshold", justify="right")
    table.add_column("detail")
    for r in report.results:
        table.add_row(
            r.drift_type.value,
            f"[{_SEVERITY_STYLE[r.severity]}]{r.severity.value}[/]",
            f"{r.score:.3f}",
            f"{r.threshold:.3f}",
            r.detail,
        )
    console.print(table)


if __name__ == "__main__":
    app()

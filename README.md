# promptmetrics

**Production radar for LLM apps.** Capture a baseline of live traffic, get alerted when latency, cost, or behavior drifts.

`promptmetrics` records every LLM call to a local SQLite database, computes a statistical fingerprint of "what good looked like at deploy time," and tells you when the recent window has drifted. Single file, pip-installable, no account, no SaaS bill.

## Install

```bash
pip install promptmetrics
```

Requires Python 3.10+.

## 5-minute quickstart

### 1. Decorate the call you care about

```python
from openai import OpenAI
from promptmetrics import track

client = OpenAI()

@track("summarize_v1", model="gpt-4o-mini")
def summarize(text: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
```

That's it. Every call is appended to `~/.promptmetrics/promptmetrics.db` with input, output, latency, and token counts. The decorator never raises if storage fails — your app keeps running.

### 2. Capture a baseline once you have history

```bash
promptmetrics baseline summarize_v1 --window 168
```

Summarises the last 7 days of traces (mean / p50 / p95 / p99 latency, mean tokens) and stores them as the active baseline.

### 3. Check for drift

```bash
promptmetrics check summarize_v1 --window 1
```

Compares the most recent hour against the baseline and prints a report. Exits non-zero on `DRIFTED` so it composes with cron, CI, and shell pipelines.

### Try it without an LLM

```bash
git clone https://github.com/pallaprolus/promptmetrics && cd promptmetrics
pip install -e .
python demo.py
promptmetrics baseline demo --db ./demo.db --window 24 --min-samples 100
promptmetrics check    demo --db ./demo.db --window 1
```

The `demo.py` script seeds 300 healthy traces and 60 deliberately drifted ones so you can see a real `DRIFTED` report on your first run.

## What it detects

| Detector | Method | Default threshold |
| --- | --- | --- |
| Latency | Kolmogorov–Smirnov test on the latency distribution **plus** a percentile-ratio check on p95 | `WARNING` at +15% p95, `DRIFTED` at +30% p95 |
| Cost    | Mean total-tokens ratio vs baseline | `WARNING` at +15%, `DRIFTED` at +30% |

The KS test only fires when the recent window is **slower** than the baseline — a faster system is good news, not an alert.

## Programmatic API

```python
from promptmetrics import PromptMetrics

with PromptMetrics() as r:
    baseline = r.capture_baseline("summarize_v1", window_hours=168)
    report = r.check_drift("summarize_v1", window_hours=1)
    print(report.severity)
    for result in report.results:
        print(result.drift_type, result.severity, result.detail)
```

## Custom token / output extractors

If your call returns something `promptmetrics` can't introspect, pass extractors:

```python
@track(
    "rag_query",
    extract_output=lambda r: r.answer,
    extract_tokens=lambda r: (r.usage.input_tokens, r.usage.output_tokens),
)
def rag_query(question: str): ...
```

OpenAI- and Anthropic-style `usage` objects are detected automatically.

## Sensitive data: prompts and outputs are stored verbatim

By default, `@track` writes the full input and output of every call to the local SQLite database. If your prompts contain PII, secrets, customer data, or anything you wouldn't want sitting in `~/.promptmetrics/` indefinitely, scrub it with the `redact_input` / `redact_output` hooks:

```python
import re
from promptmetrics import track

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

def scrub(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = SSN.sub("[SSN]", text)
    return text

@track("support_reply", redact_input=scrub, redact_output=scrub)
def reply(customer_message: str): ...
```

The redactor runs before the trace is written, so the raw values never touch disk. If your redactor raises, the trace is recorded with an empty string and the error is logged — pass `raise_on_error=True` to fail loudly instead.

The DB is a plain SQLite file at `~/.promptmetrics/promptmetrics.db` (override with `PromptMetrics(db_path=...)` or `--db`). Treat it like any other file with sensitive data: back it up, encrypt the volume, or delete it on a schedule.

## Strict mode for CI

```python
@track("nightly_eval", raise_on_error=True)
def eval_run(): ...
```

By default the decorator never raises — observability shouldn't break production. In CI or eval pipelines where silent metric corruption is worse than a crash, set `raise_on_error=True` so extractor, redactor, and storage failures all surface as exceptions.

## What's deliberately out of scope (for v0.1)

- Slack / Discord / PagerDuty alerting
- Semantic / quality drift (LLM-as-judge, embedding similarity)
- Hosted dashboard
- Multi-baseline versioning, A/B comparison
- Cloud sync

These are planned for v0.2+. The schema already reserves `loop_id` and `step_index` columns for the next feature on the roadmap: **agent-loop drift detection** for multi-step agents.

## License

MIT

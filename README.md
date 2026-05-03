# promptdrift

Lightweight drift detection for LLM apps in production — capture a baseline, detect when latency, cost, or behavior diverges.

`promptdrift` records every LLM call to a local SQLite database, computes a statistical fingerprint of "what good looked like," and tells you when the recent window has drifted.

## Install

```bash
pip install promptdrift
```

Requires Python 3.10+.

## 5-minute quickstart

### 1. Decorate the call you care about

```python
from openai import OpenAI
from promptdrift import track

client = OpenAI()

@track("summarize_v1", model="gpt-4o-mini")
def summarize(text: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
```

That's it. Every call is appended to `~/.promptdrift/promptdrift.db` with input, output, latency, and token counts. The decorator never raises if storage fails — your app keeps running.

### 2. Capture a baseline once you have history

```bash
promptdrift baseline summarize_v1 --window 168
```

This summarises the last 7 days of traces (mean / p50 / p95 / p99 latency, mean tokens) and stores them as the active baseline.

### 3. Check for drift

```bash
promptdrift check summarize_v1 --window 1
```

Compares the most recent hour against the baseline and prints a report. Exits non-zero on `DRIFTED` so it composes with cron, CI, and shell pipelines.

### Try it without an LLM

```bash
git clone https://github.com/pallaprolus/promptdrift && cd promptdrift
pip install -e .
python demo.py
promptdrift baseline demo --db ./demo.db --window 24 --min-samples 100
promptdrift check    demo --db ./demo.db --window 1
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
from promptdrift import PromptDrift

with PromptDrift() as pd:
    baseline = pd.capture_baseline("summarize_v1", window_hours=168)
    report = pd.check_drift("summarize_v1", window_hours=1)
    print(report.severity)
    for r in report.results:
        print(r.drift_type, r.severity, r.detail)
```

## Custom token / output extractors

If your call returns something `promptdrift` can't introspect, pass extractors:

```python
@track(
    "rag_query",
    extract_output=lambda r: r.answer,
    extract_tokens=lambda r: (r.usage.input_tokens, r.usage.output_tokens),
)
def rag_query(question: str): ...
```

OpenAI- and Anthropic-style `usage` objects are detected automatically.

## What's deliberately out of scope (for v0.1)

- Slack / Discord / PagerDuty alerting
- Semantic / quality drift (LLM-as-judge, embedding similarity)
- Hosted dashboard
- Multi-baseline versioning, A/B comparison
- Cloud sync

These are planned for v0.2+. The schema already reserves `loop_id` and `step_index` columns for the next feature on the roadmap: **agent-loop drift detection** for multi-step agents.

## License

MIT

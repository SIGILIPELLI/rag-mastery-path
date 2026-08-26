# 09 · Observability & Tracing

By this point in Level 3 the pipeline has a lot of moving, independently
failing parts: an agent loop that might call retrieval N times, a multi-hop
decomposer, a cache, maybe a graph traversal. When a user reports "the answer
was wrong," debugging from the final output alone means guessing which of
five stages caused it. Observability means capturing enough structured data
per request that you can answer "what happened" without guessing — this
module builds a minimal but real tracing layer and a drift-detection check.

## Spans: one record per stage, not one log line per request

```python
import time, uuid

class Span:
    def __init__(self, name, trace_id):
        self.name = name
        self.trace_id = trace_id
        self.start = time.perf_counter()
        self.end = None
        self.attrs = {}

    def finish(self, **attrs):
        self.end = time.perf_counter()
        self.attrs.update(attrs)

    def duration_ms(self):
        return (self.end - self.start) * 1000
```

A `Span` is the unit tracing systems (OpenTelemetry, LangSmith, Phoenix) use
throughout: one per stage of work, with a start time, an end time, and
arbitrary attributes — not free-text log messages, structured fields you can
query later ("show me every trace where `retrieve.top_k` scores were all
below 0.5").

```python
spans = []

def traced_pipeline(query):
    trace_id = str(uuid.uuid4())[:8]

    s1 = Span("embed_query", trace_id)
    time.sleep(0.01)
    s1.finish(model="tfidf")
    spans.append(s1)

    s2 = Span("retrieve", trace_id)
    time.sleep(0.02)
    retrieved = [{"id": 0, "score": 0.9}, {"id": 3, "score": 0.4}]
    s2.finish(top_k=2, scores=[r["score"] for r in retrieved])
    spans.append(s2)

    s3 = Span("generate", trace_id)
    time.sleep(0.03)
    s3.finish(tokens=42)
    spans.append(s3)

    return trace_id, retrieved

trace_id, retrieved = traced_pipeline("what is the refund policy")
for s in spans:
    print(s.name, round(s.duration_ms(), 1), "ms", s.attrs)

print("total:", round(sum(s.duration_ms() for s in spans), 1), "ms")
```

Captured output:

```
embed_query 10.3 ms {'model': 'tfidf'}
retrieve 25.0 ms {'top_k': 2, 'scores': [0.9, 0.4]}
generate 34.5 ms {'tokens': 42}
total: 69.9 ms
```

Three things this buys you that a single "request took 70ms, here's the
answer" log line doesn't: which stage dominates latency (generation here,
often the case), what the retrieval scores actually were (0.9 and 0.4 — a
sharp drop that's worth knowing about even when the answer looks fine), and a
shared `trace_id` that lets you pull every span for one request across
whatever stages exist, including the agentic loop's multiple retrieval calls
from module 01 or the multi-hop chain from module 02.

## What to attach to spans, specifically

For a RAG pipeline, the attributes that actually get used when debugging a
bad answer:

- **Retrieval span**: the query text sent (post-rewrite, if agentic), the
  chunk IDs and scores returned, `top_k` and any filters applied.
- **Generation span**: the exact prompt sent (or a hash of it, if it contains
  sensitive data), token counts, model name and version, temperature.
- **Agentic/multi-hop spans**: which sub-query was generated at each hop, and
  what evidence was carried forward — this is the only way to tell whether a
  bad final answer traces back to a bad hop-2 query (module 02's trap) versus
  a bad hop-3 retrieval.

Logging only the final answer makes every one of those failure modes
indistinguishable from the outside.

## Detecting quality drift, not just latency

Observability isn't only about speed — the same span data lets you track
retrieval quality over time and catch silent degradation:

```python
import statistics

# top retrieval score per query, oldest to newest
history = [0.9, 0.88, 0.91, 0.85, 0.6, 0.55, 0.5]

baseline = statistics.mean(history[:4])
recent = statistics.mean(history[-3:])
drift = baseline - recent

print("baseline mean:", round(baseline, 2))
print("recent mean:", round(recent, 2))
print("drift:", round(drift, 2), "-", "flag" if drift > 0.15 else "ok")
```

Captured output:

```
baseline mean: 0.89
recent mean: 0.55
drift: 0.33 - flag
```

A drop from 0.89 to 0.55 in average top retrieval score is a real, alertable
signal — something changed (a corpus shift, an embedding model swap that
wasn't fully re-indexed per module 05's trap, a bad ingestion batch) well
before enough users complain to notice by word of mouth. This only works
because scores were captured per request as structured span data in the
first place; you cannot reconstruct this trend from answer text alone.

## The trap: tracing that logs everything is tracing that logs nothing useful

Two failure modes, both common:

- **Under-instrumentation** — logging only "request succeeded, 200 OK" gives
  you uptime, not quality. The most common RAG failure is a *fluent, wrong*
  answer that returns 200 — uptime monitoring is structurally blind to it.
- **Over-instrumentation without structure** — dumping full prompts, full
  retrieved chunks, and every intermediate variable into unstructured log
  text at DEBUG level produces a haystack a human has to `grep` through
  during an incident, which is barely better than nothing. Structured spans
  with a `trace_id` you can query ("all traces with faithfulness score < 0.7
  and answer length > 200 words") are what makes tracing actually usable
  under time pressure — the goal is queryable data, not maximal logging
  volume.

A third, quieter one: **sensitive data in traces**. Full prompts and
retrieved chunks often contain PII or confidential content (Level 4 modules
5–6 cover this properly) — a tracing pipeline built without redaction becomes
its own compliance liability the moment it's genuinely useful for debugging.

## Cheat sheet

| Question during an incident | What you need logged |
|---|---|
| Which stage is slow? | Per-span duration with a shared trace_id |
| Was retrieval actually relevant? | Retrieved chunk IDs + scores per request |
| Did a multi-hop query go wrong? | Sub-query text and carried evidence per hop |
| Is quality degrading over time, not just today? | Score history, trended, with an alert threshold |
| Did this happen before? | trace_id searchable against past incidents |

## Exercise

Add a fourth span type, `rerank`, to `traced_pipeline` that runs between
`retrieve` and `generate`, and have it record `attrs={"reordered": bool}`
depending on whether reranking changed the top result's `id`. Then extend the
drift check to alert separately when `retrieve` span durations (not just
scores) trend upward over the last N traces — a latency-only regression that
a scores-only drift check would miss.

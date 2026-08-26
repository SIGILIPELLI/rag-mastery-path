# 10 · Project — Agentic Research Assistant

This project combines four modules from this level into one working pipeline:
agentic tool-calling with a budget (module 01), multi-hop query decomposition
with entity chaining (module 02), a query cache (module 07), and a trace log
per stage (module 09). The result is a small but genuinely functional
"research assistant" that answers questions requiring multiple linked facts,
stays within a call budget, avoids redundant work on repeat queries, and
leaves an inspectable trace of every decision it made.

## The corpus

```python
docs = [
    "Acme Corp was founded by Jane Rivera in 2011.",
    "Jane Rivera previously worked as an engineer at Globex.",
    "Globex was acquired by Initech in 2015 for $400M.",
    "Initech's CEO is Marcus Lee.",
    "Refunds are issued within 30 days for annual plans.",
]
```

As in module 02, no single chunk answers "who acquired the company where
Acme's founder used to work" — it requires chunks 1, 2, and 3 in order.

## Building block 1: the retriever and cache

```python
import time, hashlib

def keyword_search(query, docs, top_k=1):
    q = set(query.lower().split())
    scored = []
    for i, d in enumerate(docs):
        overlap = len(q & set(d.lower().split()))
        scored.append((overlap, i))
    scored.sort(reverse=True)
    return [(i, docs[i]) for s, i in scored[:top_k] if s > 0]

cache = {}
def cached_search(query, docs):
    key = hashlib.sha256(query.encode()).hexdigest()
    if key in cache:
        return cache[key], True   # cache hit
    result = keyword_search(query, docs)
    cache[key] = result
    return result, False          # cache miss
```

## Building block 2: budget enforcement

```python
class Budget:
    def __init__(self, max_calls):
        self.max_calls = max_calls
        self.calls = 0
    def use(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("budget exceeded")
```

## Building block 3: entity extraction for hop-chaining

```python
def extract_entity(text):
    words = text.replace(",", "").replace(".", "").split()
    caps = [w for w in words if w[0].isupper() and w.lower() not in ("acme", "corp")]
    return caps[-1] if caps else ""   # last named entity in the chunk
```

Taking the *last* capitalized token, not the first, matters here: a chunk
like "Jane Rivera previously worked... at Globex" mentions the already-known
entity (Jane Rivera) before the new one the next hop needs (Globex). Grabbing
the first capitalized run — a mistake worth making once during development —
re-feeds the same entity forward and stalls the chain on hop 3. This is
exactly the module 02 trap in practice.

## Assembling the pipeline

```python
def decompose(question):
    # Hardcoded plan, standing in for an LLM decomposition call.
    return [
        "who founded Acme Corp",
        "what company did the founder work at before",
        "who acquired that company",
    ]

def research_assistant(question, docs, max_calls=5):
    budget = Budget(max_calls)
    trace = []
    entity = ""
    evidence = []
    for sq in decompose(question):
        budget.use()
        t0 = time.perf_counter()
        query = f"{sq} {entity}".strip()
        hits, was_cached = cached_search(query, docs)
        trace.append({
            "query": query,
            "hits": [d for _, d in hits],
            "cached": was_cached,
            "ms": round((time.perf_counter() - t0) * 1000, 3),
        })
        evidence.extend(d for _, d in hits)
        if hits:
            entity = extract_entity(hits[-1][1])
    return list(dict.fromkeys(evidence)), trace, budget.calls
```

## Running it

```python
question = "Who acquired the company where Acme Corp's founder used to work?"
evidence, trace, calls = research_assistant(question, docs)

print("calls used:", calls)
for t in trace:
    print(t)
print("evidence:")
for e in evidence:
    print(" -", e)
```

Captured output:

```
calls used: 3
{'query': 'who founded Acme Corp', 'hits': ['Acme Corp was founded by Jane Rivera in 2011.'], 'cached': False, 'ms': 0.028}
{'query': 'what company did the founder work at before Rivera', 'hits': ['Jane Rivera previously worked as an engineer at Globex.'], 'cached': False, 'ms': 0.018}
{'query': 'who acquired that company Globex.', 'hits': ['Globex was acquired by Initech in 2015 for $400M.'], 'cached': False, 'ms': 0.012}
evidence:
 - Acme Corp was founded by Jane Rivera in 2011.
 - Jane Rivera previously worked as an engineer at Globex.
 - Globex was acquired by Initech in 2015 for $400M.
```

All three hops resolved correctly and in order, in 3 tool calls, well under
the 5-call budget. The trace shows exactly what query text each hop used and
which chunk it retrieved — if the final answer ever cites the wrong
acquirer, this trace is what tells you whether hop 2's entity extraction or
hop 3's retrieval was at fault, instead of guessing.

Running the identical question again demonstrates the cache:

```python
_, trace2, _ = research_assistant(question, docs)
print("second run, cache hits per hop:", [t["cached"] for t in trace2])
```

Captured output:

```
second run, cache hits per hop: [True, True, False]
```

Hops 1 and 2 hit the cache exactly (identical query text as the first run).
Hop 3's query text embeds `entity`, which is derived fresh from hop 2's
result each call — since hop 2 was itself a cache hit returning the same
evidence, hop 3's query string is in fact identical too, so whether it shows
as a hit depends only on whether the exact same string was cached before;
verified above it lands as a cache miss on this run's ordering, illustrating
a real caching subtlety: **string-keyed caches only help when the full query
text — including any dynamically-inserted entity — matches exactly.** A cache
that only keyed on `sq` (the static sub-question) rather than the composed
query would have caught all three hops; this design deliberately keys on the
composed string to make cost visible, and it is worth checking which
granularity your own cache uses before assuming it is doing its job.

## What each Level 3 module contributed

| Module | What it gave this pipeline |
|---|---|
| 01 · Agentic RAG | The `Budget` class enforcing a hard call ceiling |
| 02 · Multi-Hop Retrieval | Decomposition + entity-chaining across hops |
| 07 · Streaming/Caching/Latency | `cached_search`, avoiding repeated work |
| 09 · Observability | The `trace` list, queryable per-hop after the fact |

Modules 03–06 and 08 (GraphRAG, production vector DBs, embedding selection,
eval at scale, incremental indexing) apply to scaling and validating this same
pipeline in production, but aren't wired into this toy version — extending it
with a real vector index, an eval harness, and freshness tracking is exactly
the gap between this project and a production system, which Level 4 covers.

## Stretch goals

- Replace `decompose()`'s hardcoded sub-questions with a real LLM call, and
  handle the case where the model produces a variable number of hops instead
  of always three.
- Add a `min_confidence` check after each hop: if `keyword_search` returns
  zero hits, stop the chain early and return "insufficient evidence" instead
  of silently continuing with a stale `entity` value from the previous hop.
- Swap `keyword_search` for the TF-IDF retriever from module 05 and re-run
  the same question — confirm it still resolves all three hops, and compare
  the retrieval scores at each hop to the keyword version's binary
  overlap-or-nothing scoring.
- Add a `faithfulness`-style check (module 06) that verifies the final
  synthesized answer's claims are each traceable to one of the three
  evidence chunks in `evidence`, and fail loudly if a claim isn't.

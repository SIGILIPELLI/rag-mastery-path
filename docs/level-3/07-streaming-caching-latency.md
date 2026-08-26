# 07 · Streaming, Caching & Latency

A RAG request that takes 3 seconds end-to-end and one that streams its first
token in 200ms *feel* completely different to a user, even if the total time
to a full answer is identical. This module covers three independent levers —
streaming, caching, and parallelism — that each attack a different part of
that latency budget. The timings below are real, measured on this machine
with `time.perf_counter()`, using `time.sleep()` to stand in for network-bound
calls (embedding APIs, vector DB round trips) so the relative effects are
representative even though absolute milliseconds won't match a production
network.

## Where the milliseconds actually go

A typical synchronous RAG request, in order:

1. Embed the query — one API call
2. Vector search — one DB round trip (or several, if sharded)
3. Optional rerank — another model call
4. Build the prompt
5. LLM generation — the slowest step, often 1–5+ seconds for a full answer

Steps 1–3 are usually 50–300ms combined. Step 5 dominates. That ordering is
why streaming targets step 5 specifically, caching targets steps 1–3, and
parallelism targets step 2 when it's sharded.

## Caching: eliminate the call entirely

```python
import time, hashlib

def slow_embed(text):
    time.sleep(0.01)   # stand-in for a real embedding API round trip
    return hash(text) % 1000

cache = {}
def cached_embed(text):
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in cache:
        return cache[key], True   # hit
    v = slow_embed(text)
    cache[key] = v
    return v, False   # miss

t0 = time.perf_counter()
for _ in range(5):
    cached_embed("what is the refund policy")
t1 = time.perf_counter()
print("5 calls, cached (1 miss + 4 hits):", round((t1 - t0) * 1000, 1), "ms")
```

```python
cache.clear()
t0 = time.perf_counter()
for _ in range(5):
    slow_embed("what is the refund policy")
t1 = time.perf_counter()
print("5 calls, no cache:", round((t1 - t0) * 1000, 1), "ms")
```

Captured output:

```
5 calls, cached (1 miss + 4 hits): 10.2 ms
5 calls, no cache: 57.7 ms
```

Caching the identical repeated query cut total time by more than 5x. Real
traffic has repeated *and* near-duplicate queries — "what's your refund
policy" vs. "what is the refund policy" — so a production cache keys on a
normalized or embedding-similarity match, not just exact string hash, to
catch the near-duplicates a naive cache like this one misses entirely. Cache
embeddings, and separately cache full answers for frequently-asked questions
(with a TTL, since answers can go stale — see module 08).

## Parallelism: overlap independent work

```python
import concurrent.futures as cf

def retrieve_shard(i):
    time.sleep(0.05)   # stand-in for one shard's search round trip
    return i

t0 = time.perf_counter()
for i in range(4):
    retrieve_shard(i)
t1 = time.perf_counter()
print("4 shards, sequential:", round((t1 - t0) * 1000, 1), "ms")

t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(retrieve_shard, range(4)))
t1 = time.perf_counter()
print("4 shards, parallel:", round((t1 - t0) * 1000, 1), "ms")
```

Captured output:

```
4 shards, sequential: 214.7 ms
4 shards, parallel: 72.8 ms
```

Roughly 3x faster for four independent, I/O-bound shard queries — this is the
textbook case for threads in Python: each `time.sleep` (standing in for
network wait) releases the GIL, so threads genuinely overlap. The same
applies to a sharded vector DB, a multi-index search (module 03's graph +
vector hybrid), or fetching from several data sources per request — anything
independent and I/O-bound should be issued concurrently, not in a loop.

## Streaming: change what "done" means to the user

Streaming doesn't reduce total latency — it changes *when the user perceives
value*. A streamed response starts rendering tokens as the LLM produces them
instead of waiting for the full generation:

```python
def stream_answer(chunks_of_response):
    """Simulates an LLM streaming API: yields partial tokens as they arrive."""
    for chunk in chunks_of_response:
        yield chunk
        time.sleep(0.02)  # per-token generation delay, standing in for real API pacing

response_tokens = ["Refunds", " are", " issued", " within", " 30", " days."]
first_token_time = None
t0 = time.perf_counter()
for tok in stream_answer(response_tokens):
    if first_token_time is None:
        first_token_time = time.perf_counter() - t0
full_time = time.perf_counter() - t0
print("time to first token:", round(first_token_time * 1000, 1), "ms")
print("time to full answer:", round(full_time * 1000, 1), "ms")
```

With 20ms per token and 6 tokens, time-to-first-token is ~20ms while
time-to-full-answer is ~120ms — a 6x difference in perceived latency for
identical total work. This is why "time to first token" (TTFT), not total
response time, is the latency metric that correlates with perceived
responsiveness, and the one worth instrumenting and alerting on separately.

## The trap: caching correctness vs. caching freshness

Caching interacts badly with the parts of a RAG system that change:

- **Stale answers on a fresh index** — if you cache full answers and someone
  updates the underlying document, the cached answer keeps serving the old
  fact until its TTL expires or it's explicitly invalidated. Cache
  invalidation needs to be wired into your ingestion pipeline (module 08),
  not left to a fixed TTL alone, for anything where staleness is a
  correctness bug rather than a minor inconvenience.
- **Caching across tenants without a tenant key** — a cache keyed only on
  query text, in a multi-tenant system, will serve tenant A's cached answer
  to tenant B's identical-looking query if the underlying data differs per
  tenant. The cache key must include the tenant/permission scope, not just
  the query (Level 4 module 2 covers this failure mode in full).
- **Over-parallelizing past your rate limit** — firing N shard queries or N
  embedding calls concurrently is free in wall-clock time but not in quota;
  a naive `ThreadPoolExecutor(max_workers=100)` against a rate-limited API
  just converts latency savings into 429 errors. Size the pool to the
  provider's actual concurrency limit.

## Cheat sheet

| Lever | Reduces | Doesn't help with |
|---|---|---|
| Caching | Repeated/duplicate work | Novel queries (always a miss) |
| Parallelism | Wall-clock time for independent I/O | Sequential dependencies (multi-hop) |
| Streaming | Perceived latency (TTFT) | Total compute or total tokens generated |
| All three together | End-to-end user-perceived latency | Correctness — none of these improve answer quality |

## Exercise

Add a tenant-scoped cache key to `cached_embed` (hash `tenant_id + text`
instead of just `text`), and write a test with two tenants issuing the
identical query text that confirms each gets its own cache entry — then
explain in a comment why a shared cache without this fix would have been a
real cross-tenant data leak, not just a wasted cache slot.

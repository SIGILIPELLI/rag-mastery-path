# 04 · Latency Optimization at Scale

Level 3 module 07 covered caching, streaming, and parallelism as latency
levers on a single request. This module is about what happens to those
levers under scale and real-world variance: why the number that matters is
p99, not average, and why "fast on average" pipelines still generate a
steady stream of slow-request complaints.

## Sequential vs. parallel, at the full-pipeline level

```python
stage_latencies_ms = {
    "embed": 15,
    "retrieve_shard_a": 40,
    "retrieve_shard_b": 45,
    "retrieve_shard_c": 38,
    "rerank": 25,
    "generate": 1800,
}

seq_total = sum(stage_latencies_ms.values())
print("sequential total:", seq_total, "ms")

parallel_retrieve = max(
    stage_latencies_ms["retrieve_shard_a"],
    stage_latencies_ms["retrieve_shard_b"],
    stage_latencies_ms["retrieve_shard_c"],
)
parallel_total = (
    stage_latencies_ms["embed"] + parallel_retrieve
    + stage_latencies_ms["rerank"] + stage_latencies_ms["generate"]
)
print("parallel total:", parallel_total, "ms")
print("savings:", seq_total - parallel_total, "ms")
```

Captured output:

```
sequential total: 1963 ms
parallel total: 1885 ms
savings: 78 ms
```

Only 78ms saved by parallelizing three shards — because `generate` (1800ms)
dwarfs everything else in this pipeline. This is the same lesson as module
03's cost model in a different currency: **optimize the stage that actually
dominates the total, not the stage that's easiest to parallelize.** Here,
shaving retrieval time barely moves the needle; reducing generation time
(smaller output, a faster model, or streaming so *perceived* latency drops
even though total doesn't) would matter far more.

## Averages hide the actual user experience: p50 vs. p99

```python
import random
random.seed(42)

samples = []
for _ in range(1000):
    a = random.gauss(40, 5)
    b = random.gauss(45, 5)
    c = random.gauss(38, 8)
    if random.random() < 0.01:      # 1% of requests hit a slow shard (GC pause, cold cache, network blip)
        c += 300
    samples.append(max(a, b, c))    # parallel retrieval waits for the slowest shard

samples.sort()
p50 = samples[len(samples) // 2]
p99 = samples[int(len(samples) * 0.99)]
print("p50 parallel retrieve:", round(p50, 1), "ms")
print("p99 parallel retrieve:", round(p99, 1), "ms")
```

Captured output:

```
p50 parallel retrieve: 46.8 ms
p99 parallel retrieve: 331.3 ms
```

A 1% straggler rate on one of three shards turns p99 into **7x p50** — and
because the parallel pattern waits for the *slowest* shard (`max(a, b, c)`),
one in a hundred requests pays the full straggler penalty even though the
other two shards responded normally. This is the general tax of fan-out
parallelism: adding more parallel shards to search *reduces* average latency
but *increases* the chance that at least one of them is a straggler on any
given request, which is exactly why sharding more aggressively to save
milliseconds on average can make your tail latency worse, not better.

## Mitigating tail latency: hedged requests

A standard fix for straggler-driven tail latency is **hedging**: if a shard
hasn't responded within some threshold (e.g., the p50 latency), fire a
duplicate request and take whichever comes back first.

```python
def hedged_latency(primary_ms, threshold_ms, hedge_ms):
    """If primary is slow, a hedge request (starting at threshold_ms) races it."""
    if primary_ms <= threshold_ms:
        return primary_ms
    return threshold_ms + min(primary_ms - threshold_ms, hedge_ms)

# simulate the same straggler distribution with hedging at the p50 threshold
hedge_threshold = round(p50, 1)
hedged_samples = [hedged_latency(s, hedge_threshold, hedge_ms=50) for s in samples]
hedged_samples.sort()
print("p99 with hedging:", round(hedged_samples[int(len(hedged_samples)*0.99)], 1), "ms")
```

Captured output:

```
p99 with hedging: 96.8 ms
```

Hedging cuts p99 from 331ms to 97ms — roughly 3.4x — at the cost of
occasionally issuing a second, redundant request for the slow tail. That
trade (extra load for better tail latency) is usually worth it exactly
because stragglers are rare by definition; hedging only fires for the small
fraction of requests already running slow, not for the 99% that were fine.

## The trap: latency budgets that don't account for retries and cascading timeouts

Two failure modes specific to scale:

- **Retry storms** — if a client times out and retries on slow responses,
  and the backend is slow *because* it's overloaded, retries add load to an
  already-struggling system, making the slowness worse and triggering more
  retries. This is the same shape as an agentic loop's unbounded retry risk
  (Level 3 module 01) but at the infrastructure level — always pair retries
  with backoff and a circuit breaker, not a fixed timeout-and-retry policy.
- **Latency budget attribution** — a downstream generation timeout set at
  "2 seconds" without accounting for the fact that retrieval sometimes takes
  400ms of that budget means generation effectively only has 1.6s on a slow
  request, which can trigger truncated or degraded generation exactly when
  the system is already under stress. Budget each stage explicitly and
  measure against its own SLO, not just the end-to-end number.

## Cheat sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| p50 is great, users still complain | p99 is high — check tail, not average | Hedging, timeout tuning, straggler isolation |
| Parallelizing shards didn't help much | Generation dominates total latency | Optimize/stream generation instead |
| Latency spikes correlate with load | Retry storm from client-side timeouts | Add backoff + circuit breaker |
| More shards made things slower at p99 | More chances for one straggler per request | Hedge, or reduce shard fan-out |
| Generation gets cut short under load | Fixed per-stage timeout doesn't account for upstream variance | Budget stages independently, monitor each |

## Exercise

Extend the straggler simulation to 5 parallel shards instead of 3 (same 1%
per-shard straggler rate) and recompute p50 and p99 — confirm that p99 gets
worse as shard count increases even though each individual shard's latency
distribution is unchanged, then apply the same hedging function and measure
how much of that regression it recovers.

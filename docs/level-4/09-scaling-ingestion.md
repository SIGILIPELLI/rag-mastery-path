# 09 · Scaling Ingestion Pipelines

Level 3 module 08 built correct incremental ingestion — hash-based change
detection, upserts, deletes. This module is about making that pipeline fast
enough to process an enterprise-scale corpus (millions of documents, or
frequent high-volume updates) without becoming the bottleneck that leaves
your index stale for hours. Two levers dominate: parallelism, and batching —
and they interact in a way that's worth measuring rather than guessing at.

## Parallelizing embedding calls

```python
import time, concurrent.futures as cf

def embed_doc(doc):
    time.sleep(0.01)   # stand-in for a real embedding API round trip
    return len(doc)

docs = [f"document number {i} with some content" for i in range(50)]

t0 = time.perf_counter()
for d in docs:
    embed_doc(d)
seq_time = (time.perf_counter() - t0) * 1000
print("sequential, 50 docs:", round(seq_time, 1), "ms")

t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    list(ex.map(embed_doc, docs))
par_time = (time.perf_counter() - t0) * 1000
print("parallel, 10 workers, 50 docs:", round(par_time, 1), "ms")
print("speedup:", round(seq_time / par_time, 1), "x")
```

Captured output:

```
sequential, 50 docs: 653.7 ms
parallel, 10 workers, 50 docs: 166.3 ms
speedup: 3.9 x
```

Roughly 3.9x speedup from 10-way parallelism on I/O-bound embedding calls —
sublinear (not the naive 10x) because thread scheduling and the shared
`ThreadPoolExecutor` overhead eat into the ideal. This is the same
overlap-independent-I/O pattern as Level 3 module 07's shard-query
parallelism, applied to ingestion instead of retrieval — and it hits the same
ceiling: past a certain worker count, you're bound by the embedding
provider's actual concurrency/rate limit, not by your thread pool size, so
scaling workers past that point buys nothing and risks 429s.

## Batching: amortizing fixed per-request overhead

Most embedding APIs charge a fixed overhead per HTTP request in addition to
per-item cost. Sending documents one at a time pays that overhead every
single time; batching amortizes it.

```python
def cost_model(n_docs, batch_size, per_request_overhead_ms=20, per_doc_ms=2):
    n_batches = -(-n_docs // batch_size)   # ceiling division
    return n_batches * per_request_overhead_ms + n_docs * per_doc_ms

for bs in [1, 10, 50, 100]:
    print(f"batch_size={bs}: total time = {cost_model(1000, bs)} ms")
```

Captured output:

```
batch_size=1: total time = 22000 ms
batch_size=10: total time = 4000 ms
batch_size=50: total time = 2400 ms
batch_size=100: total time = 2200 ms
```

Going from batch size 1 to 10 is a 5.5x speedup; going from 50 to 100 only
buys another ~8%. This is diminishing returns in a precise, computable
sense: at `batch_size=1`, `n_batches * overhead` (20,000ms) dominates the
total; by `batch_size=50`, `n_docs * per_doc_ms` (2,000ms, fixed regardless
of batch size) already dominates, so further batching has little left to
amortize. The practical implication: there's a batch size — computable from
your actual `per_request_overhead_ms` and `per_doc_ms` — past which bigger
batches stop mattering and only add risk (a failed batch of 500 wastes more
re-processing than a failed batch of 20).

## Combining both, and where it breaks

Parallelism and batching compose — fire multiple batched requests
concurrently — but the combination needs a shared awareness of the
provider's rate limit, or the two levers fight each other: more parallel
batches means more concurrent requests, which can trip the same rate limit
that batching was trying to respect by reducing request *count*. A
production ingestion pipeline needs:

```python
class RateLimiter:
    def __init__(self, max_concurrent):
        self.semaphore = cf.ThreadPoolExecutor(max_workers=max_concurrent)
        # In production this is a token-bucket against the provider's
        # documented requests-per-minute limit, not just a worker count cap.

def ingest_at_scale(docs, batch_size, max_concurrent_requests):
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    limiter = RateLimiter(max_concurrent_requests)
    with limiter.semaphore as ex:
        results = list(ex.map(lambda b: [embed_doc(d) for d in b], batches))
    return results
```

The `max_concurrent_requests` cap here is doing the actual safety work — it
is the single number connecting "how fast can ingestion go" to "will the
embedding provider start rejecting requests," and it needs to be set from
the provider's documented limit, not from local trial and error against a
sandbox that doesn't reflect production quota.

## The trap: a fast pipeline that silently drops or duplicates documents under load

Speeding up ingestion introduces failure modes that a slow, careful
sequential pipeline doesn't have:

- **Partial batch failures.** If a batch of 100 documents fails midway
  (network error on document 60), does the pipeline retry the whole batch
  (risking duplicate embeddings for documents 1–59), retry only the failed
  remainder, or drop the batch silently? Each choice has a different
  correctness property, and "just retry the batch" without idempotency
  (Level 3 module 08's content-hash upsert makes this safe — re-embedding an
  unchanged document is a no-op, not a duplicate) is the naive and wrong
  default.
- **Out-of-order completion racing the change-detection logic.** If two
  updates to the same document are in flight concurrently (a fast edit
  followed by another fast edit), a slower worker processing the first
  update can complete *after* a faster worker processing the second,
  overwriting the newer content with the older one. Version numbers or
  timestamps compared at write time — not just "last write wins" — are
  needed once ingestion is concurrent, not just sequential-and-fast.
- **Backpressure ignored.** A pipeline that fires requests as fast as
  possible without checking whether the vector store's write throughput
  can keep up just moves the bottleneck downstream, often turning it into
  either dropped writes or an unbounded queue that eventually runs the
  ingestion worker out of memory.

## Cheat sheet

| Lever | Gains | Ceiling |
|---|---|---|
| Parallel requests | Overlaps I/O wait | Provider's concurrency/rate limit |
| Batching | Amortizes per-request overhead | Diminishing returns once overhead is amortized; bigger failure blast radius |
| Both combined | Multiplies the above | Rate limit is shared — parallel batches can still trip it |
| None of the above, just "run it overnight" | Simplicity | Doesn't scale to same-day freshness at enterprise volume |

## How It Actually Works

**Why embedding calls parallelize almost for free, and why that ceiling is
real.** Each chunk's embedding is computed independently — the encoder's
forward pass for chunk A doesn't depend on chunk B's output at all (unlike
generation's autoregressive per-token dependency, level-3 lesson 7), so
issuing many embedding requests concurrently is limited only by the
embedding service's rate limits and your own concurrency settings, not by
any inherent sequential dependency in the computation. The ceiling shows up
as provider-side rate limiting (requests or tokens per minute) or, for
self-hosted models, GPU memory and batch-size limits — parallelizing past
that ceiling doesn't increase throughput, it just queues requests, which is
why naive "just add more concurrent workers" plans plateau and sometimes
regress (excess concurrent connections adding overhead and retries) past a
provider-specific point.

**Why batching amortizes a fixed cost that exists independently of how much
work parallelization does.** Every API call, whether embedding one chunk or
one hundred, pays a fixed per-request overhead (network round-trip, request
parsing, model-loading amortization on the server side) on top of the
actual per-token compute cost. Batching many chunks into one request spreads
that fixed cost across all of them, while parallelizing many single-chunk
requests pays the fixed cost N times regardless of how concurrently those N
requests execute — batching and parallelism are solving different
inefficiencies (fixed-cost amortization vs. wall-clock time from serial
execution) and combining them (parallel batches, not parallel single
requests) is what actually maximizes throughput, which is exactly why
naive full parallelization without batching leaves throughput on the table.

**Why silent drops and duplicates are the predictable failure mode of a
fast, concurrent pipeline specifically, not of ingestion in general.** A
sequential pipeline that fails on document 500 of 1,000 has an obvious,
visible failure point to resume from. A pipeline running hundreds of
concurrent batched requests has many in-flight operations at any failure
moment — a batch that partially succeeds (some chunks embedded, the request
then times out) can leave the pipeline's bookkeeping unsure whether those
chunks were actually stored, and a naive retry-the-whole-batch-on-timeout
strategy can duplicate the chunks that *did* succeed while a naive
skip-on-timeout strategy can silently drop the ones that didn't — both
failure modes are specifically products of concurrency (many overlapping,
partially-observable operations) rather than something a slower, sequential
pipeline would exhibit, which is why idempotent writes (keyed by the
content hash from level-3 lesson 8, so a duplicate write is a no-op) and
explicit per-chunk status tracking are the correct fix rather than "add more
retries."

## Exercise

Using the `cost_model` function, find the batch size at which doubling it
further saves less than 5% additional time, for your own
`per_request_overhead_ms` and `per_doc_ms` estimates (measure them against a
real embedding API if you have one, or reuse the illustrative values above).
Then extend `ingest_at_scale` to track and report how many documents were
retried due to partial batch failures, distinguishing "retried and
succeeded" from "retried and gave up," so a spike in either becomes visible
in monitoring rather than silently inflating ingestion latency.

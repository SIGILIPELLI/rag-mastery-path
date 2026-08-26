# 04 · Production Vector Databases

ChromaDB, used throughout Levels 1–2, is a local, single-process, embedded
store — perfect for learning and small projects, and not what you reach for
once you have millions of vectors, multiple services querying concurrently, or
metadata filters that need to be fast. This module compares the databases you
actually deploy: **pgvector, Qdrant, Weaviate, and Pinecone** — on indexing
algorithm, filtering, and operational trade-offs. There's no local service to
install for any of these in this sandbox, so this module is manual review of
their documented behavior rather than runnable code, stated plainly where that
applies; the memory and latency math below is real arithmetic, not vendor
marketing numbers.

## The indexing algorithms underneath all of them

Every one of these databases is a UI over one of two families:

- **HNSW** (Hierarchical Navigable Small World) — a multi-layer graph where
  each vector links to its approximate nearest neighbors. Query time is
  logarithmic in dataset size; build time and memory are the cost.
- **IVF** (Inverted File Index) — vectors are clustered (k-means-style) into
  buckets ("cells"); a query only searches the nearest few buckets. Cheaper to
  build and update than HNSW, at some recall cost from bucket-boundary misses.

Qdrant and Weaviate default to HNSW. Pinecone uses a proprietary variant
described as graph-based (HNSW-family). pgvector supports both HNSW and IVFFlat
as index types you choose explicitly.

## Memory is the first wall you hit

```python
n = 10_000_000          # 10M vectors — a real mid-size production corpus
d = 768                  # embedding dimension (e.g. a BERT-family model)
bytes_per_vec = d * 4    # float32

flat_gb = n * bytes_per_vec / 1e9
print("flat index memory:", round(flat_gb, 1), "GB")

# HNSW stores the vectors AND a graph: ~M neighbor links per node per layer
M = 16                    # typical HNSW fan-out
graph_overhead_gb = n * M * 4 * 1.3 / 1e9   # 4 bytes/link, ~1.3x for multi-layer
print("HNSW graph overhead:", round(graph_overhead_gb, 2), "GB")
print("HNSW total:", round(flat_gb + graph_overhead_gb, 1), "GB")
```

Captured output:

```
flat index memory: 30.7 GB
HNSW graph overhead: 0.83 GB
HNSW total: 31.6 GB
```

At 10M vectors and 768 dimensions, you're at ~31GB before any metadata,
replication, or OS overhead — and that's the *good* case, because HNSW's graph
overhead is small relative to the vectors themselves. The number that actually
bites teams is the **31GB has to fit in RAM** for HNSW query latency to stay
low; falling back to disk turns single-digit-millisecond queries into
double-digit or worse. This is the real reason people reach for a managed
vector DB instead of "just run FAISS on a box" — the memory-provisioning
problem doesn't go away, it just becomes someone else's dashboard.

## Comparison

| | pgvector | Qdrant | Weaviate | Pinecone |
|---|---|---|---|---|
| Model | Postgres extension | Standalone service | Standalone service | Managed only, no self-host |
| Index | HNSW or IVFFlat, your choice | HNSW | HNSW | Proprietary (HNSW-family) |
| Filtering | Native SQL `WHERE`, exact | Payload filters, pre/post configurable | GraphQL filters | Metadata filters |
| Scaling | Vertical (one Postgres), or Citus/read replicas | Horizontal sharding built in | Horizontal sharding built in | Fully managed, opaque scaling |
| Ops burden | You run Postgres (you probably already do) | New service to operate | New service to operate | None — but vendor lock-in |
| Best fit | Already on Postgres, moderate scale, want joins | Self-hosted, need speed + filters | Self-hosted, want built-in hybrid search | No ops team, will pay for it |

## The trap: filtered search is not "search, then filter"

The naive mental model — "find the nearest neighbors, then throw out the ones
that don't match my metadata filter" — silently breaks recall. If a query asks
for `top_k=5` results `WHERE tenant_id = 'acme'`, and the ANN index's true
nearest 5 neighbors happen to belong to *other* tenants, post-filtering
returns fewer than 5 results — or zero — even though acme has plenty of
relevant vectors further down the ranked list.

This is exactly the mechanism behind the most severe production RAG bug
class: **a tenant sees no results (annoying) or, worse, sees another tenant's
data because a filter was applied at the wrong stage (a breach)**. The correct
approaches, in order of how the databases above actually implement it:

- **Pre-filtering**: restrict the candidate set to matching metadata *before*
  ANN search runs. Qdrant and Weaviate support this natively; it's the safest
  default for multi-tenant systems (Level 4 module 2 covers this in depth).
- **Filtered HNSW traversal**: some implementations walk the graph while
  skipping non-matching nodes, avoiding a separate pre-filter pass — faster,
  but implementation-dependent recall guarantees.
- **Post-filtering with over-fetching**: fetch `top_k * N` candidates, filter,
  hope enough survive — a real technique, but silently degrades as your
  filter becomes more selective (a filter matching 1% of vectors needs
  `N≈100` to reliably return `top_k`, which erases the speed advantage of ANN
  search in the first place).

Always check, per database, which mode a given filter syntax triggers by
default — it is not always documented as clearly as the query API itself.

## Cheat sheet

| Symptom | Likely cause |
|---|---|
| Recall drops when combined with a filter | Post-filtering instead of pre-filtering |
| Query latency degrades as data grows past RAM | HNSW graph no longer fits in memory |
| Index build is slow, queries are fast | Expected HNSW trade-off; IVF is the inverse |
| Cross-tenant data appears in results | Filter applied after ANN search, or missing entirely |
| Postgres team says "just use pgvector" | Right call if scale and filter complexity are moderate |

## Exercise

Recompute the memory math for a 100M-vector, 1536-dimension corpus (OpenAI
`text-embedding-3-large` size) with the same HNSW parameters, and figure out
at what vector count the index stops fitting in a single machine with 128GB
RAM. Then write down, in a comment, which of the four databases' scaling
models (vertical vs. built-in horizontal) you'd pick to cross that line, and
why.

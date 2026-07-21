# 05 · Retrieval

Retrieval is the moment of truth in a RAG pipeline: given a question, fetch
the chunks that contain the answer. Everything before it (chunking, embedding,
storage) exists to make this step work; everything after it (prompting,
generation) can only be as good as what it returns. This lesson covers top-k
search, choosing k, metadata filtering, similarity thresholds, and the basics
of judging retrieval quality — over a corpus big enough to be interesting.

## Setup: a corpus worth searching

```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("kb", metadata={"hnsw:space": "cosine"})

docs = [
    # (text, source, topic)
    ("Annual plans can be refunded within 14 days of purchase or renewal.", "billing.md", "refunds"),
    ("Monthly plans are non-refundable but can be cancelled at any time.", "billing.md", "refunds"),
    ("Invoices are emailed on the 1st of each month to the account owner.", "billing.md", "invoicing"),
    ("Contact billing@example.com for invoice and payment questions.", "billing.md", "invoicing"),
    ("To reset your password, click 'Forgot password' on the login page.", "account.md", "auth"),
    ("Reset links expire after 30 minutes for security reasons.", "account.md", "auth"),
    ("Two-factor authentication can be enabled under Settings > Security.", "account.md", "auth"),
    ("Deploys to production happen automatically when main is merged.", "engineering.md", "deploys"),
    ("Rollbacks are performed with the 'deploy --rollback' command.", "engineering.md", "deploys"),
    ("Staging mirrors production and refreshes its data nightly.", "engineering.md", "deploys"),
    ("Data can be exported as CSV or JSON from the Reports page.", "features.md", "export"),
    ("API rate limits are 1000 requests per minute per key.", "features.md", "api"),
]

collection.add(
    ids=[f"c{i}" for i in range(len(docs))],
    embeddings=model.encode([t for t, _, _ in docs]).tolist(),
    documents=[t for t, _, _ in docs],
    metadatas=[{"source": s, "topic": tp} for _, s, tp in docs],
)
```

## Top-k similarity search

A reusable retrieve function — this exact shape appears in every RAG system:

```python
def retrieve(question: str, k: int = 3, where: dict | None = None) -> list[dict]:
    res = collection.query(
        query_embeddings=[model.encode(question).tolist()],
        n_results=k,
        where=where,
    )
    return [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]

for hit in retrieve("how do I undo a bad release?"):
    print(f"{hit['distance']:.3f}  [{hit['meta']['source']}]  {hit['text']}")
```

```text
0.517  [engineering.md]  Rollbacks are performed with the 'deploy --rollback' command.
0.671  [engineering.md]  Deploys to production happen automatically when main is merged.
0.729  [engineering.md]  Staging mirrors production and refreshes its data nightly.
```

"Undo a bad release" shares no keywords with "rollback" — semantic retrieval
finds it anyway, and the entire top-3 comes from the right file.

### Choosing k

- **k too small (1):** one near-miss and the answer isn't in the context at
  all. No redundancy.
- **k too large (20):** the answer is present but buried in noise — which
  costs tokens and measurably hurts the LLM's accuracy (lesson 9's
  "lost in the middle").
- **Start at k=3–5** for well-chunked corpora. The real answer comes from
  evaluation (lesson 8): raise k until hit-rate plateaus, then stop.

## Metadata filtering

Semantic similarity can't express hard constraints — "only current policy
docs", "only this customer's data". Filters can:

```python
# Without filter: "refresh" matches the staging doc from engineering.md
print(retrieve("when does data refresh?", k=1)[0]["text"])
# Staging mirrors production and refreshes its data nightly.

# The user was asking about *exports* — scope the search:
print(retrieve("when does data refresh?", k=1, where={"source": "features.md"})[0]["text"])
# Data can be exported as CSV or JSON from the Reports page.
```

Operators compose: `where={"topic": {"$in": ["refunds", "invoicing"]}}`,
`where={"$and": [{"source": "billing.md"}, {"topic": "refunds"}]}`. In
multi-tenant systems (Level 4), a tenant filter isn't an optimization — it's
the security boundary.

## When retrieval finds nothing good

Every query returns *k results* — nearest neighbors always exist, even for a
question your corpus can't answer:

```python
for hit in retrieve("what is the meaning of life?", k=2):
    print(f"{hit['distance']:.3f}  {hit['text']}")
# 0.943  Staging mirrors production and refreshes its data nightly.
# 0.955  Two-factor authentication can be enabled under Settings > Security.
```

Distances near 1.0 are the store shrugging. If you pass these to the LLM as
"context", you're *inviting* hallucination. Defend with a threshold:

```python
def retrieve_confident(question: str, k: int = 3, max_distance: float = 0.8) -> list[dict]:
    hits = [h for h in retrieve(question, k) if h["distance"] <= max_distance]
    return hits   # may legitimately be empty!

hits = retrieve_confident("what is the meaning of life?")
if not hits:
    print("No relevant documents found.")   # ← the honest answer
```

The 0.8 cutoff here is illustrative, not universal — thresholds depend on the
embedding model and corpus. Calibrate by eyeballing distances for a handful of
known-answerable and known-unanswerable questions (or properly, with the eval
set you'll build in lesson 8).

## Judging retrieval quality

Before any formal metrics, use the two-question smoke test on every pipeline
change:

1. **Is the answer in the top-k?** (If not: nothing downstream can save you.)
2. **How much of the top-k is junk?** (Signal-to-noise for the LLM.)

Lesson 8 turns these into numbers — hit rate and MRR over a golden question
set. The habit to build now: whenever an answer looks wrong, **print the
retrieved chunks first**. Nine times out of ten, the bug is visible right
there.

## Cheat sheet

| Concept | Takeaway |
|---------|----------|
| Top-k search | Return the k nearest chunks to the query vector |
| Choosing k | Start 3–5; tune with eval, not intuition |
| `where=` filter | Hard constraints inside the search; also the tenant/security boundary |
| Filter operators | `$in`, `$ne`, `$gt`, `$and`, `$or` |
| No-answer queries | Still return k results — check distances |
| Distance threshold | Drop hits above ~0.8 distance (calibrate per model/corpus) |
| Empty result | A feature, not a bug — enables honest "I don't know" |
| Debug habit | Always print retrieved chunks before blaming the LLM |

## Exercise

Using the corpus above (or your own from lesson 4's exercise), write 6 test
questions: four answerable (one per source file) and two unanswerable. For
each, run `retrieve(q, k=3)` and record (a) whether the correct chunk appears
in the top 3, and at which rank, and (b) the top hit's distance. Then pick a
`max_distance` threshold that accepts all four answerable questions and
rejects both unanswerable ones. Is such a threshold possible for your corpus?
If not, which question makes it impossible — and would a different phrasing,
chunk, or filter fix it?

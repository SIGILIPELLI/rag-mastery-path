# 04 · Vector Stores (ChromaDB)

In lesson 2 you searched a corpus by comparing the query vector against every
document vector in a Python loop. A **vector store** (or vector database) is
the production version of that loop: it stores vectors alongside their text
and metadata, indexes them for fast nearest-neighbor search, filters on
metadata, and persists everything to disk. This lesson uses
[ChromaDB](https://www.trychroma.com/) — free, local, no API key — and also
shows when a plain numpy array is honestly all you need.

## What a vector database actually does

Four jobs:

1. **Store** — each record is `(id, vector, document text, metadata dict)`.
2. **Index** — organize vectors so "find the k nearest" doesn't require
   comparing against every vector. At scale this uses approximate
   nearest-neighbor (ANN) structures like HNSW graphs: near-perfect results
   in a fraction of the time.
3. **Query** — take a query vector, return the top-k nearest records with
   their distances.
4. **Filter** — restrict search to records whose metadata matches
   (`source == "policy.md"`), which pure vector math can't do.

## First steps with ChromaDB

```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()          # in-memory for now; persistence below

collection = client.create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},   # use cosine distance (default is L2)
)
```

A **collection** is Chroma's unit of organization — one collection per corpus
per embedding model (remember the same-model rule from lesson 2).

## Adding chunks with metadata

We embed with our own model and pass vectors explicitly — that way you always
know exactly which model is in play:

```python
chunks = [
    "Annual plans can be refunded within 14 days of purchase or renewal.",
    "Monthly plans are non-refundable but can be cancelled at any time.",
    "To reset your password, click 'Forgot password' on the login page.",
    "Deploys to production happen automatically when main is merged.",
]
metadatas = [
    {"source": "policy.md", "section": "Refunds"},
    {"source": "policy.md", "section": "Refunds"},
    {"source": "help.md",   "section": "Account"},
    {"source": "eng.md",    "section": "Deploys"},
]

collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],   # ids must be unique strings
    embeddings=model.encode(chunks).tolist(),
    documents=chunks,
    metadatas=metadatas,
)
print(collection.count())  # 4
```

!!! warning "Chroma can embed for you — know what you're getting"
    If you omit `embeddings=`, Chroma silently embeds documents with its own
    default model (also `all-MiniLM-L6-v2`, as it happens). Convenient, but if
    you ever query with vectors from a *different* model, results turn to
    noise with no error. Passing embeddings explicitly keeps the contract
    visible.

## Querying

```python
q = "can I get my money back?"
results = collection.query(
    query_embeddings=[model.encode(q).tolist()],
    n_results=2,
)

for doc, meta, dist in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(f"{dist:.3f}  [{meta['source']} § {meta['section']}]  {doc}")
```

```text
0.362  [policy.md § Refunds]  Annual plans can be refunded within 14 days of purchase or renewal.
0.616  [policy.md § Refunds]  Monthly plans are non-refundable but can be cancelled at any time.
```

Chroma returns **cosine distance** (`1 - similarity`), so *smaller is better*:
0.362 distance is the 0.638 similarity you computed by hand in lesson 2. The
results are lists-of-lists because you can send several queries at once —
hence the `[0]` indexing.

### Metadata filtering

```python
results = collection.query(
    query_embeddings=[model.encode("how do releases work?").tolist()],
    n_results=2,
    where={"source": "eng.md"},          # only search engineering docs
)
print(results["documents"][0])
# ['Deploys to production happen automatically when main is merged.']
```

Filters (`where={"source": ...}`, with operators like `$in`, `$ne`, `$gt`)
run *inside* the store, combining structured constraints with semantic search
— the workhorse of "search only this product's docs" features. More in
lesson 5.

## Persistence

Swap one line and the index survives restarts:

```python
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="docs", metadata={"hnsw:space": "cosine"}
)
```

Everything is written to `./chroma_db/`. On the next run,
`get_or_create_collection` picks up the existing data — so ingestion becomes
something you run only when documents change. Updates use the same ids:
`collection.upsert(ids=..., ...)` overwrites existing records,
`collection.delete(ids=[...])` removes them.

## When a numpy array is honestly enough

A vector DB is infrastructure. Below a few tens of thousands of chunks,
brute-force numpy search is simpler and *fast*:

```python
import numpy as np

class TinyVectorStore:
    def __init__(self, model):
        self.model = model
        self.texts: list[str] = []
        self.vecs: np.ndarray | None = None

    def add(self, texts: list[str]) -> None:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        self.texts += texts
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])

    def query(self, text: str, k: int = 3) -> list[tuple[float, str]]:
        q = self.model.encode([text], normalize_embeddings=True)[0]
        sims = self.vecs @ q                     # normalized → dot = cosine sim
        top = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.texts[i]) for i in top]

store = TinyVectorStore(model)
store.add(chunks)
print(store.query("refund", k=2))
```

Thirty lines, no dependencies beyond numpy, exact (not approximate) results,
and for 10,000 chunks a query is a few milliseconds. Reach for a real vector
store when you need **persistence, metadata filtering, incremental updates,
or scale** — not because the tutorial you read used one.

## Cheat sheet

| Operation | ChromaDB code |
|-----------|---------------|
| In-memory client | `chromadb.Client()` |
| Persistent client | `chromadb.PersistentClient(path="./chroma_db")` |
| Create/open collection | `client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})` |
| Add records | `collection.add(ids=, embeddings=, documents=, metadatas=)` |
| Query | `collection.query(query_embeddings=[...], n_results=k)` |
| Filter | `where={"source": "policy.md"}` |
| Update / delete | `collection.upsert(...)` / `collection.delete(ids=[...])` |
| Distance semantics | Cosine *distance*: smaller = more similar |
| Skip the DB when | Small corpus, no persistence/filtering needs → numpy |

## Exercise

Build a persistent index of the lesson-3 chunker's output: take 2–3 real text
or Markdown files, run `chunk_text` over each, and add every chunk to a
persistent ChromaDB collection with metadata `{"source": filename, "chunk":
i}`. Run the script twice and prove persistence works (hint: guard ingestion
with `if collection.count() == 0:` or use deterministic ids + `upsert`). Then
query it with three questions — one answerable from each file — and check that
the top hit's `source` metadata points at the right file each time.

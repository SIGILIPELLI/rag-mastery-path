---
description: "Incremental Indexing & Freshness — Every pipeline so far builds an index once, from a fixed snapshot of documents, and never touches it again. Real…"
---

# 08 · Incremental Indexing & Freshness

Every pipeline so far builds an index once, from a fixed snapshot of
documents, and never touches it again. Real corpora change: tickets get
resolved, policies get updated, pages get deleted. Re-embedding and
re-indexing the entire corpus on every change doesn't scale — for a
million-chunk corpus, a one-line edit to one document shouldn't cost a
million embedding calls. This module covers content hashing, upserts and
deletes, and the change-detection pattern (CDC) that makes incremental
updates correct instead of just fast.

## Content hashing: know what actually changed

```python
import hashlib

def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

class Store:
    def __init__(self):
        self.docs = {}  # doc_id -> (content_hash, content, version)

store = Store()

def upsert(store, doc_id, content):
    h = content_hash(content)
    existing = store.docs.get(doc_id)
    if existing and existing[0] == h:
        return "unchanged"          # identical content — skip re-embedding
    version = (existing[2] + 1) if existing else 1
    store.docs[doc_id] = (h, content, version)
    return "reindexed" if existing else "inserted"
```

```python
print(upsert(store, "doc1", "Refunds within 30 days."))
print(upsert(store, "doc1", "Refunds within 30 days."))  # re-synced, no change
print(upsert(store, "doc1", "Refunds within 45 days."))  # actually changed
print(store.docs["doc1"])
```

Captured output:

```
inserted
unchanged
reindexed
('766b6d184198d2aaf44d300bd9e5d0c065d1f5794a3898acfb00d11ea0e7de43', 'Refunds within 45 days.', 2)
```

The second call — same content, re-synced from source — correctly does
nothing, avoiding a wasted embedding call. This matters at scale: a nightly
sync job that re-reads every source document should only pay embedding cost
for the documents that actually changed, and the hash comparison is what
tells you which ones those are, without needing to diff content byte-by-byte
or trust source-system timestamps (which are frequently wrong or absent).

## Deletes: the part teams forget

An index that only handles inserts and updates silently accumulates stale
entries for documents that were removed from the source. A retriever will
happily keep returning a deleted refund policy forever unless deletion is
handled explicitly:

```python
def sync(store, source_docs):
    """Full reconciliation: source_docs is the authoritative current set."""
    seen = set()
    actions = {"inserted": 0, "reindexed": 0, "unchanged": 0, "deleted": 0}
    for doc_id, content in source_docs.items():
        seen.add(doc_id)
        actions[upsert(store, doc_id, content)] += 1
    for doc_id in [d for d in store.docs if d not in seen]:
        del store.docs[doc_id]
        actions["deleted"] += 1
    return actions

source = {"doc1": "Refunds within 45 days.", "doc2": "New doc."}
print(sync(store, source))
print(sorted(store.docs.keys()))
```

Captured output:

```
{'inserted': 1, 'reindexed': 0, 'unchanged': 1, 'deleted': 0}
['doc1', 'doc2']
```

`doc1` was unchanged (correct — same hash as before), `doc2` was newly
inserted, and nothing was deleted because both source documents were present.
Had a previously-indexed `doc3` been missing from `source_docs`, it would show
up in `deleted` and be removed from `store.docs` — this reconciliation pattern
(diff the full current source set against the full indexed set) is what
catches deletions that a pure "process each incoming change" pipeline misses,
because there's no explicit "delete" event to process — the document just
stops appearing.

## Change-data-capture (CDC) vs. full reconciliation

The `sync()` function above does a **full reconciliation**: read every source
document, hash-compare against every indexed document. That's correct but
`O(corpus size)` per sync, even when only one document changed. **CDC**
inverts this: the source system emits an event (webhook, database trigger,
message queue) exactly when something changes, and the indexer processes
only that event — `O(1)` per change instead of `O(corpus size)` per sync.

| | Full reconciliation | CDC |
|---|---|---|
| Cost per sync | O(corpus size) | O(changes) |
| Catches deletes | Yes, naturally | Only if delete events are emitted reliably |
| Failure mode | Slow at scale | Missed events = permanently stale entries |
| When to use | Small/medium corpus, or as a periodic correctness sweep | Large corpus with a reliable event source |

Production systems commonly run both: CDC for low-latency updates, plus a
slower full-reconciliation sweep (nightly or weekly) as a correctness
backstop for any CDC events that were dropped, delayed, or never emitted
because of an upstream bug.

## Freshness metadata in answers

Once an index can go stale between syncs, the answer itself should say so
when relevant:

```python
def answer_with_freshness(retrieved_doc, current_time, source):
    version, indexed_at = source[retrieved_doc]
    age_hours = (current_time - indexed_at) / 3600
    note = f" (source last verified {age_hours:.0f}h ago)" if age_hours > 24 else ""
    return note
```

Surfacing "last verified Xh ago" alongside an answer is cheap to compute and
meaningfully changes how a user should trust a time-sensitive fact (pricing,
policy, inventory) versus a stable one (a product's founding date). This is a
UX decision as much as an engineering one, but it depends entirely on
`indexed_at` metadata that has to be tracked at ingestion time — it can't be
reconstructed after the fact.

## The trap: re-embedding cost hides in "just update the metadata"

A common shortcut: "the document's category tag changed, but the text
didn't — just update the metadata field, skip re-embedding." That's correct
*if* your vector store keeps embeddings and metadata separable and your
retrieval doesn't re-derive anything from metadata into the embedding space.
It's wrong if metadata is baked into the embedded text (a common pattern:
`f"[category: {cat}] {chunk_text}"`) — then a metadata-only change silently
invalidates the embedding, and skipping re-embedding leaves a subtly
mismatched vector in the index indefinitely. Know which pattern your
pipeline uses before deciding a change is "metadata-only."

## Cheat sheet

| Symptom | Cause | Fix |
|---|---|---|
| Stale info in answers | Deletes not handled, or sync not running | Add explicit delete reconciliation |
| Full re-embed on every sync | No content hashing | Hash-compare before re-embedding |
| Occasional permanently-stale doc | CDC event dropped, no backstop | Add periodic full reconciliation |
| Metadata update produces wrong answers | Metadata baked into embedded text | Re-embed on metadata change too, or separate metadata from embedded text |
| Users trust stale time-sensitive facts | No freshness surfaced in answer | Track and show `indexed_at` age |

## How It Actually Works

**Why content hashing, not modification timestamps, is the correct
"did this change" check.** File-system mtimes get touched by operations that
don't change content at all (a re-save with identical text, a permissions
change, a copy during backup/sync) and can be missing or unreliable across
storage backends (object stores like S3 handle timestamps differently than a
local filesystem, and a CMS export can reset every mtime to the export
time). Hashing the actual chunk content (a fast non-cryptographic hash like
xxHash, or SHA-256 if you need collision resistance) and comparing against
the hash stored at last ingestion gives a check that is true content
equality, immune to metadata noise — if the hash matches, re-embedding that
chunk is provably wasted work, because the embedding model is a
deterministic function of the input text and would produce the identical
vector again.

**Why deletes are structurally easy to forget and expensive to get wrong.**
Re-ingestion pipelines are naturally biased toward "process what's here" —
walk the current document set, hash and diff each one, upsert what changed.
A deleted source document simply isn't in that walk anymore, so nothing in
the natural flow of the pipeline notices its absence; its chunks and vectors
silently remain in the index forever unless the pipeline explicitly diffs
"what's indexed" against "what currently exists" and removes the difference.
This is a pure liability, not a performance issue: a stale-but-still-indexed
chunk from a deleted document can retrieve and get cited in an answer for a
document that, from the user's perspective, no longer exists — arguably
worse than the "stale index" failure mode in level-1 lesson 9, because the
source of truth doesn't just disagree with the index, it has no record of
the fact at all anymore.

**Why re-embedding cost hides specifically in metadata-only updates.**
Updating a chunk's metadata (say, its `access_tier` after a permissions
change) feels like it should be nearly free — but if the update path
re-runs the same code path as full re-ingestion, it re-embeds text whose
embedding hasn't changed at all, paying the embedding model's full
per-chunk cost for zero semantic benefit. The content hash from above is
exactly the guard that should gate this: hash matches → update the metadata
fields in place in the vector store without touching the embedding; hash
differs → re-embed. Skipping this check is how "just update the metadata"
silently becomes "re-embed the entire corpus" at scale.

## Exercise

Add a `deleted_ids` return value to `sync()` (currently it only counts them)
and use it to also remove those documents' vectors from a parallel
`embeddings` dict, keeping the two stores consistent. Then simulate a CDC
event being dropped — call `upsert` for `doc3` but never let it reach
`source_docs` — and show that only a subsequent `sync()` call (the
reconciliation backstop), not another CDC event, catches and removes it.

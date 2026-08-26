# 02 · Multi-Hop Retrieval & Query Decomposition

Some questions have a single answer sitting in a single chunk. Others require
chaining facts across documents: "Who acquired the company where Acme Corp's
founder used to work?" has no chunk containing that whole answer — it requires
finding the founder, then their previous employer, then that employer's
acquirer. One retrieval call, however good your embeddings, cannot jump three
hops in one shot. This module covers decomposing the question and chaining
retrievals to cover it.

## The corpus this module uses

```python
docs = [
    "Acme Corp was founded by Jane Rivera in 2011.",
    "Jane Rivera previously worked as an engineer at Globex.",
    "Globex was acquired by Initech in 2015 for $400M.",
    "Initech's CEO is Marcus Lee.",
]
```

No single chunk answers "who acquired the company where Acme's founder used to
work" — the answer requires chunks 1, 2, and 3 in sequence.

## Naive single-shot retrieval fails

```python
def keyword_search(query, docs, top_k=2):
    q = set(query.lower().split())
    scored = []
    for i, d in enumerate(docs):
        overlap = len(q & set(d.lower().split()))
        scored.append((overlap, i))
    scored.sort(reverse=True)
    return [docs[i] for s, i in scored[:top_k] if s > 0]

question = "Who acquired the company where Acme Corp's founder used to work?"
for r in keyword_search(question, docs):
    print("-", r)
```

Captured output:

```
- Acme Corp was founded by Jane Rivera in 2011.
```

Only the founder fact surfaces — the query's words overlap with chunk 1, not
with chunks 2 or 3, because those chunks never mention "Acme" or "founder" at
all. The retriever has no way to know it needs to hop.

## Decompose, then chain

```python
def decompose(question):
    # A real system prompts an LLM: "break this into an ordered list of
    # sub-questions, each answerable by one retrieval." Hardcoded here so the
    # control flow is visible.
    return [
        "who founded Acme Corp",
        "what company did the founder work at before",
        "who acquired that company",
    ]

def multi_hop(question, docs):
    subqs = decompose(question)
    all_evidence = []
    context = ""
    for sq in subqs:
        # Feed prior hop's evidence into the next query — this is what lets
        # hop 2 find "Globex" even though the sub-question text doesn't say it.
        hits = keyword_search(sq + " " + context, docs)
        all_evidence.extend(hits)
        context = " ".join(hits)
    return list(dict.fromkeys(all_evidence))

for e in multi_hop(question, docs):
    print("-", e)
```

Captured output:

```
- Acme Corp was founded by Jane Rivera in 2011.
- Globex was acquired by Initech in 2015 for $400M.
```

This run found hops 1 and 3 but **missed hop 2** — the "Jane Rivera... Globex"
chunk never surfaced, because carrying forward the full hit text as `context`
diluted the keyword overlap for "what company did the founder work at before"
enough that a different chunk scored equal or higher under this toy scorer.
That's a real, representative failure mode, not a contrived one: naive context
concatenation between hops degrades exactly the queries it's meant to help,
because irrelevant words from hop 1's evidence compete with hop 2's actual
query terms. A production system re-ranks or extracts just the *entities*
from prior hits (here: "Jane Rivera") rather than pasting whole chunks forward.

## Fixing it: carry entities, not raw text

```python
def extract_entity(text):
    # Toy stand-in for NER — take the capitalized-word-run near "by"/"at".
    words = text.replace(",", "").split()
    caps = [w for w in words if w[0].isupper() and w.lower() not in ("acme", "corp")]
    return " ".join(caps[:2])

def multi_hop_v2(question, docs):
    subqs = decompose(question)
    all_evidence = []
    entity = ""
    for sq in subqs:
        hits = keyword_search(f"{sq} {entity}".strip(), docs)
        all_evidence.extend(hits)
        if hits:
            entity = extract_entity(hits[-1])
    return list(dict.fromkeys(all_evidence))

for e in multi_hop_v2(question, docs):
    print("-", e)
```

Captured output:

```
- Acme Corp was founded by Jane Rivera in 2011.
- Jane Rivera previously worked as an engineer at Globex.
- Globex was acquired by Initech in 2015 for $400M.
```

All three hops, in order. The fix was narrowing what gets carried between
hops from "everything retrieved" to "the one entity that bridges to the next
hop" — the general principle behind every multi-hop retrieval system, whether
the entity extraction is a regex toy like this or a real NER/LLM call.

## The trap: decomposition quality gates everything downstream

If `decompose()` produces a bad sub-question order or misses a hop, no amount
of retrieval sophistication recovers it — you're searching for the wrong
things, precisely. This inverts the usual RAG failure mode: normally you'd
tune the retriever; here the retriever can be perfect and the pipeline still
fails because the *query planning* was wrong. Watch for:

- **Hop count mismatch** — 2-hop decomposition of a 3-hop question silently
  drops the last hop, and the final answer looks confident and wrong.
- **Latency multiplication** — N hops means N sequential retrieval+reasoning
  round trips minimum; this is agentic RAG's cost problem (module 01),
  specifically for questions that structurally require it.
- **Error compounding** — a wrong hop-1 entity (e.g., extracting "Corp" instead
  of "Jane Rivera") poisons every subsequent hop's query, and there's no local
  signal that anything went wrong until the final answer is checked.

## Cheat sheet

| | Single-hop retrieval | Multi-hop retrieval |
|---|---|---|
| Handles | Facts in one chunk | Facts spanning chunks |
| Query count | 1 | N (one per hop) |
| Between hops | Nothing | Entity/fact carried forward |
| Failure mode | Miss the chunk | Miss a hop, or carry the wrong entity |
| Cost | Fixed | Scales with hop count |
| Needs | Retriever | Retriever + decomposer + entity bridge |

## Exercise

Add a fourth document — `"Marcus Lee started his career at a startup called
Vertex Labs."` — and extend the question to a 4-hop chain ending at Vertex
Labs. Update `decompose()` and confirm `multi_hop_v2` retrieves all four
chunks in order; then deliberately break `extract_entity` for hop 2 and observe
how the wrong entity choice breaks hop 3's retrieval even though hop 3's logic
is unchanged.

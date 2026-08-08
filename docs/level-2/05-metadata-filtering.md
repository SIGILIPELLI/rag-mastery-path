# 05 · Metadata Design & Filtered Retrieval

Semantic similarity cannot express "only documents this user is allowed to
see", "only the current policy version", or "only tickets from last quarter".
Those are not fuzzy preferences — they are hard constraints, and no amount of
embedding quality will enforce them. Metadata will.

Level 1 lesson 5 used a `where=` filter as a convenience. This lesson treats
metadata as what it actually is in production: a schema you design up front,
a correctness boundary, and — in multi-tenant systems — a security boundary.

!!! note "What ran here"
    All code below ran locally with `rank-bm25` and a hand-rolled filter
    predicate. Using plain Python for the filter is deliberate: it makes the
    pre-filter vs post-filter distinction visible, which is exactly the thing
    vector-store APIs hide from you.

## Designing the schema

Metadata you did not capture at ingestion time does not exist. Re-indexing a
large corpus to add a field is expensive and, if the source is gone, sometimes
impossible. So the design question is: **what will someone want to filter on
in six months?**

A practical starting schema:

```python
{
    "source": "billing.md",       # provenance - always capture this
    "doc_type": "policy",         # policy | runbook | howto | reference
    "year": 2025,                 # recency, for staleness filtering
    "tenant_id": "acme",          # security boundary in multi-tenant systems
    "version": "2.3",             # which product version this describes
    "acl": ["employees"],         # who may see it
    "indexed_at": 1754600000,     # for incremental re-indexing
}
```

Four rules that will save you a re-index:

1. **Always store provenance** (`source`, and ideally a URL and heading path).
   You need it for citations, and citations are what make RAG trustworthy.
2. **Store dates as numbers**, not strings. `year: 2025` supports `$gte`;
   `"2025-01-01"` supports string equality and lies to you about ordering.
3. **Keep values low-cardinality where you can.** `doc_type: "policy"` filters
   efficiently; a free-text `summary` field does not filter at all.
4. **Capture the security fields on day one**, even in a single-tenant
   prototype. Retrofitting `tenant_id` after launch is a migration *and* an
   incident report.

## Filtered retrieval

```python
def matches(meta, where):
    for key, cond in (where or {}).items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$gte" and not (val >= target): return False
                if op == "$in" and val not in target: return False
                if op == "$ne" and val == target: return False
        elif val != cond:
            return False
    return True

def search(query, k=3, where=None, prefilter=True):
    scores = bm25.get_scores(tok(query))
    if prefilter:
        pool = [i for i in range(len(TEXTS)) if matches(META[i], where)]
    else:
        pool = sorted(range(len(TEXTS)), key=lambda i: -scores[i])[:k]
        pool = [i for i in pool if matches(META[i], where)]
    order = sorted(pool, key=lambda i: -scores[i])[:k]
    return [(IDS[i], scores[i], META[i], TEXTS[i]) for i in order]
```

```text
query: 'support response time'

no filter:
   2.20 [d1] policy    2025  Monthly plans are non-refundable but can be cancelle
   1.73 [d15] policy    2023  Community support is handled on the public forum wit
   1.66 [d14] policy    2025  Enterprise customers get a dedicated support channel

where doc_type='policy' and year>=2025:
   2.20 [d1] policy    2025  Monthly plans are non-refundable but can be cancelle
   1.66 [d14] policy    2025  Enterprise customers get a dedicated support channel
   0.00 [d0] policy    2025  Annual plans can be refunded within 14 days of purch
```

The filter dropped the 2023 community-support document. If that document
describes a policy you have since changed, this filter is the difference
between a correct answer and a confidently outdated one — a failure mode users
find much harder to forgive than "I don't know".

But notice the third row of the filtered result: **score 0.00**. Once the filter
shrank the pool, BM25 padded the top-3 with a document that matches nothing.
Filters make zero-score results *more* likely, so the `score > 0` guard from
lesson 1 becomes more important, not less.

## Pre-filter vs post-filter

This is the distinction that quietly breaks production systems.

- **Pre-filter**: restrict the candidate set *first*, then rank within it.
- **Post-filter**: retrieve the global top-k, *then* discard non-matching hits.

```python
w = {"source": "deploys.md"}
print("pre-filter :", [r[0] for r in search("data", k=3, where=w, prefilter=True)])
print("post-filter:", [r[0] for r in search("data", k=3, where=w, prefilter=False)])
```

```text
pre-filter : ['d10', 'd8', 'd9']
post-filter: ['d10']
```

Same query, same filter, same `k`. Pre-filtering returns three documents;
post-filtering returns **one**. Two of the three matching documents were never
in the global top-3, so post-filtering threw them away before the filter ever
ran.

This is the **k-collapse trap**, and it gets worse as the filter gets more
selective. Filter to one tenant out of a thousand and post-filtering will
usually return an empty list from a corpus that contains a perfect answer — and
your pipeline will report "no relevant documents found" with total confidence.

The nasty part is that both behave *identically in testing* with loose filters,
and diverge only under the narrow filters real users hit.

| | Pre-filter | Post-filter |
|---|---|---|
| Result count at k | Reliable | Collapses under selective filters |
| Recall with narrow filters | Correct | Silently catastrophic |
| Cost | Filter must be indexed, or you scan | Cheap — reuses the plain search |
| ANN index interaction | Can degrade HNSW graph traversal | Index untouched |
| Safe for security filters | **Yes** | **No** |
| Typical fix when slow | Over-fetch (`k × 10`) then post-filter | — |

Know which one your vector store does. Chroma and Qdrant pre-filter; several
libraries and naive implementations post-filter. If yours post-filters and you
cannot change it, **over-fetch aggressively** — retrieve `k × 20` and filter
down — and monitor how often you still come up short.

## Metadata as a security boundary

In a multi-tenant system, `where={"tenant_id": current_user.tenant}` is not an
optimization. It is the entire thing preventing customer A from reading
customer B's documents through your chatbot.

Treat it accordingly:

- **Never** build the filter from LLM output or from anything in the user's
  message. Build it server-side from the authenticated session.
- Make it structurally impossible to query without it — wrap the store so the
  tenant filter is injected by the retriever, not passed by callers.
- Pre-filter only. A post-filtered security check that returns fewer results is
  a bug; one that returns *more* is a breach.
- Test it explicitly with a golden query designed to surface another tenant's
  document, and assert it comes back empty.

## Traps

- **Over-filtering into emptiness.** Stacking `doc_type` + `year` + `version`
  can leave zero candidates. Decide the fallback deliberately: relax the
  soft filters (recency) while keeping hard ones (tenant, ACL), and tell the
  user you widened the search.
- **Filtering on fields the user did not ask about.** Auto-applying
  `year >= 2025` hides legitimate historical questions. Recency is usually
  better as a *ranking boost* than a hard filter.
- **High-cardinality filters.** Filtering by a unique `document_id` per query
  means you are not searching, you are looking up. Fine — just do not pay for
  the vector store.
- **Metadata drift.** `doc_type` becomes `"how-to"`, `"howto"`, and `"HowTo"`
  across three ingestion runs, and filters silently match a third of what they
  should. Validate against an enum at ingestion; reject unknown values loudly.
- **Bloated metadata.** Storing entire chunk text in metadata inflates the index
  and slows filtering. Metadata is for filtering and citation, not storage.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| Capture at ingestion | Fields you skip now require a full re-index later |
| Dates as numbers | Enables `$gte`; strings do not order correctly |
| Pre-filter | Filter then rank — correct result counts |
| Post-filter | Rank then filter — collapses under narrow filters |
| k-collapse | Selective filter + post-filter = empty results |
| Over-fetch | The mitigation if you are stuck with post-filtering |
| `tenant_id` | Security boundary; server-side, pre-filtered, always |
| Never LLM-generated | Filters come from the session, not the model |
| Recency | Prefer a ranking boost over a hard filter |

## Exercise

Design and stress-test a metadata schema for a corpus with real constraints.

Take 20+ documents and attach at least five fields, including one date, one
low-cardinality enum, and one `tenant_id` with at least two tenants. Write 10
queries where the *correct* answer depends on the filter — for example, the
same question that must resolve differently for tenant A and tenant B, or a
policy that changed between 2023 and 2025.

Then measure:

1. Hit rate with no filter, with pre-filtering, and with post-filtering. Report
   the average number of results actually returned at `k=5` for each. Quantify
   the k-collapse.
2. Write an adversarial test: a query phrased to elicit tenant B's document
   while authenticated as tenant A. Assert it returns nothing. Then try to make
   it fail by moving the filter from pre to post.
3. Add a fallback that relaxes soft filters when the result set is empty, but
   never relaxes `tenant_id`. How do you surface to the user that the search was
   widened — and why does silently widening it undermine trust?

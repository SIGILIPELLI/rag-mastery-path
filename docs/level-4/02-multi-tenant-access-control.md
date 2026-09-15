---
description: "Multi-Tenant RAG & Access Control — This is the module where a subtle-looking bug is actually a data breach. In a multi-tenant RAG system — one deployment…"
---

# 02 · Multi-Tenant RAG & Access Control

This is the module where a subtle-looking bug is actually a data breach. In a
multi-tenant RAG system — one deployment serving many customers, teams, or
users with different data access rights — the difference between "filter
before ranking" and "filter after ranking" isn't a performance detail. It's
the difference between correct isolation and a system that can, under the
right query, either deny a tenant their own data or hand them someone else's.

## The setup

```python
docs = [
    {"id": 0, "tenant": "acme",   "text": "Acme's Q3 revenue was $2M."},
    {"id": 1, "tenant": "acme",   "text": "Acme's refund policy is 30 days."},
    {"id": 2, "tenant": "globex", "text": "Globex's Q3 revenue was $5M."},
    {"id": 3, "tenant": "globex", "text": "Globex's refund policy is 14 days."},
]

def keyword_search(query, corpus, top_k=1):
    q = set(query.lower().split())
    scored = []
    for d in corpus:
        overlap = len(q & set(d["text"].lower().split()))
        scored.append((overlap, d))
    scored.sort(key=lambda x: -x[0])
    return [d for s, d in scored[:top_k] if s > 0]
```

## The vulnerable pattern: filter after ranking

```python
def search_postfilter_broken(query, docs, tenant, top_k=1):
    results = keyword_search(query, docs, top_k=top_k)   # ranks across ALL tenants first
    return [d for d in results if d["tenant"] == tenant]  # then discards non-matches
```

## The correct pattern: filter before ranking

```python
def search_prefilter_safe(query, docs, tenant, top_k=1):
    scoped = [d for d in docs if d["tenant"] == tenant]   # restrict candidates first
    return keyword_search(query, scoped, top_k=top_k)      # rank only within tenant's data
```

These look almost identical — same two operations, different order. Here's
why the order is the entire security model:

```python
query = "globex's q3 revenue was $5m."

print("global top-1, no filter at all:", keyword_search(query, docs, top_k=1))
print("broken postfilter, tenant=acme:", search_postfilter_broken(query, docs, "acme", top_k=1))
print("safe prefilter, tenant=acme:   ", search_prefilter_safe(query, docs, "acme", top_k=1))
```

Captured output:

```
global top-1, no filter at all: [{'id': 2, 'tenant': 'globex', 'text': "Globex's Q3 revenue was $5M."}]
broken postfilter, tenant=acme: []
safe prefilter, tenant=acme:    [{'id': 0, 'tenant': 'acme', 'text': "Acme's Q3 revenue was $2M."}]
```

The query happens to word-match Globex's document more closely than Acme's.
Under the broken pattern, `keyword_search` picks the single best global match
— Globex's chunk — *before* tenant filtering ever runs. Filtering then
discards it because it belongs to the wrong tenant, and **acme gets zero
results for a question their own data actually answers.** That's the visible,
annoying failure. The dangerous one is one line away: if that same
`top_k=1` result had *not* been filtered — a missing filter call, a
forgotten `tenant=` parameter on one code path, a new endpoint added without
the check — Globex's revenue figure would have been served directly into
Acme's session. Denial and leak are the same root cause (ranking sees data it
should never have been allowed to touch) with different downstream code
deciding which one you get.

## Why "just filter afterward" keeps happening

Post-filtering is what you get by default if you bolt tenant isolation onto
an existing single-tenant retriever without touching its ranking step —
which is exactly the migration path most systems take: ship single-tenant,
add customers, retrofit isolation. It's also what a naive `top_k`-then-filter
implementation produces even when someone remembered to filter at all,
because "search, then check permissions on the results" is the intuitive
order to write it in. The fix is structural, not a bug-fix: **the permission
scope has to be an input to retrieval, not a post-processing step on its
output.**

## Row-level security as a stronger default

Where the underlying store supports it (pgvector on Postgres is the clean
case), enforcing tenant scope at the *database* layer via row-level security
policies means no application code path can accidentally skip it — not even
a new endpoint someone forgets to add a filter to:

```sql
-- Illustrative Postgres RLS policy — not run in this sandbox (no live DB here),
-- reviewed for correctness against pgvector's documented RLS support.
CREATE POLICY tenant_isolation ON document_chunks
    USING (tenant_id = current_setting('app.current_tenant')::text);
```

This moves the isolation guarantee from "every retrieval code path
remembered to filter" to "the database itself refuses to return rows outside
the current session's tenant" — a strictly stronger guarantee, at the cost of
needing your connection pooling to set `app.current_tenant` correctly per
request, which is its own thing to get right and audit.

## The trap: permission checks that only cover retrieval, not generation

Filtering retrieval correctly is necessary but not sufficient. Two related
leaks that survive a correct retrieval filter:

- **Cache leakage** (module 07's trap, worth repeating here): a full-answer
  cache keyed only on query text, shared across tenants, serves tenant A's
  cached answer — built from tenant A's data — to tenant B's identical-
  looking query. The cache key must include the tenant scope.
- **Cross-session context bleed in agentic pipelines**: an agentic loop
  (Level 3 module 01) that reuses a conversation-level scratchpad or an
  entity-carry variable (Level 3 module 02) across requests without
  resetting it per-tenant can leak entities extracted from one tenant's
  documents into another tenant's follow-up query construction.

Both are variants of the same root lesson: **any state that persists across
requests — cache, scratchpad, session context — needs the tenant scope
threaded through it as deliberately as the retrieval filter does.**

## Cheat sheet

| Symptom | Likely cause |
|---|---|
| Tenant gets zero results for answerable questions | Post-filtering after top_k truncation |
| Tenant sees another tenant's data | Missing filter, or post-filtering with no truncation loss (rarer, still possible) |
| New endpoint has a leak old ones don't | Filter enforced in application code per-path, not centrally |
| Leak survives a retrieval-filter code review | Check the cache and any cross-request scratchpad state next |
| Team says "we filter after ranking, it's fine, we just top_k a bit higher" | Overfetch-then-filter — recall degrades as filter selectivity increases (Level 3 module 04) and doesn't remove the structural risk |

## How It Actually Works

**Why "filter after ranking" is a recall bug that becomes a security bug
under multi-tenancy, mechanically.** Level-3 lesson 4 already showed that
post-filtering an ANN search's top-k can return fewer than k results when
many top matches fail the predicate. Under single-tenant metadata filtering
that's a quality annoyance; under multi-tenant access control the same
mechanism becomes a breach vector the moment the *unfiltered* top-k results
are used anywhere before the filter runs — for example, if a reranker
(level-2 lesson 3) scores the pre-filter candidate set, or if retrieved
text briefly enters a shared cache (level-3 lesson 7) keyed without tenant
scope, another tenant's data has already been read, processed, or cached
by code paths the access-control filter never touched. The vulnerability
isn't in the filter's correctness — it's in every place upstream of the
filter that touched the data first.

**Why filtering before ranking is the only version that closes this
structurally, not just usually.** Restricting the candidate set to the
requesting tenant's vectors *before* the ANN search or reranking step runs
means other tenants' vectors are never fetched, never scored, never placed
in a cache keyed insufficiently, and never assembled into a prompt — the
same principle as level-2 lesson 5's "metadata as a security boundary,"
now applied at the point where the actual attack surface (accidental
exposure through an intermediate processing stage, not just the final
answer) lives. Row-level security pushes this even further down: enforcing
the tenant predicate inside the database or vector store's own query engine
means the isolation guarantee doesn't depend on every application code path
remembering to add the filter correctly — a single missed `WHERE tenant_id
= ?` in a new endpoint is a silent bypass under application-level filtering,
but is a structural impossibility under database-level row security,
because the row simply doesn't exist in the query's visible result set
regardless of what code executed it.

**Why permission checks stopping at retrieval leave the generation step
exposed anyway.** Filtering retrieval correctly guarantees the LLM's context
window never contains another tenant's chunk — but it does not, by itself,
guarantee the model's *response* respects further constraints (a chunk
correctly retrieved for the current tenant might still contain a
sub-document-level permission boundary, e.g. an HR document one manager can
see fully and another can only see redacted). Generation-time checks — output
filtering, or a second-pass permission validation against what was actually
included in the final prompt — exist because "the right document reached
context" and "the right information reached the user" are not the same
guarantee once documents themselves contain heterogeneous permission
levels.

## 🔀 Related lessons on other tracks

- [Cybersecurity — 05 · Authentication & Access Control](https://sigilipelli.github.io/cybersecurity-mastery-path/level-1/05-authentication-access-control/)
- [ETL & Data Lake — 03 · Security & Access Control Patterns for Data Lakes](https://sigilipelli.github.io/etl-datalake-mastery-path/level-4/03-security-access-control/)

## Exercise

Write a test with three tenants and a query engineered (like the one above)
to globally rank another tenant's document first, and assert that
`search_prefilter_safe` never returns a document whose `tenant` field
doesn't match the requested tenant, across 100 randomized queries built from
shuffled words in the corpus. Then run the same test against
`search_postfilter_broken` and confirm it fails — capturing, in the
assertion message, the exact query that triggered either a leak or an
unnecessary empty result.

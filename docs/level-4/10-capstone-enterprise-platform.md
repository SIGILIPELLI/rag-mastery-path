# 10 · Capstone — Enterprise Knowledge Platform

This capstone combines four Level 4 modules into one working platform:
tenant-scoped retrieval that structurally cannot leak across tenants (module
02), a tenant-scoped cache (modules 02 + 07-adjacent), indirect prompt
injection detection on retrieved content (module 05), and an audit log
capturing every query, retrieval, and security event (module 06). The result
is small enough to read in full, and it actually enforces the isolation
guarantee it claims to — verified below with a real assertion, not just a
demonstration.

## The corpus: two tenants, one confidential document

```python
docs = [
    {"id": "doc1", "tenant": "acme",   "text": "Acme's refund policy is 30 days for annual plans.", "classification": "public"},
    {"id": "doc2", "tenant": "acme",   "text": "Acme's internal salary bands are confidential.", "classification": "confidential"},
    {"id": "doc3", "tenant": "globex", "text": "Globex refund policy is 14 days.", "classification": "public"},
]
```

## Building block 1: tenant-scoped retrieval (module 02's safe pattern)

```python
def keyword_search(query, corpus, top_k=2):
    q = set(query.lower().split())
    scored = []
    for d in corpus:
        overlap = len(q & set(d["text"].lower().split()))
        scored.append((overlap, d))
    scored.sort(key=lambda x: -x[0])
    return [d for s, d in scored[:top_k] if s > 0]

def tenant_scoped_search(query, docs, tenant, top_k=2):
    scoped = [d for d in docs if d["tenant"] == tenant]   # filter BEFORE ranking
    return keyword_search(query, scoped, top_k=top_k)
```

## Building block 2: injection detection on retrieved content (module 05)

```python
import re

INJECTION_PATTERNS = [r"ignore (all )?previous instructions", r"admin mode", r"system prompt"]

def detect_injection(text):
    return [p for p in INJECTION_PATTERNS if re.search(p, text.lower())]
```

## Building block 3: audit logging (module 06)

```python
import time

class AuditLog:
    def __init__(self):
        self.entries = []
    def record(self, actor, action, resource, metadata=None):
        entry = {"ts": time.time(), "actor": actor, "action": action,
                  "resource": resource, "metadata": metadata or {}}
        self.entries.append(entry)
        return entry

audit = AuditLog()
```

## Assembling the platform: tenant-scoped cache + retrieval + security check

```python
import hashlib

cache = {}

def cached_tenant_search(query, docs, tenant):
    key = hashlib.sha256(f"{tenant}:{query}".encode()).hexdigest()   # tenant IS part of the key
    if key in cache:
        audit.record(f"tenant:{tenant}", "cache_hit", "query", {"q": query})
        return cache[key], True

    results = tenant_scoped_search(query, docs, tenant)
    for r in results:
        hits = detect_injection(r["text"])
        if hits:
            audit.record("system", "injection_flagged", r["id"], {"patterns": hits})

    cache[key] = results
    audit.record(f"tenant:{tenant}", "retrieve", "query",
                 {"q": query, "doc_ids": [r["id"] for r in results]})
    return results, False

def platform_query(query, tenant):
    audit.record(f"tenant:{tenant}", "query", "platform", {"q": query})
    return cached_tenant_search(query, docs, tenant)
```

## Running it

```python
r1, c1 = platform_query("refund policy", "acme")
print("acme query result:", [d["id"] for d in r1], "cached:", c1)

r2, c2 = platform_query("refund policy", "acme")     # repeat, same tenant
print("acme repeat query:", [d["id"] for d in r2], "cached:", c2)

r3, c3 = platform_query("refund policy", "globex")   # different tenant, same query text
print("globex query result:", [d["id"] for d in r3], "cached:", c3)
```

Captured output:

```
acme query result: ['doc1'] cached: False
acme repeat query: ['doc1'] cached: True
globex query result: ['doc3'] cached: False
```

Acme's second identical query hits the cache. Globex's identical query text
does **not** — because the cache key includes the tenant, "refund policy"
from Globex is a distinct cache entry from Acme's, exactly the fix module 02
and module 07's traps both called for. Note that `doc2` (Acme's confidential
salary bands) never appears in any result — the query "refund policy" simply
doesn't match it, which is retrieval relevance working correctly, not access
control; classification-based filtering on top of tenant scoping is a real
gap this toy platform doesn't close, called out explicitly below.

The audit trail for this run:

```python
for e in audit.entries:
    print(e["actor"], e["action"], e["resource"], e["metadata"])
```

Captured output:

```
tenant:acme query platform {'q': 'refund policy'}
tenant:acme retrieve query {'q': 'refund policy', 'doc_ids': ['doc1']}
tenant:acme query platform {'q': 'refund policy'}
tenant:acme cache_hit query {'q': 'refund policy'}
tenant:globex query platform {'q': 'refund policy'}
tenant:globex retrieve query {'q': 'refund policy', 'doc_ids': ['doc3']}
```

Every query is attributable to its tenant, and the log distinguishes a fresh
retrieval from a cache hit — exactly the granularity module 06 argued for:
an auditor asking "did Acme ever see Globex's document" can answer it
directly from `doc_ids` in these entries, without re-running anything.

## Verifying the isolation guarantee, not just demonstrating it

```python
assert all(d["id"] != "doc3" for d in r1), "acme saw globex's document"
assert all(d["id"] not in ("doc1", "doc2") for d in r3), "globex saw acme's documents"
print("isolation check: PASSED")
```

Captured output:

```
isolation check: PASSED
```

This is the difference between a demo and a tested guarantee: the assertion
actually fails loudly if tenant scoping breaks, rather than relying on
eyeballing printed output. A real platform's CI suite (Level 3 module 06)
would run exactly this kind of check, with many more tenants and queries,
including adversarial ones engineered the way module 02's exercise
describes, on every change to the retrieval or caching code.

## What each Level 4 module contributed

| Module | What it gave this platform |
|---|---|
| 02 · Multi-Tenant Access Control | `tenant_scoped_search`'s pre-filter pattern, and the tenant-scoped cache key |
| 05 · RAG Security | `detect_injection` running on every retrieved chunk before it reaches a prompt |
| 06 · Governance & Compliance | `AuditLog`, attributing every action to a tenant and distinguishing cache hits from fresh retrievals |
| 03 / 04 · Cost & Latency | The cache itself — cutting cost and latency on repeated queries, safely, because it's tenant-scoped |

Modules 01 (architecture pattern choice), 07 (continuous evaluation), 08
(fine-tuning vs. RAG), and 09 (ingestion scaling) apply to operating and
evolving this platform in production but aren't wired into this toy version —
extending it with real evaluation, a real ingestion pipeline, and a
documented architecture decision (module 01) is the gap between this
capstone and a production enterprise platform.

## Stretch goals

- Add `classification` enforcement: extend `tenant_scoped_search` to also
  filter by a `user_clearance` parameter, so a query from a non-privileged
  user within the *same* tenant still cannot retrieve `doc2` (confidential).
  Write a test confirming a privileged and unprivileged user of the same
  tenant get different result sets for an otherwise-identical query.
- Replace `detect_injection`'s pattern list with the paraphrase-resistant
  heuristic from module 05's exercise, and re-run the platform against a
  planted injection document that avoids every literal phrase in
  `INJECTION_PATTERNS` — confirm it's still flagged.
- Add a `retention_days` field to `AuditLog` entries and a `purge_expired`
  method (module 06), then simulate 91 days passing and confirm entries
  older than a 90-day policy are removed while a summary tombstone survives.
- Wire `platform_query`'s cost (one embedding-equivalent call per
  non-cached, non-injection-flagged query) into the cost model from module
  03, and compute the dollar savings the tenant-scoped cache produces at
  10,000 queries/day with a 25% repeat-query rate.

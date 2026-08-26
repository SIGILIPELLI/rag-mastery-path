# 06 · Governance, Compliance & Auditability

"It worked in testing" isn't a defense in a regulated audit. Enterprise RAG
systems handling legal, healthcare, or financial data need to answer, after
the fact and with evidence: who asked what, what data informed the answer,
who could have seen it, and for how long that record is kept. This module
builds the audit-logging and data-lineage primitives that answer those
questions, and covers where they typically fall short.

## Audit logging: every action, attributable and tamper-evident

```python
import time, json, hashlib

class AuditLog:
    def __init__(self):
        self.entries = []

    def record(self, actor, action, resource, metadata=None):
        entry = {
            "ts": time.time(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "metadata": metadata or {},
        }
        # A hash of the entry's own content — not a secure chain by itself, but a
        # cheap way to detect if an entry was altered after the fact.
        entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:12]
        self.entries.append(entry)
        return entry

log = AuditLog()
log.record("user:alice", "query", "legal-contracts-index", {"question": "termination clause?"})
log.record("system", "retrieve", "doc-4821", {"tenant": "acme"})
log.record("system", "generate", "llm-call", {"model": "claude", "tokens": 300})

for e in log.entries:
    print(e["actor"], e["action"], e["resource"])
```

Captured output:

```
user:alice query legal-contracts-index
system retrieve doc-4821
system generate llm-call
```

Three separate log entries for one user request — the query, the retrieval,
and the generation — each independently attributable. This granularity
matters: "alice asked a question" doesn't tell an auditor *which document*
informed the answer she received, and a single combined log line tends to
lose that detail the moment logging is treated as a debugging convenience
rather than a compliance record.

## Data lineage: "what data informed this answer"

```python
def lineage_for_answer(retrieved_doc_ids, doc_metadata):
    return [doc_metadata[d] for d in retrieved_doc_ids if d in doc_metadata]

doc_metadata = {
    "doc-4821": {
        "source": "contract_2023.pdf",
        "classification": "confidential",
        "ingested_at": "2024-01-15",
    },
}

print(lineage_for_answer(["doc-4821"], doc_metadata))
```

Captured output:

```
[{'source': 'contract_2023.pdf', 'classification': 'confidential', 'ingested_at': '2024-01-15'}]
```

Given only a `trace_id` and a retrieved-document-ID list (Level 3 module 09's
tracing gives you exactly this), lineage tracking answers "show me every
source document that could have influenced this specific answer" — the
question a compliance review or a "right to know what data was used about
me" request actually asks. Note this depends entirely on `doc_metadata`
being populated at ingestion time (source file, classification level,
ingestion date) — if that metadata isn't captured when a document enters the
index, it can't be reconstructed afterward, and lineage tracking silently
degrades to "we know a document was retrieved, not what it was or where it
came from."

## Retention policy enforcement

```python
def apply_retention(log, max_age_seconds, now):
    kept = [e for e in log.entries if now - e["ts"] <= max_age_seconds]
    removed = len(log.entries) - len(kept)
    log.entries = kept
    return removed

now = time.time() + 100
removed = apply_retention(log, max_age_seconds=50, now=now)
print("removed:", removed, "remaining:", len(log.entries))
```

Captured output:

```
removed: 3 remaining: 0
```

All three entries aged past the 50-second retention window and were purged.
In production this cuts the other way as often as it cuts toward deletion:
regulations like GDPR require deleting data on request within a bounded
time, while others (financial recordkeeping, some healthcare regulations)
require *retaining* audit records for years. **The retention policy is a
compliance requirement, not an engineering default** — get the specific
retention period from whoever owns regulatory compliance for your
deployment, per data category, rather than picking a number that seems
reasonable.

## The trap: logs that can't answer the question that gets asked

Every piece above is necessary but the combination has to actually answer
real audit questions, and it's easy to build a system that has "logging" and
still can't answer them:

- **"Did this user have permission to see this data at the time they saw
  it?"** requires cross-referencing the audit log against the access-control
  state *as it was at that timestamp*, not as it is now — a permission that
  was later revoked doesn't retroactively make a past access improper if it
  was valid then, but the log needs enough detail to reconstruct that,
  meaning "what were this user's permissions at time T" has to be
  answerable, which usually requires versioning access-control changes
  (module 02), not just logging the current state.
- **"Delete everything about user X"** (a GDPR right-to-erasure request)
  requires the audit log itself to support redaction of a specific actor's
  entries without breaking the tamper-evidence of the surrounding log — a
  genuinely hard problem that naive append-only or hash-chained logs don't
  solve for free, and one worth designing for before the first real request
  arrives rather than after.
- **Audit logs that are themselves unsecured.** A log containing "user asked
  about acquisition rumors, retrieved doc-4821 (confidential M&A memo)" is
  itself sensitive data, sometimes more sensitive than the original query.
  Access to the audit log needs its own access control, and it needs to be
  in scope for the same security review as the primary system (module 05).

## Cheat sheet

| Compliance question | What has to be logged/tracked to answer it |
|---|---|
| Who asked what, and when? | Actor + action + timestamp per request |
| What data informed this specific answer? | Retrieved doc IDs + doc metadata, linked by trace_id |
| Did this access comply with policy at the time? | Access-control state versioned over time, not just current |
| Can we delete one user's records on request? | Log design that supports targeted redaction |
| Are we retaining data the required duration — no more, no less? | Explicit, per-category retention policy, enforced in code |
| Is the audit log itself secure? | Access control and review scope on the log, same as the primary system |

## Exercise

Extend `AuditLog` with a `redact_actor(actor_id)` method that removes all
entries for a given actor while leaving a tombstone entry recording that a
redaction occurred (actor, timestamp, and count of entries removed, but not
their content) — then verify that after redaction, `lineage_for_answer` for
any of that actor's past queries can no longer reconstruct what they asked,
while the fact that *a* redaction happened remains auditable.

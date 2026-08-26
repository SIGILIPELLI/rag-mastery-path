# 01 · Enterprise RAG Architecture Patterns

Everything through Level 3 is one pipeline, one corpus, one set of users. An
enterprise deployment usually has none of those luxuries: multiple teams with
different corpora, different access rules, different latency and cost
budgets, and a platform team that has to support all of them without
rebuilding the pipeline per team. This module is architectural — comparing
the patterns enterprises actually converge on — with runnable Python only
where the trade-off is genuinely a code-level decision (routing logic,
config-driven pipelines); the rest is deliberate, stated manual review, since
"which architecture fits your org" isn't something a toy script can validate.

## Three patterns, and when each wins

**1. Single shared index, metadata-filtered per tenant/team.** One vector
store, one ingestion pipeline; every document is tagged with `team_id` or
`tenant_id`, and every query filters on it. Cheapest to operate, and the
riskiest — a missing or wrong filter leaks data across tenants (Level 4
module 2 covers this failure mode exhaustively). Fits well when tenants are
internal teams with moderate trust and a platform team enforcing filters
centrally.

**2. Index-per-tenant.** Separate vector collections (or separate database
instances) per tenant. No shared-filter risk by construction — a bug can't
leak tenant A's data into tenant B's query because there's no code path that
touches both. Costs more operationally: N tenants means N indexes to
provision, monitor, and keep fresh, and small tenants pay a fixed per-index
overhead disproportionate to their data volume.

**3. Hub-and-spoke with a shared retrieval service.** Teams own their data
and ingestion; a central service exposes a common retrieval API, handles
auth, routing, and observability uniformly, and enforces org-wide policies
(module 05, 06) in one place instead of per-team reimplementation. This is
the pattern most enterprises grow into after starting with pattern 1, because
it's the only one of the three that scales the *platform team's* effort
sub-linearly with the number of internal RAG consumers.

| | Shared index + filters | Index-per-tenant | Hub-and-spoke |
|---|---|---|---|
| Isolation guarantee | Enforced in query code | Structural | Enforced in gateway code |
| Ops overhead | Lowest | Scales with tenant count | Moderate, centralized |
| Best for | Few, trusted internal teams | Regulated/high-trust-boundary tenants | Many teams, shared platform investment |
| Biggest risk | Filter bug = data leak | Cost sprawl, index drift | Gateway becomes a bottleneck/SPOF |

## Config-driven pipelines: the part that is actually code

What *is* a concrete, runnable engineering decision is whether each team's
pipeline is a bespoke script or a declarative config against a shared engine.
The shared-engine version scales a platform team's effort; the bespoke
version scales the number of pipelines someone has to individually
understand during an incident.

```python
from dataclasses import dataclass, field

@dataclass
class PipelineConfig:
    team_id: str
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_model: str = "default-small"
    rerank: bool = False
    allowed_sources: list = field(default_factory=list)

def build_pipeline(config: PipelineConfig):
    # A real version wires this into your retriever/generator factory.
    # Returning a summary dict here to make the routing decision inspectable.
    return {
        "team_id": config.team_id,
        "index_name": f"idx_{config.team_id}",
        "chunking": (config.chunk_size, config.chunk_overlap),
        "retrieval": {"top_k": config.top_k, "rerank": config.rerank},
        "embedding_model": config.embedding_model,
    }

legal_team = PipelineConfig(team_id="legal", chunk_size=256, rerank=True,
                             allowed_sources=["contracts", "policies"])
support_team = PipelineConfig(team_id="support", chunk_size=512, top_k=8)

for cfg in (legal_team, support_team):
    print(build_pipeline(cfg))
```

Captured output:

```
{'team_id': 'legal', 'index_name': 'idx_legal', 'chunking': (256, 64), 'retrieval': {'top_k': 5, 'rerank': True}, 'embedding_model': 'default-small'}
{'team_id': 'support', 'index_name': 'idx_support', 'chunking': (512, 64), 'retrieval': {'top_k': 8, 'rerank': False}, 'embedding_model': 'default-small'}
```

One `build_pipeline` function serves both teams' different needs from data,
not from two different code paths — this is the concrete mechanism behind
"hub-and-spoke scales platform effort sub-linearly": adding team 51 is a new
`PipelineConfig`, not a new pipeline implementation to maintain.

## The trap: architecture decisions made by whoever asked first

The single most common enterprise RAG failure isn't a bug — it's **pattern 1
adopted implicitly**, one team at a time, with no one deciding it
deliberately. Team A builds a quick shared index for their docs. Team B asks
to add theirs "to the same thing, it's already there." Six months in, there's
a shared index with a dozen teams' data, filter logic that's grown ad hoc
per team, and no one who chose pattern 1 as a policy — it just accreted. By
the time someone asks "wait, can tenant X's queries actually see tenant Y's
data?" the answer is usually "we're not sure," which is the module 02
scenario this level exists to prevent.

The fix isn't picking the "best" pattern up front — it's making the choice
**explicit and reviewed** before the second team joins a shared resource, with
the isolation guarantee (structural vs. enforced-in-code) stated in writing,
because that's the fact that determines how carefully every future filter
change has to be reviewed.

## Cheat sheet

| Signal | Suggests |
|---|---|
| 2-3 internal teams, similar trust level | Shared index + filters is fine, keep filter logic centralized |
| Regulatory separation required (e.g. GDPR data residency per customer) | Index-per-tenant, or physically separate infra |
| 10+ teams, growing | Hub-and-spoke — invest in the shared retrieval service |
| "Just add it to the existing index" without a filter review | Stop — this is how implicit pattern-1 sprawl starts |
| One team's config change breaks another team's pipeline | Sign the config-driven approach was needed and isn't there yet |

## Exercise

Extend `PipelineConfig` with an `isolation_mode` field (`"shared_filtered"`,
`"per_tenant"`, or `"hub_spoke"`), and write a `validate_config` function that
raises if `isolation_mode == "shared_filtered"` but `allowed_sources` is
empty — treating "no explicit source restriction on a shared index" as a
configuration error rather than a default to silently allow. This is the
kind of guardrail-as-code that turns an implicit architecture decision into
an enforced one.

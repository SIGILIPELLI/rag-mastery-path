# 03 · Cost Optimization

At enterprise query volume, "which model should we use" is a budget line
item, not a preference. This module builds a real cost model for a RAG
pipeline — arithmetic you can rerun against your own volume and pricing —
and shows where the money actually goes, which is rarely where teams
initially guess.

## The cost model

```python
# Illustrative per-token pricing (check current provider pricing for real numbers) —
# the arithmetic pattern below is what matters, not these specific rates.
embed_cost_per_1k_tokens = 0.00002
llm_input_per_1k = 0.003
llm_output_per_1k = 0.015

queries = 1_000_000
avg_query_tokens = 20
avg_context_tokens = 1500   # retrieved chunks injected into the prompt
avg_output_tokens = 250

embed_cost = queries * (avg_query_tokens / 1000) * embed_cost_per_1k_tokens
llm_in_cost = queries * ((avg_query_tokens + avg_context_tokens) / 1000) * llm_input_per_1k
llm_out_cost = queries * (avg_output_tokens / 1000) * llm_output_per_1k
total = embed_cost + llm_in_cost + llm_out_cost

print("embed cost:      $", round(embed_cost, 2))
print("llm input cost:  $", round(llm_in_cost, 2))
print("llm output cost: $", round(llm_out_cost, 2))
print("total:           $", round(total, 2))
print("pct on generation:", round((llm_in_cost + llm_out_cost) / total * 100, 1), "%")
```

Captured output:

```
embed cost:      $ 0.4
llm input cost:  $ 4560.0
llm output cost: $ 3750.0
total:           $ 8310.4
pct on generation: 100.0 %
```

At a million queries a month, embedding cost is a rounding error — $0.40 —
and generation is essentially the entire $8,310 bill. This is the single most
useful fact this module teaches: **teams instinctively optimize embedding
model choice and retrieval speed, and the actual money is in the prompt sent
to the LLM and the tokens it generates.** Retrieval-side optimization (Level
3 module 04, 05, 07) still matters for latency and quality, but if the goal
is specifically cost, the LLM call is where to look first.

## Lever 1: caching

```python
hit_rate = 0.30   # fraction of queries served from cache instead of a fresh LLM call
total_cached = (embed_cost + llm_in_cost + llm_out_cost) * (1 - hit_rate)
print("total with 30% cache hit rate: $", round(total_cached, 2))
print("savings:                       $", round(total - total_cached, 2))
```

Captured output:

```
total with 30% cache hit rate: $ 5817.28
savings:                       $ 2493.12
```

A 30% cache hit rate — realistic for FAQ-heavy support workloads with
repeated questions — cuts nearly $2,500/month directly, because a cache hit
skips both the retrieval and the generation cost entirely (Level 3 module 07
covers the mechanics; this is what that mechanism is worth in dollars at
scale).

## Lever 2: shrinking the context

```python
avg_context_tokens_v2 = 800   # tighter chunking/reranking keeps fewer, denser chunks
llm_in_cost_v2 = queries * ((avg_query_tokens + avg_context_tokens_v2) / 1000) * llm_input_per_1k
total_v2 = embed_cost + llm_in_cost_v2 + llm_out_cost
print("total with smaller context: $", round(total_v2, 2))
print("savings from context reduction: $", round(total - total_v2, 2))
```

Captured output:

```
total with smaller context: $ 6210.4
savings from context reduction: $ 2100.0
```

Cutting average injected context from 1500 to 800 tokens — via better
chunking, reranking down to fewer but more relevant chunks, or a smaller
`top_k` — saves $2,100/month here, because input tokens are billed on every
single request whether or not the model needed all of them. This is the
direct cost argument for the reranking and retrieval-quality work in Level 3:
a reranker that reliably lets you drop `top_k` from 8 to 4 pays for its own
inference cost many times over at volume, *if* it doesn't hurt answer
quality — which is exactly why module 06/07's eval harness has to run
alongside any cost change, not after it.

## Lever 3: model tiering

Not every query needs the most expensive model. A common pattern: route
simple factual lookups to a cheap, fast model, and route genuinely complex
multi-hop or reasoning-heavy questions (Level 3 modules 01–02) to a stronger,
pricier one.

```python
def route_by_complexity(question, hop_count_estimate):
    # A real router uses a cheap classifier or heuristic; hop_count_estimate
    # stands in for "how many retrieval/reasoning steps this looks like it needs."
    return "cheap_model" if hop_count_estimate <= 1 else "strong_model"

cheap_price_per_1k_out = 0.0015   # ~10x cheaper than the strong model used above
mix = {"cheap_model": 0.7, "strong_model": 0.3}   # 70% of traffic is simple lookups

out_cost_tiered = queries * (avg_output_tokens / 1000) * (
    mix["cheap_model"] * cheap_price_per_1k_out
    + mix["strong_model"] * llm_output_per_1k
)
print("output cost, single strong model: $", round(llm_out_cost, 2))
print("output cost, tiered routing:      $", round(out_cost_tiered, 2))
print("savings:                          $", round(llm_out_cost - out_cost_tiered, 2))
```

```
output cost, single strong model: $ 3750.0
output cost, tiered routing:      $ 1387.5
savings:                          $ 2362.5
```

Routing 70% of traffic to a 10x-cheaper model for output tokens alone saves
$2,362.50/month on this workload's output cost — the input-token side of
tiering follows the identical arithmetic. The catch is entirely in
`route_by_complexity`'s accuracy: misrouting a genuinely complex question to
the cheap model doesn't just risk a wrong answer, it risks an *expensive*
wrong answer if the user has to re-ask and the system retries with the strong
model anyway, paying for both.

## The trap: optimizing cost against yesterday's traffic mix

Every number above assumes a fixed traffic profile — same query complexity
distribution, same cache hit rate, same context size needs. Two ways this
breaks in production:

- **A product launch shifts the mix** — if a new feature drives more complex,
  multi-hop questions, a router tuned on old traffic sends them to the cheap
  model, degrading quality silently while the cost dashboard looks great
  (cost went down because low-cost-tier volume looks like it increased, when
  actually complexity was misclassified).
- **Cache hit rate isn't stable** — a cache tuned for FAQ-heavy support
  traffic assumes repeated questions; a workload shift toward unique,
  per-user questions collapses the hit rate and the "$2,493/month savings"
  above evaporates without any code change, silently, unless hit rate itself
  is monitored as a metric (Level 3 module 09).

The fix is the same principle as module 06's eval-gaming trap: never treat a
cost optimization as a one-time win. Track cost-per-query, cache hit rate,
and router accuracy continuously, the same way you'd track retrieval
recall — because a cost lever that degrades silently is exactly as dangerous
as a quality metric that does.

## Cheat sheet

| Lever | Saves | Risk if done carelessly |
|---|---|---|
| Caching | Skips full request cost on hits | Stale answers (Level 3 module 08), cross-tenant leak if unscoped (module 02) |
| Smaller context / better reranking | Input token cost, every request | Answer quality drop if chunks cut were actually needed |
| Model tiering | Output (and input) cost on routed-down traffic | Misrouted complex queries get worse answers, possibly re-run at full cost anyway |
| None of the above, just "use a cheaper model everywhere" | Uniform cost cut | Uniform quality cut — no way to recover just the queries that needed the stronger model |

## Exercise

Recompute `total` for your own actual (or estimated) query volume, average
context size, and provider pricing, then apply all three levers together
(cache hit rate, reduced context, tiered routing) and calculate the combined
savings — confirm whether the three levers are additive or whether applying
them together (e.g., caching already removes the queries that would have
been tiered) double-counts some of the savings.

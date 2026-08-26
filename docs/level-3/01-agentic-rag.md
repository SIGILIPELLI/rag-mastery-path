# 01 · Agentic RAG

Every pipeline through Level 2 retrieves once, then answers. That works when
the question maps cleanly onto one search. It breaks the moment the question
needs *judgment about retrieval itself*: "is this evidence enough?", "should I
rephrase and try again?", "do I need a second, different search to cover the
other half of the question?" Agentic RAG is the pattern where the LLM controls
those decisions, instead of your pipeline code.

Concretely: retrieval stops being a fixed step and becomes a **tool** the
model can call, zero or more times, in a loop it steers.

## The shape of the loop

```python
def search_tool(query, docs, top_k=2):
    """Toy BM25-style keyword search standing in for a real retriever."""
    q = set(query.lower().split())
    scored = []
    for i, d in enumerate(docs):
        overlap = len(q & set(d.lower().split()))
        scored.append((overlap, i))
    scored.sort(reverse=True)
    return [docs[i] for s, i in scored[:top_k] if s > 0]
```

A real system hands the LLM a tool schema (`search(query: str) -> list[str]`)
and lets *the model* write the query string and decide whether to call it
again. Here we simulate the model's decisions explicitly so the control flow
is inspectable:

```python
class Budget:
    """Hard cap on tool calls — see the trap below."""
    def __init__(self, max_calls):
        self.max_calls = max_calls
        self.calls = 0

    def use(self):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("tool budget exceeded")

def agent_loop(question, docs, max_calls=3):
    budget = Budget(max_calls)
    # A real agent generates these; we hardcode the model's plan for clarity.
    subqueries = [question, "annual plan refund", "enterprise refund terms"]
    evidence = []
    for sq in subqueries:
        budget.use()
        evidence.extend(search_tool(sq, docs))
    return list(dict.fromkeys(evidence)), budget.calls
```

Run it:

```python
docs = [
    "The refund window is 30 days from purchase for annual plans.",
    "Monthly plans are non-refundable after 7 days.",
    "Enterprise contracts have custom refund terms in the MSA.",
]

evidence, calls = agent_loop("what is the refund policy for annual plans?", docs)
print("calls used:", calls)
for e in evidence:
    print("-", e)
```

Captured output:

```
calls used: 3
- The refund window is 30 days from purchase for annual plans.
- Enterprise contracts have custom refund terms in the MSA.
```

Three tool calls surfaced two distinct, relevant chunks that a single search
for "refund policy annual plans" might have missed if the enterprise clause
lived in a differently-worded document. That's the entire value proposition
of agentic RAG in one run: **more, better-targeted retrieval, paid for in
extra tool calls and latency.**

## What actually changes vs. Level 1–2

| Fixed pipeline | Agentic pipeline |
|---|---|
| Retrieve once, always | Retrieve 0–N times, model's choice |
| One query: the user's | Model rewrites/splits into its own queries |
| Always answers | Can decide "insufficient evidence, need another search" |
| Latency ≈ 1 retrieval + 1 generation | Latency ≈ N retrievals + reasoning between them |
| Cost is predictable | Cost varies per question, sometimes wildly |

The mechanism giving the model this control is a **tool-use / function-calling**
API: you describe `search(query: str)` in JSON schema, the model emits a
tool-call instead of (or before) a final answer, your code executes it and
feeds the result back in, and the model decides whether to call again or
answer. The loop above is that protocol with the model's turn hardcoded so you
can see the mechanics without needing a live API key.

## The trap: unbounded loops are a production incident waiting to happen

An agent that decides for itself when to stop searching can also decide,
because of a bad prompt, a confusing corpus, or a model that gets stuck in a
"let me check one more thing" spiral, to never stop. This is not hypothetical
— it is the single most common agentic-RAG production failure, and it costs
real API dollars per stuck request, not just latency.

```python
try:
    agent_loop("some tricky question", docs, max_calls=2)
except RuntimeError as e:
    print("caught:", e)
```

```
caught: tool budget exceeded
```

The fix is always the same shape, layered:

1. **Hard call budget** — a `Budget` object like above, enforced in code the
   model cannot talk its way around (never just a prompt instruction like "make
   at most 3 searches" — models don't reliably obey it).
2. **Wall-clock timeout** on the whole loop, independent of call count, because
   a single call can itself hang.
3. **Diminishing-returns check** — if the last two searches returned near-
   identical evidence (compare with a cheap overlap or embedding-similarity
   score), stop and answer with what you have rather than calling again.
4. **Cost ceiling per request**, logged and alertable, because call count
   alone doesn't capture cost if queries vary in retrieved-context size.

A second, quieter trap: agentic loops make **retrieval quality harder to
debug**, because a bad final answer might trace back to hop 1's query, hop 2's
query, or a stopping decision that fired too early. Section 09 (Observability)
covers tracing this properly — for now, log every subquery and every result
set, not just the final answer.

## Cheat sheet

| Decision | Fixed RAG | Agentic RAG |
|---|---|---|
| When to retrieve | Always, once | Model decides, 0–N times |
| Query text | User's question verbatim | Model-generated, can differ per call |
| Stopping condition | N/A | Budget, timeout, or model's own judgment |
| Cost predictability | High | Low — variance per question |
| Failure mode | Under-retrieval | Runaway loops, latency spikes |
| Needs | Retriever | Retriever + tool schema + budget enforcement + tracing |

## Exercise

Extend `agent_loop` so it stops early when a new subquery's results are
already fully contained in `evidence` (no new chunks), and add a wall-clock
budget using `time.monotonic()` alongside the call-count budget. Run it against
a question where the second subquery is redundant with the first, and confirm
the loop exits after 2 calls instead of 3.

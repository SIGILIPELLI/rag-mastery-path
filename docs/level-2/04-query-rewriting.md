# 04 · Query Rewriting & Expansion

Lessons 1–3 all optimized the same thing: what happens *after* the query
arrives. But the query itself is usually the weakest link in the pipeline. Users
type "and how do I undo it?", or "card got rejected", or three words with a
typo. Your documents are written in careful, formal, complete prose. The
mismatch is not a retrieval bug — it is a *translation* problem.

Query rewriting spends a little compute before retrieval to turn a bad query
into one or more good ones.

!!! note "What ran here, and what didn't"
    All retrieval, fusion, and comparison code below **ran locally** against the
    16-document corpus using `rank-bm25`. The query variants are **hard-coded**
    rather than LLM-generated, because this run had no API key. That changes
    nothing about the mechanics: an LLM call returns a list of strings, and the
    fusion logic that consumes it is identical. The LLM prompts shown are
    production-shaped but were not executed.

## The baseline failure

```python
def rank(query, k=5):
    s = bm25.get_scores(tok(query))
    hits = [i for i in range(len(s)) if s[i] > 0]      # a 0.0 score is not a hit
    return sorted(hits, key=lambda i: -s[i])[:k]

original = "undo a bad release"
print(f"BM25 top3: {[IDS[i] for i in rank(original, 3)]}")
```

```text
original: 'undo a bad release'
  BM25 top3: []  <- zero hits
```

An empty list. Honest, and useless. The answer (`d9`, the rollback runbook)
exists; the query simply shares no vocabulary with it.

Note the `s[i] > 0` filter — without it this function returns the first three
documents in corpus order with scores of 0.0, which downstream code cannot
distinguish from real hits. Lesson 1 flagged this; it matters most here, because
rewriting pipelines fuse *many* rankings and all-zero rankings poison the fusion.

## Multi-query retrieval

Rather than trusting one phrasing, generate several and fuse the results with
RRF from lesson 2.

```python
# In production an LLM generates these. Here they are hard-coded so the
# example runs with no API key -- the fusion logic is identical either way.
VARIANTS = [
    "how to roll back a deployment",
    "revert production to a previous version",
    "rollback command for deploys",
]

for v in VARIANTS:
    print(f"variant {v!r}\n  top3: {[IDS[i] for i in rank(v, 3)]}")

fused = rrf([rank(v) for v in VARIANTS] + [rank(original)])
for i in fused:
    print(f"  [{IDS[i]}] {TEXTS[i]}")
```

```text
variant 'how to roll back a deployment'
  top3: []
variant 'revert production to a previous version'
  top3: ['d8', 'd10']
variant 'rollback command for deploys'
  top3: ['d9', 'd8']

fused over 1 original + 3 variants:
  [d8] Deploys to production happen automatically when main is merged.
  [d9] Rollbacks are performed with the deploy --rollback command.
  [d10] Staging mirrors production and refreshes its data nightly.
```

From nothing to a top-3 that contains the answer. Two observations worth more
than the win itself:

- **One variant still returned nothing.** "How to roll back a deployment" got
  zero hits — the analyzer stems `deployment` differently from `deploys`.
  Generating variants is a shotgun, and some pellets miss. That is fine; the
  fusion absorbs it.
- **`d8` outranked `d9`.** The *correct* answer is `d9`, but `d8` appeared in
  two variant rankings and `d9` in one. This is RRF's consensus bias from
  lesson 2 doing exactly what it is designed to do, and it is the argument for
  putting a reranker (lesson 3) after fusion rather than trusting fused order.

The production version of the variant generator:

```python
MULTI_QUERY_PROMPT = """Generate 3 alternative phrasings of the user's question
for a documentation search engine. Vary the vocabulary: use synonyms, formal
terminology, and the words a technical writer would use. Output one per line,
no numbering, no commentary.

Question: {question}"""

# resp = client.messages.create(model=..., max_tokens=200,
#     messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=q)}])
# variants = [l.strip() for l in resp_text.splitlines() if l.strip()]
```

## Conversational rewriting

The most common production failure is not exotic vocabulary — it is pronouns.

```python
history = [("How do I deploy to production?", "Merge to main; it deploys automatically.")]
followup = "and how do I undo it?"
standalone = "how do I undo a deploy to production (rollback)?"

print(f"  as-is    : {[IDS[i] for i in rank(followup, 3)]}")
print(f"  rewritten: {[IDS[i] for i in rank(standalone, 3)]}")
```

```text
  history  : How do I deploy to production?
  follow-up: 'and how do I undo it?'
  as-is    : []
  rewritten: ['d9', 'd8', 'd10']
```

Zero hits versus the correct document at rank 1. "It" carries the entire
meaning of the question, and retrieval has no idea what "it" refers to.

If you build a chat interface over RAG, **this rewrite is not optional**. It is
the difference between a bot that works on turn 1 and dies on turn 2, and one
that holds a conversation.

```python
REWRITE_PROMPT = """Given the conversation history, rewrite the follow-up
question as a standalone question that makes sense with no context. Preserve
the user's intent exactly. If it is already standalone, return it unchanged.
Output only the rewritten question.

History:
{history}

Follow-up: {question}"""
```

## HyDE: hypothetical document embeddings

HyDE inverts the problem. Queries and documents are different *kinds* of text —
one is a short question, the other is expository prose — so comparing them
directly is comparing apples to oranges. HyDE asks an LLM to *hallucinate a
plausible answer*, then retrieves using that fake answer as the query.

```python
HYDE_PROMPT = """Write a short factual paragraph that would appear in
documentation answering this question. Invent plausible specifics. Do not
hedge, do not say you are unsure.

Question: {question}"""

# "undo a bad release" -> "To undo a bad release, use the rollback command.
#  Rollbacks revert production to the previously deployed version..."
```

The invented paragraph is likely wrong on the facts, and that is fine — it is
never shown to the user. It is used **only as a retrieval probe**, and it looks
far more like your documents than the original three-word query did.

HyDE works best with dense retrieval, where the generated prose lands near real
documents in embedding space. With BM25 it degenerates into keyword expansion —
useful, but less dramatic.

## Comparing the techniques

| Technique | Fixes | Extra LLM calls | Added latency | When to reach for it |
|---|---|---|---|---|
| Multi-query + RRF | Vocabulary mismatch | 1 | ~300–800 ms + N retrievals | Broad, general-purpose gain |
| Conversational rewrite | Pronouns, ellipsis | 1 | ~300–500 ms | **Mandatory** for any chat UI |
| HyDE | Query/document style gap | 1 | ~500 ms–2 s (long output) | Short queries, dense retrieval |
| Synonym/acronym expansion | Domain jargon (`2FA`) | 0 (dictionary) | ~0 ms | Known, stable vocabulary |
| Step-back prompting | Overly specific queries | 1 | ~300–500 ms | "Why did X fail in v2.3.1?" |
| Do nothing | — | 0 | 0 ms | Long, well-formed queries |

Note row 4: a hand-maintained acronym dictionary costs nothing, adds no latency,
and in lesson 10 it is what takes hit rate from 0.80 to 1.00. **Try the free fix
before the expensive one.**

## Traps

- **Latency stacks.** Rewriting is a blocking LLM call before retrieval even
  starts. Multi-query with N variants also multiplies your retrieval calls
  (parallelize them) and your reranker's candidate pool. A pipeline with
  rewrite + 4 retrievals + cross-encoder rerank + generation can easily exceed
  three seconds.
- **Rewriting away the exact token.** This is the sharp one. A user searching
  `E4022` gets "helpfully" rewritten to "billing address verification error" —
  and you have just destroyed the one query type BM25 was perfect at. **Always
  retrieve with the original query alongside the variants**, which is why the
  fusion above includes `rank(original)`.
- **Drift.** Each rewrite is a chance to change what the user meant. "Can I get
  a refund?" becoming "what is the cancellation policy?" retrieves confidently
  and answers a different question.
- **HyDE feeding its own hallucination.** If the generated document invents a
  command flag that does not exist, retrieval may surface loosely related
  content, and the final answer can inherit the invention. Ground the final
  generation strictly in retrieved text, never in the hypothetical document.
- **Rewriting everything.** Long, precise questions are usually better left
  alone. Gate rewriting on query length or an initial low-confidence retrieval
  result, so you pay the cost only when it can help.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| Multi-query | N phrasings, retrieve each, fuse with RRF |
| Always keep the original | Protects exact-token queries from rewrite damage |
| Conversational rewrite | Resolve pronouns; mandatory for chat |
| HyDE | Retrieve with a fake answer, not the question |
| Acronym dictionary | Free, instant, often the biggest single win |
| Filter zero scores | Before fusing, or you fuse corpus order |
| Fused order ≠ correct order | Put a reranker after fusion |
| Cost | Every technique is +1 LLM call on the critical path |

## Exercise

Build a rewriting layer and prove each piece earns its latency.

Take your lesson 1 query set and add six conversational pairs — a first
question plus a follow-up that is meaningless standalone ("what about the
monthly one?", "and if that fails?").

Measure hit rate and MRR for four configurations: (a) raw query, (b) raw +
hard-coded acronym expansion, (c) multi-query fusion with 3 variants, and
(d) conversational rewrite applied to the follow-ups. Record mean added latency
for each.

Then answer with evidence:

1. Which configuration gives the best quality *per millisecond*? Is it the one
   you expected?
2. Construct a query where rewriting makes retrieval **worse**. (Start with an
   error code.) What guard would you add to the rewrite step to prevent it?
3. For the conversational pairs, what happens to (a) and (c) on the follow-up
   turns? Explain why no amount of query *expansion* substitutes for pronoun
   *resolution*.

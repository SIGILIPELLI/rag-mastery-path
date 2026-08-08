# 03 · Reranking with Cross-Encoders

Retrieval has to be fast, because it scores every document in your corpus.
Precision has to be high, because whatever lands in the top 3 is what the LLM
sees. Those two demands pull in opposite directions, and the standard
resolution is a **two-stage pipeline**: cast a wide, cheap net, then spend real
compute re-scoring only the survivors.

This is the single highest-leverage upgrade in Level 2. It routinely moves the
right document from rank 7 to rank 1 without touching your index.

## Bi-encoders vs cross-encoders

Everything in Level 1 was a **bi-encoder**: the query and each document are
embedded *independently*, and similarity is a dot product between two vectors
that never met.

```text
bi-encoder     query --> [encoder] --> vec_q  \
                                               >-- cosine --> score
               doc   --> [encoder] --> vec_d  /

cross-encoder  (query, doc) --> [encoder] --> score
```

A **cross-encoder** feeds the query and document through the model *together*,
so every query token can attend to every document token. It can tell that "the
card was declined" refers to the same event as "payment card was declined by
the issuing bank", rather than just noticing they share a topic.

The catch is arithmetic. A bi-encoder embeds each document **once, at index
time**, so query time is a vector lookup. A cross-encoder must run a full
forward pass for **every (query, document) pair, at query time**. Scoring a
100,000-document corpus with a cross-encoder is not slow — it is impossible.

Hence: retrieve 50 with the bi-encoder and BM25, rerank those 50.

!!! warning "What ran here, and what didn't"
    Real cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`, BGE-reranker,
    Cohere Rerank) require `torch` + `sentence-transformers`, which were **not
    installed** for this run. The code below uses a lightweight **pair-scoring
    stand-in** that is structurally honest — it scores the `(query, document)`
    pair jointly rather than embedding each side separately — so the pipeline
    shape, the API, and the rank movement are all real. The *quality* and
    especially the *cost* numbers are not representative; the prose gives real
    figures for production models.

## The two-stage pipeline

```python
def retrieve(query, k):
    """Stage 1: fast, wide, cheap. Scores the whole corpus."""
    s = bm25.get_scores(tok(query))
    return sorted(range(len(s)), key=lambda i: -s[i])[:k]

def pair_score(query, doc):
    """Stage 2 stand-in: scores the (query, doc) PAIR jointly instead of
    embedding each side independently. A real cross-encoder replaces this."""
    q = [w for w in tok(query) if w not in STOP]
    d = tok(doc)
    if not q:
        return 0.0
    dset = set(d)
    coverage = sum(1 for w in q if w in dset) / len(q)
    positions = [d.index(w) for w in q if w in dset]
    proximity = 1.0 / (1.0 + (max(positions) - min(positions))) if len(positions) > 1 else 0.5
    brevity = 1.0 / (1.0 + abs(len(d) - 12) / 12)
    return 0.6 * coverage + 0.25 * proximity + 0.15 * brevity

query = "what is the refund policy for annual plans"

candidates = retrieve(query, 8)
reranked = sorted(candidates, key=lambda i: -pair_score(query, TEXTS[i]))[:3]
```

```text
stage 1 - BM25 top 8 (wide net):
  1. [d0] Annual plans can be refunded within 14 days of purchase or r
  2. [d6] Reset links expire after 30 minutes for security reasons.
  3. [d13] The webhook retry policy attempts delivery five times over o
  4. [d8] Deploys to production happen automatically when main is merg
  5. [d15] Community support is handled on the public forum with no SLA
  6. [d1] Monthly plans are non-refundable but can be cancelled at any
  7. [d2] Invoices are emailed on the first of each month to the accou
  8. [d3] Error E4021 means the payment card was declined by the issui

stage 2 - reranked top 3:
  1. 0.575  [d0] Annual plans can be refunded within 14 days of purchase or r
  2. 0.425  [d1] Monthly plans are non-refundable but can be cancelled at any
  3. 0.413  [d13] The webhook retry policy attempts delivery five times over o
```

Look at what moved. BM25's top 3 was `[d0, d6, d13]` — the correct answer, then
a document about **password reset links**, then webhook retries. Two of the
three chunks going into the LLM prompt were pure noise.

After reranking the top 3 is `[d0, d1, d13]`. `d1` — "Monthly plans are
non-refundable" — climbed from **rank 6 to rank 2**. For a question about
refund policy, that is exactly the context a good answer needs: it lets the LLM
contrast annual with monthly instead of answering half the question.

The junk `d6` was evicted entirely. **This is the whole value proposition:
reranking does not find new documents, it fixes the order of the ones you
already had.** Which also means it cannot rescue you if the answer never made
it into the candidate pool — recall is stage 1's job, forever.

## The cost, honestly

```text
retrieve(k=8)     0.041 ms
rerank 8 pairs    0.027 ms
rerank / retrieve 0.7x
```

Those measurements are real, and they are **not representative** — the stand-in
reranker is pure Python string work, so it is cheaper than BM25 here. A real
cross-encoder is a transformer forward pass per pair. Realistic figures:

| Stage | Typical latency | Notes |
|---|---|---|
| BM25 / vector search, top-50 | 5–20 ms | Scales with corpus, sublinear |
| MiniLM cross-encoder, 50 pairs, GPU | 30–60 ms | Batched in one forward pass |
| MiniLM cross-encoder, 50 pairs, CPU | 300–900 ms | Often the whole latency budget |
| Hosted rerank API, 50 docs | 100–300 ms | Plus network, plus per-call cost |
| LLM-as-reranker, 50 docs | 1–5 s | Highest quality, rarely worth it |

The decision rule: **reranking latency scales linearly with candidate count**,
so `top_n` is your cost dial. Going from 50 to 100 candidates doubles rerank
cost for a usually-marginal recall gain. Measure the curve on your golden set
(lesson 10) and pick the knee.

## Choosing the candidate count

| `top_n` retrieved | Recall of gold doc | Rerank cost | Verdict |
|---|---|---|---|
| 5 | Low — misses are unrecoverable | Negligible | Under-retrieving |
| 20 | Good for focused corpora | Low | Fine default for small corpora |
| 50 | Strong | Moderate | The common production choice |
| 100 | Marginally better | 2× of 50 | Only if eval proves it |
| 500 | Diminishing | Prohibitive | Almost never |

Retrieve 50, rerank, keep 3–5. That single line is the default worth memorizing.

## Traps

- **Reranking cannot fix recall.** If the gold document is at rank 87 and you
  retrieve 50, no reranker will ever see it. When hit rate is bad, widen stage 1
  or improve chunking — do not add a reranker and hope.
- **Latency budget blowout.** A CPU cross-encoder over 50 candidates can cost
  more than everything else in your pipeline combined. Rerank on GPU, cache
  aggressively for repeated queries, or cut `top_n`.
- **Chunk size fights the reranker.** Cross-encoders have a token limit
  (commonly 512 for the query *and* document together). Chunks longer than that
  get silently truncated, so the reranker scores only the opening of each chunk
  — and a chunk whose answer lives in its last paragraph gets scored on its
  first. Long chunks help the LLM read context but hurt the reranker; this is a
  real tradeoff, not a free choice.
- **Trusting reranker scores as probabilities.** Raw cross-encoder outputs are
  logits, not calibrated confidence. A top score of 4.2 means "best of these
  50", not "correct". Calibrate a threshold on your own data before using it to
  abstain.
- **Reranking a pool of noise.** If stage 1 returns 50 irrelevant documents, the
  reranker returns the 3 *least* irrelevant, with high confidence. Weak
  retrieval plus a confident reranker is one of the most effective ways to
  produce a fluent, well-cited, completely wrong answer.
- **Forgetting the abstain path.** Keep a floor on the reranker score. Empty
  results remain a valid, honest outcome.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| Bi-encoder | Encodes query and doc separately; fast, indexable |
| Cross-encoder | Encodes the pair jointly; accurate, query-time only |
| Pipeline | Retrieve 50 (cheap) → rerank → keep 3–5 |
| What it fixes | Ordering and precision, never recall |
| Cost driver | Linear in candidate count — `top_n` is the dial |
| Token limit | ~512 tokens per pair; long chunks get truncated |
| Free local models | `ms-marco-MiniLM-L-6-v2`, BGE-reranker (need torch) |
| Guardrail | Score floor, or you rank noise confidently |

## Exercise

Instrument the two-stage pipeline and produce a **cost/quality curve** on your
golden set from lesson 1.

For `top_n` in `{5, 10, 20, 50}`, measure at final `k=3`: hit rate, MRR, and
mean wall-clock latency split by stage. Plot quality against latency and find
the knee — the point past which more candidates buy nothing.

Then investigate the two failure modes directly:

1. **Recall ceiling.** For every query the reranked top-3 misses, check whether
   the gold document was in the stage-1 pool at all. Split your misses into
   "reranker's fault" and "retrieval's fault". Which dominates? That tells you
   where to spend your next hour.
2. **The truncation tradeoff.** Re-chunk your corpus at roughly double the
   length and rerun. Hit rate will likely move in *opposite* directions for
   retrieval and reranking. Explain which effect won and why.

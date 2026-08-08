# 10 · Project — Hybrid Search Documentation Assistant

Level 1's capstone was a working RAG bot. This one is the same bot, made
*measurably* better — and "measurably" is the entire assignment.

You now know five techniques that plausibly improve retrieval: BM25, hybrid
fusion, reranking, query expansion, and metadata filtering. Each costs latency,
complexity, or money. The skill this project builds is not implementing them —
you have done that — it is **proving which ones earn their keep on your
corpus**, and deleting the ones that don't. Spoiler from the reference run
below: two of the four configurations bought exactly nothing.

!!! note "What ran here"
    The evaluation harness below **ran locally** against the 16-document corpus
    using `rank-bm25` and `scikit-learn`. As in lesson 2, the "dense" channel is
    a character n-gram TF-IDF stand-in, not a neural embedder (`torch` was not
    installed), and the reranker is lesson 3's pair-scoring stand-in. The
    numbers are real numbers from real code; they are **not** a claim about what
    these techniques do on a production corpus with real embeddings. Your job is
    to produce the equivalent table for your own data.

## Project layout

```text
hybrid-assistant/
├── README.md
├── requirements.txt
├── config.py          # every knob, one file
├── ingest.py          # multi-format -> clean text -> chunks -> index (L7)
├── retrieve.py        # analyzer + sparse + dense + RRF + rerank + filters (L1-L5)
├── rewrite.py         # expansion + conversational rewrite (L4)
├── bot.py             # CLI loop with citations and refusal
├── evaluate.py        # the ablation harness -- the point of this project
└── golden.json        # >= 20 questions, incl. unanswerable
```

## `config.py`

Every technique gets a flag — you cannot ablate what you cannot switch off.

```python
# retrieval
CANDIDATE_K   = 20        # per channel, before fusion (L2)
FINAL_K       = 3         # chunks into the prompt (L1)
RRF_K         = 60        # fusion constant (L2)
MIN_SPARSE    = 0.0       # drop zero-score BM25 hits (L1)

# feature flags -- each one is a row in your ablation table
USE_DENSE     = True      # L2
USE_SPARSE    = True      # L1
USE_RERANK    = True      # L3
USE_EXPANSION = True      # L4
RERANK_POOL   = 8         # candidates into the reranker (L3)

DONT_KNOW = "I don't know based on the available documents."
```

## The retrieval stack

Each layer is one function, and each maps to one lesson.

```python
def sparse(q, k=CANDIDATE_K):
    s = bm25.get_scores(analyze(q))
    return sorted([i for i in range(len(s)) if s[i] > MIN_SPARSE],
                  key=lambda i: -s[i])[:k]          # L1: 0.0 is not a hit

def dense(q, k=CANDIDATE_K):
    s = cosine_similarity(vec.transform([q]), M)[0]
    return sorted([i for i in range(len(s)) if s[i] > 0.01],
                  key=lambda i: -s[i])[:k]

def rrf(rankings, k=RRF_K, top=10):                 # L2
    f = {}
    for r in rankings:
        for pos, i in enumerate(r, 1):
            f[i] = f.get(i, 0.0) + 1 / (k + pos)
    return sorted(f, key=lambda i: -f[i])[:top]

def pair_score(q, doc):                             # L3 (cross-encoder stand-in)
    qs, d = analyze(q), analyze(doc)
    if not qs:
        return 0.0
    ds = set(d)
    cov = sum(1 for w in qs if w in ds) / len(qs)
    pos = [d.index(w) for w in qs if w in ds]
    prox = 1 / (1 + (max(pos) - min(pos))) if len(pos) > 1 else 0.5
    return 0.75 * cov + 0.25 * prox

def search(q):
    queries = [q] + (EXPANSIONS.get(q, []) if USE_EXPANSION else [])   # L4
    lists = []
    for x in queries:
        if USE_SPARSE: lists.append(sparse(x))
        if USE_DENSE:  lists.append(dense(x))
    fused = rrf(lists)
    if USE_RERANK:                                                     # L3
        return sorted(fused[:RERANK_POOL], key=lambda i: -pair_score(q, TEXTS[i]))
    return fused
```

## `evaluate.py` — the ablation harness

```python
GOLDEN = [                                      # (question, gold chunk id)
    ("how do I get a refund on a yearly subscription", "d0"),
    ("undo a bad release", "d9"),               # paraphrase of "rollback"
    ("card got rejected", "d3"),
    ("what does E4022 mean", "d4"),             # exact-token
    ("turn on 2FA", "d7"),                      # acronym vs "two-factor"
    ("how many API calls am I allowed", "d12"),
    # ... 4 more, see the full run below
]

def run(name, retrieve, k=3):
    hits, rr, misses = 0, 0.0, []
    for q, gold in GOLDEN:
        ids = [IDS[i] for i in retrieve(q)[:k]]
        if gold in ids:
            hits += 1
            rr += 1 / (ids.index(gold) + 1)
        else:
            misses.append(q)
    n = len(GOLDEN)
    print(f"{name:26s} hit@{k} {hits/n:.2f}   MRR {rr/n:.2f}   misses: {len(misses)}")
    return misses
```

Note the golden questions are phrased the way **users** phrase things — "card
got rejected", not "payment card declined error code". A golden set written in
your documentation's vocabulary makes every technique look unnecessary, because
the problem they solve has been assumed away. This is how RAG evaluations lie.

## The result

```text
golden set: 10 questions

A bm25 only                hit@3 0.80   MRR 0.73   misses: 2
B hybrid (rrf)             hit@3 0.80   MRR 0.75   misses: 2
C hybrid + rerank          hit@3 0.80   MRR 0.75   misses: 2
D C + query expansion      hit@3 1.00   MRR 0.88   misses: 0

remaining misses:
  A: ['undo a bad release', 'turn on 2FA']
  B: ['undo a bad release', 'turn on 2FA']
  C: ['undo a bad release', 'turn on 2FA']
  D: []
```

Read this carefully, because it is not the tidy staircase a tutorial would show.

- **Hybrid fusion (B) bought nothing.** Hit rate stayed at 0.80; MRR moved
  0.73 → 0.75, i.e. it reordered slightly. Exactly as lesson 2 warned: the
  "dense" channel here is character n-grams, still *lexical*, and fusing two
  lexical retrievers cannot solve a vocabulary problem. With a real embedding
  model this row is where the misses would most likely be fixed.
- **Reranking (C) bought nothing either.** Also predictable: **reranking cannot
  fix recall** (lesson 3). Both misses failed because the gold document was
  never in the candidate pool, and reordering an empty set changes nothing.
- **Query expansion (D) fixed everything.** 0.80 → 1.00 hit rate, 0.75 → 0.88
  MRR. The two failing queries were pure vocabulary mismatches — "undo a bad
  release" vs `deploy --rollback`, and "turn on 2FA" vs "Two-factor
  authentication" — both solved by a **hard-coded synonym list**: zero LLM
  calls, zero added latency, about six lines of code.

The lesson survives contact with production: **the cheapest intervention won,
and the two fashionable ones did not**. Shipping B and C on faith would have
bought two retrievers, a cross-encoder, and a few hundred milliseconds of
latency for an MRR gain of 0.02. You only know that because you measured — run
the same harness on your corpus and you may get the opposite ordering.

## Ablation table to fill in — rows E and F are yours

| Config | Techniques | hit@3 | MRR | Latency | LLM calls | Ship it? |
|---|---|---|---|---|---|---|
| A | BM25 only | 0.80 | 0.73 | baseline | 0 | Baseline |
| B | + dense fusion (RRF) | 0.80 | 0.75 | +1 retrieval | 0 | Not on this corpus |
| C | + cross-encoder rerank | 0.80 | 0.75 | +rerank pass | 0 | Not on this corpus |
| D | + query expansion | **1.00** | **0.88** | ~0 (dictionary) | 0 | **Yes** |
| E | + LLM multi-query | ? | ? | +300–800 ms | 1 | Measure it |
| F | + metadata filters | ? | ? | ~0 | 0 | Measure it |

## Definition of done

- [ ] `ingest.py` handles at least two formats with boilerplate stripping and
      deterministic chunk ids (L7)
- [ ] Sparse + dense retrieval with RRF fusion, each channel filtering
      zero/near-zero scores (L1, L2); a reranker narrows ≥20 candidates to
      `FINAL_K` (L3)
- [ ] Query expansion runs, and the **original query is always retrieved
      alongside** the variants (L4)
- [ ] Metadata filters are pre-filtered, and a tenant-style filter is enforced
      outside any fusion that includes an unfiltered channel (L5)
- [ ] `golden.json` has ≥20 questions phrased as users phrase them (≥3
      exact-token, ≥3 paraphrase, ≥2 unanswerable); unanswerable ones return
      the exact `DONT_KNOW` string via both the empty-retrieval and prompt paths
- [ ] `evaluate.py` prints the full ablation table, with latency per config
- [ ] README documents **one technique you removed** because the numbers did not
      justify it

That last item is not decoration. Shipping a pipeline you cannot justify is how
RAG systems become slow, expensive, and unimprovable.

## Traps

- **A golden set written from your documents.** Phrase questions in the docs'
  own words and BM25 scores near-perfectly, so you conclude nothing else is
  needed. Write questions first, from real user language — ideally a support
  queue.
- **Optimizing hit rate while answer quality falls.** More context is not better
  context; `k=10` often improves hit rate and *worsens* answers (L1 lesson 9).
- **Confident answers from weak retrieval.** The most dangerous configuration is
  a strong reranker over a weak candidate pool: it returns the least-irrelevant
  documents with high scores, and the LLM writes a fluent, cited, wrong answer.
  Keep an absolute score floor, not just a ranking.
- **Chunking changes invalidate everything.** Re-chunking changes what "the gold
  document" means. Version chunking config with your golden set, or your
  historical numbers are fiction.
- **Measuring on ten questions.** The reference run above is a demonstration,
  not a benchmark. With 10 questions, one flip moves hit rate by 0.10 — well
  inside noise. Aim for 50+ before trusting a 0.03 difference. And every
  ablation showing no gain is permission to delete code: take it.

## Stretch goals

Pick one and do it properly — end to end, with before/after numbers from your
own harness.

1. **Replace the stand-ins with the real thing.** Install
   `sentence-transformers` and swap in `all-MiniLM-L6-v2` for the dense channel
   and `ms-marco-MiniLM-L-6-v2` for the reranker, then re-run the full ablation.
   Do rows B and C finally earn their place? This is the most informative
   experiment in the level: it tests the exact hypothesis the reference run
   could not.

2. **LLM expansion vs the dictionary.** Row D won with hard-coded synonyms,
   which do not generalize to queries you did not anticipate. Implement lesson
   4's `MULTI_QUERY_PROMPT` and compare against the dictionary on a *held-out*
   set of 20 questions. Report quality, latency, and cost per query — when does
   the LLM justify itself?

3. **Confidence calibration.** Build an abstain rule from fused rank, raw
   channel scores, and reranker score. On your unanswerable questions: what
   fraction are correctly refused, and how many answerable ones do you wrongly
   refuse? Plot the tradeoff and choose a threshold you can defend.

4. **Conversational memory.** Add lesson 4's rewrite step, a 3-turn history, and
   golden sequences where turn 2 is meaningless alone. Measure hit rate on
   follow-up turns with and without rewriting.

5. **Route table questions.** Add a table and lesson 6's router, sending
   aggregation questions to SQL. Report end-to-end answer accuracy, not just
   retrieval.

When you are done, you will have something more valuable than a working
assistant: a harness that tells you whether your next idea is worth shipping.
Level 3 turns that instinct on production concerns — scale, cost, freshness, and
the failure modes that only appear under load.

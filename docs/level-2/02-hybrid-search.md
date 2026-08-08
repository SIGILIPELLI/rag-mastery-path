# 02 · Hybrid Search (Dense + Sparse)

Lesson 1 ended with a corpus where BM25 scored `0.00` on every document for the
query "undo a bad release" — while Level 1's embedding-based retriever found
the rollback document at distance 0.517 for essentially the same question. Two
retrievers, opposite blind spots. Hybrid search is the obvious move: run both,
then merge the results.

The merge is where it gets interesting, because dense and sparse scores live on
incompatible scales and cannot simply be added.

!!! warning "What ran here, and what didn't"
    The fusion code below is real and executed. But the "dense" channel in these
    examples is a **character n-gram TF-IDF vectorizer**, not a neural embedding
    model — `torch`/`sentence-transformers` were not installed for this run.
    That stand-in is genuinely useful for showing the *mechanics* of fusion, and
    it is honest about its limits: it is still lexical, so it does **not** solve
    vocabulary mismatch. Where that matters, the text says so explicitly and
    describes what a real embedding model would do instead.

## Two rankings, one query

```python
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

bm25 = BM25Okapi([tok(t) for t in TEXTS])

# Stand-in for a dense embedding model. In production this is
# model.encode(...) against a vector store -- the interface is the same:
# query in, ranked list of document indices out.
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
MATRIX = vec.fit_transform(TEXTS)

def sparse_rank(query, k=10):
    scores = bm25.get_scores(tok(query))
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

def dense_rank(query, k=10):
    sims = cosine_similarity(vec.transform([query]), MATRIX)[0]
    return sorted(range(len(sims)), key=lambda i: -sims[i])[:k]
```

The key design point: **both retrievers expose the same interface** — a query
string in, an ordered list of document indices out. That uniformity is what
makes fusion a ten-line function instead of a refactor.

## Why you cannot just add the scores

```python
q = "E4022 billing address"
sparse = bm25.get_scores(tok(q))
dense = cosine_similarity(vec.transform([q]), M)[0]

print(f"BM25 : min {sparse.min():.3f}  max {sparse.max():.3f}")
print(f"dense: min {dense.min():.3f}  max {dense.max():.3f}")
```

```text
BM25 : min 0.000  max 7.506
dense: min 0.000  max 0.631
```

BM25 scores are **unbounded** — they depend on corpus statistics, query length,
and IDF, and they change as you add documents. Cosine similarity is bounded to
`[-1, 1]`. Add them directly and BM25 wins every tie by an order of magnitude;
the dense channel is decoration.

The usual patch is min-max normalization per query:

```python
def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else x * 0

ns, nd = minmax(sparse), minmax(dense)
for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
    combined = alpha * nd + (1 - alpha) * ns
    order = sorted(range(len(combined)), key=lambda i: -combined[i])[:3]
    print(f"  alpha={alpha:.1f}  {[IDS[i] for i in order]}")
```

```text
  alpha=0.0  ['d4', 'd0', 'd1']
  alpha=0.3  ['d4', 'd3', 'd10']
  alpha=0.5  ['d4', 'd3', 'd10']
  alpha=0.7  ['d4', 'd3', 'd10']
  alpha=1.0  ['d4', 'd3', 'd10']
```

`d4` is the correct answer and it holds rank 1 throughout — but notice how the
*rest* of the list flips the moment `alpha` leaves 0, and then never moves
again. That flatness is a warning sign, not a success: on a 16-document corpus
there is not enough signal for `alpha` to be meaningfully tunable.

Min-max normalization also has a nasty property in production: it is computed
**per query**, so the top hit always normalizes to exactly 1.0 whether it is a
perfect match or garbage. You destroy the absolute-confidence information you
need for the "I don't know" threshold from Level 1 lesson 5.

## Reciprocal Rank Fusion

RRF sidesteps the scale problem entirely by throwing scores away and using only
**ranks**:

$$\text{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(d)}$$

```python
def rrf(rankings, k=60, top=3):
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    order = sorted(fused, key=lambda i: -fused[i])[:top]
    return [(IDS[i], fused[i], TEXTS[i]) for i in order]

for q in ["E4022 billing address", "undo a bad release", "refund an annual plan"]:
    s, d = sparse_rank(q), dense_rank(q)
    print(f"=== {q!r}")
    print(f"  BM25   top3: {[IDS[i] for i in s[:3]]}")
    print(f"  dense  top3: {[IDS[i] for i in d[:3]]}")
    for doc_id, score, text in rrf([s, d]):
        print(f"    {score:.4f}  [{doc_id}] {text[:58]}")
```

```text
=== 'E4022 billing address'
  BM25   top3: ['d4', 'd0', 'd1']
  dense  top3: ['d4', 'd3', 'd10']
  fused  top3:
    0.0328  [d4] Error E4022 means the billing address failed AVS verificat
    0.0315  [d3] Error E4021 means the payment card was declined by the iss
    0.0311  [d0] Annual plans can be refunded within 14 days of purchase or

=== 'undo a bad release'
  BM25   top3: ['d14', 'd0', 'd1']
  dense  top3: ['d0', 'd14', 'd7']
  fused  top3:
    0.0325  [d14] Enterprise customers get a dedicated support channel and a
    0.0325  [d0] Annual plans can be refunded within 14 days of purchase or
    0.0310  [d1] Monthly plans are non-refundable but can be cancelled at a

=== 'refund an annual plan'
  BM25   top3: ['d0', 'd1', 'd2']
  dense  top3: ['d0', 'd1', 'd2']
  fused  top3:
    0.0328  [d0] Annual plans can be refunded within 14 days of purchase or
    0.0328  [d1] Monthly plans are non-refundable but can be cancelled at a
```

Query 1 works: `d4` is confirmed at the top by both channels, and `d3` (the
sibling error code) is promoted because the dense channel liked it.

**Query 2 fails, and the failure is the lesson.** `undo a bad release` still
returns nothing useful. Fusing two lexical retrievers cannot invent semantic
understanding neither one has. With a real sentence-embedding model —
`all-MiniLM-L6-v2` or similar — the dense channel would rank `d9` ("Rollbacks
are performed with the `deploy --rollback` command") first, and RRF would carry
it straight to the top. That is precisely the result Level 1 lesson 5 showed.

Do not let a fusion layer convince you that you have a semantic retriever. Fusion
combines the strengths you actually have; it does not create new ones.

### Why `k = 60`

The `k` constant damps the influence of top ranks. With `k=60`, rank 1 scores
`1/61 = 0.0164` and rank 10 scores `1/70 = 0.0143` — only a 15% gap. Small `k`
(say 1) makes rank 1 dominate: `1/2` vs `1/11`, a 5.5× gap.

- **Small `k`** → trust each retriever's top hit strongly; one confident
  retriever can win alone.
- **Large `k`** → reward documents that *both* retrievers rank decently.
  Consensus beats individual confidence.

60 is the value from the original RRF paper and is a genuinely good default.
It is also why `d8` sometimes edges out a better `d9` in fused lists: appearing
in more lists beats ranking higher in one.

## RRF vs weighted scores

| | Weighted normalized scores | Reciprocal Rank Fusion |
|---|---|---|
| Needs score normalization | Yes, and it is fragile | No |
| Tunable per corpus | Yes, via `alpha` | Barely, via `k` |
| Keeps confidence info | Only if normalized globally | No — ranks only |
| Handles 3+ retrievers | Awkward — weights must sum | Trivial — add lists |
| Robust to a broken retriever | Poor — bad scores leak in | Good — bad ranks dilute |
| Supports "I don't know" | Yes, with raw scores | Needs a separate check |
| Typical use | You have eval data to tune on | Sensible default, day one |

**Start with RRF.** Move to weighted fusion only when you have a golden set
(lesson 10) proving a specific `alpha` beats it.

## Traps

- **Losing the abstain path.** RRF outputs are always positive and always
  ranked, so "nothing matched" looks identical to "great match". Keep the raw
  scores alongside the fused rank and apply your Level 1 distance/score
  threshold *before* fusion. Otherwise hybrid search becomes a hallucination
  machine that never says "I don't know".
- **Fusing all-zero rankings.** If BM25 returns nothing above 0.0, its ranking
  is corpus order, and RRF will dutifully fuse that noise into the result.
  Filter `score > 0` per channel first.
- **Different chunk sets per channel.** If your BM25 index and vector store were
  built from different chunkings, fusion is comparing incomparable things. Index
  once, share ids across both channels.
- **Fetching too few candidates.** Fuse the top 20–50 from each channel, not the
  top 3. Fusion needs overlap to work with; it is a reranking of a candidate
  pool, and lesson 3 depends on that pool being wide.
- **Double latency.** You now run two retrievers per query. They are trivially
  parallelizable — issue both concurrently rather than sequentially.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| Why hybrid | Sparse and dense fail on disjoint query types |
| Score scales | BM25 unbounded; cosine in `[-1, 1]` — never add raw |
| Min-max norm | Per-query; destroys absolute confidence |
| RRF formula | `sum(1 / (k + rank))` over all rankings |
| `k` = 60 | Paper default; larger `k` rewards consensus |
| Candidate depth | Fuse top 20–50 per channel, not top 3 |
| Fusion is not magic | It cannot add capability neither retriever has |
| Abstain | Threshold on raw scores before fusing |

## Exercise

Reuse the 12-query audit set from lesson 1. Build a hybrid retriever with both
channels and evaluate three configurations at `k=3`: BM25 alone, dense alone,
and RRF-fused. Record hit rate and MRR for each, broken down by your three
query groups (exact-token, keyword, paraphrase).

Then push on the parts that matter:

1. Sweep RRF's `k` over `{1, 10, 60, 200}`. Which group is most sensitive, and
   why does that match the intuition about consensus vs confidence?
2. Vary candidate depth per channel over `{3, 10, 50}` with `k=60` fixed. Find
   the depth where hit rate stops improving — that is your production setting.
3. Deliberately break one channel (return a shuffled list) and re-measure. How
   much does RRF degrade compared to weighted fusion with `alpha=0.5`? This is
   the robustness argument for RRF, in numbers you generated yourself.

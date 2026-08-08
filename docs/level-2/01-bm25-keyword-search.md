# 01 · Keyword Search & BM25

Level 1 built everything on embeddings, and embeddings have one blind spot that
matters enormously in practice: they are bad at *exact tokens*. Ask a vector
store about error code `E4022`, an order id, a config flag, or a rare product
name, and it will happily hand you something semantically adjacent and
completely wrong. Keyword search — the fifty-year-old technology embeddings
were supposed to replace — nails those queries cold.

This lesson builds BM25 from scratch so you know exactly what it scores, then
switches to a library and breaks it on purpose. The failures you see here are
what lesson 2's hybrid search exists to fix.

!!! note "What actually ran"
    Every code block and every output block on this page was executed locally
    with `rank-bm25` and Python's standard library. No embedding model, no API
    key, no GPU. That's the point of starting here: sparse retrieval is cheap
    enough to run anywhere.

## The intuition: rare words carry the signal

A document matching your query on "the" tells you nothing; one matching on
"E4022" tells you almost everything. BM25 formalizes that with three ideas:
**term frequency** (more occurrences means more relevant, with diminishing
returns), **inverse document frequency** (rare terms are worth more), and
**length normalization** (a term in a 10-word doc is stronger evidence than the
same term in a 1000-word doc).

## BM25 from scratch

```python
import math, re
from collections import Counter

CORPUS = [
    "Error E4021 means the payment card was declined by the issuing bank.",
    "Annual plans can be refunded within 14 days of purchase or renewal.",
    "Rollbacks are performed with the deploy --rollback command.",
    "Monthly plans are non-refundable but can be cancelled at any time.",
    "Deploys to production happen automatically when main is merged.",
]

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

docs = [tokenize(d) for d in CORPUS]
N = len(docs)
avgdl = sum(len(d) for d in docs) / N

df = Counter()
for d in docs:
    for term in set(d):
        df[term] += 1

def idf(term):
    n = df.get(term, 0)
    return math.log((N - n + 0.5) / (n + 0.5) + 1)

def bm25(query, k1=1.5, b=0.75):
    q = tokenize(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            num = tf[term] * (k1 + 1)
            den = tf[term] + k1 * (1 - b + b * dl / avgdl)
            s += idf(term) * num / den
        scores.append(s)
    return scores

for q in ["E4021", "refund an annual plan"]:
    print(f"query: {q!r}")
    for score, text in sorted(zip(bm25(q), CORPUS), reverse=True)[:3]:
        print(f"  {score:.3f}  {text}")
    print()
```

```text
query: 'E4021'
  1.309  Error E4021 means the payment card was declined by the issuing bank.
  0.000  Rollbacks are performed with the deploy --rollback command.
  0.000  Monthly plans are non-refundable but can be cancelled at any time.

query: 'refund an annual plan'
  1.309  Annual plans can be refunded within 14 days of purchase or renewal.
  0.000  Rollbacks are performed with the deploy --rollback command.
  0.000  Monthly plans are non-refundable but can be cancelled at any time.
```

The `E4021` query is a clean win — an embedding model would have to *memorize*
that token to compete. But look closely at the second query: it scored 1.309,
the exact same score as a single-term match. `refund` did not match
`refunded`, and `plan` did not match `plans`. Only `annual` matched.

The two knobs: **`k1`** (default ~1.2–1.5) controls TF saturation — lower means
the second occurrence of a term barely helps. **`b`** (default 0.75) controls
length normalization: `b=0` ignores document length, `b=1` normalizes fully.

## Using a real library, and breaking it

```python
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi([tokenize(t) for t in TEXTS])   # TEXTS: 16-doc support corpus

def search(query, k=3):
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(IDS[i], scores[i], TEXTS[i]) for i in order]

for q in ["E4022", "how do I roll back a deploy", "refund policy"]:
    print(f"query: {q!r}")
    for doc_id, score, text in search(q):
        print(f"  {score:5.2f}  [{doc_id}] {text}")
    print()
```

```text
query: 'E4022'
   2.50  [d4] Error E4022 means the billing address failed AVS verification.
   0.00  [d0] Annual plans can be refunded within 14 days of purchase or renewal.
   0.00  [d1] Monthly plans are non-refundable but can be cancelled at any time.

query: 'how do I roll back a deploy'
   3.20  [d14] Enterprise customers get a dedicated support channel and a 4 hour SLA.
   2.62  [d9] Rollbacks are performed with the deploy --rollback command.
   0.00  [d0] Annual plans can be refunded within 14 days of purchase or renewal.

query: 'refund policy'
   2.29  [d13] The webhook retry policy attempts delivery five times over one hour.
   0.00  [d0] Annual plans can be refunded within 14 days of purchase or renewal.
   0.00  [d1] Monthly plans are non-refundable but can be cancelled at any time.
```

Three queries, two disasters. This is not a broken library — this is what raw
BM25 does, and why every real search stack has an *analyzer* in front of it.
For `roll back a deploy`, the top hit is an SLA document about enterprise
support: it won because the stopword `a` appears **twice** in it, so a short
document stuffed with common words beat an exact topical match. For `refund
policy`, the webhook retry *policy* wins while the two documents that literally
describe refunds score **0.00** — `refunded` and `non-refundable` are different
strings from `refund`.

## The fix: a real analyzer

Tokenization is not a detail you get to skip. Removing stopwords and reducing
words to a common stem changes the ranking completely:

```python
STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of",
             "and", "or", "in", "on", "for", "with", "how", "do", "i", "my",
             "can", "at", "by", "it", "this", "that", "from", "but"}

def stem(word):
    for suffix in ("ability", "able", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word   # a real stemmer (Porter/Snowball) is far better

def analyze(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]

bm25 = BM25Okapi([analyze(t) for t in TEXTS])
```

```text
query: 'how do I roll back a deploy'  -> tokens ['roll', 'back', 'deploy']
   2.09  [d9] Rollbacks are performed with the deploy --rollback command.
   1.83  [d8] Deploys to production happen automatically when main is merged.

query: 'refund policy'  -> tokens ['refund', 'policy']
   2.06  [d13] The webhook retry policy attempts delivery five times over one hour.
   1.83  [d1] Monthly plans are non-refundable but can be cancelled at any time.

query: 'undo a bad release'  -> tokens ['undo', 'bad', 'release']
   0.00  [d0] Annual plans can be refunded within 14 days of purchase or renewal.
   0.00  [d1] Monthly plans are non-refundable but can be cancelled at any time.
```

The deploy query is fixed. `refund policy` is half-fixed — the refund document
climbed into the top 2, but "policy" is a generic word BM25 cannot discount
contextually.

The third query is the important one. **`undo a bad release` scores 0.00 on
every document in the corpus.** The answer sits right there in `d9`, and BM25
cannot see it, because query and document share zero tokens. No amount of
stemming or stopword tuning fixes a vocabulary mismatch.

Note also the silent trap: every score is 0.00, yet the function still returned
three documents in corpus order. **A ranking of all-zero scores is not a
ranking.** Filter on `score > 0` before handing anything to an LLM, or you feed
it confidently-formatted noise — the most reliable way to manufacture a
hallucination.

## Sparse vs dense: complementary failure modes

| Query type | BM25 | Dense embeddings |
|---|---|---|
| Error codes, ids, SKUs (`E4022`) | Excellent | Poor — unseen tokens |
| Rare jargon, product names | Excellent | Poor unless in training data |
| Exact quoted phrases | Excellent | Mediocre |
| Paraphrase (`undo a bad release`) | **Zero** | Excellent |
| Synonyms (`2FA` / `two-factor`) | **Zero** | Excellent |
| Cross-lingual | Zero | Good with multilingual models |
| Cost to index | Trivial, CPU | GPU or API, per-token cost |
| Explainability | Total — you see the terms | Opaque |

Read that table as the argument for lesson 2: these are not competing options
where one wins, they fail on *disjoint* query sets.

## Traps

- **Skipping the analyzer.** Raw `.split()` gives you the stopword disaster
  above. Production engines (Elasticsearch, OpenSearch, Postgres FTS) ship
  proper analyzers with real stemmers — use them rather than the toy `stem()`
  here, which will mangle words like "address" or "business". Prefer
  under-stemming; aggressive stemmers collapse "university" and "universe".
- **Treating a 0.0 score as a weak match.** It is not weak, it is *nothing*.
- **Chunk length interacts with `b`.** BM25 penalizes long documents. If your
  chunker emits a mix of 100-token and 2000-token chunks, the short ones win
  systematically regardless of relevance. Consistent chunk sizes are a
  retrieval-quality decision, not just a tidiness one.
- **Assuming BM25 is obsolete.** It remains a brutally strong baseline. Many
  "our RAG is bad" incidents are fixed by adding BM25, not a bigger model.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| TF saturation (`k1`) | Repeated terms give diminishing returns |
| IDF | Rare terms score higher; stopwords score near zero |
| Length norm (`b`) | Short docs favored; 0.75 is a sane default |
| Analyzer | Lowercase, strip stopwords, stem — not optional |
| Score 0.0 | No lexical overlap at all — drop the hit |
| Best at | Ids, codes, jargon, exact phrases |
| Blind to | Paraphrase and synonyms |
| Library | `rank-bm25` for learning; Elasticsearch/Postgres FTS in production |

## Exercise

Take the 16-document corpus (or your own from Level 1) and build a
**query-type audit**. Write 12 queries in three groups of four: exact-token
queries (codes, ids, quoted phrases), keyword queries that share vocabulary
with the docs, and pure-paraphrase queries that deliberately share *no* words
with the target document.

Run all 12 through BM25 with the analyzer, recording the gold document's rank
and the top hit's score. Then answer with numbers: what fraction of each group
does BM25 get at rank 1, how many paraphrase queries return an all-zero
ranking, and does *any* setting of `b` from 0.0 to 1.0 rescue a paraphrase
query? (It will not — proving that to yourself is the point.)

Keep this query set. You will reuse it in lesson 2 to measure exactly how much
hybrid search buys you, and again in lesson 10 to grade the finished assistant.

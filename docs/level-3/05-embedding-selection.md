# 05 · Embedding Model Selection & Fine-Tuning

MTEB leaderboard rank is the number everyone quotes and the number that
matters least for your corpus. A model that tops MTEB's average across 56
benchmark datasets can still underperform a smaller model on *your* support
tickets, *your* legal contracts, or *your* codebase — because MTEB's average
is exactly that, an average across domains none of which is yours. This
module covers benchmarking on your own data, the dimension/cost trade-off,
and when fine-tuning beats picking a different off-the-shelf model.

As in Level 2, `torch`/`sentence-transformers` don't fit in this environment.
Everything below runs on TF-IDF (`scikit-learn`), which is real, runnable, and
— usefully — bad in a very informative way for this specific lesson.

## Benchmarking on your own data, for real

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

docs = [
    "Refunds are issued within 30 days for annual plans.",
    "Our API rate limit is 1000 requests per minute.",
    "Password reset emails expire after one hour.",
    "Annual subscribers get a 30-day money-back guarantee.",
    "The API returns HTTP 429 when rate limited.",
]

# (query, index of the doc that actually answers it)
eval_set = [
    ("what is the refund policy", 0),
    ("how many api calls can I make", 1),
    ("how long is a password reset link valid", 2),
]

vec = TfidfVectorizer()
X = vec.fit_transform(docs)

def top1_accuracy(vectorizer, X, eval_set):
    hits = 0
    for q, rel_idx in eval_set:
        qv = vectorizer.transform([q])
        sims = cosine_similarity(qv, X)[0]
        hits += int(np.argmax(sims) == rel_idx)
    return hits / len(eval_set)

print("tfidf top-1 accuracy:", top1_accuracy(vec, X, eval_set))
```

Captured output:

```
tfidf top-1 accuracy: 0.3333333333333333
```

One out of three. Let's see exactly why:

```python
for q, rel in eval_set:
    qv = vec.transform([q])
    sims = cosine_similarity(qv, X)[0]
    print(q, "-> retrieved:", np.argmax(sims), "expected:", rel, "sims:", sims.round(2))
```

Captured output:

```
what is the refund policy -> retrieved: 4 expected: 0 sims: [0.   0.25 0.   0.   0.26]
how many api calls can I make -> retrieved: 4 expected: 1 sims: [0.   0.28 0.   0.   0.3 ]
how long is a password reset link valid -> retrieved: 2 expected: 2 sims: [0.   0.2  0.44 0.   0.  ]
```

Only the third query — which happens to share the exact words "password" and
"reset" with its target chunk — retrieves correctly. The first two fail
because "refund policy" shares zero exact tokens with "Refunds are issued...
money-back guarantee," and "how many api calls" shares zero exact tokens with
"rate limit... 1000 requests." **This is the textbook case for a real semantic
embedding model over TF-IDF**: "refund" and "money-back" mean the same thing
to a human and to a transformer embedding, but TF-IDF only counts shared
character sequences, and gets a near-tie on the wrong document (0.26 vs. 0.25)
purely from incidental overlap. Trying `ngram_range=(1,2)` doesn't fix this —
verified separately, it produces the identical 0.33 accuracy, because the
problem isn't phrase granularity, it's the total absence of any word-meaning
signal. This gap is exactly what production systems pay for a transformer
embedding model to close.

## Dimension vs. cost

```python
dims = {
    "MiniLM-L6 (384d)": 384,
    "bge-base (768d)": 768,
    "text-embedding-3-large (3072d)": 3072,
}
n_vectors = 5_000_000
for name, d in dims.items():
    gb = n_vectors * d * 4 / 1e9   # float32
    print(name, "->", round(gb, 1), "GB for", n_vectors, "vectors")
```

Captured output:

```
MiniLM-L6 (384d) -> 7.7 GB for 5000000 vectors
bge-base (768d) -> 15.4 GB for 5000000 vectors
text-embedding-3-large (3072d) -> 61.4 GB for 5000000 vectors
```

An 8x dimension increase (384 → 3072) is an 8x memory bill, linearly — and
that's before the HNSW graph overhead from module 04. Higher-dimension models
usually *do* score better on retrieval quality, but "usually" is doing a lot
of work: the only way to know if the extra dimensions buy you anything on
*your* corpus is the accuracy benchmark above, run against each candidate
model, not the MTEB leaderboard.

## When to fine-tune instead of switching models

Fine-tuning an embedding model means training it further on pairs from your
own domain: `(query, relevant_chunk)` positive pairs, ideally with hard
negatives (chunks that look relevant but aren't). It's the right call when:

- Your benchmark shows every off-the-shelf model plateaus below your quality
  bar, and the failures are domain-vocabulary mismatches (medical, legal,
  internal jargon) rather than genuine ambiguity.
- You have — or can generate — at least a few thousand labeled query/chunk
  pairs. Fewer than that, fine-tuning tends to overfit and generalizes worse
  than the base model it started from.
- You can afford the *operational* cost: a fine-tuned model needs versioning,
  re-training when your domain vocabulary shifts, and a decision about
  whether every embedding in your index needs to be regenerated when the
  model changes (it does — old and new embeddings from different model
  versions are not comparable, and mixing them silently corrupts similarity
  scores).

If your benchmark instead shows one off-the-shelf model beating another by a
wide margin, switch models first — it's free and reversible. Fine-tuning is
for closing a gap no available model closes.

## The trap: eval set contamination and non-representative queries

The `eval_set` above has three questions, hand-picked to look like production
queries. Two failure modes to guard against when you build a real one:

- **Vocabulary-matched eval queries** — if you write eval questions by
  paraphrasing chunk text closely, you overstate every model's accuracy
  (including TF-IDF's) because the eval set never tests the paraphrase gap
  real users create. Pull real query logs, or have someone *not* looking at
  the source docs write the questions.
- **Re-embedding staleness** — after fine-tuning or switching models, every
  vector in your index was computed by the *old* model. Cosine similarity
  between an old-model chunk vector and a new-model query vector is
  meaningless — not degraded, meaningless. The entire index must be
  re-embedded and re-indexed on a model change, which is itself a real-cost
  migration to plan for (module 08 covers doing this without downtime).

## Cheat sheet

| Signal | Action |
|---|---|
| MTEB rank looks great, your eval accuracy doesn't | Trust your eval, not MTEB |
| Failures are exact-word matches only | Vocabulary/semantic gap — try a stronger embedding model |
| Failures persist across every off-the-shelf model tried | Candidate for fine-tuning, if you have labeled pairs |
| Just switched embedding models | Re-embed the entire index — old and new vectors aren't comparable |
| Eval questions read like paraphrased chunks | Rebuild eval set from real user queries |

## Exercise

Add two more `(query, relevant_doc_index)` pairs to `eval_set` where the query
uses a synonym for a word in its target chunk (e.g., "outage" for "downtime"),
re-run `top1_accuracy`, and confirm TF-IDF still fails those specifically —
then write, in a comment, what output you'd expect if you swapped in a real
sentence-transformer model for the same eval set.

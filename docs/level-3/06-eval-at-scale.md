# 06 · Evaluation at Scale

Level 1's eval harness checked a handful of hand-written question/answer
pairs against a handful of documents. That's enough to catch a broken
pipeline. It's not enough to catch a pipeline that's 5% worse than last week,
or one that's great on the ten questions you thought of and bad on the
thousand real ones users actually ask. This module scales the harness up:
synthetic question generation, RAGAS-style metrics implemented from scratch,
and a CI regression gate.

## The metrics, implemented for real

RAGAS (and similar frameworks) score a RAG pipeline on more than "did it get
the answer right." Here are the four that matter most, as runnable Python —
these are simplified versions of the real definitions, not API calls to the
RAGAS library, but the logic matches:

```python
def context_precision(retrieved, relevant_set):
    """Of what you retrieved, how much was actually relevant?"""
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant_set)
    return hits / len(retrieved)

def context_recall(retrieved, relevant_set):
    """Of what was relevant, how much did you retrieve?"""
    if not relevant_set:
        return 1.0
    hits = sum(1 for r in relevant_set if r in retrieved)
    return hits / len(relevant_set)

def faithfulness(answer, context_text):
    """Of the answer's claims, how many are actually supported by context?"""
    claims = [c.strip() for c in answer.split(".") if c.strip()]
    ctx_words = set(context_text.lower().split())
    supported = 0
    for c in claims:
        words = set(w.strip(",") for w in c.lower().split())
        overlap = len(words & ctx_words)
        if overlap / max(len(words), 1) > 0.5:
            supported += 1
    return supported / max(len(claims), 1)
```

Run precision and recall on a retrieval result:

```python
retrieved = [0, 3, 4]     # doc indices your retriever returned
relevant = {0, 3}         # doc indices that actually answer the question

print("precision:", context_precision(retrieved, relevant))
print("recall:", context_recall(retrieved, relevant))
```

Captured output:

```
precision: 0.6666666666666666
recall: 1.0
```

Full recall (found everything relevant) with only 67% precision (one of three
retrieved chunks was noise) — a completely normal, and importantly *not
contradictory*, pair of scores. A pipeline can hit 100% recall by
over-retrieving, which is why precision has to be tracked alongside it, not
instead of it.

Now faithfulness, on a correct vs. a hallucinated answer over the same context:

```python
context = "Refunds are issued within 30 days for annual plans."
good_answer = "Refunds are issued within 30 days for annual plans."
bad_answer = "Refunds are issued within 90 days and require manager approval."

print("faithfulness (good):", faithfulness(good_answer, context))
print("faithfulness (bad):", faithfulness(bad_answer, context))
```

Captured output:

```
faithfulness (good): 1.0
faithfulness (bad): 0.0
```

The bad answer isn't garbled text — it reads fluently and confidently. That's
exactly why faithfulness has to be measured against the *retrieved context*,
not judged on fluency: an LLM asked to answer from context will still
sometimes substitute its own (wrong) prior knowledge, and nothing about the
output's grammar signals that.

## The trap: metrics that are trivially gameable

```python
all_docs = list(range(20))
print("recall retrieving everything:", context_recall(all_docs, relevant))
print("precision retrieving everything:", context_precision(all_docs, relevant))
```

Captured output:

```
recall retrieving everything: 1.0
precision retrieving everything: 0.1
```

A pipeline that returns every document in the corpus scores a perfect 1.0 on
recall — technically true, and completely useless as a signal, because
precision collapses to 0.1. This is the general shape of **eval metric
gaming**: any single metric, optimized in isolation, has a degenerate strategy
that satisfies it while destroying the system's actual usefulness. The same
pattern hits faithfulness (an answer that just quotes the context verbatim,
with no synthesis, scores perfect faithfulness while being a worse answer than
one that reasons over the context) and answer relevance (a longer, more
generic answer often scores as "relevant" without being more useful).

The fix is never a single number — track precision *and* recall *and*
faithfulness *and* answer relevance together, and be suspicious of any
pipeline change that improves one sharply while the others move the wrong
way. That correlation is usually the tell that something's being gamed rather
than genuinely improved.

## Synthetic question generation at scale

Hand-writing eval questions doesn't scale past a few dozen. The standard
approach: for each chunk, prompt an LLM to generate a question that chunk
answers, then hold out that (question, chunk) pair as a labeled eval example.

```python
def generate_eval_pairs(chunks, question_generator):
    """question_generator: chunk -> question string (an LLM call in production)"""
    pairs = []
    for i, chunk in enumerate(chunks):
        q = question_generator(chunk)
        pairs.append({"question": q, "relevant_chunk_id": i})
    return pairs

# Toy generator standing in for the LLM call — extracts the chunk's subject.
def toy_question_generator(chunk):
    first_words = " ".join(chunk.split()[:3])
    return f"What does the document say about {first_words}?"

chunks = [
    "Refunds are issued within 30 days for annual plans.",
    "API rate limits reset every 60 seconds.",
]
pairs = generate_eval_pairs(chunks, toy_question_generator)
for p in pairs:
    print(p)
```

Captured output:

```
{'question': 'What does the document say about Refunds are issued?', 'relevant_chunk_id': 0}
{'question': 'What does the document say about API rate limits?', 'relevant_chunk_id': 1}
```

The toy generator produces awkward but usable questions — a real LLM call
produces natural ones. Either way, this scales linearly with corpus size:
10,000 chunks gives 10,000 labeled eval pairs with no manual labeling, which
is the entire point. The trade-off: synthetic questions tend to be
vocabulary-matched to their source chunk (module 05's trap, again), so they
systematically overstate retrieval accuracy versus real user queries. Mix
synthetic pairs with a smaller set of real, logged user queries for an
honest number.

## CI regression suites

The eval harness only earns its keep once it runs automatically and blocks
regressions:

```python
def run_regression_suite(pipeline, eval_pairs, min_recall=0.85):
    scores = []
    for pair in eval_pairs:
        retrieved = pipeline(pair["question"])
        scores.append(context_recall(retrieved, {pair["relevant_chunk_id"]}))
    avg = sum(scores) / len(scores)
    if avg < min_recall:
        raise AssertionError(f"regression: recall {avg:.2f} below floor {min_recall}")
    return avg
```

Wire this into CI as a real test that fails the build, with a `min_recall`
floor set from your current baseline minus a small margin — not from an
aspirational target. The floor's job is catching regressions, not chasing
improvement; raise it deliberately after a verified quality gain, never as a
guess.

## Cheat sheet

| Metric | Answers | Gameable by |
|---|---|---|
| Context precision | Is retrieval noisy? | N/A alone — pairs with recall |
| Context recall | Did retrieval miss anything? | Retrieving everything |
| Faithfulness | Is the answer grounded in context? | Verbatim-quoting context with no synthesis |
| Answer relevance | Does the answer address the question? | Longer, vaguer answers |
| CI regression gate | Did this change make things worse? | A floor set too low to ever trip |

## Exercise

Extend `run_regression_suite` to also compute average faithfulness (using a
provided `answer_generator`) and fail the build if faithfulness drops below
0.8 even when recall is fine. Run it against a `pipeline` that retrieves
correctly but an `answer_generator` that sometimes hallucinates, and confirm
the suite catches the faithfulness regression that a recall-only gate would
miss entirely.

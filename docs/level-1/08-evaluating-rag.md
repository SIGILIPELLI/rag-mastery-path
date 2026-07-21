# 08 · Evaluating RAG

"It looks right when I try it" is not evaluation. RAG systems fail quietly —
a chunking tweak that helps five questions silently breaks three others — and
without numbers you'll never notice. Evaluation is hard here because there are
*two* things to measure (did we retrieve the right text? did we generate a
good answer?) and the second is a judgment call. This lesson builds a small
but real eval harness: a golden question set, retrieval metrics (hit rate and
MRR), and LLM-as-judge scoring for answer quality.

## Why RAG eval is genuinely hard

- **Two coupled stages.** A wrong answer might be retrieval's fault (evidence
  never arrived) or generation's fault (evidence arrived, model fumbled).
  End-to-end scores alone can't tell you which — so we measure the stages
  separately.
- **No single right answer.** "14 days" and "Annual plans can be refunded for
  two weeks after purchase" are both correct. Exact-match scoring fails;
  something has to judge *meaning*.
- **Small samples mislead.** With 5 test questions, one flake swings your
  score by 20 points. You need enough questions to see signal — 20+ to start,
  growing over time.

## The golden question set

The foundation: questions with known answers *and* known evidence locations.
Keep it in a JSON file next to your pipeline, and grow it every time a user
hits a failure:

```python
# golden.json
GOLDEN = [
    {
        "question": "Can I get a refund on an annual plan?",
        "expected_answer": "Yes, within 14 days of purchase or renewal.",
        "relevant_ids": ["billing.md-0"],       # chunk ids that contain the evidence
    },
    {
        "question": "How do I undo a bad production deploy?",
        "expected_answer": "Run the 'deploy --rollback' command.",
        "relevant_ids": ["engineering.md-0"],
    },
    {
        "question": "When are invoices sent out?",
        "expected_answer": "On the 1st of each month, to the account owner.",
        "relevant_ids": ["billing.md-0"],
    },
    {
        "question": "Do you offer student discounts?",
        "expected_answer": "I don't know based on the available documents.",
        "relevant_ids": [],                     # unanswerable — by design!
    },
]
```

Include unanswerable questions on purpose: a pipeline that never says "I don't
know" fails them all, and that's a bug worth catching.

## Retrieval metrics: hit rate and MRR

Retrieval eval is mechanical — no LLM, no judgment, fast enough to run on
every change:

- **Hit rate@k** — fraction of questions where *any* relevant chunk appears
  in the top k. The "can the answer even be found?" number.
- **MRR (Mean Reciprocal Rank)** — average of `1/rank` of the first relevant
  chunk (0 if absent). Rewards putting the evidence *first*, where the LLM
  pays most attention.

```python
def eval_retrieval(golden: list[dict], k: int = 4) -> None:
    answerable = [g for g in golden if g["relevant_ids"]]
    hits, rr = 0, 0.0
    for g in answerable:
        res = col.query(
            query_embeddings=[model.encode(g["question"]).tolist()], n_results=k
        )
        returned_ids = res["ids"][0]
        ranks = [returned_ids.index(rid) + 1
                 for rid in g["relevant_ids"] if rid in returned_ids]
        if ranks:
            hits += 1
            rr += 1.0 / min(ranks)
        else:
            print(f"  MISS: {g['question']!r} -> got {returned_ids}")
    n = len(answerable)
    print(f"hit_rate@{k}: {hits/n:.2f}   MRR: {rr/n:.2f}   ({n} questions)")

eval_retrieval(GOLDEN)
# hit_rate@4: 1.00   MRR: 0.83   (3 questions)
```

Reading the numbers: hit rate 1.0 means the evidence always arrived; MRR 0.83
means it usually — not always — arrived at rank 1. If hit rate is low, fix
chunking/embedding/k *before* touching prompts: generation can't cite what
retrieval never delivered.

## Answer quality: LLM-as-judge

For the end-to-end answer, use a second LLM call to compare the pipeline's
answer against the expected answer. Crude exact-match fails on paraphrases;
a judge handles them:

```python
import anthropic, json

judge = anthropic.Anthropic()   # needs ANTHROPIC_API_KEY

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Expected answer (ground truth): {expected}
System's answer: {actual}

Grade the system's answer:
- "correct": factually matches the expected answer (wording may differ)
- "partial": contains the right facts but is incomplete or adds unsupported claims
- "wrong": contradicts the expected answer, or answers when it should have refused

Reply with JSON only: {{"grade": "...", "reason": "..."}}"""

def judge_answer(question: str, expected: str, actual: str) -> dict:
    resp = judge.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, expected=expected, actual=actual)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text)

def eval_answers(golden: list[dict]) -> None:
    grades = []
    for g in golden:
        actual = ask(g["question"])            # lesson 7's pipeline
        result = judge_answer(g["question"], g["expected_answer"], actual)
        grades.append(result["grade"])
        print(f"{result['grade']:8s} {g['question']}")
    n = len(grades)
    print(f"\ncorrect: {grades.count('correct')/n:.0%}  "
          f"partial: {grades.count('partial')/n:.0%}  "
          f"wrong: {grades.count('wrong')/n:.0%}")
```

```text
correct  Can I get a refund on an annual plan?
correct  How do I undo a bad production deploy?
partial  When are invoices sent out?
correct  Do you offer student discounts?

correct: 75%  partial: 25%  wrong: 0%
```

LLM-as-judge caveats, honestly stated: judges are ~90–95% aligned with humans
on clear-cut grades, worse on "partial"; they drift if you change judge model
or prompt; and they cost an API call per question. Mitigations: keep the
rubric tight and categorical (not 1–10 scores), spot-check a sample of grades
by hand, and pin the judge model version.

## The workflow that makes eval pay off

1. Build the golden set (start with 20 questions; include unanswerables).
2. Record baseline: `hit_rate`, `MRR`, `%correct`.
3. Change **one thing** (chunk size, k, threshold, prompt wording).
4. Re-run. Keep the change only if the numbers improve.
5. Every real-world failure becomes a new golden question — the set grows
   into your regression suite.

This loop — not any single metric — is what separates a demo from a system.

## Cheat sheet

| Metric | What it measures | Cost |
|--------|------------------|------|
| Hit rate@k | Evidence present in top-k at all | Free, fast — run constantly |
| MRR | How high the first relevant chunk ranks | Free, fast |
| LLM-as-judge grade | End-to-end answer correctness | One API call per question |
| Unanswerable questions | "I don't know" discipline | Include ~20% of the set |
| Golden set | Ground truth: question + answer + evidence ids | Grows from real failures |
| Debug rule | Low hit rate → fix retrieval first | — |

## Exercise

Build a golden set of 10 questions (including 2 unanswerable) for the corpus
you indexed in lesson 7, and run `eval_retrieval` at k=1, k=3, and k=5.
Then make one deliberate degradation — re-ingest with `max_chars=200` (tiny
chunks) — and re-run. Which metric moves more, hit rate or MRR, and why?
Finally, run `eval_answers` on both versions and check whether the retrieval
regression is visible in the end-to-end grades — you've just witnessed (or
refuted) the "garbage in, garbage out" chain empirically.

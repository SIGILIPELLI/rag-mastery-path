# 08 · Fine-Tuning vs. RAG vs. Hybrid

Every level up to this one has assumed RAG is the answer and focused on
doing it well. This module steps back: RAG isn't always the right tool.
Fine-tuning solves a different problem, and a lot of enterprise projects
pick the wrong one — or reach for both without a clear reason for either —
because the two get pitched as competitors when they're usually
complementary.

## What each actually changes

- **RAG** changes what the model *knows about right now*, by handing it
  facts at query time. The model's weights never change.
- **Fine-tuning** changes what the model *does by default* — its tone,
  format, task-specific behavior, or domain vocabulary — by further training
  its weights on examples. The knowledge available to it at inference time
  doesn't grow; how it responds does.

This distinction predicts the failure mode of using the wrong one: fine-tune
a model to "know" your company's current pricing, and it's wrong the moment
pricing changes — the fact is baked into weights that don't update without a
whole new training run. RAG a model to "sound like our brand voice," and it
still doesn't, because retrieved documents give it facts, not a persistent
behavioral pattern to imitate on every response regardless of what's
retrieved.

## The cost/update-speed asymmetry, quantified

```python
rag_update_cost = 0.00002 * (200 / 1000)     # one embedding call for one changed chunk
finetune_cost_per_run = 50                    # illustrative small fine-tune job
finetune_time_hours = 3
rag_update_time_seconds = 2

print("RAG single-fact update cost: $", rag_update_cost)
print("Fine-tune update cost: $", finetune_cost_per_run, "over", finetune_time_hours, "hours")
print("cost ratio finetune/rag:", finetune_cost_per_run / rag_update_cost)
```

Captured output:

```
RAG single-fact update cost: $ 4.000000000000001e-06
Fine-tune update cost: $ 50 over 3 hours
cost ratio finetune/rag: 12499999.999999998
```

Updating one fact costs roughly twelve and a half million times more via
fine-tuning than via RAG in this illustration, and takes hours instead of
seconds. This is the single clearest argument for RAG over fine-tuning
whenever the underlying facts change with any regularity — pricing, policy,
inventory, personnel, anything with a "last updated" date. It is also why
"just fine-tune it on our knowledge base" is usually the wrong instinct for
a knowledge-freshness problem, even though it sounds like the more thorough
approach.

## Where fine-tuning wins instead

Fine-tuning is the right tool when the problem isn't "the model doesn't know
X," it's "the model doesn't behave like Y":

- **Structured output compliance** — reliably producing a specific JSON
  schema, a specific citation format, or a house style, more consistently
  than prompting alone achieves.
- **Domain vocabulary and phrasing** — a model fine-tuned on medical or
  legal text tends to produce more natural, precise phrasing in that domain
  than a general model prompted with the same facts via RAG.
- **Task specialization** — a smaller, cheaper model fine-tuned narrowly for
  one repeated task (classification, extraction, a fixed-format summary) can
  match a larger general model's quality on that task at a fraction of the
  inference cost — this is a distinct axis from cost optimization (module
  03)'s model-tiering, but the same underlying principle: don't pay for
  general capability you don't need.
- **Embedding fine-tuning specifically** (Level 3 module 05) — tuning the
  *retriever's* embedding model on domain query/chunk pairs, which is a
  fine-tuning technique in service of a RAG system, not a competitor to it.

## The hybrid pattern most production systems actually land on

```python
def answer(question, model, retriever):
    context = retriever(question)              # RAG: fresh facts
    return model.generate(question, context)   # fine-tuned model: consistent format/tone

# model = a model fine-tuned for house style + structured citations
# retriever = a standard RAG pipeline supplying current facts
```

This is not a compromise — it's the two techniques doing what each is
actually good at: fine-tuning shapes *how* the model responds (format,
tone, domain fluency), RAG supplies *what* it responds with (current,
retrievable facts). A model fine-tuned to always cite sources in a specific
format, fed context from a RAG pipeline, gets both properties neither
technique alone provides — the fine-tuning doesn't need to encode any actual
facts, only the response pattern, which is exactly the kind of stable target
fine-tuning handles well and knowledge-freshness handles badly.

## The trap: choosing based on what's technically impressive, not the actual problem

The recurring mistake is starting from "we have budget/appetite for
fine-tuning" or "RAG is the standard architecture now" and working backward
to justify it, instead of starting from the failure mode observed:

- If eval failures (Level 3 module 06 / Level 4 module 07) look like
  **outdated or missing facts**, that's a retrieval-coverage or
  freshness problem (Level 3 module 08) — fine-tuning cannot fix it, because
  the facts still aren't in the model's weights after training unless you
  retrain on every update, which is the exact problem you started with.
- If failures look like **wrong format, wrong tone, or wrong task framing
  despite correct facts being retrieved**, that's a behavioral problem — more
  retrieval, bigger `top_k`, a better reranker (Level 3 modules 04, 05) won't
  fix it, because the facts were already right.
- If failures look like **both**, you likely need both, applied to their
  respective symptom — not a bigger version of either one alone.

Diagnose from the failure, not from which technique your team already knows
how to operate.

## Cheat sheet

| Symptom | Points to |
|---|---|
| Facts are correct at training time, stale by launch | RAG, not fine-tuning |
| Correct facts, wrong format/tone/structure | Fine-tuning, not more retrieval |
| Need a cheap, narrow, high-volume task done well | Fine-tuned small model |
| Facts change frequently, format must stay consistent | Hybrid — fine-tuned model + RAG context |
| "Let's fine-tune on our whole knowledge base" | Almost always the wrong call — that's what RAG is for |

## Exercise

Take one real eval failure from your own project (or one from Level 3 module
06's synthetic set) and classify it as a knowledge-freshness failure, a
behavioral/format failure, or both — then write out, concretely, which
change (retrieval fix, fine-tuning run, or both) would actually resolve it,
and what a regression test for that fix would check.

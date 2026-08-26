# 07 · Continuous Evaluation & Feedback Loops

Level 3 module 06 built a CI regression suite that runs against a fixed eval
set. That catches regressions in code you changed. It does not catch quality
drift from a shifting corpus, a shifting user population, or a model
provider silently updating a model behind a stable API name — none of which
show up as a code diff. This module closes that gap with continuous,
production-traffic evaluation and feedback loops that feed back into the
eval set itself.

## Detecting a real regression from noisy feedback

```python
import random
random.seed(7)

def simulate_feedback(n, base_rate):
    """Thumbs-up rate at `base_rate`, simulating real user feedback noise."""
    return [1 if random.random() < base_rate else 0 for _ in range(n)]

week1 = simulate_feedback(200, 0.85)
week2 = simulate_feedback(200, 0.85)
week3 = simulate_feedback(200, 0.65)   # a real quality regression happened here

def rate(feedback):
    return sum(feedback) / len(feedback)

r1, r2, r3 = rate(week1), rate(week2), rate(week3)
print("week1:", r1, "week2:", r2, "week3:", r3)
```

Captured output:

```
week1: 0.86 week2: 0.845 week3: 0.65
```

Even with an unchanged underlying 85% satisfaction rate, weeks 1 and 2 don't
match exactly (0.86 vs 0.845) — that's expected sampling noise, not drift.
Week 3's drop to 0.65 is a real, injected regression. The engineering
question is distinguishing the two:

```python
def flag_regression(baseline_rate, current_rate, threshold=0.1):
    return (baseline_rate - current_rate) > threshold

baseline = (r1 + r2) / 2
print("baseline:", round(baseline, 3))
print("week3 flagged:", flag_regression(baseline, r3))
```

Captured output:

```
baseline: 0.853
week3 flagged: True
```

A fixed threshold (here, 10 percentage points) separates real regressions
from sampling noise — set too tight, normal week-to-week variance
constantly false-alarms; set too loose, real regressions like a bad model
update slip through for weeks. The right threshold depends on your feedback
volume: with only 200 samples/week, the noise floor between week 1 and week
2 was already 1.5 points, so a 3-point threshold would have been useless
noise; a 10-point threshold correctly stays quiet on noise and fires on the
real regression. Higher traffic volume lets you tighten the threshold and
catch smaller regressions faster — this is a statistics problem as much as
an engineering one.

## Sampling production traffic for deeper review

Thumbs-up/down feedback is cheap but shallow — most users don't leave
feedback at all, and a "thumbs up" doesn't tell you *which* metric (module
06: faithfulness, relevance, precision) is at issue. Continuous evaluation
supplements it by sampling a slice of real traffic for a full metric
computation (human review or an LLM-judge pass, not just implicit feedback):

```python
def sample_for_review(requests, rate=0.05, seed=1):
    rnd = random.Random(seed)
    return [r for r in requests if rnd.random() < rate]

reqs = list(range(1000))
sampled = sample_for_review(reqs)
print("sampled for review:", len(sampled))
```

Captured output:

```
sampled for review: 54
```

Roughly 5% of 1,000 requests, as configured — sampling rate is a direct
trade-off between review cost (human time, or LLM-judge API cost) and how
quickly a regression affecting only a subset of query types gets noticed. A
regression narrow enough to affect 2% of traffic can hide in a 5% sample for
a long time; widen the sample, or stratify it (sample more heavily from
query categories with historically lower scores) rather than sampling
uniformly at random once volume allows it.

## Closing the loop: failures become eval cases

The actual value of continuous evaluation isn't the dashboard — it's feeding
discovered failures back into the regression suite so they can never
regress silently again:

```python
def augment_eval_set(eval_set, flagged_failures):
    return eval_set + [f for f in flagged_failures if f not in eval_set]

eval_set = ["q1", "q2"]
failures = ["q2", "q3"]   # q2 already known, q3 is new
print(augment_eval_set(eval_set, failures))
```

Captured output:

```
['q1', 'q2', 'q3']
```

Every production failure a human or LLM-judge flags during sampled review
becomes a permanent regression test case, deduplicated against what's
already tracked. This is what turns continuous evaluation from a monitoring
dashboard into an actual quality ratchet — Level 3's CI suite only protects
against regressions on cases someone already thought to write down; this
loop is how that eval set grows to cover cases someone only discovered by
looking at real traffic.

## The trap: feedback loops that reward the wrong thing

- **Selection bias in explicit feedback.** Users who leave thumbs-down
  feedback are disproportionately the ones with a bad enough experience to
  bother — a feedback-rate-based quality score is measuring "rate of
  extremely bad outcomes," not average quality, and can look stable while
  average quality erodes quietly.
- **LLM-judge drift mirroring the system it judges.** If the same model
  family powers both generation and the LLM-judge doing sampled review, a
  systematic blind spot in the model (a factual error type it consistently
  gets wrong) can go undetected because the judge shares the same blind
  spot. Mixing judge models, or anchoring judge prompts to explicit rubrics
  rather than open-ended "is this good," reduces but doesn't eliminate this.
- **Eval-set growth without pruning.** `augment_eval_set` only ever adds
  cases. An eval suite that grows for years without removing cases that no
  longer reflect real usage (a product feature that was deprecated, a query
  pattern users stopped asking) becomes slow to run and full of noise that
  obscures genuinely new regressions — schedule periodic review of what's in
  the suite, not just what's added to it.

## Cheat sheet

| Signal source | Catches | Misses |
|---|---|---|
| Explicit thumbs up/down | Severe, obvious failures | Mediocre-but-not-bad answers (no signal) |
| Sampled traffic + judge/human review | Deeper metric failures (faithfulness, relevance) | Anything outside the sampled slice |
| Fixed-threshold regression alert | Real week-over-week drops | Regressions smaller than the noise floor |
| Eval-set augmentation from failures | Any specific failure recurring | Novel failure types not yet seen once |
| CI suite alone (Level 3 module 06) | Regressions from code changes | Drift from corpus/model/user-base shift with no code change |

## Exercise

Extend `flag_regression` to use a statistical test (a two-proportion z-test,
or a simple bootstrap confidence interval) instead of a fixed percentage-point
threshold, so the alert sensitivity automatically adjusts to sample size —
confirm it stays quiet on the week1-vs-week2 noise at n=200 but also correctly
flags a smaller, real 5-point regression once you increase `n` to 2,000 in
the simulation.

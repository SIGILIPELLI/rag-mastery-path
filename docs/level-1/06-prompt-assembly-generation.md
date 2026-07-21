# 06 · Prompt Assembly & Generation

You can retrieve perfect chunks and still get a bad answer if the prompt is
sloppy. Prompt assembly is where retrieval meets generation: you format the
retrieved chunks into a context block, wrap them with instructions that keep
the model grounded, and make the API call. This lesson builds a solid
`build_prompt()` and `generate()` you'll reuse in lessons 7–10, including
source citations and an honest "I don't know" path.

!!! info "API key needed for this lesson"
    The generation call uses the Anthropic API: `pip install anthropic` and
    set the `ANTHROPIC_API_KEY` environment variable (create a key at
    console.anthropic.com). This is the only paid/networked piece of the
    pipeline — and the pattern is identical for any chat-completions API
    (OpenAI, local models via Ollama, etc.): same prompt in, same answer out.

## Assembling the context block

Give every chunk a numbered tag carrying its source metadata — the tags are
what make citations possible:

```python
def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a numbered, source-tagged context block."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        src = hit["meta"].get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{hit['text']}")
    return "\n\n".join(blocks)
```

Example output:

```text
[1] (source: billing.md)
Annual plans can be refunded within 14 days of purchase or renewal.

[2] (source: billing.md)
Monthly plans are non-refundable but can be cancelled at any time.
```

## The grounded prompt

Four ingredients, each doing a real job:

```python
def build_prompt(question: str, hits: list[dict]) -> str:
    context = build_context(hits)
    return f"""You are a helpful assistant answering questions using ONLY the \
provided context.

Rules:
- Base your answer solely on the context below. Do not use outside knowledge.
- Cite the context passages you used, like [1] or [1][3].
- If the context does not contain the answer, say exactly: \
"I don't know based on the available documents." Do not guess.
- Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
```

Why each rule earns its place:

- **"ONLY the provided context"** — this is the grounding instruction, the
  line between RAG and a model freestyling with decoration. Without it the
  model happily blends retrieved facts with trained-in guesses, and you can't
  tell which is which.
- **Citations** — force the model to tie claims to passages. Users can
  verify, you can debug, and the model itself gets more careful when it must
  point at evidence.
- **The exact "I don't know" sentence** — models are biased toward being
  helpful, and an unanchored "say if you're unsure" gets ignored. A verbatim
  escape phrase is easy for the model to emit and trivial for your code to
  detect.
- **Concise** — retrieved-context answers tend to bloat with restated
  context. Say what you want.

## The generation call

```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment

def generate(prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
```

Wire it to lesson 5's retriever and run the whole query phase:

```python
question = "Can I get a refund on my annual plan?"
hits = retrieve(question, k=3)                      # lesson 5
answer = generate(build_prompt(question, hits))
print(answer)
```

```text
Yes — annual plans can be refunded within 14 days of purchase or renewal [1].
Note that monthly plans are non-refundable [2].
```

That's a grounded, cited answer. Total new code in this lesson: ~40 lines.

## Grounding vs. hallucination: prove it works

The test that matters is a question the context *can't* answer:

```python
question = "Do you offer student discounts?"
hits = retrieve(question, k=3)          # returns refund/billing chunks — nearest, not relevant
print(generate(build_prompt(question, hits)))
# I don't know based on the available documents.
```

Without the grounding rules, the same model given the same chunks will often
produce something like "Many plans offer student discounts, contact
support..." — fluent, plausible, invented. Run both versions once and the
value of the four rules stops being theoretical.

Two caveats to stay honest about:

- **Grounding is strong, not absolute.** Models occasionally leak outside
  knowledge or over-summarize the context. Instructions cut hallucination
  dramatically; evaluation (lesson 8) is how you *measure* what remains.
- **Garbage context in, garbage answer out.** If retrieval returns junk, the
  model must choose between junk and refusal. Combine the prompt rules with
  lesson 5's distance threshold: when `retrieve_confident()` returns nothing,
  skip the LLM call entirely and return the "I don't know" string yourself —
  cheaper and safer.

```python
def answer_question(question: str) -> str:
    hits = retrieve_confident(question, k=3, max_distance=0.8)
    if not hits:
        return "I don't know based on the available documents."
    return generate(build_prompt(question, hits))
```

## Cheat sheet

| Piece | Purpose |
|-------|---------|
| Numbered context blocks `[1] (source: x)` | Enable citations + debugging |
| "Answer ONLY from context" | The grounding instruction |
| "Cite passages like [1]" | Verifiable, evidence-tied claims |
| Exact "I don't know..." phrase | Reliable, detectable refusal path |
| "Keep it concise" | Prevents context-restating bloat |
| `client.messages.create(model=, max_tokens=, messages=)` | The generation call |
| Threshold before generate | Skip the LLM when retrieval found nothing |
| Provider-agnostic | Same prompt works on any chat-completions API |

## Exercise

Write an *ablation test*: take one answerable question and one unanswerable
question, and run each through (a) the full `build_prompt` above and (b) a
stripped prompt containing only the context and the question with no rules.
Compare the four outputs. Which rules changed the behavior, and on which
question? Then add a fifth rule of your own design (for example: "answer in a
single sentence" or "quote the exact policy wording") and verify the model
follows it.

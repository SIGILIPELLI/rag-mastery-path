# 01 · What Is RAG & Why It Exists

Retrieval-Augmented Generation (RAG) is a simple idea with a big payoff:
instead of asking a language model to answer from whatever it happened to
memorize during training, you **look up the relevant text first** and hand it
to the model along with the question. The model's job shrinks from "know
everything" to "read this and answer" — a task LLMs are extremely good at.
This lesson explains the problems RAG solves, the architecture at a high
level, and when you'd choose RAG over the alternatives.

## The two problems RAG solves

### 1. Knowledge cutoffs and private data

An LLM's knowledge is frozen at its training cutoff, and it has never seen
your private data at all. Ask a plain model about *your* company's refund
policy, *yesterday's* incident report, or *internal* API docs and it has
exactly two options: refuse, or make something up.

```text
User: What is our refund window for annual plans?

Plain LLM: Typically, companies offer a 30-day refund window...   ← a guess
RAG LLM:   Per policy.md, annual plans can be refunded within
           14 days of purchase or renewal.                        ← grounded
```

### 2. Hallucination

When a model doesn't know something, it doesn't return `null` — it produces
the most *plausible-sounding* continuation. That's what generation is. The
output is fluent, confident, and sometimes wrong, which is far more dangerous
than an obvious error. Grounding the model in retrieved text doesn't make
hallucination impossible, but it changes the failure profile: the model can
now quote and cite real passages, and you can instruct it to say "I don't
know" when the retrieved text doesn't contain the answer (lesson 6 covers
exactly how).

## Why not just put everything in the prompt?

Modern models accept hundreds of thousands of tokens of context, so a fair
question is: why not paste the whole knowledge base into every request?

Three reasons:

1. **It doesn't fit.** A context window of even 1M tokens is roughly 3–4 MB of
   text. Real document collections — wikis, ticket archives, codebases — are
   gigabytes. There is no window big enough.
2. **It costs too much.** You pay per input token, on *every* request. Sending
   200K tokens of context to answer a question whose answer lives in one
   paragraph is paying for 199,900 tokens of noise.
3. **Accuracy degrades.** Models are measurably worse at using facts buried in
   the middle of a huge context ("lost in the middle" — lesson 9). A short,
   relevant context outperforms a long, mostly-irrelevant one.

Retrieval is the fix: spend a few milliseconds finding the 5–10 passages that
matter, and send only those.

## The RAG architecture

Every RAG system, from a 100-line script to an enterprise platform, has the
same two phases:

```text
INGESTION (offline, run when documents change)
  documents ──▶ chunk ──▶ embed ──▶ store in vector DB

QUERY (online, run per question)
  question ──▶ embed ──▶ retrieve top-k chunks ──▶ assemble prompt ──▶ LLM ──▶ answer
```

- **Chunking** splits documents into retrievable pieces (lesson 3).
- **Embedding** turns text into vectors so "similar meaning" becomes "nearby
  in space" (lesson 2).
- **The vector store** indexes those vectors for fast nearest-neighbor search
  (lesson 4).
- **Retrieval** finds the chunks most similar to the question (lesson 5).
- **Prompt assembly + generation** stitches the chunks and the question into a
  prompt and calls the model (lesson 6).

Here is the whole query phase as pseudocode-that-is-almost-real-Python — by
lesson 7 you will have written every line of it for real:

```python
question = "What is the refund window for annual plans?"

q_vector = embed(question)                       # lesson 2
chunks = vector_store.query(q_vector, top_k=5)   # lessons 4-5
prompt = build_prompt(question, chunks)          # lesson 6
answer = llm(prompt)                             # lesson 6
```

## RAG vs. fine-tuning vs. long context

RAG is not the only way to get knowledge into a model. Choose deliberately:

- **RAG** injects knowledge *at query time*. Updating knowledge = updating the
  index, which takes seconds and costs nothing. Answers can cite sources. Best
  for factual lookup over changing or private corpora.
- **Fine-tuning** bakes patterns into the model's *weights*. It is the right
  tool for changing *behavior* — style, format, domain vocabulary — but a poor
  tool for injecting facts: it's expensive, slow to update, can't cite
  sources, and models still hallucinate about fine-tuned facts.
- **Long context** (just paste the documents in) is genuinely the simplest
  option and the right one when the corpus is small — a handful of files that
  comfortably fit in the window. No index to build or keep fresh. It stops
  working when the corpus grows, when cost matters, or when latency matters.

These compose: many production systems use RAG for facts *and* a fine-tuned
model for tone, inside a generous context window.

## Cheat sheet

| Term | Meaning |
|------|---------|
| RAG | Retrieve relevant text, then generate an answer grounded in it |
| Hallucination | Fluent, confident, fabricated model output |
| Knowledge cutoff | Date after which the model has seen nothing |
| Ingestion phase | Offline: chunk → embed → store |
| Query phase | Online: embed question → retrieve → prompt → generate |
| Grounding | Constraining the answer to retrieved evidence |
| Fine-tuning | Changing model weights — for behavior, not facts |
| Long context | Skipping retrieval by pasting docs into the prompt — fine for small corpora |

## Exercise

Without writing any code yet, pick a real document collection you have access
to (team wiki, course notes, a project's docs folder) and write down: (1) three
questions a plain LLM would likely hallucinate about it, (2) roughly how large
the collection is in files and megabytes, and (3) whether long context alone
could handle it — and if not, which of the three reasons (size, cost,
accuracy) rules it out. Keep this corpus in mind: you'll index a folder like it
for real in lesson 7 and build a full Q&A bot over it in lesson 10.

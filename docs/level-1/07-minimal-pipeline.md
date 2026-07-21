# 07 · A Minimal End-to-End Pipeline

Time to put lessons 2–6 together into one file you can actually run: a
~100-line script that ingests a folder of text files, chunks them, embeds
them, stores them in a persistent ChromaDB collection, and then answers
questions with cited sources. No framework, no magic — every line is something
you built in a previous lesson. This script is also the skeleton the capstone
project (lesson 10) grows from.

## Setup

```bash
pip install sentence-transformers chromadb anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # only needed for the answer step
mkdir -p my_docs
```

Put a few `.txt` or `.md` files in `my_docs/`. If you don't have any handy:

```bash
cat > my_docs/billing.md <<'EOF'
# Billing

Annual plans can be refunded within 14 days of purchase or renewal.
Monthly plans are non-refundable but can be cancelled at any time.

Invoices are emailed on the 1st of each month to the account owner.
EOF

cat > my_docs/engineering.md <<'EOF'
# Engineering

Deploys to production happen automatically when main is merged.
Rollbacks are performed with the 'deploy --rollback' command.

Staging mirrors production and refreshes its data nightly.
EOF
```

## The pipeline: `rag.py`

```python
"""Minimal end-to-end RAG pipeline: ingest -> chunk -> embed -> store -> retrieve -> generate.

Usage:
    python rag.py ingest ./my_docs        # (re)build the index
    python rag.py ask "your question"     # answer a question (needs ANTHROPIC_API_KEY)
"""
import re
import sys
from pathlib import Path

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "./chroma_db"
COLLECTION = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_DISTANCE = 0.8   # retrieval confidence threshold (lesson 5)

model = SentenceTransformer(EMBED_MODEL)
db = chromadb.PersistentClient(path=DB_PATH)


# ---------- ingestion (lessons 3-4) ----------

def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Pack paragraphs (falling back to sentences) into chunks of <= max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            units.append(p)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())
    chunks, current, size = [], [], 0
    for unit in units:
        if current and size + len(unit) > max_chars:
            chunks.append("\n".join(current))
            current, size = current[-1:], len(current[-1])   # 1-unit overlap
        current.append(unit)
        size += len(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks


def ingest(folder: str) -> None:
    try:
        db.delete_collection(COLLECTION)          # rebuild from scratch
    except Exception:
        pass
    col = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    for path in sorted(Path(folder).glob("**/*")):
        if path.suffix not in {".txt", ".md"}:
            continue
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.name}-{i}")
            texts.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    col.add(ids=ids, embeddings=model.encode(texts).tolist(),
            documents=texts, metadatas=metas)
    print(f"Indexed {len(texts)} chunks from {folder} into {DB_PATH}")


# ---------- query (lessons 5-6) ----------

def retrieve(question: str, k: int = 4) -> list[dict]:
    col = db.get_collection(COLLECTION)
    res = col.query(query_embeddings=[model.encode(question).tolist()], n_results=k)
    hits = [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    return [h for h in hits if h["distance"] <= MAX_DISTANCE]


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] (source: {h['meta']['source']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    return f"""You are a helpful assistant answering questions using ONLY the provided context.

Rules:
- Base your answer solely on the context below. Do not use outside knowledge.
- Cite the context passages you used, like [1] or [1][3].
- If the context does not contain the answer, say exactly: "I don't know based on the available documents."
- Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""


def ask(question: str) -> str:
    hits = retrieve(question)
    if not hits:
        return "I don't know based on the available documents."
    client = anthropic.Anthropic()   # needs ANTHROPIC_API_KEY; any chat API works here
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    answer = "".join(b.text for b in response.content if b.type == "text")
    sources = ", ".join(sorted({h["meta"]["source"] for h in hits}))
    return f"{answer}\n\n(sources searched: {sources})"


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "ingest":
        ingest(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        print(ask(" ".join(sys.argv[2:])))
    else:
        print(__doc__)
```

## Running it

```bash
$ python rag.py ingest ./my_docs
Indexed 4 chunks from ./my_docs into ./chroma_db

$ python rag.py ask "how do I undo a bad deploy?"
Rollbacks are performed with the 'deploy --rollback' command [1].

(sources searched: engineering.md)

$ python rag.py ask "do you offer student discounts?"
I don't know based on the available documents.
```

Ingestion runs once (and only when documents change); asking uses the
persisted index. The unanswerable question never even reaches the API — the
distance threshold catches it first.

## What to notice about the design

- **Two entry points, one file.** `ingest` is the offline phase, `ask` is the
  online phase — the exact split from lesson 1's architecture diagram.
- **Everything is inspectable.** Add `print(hits)` in `ask()` and you see
  precisely what the model saw. When an answer is wrong, that one print
  statement is your debugger (lesson 5's habit).
- **Deterministic chunk ids** (`filename-i`) mean re-ingesting a changed file
  could `upsert` instead of rebuild — the simple rebuild is fine at this
  scale, and Level 3 covers incremental indexing properly.
- **The LLM is a plug.** Swap `generate` internals for OpenAI, a local Ollama
  model, or anything else with a chat endpoint; nothing else changes. The
  pipeline is the product; the model is a component.

## Cheat sheet

| Stage | Function | Lesson it came from |
|-------|----------|---------------------|
| Chunk | `chunk_text()` — paragraph-aware, 1-unit overlap | 3 |
| Embed | `model.encode(texts)` — one model for docs & queries | 2 |
| Store | `PersistentClient` + cosine collection | 4 |
| Retrieve | top-k + distance threshold | 5 |
| Prompt | numbered, source-tagged context + grounding rules | 6 |
| Generate | `messages.create(...)` — provider-swappable | 6 |
| CLI | `ingest <folder>` / `ask <question>` | — |

## Exercise

Extend `rag.py` in three small ways: (1) add a `--show-chunks` flag to `ask`
that prints each retrieved chunk with its distance before the answer; (2) add
a `sources` command that lists every distinct `source` in the collection with
its chunk count (hint: `col.get()` returns all records); (3) point `ingest` at
a real folder of your own notes or docs — at least 10 files — and find one
question where the answer is wrong or missing. Using `--show-chunks`, diagnose
whether the failure is in retrieval (right chunk absent) or generation (right
chunk present, bad answer). That diagnosis skill is exactly what lessons 8 and
9 formalize.

# Level 1 · Entry <span class="level-badge">Foundations</span>

Goal: understand every moving part of a RAG pipeline — embeddings, chunking,
vector storage, retrieval, and grounded generation — and ship a working docs
Q&A bot built from those parts in plain Python.

## Modules

1. [What Is RAG & Why It Exists](01-what-is-rag.md)
2. [Embeddings & Semantic Similarity](02-embeddings.md)
3. [Chunking Strategies](03-chunking.md)
4. [Vector Stores (ChromaDB)](04-vector-stores.md)
5. [Retrieval](05-retrieval.md)
6. [Prompt Assembly & Generation](06-prompt-assembly-generation.md)
7. [A Minimal End-to-End Pipeline](07-minimal-pipeline.md)
8. [Evaluating RAG](08-evaluating-rag.md)
9. [Common Failure Modes](09-failure-modes.md)
10. [Project — Docs Q&A Bot](10-capstone-docs-qa-bot.md)

By the end of this level you'll be able to take a folder of documents, index it
into a vector store, answer questions over it with cited sources, measure how
well your pipeline retrieves and answers, and recognize (and fix) the most
common ways RAG systems fail.

!!! info "Setup for this level"
    ```bash
    pip install sentence-transformers chromadb anthropic
    ```
    `sentence-transformers` and `chromadb` run locally and are free — no API
    key, no network calls after the first model download. Only the generation
    lessons (06, 07, 08, 10) additionally use the Anthropic API, which needs an
    `ANTHROPIC_API_KEY` environment variable. Everything else runs without one.

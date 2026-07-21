# RAG Pipelines Mastery Path

A structured, module-wise training program on **Retrieval-Augmented Generation
(RAG)** — from your first embedding to production-grade, master-level RAG
systems — with runnable Python code in every module and a hands-on project at
the end of each level.

RAG is the technique behind almost every "chat with your documents" product:
instead of hoping a language model memorized your data, you **retrieve** the
relevant passages at question time and let the model **generate** an answer
grounded in them. This site teaches that pipeline from first principles, in
plain Python — no framework required to understand what's actually happening.

## How the program is organized

| Level | Focus | Modules |
|-------|-------|---------|
| [Level 1 · Entry](level-1/index.md) | Embeddings, chunking, vector stores, retrieval, grounded generation, a full working pipeline | 9 topics + 1 project |
| [Level 2 · Intermediate](level-2/index.md) | BM25 & hybrid search, reranking, query rewriting, table RAG, LangChain & LlamaIndex | 9 topics + 1 project |
| [Level 3 · Advanced](level-3/index.md) | Agentic RAG, multi-hop retrieval, GraphRAG, production vector DBs, eval at scale | 9 topics + 1 project |
| [Level 4 · Master](level-4/index.md) | Enterprise architecture, multi-tenancy, cost/latency optimization, RAG security | 9 topics + 1 capstone |

## What you need

- **Python 3.10+** and `pip`. Level 1 uses two free, local, no-API-key
  libraries: [`sentence-transformers`](https://www.sbert.net/) for embeddings
  and [`chromadb`](https://www.trychroma.com/) for the vector store.
- The **generation** step (turning retrieved text into an answer) is shown with
  the Anthropic API (`pip install anthropic`), which requires an API key. Every
  other part of the pipeline runs fully offline, and any chat-completions API
  works the same way — the lessons flag exactly where a key is needed.

## How to use this site

- Work through each level in order — later modules assume earlier ones.
- Every topic page has runnable code — copy it into a local `.py` file and run
  it. Code that needs an API key says so explicitly.
- Each level ends with a project that combines everything learned in that level.
- Use the search bar (top of the page) to jump straight to a topic.

Start here → [Level 1 · Entry](level-1/index.md)

## Related tracks

RAG sits between machine learning and LLM application development. Two sister
sites cover the neighboring ground:

- [AI/ML Mastery Path](https://sigilipelli.github.io/ai-ml-mastery-path/) — machine-learning foundations
- [LLM Development Mastery Path](https://sigilipelli.github.io/llm-dev-mastery-path/) — building LLM applications

## More from the Mastery Path series

Free, module-wise, entry-to-master training for languages and platforms:

- [Python Mastery Path](https://sigilipelli.github.io/python-mastery-path/)
- [Java Mastery Path](https://sigilipelli.github.io/java-mastery-path/)
- [JavaScript Mastery Path](https://sigilipelli.github.io/javascript-mastery-path/)
- [TypeScript Mastery Path](https://sigilipelli.github.io/typescript-mastery-path/)
- [SQL Mastery Path](https://sigilipelli.github.io/sql-mastery-path/)
- [Go Mastery Path](https://sigilipelli.github.io/go-mastery-path/)
- [Rust Mastery Path](https://sigilipelli.github.io/rust-mastery-path/)
- [AWS Mastery Path](https://sigilipelli.github.io/aws-mastery-path/)
- [Azure Mastery Path](https://sigilipelli.github.io/azure-mastery-path/)
- [GCP Mastery Path](https://sigilipelli.github.io/gcp-mastery-path/)

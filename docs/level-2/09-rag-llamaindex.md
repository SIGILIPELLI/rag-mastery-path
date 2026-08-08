# 09 · RAG with LlamaIndex

LangChain is a general framework for LLM applications that happens to do RAG
well. LlamaIndex is a framework built *specifically* for connecting LLMs to
data, and the difference shows in its defaults: a working RAG pipeline is three
lines, and the interesting abstractions are all about how documents relate to
each other.

Where LangChain gives you `Retriever` and lets you compose, LlamaIndex gives you
`Index` and `QueryEngine` and opinions about how retrieval should work.

!!! warning "None of this lesson's code was executed"
    `llama-index` and its dependencies were **not installed** for this run — the
    environment was capped at `rank-bm25`, `scikit-learn`, and `pypdf`, and the
    full stack pulls in `torch` via the embedding models. The code below matches
    the current `llama-index-core` v0.11+ package layout and is correct in
    shape, but unlike lessons 1–7 there are **no verified output blocks**.
    Verify against the docs for your installed version.

## The three-line version, and why to distrust it

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("what is the refund policy?")
```

That genuinely works, and it is a fair demo. It is also the most dangerous
snippet in this level, because it silently chooses your chunk size (1024 tokens,
20 overlap), your embedding model, your `top_k` (2), your prompt template, and
your LLM — every knob Level 1 taught you to tune, decided by a library that has
never seen your corpus.

Use it to check your data loads. Do not use it to decide whether RAG works for
your problem.

## The vocabulary

| LlamaIndex concept | What it actually is | Your equivalent |
|---|---|---|
| `Document` | A source file with metadata | Your `(text, meta)` dict |
| `Node` | A chunk, plus relationships to neighbours | Your chunk — with links |
| `NodeParser` / splitter | Chunking strategy | Your `chunk()` (L1 lesson 3) |
| `Index` | Nodes + a retrieval structure over them | Your Chroma collection |
| `Retriever` | Query → nodes | Your `retrieve()` |
| `NodePostprocessor` | Filter/rerank after retrieval | Lesson 3's reranker |
| `ResponseSynthesizer` | Context → prompt → answer | Your prompt assembly (L1 lesson 6) |
| `QueryEngine` | The whole pipeline, callable | Your `answer()` |

The concept without a Level 1 counterpart is **`Node` relationships**. A node
knows its `PREVIOUS`, `NEXT`, `PARENT`, and `SOURCE` nodes. That graph is what
powers LlamaIndex's most distinctive feature, below.

## Building it explicitly

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-sonnet-5", max_tokens=1000)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

documents = SimpleDirectoryReader("./docs", recursive=True).load_data()
index = VectorStoreIndex.from_documents(documents)
```

`Settings` is a global. That is convenient for scripts and a genuine liability
in a service where different indexes want different models — prefer passing
`embed_model=` and `llm=` explicitly once you are past prototyping.

`SentenceSplitter` respects sentence boundaries rather than cutting at a
character count, which is the same improvement `RecursiveCharacterTextSplitter`
offers in LangChain, arrived at from a different direction.

## Hybrid retrieval and reranking

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

vector_retriever = index.as_retriever(similarity_top_k=20)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=20
)

# lessons 2 + 4: fuses retrievers AND generates query variants
fusion = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4,               # 1 original + 3 LLM-generated variants
    mode="reciprocal_rerank",    # RRF, k=60
    use_async=True,
)

# lesson 3
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=4
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=fusion, node_postprocessors=[rerank]
)
```

`QueryFusionRetriever` is the neatest abstraction in either framework: it
collapses lessons 2 and 4 into one object, running multi-query expansion and
RRF fusion together, asynchronously. Set `num_queries=1` to disable expansion
and get pure hybrid fusion.

Be aware of what `num_queries=4` costs: one LLM call to generate variants, then
`4 queries × 2 retrievers = 8` retrieval calls, then a cross-encoder over the
fused pool, then generation. That is lesson 4's latency warning, packaged so
neatly you may not notice you opted in.

Metadata filtering (lesson 5) is explicit and typed:

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

filters = MetadataFilters(filters=[
    MetadataFilter(key="doc_type", value="policy", operator=FilterOperator.EQ),
    MetadataFilter(key="year", value=2025, operator=FilterOperator.GTE),
])
retriever = index.as_retriever(similarity_top_k=10, filters=filters)
```

As in LangChain, the BM25 channel does not honour these filters. If the filter
is a tenant boundary, enforce it at the store level or outside the fusion — do
not assume the abstraction covers every channel.

## What LlamaIndex does that is genuinely distinctive

**Auto-merging retrieval.** Index the same content at several chunk sizes
(2048 / 512 / 128) as a parent-child hierarchy. Retrieve on the small, precise
leaves; if several leaves from the same parent are retrieved,
`AutoMergingRetriever` replaces them with the parent chunk. You get the
precision of small chunks with the context of large ones — a direct answer to
Level 1 lesson 3's chunking tradeoff, and the strongest argument for the `Node`
relationship model.

```python
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever

nodes = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
).get_nodes_from_documents(documents)
```

**Sentence-window retrieval.** Embed single sentences for maximum retrieval
precision, but pass the surrounding ±3 sentences to the LLM via
`MetadataReplacementPostProcessor`. Same tradeoff, different resolution.

**Recursive / multi-document retrieval.** Index summaries of whole documents,
retrieve to the right document first, then retrieve within it. This is the
router pattern lesson 6 sketched for tables, generalized.

## Framework comparison

| | Plain Python | LangChain | LlamaIndex |
|---|---|---|---|
| Lines for basic RAG | ~80 | ~30 | ~3 |
| Learning curve | Concepts only | Moderate | Low to start, steep for internals |
| Chunking strategies | Whatever you write | Recursive, token-aware | Hierarchical, sentence-window, semantic |
| Hybrid + expansion | Lessons 2 + 4 | `EnsembleRetriever` + `MultiQueryRetriever` | `QueryFusionRetriever` (both, one object) |
| Agents / tool use | Manual | Strong (LangGraph) | Present, less central |
| Non-RAG LLM apps | — | Strong | Weak — not the goal |
| Debuggability | Total | Layered | Layered, more magic defaults |
| Best at | Control, latency | Composition, integrations | Retrieval sophistication over documents |

Rough guidance: **LlamaIndex if your problem is "I have a lot of documents and
retrieval quality is the hard part."** **LangChain if RAG is one component of a
larger LLM application** with tools, agents, and branching control flow.
**Plain Python once the pipeline is settled** and latency or debuggability
dominates.

They are not exclusive. A common production shape is LlamaIndex for ingestion
and retrieval, exposed as a plain function that the rest of the system calls.

## Traps

- **Shipping the three-line version.** Every default is a guess about your data.
  `similarity_top_k=2` in particular is far below the 3–5 Level 1 recommends and
  nowhere near the 20–50 that lesson 3's reranking needs.
- **Global `Settings` in a service.** Convenient in a notebook, a source of
  cross-request surprises in a server.
- **Not realising fusion calls an LLM.** `QueryFusionRetriever` with
  `num_queries > 1` puts a generation call on your retrieval critical path.
- **Hidden token spend.** Some index types (summary/tree indexes) call the LLM
  at *index* time over your whole corpus. On a large corpus, building an index
  can be the most expensive thing you do all week. Check before you run it.
- **Assuming filters cover every retriever.** As in LangChain, the sparse
  channel is the gap.
- **Treating "3 lines" as a quality claim.** Terseness of setup is unrelated to
  retrieval quality. Only your golden set answers that.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| `Document` → `Node` | Nodes are chunks that know their neighbours |
| `Settings` | Global defaults; pass explicitly in services |
| `SentenceSplitter` | Sentence-aware chunking, better than fixed-size |
| `QueryFusionRetriever` | Lessons 2 + 4 in one object; costs an LLM call |
| `SentenceTransformerRerank` | Lesson 3's cross-encoder stage |
| Auto-merging | Small chunks to retrieve, large chunks to read |
| Sentence-window | Embed one sentence, return its neighbours |
| Defaults | `top_k=2`, 1024-token chunks — always override |
| Choose it when | Retrieval quality over many documents is the hard part |

## Exercise

Build the same pipeline a third time — LlamaIndex — and run your lesson 1 golden
set against all three implementations (hand-built, LangChain, LlamaIndex).

Produce one table: hit rate, MRR, end-to-end latency, LLM calls per query, and
lines of code. Then dig into what only this framework offers:

1. **Auto-merging vs flat chunks.** Build both a flat 512-token index and a
   hierarchical `[2048, 512, 128]` index with `AutoMergingRetriever`. Measure
   retrieval hit rate *and* final answer quality. Does the extra context help
   the LLM enough to justify the token cost?
2. **Sentence-window retrieval.** Compare against auto-merging on the same set.
   Which handles your "needle in a long document" questions better?
3. **Turn `QueryFusionRetriever`'s `num_queries` from 1 to 6** and plot quality
   against latency and LLM calls. Where is the knee — and is it in the same
   place as your hand-rolled multi-query result from lesson 4?

Write a short recommendation for your own corpus: which of the three
implementations would you ship, and what specific number in your table justifies
it? Lesson 10 builds the assistant you will defend with that answer.

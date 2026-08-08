# 08 · RAG with LangChain

You have now built every RAG component by hand: chunking, embedding, a vector
store, BM25, fusion, reranking, query rewriting, and metadata filters. That was
the right order. Frameworks are much easier to evaluate when you already know
what they are wrapping — and much harder to be fooled by.

LangChain's genuine value is not that it does something you cannot. It is
composition: a large catalogue of loaders, splitters, store adapters, and
retrievers that all share an interface, so swapping Chroma for Qdrant or adding
a reranker is a one-line change rather than a rewrite.

!!! warning "None of this lesson's code was executed"
    LangChain and its integration packages were **not installed** for this run —
    the environment was deliberately kept to `rank-bm25`, `scikit-learn`, and
    `pypdf`, and the full stack (`langchain`, `langchain-community`,
    `sentence-transformers`, `torch`) does not fit the disk budget. The code
    below is written against the current package layout and is correct in shape,
    but unlike lessons 1–7 there are **no verified output blocks**. Treat it as
    a map of the API surface, not as tested code. LangChain's APIs also move
    faster than most — check the version's docs before copying.

## The mapping

Every LangChain concept has a direct counterpart in code you already wrote:

| Your Level 1 / 2 code | LangChain abstraction | What it adds |
|---|---|---|
| `open(path).read()` + cleanup | `DocumentLoader` | 100+ format loaders (lesson 7) |
| Your `chunk()` function | `TextSplitter` | Recursive, token-aware, language-aware |
| `dict` of text + metadata | `Document` | `page_content` + `metadata`, used everywhere |
| `model.encode(...)` | `Embeddings` | One interface over OpenAI/HF/local |
| Chroma calls | `VectorStore` | Same API across ~50 backends |
| `retrieve(q, k)` | `Retriever` | Composable; the core interface |
| Your `rrf()` | `EnsembleRetriever` | Weighted RRF over N retrievers |
| Your `pair_score()` | `ContextualCompressionRetriever` | Wraps a cross-encoder reranker |
| Your multi-query fusion | `MultiQueryRetriever` | LLM generates variants, fuses |
| f-string prompt assembly | `ChatPromptTemplate` + LCEL | Piping, streaming, batching |

Read that table as the actual curriculum of this lesson. If a row does not make
sense, the fix is to revisit the lesson, not the docs.

## Ingestion and indexing

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = DirectoryLoader("./docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()          # -> list[Document], metadata includes source + page

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],   # try paragraph, then line, then word
)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
```

`RecursiveCharacterTextSplitter` is worth understanding rather than accepting.
It tries the separators **in order**, splitting on paragraph breaks first and
falling back to progressively finer boundaries only when a chunk is still too
large. That is a better default than the fixed-size splitter from Level 1 lesson
3, and it is the piece most worth stealing even if you skip the rest of the
framework.

Note that `split_documents` propagates `metadata` from each source document to
every chunk automatically — including `source` and `page` from the PDF loader.
That is the provenance plumbing lesson 7 insisted on, done for free.

## Hybrid retrieval, reranking, and the chain

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# lesson 2: two channels, fused
dense = store.as_retriever(search_kwargs={"k": 20})
sparse = BM25Retriever.from_documents(chunks)
sparse.k = 20

hybrid = EnsembleRetriever(retrievers=[sparse, dense], weights=[0.4, 0.6])

# lesson 3: retrieve wide, rerank down
reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=4,
)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=hybrid
)
```

That is lessons 2 and 3 in twelve lines, and it is a fair demonstration of what
the framework buys you. It is also where you should be most alert:
`EnsembleRetriever` uses RRF under the hood, so those `weights` scale each
retriever's *reciprocal rank contribution* — they are not the `alpha` of a
normalized-score blend from lesson 2. If you did not know RRF, you would
mis-tune this and never know why it felt insensitive.

Metadata filtering (lesson 5) rides along on the retriever:

```python
dense = store.as_retriever(search_kwargs={
    "k": 20,
    "filter": {"doc_type": "policy"},     # Chroma pre-filters
})
```

`BM25Retriever`, by contrast, is an in-memory implementation with **no filter
support** — so a filter applied only to the dense channel silently lets
unfiltered sparse results through the ensemble. If your filter is a security
boundary, that is a breach, not a bug. Enforce tenant filters *outside* the
ensemble, or use a sparse backend that supports them.

### Generation with LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic

PROMPT = ChatPromptTemplate.from_template("""Answer using only the context below.
If the context does not contain the answer, say exactly:
"I don't know based on the available documents."

Context:
{context}

Question: {question}""")

def format_docs(docs):
    return "\n\n".join(f"[{i}] ({d.metadata.get('source','?')}) {d.page_content}"
                       for i, d in enumerate(docs, 1))

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | ChatAnthropic(model="claude-sonnet-5", max_tokens=1000)
    | StrOutputParser()
)

# answer = chain.invoke("what is the refund policy for annual plans?")
```

The `|` operator (LCEL) composes runnables, and every composed chain gets
`.invoke()`, `.batch()`, `.stream()`, and async variants for free. Streaming and
batching are real wins — implementing them by hand across a pipeline is tedious.

## Framework or plain Python?

| Situation | Recommendation |
|---|---|
| Prototyping, unsure of the stack | LangChain — swap components cheaply |
| Many input formats | LangChain — the loaders alone justify it |
| One well-understood pipeline in production | Plain Python — fewer layers to debug |
| Latency-critical path | Plain Python — you control every call |
| Team already fluent in it | LangChain — consistency beats purity |
| Unusual retrieval logic | Plain Python — or you fight the abstraction |
| Need tracing/observability | LangChain + LangSmith is genuinely strong |

The honest summary: LangChain is excellent for **exploration** and for corpora
with messy, varied inputs. Its cost is indirection — a bug in your retrieval
becomes a bug somewhere inside three layers of `Runnable`, and stack traces get
long. Many teams prototype in LangChain and then rewrite the ~200 lines that
survive.

## Traps

- **Accepting `chunk_size=1000` without measuring.** The default is a guess about
  your corpus that the library cannot possibly make. Level 1 lesson 3 and your
  golden set decide this, not the docs.
- **Assuming abstractions are equivalent.** `EnsembleRetriever` weights are RRF
  weights. `BM25Retriever` ignores filters. `as_retriever()` defaults to `k=4`.
  Each is defensible; each will surprise you.
- **Silent version drift.** LangChain has repeatedly moved classes between
  packages (`langchain` → `langchain_community` → provider packages). Pin
  versions, and be skeptical of any tutorial without one.
- **Debugging through layers.** When retrieval looks wrong, call
  `retriever.invoke(query)` **directly** and print the documents before touching
  the chain. The Level 1 habit — print the retrieved chunks first — matters more
  here, not less.
- **Hidden cost multiplication.** `MultiQueryRetriever` makes an LLM call per
  query and N retrieval calls; add a cross-encoder and a generation call and one
  user question is 3+ model calls. Trace it before you ship it.
- **Losing the abstain path.** Compression and reranking can return an empty
  list. Make sure `format_docs([])` produces context that triggers your refusal
  string rather than an empty prompt the LLM fills with invention.

## Cheat sheet

| Concept | Takeaway |
|---|---|
| `Document` | `page_content` + `metadata`; the universal currency |
| `RecursiveCharacterTextSplitter` | Ordered separators; better default than fixed-size |
| `Retriever` | The one interface that makes composition work |
| `EnsembleRetriever` | Lesson 2's RRF; weights are on reciprocal ranks |
| `ContextualCompressionRetriever` | Lesson 3's rerank stage |
| `MultiQueryRetriever` | Lesson 4's expansion; costs an LLM call |
| LCEL `\|` | Free streaming, batching, async |
| Real value | Loaders + swappability + tracing |
| Real cost | Indirection, version churn, hidden call counts |

## Exercise

Port your Level 1 capstone to LangChain, then interrogate the port.

Rebuild ingestion with `DirectoryLoader` + `RecursiveCharacterTextSplitter`, and
retrieval as `EnsembleRetriever` (BM25 + dense) wrapped in
`ContextualCompressionRetriever`. Run your golden question set from lesson 1
against both implementations.

Then answer with evidence, not impressions:

1. **Do the metrics match?** If hit rate differs from your hand-built pipeline,
   find out why — chunk boundaries, default `k`, or fusion weights. Do not
   accept "the framework does it differently."
2. **Count the calls.** Instrument every LLM and embedding call for one user
   question, in both implementations. Compare totals and end-to-end latency.
3. **Test the filter gap.** Apply a metadata filter to the ensemble and check
   whether unfiltered documents leak through the BM25 channel. Write the
   assertion that would have caught it, and put it in your test suite.
4. **Count the lines.** Which implementation is shorter? Which would you rather
   debug at 3am? Write down your answer — you will revisit it after lesson 9.

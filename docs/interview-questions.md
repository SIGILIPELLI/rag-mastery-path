---
title: "RAG Pipelines Interview Questions and Answers (179 Questions)"
description: "179 RAG Pipelines interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the RAG Pipelines Mastery Path."
---

# RAG Pipelines Interview Questions and Answers

**179 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free RAG Pipelines Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Entry

### [01 · What Is RAG &amp; Why It Exists](level-1/01-what-is-rag.md)

**Q1.** What is the core idea of RAG?

**Answer:** Look up relevant text first and hand it to the model with the question, shrinking its job from 'know everything' to 'read this and answer'

LLMs are extremely good at reading provided text and answering.

**Q2.** Which two problems does RAG solve?

**Answer:** Knowledge cutoffs/private data, and hallucination

A plain model can only refuse or guess about private or recent data.

**Q3.** Why not paste the entire knowledge base into every prompt?

**Answer:** It doesn't fit, it costs too much on every request, and accuracy degrades with facts buried in the middle

A 1M-token window is roughly 3–4 MB; real collections are gigabytes.

**Q4.** How does grounding change the hallucination failure profile?

**Answer:** The model can quote and cite real passages and be told to say 'I don't know' when the text lacks the answer

It doesn't make hallucination impossible.

### [02 · Embeddings &amp; Semantic Similarity](level-1/02-embeddings.md)

**Q5.** What property makes embeddings useful for retrieval?

**Answer:** Texts that mean similar things get close vectors, even with no words in common

'How do I get my money back?' lands near 'refund policy'.

**Q6.** What does a cosine similarity near 0 mean?

**Answer:** The texts are unrelated

1.0 is the same direction; −1.0 is opposite (rare).

**Q7.** What is the vector shape of all-MiniLM-L6-v2 embeddings?

**Answer:** 384 dimensions

Each text becomes a point in a 384-dimensional space.

**Q8.** Why does semantic search beat keyword search for 'money back' vs 'refund'?

**Answer:** Keyword search can't match words with no overlap, while embeddings capture meaning

The individual numbers are meaningless; relative position matters.

### [03 · Chunking Strategies](level-1/03-chunking.md)

**Q9.** Why does chunking matter so much?

**Answer:** Retrieval returns chunks, the LLM reads chunks and evaluation scores chunks, so wrong chunks can't be fixed downstream

It's the least glamorous and most consequential decision.

**Q10.** What two forces pull chunk size in opposite directions?

**Answer:** Embedding models truncate long input, while the LLM needs enough context in each chunk

Chunks must be small enough for one coherent topic and large enough to be useful evidence.

**Q11.** Why use overlap in fixed-size chunking?

**Answer:** A fixed slicer will cut ideas in half; overlap ensures at least one chunk holds the whole thought

Typical overlap is 10–20%, at the cost of near-duplicate chunks.

**Q12.** For 2000 'A' characters with chunk_size 800 and overlap 150, what are the chunk lengths?

**Answer:** 800, 800, 700, 50

The start advances by chunk_size − overlap = 650: chunks begin at 0, 650, 1300 and 1950, so the lengths are 800, 800, 700 and 50. (The lesson's printed output shows 800 for the third chunk, which is a typo.)

### [04 · Vector Stores (ChromaDB)](level-1/04-vector-stores.md)

**Q13.** What are a vector database's four jobs?

**Answer:** Store, index, query and filter

Records are (id, vector, text, metadata).

**Q14.** Why index vectors with ANN structures such as HNSW?

**Answer:** To find nearest neighbors without comparing against every vector, with near-perfect results in a fraction of the time

It's approximate nearest-neighbor search.

**Q15.** What does metadata filtering add?

**Answer:** Restricting search to records whose metadata matches, such as source == 'policy.md', which pure vector math can't do

Filtering complements similarity.

**Q16.** What is the same-model rule for a collection?

**Answer:** One collection per corpus per embedding model; queries must use the same model that embedded the documents

Mixing models breaks similarity.

### [05 · Retrieval](level-1/05-retrieval.md)

**Q17.** What does the shared retrieve(question, k, where) function return?

**Answer:** The top-k nearest chunks, optionally restricted by metadata

This exact shape appears in every RAG system.

**Q18.** What does a similarity threshold (MAX_DISTANCE) do?

**Answer:** Drops weak matches so the system can say it doesn't know instead of using irrelevant chunks

It's a retrieval confidence threshold.

**Q19.** Why does retrieval matter more than anything after it?

**Answer:** Generation can only be as good as the chunks it receives

Everything after retrieval depends on what it returns.

**Q20.** What does metadata filtering let you do in a knowledge base?

**Answer:** Search only a source or topic, e.g. topic == 'refunds'

The corpus tags each chunk with source and topic.

### [06 · Prompt Assembly &amp; Generation](level-1/06-prompt-assembly-generation.md)

**Q21.** Why tag each context chunk with a number and source?

**Answer:** The tags make citations like [1] possible

build_context formats '[i] (source: ...)'.

**Q22.** Why does the prompt say to use ONLY the provided context?

**Answer:** It's the grounding instruction separating RAG from the model freestyling; otherwise it blends retrieved facts with guesses

Without it you can't tell facts from guesses.

**Q23.** What should the model do if the context lacks the answer?

**Answer:** Say exactly 'I don't know based on the available documents.' and not guess

An honest don't-know path is part of the design.

**Q24.** Which piece is the only paid/networked part of the pipeline?

**Answer:** The generation call to the Anthropic API

The pattern is the same for any chat-completions API.

### [07 · A Minimal End-to-End Pipeline](level-1/07-minimal-pipeline.md)

**Q25.** What stages does the minimal rag.py pipeline contain?

**Answer:** Ingest → chunk → embed → store → retrieve → generate

It's about 100 lines with no framework.

**Q26.** What does the persistent ChromaDB path give you?

**Answer:** The index survives between runs, so you ingest once and ask many times

The script has 'ingest' and 'ask' commands.

**Q27.** How does chunk_text pack content?

**Answer:** Paragraphs (falling back to sentences) into chunks of at most max_chars

It avoids cutting mid-idea.

**Q28.** What is the pipeline the skeleton for?

**Answer:** The capstone project in lesson 10

The capstone grows from it.

### [08 · Evaluating RAG](level-1/08-evaluating-rag.md)

**Q29.** Why is RAG evaluation hard?

**Answer:** Two coupled stages (retrieval and generation), no single right answer, and small samples mislead

Measure the stages separately.

**Q30.** What is in a golden question set?

**Answer:** Questions with expected answers and known relevant chunk ids, including deliberate unanswerable questions

A pipeline that never says 'I don't know' fails the unanswerable ones.

**Q31.** Why include unanswerable questions on purpose?

**Answer:** To catch pipelines that never admit they don't know

Those expected answers are 'I don't know based on the available documents.'

**Q32.** How many questions should you start with?

**Answer:** 20+, growing over time as users hit failures

With 5 questions one flake swings your score by 20 points.

### [09 · Common Failure Modes](level-1/09-failure-modes.md)

**Q33.** How do you confirm a bad-chunking failure?

**Answer:** Print the retrieved chunks; if boundaries sever sentences, tables or Q→A pairs, that's it

Fix with structure-aware chunking, more overlap and one topic per chunk.

**Q34.** Why do embeddings miss exact identifiers like ERR_4012 or part numbers?

**Answer:** Embeddings treat exact tokens as noise; add keyword/BM25 search alongside dense retrieval (hybrid)

Hybrid search exists because embeddings whiff on exact tokens.

**Q35.** A user says 'can't log in' but docs say 'authentication failure'. What kind of retrieval miss is this?

**Answer:** A vocabulary mismatch the embedding model can't bridge

Check whether the right chunk is at rank 7 (ranking) or rank 400 (vocabulary).

**Q36.** What does context overflow look like?

**Answer:** API errors about maximum context length, or costs and latency creeping up as k is raised

Top-k chunks, scaffolding and history exceed the window or waste money.

### [10 · Project — Docs Q&amp;A Bot](level-1/10-capstone-docs-qa-bot.md)

**Q37.** What does the capstone Docs Q&amp;A bot do?

**Answer:** Ingests docs, answers in a CLI loop with cited sources, refuses ungrounded questions, and ships a golden-set eval

It's a tiny multi-file project.

**Q38.** Why put every tunable in config.py?

**Answer:** Evaluation tells you how to set them, and they live in one place

For example CHUNK_MAX_CHARS, TOP_K and MAX_DISTANCE.

**Q39.** What does MAX_DISTANCE = 0.8 do in the bot?

**Answer:** Drops hits above that cosine distance so weak matches aren't used

It supports the honest don't-know path.

**Q40.** What does evaluate.py measure?

**Answer:** Hit rate, MRR and an LLM-judge score on the golden set

It lets you keep proving it works as you change things.

## Level 2 · Intermediate

### [01 · Keyword Search &amp; BM25](level-2/01-bm25-keyword-search.md)

**Q41.** What blind spot do embeddings have that BM25 covers?

**Answer:** Exact tokens such as error codes, order ids and rare product names

A vector store returns something semantically adjacent but wrong.

**Q42.** Which three ideas does BM25 formalize?

**Answer:** Term frequency (diminishing returns), inverse document frequency (rare terms matter more) and length normalization

A term in a 10-word doc is stronger evidence than in a 1000-word doc.

**Q43.** Why does matching 'E4022' tell you more than matching 'the'?

**Answer:** Rare words carry the signal (high IDF)

Common words appear everywhere.

**Q44.** Why is sparse retrieval cheap?

**Answer:** It needs no embedding model, API key or GPU

The lesson ran everything with rank-bm25 and the standard library.

### [02 · Hybrid Search (Dense + Sparse)](level-2/02-hybrid-search.md)

**Q45.** Why run both dense and sparse retrieval?

**Answer:** They have opposite blind spots: BM25 misses 'undo a bad release' while embeddings find the rollback doc, and embeddings miss exact tokens

Hybrid search merges their results.

**Q46.** Why can't you just add BM25 and dense scores?

**Answer:** They live on incompatible scales (BM25 up to ~7.5, cosine up to ~0.63)

Fusion needs a scale-free method such as rank-based merging.

**Q47.** Why does the key design point make fusion a ten-line function?

**Answer:** Both retrievers expose the same interface: a query in, an ordered list of document indices out

No refactor needed.

**Q48.** What was the 'dense' channel in that lesson's examples, honestly?

**Answer:** A character n-gram TF-IDF stand-in, still lexical, so it doesn't solve vocabulary mismatch

The lesson states what ran and what didn't.

### [03 · Reranking with Cross-Encoders](level-2/03-reranking.md)

**Q49.** What is the two-stage pipeline in reranking?

**Answer:** Cast a wide, cheap net (e.g. 50 candidates), then re-score only the survivors with a cross-encoder

It routinely moves the right document from rank 7 to rank 1 without touching the index.

**Q50.** How does a cross-encoder differ from a bi-encoder?

**Answer:** It feeds the query and document through the model together so every query token attends to every document token

A bi-encoder embeds the query and documents separately.

**Q51.** Why can't you cross-encode a 100,000-document corpus at query time?

**Answer:** It needs a full forward pass per (query, document) pair, which is impossible at that scale

So retrieve about 50 first, then rerank those.

**Q52.** What did the lesson say honestly about its reranker?

**Answer:** It used a pair-scoring stand-in; pipeline shape and rank movement are real but quality and cost numbers aren't representative

Real cross-encoders need torch and sentence-transformers.

### [04 · Query Rewriting &amp; Expansion](level-2/04-query-rewriting.md)

**Q53.** Why is the query often the weakest link?

**Answer:** Users type short, typo'd or context-dependent queries while documents are careful formal prose, a translation problem

Rewriting turns a bad query into one or more good ones.

**Q54.** What is multi-query retrieval?

**Answer:** Generate several phrasings and fuse their rankings (e.g. with RRF)

The variants were hard-coded in the lesson; production uses an LLM.

**Q55.** Why does the rank function filter s[i] &gt; 0?

**Answer:** A 0.0 score is not a hit; otherwise it returns corpus-order documents that fusion can't distinguish from real hits

All-zero rankings poison the fusion.

**Q56.** What does an empty BM25 result for 'undo a bad release' show?

**Answer:** The query shares no vocabulary with the rollback runbook

The answer exists; the words don't overlap.

### [05 · Metadata Design &amp; Filtered Retrieval](level-2/05-metadata-filtering.md)

**Q57.** Why does semantic similarity not replace metadata filters?

**Answer:** It can't express hard constraints like 'only documents this user may see' or 'only the current policy version'

These are correctness and security boundaries.

**Q58.** Why capture metadata at ingestion time?

**Answer:** Metadata you didn't capture doesn't exist; re-indexing to add a field is expensive and sometimes impossible

Ask what someone will want to filter on in six months.

**Q59.** Why store dates as numbers rather than strings?

**Answer:** Numbers support $gte and ordering; strings only support equality and mislead about ordering

year: 2025 supports range filters.

**Q60.** Why capture security fields like tenant_id on day one?

**Answer:** Retrofitting after launch is a migration and an incident report

In multi-tenant systems metadata is a security boundary.

### [06 · Structured &amp; Table RAG](level-2/06-structured-table-rag.md)

**Q61.** Why do tables break prose-style RAG?

**Answer:** A cell like 99 is meaningless without its column header and row label, and chunking scatters that context

Chunks that each look fine say nothing.

**Q62.** Why can't retrieval answer 'how many seats total?'

**Answer:** Those are aggregations; retrieval cannot sum, so no top-k tuning computes SUM(seats)

Use text-to-SQL or code for aggregation.

**Q63.** What is row-level verbalization?

**Answer:** Turn each row into a self-contained sentence that carries its own headers

Strategy B in the lesson.

**Q64.** What is the problem with the whole table as one chunk?

**Answer:** Every row retrieves together, the LLM may read the wrong row, and it doesn't scale to large tables

For example Enterprise's SLA pulled onto the Team plan.

### [07 · Multi-Format Ingestion (PDF/HTML/Docs)](level-2/07-multi-format-ingestion.md)

**Q65.** Why is ingestion where most RAG quality is won or lost?

**Answer:** A retriever can't outperform its input; repeated boilerplate pollutes BM25 statistics and drags embeddings together

Every chunk inheriting a header becomes a high-frequency term.

**Q66.** What does a PDF actually contain?

**Answer:** Instructions for drawing glyphs at coordinates, so reading order and headings are reconstructed heuristically

The parser may get it wrong.

**Q67.** How does the lesson detect boilerplate without hard-coding it?

**Answer:** Text that repeats across pages (plus patterns like 'Page N of M') is boilerplate

You don't need to know what it says.

**Q68.** What was not executed in this lesson?

**Answer:** The .docx section (described in prose) and OCR

python-docx wasn't installed for the run.

### [08 · RAG with LangChain](level-2/08-rag-langchain.md)

**Q69.** What is LangChain's genuine value for RAG?

**Answer:** Composition: many loaders, splitters, store adapters and retrievers sharing an interface, so swapping a store or adding a reranker is a one-line change

It wraps what you already built by hand.

**Q70.** Which LangChain abstraction corresponds to your own rrf() fusion?

**Answer:** EnsembleRetriever (weighted RRF over N retrievers)

ContextualCompressionRetriever wraps a cross-encoder reranker.

**Q71.** Why does the lesson warn that none of its code was executed?

**Answer:** The packages weren't installed, so there are no verified outputs; treat it as a map of the API surface

Check the version's docs before copying.

**Q72.** Why learn the pieces by hand before using a framework?

**Answer:** Frameworks are easier to evaluate, and harder to be fooled by, when you know what they wrap

The mapping table is the curriculum.

### [09 · RAG with LlamaIndex](level-2/09-rag-llamaindex.md)

**Q73.** How does LlamaIndex differ from LangChain?

**Answer:** It is built specifically for connecting LLMs to data, with Index and QueryEngine abstractions and opinions about retrieval

LangChain gives you a Retriever to compose.

**Q74.** Why is the three-line LlamaIndex version dangerous?

**Answer:** It silently chooses chunk size, embedding model, top_k, prompt and LLM, every knob you should tune

Use it to check data loads, not to decide whether RAG works for your problem.

**Q75.** What is a LlamaIndex Node?

**Answer:** A chunk plus relationships to its neighbours

A Document is the source file with metadata.

**Q76.** What does a QueryEngine represent?

**Answer:** The whole pipeline, callable

It maps to your own answer function.

### [10 · Project — Hybrid Search Documentation Assistant](level-2/10-project-hybrid-assistant.md)

**Q77.** What is the skill this project builds?

**Answer:** Proving which techniques earn their keep on your corpus, and deleting those that don't

In the reference run, two of four configurations bought nothing.

**Q78.** Why does config.py give every technique a feature flag?

**Answer:** You cannot ablate what you cannot switch off

Each flag is a row in the ablation table.

**Q79.** Which five techniques does the project test?

**Answer:** BM25, hybrid fusion, reranking, query expansion and metadata filtering

Each costs latency, complexity or money.

**Q80.** How should you read the reference numbers?

**Answer:** They're real numbers from real code but not a claim about production corpora with real embeddings; produce your own table

The dense channel was a TF-IDF stand-in.

## Level 3 · Advanced

### [01 · Agentic RAG](level-3/01-agentic-rag.md)

**Q81.** What makes RAG 'agentic'?

**Answer:** Retrieval becomes a tool the model can call zero or more times in a loop it steers, deciding whether evidence is enough or to rephrase

The LLM controls retrieval decisions instead of your pipeline code.

**Q82.** Why does the loop have a Budget capping tool calls?

**Answer:** To stop runaway loops and cost; exceeding it raises an error

It's the trap the lesson warns about.

**Q83.** What question shapes break single-shot RAG?

**Answer:** Ones needing judgment about retrieval itself: is the evidence enough, should I rephrase, do I need a second different search

One search can't cover the other half of the question.

**Q84.** Why does the lesson hardcode the model's subqueries?

**Answer:** So the control flow is inspectable; a real agent generates them

A real system hands the LLM a search tool schema.

### [02 · Multi-Hop Retrieval &amp; Query Decomposition](level-3/02-multi-hop-retrieval.md)

**Q85.** Why does single-shot retrieval fail on 'Who acquired the company where Acme's founder used to work?'

**Answer:** No chunk holds the whole answer; the facts are in separate chunks that don't share the query's words

Only the founder chunk overlaps with the query.

**Q86.** What is the decompose-then-chain approach?

**Answer:** Break the question into ordered sub-questions and feed each hop's evidence into the next query

It lets hop 2 find 'Globex' even though the sub-question doesn't name it.

**Q87.** How many hops does the example need?

**Answer:** Three: founder, previous employer, acquirer

Jane Rivera → Globex → Initech.

**Q88.** Who acquired Globex in the corpus?

**Answer:** Initech, in 2015 for $400M

The chain ends at chunk 3.

### [03 · GraphRAG &amp; Knowledge Graphs](level-3/03-graphrag.md)

**Q89.** What does GraphRAG retrieve by?

**Answer:** Traversal of an explicit entity-relation graph instead of text similarity

It answers 'how are these entities connected?'

**Q90.** Why does a vector retriever fail to surface Initech for the query 'Jane Rivera'?

**Answer:** Nothing about Initech is semantically close to her name; the connection is structural

Two hops in the graph reach Initech.

**Q91.** What do community summaries add in Microsoft-style GraphRAG?

**Answer:** The graph is clustered into communities and pre-summarized, so a query can retrieve one dense summary instead of walking edges

It avoids edge walks at query time.

**Q92.** What is networkx standing in for?

**Answer:** A graph store such as Neo4j in production

The traversal logic is the same.

### [04 · Production Vector Databases](level-3/04-production-vector-dbs.md)

**Q93.** What are the two index families underneath production vector databases?

**Answer:** HNSW (multi-layer neighbor graph) and IVF (clustered buckets)

HNSW has logarithmic query time; IVF is cheaper to build/update at some recall cost.

**Q94.** How much memory do 10M float32 vectors of 768 dimensions need as a flat index?

**Answer:** About 30.7 GB

10,000,000 × 768 × 4 bytes.

**Q95.** Which databases does the lesson compare?

**Answer:** pgvector, Qdrant, Weaviate and Pinecone

Chroma is local, single-process and embedded.

**Q96.** Why was this lesson manual review rather than runnable code?

**Answer:** No local service for these databases was available in the sandbox; the memory math is real arithmetic

The lesson says so plainly.

### [05 · Embedding Model Selection &amp; Fine-Tuning](level-3/05-embedding-selection.md)

**Q97.** Why does MTEB leaderboard rank matter least for your corpus?

**Answer:** It's an average over 56 datasets in domains none of which is yours, so a top model can underperform on your tickets, contracts or code

Benchmark on your own data.

**Q98.** How do you benchmark an embedding model on your data?

**Answer:** Make an eval set of (query, answering doc) pairs and measure top-1 (or recall@k) accuracy

The lesson's TF-IDF baseline scored 1/3.

**Q99.** Why did TF-IDF score only 0.33 on the eval set?

**Answer:** It's lexical: 'refund' and 'money-back' don't overlap, and shared words like 'api' mislead

It's bad in a very informative way for this lesson.

**Q100.** What does the lesson weigh besides accuracy?

**Answer:** The dimension/cost trade-off and when fine-tuning beats picking another off-the-shelf model

Higher dimensions cost more storage and compute.

### [06 · Evaluation at Scale](level-3/06-eval-at-scale.md)

**Q101.** What does context precision measure?

**Answer:** Of what you retrieved, how much was relevant

Recall is the reverse.

**Q102.** retrieved = [0, 3, 4] and relevant = {0, 3}. What are precision and recall?

**Answer:** About 0.667 and 1.0

Two of three retrieved are relevant; both relevant docs were found.

**Q103.** Why track precision alongside recall?

**Answer:** A pipeline can hit 100% recall by over-retrieving

Full recall with 67% precision is a normal pair of scores.

**Q104.** What does faithfulness measure?

**Answer:** Of the answer's claims, how many are supported by the context

It catches hallucination.

### [07 · Streaming, Caching &amp; Latency](level-3/07-streaming-caching-latency.md)

**Q105.** Which step dominates a synchronous RAG request's latency?

**Answer:** LLM generation (1–5+ seconds)

Steps 1–3 are usually 50–300ms combined.

**Q106.** Which lever targets which step?

**Answer:** Streaming targets generation; caching targets embedding/search/rerank; parallelism targets sharded vector search

Each attacks a different part of the budget.

**Q107.** Why does a production cache key on normalized or similarity matches?

**Answer:** Near-duplicate queries like 'what's your refund policy' vs 'what is the refund policy' would miss an exact-hash cache

The lesson's exact-hash cache missed them.

**Q108.** What did caching achieve in the measured example?

**Answer:** 5 calls dropped from 57.7 ms to 10.2 ms, more than 5x faster

One miss and four hits.

### [08 · Incremental Indexing &amp; Freshness](level-3/08-incremental-indexing.md)

**Q109.** What does content hashing tell you?

**Answer:** Which documents actually changed, so unchanged ones skip re-embedding, without trusting source timestamps

Timestamps are frequently wrong or absent.

**Q110.** What does upsert return for identical content re-synced?

**Answer:** 'unchanged', skipping the embedding call

Different content returns 'reindexed' with version + 1.

**Q111.** What is 'the part teams forget' in incremental indexing?

**Answer:** Deletes: otherwise a deleted refund policy keeps being retrieved forever

Reconcile against the authoritative source set.

**Q112.** Why isn't re-embedding the whole corpus on every change acceptable?

**Answer:** A one-line edit shouldn't cost a million embedding calls

It doesn't scale.

### [09 · Observability &amp; Tracing](level-3/09-observability.md)

**Q113.** Why record spans per stage rather than one log line per request?

**Answer:** Structured per-stage records answer 'what happened' without guessing which of five stages failed

You can query fields like retrieve.top_k scores.

**Q114.** What does a Span hold?

**Answer:** A name, start and end time and arbitrary attributes

It's the unit used by OpenTelemetry, LangSmith and Phoenix.

**Q115.** In the example, which stage dominated latency?

**Answer:** generate (about 34.5 of 69.9 ms)

Tracing shows which stage dominates and what the retrieval scores were.

**Q116.** What does the module add besides tracing?

**Answer:** A drift-detection check

It also helps debug reports that 'the answer was wrong'.

### [10 · Project — Agentic Research Assistant](level-3/10-project-agentic-assistant.md)

**Q117.** Which four modules does the agentic research assistant combine?

**Answer:** Agentic tool-calling with a budget, multi-hop decomposition, a query cache and a trace log per stage

It answers multi-fact questions within a call budget.

**Q118.** Why does extract_entity take the last capitalized token?

**Answer:** A chunk mentions the already-known entity first, then the new one the next hop needs

Jane Rivera comes before Globex in the chunk.

**Q119.** What does the cache avoid?

**Answer:** Redundant work on repeat queries

cached_search returns a hit flag.

**Q120.** What does the trace leave behind?

**Answer:** An inspectable record of every decision

It supports debugging.

## Level 4 · Master

### [01 · Enterprise RAG Architecture Patterns](level-4/01-enterprise-architecture.md)

**Q121.** What is the risk of a single shared index with per-tenant metadata filters?

**Answer:** A missing or wrong filter leaks data across tenants

It is the cheapest to operate and the riskiest.

**Q122.** What does index-per-tenant guarantee?

**Answer:** Structural isolation: no code path touches two tenants' data, at the cost of N indexes to run

Small tenants pay disproportionate fixed overhead.

**Q123.** Why do enterprises grow into hub-and-spoke?

**Answer:** A central retrieval service handles auth, routing, observability and policies once, scaling the platform team's effort sub-linearly

Teams own their data and ingestion.

**Q124.** Why is much of this module deliberately manual review?

**Answer:** 'Which architecture fits your org' can't be validated by a toy script

Runnable code appears only for genuinely code-level decisions.

### [02 · Multi-Tenant RAG &amp; Access Control](level-4/02-multi-tenant-access-control.md)

**Q125.** What is the vulnerable pattern?

**Answer:** Filter after ranking: rank across all tenants, then discard non-matches

The order of two operations is the whole security model.

**Q126.** What happens in the lesson when tenant acme queries Globex's wording with a post-filter?

**Answer:** It returns [] (denies acme their own data) because the top global hit belonged to Globex

A post-filter can deny a tenant their own data or leak another's.

**Q127.** What does the safe pre-filter do?

**Answer:** Restrict candidates to the tenant's documents first, then rank only within them

It returned acme's Q3 revenue.

**Q128.** Why is this bug a data breach rather than a performance detail?

**Answer:** Under the right query the system can hand a tenant someone else's data

Isolation must be enforced before ranking.

### [03 · Cost Optimization](level-4/03-cost-optimization.md)

**Q129.** In the cost model, where does almost all the money go?

**Answer:** The LLM call: prompt input and generated output tokens (embedding cost was $0.40 of $8,310)

Teams instinctively optimize embeddings and retrieval speed, but the money is in generation.

**Q130.** For 1M queries, ~1,520 input tokens and 250 output tokens each, what is the approximate total?

**Answer:** About $8,310

LLM input $4,560 plus output $3,750 plus tiny embedding cost.

**Q131.** What does a 30% cache hit rate save in the example?

**Answer:** About $2,493 (total drops to $5,817)

Cached queries skip a fresh LLM call.

**Q132.** If the goal is specifically cost, where should you look first?

**Answer:** The LLM call: context tokens and output length

Retrieval optimization still matters for latency and quality.

### [04 · Latency Optimization at Scale](level-4/04-latency-optimization.md)

**Q133.** Why did parallelizing three retrieval shards save only 78 ms?

**Answer:** Generation (1800 ms) dwarfs everything else; optimize the stage that dominates, not the easiest to parallelize

Sequential 1963 ms vs parallel 1885 ms.

**Q134.** What does a 1% straggler on one of three shards do to p99?

**Answer:** It turns p99 into about 7x p50 (46.8 ms vs 331.3 ms)

Parallel retrieval waits for the slowest shard.

**Q135.** Why is p99, not the average, the number that matters?

**Answer:** Averages hide the slow requests that generate a steady stream of complaints

Fast on average still has slow-request complaints.

**Q136.** How can you reduce perceived latency even if total time doesn't change?

**Answer:** Stream tokens

Streaming makes the first words appear quickly.

### [05 · RAG Security](level-4/05-rag-security.md)

**Q137.** What is indirect prompt injection in RAG?

**Answer:** A malicious instruction planted in a document that later gets retrieved into the LLM's context, where the model may follow it

The attack doesn't need to come from the user's message.

**Q138.** Why do RAG systems have a larger attack surface than a plain chatbot?

**Answer:** Any ingestion source (tickets, scraped pages, uploads, emails) is a potential injection vector

Ingestion pipelines are built for correctness and freshness, not adversarial content.

**Q139.** What is the security model for retrieved context?

**Answer:** Untrusted input, not just data

Treat it that way throughout.

**Q140.** What is a first line of defense shown in the lesson?

**Answer:** Pattern-based flagging of phrases like 'ignore previous instructions' or 'api key'

It is only a first line, not a complete defense.

### [06 · Governance, Compliance &amp; Auditability](level-4/06-governance-compliance.md)

**Q141.** What must enterprise RAG answer in an audit?

**Answer:** Who asked what, what data informed the answer, who could have seen it, and how long the record is kept

'It worked in testing' isn't a defense.

**Q142.** Why log query, retrieval and generation as separate entries?

**Answer:** Each is independently attributable, so an auditor can see which document informed which answer

A single combined line loses that detail.

**Q143.** What does the entry hash provide?

**Answer:** A cheap way to detect that an entry was altered after the fact (not a secure chain by itself)

The lesson is explicit about its limits.

**Q144.** What is data lineage?

**Answer:** Mapping an answer back to the source documents and their metadata that informed it

For example source, classification and ingestion date.

### [07 · Continuous Evaluation &amp; Feedback Loops](level-4/07-continuous-evaluation.md)

**Q145.** What does continuous evaluation catch that a CI regression suite doesn't?

**Answer:** Quality drift from a shifting corpus or user population, or a provider silently updating a model, none of which show up as a code diff

It uses production-traffic evaluation and feedback loops.

**Q146.** In the example, thumbs-up rates were 0.86, 0.845 and 0.65 across three weeks. Which drop is real?

**Answer:** Week 3's drop to 0.65, an injected regression; the week 1 vs 2 gap is sampling noise

A 10-point threshold stays quiet on noise and fires on the real drop.

**Q147.** What happens if the regression threshold is too tight or too loose?

**Answer:** Too tight false-alarms on normal variance; too loose lets real regressions slip through for weeks

The right threshold depends on feedback volume.

**Q148.** How does higher traffic volume help?

**Answer:** It lets you tighten the threshold and catch smaller regressions faster

This is a statistics problem as much as an engineering one.

### [08 · Fine-Tuning vs. RAG vs. Hybrid](level-4/08-finetuning-vs-rag.md)

**Q149.** What does RAG change versus fine-tuning?

**Answer:** RAG changes what the model knows right now by handing it facts at query time; fine-tuning changes what it does by default (tone, format, behavior)

RAG never changes weights.

**Q150.** What happens if you fine-tune a model to know current pricing?

**Answer:** It's wrong the moment pricing changes, since facts are baked into weights needing a whole new training run

Use RAG for facts that change.

**Q151.** What happens if you use RAG to make a model sound like your brand voice?

**Answer:** It doesn't, since retrieved documents give facts, not a persistent behavioral pattern

That's a job for fine-tuning.

**Q152.** In the illustration, how much more does updating one fact cost via fine-tuning than via RAG?

**Answer:** About 12.5 million times, and hours instead of seconds

RAG: one embedding call for one chunk; fine-tune: a $50, 3-hour run.

### [09 · Scaling Ingestion Pipelines](level-4/09-scaling-ingestion.md)

**Q153.** Which two levers dominate scaling ingestion?

**Answer:** Parallelism and batching

They interact in ways worth measuring.

**Q154.** Why was 10-way parallelism only 3.9x faster?

**Answer:** Thread-scheduling overhead, and past a point you're bound by the embedding provider's rate limit, risking 429s

I/O-bound calls overlap, but not perfectly.

**Q155.** Why does batching help?

**Answer:** It amortizes the fixed per-request overhead that single-doc calls pay every time

Total time falls steeply as batch size grows.

**Q156.** What is the risk of an ingestion pipeline that's too slow?

**Answer:** The index stays stale for hours

At millions of documents it can become the bottleneck.

### [10 · Capstone — Enterprise Knowledge Platform](level-4/10-capstone-enterprise-platform.md)

**Q157.** Which four pieces does the capstone combine?

**Answer:** Tenant-scoped retrieval, a tenant-scoped cache, injection detection on retrieved content and an audit log

It enforces isolation with a real assertion.

**Q158.** How does tenant_scoped_search stay safe?

**Answer:** It filters to the tenant's documents before ranking

That's module 2's safe pattern.

**Q159.** Why is the cache tenant-scoped?

**Answer:** A shared cache could return one tenant's cached answer to another

The cache key includes the tenant.

**Q160.** What does the audit log capture?

**Answer:** Every query, retrieval and security event

It supports compliance review.

## Mixed questions across all levels

**Q161.** What does this print?

```
def chunk_fixed(text, size=800, overlap=150):
    out=[]; start=0
    while start < len(text):
        out.append(text[start:start+size]); start += size-overlap
    return out
print([len(c) for c in chunk_fixed('A'*2000)])
```

**Answer:** [800, 800, 700, 50]

Starts advance by 650: chunks begin at 0, 650, 1300 and 1950, with lengths 800, 800, 700 and 50.

**Q162.** What does this print?

```
def precision(r, rel): return sum(x in rel for x in r)/len(r)
def recall(r, rel): return sum(x in r for x in rel)/len(rel)
r=[0,3,4]; rel={0,3}
print(round(precision(r,rel),3), recall(r,rel))
```

**Answer:** 0.667 1.0

Two of three retrieved are relevant; both relevant docs were retrieved.

**Q163.** What does this print?

```
def rrf(rankings,k=60):
    s={}
    for r in rankings:
        for i,d in enumerate(r):
            s[d]=s.get(d,0)+1/(k+i+1)
    return sorted(s,key=lambda d:-s[d])
print(rrf([['a','b','c'],['b','a','d']]))
```

**Answer:** ['a', 'b', 'c', 'd']

a and b tie on the fused score; sorted() is stable so a stays first, then c beats d.

**Q164.** What does this print?

```
import hashlib
h=lambda t: hashlib.sha256(t.encode()).hexdigest()
print(h('Refunds within 30 days.')==h('Refunds within 30 days.'), h('30 days')==h('45 days'))
```

**Answer:** True False

Content hashing detects whether a document changed, so unchanged ones skip re-embedding.

**Q165.** What does this print?

```
docs=[('acme','a rev $2M'),('globex','g rev $5M')]
q={'g','rev'}
post=[d for d in sorted(docs,key=lambda d:-len(q&set(d[1].split())))[:1] if d[0]=='acme']
pre=sorted([d for d in docs if d[0]=='acme'],key=lambda d:-len(q&set(d[1].split())))[:1]
print(post, pre)
```

**Answer:** [] [('acme', 'a rev $2M')]

Filtering after ranking loses the tenant's own data when another tenant ranks first; filter before ranking.

**Q166.** Users type 'card got rejected' but docs say 'payment declined'. What helps most?

**Answer:** Query rewriting/expansion plus dense (embedding) retrieval

It's a vocabulary mismatch.

**Q167.** Queries contain error codes like E4022 that embeddings mangle. What do you add?

**Answer:** BM25 keyword search fused with dense retrieval (hybrid)

Embeddings are poor at exact tokens.

**Q168.** The right chunk is at rank 7 but the LLM sees only the top 3. What is the highest-leverage fix?

**Answer:** Retrieve ~50 then rerank with a cross-encoder

Reranking can move the right document from rank 7 to rank 1.

**Q169.** A question needs facts chained across three documents. What pattern helps?

**Answer:** Query decomposition with multi-hop retrieval

Each hop's evidence feeds the next query.

**Q170.** You must answer 'how many seats total across plans?' from a table. What approach fits?

**Answer:** Text-to-SQL or code over structured data; retrieval can't aggregate

Aggregations need computation, not retrieval.

**Q171.** You want to know if a technique earns its keep. What do you do?

**Answer:** Run an ablation on a golden set with a flag per technique

Delete techniques that don't help your corpus.

**Q172.** Your RAG answers are fluent but cite nothing relevant. Which metric flags that?

**Answer:** Faithfulness (are claims supported by the context) plus context precision

Faithfulness catches unsupported claims.

**Q173.** A recall of 100% with low precision tells you what?

**Answer:** You over-retrieve; noise is entering the context

Track precision alongside recall.

**Q174.** A policy doc is updated nightly. How do you keep the index fresh cheaply?

**Answer:** Content hashing with upserts and deletes, re-embedding only changed documents

Don't forget deletes, or removed policies keep being retrieved.

**Q175.** A customer-uploaded PDF contains 'IGNORE ALL PREVIOUS INSTRUCTIONS'. What is this?

**Answer:** Indirect prompt injection; treat retrieved context as untrusted input

Any ingestion source is an injection vector.

**Q176.** A multi-tenant system filters results after ranking. What can go wrong?

**Answer:** It can deny a tenant their own data or leak another's; filter before ranking

The order of operations is the security model.

**Q177.** Your facts change weekly but you want the model to answer in a fixed brand voice. What do you combine?

**Answer:** RAG for facts plus fine-tuning (or prompting) for voice

They're complementary, not competitors.

**Q178.** At a million queries a month, where does most of the cost go?

**Answer:** LLM input/output tokens, not embeddings

Reduce context size, output length or use caching.

**Q179.** Your mean latency is fine but users complain. What should you check?

**Answer:** p99 latency, since stragglers dominate the tail

A 1% straggler can make p99 7x p50.

## Practical coding questions

### Chunk with overlap

Write `chunk_fixed(text, size, overlap)` that slices text into `size`-character chunks, advancing by `size - overlap`, and print the chunk lengths for 'A' * 25 with size 10 and overlap 3.

Starter code:

```
def chunk_fixed(text, size, overlap):
    # your code here
    pass

print([len(c) for c in chunk_fixed("A" * 25, 10, 3)])
```

Expected output:

```text
[10, 10, 10, 4]
```

??? success "Show a solution"

    ```
    def chunk_fixed(text, size, overlap):
        out, start = [], 0
        while start < len(text):
            out.append(text[start:start + size])
            start += size - overlap
        return out

    print([len(c) for c in chunk_fixed("A" * 25, 10, 3)])
    ```

    The start advances by 7 each time: chunks start at 0, 7, 14 and 21.

### Reciprocal rank fusion

Write `rrf(rankings, k=60)` that fuses ranked lists using score 1/(k + rank) (rank starting at 1) and returns document ids best-first. Print the fusion of [['a','b','c'], ['c','b','d']].

Starter code:

```
def rrf(rankings, k=60):
    # your code here
    pass

print(rrf([["a", "b", "c"], ["c", "b", "d"]]))
```

Expected output:

```text
['c', 'b', 'a', 'd']
```

??? success "Show a solution"

    ```
    def rrf(rankings, k=60):
        scores = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
        return sorted(scores, key=lambda d: -scores[d])

    print(rrf([["a", "b", "c"], ["c", "b", "d"]]))
    ```

    c scores 1/63 + 1/61 (rank 3 in one list, rank 1 in the other), slightly more than b's 1/62 + 1/62, so c comes first, then b, then a (1/61) and d (1/63). RRF rewards documents that rank well in several lists.

---

More free practice: the [RAG Pipelines Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).

---
title: "RAG Pipelines Cheat Sheet: Quick Reference, Beginner to Advanced"
description: "A free RAG Pipelines cheat sheet: 37 quick-reference tables from every level of the RAG Pipelines Mastery Path, each linked to its full lesson."
---

# RAG Pipelines Cheat Sheet

Quick-reference tables from **37 lessons** of the free RAG Pipelines Mastery Path, in course order from beginner to advanced. Each section links to the full lesson with explanations and runnable examples.

## Level 1 · Entry

### [01 · What Is RAG &amp; Why It Exists](level-1/01-what-is-rag.md)

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

### [02 · Embeddings &amp; Semantic Similarity](level-1/02-embeddings.md)

| Concept | Takeaway |
|---------|----------|
| Embedding | Text → fixed-length vector; similar meaning → nearby vectors |
| `model.encode(text)` | Returns a numpy array (`(384,)` for MiniLM) |
| Cosine similarity | `dot(a,b)/(|a||b|)`; 1 = same meaning, 0 = unrelated |
| Cosine distance | `1 - similarity` (what ChromaDB reports) |
| Search | Embed corpus + query, sort by similarity |
| Same-model rule | Query and documents must use the same embedding model |
| Truncation | Models cut off long inputs → chunk before embedding |
| Score intuition | Compare rankings, not absolute values |

### [03 · Chunking Strategies](level-1/03-chunking.md)

| Strategy | How it splits | Use when |
|----------|---------------|----------|
| Fixed-size + overlap | Every N chars, repeat last M | No structure to exploit; baseline |
| Sentence/paragraph-aware | Pack whole units to a size budget | Default for prose |
| Structural (headings/tags) | On document structure | Markdown, HTML, code |
| Overlap | Repeat boundary content | Mitigate mid-thought cuts (10–20%) |
| Size sweet spot | ~200–500 words to start | Tune with evaluation, not vibes |

### [04 · Vector Stores (ChromaDB)](level-1/04-vector-stores.md)

| Operation | ChromaDB code |
|-----------|---------------|
| In-memory client | `chromadb.Client()` |
| Persistent client | `chromadb.PersistentClient(path="./chroma_db")` |
| Create/open collection | `client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})` |
| Add records | `collection.add(ids=, embeddings=, documents=, metadatas=)` |
| Query | `collection.query(query_embeddings=[...], n_results=k)` |
| Filter | `where={"source": "policy.md"}` |
| Update / delete | `collection.upsert(...)` / `collection.delete(ids=[...])` |
| Distance semantics | Cosine *distance*: smaller = more similar |
| Skip the DB when | Small corpus, no persistence/filtering needs → numpy |

### [05 · Retrieval](level-1/05-retrieval.md)

| Concept | Takeaway |
|---------|----------|
| Top-k search | Return the k nearest chunks to the query vector |
| Choosing k | Start 3–5; tune with eval, not intuition |
| `where=` filter | Hard constraints inside the search; also the tenant/security boundary |
| Filter operators | `$in`, `$ne`, `$gt`, `$and`, `$or` |
| No-answer queries | Still return k results — check distances |
| Distance threshold | Drop hits above ~0.8 distance (calibrate per model/corpus) |
| Empty result | A feature, not a bug — enables honest "I don't know" |
| Debug habit | Always print retrieved chunks before blaming the LLM |

### [06 · Prompt Assembly &amp; Generation](level-1/06-prompt-assembly-generation.md)

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

### [07 · A Minimal End-to-End Pipeline](level-1/07-minimal-pipeline.md)

| Stage | Function | Lesson it came from |
|-------|----------|---------------------|
| Chunk | `chunk_text()` — paragraph-aware, 1-unit overlap | 3 |
| Embed | `model.encode(texts)` — one model for docs & queries | 2 |
| Store | `PersistentClient` + cosine collection | 4 |
| Retrieve | top-k + distance threshold | 5 |
| Prompt | numbered, source-tagged context + grounding rules | 6 |
| Generate | `messages.create(...)` — provider-swappable | 6 |
| CLI | `ingest <folder>` / `ask <question>` | — |

### [08 · Evaluating RAG](level-1/08-evaluating-rag.md)

| Metric | What it measures | Cost |
|--------|------------------|------|
| Hit rate@k | Evidence present in top-k at all | Free, fast — run constantly |
| MRR | How high the first relevant chunk ranks | Free, fast |
| LLM-as-judge grade | End-to-end answer correctness | One API call per question |
| Unanswerable questions | "I don't know" discipline | Include ~20% of the set |
| Golden set | Ground truth: question + answer + evidence ids | Grows from real failures |
| Debug rule | Low hit rate → fix retrieval first | — |

### [09 · Common Failure Modes](level-1/09-failure-modes.md)

| Failure | Signature | First fix |
|---------|-----------|-----------|
| Bad chunking | Truncated/mixed-topic chunks in top-k | Structure-aware chunking + overlap |
| Retrieval miss | Right chunk absent from top-k | Hybrid search; enrich chunks; raise k |
| Context overflow | Length errors; cost creep | Token budget, dedupe, cap history |
| Lost in the middle | Evidence present, answer wrong | Fewer/better chunks; best evidence first |
| Stale index | Correctly cited outdated facts | Scheduled re-ingest; `indexed_at` metadata |
| Prompt injection | Rule-breaking on specific queries | Delimit docs as data; curate corpus; gate actions |

### [10 · Project — Docs Q&amp;A Bot](level-1/10-capstone-docs-qa-bot.md)

| File | Responsibility | Key lessons |
|------|----------------|-------------|
| `config.py` | Every tunable, one place | 8 |
| `ingest.py` | chunk → embed → store (rebuildable) | 3, 4 |
| `bot.py` | retrieve → threshold → budget → prompt → generate → cite | 5, 6, 9 |
| `evaluate.py` | hit rate + MRR + LLM-judge | 8 |
| `golden.json` | Ground truth incl. unanswerables | 8 |
| `docs/` | The corpus — treat as untrusted input | 9 |

## Level 2 · Intermediate

### [01 · Keyword Search &amp; BM25](level-2/01-bm25-keyword-search.md)

| Concept | Takeaway |
|---|---|
| TF saturation (`k1`) | Repeated terms give diminishing returns |
| IDF | Rare terms score higher; stopwords score near zero |
| Length norm (`b`) | Short docs favored; 0.75 is a sane default |
| Analyzer | Lowercase, strip stopwords, stem — not optional |
| Score 0.0 | No lexical overlap at all — drop the hit |
| Best at | Ids, codes, jargon, exact phrases |
| Blind to | Paraphrase and synonyms |
| Library | `rank-bm25` for learning; Elasticsearch/Postgres FTS in production |

### [02 · Hybrid Search (Dense + Sparse)](level-2/02-hybrid-search.md)

| Concept | Takeaway |
|---|---|
| Why hybrid | Sparse and dense fail on disjoint query types |
| Score scales | BM25 unbounded; cosine in `[-1, 1]` — never add raw |
| Min-max norm | Per-query; destroys absolute confidence |
| RRF formula | `sum(1 / (k + rank))` over all rankings |
| `k` = 60 | Paper default; larger `k` rewards consensus |
| Candidate depth | Fuse top 20–50 per channel, not top 3 |
| Fusion is not magic | It cannot add capability neither retriever has |
| Abstain | Threshold on raw scores before fusing |

### [03 · Reranking with Cross-Encoders](level-2/03-reranking.md)

| Concept | Takeaway |
|---|---|
| Bi-encoder | Encodes query and doc separately; fast, indexable |
| Cross-encoder | Encodes the pair jointly; accurate, query-time only |
| Pipeline | Retrieve 50 (cheap) → rerank → keep 3–5 |
| What it fixes | Ordering and precision, never recall |
| Cost driver | Linear in candidate count — `top_n` is the dial |
| Token limit | ~512 tokens per pair; long chunks get truncated |
| Free local models | `ms-marco-MiniLM-L-6-v2`, BGE-reranker (need torch) |
| Guardrail | Score floor, or you rank noise confidently |

### [04 · Query Rewriting &amp; Expansion](level-2/04-query-rewriting.md)

| Concept | Takeaway |
|---|---|
| Multi-query | N phrasings, retrieve each, fuse with RRF |
| Always keep the original | Protects exact-token queries from rewrite damage |
| Conversational rewrite | Resolve pronouns; mandatory for chat |
| HyDE | Retrieve with a fake answer, not the question |
| Acronym dictionary | Free, instant, often the biggest single win |
| Filter zero scores | Before fusing, or you fuse corpus order |
| Fused order ≠ correct order | Put a reranker after fusion |
| Cost | Every technique is +1 LLM call on the critical path |

### [05 · Metadata Design &amp; Filtered Retrieval](level-2/05-metadata-filtering.md)

| Concept | Takeaway |
|---|---|
| Capture at ingestion | Fields you skip now require a full re-index later |
| Dates as numbers | Enables `$gte`; strings do not order correctly |
| Pre-filter | Filter then rank — correct result counts |
| Post-filter | Rank then filter — collapses under narrow filters |
| k-collapse | Selective filter + post-filter = empty results |
| Over-fetch | The mitigation if you are stuck with post-filtering |
| `tenant_id` | Security boundary; server-side, pre-filtered, always |
| Never LLM-generated | Filters come from the session, not the model |
| Recency | Prefer a ranking boost over a hard filter |

### [06 · Structured &amp; Table RAG](level-2/06-structured-table-rag.md)

| Concept | Takeaway |
|---|---|
| Why tables break | Cells are meaningless without header + row label |
| Whole-table chunk | Causes cross-row confusion; does not scale |
| Row-level chunks | Self-contained, retrievable, the default fix |
| Verbalization | Write rows as sentences — matches how users ask |
| Table summary | A router chunk, not an answer source |
| Aggregation | Retrieval cannot sum — route to SQL |
| Text-to-SQL safety | Read-only, `SELECT`-only, `LIMIT`, timeout |
| Production pattern | Route: lookups → chunks, math → SQL |

### [07 · Multi-Format Ingestion (PDF/HTML/Docs)](level-2/07-multi-format-ingestion.md)

| Concept | Takeaway |
|---|---|
| PDFs have no paragraphs | Reading order is reconstructed, sometimes wrongly |
| Boilerplate detection | Repeats across pages — find by frequency, not hard-coding |
| Templated footers | Need regex; frequency counting misses them |
| Line unwrapping | Join breaks before lowercase, after non-punctuation |
| Multi-column PDFs | Naive extraction scrambles text silently |
| HTML | Never regex-strip tags — you index CSS and JS |
| `.docx` tables | Live in `doc.tables`, skipped by `doc.paragraphs` |
| Empty extraction | Assert a minimum length; never index silently |
| Golden rule | Read the extracted text before indexing at scale |

### [08 · RAG with LangChain](level-2/08-rag-langchain.md)

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

### [09 · RAG with LlamaIndex](level-2/09-rag-llamaindex.md)

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

## Level 3 · Advanced

### [01 · Agentic RAG](level-3/01-agentic-rag.md)

| Decision | Fixed RAG | Agentic RAG |
|---|---|---|
| When to retrieve | Always, once | Model decides, 0–N times |
| Query text | User's question verbatim | Model-generated, can differ per call |
| Stopping condition | N/A | Budget, timeout, or model's own judgment |
| Cost predictability | High | Low — variance per question |
| Failure mode | Under-retrieval | Runaway loops, latency spikes |
| Needs | Retriever | Retriever + tool schema + budget enforcement + tracing |

### [02 · Multi-Hop Retrieval &amp; Query Decomposition](level-3/02-multi-hop-retrieval.md)

| | Single-hop retrieval | Multi-hop retrieval |
|---|---|---|
| Handles | Facts in one chunk | Facts spanning chunks |
| Query count | 1 | N (one per hop) |
| Between hops | Nothing | Entity/fact carried forward |
| Failure mode | Miss the chunk | Miss a hop, or carry the wrong entity |
| Cost | Fixed | Scales with hop count |
| Needs | Retriever | Retriever + decomposer + entity bridge |

### [03 · GraphRAG &amp; Knowledge Graphs](level-3/03-graphrag.md)

| | Vector RAG | GraphRAG |
|---|---|---|
| Finds | Semantically similar text | Structurally connected entities |
| Build cost | 1 embedding call/chunk | LLM extraction + resolution + summarization |
| Update cost | Append embedding | Re-resolve entities, maybe re-summarize |
| Query cost | 1 similarity search | Traversal (cheap) or community lookup |
| Best for | "Find X" | "How does X relate to Y" |
| Failure mode | Miss paraphrases | Miss connections if extraction/resolution is bad |

### [04 · Production Vector Databases](level-3/04-production-vector-dbs.md)

| Symptom | Likely cause |
|---|---|
| Recall drops when combined with a filter | Post-filtering instead of pre-filtering |
| Query latency degrades as data grows past RAM | HNSW graph no longer fits in memory |
| Index build is slow, queries are fast | Expected HNSW trade-off; IVF is the inverse |
| Cross-tenant data appears in results | Filter applied after ANN search, or missing entirely |
| Postgres team says "just use pgvector" | Right call if scale and filter complexity are moderate |

### [05 · Embedding Model Selection &amp; Fine-Tuning](level-3/05-embedding-selection.md)

| Signal | Action |
|---|---|
| MTEB rank looks great, your eval accuracy doesn't | Trust your eval, not MTEB |
| Failures are exact-word matches only | Vocabulary/semantic gap — try a stronger embedding model |
| Failures persist across every off-the-shelf model tried | Candidate for fine-tuning, if you have labeled pairs |
| Just switched embedding models | Re-embed the entire index — old and new vectors aren't comparable |
| Eval questions read like paraphrased chunks | Rebuild eval set from real user queries |

### [06 · Evaluation at Scale](level-3/06-eval-at-scale.md)

| Metric | Answers | Gameable by |
|---|---|---|
| Context precision | Is retrieval noisy? | N/A alone — pairs with recall |
| Context recall | Did retrieval miss anything? | Retrieving everything |
| Faithfulness | Is the answer grounded in context? | Verbatim-quoting context with no synthesis |
| Answer relevance | Does the answer address the question? | Longer, vaguer answers |
| CI regression gate | Did this change make things worse? | A floor set too low to ever trip |

### [07 · Streaming, Caching &amp; Latency](level-3/07-streaming-caching-latency.md)

| Lever | Reduces | Doesn't help with |
|---|---|---|
| Caching | Repeated/duplicate work | Novel queries (always a miss) |
| Parallelism | Wall-clock time for independent I/O | Sequential dependencies (multi-hop) |
| Streaming | Perceived latency (TTFT) | Total compute or total tokens generated |
| All three together | End-to-end user-perceived latency | Correctness — none of these improve answer quality |

### [08 · Incremental Indexing &amp; Freshness](level-3/08-incremental-indexing.md)

| Symptom | Cause | Fix |
|---|---|---|
| Stale info in answers | Deletes not handled, or sync not running | Add explicit delete reconciliation |
| Full re-embed on every sync | No content hashing | Hash-compare before re-embedding |
| Occasional permanently-stale doc | CDC event dropped, no backstop | Add periodic full reconciliation |
| Metadata update produces wrong answers | Metadata baked into embedded text | Re-embed on metadata change too, or separate metadata from embedded text |
| Users trust stale time-sensitive facts | No freshness surfaced in answer | Track and show `indexed_at` age |

### [09 · Observability &amp; Tracing](level-3/09-observability.md)

| Question during an incident | What you need logged |
|---|---|
| Which stage is slow? | Per-span duration with a shared trace_id |
| Was retrieval actually relevant? | Retrieved chunk IDs + scores per request |
| Did a multi-hop query go wrong? | Sub-query text and carried evidence per hop |
| Is quality degrading over time, not just today? | Score history, trended, with an alert threshold |
| Did this happen before? | trace_id searchable against past incidents |

## Level 4 · Master

### [01 · Enterprise RAG Architecture Patterns](level-4/01-enterprise-architecture.md)

| Signal | Suggests |
|---|---|
| 2-3 internal teams, similar trust level | Shared index + filters is fine, keep filter logic centralized |
| Regulatory separation required (e.g. GDPR data residency per customer) | Index-per-tenant, or physically separate infra |
| 10+ teams, growing | Hub-and-spoke — invest in the shared retrieval service |
| "Just add it to the existing index" without a filter review | Stop — this is how implicit pattern-1 sprawl starts |
| One team's config change breaks another team's pipeline | Sign the config-driven approach was needed and isn't there yet |

### [02 · Multi-Tenant RAG &amp; Access Control](level-4/02-multi-tenant-access-control.md)

| Symptom | Likely cause |
|---|---|
| Tenant gets zero results for answerable questions | Post-filtering after top_k truncation |
| Tenant sees another tenant's data | Missing filter, or post-filtering with no truncation loss (rarer, still possible) |
| New endpoint has a leak old ones don't | Filter enforced in application code per-path, not centrally |
| Leak survives a retrieval-filter code review | Check the cache and any cross-request scratchpad state next |
| Team says "we filter after ranking, it's fine, we just top_k a bit higher" | Overfetch-then-filter — recall degrades as filter selectivity increases (Level 3 module 04) and doesn't remove the structural risk |

### [03 · Cost Optimization](level-4/03-cost-optimization.md)

| Lever | Saves | Risk if done carelessly |
|---|---|---|
| Caching | Skips full request cost on hits | Stale answers (Level 3 module 08), cross-tenant leak if unscoped (module 02) |
| Smaller context / better reranking | Input token cost, every request | Answer quality drop if chunks cut were actually needed |
| Model tiering | Output (and input) cost on routed-down traffic | Misrouted complex queries get worse answers, possibly re-run at full cost anyway |
| None of the above, just "use a cheaper model everywhere" | Uniform cost cut | Uniform quality cut — no way to recover just the queries that needed the stronger model |

### [04 · Latency Optimization at Scale](level-4/04-latency-optimization.md)

| Symptom | Likely cause | Fix |
|---|---|---|
| p50 is great, users still complain | p99 is high — check tail, not average | Hedging, timeout tuning, straggler isolation |
| Parallelizing shards didn't help much | Generation dominates total latency | Optimize/stream generation instead |
| Latency spikes correlate with load | Retry storm from client-side timeouts | Add backoff + circuit breaker |
| More shards made things slower at p99 | More chances for one straggler per request | Hedge, or reduce shard fan-out |
| Generation gets cut short under load | Fixed per-stage timeout doesn't account for upstream variance | Budget stages independently, monitor each |

### [05 · RAG Security](level-4/05-rag-security.md)

| Risk | Where it enters | Primary mitigation |
|---|---|---|
| Indirect prompt injection | Any ingested/retrieved document | Role separation, least-privilege tools, output validation |
| Tool-enabled injection escalation | Agentic pipelines with side-effect tools | Scope tool permissions to the minimum needed |
| PII/secrets in the index | Ingestion without scrubbing | Redact at ingestion time, before embedding |
| Access-control leakage via errors | Verbose "not found" / "not authorized" messages | Generic denial messages, audited error text |
| False sense of security from keyword filters | Pattern-matching treated as sufficient | Treat as a tripwire only, layer real structural defenses |

### [06 · Governance, Compliance &amp; Auditability](level-4/06-governance-compliance.md)

| Compliance question | What has to be logged/tracked to answer it |
|---|---|
| Who asked what, and when? | Actor + action + timestamp per request |
| What data informed this specific answer? | Retrieved doc IDs + doc metadata, linked by trace_id |
| Did this access comply with policy at the time? | Access-control state versioned over time, not just current |
| Can we delete one user's records on request? | Log design that supports targeted redaction |
| Are we retaining data the required duration — no more, no less? | Explicit, per-category retention policy, enforced in code |
| Is the audit log itself secure? | Access control and review scope on the log, same as the primary system |

### [07 · Continuous Evaluation &amp; Feedback Loops](level-4/07-continuous-evaluation.md)

| Signal source | Catches | Misses |
|---|---|---|
| Explicit thumbs up/down | Severe, obvious failures | Mediocre-but-not-bad answers (no signal) |
| Sampled traffic + judge/human review | Deeper metric failures (faithfulness, relevance) | Anything outside the sampled slice |
| Fixed-threshold regression alert | Real week-over-week drops | Regressions smaller than the noise floor |
| Eval-set augmentation from failures | Any specific failure recurring | Novel failure types not yet seen once |
| CI suite alone (Level 3 module 06) | Regressions from code changes | Drift from corpus/model/user-base shift with no code change |

### [08 · Fine-Tuning vs. RAG vs. Hybrid](level-4/08-finetuning-vs-rag.md)

| Symptom | Points to |
|---|---|
| Facts are correct at training time, stale by launch | RAG, not fine-tuning |
| Correct facts, wrong format/tone/structure | Fine-tuning, not more retrieval |
| Need a cheap, narrow, high-volume task done well | Fine-tuned small model |
| Facts change frequently, format must stay consistent | Hybrid — fine-tuned model + RAG context |
| "Let's fine-tune on our whole knowledge base" | Almost always the wrong call — that's what RAG is for |

### [09 · Scaling Ingestion Pipelines](level-4/09-scaling-ingestion.md)

| Lever | Gains | Ceiling |
|---|---|---|
| Parallel requests | Overlaps I/O wait | Provider's concurrency/rate limit |
| Batching | Amortizes per-request overhead | Diminishing returns once overhead is amortized; bigger failure blast radius |
| Both combined | Multiplies the above | Rate limit is shared — parallel batches can still trip it |
| None of the above, just "run it overnight" | Simplicity | Doesn't scale to same-day freshness at enterprise volume |

---

Go deeper with the full [RAG Pipelines Mastery Path](index.md) curriculum or test yourself with the [interview questions](interview-questions.md).

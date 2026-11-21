---
title: "Final Quiz"
description: "Test what you have learned across all four levels of RAG Pipelines Mastery Path with a 19-question final quiz."
---

# Final Quiz

19 questions drawn from all four levels of this course. You need **70% (14 out of 19)** to pass —
you can retake it as many times as you like. Your best result is saved in this browser.

<div id="mp-quiz-root"></div>

<script type="application/json" id="mp-quiz-data">
[
  {
    "question": "What does this print?\n```\ndef chunk_fixed(text, size=800, overlap=150):\n    out=[]; start=0\n    while start < len(text):\n        out.append(text[start:start+size]); start += size-overlap\n    return out\nprint([len(c) for c in chunk_fixed('A'*2000)])\n```",
    "options": [
      "[800, 800, 700, 50]",
      "[650, 650, 650, 50]",
      "[800, 800, 800, 50]",
      "[800, 800, 400]"
    ],
    "correctIndex": 0,
    "explanation": "Starts advance by 650: chunks begin at 0, 650, 1300 and 1950, with lengths 800, 800, 700 and 50."
  },
  {
    "question": "What does this print?\n```\ndef precision(r, rel): return sum(x in rel for x in r)/len(r)\ndef recall(r, rel): return sum(x in r for x in rel)/len(rel)\nr=[0,3,4]; rel={0,3}\nprint(round(precision(r,rel),3), recall(r,rel))\n```",
    "options": [
      "1.0 1.0",
      "0.5 0.5",
      "1.0 0.667",
      "0.667 1.0"
    ],
    "correctIndex": 3,
    "explanation": "Two of three retrieved are relevant; both relevant docs were retrieved."
  },
  {
    "question": "What does this print?\n```\ndef rrf(rankings,k=60):\n    s={}\n    for r in rankings:\n        for i,d in enumerate(r):\n            s[d]=s.get(d,0)+1/(k+i+1)\n    return sorted(s,key=lambda d:-s[d])\nprint(rrf([['a','b','c'],['b','a','d']]))\n```",
    "options": [
      "['b', 'a', 'c', 'd']",
      "['c', 'd', 'a', 'b']",
      "['a', 'b', 'c', 'd']",
      "['a', 'b', 'd', 'c']"
    ],
    "correctIndex": 2,
    "explanation": "a and b tie on the fused score; sorted() is stable so a stays first, then c beats d."
  },
  {
    "question": "What does this print?\n```\nimport hashlib\nh=lambda t: hashlib.sha256(t.encode()).hexdigest()\nprint(h('Refunds within 30 days.')==h('Refunds within 30 days.'), h('30 days')==h('45 days'))\n```",
    "options": [
      "False True",
      "True True",
      "True False",
      "False False"
    ],
    "correctIndex": 2,
    "explanation": "Content hashing detects whether a document changed, so unchanged ones skip re-embedding."
  },
  {
    "question": "What does this print?\n```\ndocs=[('acme','a rev $2M'),('globex','g rev $5M')]\nq={'g','rev'}\npost=[d for d in sorted(docs,key=lambda d:-len(q&set(d[1].split())))[:1] if d[0]=='acme']\npre=sorted([d for d in docs if d[0]=='acme'],key=lambda d:-len(q&set(d[1].split())))[:1]\nprint(post, pre)\n```",
    "options": [
      "[('globex', 'g rev $5M')] []",
      "[('acme', 'a rev $2M')] [('acme', 'a rev $2M')]",
      "[] [('acme', 'a rev $2M')]",
      "[] []"
    ],
    "correctIndex": 2,
    "explanation": "Filtering after ranking loses the tenant's own data when another tenant ranks first; filter before ranking."
  },
  {
    "question": "Users type 'card got rejected' but docs say 'payment declined'. What helps most?",
    "options": [
      "Reduce k to 1",
      "Add overlap",
      "Query rewriting/expansion plus dense (embedding) retrieval",
      "Delete the docs"
    ],
    "correctIndex": 2,
    "explanation": "It's a vocabulary mismatch."
  },
  {
    "question": "Queries contain error codes like E4022 that embeddings mangle. What do you add?",
    "options": [
      "More chunks",
      "BM25 keyword search fused with dense retrieval (hybrid)",
      "A larger LLM",
      "A cache"
    ],
    "correctIndex": 1,
    "explanation": "Embeddings are poor at exact tokens."
  },
  {
    "question": "The right chunk is at rank 7 but the LLM sees only the top 3. What is the highest-leverage fix?",
    "options": [
      "Increase chunk size",
      "Retrieve ~50 then rerank with a cross-encoder",
      "Add a cache",
      "Lower the temperature"
    ],
    "correctIndex": 1,
    "explanation": "Reranking can move the right document from rank 7 to rank 1."
  },
  {
    "question": "A question needs facts chained across three documents. What pattern helps?",
    "options": [
      "Lower overlap",
      "Query decomposition with multi-hop retrieval",
      "Raising k to 100",
      "Fine-tuning"
    ],
    "correctIndex": 1,
    "explanation": "Each hop's evidence feeds the next query."
  },
  {
    "question": "You must answer 'how many seats total across plans?' from a table. What approach fits?",
    "options": [
      "Use a reranker",
      "Raise top_k",
      "Text-to-SQL or code over structured data; retrieval can't aggregate",
      "Embed the whole table once"
    ],
    "correctIndex": 2,
    "explanation": "Aggregations need computation, not retrieval."
  },
  {
    "question": "You want to know if a technique earns its keep. What do you do?",
    "options": [
      "Try it once",
      "Run an ablation on a golden set with a flag per technique",
      "Ask the LLM",
      "Trust the leaderboard"
    ],
    "correctIndex": 1,
    "explanation": "Delete techniques that don't help your corpus."
  },
  {
    "question": "Your RAG answers are fluent but cite nothing relevant. Which metric flags that?",
    "options": [
      "Faithfulness (are claims supported by the context) plus context precision",
      "Chunk count",
      "Cost",
      "Latency"
    ],
    "correctIndex": 0,
    "explanation": "Faithfulness catches unsupported claims."
  },
  {
    "question": "A recall of 100% with low precision tells you what?",
    "options": [
      "You over-retrieve; noise is entering the context",
      "Embeddings are bad",
      "The index is empty",
      "The system is perfect"
    ],
    "correctIndex": 0,
    "explanation": "Track precision alongside recall."
  },
  {
    "question": "A policy doc is updated nightly. How do you keep the index fresh cheaply?",
    "options": [
      "Never update",
      "Rebuild everything nightly",
      "Use timestamps only",
      "Content hashing with upserts and deletes, re-embedding only changed documents"
    ],
    "correctIndex": 3,
    "explanation": "Don't forget deletes, or removed policies keep being retrieved."
  },
  {
    "question": "A customer-uploaded PDF contains 'IGNORE ALL PREVIOUS INSTRUCTIONS'. What is this?",
    "options": [
      "A formatting error",
      "A chunking bug",
      "A cache hit",
      "Indirect prompt injection; treat retrieved context as untrusted input"
    ],
    "correctIndex": 3,
    "explanation": "Any ingestion source is an injection vector."
  },
  {
    "question": "A multi-tenant system filters results after ranking. What can go wrong?",
    "options": [
      "Nothing",
      "It can deny a tenant their own data or leak another's; filter before ranking",
      "Slower embeddings",
      "Larger indexes"
    ],
    "correctIndex": 1,
    "explanation": "The order of operations is the security model."
  },
  {
    "question": "Your facts change weekly but you want the model to answer in a fixed brand voice. What do you combine?",
    "options": [
      "Fine-tune for facts",
      "RAG for facts plus fine-tuning (or prompting) for voice",
      "RAG for voice only",
      "Neither"
    ],
    "correctIndex": 1,
    "explanation": "They're complementary, not competitors."
  },
  {
    "question": "At a million queries a month, where does most of the cost go?",
    "options": [
      "LLM input/output tokens, not embeddings",
      "Storage",
      "Vector search",
      "Embeddings"
    ],
    "correctIndex": 0,
    "explanation": "Reduce context size, output length or use caching."
  },
  {
    "question": "Your mean latency is fine but users complain. What should you check?",
    "options": [
      "Embedding dimensions",
      "Chunk size",
      "The mean",
      "p99 latency, since stragglers dominate the tail"
    ],
    "correctIndex": 3,
    "explanation": "A 1% straggler can make p99 7x p50."
  }
]
</script>

## Coding challenges

Try these on your own computer, then compare your output. Nothing you write is sent anywhere.

### Chunk with overlap

Write `chunk_fixed(text, size, overlap)` that slices text into `size`-character chunks, advancing by `size - overlap`, and print the chunk lengths for 'A' * 25 with size 10 and overlap 3.

**Starter code**

```python
def chunk_fixed(text, size, overlap):
    # your code here
    pass

print([len(c) for c in chunk_fixed("A" * 25, 10, 3)])
```

**Expected output**

```text
[10, 10, 10, 4]
```

<details><summary>Show solution and explanation</summary>

<pre><code>def chunk_fixed(text, size, overlap):
    out, start = [], 0
    while start &lt; len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out

print([len(c) for c in chunk_fixed(&quot;A&quot; * 25, 10, 3)])</code></pre>

<p>The start advances by 7 each time: chunks start at 0, 7, 14 and 21.</p>
</details>

### Reciprocal rank fusion

Write `rrf(rankings, k=60)` that fuses ranked lists using score 1/(k + rank) (rank starting at 1) and returns document ids best-first. Print the fusion of [['a','b','c'], ['c','b','d']].

**Starter code**

```python
def rrf(rankings, k=60):
    # your code here
    pass

print(rrf([["a", "b", "c"], ["c", "b", "d"]]))
```

**Expected output**

```text
['c', 'b', 'a', 'd']
```

<details><summary>Show solution and explanation</summary>

<pre><code>def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])

print(rrf([[&quot;a&quot;, &quot;b&quot;, &quot;c&quot;], [&quot;c&quot;, &quot;b&quot;, &quot;d&quot;]]))</code></pre>

<p>c scores 1/63 + 1/61 (rank 3 in one list, rank 1 in the other), slightly more than b&#x27;s 1/62 + 1/62, so c comes first, then b, then a (1/61) and d (1/63). RRF rewards documents that rank well in several lists.</p>
</details>

---
title: "Learn RAG (Retrieval-Augmented Generation) Free: Full Course"
description: "Free course on building RAG pipelines -- embeddings, vector stores, and retrieval, with real hands-on projects. Part of a 37-course free learning library."
---

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

- [AI/ML Mastery Path](https://sigilipelli.github.io/ai-ml-skillmastery/) — machine-learning foundations
- [LLM Development Mastery Path](https://sigilipelli.github.io/llm-dev-skillmastery/) — building LLM applications

🎥 **Prefer video?** Watch the [Mastery Path video series](https://youtube.com/@sigilipelli) on YouTube — Shorts and full walkthroughs of these lessons.

## More from the Mastery Path series

Free, structured, module-wise training across 59 other languages, platforms and disciplines:

<div class="mastery-grid-wrap">
<p class="mastery-grid-category">Languages</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/python-skillmastery/">🐍 Python</a>
  <a href="https://sigilipelli.github.io/java-skillmastery/">☕ Java</a>
  <a href="https://sigilipelli.github.io/javascript-skillmastery/">🟨 JavaScript</a>
  <a href="https://sigilipelli.github.io/typescript-skillmastery/">🔷 TypeScript</a>
  <a href="https://sigilipelli.github.io/csharp-skillmastery/">🔵 C#</a>
  <a href="https://sigilipelli.github.io/shell-skillmastery/">🐚 Shell/Bash</a>
  <a href="https://sigilipelli.github.io/powershell-skillmastery/">💻 PowerShell</a>
  <a href="https://sigilipelli.github.io/c-skillmastery/">🇨 C</a>
  <a href="https://sigilipelli.github.io/cpp-skillmastery/">➕ C++</a>
  <a href="https://sigilipelli.github.io/go-skillmastery/">🐹 Go</a>
  <a href="https://sigilipelli.github.io/rust-skillmastery/">🦀 Rust</a>
  <a href="https://sigilipelli.github.io/sql-skillmastery/">🗄️ SQL</a>
  <a href="https://sigilipelli.github.io/ruby-skillmastery/">💎 Ruby</a>
  <a href="https://sigilipelli.github.io/php-skillmastery/">🐘 PHP</a>
  <a href="https://sigilipelli.github.io/kotlin-skillmastery/">🟣 Kotlin</a>
  <a href="https://sigilipelli.github.io/swift-skillmastery/">🐦 Swift</a>
  <a href="https://sigilipelli.github.io/dart-skillmastery/">🎯 Dart</a>
  <a href="https://sigilipelli.github.io/scala-skillmastery/">🔴 Scala</a>
  <a href="https://sigilipelli.github.io/r-skillmastery/">📊 R</a>
  <a href="https://sigilipelli.github.io/matlab-skillmastery/">🟧 MATLAB</a>
</div>
<p class="mastery-grid-category">Testing & QA</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/java-testing-skillmastery/">🧪 Java Testing</a>
  <a href="https://sigilipelli.github.io/cpp-testing-skillmastery/">🧪 C/C++ Testing</a>
  <a href="https://sigilipelli.github.io/python-testing-skillmastery/">🧪 Python Testing</a>
  <a href="https://sigilipelli.github.io/automotive-testing-skillmastery/">🚗 Automotive Testing</a>
</div>
<p class="mastery-grid-category">Security</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/cybersecurity-skillmastery/">🛡️ Cybersecurity</a>
</div>
<p class="mastery-grid-category">Cloud Platforms</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/aws-skillmastery/">☁️ AWS</a>
  <a href="https://sigilipelli.github.io/azure-skillmastery/">☁️ Azure</a>
  <a href="https://sigilipelli.github.io/gcp-skillmastery/">☁️ GCP</a>
  <a href="https://sigilipelli.github.io/ibm-cloud-skillmastery/">☁️ IBM Cloud</a>
  <a href="https://sigilipelli.github.io/adobe-skillmastery/">🎨 Adobe</a>
</div>
<p class="mastery-grid-category">Data & Analytics</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/data-engineering-skillmastery/">🛠️ Data Engineering</a>
  <a href="https://sigilipelli.github.io/data-science-skillmastery/">📈 Data Science</a>
  <a href="https://sigilipelli.github.io/tableau-skillmastery/">📊 Tableau</a>
  <a href="https://sigilipelli.github.io/excel-skillmastery/">📗 Excel</a>
  <a href="https://sigilipelli.github.io/pyspark-skillmastery/">⚡ PySpark</a>
  <a href="https://sigilipelli.github.io/etl-datalake-skillmastery/">🏞️ ETL & Data Lake</a>
</div>
<p class="mastery-grid-category">AI / ML / LLM</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/ai-ml-skillmastery/">🤖 AI/ML</a>
  <a href="https://sigilipelli.github.io/llm-dev-skillmastery/">🧠 LLM Dev</a>
  <a href="https://sigilipelli.github.io/edge-ai-skillmastery/">📱 Edge AI</a>
  <a href="https://sigilipelli.github.io/claude-training-skillmastery/">🔶 Claude Training</a>
  <a href="https://sigilipelli.github.io/ai-tools-skillmastery/">🧰 AI Tools</a>
  <a href="https://sigilipelli.github.io/ml-math-skillmastery/">➗ ML Math Foundations</a>
</div>
<p class="mastery-grid-category">Embedded Systems</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/embedded-skillmastery/">🔌 Embedded</a>
  <a href="https://sigilipelli.github.io/embedded-linux-skillmastery/">🐧 Embedded Linux</a>
  <a href="https://sigilipelli.github.io/embedded-python-skillmastery/">🐍 Embedded Python</a>
  <a href="https://sigilipelli.github.io/freertos-skillmastery/">⏱️ FreeRTOS</a>
  <a href="https://sigilipelli.github.io/s32k-skillmastery/">🔧 S32K</a>
</div>
<p class="mastery-grid-category">Leadership & Management</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/product-manager-skillmastery/">📋 Product Manager</a>
  <a href="https://sigilipelli.github.io/product-lead-skillmastery/">🧭 Product Lead</a>
  <a href="https://sigilipelli.github.io/project-manager-skillmastery/">📅 Project Manager</a>
  <a href="https://sigilipelli.github.io/ai-manager-skillmastery/">🤖 AI Manager</a>
  <a href="https://sigilipelli.github.io/servant-leadership-skillmastery/">🤝 Servant Leadership</a>
</div>
<p class="mastery-grid-category">Professional Skills</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/english-fluency-skillmastery/">🗣️ English Fluency & IELTS</a>
  <a href="https://sigilipelli.github.io/workday-skillmastery/">🧑‍💼 Workday</a>
</div>
<p class="mastery-grid-category">Process & APIs</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/agile-skillmastery/">🔄 Agile/Scrum/Kanban</a>
  <a href="https://sigilipelli.github.io/rest-api-skillmastery/">🔗 REST API</a>
  <a href="https://sigilipelli.github.io/playwright-skillmastery/">🎭 Playwright</a>
</div>
<p class="mastery-grid-category">Infrastructure & Ops</p>
<div class="mastery-grid">
  <a href="https://sigilipelli.github.io/server-ops-skillmastery/">🖥️ Server Ops</a>
  <a href="https://sigilipelli.github.io/nodemcu-skillmastery/">📶 NodeMCU/IoT</a>
</div>
</div>

# 06 · Structured & Table RAG

Every RAG pipeline built so far assumes documents are prose: a paragraph means
something on its own, and cutting between paragraphs is safe. Tables violate
both assumptions. A table cell containing `99` is meaningless without its column
header (`Seats`) and its row label (`Team`). Chunk a table naively and you
scatter that context across three chunks that each look fine and say nothing.

Worse, the questions people ask about tables — "how many seats total?", "which
plans are under $100?" — are *aggregations*. Retrieval cannot sum. No amount of
top-k tuning will make a retriever compute `SUM(seats)`.

!!! note "What ran here"
    All three strategies below ran locally using `rank-bm25` and Python's
    built-in `sqlite3`. No API key, no LLM. The text-to-SQL section shows
    hand-written SQL where production would have an LLM generate it — the
    execution and safety story is identical.

## The table

```python
TABLE_MD = """\
| Plan       | Price/mo | Seats | SLA    | Refundable |
|------------|----------|-------|--------|------------|
| Free       | 0        | 1     | none   | n/a        |
| Starter    | 29       | 5     | none   | 14 days    |
| Team       | 99       | 25    | 99.5%  | 14 days    |
| Enterprise | 499      | 250   | 99.9%  | 30 days    |
"""
```

## Strategy A: the whole table as one chunk

The default behavior of most splitters, and the reason table questions fail.

```text
one chunk, 336 chars, 4 rows fused together
```

It survives a fixed-size splitter here because it is small. Its real problems:

- **Every row retrieves together.** Ask about the Team plan and the LLM receives
  all four plans. It must now do the lookup itself, in-context, and it will
  sometimes read the wrong row — pulling Enterprise's SLA onto the Team plan.
  Fluent, cited, wrong.
- **It does not scale.** A 200-row table blows past both your context budget and
  a cross-encoder's 512-token window (lesson 3).
- **Markdown pipe alignment is fragile.** Once a splitter cuts mid-table, the
  remaining fragment has no header row at all, and the numbers become
  unlabelled.

## Strategy B: row-level chunks, verbalized

Turn each row into a self-contained sentence that carries its own headers.

```python
def parse(md):
    rows = [l.strip().strip("|").split("|") for l in md.strip().splitlines()]
    header = [c.strip() for c in rows[0]]
    return header, [[c.strip() for c in r] for r in rows[2:]]

header, rows = parse(TABLE_MD)

row_chunks = []
for r in rows:
    sentence = f"The {r[0]} plan costs ${r[1]} per month, includes {r[2]} seats, " \
               f"has an SLA of {r[3]}, and is refundable for {r[4]}."
    row_chunks.append(sentence)

bm25 = BM25Okapi([tok(c) for c in row_chunks])
q = "how many seats does the Team plan include"
scores = bm25.get_scores(tok(q))
best = max(range(len(scores)), key=lambda i: scores[i])
print(f"-> {row_chunks[best]}")
```

```text
--- strategy B: row-level chunks, verbalized ---
  The Free plan costs $0 per month, includes 1 seats, has an SLA of none, and is refundable for n/a.
  The Starter plan costs $29 per month, includes 5 seats, has an SLA of none, and is refundable for 14 days.
  The Team plan costs $99 per month, includes 25 seats, has an SLA of 99.5%, and is refundable for 14 days.
  The Enterprise plan costs $499 per month, includes 250 seats, has an SLA of 99.9%, and is refundable for 30 days.

query: 'how many seats does the Team plan include'
  -> The Team plan costs $99 per month, includes 25 seats, has an SLA of 99.5%, and is refundable for 14 days.
```

Exactly one row retrieved, carrying its own labels. The LLM cannot cross rows
because it never sees the other rows.

Verbalization matters more than it looks. `Team | 99 | 25 | 99.5% | 14 days` and
"The Team plan costs $99 per month, includes 25 seats" contain identical
information, but only the second one matches how users phrase questions — which
is what both BM25 and an embedding model are scoring against.

Two practical refinements:

- **Prepend a table caption** to every row chunk ("From the Pricing Plans
  table: ...") so a row retrieved in isolation still says where it came from.
- **Keep the raw row in metadata** so you can render the real table in the
  answer while retrieving on the verbalized text.

And the honest limitation: **strategy B still cannot aggregate.** Ask "how many
seats across all plans" and you retrieve one row, or a few, and the LLM does
arithmetic on partial data. It will produce a number. The number will be wrong.

## Strategy C: text-to-SQL

If the data is genuinely tabular, stop pretending it is text.

```python
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE plans (plan TEXT, price INT, seats INT, sla TEXT, refundable TEXT)")
db.executemany("INSERT INTO plans VALUES (?,?,?,?,?)",
               [(r[0], int(r[1]), int(r[2]), r[3], r[4]) for r in rows])

for label, sql in [
    ("total seats across all plans", "SELECT SUM(seats) FROM plans"),
    ("plans under $100",             "SELECT plan, price FROM plans WHERE price < 100"),
    ("most expensive plan",          "SELECT plan, price FROM plans ORDER BY price DESC LIMIT 1"),
]:
    print(f"  {label:30s} -> {db.execute(sql).fetchall()}")
```

```text
  total seats across all plans   -> [(281,)]
  plans under $100               -> [('Free', 0), ('Starter', 29), ('Team', 99)]
  most expensive plan            -> [('Enterprise', 499)]
```

`281` is *correct*, and it is correct by construction rather than by luck. No
retrieval strategy produces that. The LLM's job shifts from "compute the answer"
to "write the query and phrase the result" — and databases are extremely good
at the part LLMs are worst at.

In production the LLM generates the SQL from the schema:

```python
SQL_PROMPT = """Given this schema, write one SQLite SELECT query answering the
question. Output only SQL, no explanation, no markdown fences.

Schema: plans(plan TEXT, price INT, seats INT, sla TEXT, refundable TEXT)
Question: {question}"""
```

Non-negotiable guardrails: connect **read-only**, allow only `SELECT`, reject
anything containing `;`, `DROP`, `INSERT`, `UPDATE`, `DELETE`, or `ATTACH`,
enforce a `LIMIT`, and set a query timeout. A user asking "delete all plans" must
not be one prompt injection away from a generated `DROP TABLE`. Treat generated
SQL exactly like untrusted user input — because that is what it is.

## Choosing a strategy

| Strategy | Lookup ("Team's SLA?") | Aggregation ("total seats?") | Comparison ("cheapest with 99.9%?") | Scales to 10k rows | Needs schema |
|---|---|---|---|---|---|
| A: whole table chunk | Unreliable — row confusion | No | Sometimes, if it fits | No | No |
| B: row chunks, verbalized | **Reliable** | No | Only if all rows retrieved | Yes | No |
| C: text-to-SQL | Reliable | **Reliable** | **Reliable** | Yes | Yes |
| D: table summary chunk | No | No | No | Yes | No |
| B + C hybrid | Reliable | Reliable | Reliable | Yes | Yes |

**Strategy D** deserves a mention: index a one-paragraph LLM-written summary of
the table ("This table lists four pricing plans from Free to Enterprise,
covering price, seats, SLA and refund terms") purely as a *router*. It answers
nothing, but it makes the table findable for vague queries like "what plans do
you offer", which then route to B or C.

The production pattern is the hybrid: **route** the question. Lookup and
descriptive questions go to row chunks; anything involving counting, summing,
filtering by threshold, or superlatives goes to SQL. Mixed questions ("which
plans are refundable and what does the refund policy say?") need both — SQL for
the rows, vector search for the prose policy — with the results merged in the
prompt.

## Traps

- **Splitting a table mid-body.** The second fragment loses the header row and
  every number becomes anonymous. Detect table boundaries before chunking and
  treat them as atomic.
- **Merged and multi-level headers.** Real-world spreadsheets and PDF tables use
  spanning cells that flatten into garbage. These usually need manual mapping;
  budget for it.
- **Units and footnotes.** "Price" in USD, "Latency" in ms, an asterisk pointing
  to a footnote three pages later. Verbalize units *into* the row text or the
  LLM will guess.
- **Believing LLM arithmetic.** An LLM summing a retrieved column returns a
  confident, plausible, frequently wrong number — and unlike a hallucinated
  fact, nobody thinks to check a number. If a question needs math, route it to
  code.
- **Text-to-SQL on a schema the model cannot see.** Cryptic column names
  (`amt_usd_net_q`) produce wrong-but-valid SQL. Ship column descriptions in the
  prompt, and prefer views with readable names.
- **Stale row chunks.** When the source table changes, every derived row chunk
  must be re-generated. Key chunks by row id so updates are surgical.

## Cheat sheet

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

## Exercise

Take a table with at least 15 rows and 5 columns — a real one, from your own
docs or a public dataset.

Implement strategies A, B, and C. Write 12 questions in three groups: 4 pure
lookups, 4 aggregations (sum, count, average, max), and 4 comparisons or
threshold filters ("which rows have X above Y?").

Measure accuracy of the **final answer** (not just retrieval) for each strategy
on each group, and build the 3×3 grid.

Then go further:

1. Write a **router**: a small classifier (keyword rules are fine — `total`,
   `how many`, `average`, `most`, `least`) that sends each question to B or C.
   What is the router's accuracy, and what does an end-to-end pipeline score
   with it compared to always using B or always using C?
2. Find a question your router gets wrong and trace the failure end to end. Was
   it routed wrong, retrieved wrong, or computed wrong?
3. Add a mixed question that needs both a SQL result and a prose policy chunk.
   How do you assemble both into one prompt without the LLM confusing which
   source said what?

# 03 · GraphRAG & Knowledge Graphs

Vector retrieval finds chunks that are *semantically similar* to a query.
It's bad at a different, common question shape: "how are these two entities
connected?" or "what else relates to this thing, even if it's never mentioned
in the same sentence?" GraphRAG answers those by building an explicit graph of
entities and relations, and retrieving by *traversal* instead of similarity.

We use `networkx` here as a lightweight, real, runnable stand-in for a graph
store (Neo4j, in production). The traversal logic is the same; only the
storage and query language differ.

## Building the graph

```python
import networkx as nx

# In production, an LLM extracts (subject, relation, object) triples from
# your chunks. Hardcoded here — extraction quality is covered in the trap below.
triples = [
    ("Jane Rivera", "founded", "Acme Corp"),
    ("Jane Rivera", "worked_at", "Globex"),
    ("Initech", "acquired", "Globex"),
    ("Marcus Lee", "ceo_of", "Initech"),
    ("Acme Corp", "competitor_of", "Initech"),
]

G = nx.DiGraph()
for s, r, o in triples:
    G.add_edge(s, o, relation=r)

print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
```

Captured output:

```
nodes: 5 edges: 5
```

## Retrieval by traversal, not similarity

```python
def graph_search(entity, hops=2):
    found = {entity}
    frontier = {entity}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            for nbr in list(G.successors(n)) + list(G.predecessors(n)):
                nxt.add(nbr)
        found |= nxt
        frontier = nxt
    return found

print(sorted(graph_search("Jane Rivera", hops=2)))
```

Captured output:

```
['Acme Corp', 'Globex', 'Initech', 'Jane Rivera']
```

Two hops out from "Jane Rivera" reaches "Initech" — the entity that acquired
her former employer — with **zero text similarity** between "Jane Rivera" and
"Initech" anywhere in the source. A vector retriever, given the query "Jane
Rivera", would never surface the Initech chunk; nothing about it is
semantically close to her name. This is GraphRAG's actual value: connections
that exist structurally but not lexically or semantically.

## Community summaries

Real GraphRAG (Microsoft's variant, notably) doesn't stop at traversal — it
clusters the graph into communities and pre-summarizes each one, so a query
about "the Acme/Initech cluster" can retrieve one dense summary instead of
walking edges at query time. A cheap approximation is connected components:

```python
comms = list(nx.connected_components(G.to_undirected()))
print("communities:", len(comms))
print([sorted(c) for c in comms])
```

Captured output:

```
communities: 1
[['Acme Corp', 'Globex', 'Initech', 'Jane Rivera', 'Marcus Lee']]
```

With only five triples, everything is one connected graph — realistic
corpora fragment into dozens or hundreds of communities, each summarizable
independently. That summarization step is what makes GraphRAG's query-time
cost tractable: without it, every query pays the full graph-construction-time
cost of understanding relationships, at request latency.

## The trap: construction cost is the real bill, not query time

Graph traversal at query time is fast — the example above ran in
microseconds. What's expensive, and what teams underestimate, is **building**
the graph:

- **Entity extraction** requires an LLM call per chunk (or per document) to
  pull out triples — for a 100K-chunk corpus, that's 100K+ LLM calls before
  you've answered a single user query. Vector indexing, by comparison, is one
  embedding call per chunk with no reasoning required.
- **Entity resolution** — "Jane Rivera", "J. Rivera", and "the Acme founder"
  must collapse to one node, or your graph fragments into duplicates that
  each look sparsely connected. Getting this wrong silently loses the
  multi-hop connections GraphRAG exists to provide.
- **Community summarization** repeats the LLM cost at a coarser grain, and
  needs re-running whenever the graph changes meaningfully — it doesn't
  incrementally update the way appending a vector to an index does.
- **Staleness** — a vector index update is "add this chunk's embedding." A
  graph update might require re-resolving entities and re-summarizing an
  entire community, which teams often batch nightly instead of doing live,
  meaning graph answers can lag fresher-but-simpler vector answers.

The rule of thumb: reach for GraphRAG when your queries are genuinely
relational ("what connects X and Y", "who else is affected by Z") and you can
afford the extraction pipeline. For queries that are really just "find the
similar chunk," a vector index is cheaper to build, cheaper to update, and
just as accurate.

## Cheat sheet

| | Vector RAG | GraphRAG |
|---|---|---|
| Finds | Semantically similar text | Structurally connected entities |
| Build cost | 1 embedding call/chunk | LLM extraction + resolution + summarization |
| Update cost | Append embedding | Re-resolve entities, maybe re-summarize |
| Query cost | 1 similarity search | Traversal (cheap) or community lookup |
| Best for | "Find X" | "How does X relate to Y" |
| Failure mode | Miss paraphrases | Miss connections if extraction/resolution is bad |

## Exercise

Add a sixth triple, `("Marcus Lee", "mentored", "Jane Rivera")`, creating a
cycle back into the existing cluster. Re-run `graph_search("Jane Rivera",
hops=1)` and confirm "Marcus Lee" now appears at hop 1 instead of hop 3+ —
then explain in a comment why this one new edge shortens every future query
that needs to connect these two people, without touching the vector index at
all.

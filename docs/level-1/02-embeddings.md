# 02 · Embeddings & Semantic Similarity

Embeddings are the engine that makes retrieval-by-meaning possible. An
embedding model turns a piece of text into a fixed-length list of numbers — a
**vector** — with one crucial property: texts that mean similar things get
vectors that are close together, even if they share no words. "How do I get my
money back?" and "refund policy" land near each other; "refund policy" and
"deployment guide" do not. This lesson builds that intuition with real,
runnable code using `sentence-transformers`, which runs locally and free — no
API key.

## What a vector embedding is

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# ~80 MB download on first run, then cached locally. Runs on CPU fine.
model = SentenceTransformer("all-MiniLM-L6-v2")

vec = model.encode("How do I get my money back?")
print(type(vec), vec.shape)
# <class 'numpy.ndarray'> (384,)
print(vec[:5])
# [-0.01429  0.0674   0.0021  -0.0805   0.0344]   (your numbers will differ slightly)
```

Every text — a word, a sentence, a paragraph — becomes a point in a
384-dimensional space. The individual numbers are meaningless on their own;
what matters is **where the point sits relative to other points**. The model
learned, from hundreds of millions of sentence pairs, to place paraphrases
near each other and unrelated text far apart.

## Measuring similarity: cosine similarity

The standard closeness measure is **cosine similarity** — the cosine of the
angle between two vectors:

- `1.0` → same direction (same meaning)
- `~0.0` → unrelated
- `-1.0` → opposite direction (rare in practice with these models)

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

a = model.encode("How do I get my money back?")
b = model.encode("What is the refund policy?")
c = model.encode("How do I deploy the app to production?")

print(f"money-back vs refund:  {cosine_similarity(a, b):.3f}")
print(f"money-back vs deploy:  {cosine_similarity(a, c):.3f}")
# money-back vs refund:  0.688
# money-back vs deploy:  0.083
```

Two sentences with **zero words in common** score high; two questions with
similar surface form ("How do I ...") but different meaning score near zero.
That is the whole magic of semantic search — keyword search could never match
"money back" to "refund".

!!! info "Cosine similarity vs. cosine distance"
    You'll see both terms. **Distance** is just `1 - similarity`, so identical
    meaning = distance 0. ChromaDB (lesson 4) returns distances; most papers
    talk in similarities. Same information, flipped sign.

## Similarity search over a small corpus

Retrieval is nothing more than "embed everything, then find the nearest
vectors to the query." Here it is with plain numpy — this *is* a vector search
engine, just without an index:

```python
corpus = [
    "Annual plans can be refunded within 14 days of purchase.",
    "To reset your password, click 'Forgot password' on the login page.",
    "We support exporting your data as CSV or JSON at any time.",
    "Deploys to production happen automatically when main is merged.",
    "Contact billing@example.com for invoice and payment questions.",
]

corpus_vecs = model.encode(corpus)          # shape: (5, 384)

query = "can I get a refund?"
q_vec = model.encode(query)

scores = [cosine_similarity(q_vec, v) for v in corpus_vecs]
for score, text in sorted(zip(scores, corpus), reverse=True):
    print(f"{score:.3f}  {text}")
```

```text
0.638  Annual plans can be refunded within 14 days of purchase.
0.238  Contact billing@example.com for invoice and payment questions.
0.116  We support exporting your data as CSV or JSON at any time.
0.052  To reset your password, click 'Forgot password' on the login page.
0.049  Deploys to production happen automatically when main is merged.
```

The ranking is exactly what a human would produce: the refund sentence first,
the billing contact second (related topic), everything else nowhere. Note the
absolute numbers (0.638, not 0.99) — with real models, "very similar" often
means scores in the 0.5–0.8 range. **Rankings matter more than raw values.**

## Visualizing the space

You can *see* the clustering by projecting the 384 dimensions down to 2 with
PCA (`pip install scikit-learn matplotlib` if you want to run this):

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

texts = [
    "refund my order", "give me my money back", "cancel and refund",
    "deploy to production", "release the new version", "ship the build",
    "reset my password", "I forgot my login", "can't sign in",
]
vecs = model.encode(texts)
xy = PCA(n_components=2).fit_transform(vecs)

plt.figure(figsize=(7, 5))
plt.scatter(xy[:, 0], xy[:, 1])
for (x, y), t in zip(xy, texts):
    plt.annotate(t, (x, y), fontsize=8)
plt.title("Sentence embeddings, projected to 2D")
plt.show()
```

Three tight clusters appear — refunds, deploys, logins — with no labels or
keywords involved. The geometry *is* the meaning.

## Practical notes for RAG

- **One model for both sides.** The query and the documents must be embedded
  by the **same model**. Vectors from different models live in different
  spaces; comparing them is meaningless.
- **Batch your encoding.** `model.encode(list_of_texts)` is far faster than a
  loop of single calls.
- **Model choice.** `all-MiniLM-L6-v2` (384 dims) is small, fast, and a great
  default for learning and prototypes. Larger models (e.g.
  `all-mpnet-base-v2`, 768 dims) rank better at ~3× the compute. Hosted
  embedding APIs exist too — the concepts are identical.
- **Input limits.** These models truncate long inputs (256 word-pieces for
  MiniLM). Embedding a 10-page document as one vector silently throws away
  most of it — which is exactly why chunking (next lesson) exists.

## Cheat sheet

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

## Exercise

Build a tiny "semantic FAQ matcher": create a list of 8–10 FAQ answers from
any domain you like, embed them once, then write a loop that reads a question
from `input()`, embeds it, and prints the top-2 most similar FAQ answers with
their similarity scores. Test it with (a) a question that paraphrases an FAQ
using completely different words, and (b) a question your FAQ doesn't cover —
and note what scores the second case produces. What threshold would you pick
to answer "sorry, I don't know that one"?

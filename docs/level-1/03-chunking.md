# 03 · Chunking Strategies

Chunking — splitting documents into retrievable pieces — is the least
glamorous and most consequential decision in a RAG pipeline. Retrieval returns
chunks, the LLM reads chunks, and your evaluation scores chunks: if the chunks
are wrong, nothing downstream can fix them. This lesson covers why chunking
matters, the main strategies with their tradeoffs, and a working chunker in
plain Python that you'll reuse in the rest of Level 1.

## Why chunking matters

Two forces pull in opposite directions:

- **Embedding models truncate.** MiniLM reads ~256 word-pieces; whatever
  follows is invisible to search. A whole document embedded as one vector is
  mostly unsearchable, and its one vector is a blurry average of every topic
  it contains.
- **The LLM needs context.** A chunk of 15 characters retrieves precisely but
  tells the model nothing. If the answer spans a definition on one line and a
  caveat two sentences later, a too-small chunk delivers half an answer.

So chunks must be **small enough to embed as one coherent topic** and **large
enough to be useful evidence**. For prose, 200–500 words (or 500–1500
characters) is the classic starting zone — then you tune with evaluation
(lesson 8), not guesswork.

## Strategy 1: fixed-size chunks with overlap

The baseline: slice every N characters, letting each chunk share its tail
with the next chunk's head.

```python
def chunk_fixed(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks

text = "A" * 2000  # pretend document
for i, c in enumerate(chunk_fixed(text)):
    print(i, len(c))
# 0 800
# 1 800
# 2 800
# 3 50
```

**Overlap** exists because a fixed slicer *will* cut sentences and ideas in
half. If the boundary falls mid-thought, the overlap ensures at least one
chunk contains the whole thought. Typical overlap: 10–20% of chunk size. The
costs: duplicated storage and near-duplicate chunks in your top-k.

Fixed-size is dumb but predictable — a fine baseline and the right choice for
text with no exploitable structure.

## Strategy 2: sentence/paragraph-aware chunks

Better: never cut mid-sentence. Split on natural boundaries, then pack
units into chunks up to a size budget. This is the chunker we'll actually use
going forward:

```python
import re

def chunk_text(text: str, max_chars: int = 1000, overlap_units: int = 1) -> list[str]:
    """Pack paragraphs (falling back to sentences) into chunks of <= max_chars.

    overlap_units: how many trailing units to repeat at the start of the
    next chunk, for boundary continuity.
    """
    # Split into paragraphs; explode any oversized paragraph into sentences.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            units.append(p)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip())

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) > max_chars:
            chunks.append("\n".join(current))
            current = current[-overlap_units:]  # carry tail units forward
            size = sum(len(u) for u in current)
        current.append(unit)
        size += len(unit)
    if current:
        chunks.append("\n".join(current))
    return chunks
```

Try it:

```python
doc = """Refund Policy

Annual plans can be refunded within 14 days of purchase or renewal.
Monthly plans are non-refundable but can be cancelled at any time.

Password Resets

To reset your password, click 'Forgot password' on the login page.
Reset links expire after 30 minutes for security reasons.
"""

for i, c in enumerate(chunk_text(doc, max_chars=200)):
    print(f"--- chunk {i} ({len(c)} chars) ---")
    print(c)
```

Every chunk is made of whole sentences and tends to stay on one topic — which
is exactly what makes its embedding sharp.

## Strategy 3: structural chunking

Real documents have structure — Markdown headings, HTML tags, code function
boundaries. Splitting on structure keeps semantic units intact *and* gives you
metadata for free (which section did this chunk come from?):

```python
def chunk_markdown(text: str) -> list[dict]:
    """Split a markdown document on headings; return chunks with section metadata."""
    sections = re.split(r"(?m)^(#{1,3} .+)$", text)
    chunks = []
    heading = "Introduction"
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,3} ", part):
            heading = part.lstrip("# ").strip()
        else:
            # large sections still get sub-chunked by the size-aware packer
            for piece in chunk_text(part, max_chars=1000):
                chunks.append({"text": piece, "section": heading})
    return chunks

for c in chunk_markdown("# Refunds\nAnnual plans: 14 days.\n\n# Deploys\nMerge to main."):
    print(c)
# {'text': 'Annual plans: 14 days.', 'section': 'Refunds'}
# {'text': 'Merge to main.', 'section': 'Deploys'}
```

That `section` metadata pays off twice: it can be embedded with the text
("Refunds: Annual plans: 14 days.") for better retrieval, and it becomes the
citation you show users (lesson 6).

## Choosing chunk size: the tradeoffs

| Smaller chunks | Larger chunks |
|----------------|---------------|
| Sharper, single-topic embeddings | Blurrier, averaged embeddings |
| More precise retrieval hits | Retrieval drags in off-topic text |
| Risk: answer split across chunks | Answer more likely intact in one chunk |
| More vectors → more storage/lookups | Fewer vectors |
| Less context per hit for the LLM | More context per hit (and more noise) |

There is no universal best. FAQ-style content suits small chunks (one Q&A
each); narrative prose and legal text need larger ones; code splits best on
function boundaries. The professional answer: pick a sensible default
(paragraph-aware, ~1000 chars, 1 unit overlap), build the eval set from
lesson 8, and let hit-rate numbers choose for you.

## Cheat sheet

| Strategy | How it splits | Use when |
|----------|---------------|----------|
| Fixed-size + overlap | Every N chars, repeat last M | No structure to exploit; baseline |
| Sentence/paragraph-aware | Pack whole units to a size budget | Default for prose |
| Structural (headings/tags) | On document structure | Markdown, HTML, code |
| Overlap | Repeat boundary content | Mitigate mid-thought cuts (10–20%) |
| Size sweet spot | ~200–500 words to start | Tune with evaluation, not vibes |

## Exercise

Take any real Markdown file (a project README works well) and run all three
chunkers on it: `chunk_fixed`, `chunk_text`, and `chunk_markdown`. For each,
print the number of chunks and the first 80 characters of each chunk. Find one
concrete place where the fixed-size chunker cuts a sentence in half that the
paragraph-aware chunker keeps intact. Then embed the query "how do I install
this?" and the chunks with the lesson-2 model, and check: does the best-ranked
chunk differ between chunkers?

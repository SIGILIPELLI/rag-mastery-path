# 07 · Multi-Format Ingestion (PDF/HTML/Docs)

Every lesson so far started from a Python list of clean sentences. Real corpora
do not arrive that way. They arrive as PDFs with running headers on every page,
HTML wrapped in navigation and cookie banners, Word documents with tracked
changes, and Confluence exports full of macro residue.

Ingestion is unglamorous and it is where most RAG quality is won or lost. A
retriever cannot outperform its input: if every chunk begins with "ACME CLOUD -
INTERNAL HANDBOOK", that phrase becomes a high-frequency term that pollutes
BM25 statistics and drags every embedding toward the same region of vector
space.

!!! note "What ran here, and what didn't"
    The PDF and HTML extraction code below **ran locally**. The PDF is a real
    2-page file generated for this lesson (with genuine repeated headers and
    page footers) and parsed with `pypdf`; the HTML parser uses Python's
    standard-library `html.parser`. The `.docx` section is **described in prose**
    — `python-docx` was not installed for this run — and OCR was not attempted.

## PDF: the format that fights back

A PDF does not contain paragraphs. It contains instructions for drawing glyphs
at coordinates. Any notion of reading order, headings, or "this is a footer" is
reconstructed by the parser, heuristically, and sometimes wrongly.

```python
from pypdf import PdfReader

reader = PdfReader("handbook.pdf")
pages = [p.extract_text() for p in reader.pages]
print(f"PDF: {len(pages)} pages")
print("raw page 1:", repr(pages[0][:70]))
```

```text
PDF: 2 pages
raw page 1: 'ACME CLOUD - INTERNAL HANDBOOK\nBilling Policy\nAnnual plans can be refu'
```

The very first line of content is boilerplate. It appears on every page, and if
you chunk this directly, every chunk inherits it.

### Detecting boilerplate instead of hard-coding it

The robust trick: boilerplate is text that **repeats across pages**. You do not
need to know what it says.

```python
import re
from collections import Counter

line_counts = Counter(l.strip() for p in pages for l in p.splitlines() if l.strip())
boilerplate = {l for l, n in line_counts.items() if n >= len(pages) and len(pages) > 1}
boilerplate |= {l for l in line_counts if re.fullmatch(r"Page \d+ of \d+", l)}

def clean_page(text):
    keep = [l.strip() for l in text.splitlines()
            if l.strip() and l.strip() not in boilerplate]
    return "\n".join(keep)

cleaned = [clean_page(p) for p in pages]
```

```text
detected boilerplate: ['ACME CLOUD - INTERNAL HANDBOOK', 'Page 1 of 2', 'Page 2 of 2']

--- cleaned page 1 ---
Billing Policy
Annual plans can be refunded within 14 days of purchase.
Monthly plans are non-refundable but may be cancelled
at any time from the account settings page.
--- cleaned page 2 ---
Deployment Runbook
Deploys to production happen automatically when main
is merged. Rollbacks use the deploy --rollback command.
```

Note the two mechanisms working together. The running header was caught by
frequency (it appears on all pages). The page footers were **not** — `Page 1 of
2` and `Page 2 of 2` are different strings, each appearing once. A regex catches
those. Real corpora need both: frequency for constant boilerplate, patterns for
templated boilerplate.

Tune the threshold to `n >= len(pages) * 0.5` for long documents where the
header only starts on page 2, and always **log what you stripped**. A too-eager
rule that deletes a heading appearing on many pages is a silent recall bug.

### Line wrapping and hyphenation

Look at cleaned page 1: the sentence about monthly plans is split across two
lines because that is how it was laid out on the page. Those are *layout* line
breaks, not semantic ones. Left alone they corrupt chunk boundaries and break
phrase matching.

```python
def unwrap(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # de-hyphenate: "refund-\nable"
    text = re.sub(r"(?<![.!?:])\n(?=[a-z])", " ", text)  # join mid-sentence wraps
    return text
```

The rule: join a line break when the next line starts lowercase and the previous
line did not end in sentence punctuation. Keep it otherwise.

### Multi-column layouts

The one that silently destroys corpora. A two-column academic paper or
newsletter extracts as interleaved text — the first line of column 1, then the
first line of column 2 — producing prose that is grammatically plausible and
semantically scrambled. `pypdf` will not warn you.

If your PDFs are multi-column, you need layout-aware extraction
(`pdfplumber` with column detection, `unstructured`, or a document-AI service).
**Always eyeball the extracted text of a few documents before indexing
thousands.** Five minutes of reading beats a week of debugging bad answers.

## HTML: strip structure, not content

```python
from html.parser import HTMLParser

class Extract(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "aside"}
    BLOCK = {"p", "h1", "h2", "h3", "li", "div", "main", "tr"}
    def __init__(self):
        super().__init__()
        self.parts, self.depth = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP: self.depth += 1
        elif tag in self.BLOCK: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in self.SKIP: self.depth = max(0, self.depth - 1)
        elif tag in self.BLOCK: self.parts.append("\n")
    def handle_data(self, data):
        if self.depth == 0: self.parts.append(data)
    def text(self):
        raw = re.sub(r"[ \t]+", " ", "".join(self.parts))
        return "\n".join(l.strip() for l in raw.splitlines() if l.strip())
```

```text
--- HTML extracted ---
Refunds
Refund Policy
Annual plans can be refunded within 14 days of purchase.
Monthly plans are non-refundable.

--- what a naive tag-strip regex gives you instead ---
Refunds  .a{color:red}
```

The comparison is the point. `re.sub(r"<[^>]+>", " ", html)` is the one-liner
everyone reaches for, and it dumps **CSS rules and JavaScript source** straight
into your index. Those chunks then compete with real content in retrieval.

The parser version drops `<script>`, `<style>`, `<nav>`, and `<footer>` by
tracking nesting depth, and converts block elements into line breaks so that
`<li>` items do not run together into one unreadable sentence.

Two things to note in the output: the `<title>` text ("Refunds") survives, which
is usually desirable — capture it as metadata. And in production, use
`BeautifulSoup` or `trafilatura` rather than this class; `trafilatura` in
particular is built for main-content extraction and handles cookie banners,
sidebars, and comment sections that a tag allowlist will not.

## Word documents and the rest

For `.docx` (not executed in this run — `python-docx` was not installed):

```python
from docx import Document

doc = Document("policy.docx")
blocks = [p.text for p in doc.paragraphs if p.text.strip()]
tables = [[[c.text for c in row.cells] for row in t.rows] for t in doc.tables]
```

The traps specific to Office formats:

- **`doc.paragraphs` skips tables entirely.** Tables live in `doc.tables` and
  must be extracted separately, then handled per lesson 6. Silently losing every
  table in the corpus is a very common ingestion bug.
- **Tracked changes and comments** may extract as accepted, rejected, or
  duplicated depending on the library. Decide explicitly.
- **Headings carry structure worth keeping.** `p.style.name == "Heading 2"` lets
  you build a heading path ("Billing > Refunds > Annual") to prepend to each
  chunk — cheap context that measurably improves retrieval.
- **Legacy `.doc`** is a different format entirely. Convert with LibreOffice
  first.

## Format comparison

| Format | Library | Reliability | Main hazard |
|---|---|---|---|
| Markdown / txt | stdlib | Excellent | None — the easy case |
| HTML | `trafilatura`, BeautifulSoup | Good | Boilerplate; CSS/JS leaking in |
| PDF (text, 1 column) | `pypdf`, `pdfplumber` | Good | Headers/footers, line wrapping |
| PDF (multi-column) | `pdfplumber`, `unstructured` | Poor with naive tools | Silently scrambled reading order |
| PDF (scanned) | OCR: `tesseract`, doc-AI | Variable | Character errors; must OCR first |
| `.docx` | `python-docx` | Good | Tables skipped by default |
| `.pptx` | `python-pptx` | Fair | Fragments, not prose |
| Spreadsheets | `openpyxl`, `pandas` | Good | Treat as tables (lesson 6) |

## Traps

- **Indexing without looking.** The single most valuable ingestion habit: print
  the extracted text of five random documents and *read it*. Most catastrophic
  corpora are obvious in ten seconds and invisible in aggregate metrics.
- **Boilerplate poisoning retrieval.** Repeated text inflates document frequency
  (deflating IDF for real terms) and pulls every embedding toward a common
  centroid, flattening the similarity distribution you rely on for thresholds.
- **Empty extractions passing silently.** A scanned PDF returns `""` from
  `pypdf` with no error. Assert a minimum character count per document and route
  failures to a review queue rather than indexing an empty chunk.
- **Losing provenance.** Capture `source`, page number, and heading path during
  extraction — you cannot recover them later, and citations depend on them
  (lesson 5).
- **Encoding damage.** Smart quotes, ligatures (`ﬁ`), and non-breaking spaces
  become tokens that never match a user's typed query. Normalize with
  `unicodedata.normalize("NFKC", text)` early.
- **Non-deterministic re-ingestion.** If chunk ids change every run, you cannot
  do incremental updates and your eval set silently points at nothing. Derive
  chunk ids from a hash of source + position, not from enumeration order.

## Cheat sheet

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

## Exercise

Build an ingestion pipeline that survives contact with reality, and prove it.

Collect at least 15 documents across three formats, deliberately including one
scanned PDF, one multi-column PDF, and one HTML page with heavy navigation.
Write a `ingest.py` that dispatches on file extension, extracts text, strips
boilerplate, unwraps lines, normalizes Unicode, and emits chunks with `source`,
page/heading path, and a deterministic content-hash id.

Then build a **quality gate** that runs on every ingestion:

1. Flag any document extracting to fewer than 200 characters (catches scans and
   parse failures) and any chunk that is over 90% identical to another chunk
   (catches boilerplate that slipped through).
2. Report, per document, the ratio of alphabetic characters to total characters.
   Which of your files score lowest, and does the score correctly identify the
   ones a human would call garbage?
3. Index twice from the same sources and assert the chunk id sets are identical.
   If they are not, find the non-determinism — it is the bug that will break
   your incremental updates in lesson 10.

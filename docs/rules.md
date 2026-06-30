# Detection rules

Rules are implemented in [`src/outline_detection/detector.py`](../src/outline_detection/detector.py). Formula lists and `JUNCTION_BEFORE` live in [`src/outline_detection/utils.py`](../src/outline_detection/utils.py).

There are two families:

- **Rules A–H** — orthographic signals over a flat text string (implemented).
- **Rules I–L** — page-layout signals over OCR pages (proposed / opt-in, see [Page layout rules](#page-layout-rules-il)).

## Rule summary (A–H, orthographic)

| Rule | ID | Confidence | Pattern (plain language) | Default |
|------|-----|------------|--------------------------|---------|
| **A** | `A:yig_mgo` | 0.90 | Yig mgo ༄༅ at junction | On |
| **B** | `B:closing+break` | 0.78–0.80 | Closing formula + shad + **double newline or digit line** | recall/balanced only |
| **C** | `C:page_numbers` | 0.70 | Digit line after shad (gated on opener/closing nearby) | Off |
| **C weak** | `C:page_numbers_weak` | 0.40 | Bare digit lines (recall-only variant) | Off |
| **D** | `D:opening_formula` | 0.60–0.70 | Opening formula without yig mgo at junction | Off |
| **E** | `E:collection_title` | 0.50 | Collection title on its own line | Off |
| **F** | — | — | Removed (`བཞུགས་སོ` is usually opening, not closing) | — |
| **G** | `G:sbrul_shad` | 0.82 | ༈ section opener when guarded | On |
| **G weak** | `G:sbrul_shad_weak` | 0.45 | Unguarded ༈ | Off |
| **H** | `H:sanskrit_closing` | 0.72 | Sanskrit closing + shads before ༈/༄༅ | On |

After candidate generation, predictions below `min_confidence` are dropped, then nearby candidates are merged (highest confidence wins within `merge_window` chars).

## Page layout rules (I–L)

> **Status: implemented, opt-in.** Rules A–H above operate on a flat character
> string and look at *orthographic* signals (yig mgo, shad, closing formulas).
> Rules I–L instead look at *page structure* — how many lines each page has —
> and are aimed at **OCR output**, where a volume arrives as a sequence of
> pages. They are **off by default** and only meaningful when the input can be
> segmented into pages. Segmentation lives in
> [`src/outline_detection/page_layout.py`](../src/outline_detection/page_layout.py);
> the rules themselves are in `detector.py` (`_layout_candidates`).

### Why a separate family

OCR rarely gives clean orthographic boundaries, but it does preserve **page
layout**. A text break very often lines up with a page that is mostly empty or
much shorter than its neighbours (end of one text, start of the next). These
rules turn that intuition into boundary candidates.

They require **page-segmented input** (see "Page input" below). When the text
cannot be split into more than one page, these rules emit **no candidates**, so
continuous-stream input behaves exactly as it does today.

### Rule summary

| Rule | ID | Trigger (let `T` = `line_threshold`, default 4) | Confidence | Default |
|------|-----|-------------------------------------------------|------------|---------|
| **I** | `I:empty_page` | A page has **0 non-empty lines** | 0.75 | Off |
| **J** | `J:sparse_tail` | Page N has `nb_lines > T`, **N+1 `< T`**, **N+2 `< T`** → break **before N+2** | 0.65 | Off |
| **K** | `K:sparse_island` | Page N `> T`, **N+1 `< T`**, **N+2 `> T`** → *ambiguous*: emits **two** candidates (before N+1 and before N+2) | 0.45 each | Off |
| **L** | `L:modern_publication` | Reserved for modern-print heuristics (running headers/footers, title pages) | TBD | Off |

**Boundary position:** the character offset at the **start of the first line of
the target page** — same convention as Rule A (boundary = start of the new
text).

**Line counting:** only **non-empty** lines are counted. When
`ignore_page_number_lines` is on (default), lines that are *only* a folio/page
number are not counted either, since OCR commonly places the number alone on a
line.

### The ambiguous case (Rule K)

For the pattern `N > T`, `N+1 < T`, `N+2 > T` the break is either between N and
N+1 or between N+1 and N+2 — the line counts alone cannot decide. Rule K emits
**both** candidates at **low confidence (0.45)**, so on the `balanced` and
`precision` profiles they fall below `min_confidence` unless a nearby
orthographic rule (A / G / H) agrees and wins the merge. Set
`rule_k_sparse_island=False` to suppress this case entirely and keep only the
unambiguous Rules I/J.

### Page input

Pages are detected from the raw text via a configurable delimiter
(`page_delimiter`):

- **Form feed `\f`** (default) — the standard page separator in many OCR
  exports.
- **Blank-line fallback** — N consecutive blank lines when no `\f` is present.
- **Custom** — an explicit marker or regex.

### Enabling page layout rules

```python
from outline_detection import RuleBasedDetector

det = RuleBasedDetector(
    profile="balanced",
    rule_i_empty_page=True,
    rule_j_sparse_tail=True,
    rule_k_sparse_island=False,  # skip the ambiguous case
    line_threshold=4,
    page_delimiter="\f",
)
```

Or from the CLI (`detect` and `predict`):

```bash
outline-detect detect volume.txt \
    --rule-i-empty-page --rule-j-sparse-tail \
    --line-threshold 4 --page-delimiter ff
```

`--page-delimiter` accepts `ff` (form feed, default), `blank` / `blankN` (split
on N blank lines), or a raw regex. Pass `--no-ignore-page-number-lines` to count
lone folio numbers as page lines.

### Worked examples (input → output)

All examples use `profile="balanced"` and `line_threshold=4` (so a page is
"sparse" when it has 1–3 lines and "dense" when it has 5+). `\f` is the form
feed (page break). The returned `breakpoints` are character offsets; `details`
adds the confidence and firing rule.

**Rule I — an empty page.** The two `\f\f` create an empty middle page, so the
boundary lands at the start of the following page.

```python
detect_breakpoints(
    "page one line a\npage one line b\f\fstart of next text",
    profile="balanced", rule_i_empty_page=True, detailed=True,
)
# {'breakpoints': [33],
#  'details': [{'index': 33, 'confidence': 0.75, 'rule': 'I:empty_page'}]}
# index 33 == start of "start of next text"
```

**Rule J — dense, then two sparse pages.** Page 1 has 6 lines, pages 2 and 3
have 2 and 1 lines; the break is placed before page 3.

```python
a = "\n".join(f"dense line {i}" for i in range(6))   # 6 lines (dense)
b = "short a\nshort b"                                  # 2 lines (sparse)
c = "tail line only"                                    # 1 line  (sparse)
detect_breakpoints("\f".join([a, b, c]),
                   profile="balanced", rule_j_sparse_tail=True, detailed=True)
# {'breakpoints': [94],
#  'details': [{'index': 94, 'confidence': 0.65, 'rule': 'J:sparse_tail'}]}
# index 94 == start of "tail line only"
```

**Rule K — a sparse page between two dense pages (ambiguous).** Two
low-confidence candidates are emitted (before the sparse page and before the
next dense page); on `balanced`/`precision` they survive only if no stronger
orthographic rule overrides them in the merge.

```python
a = "\n".join(f"dense line {i}" for i in range(6))         # dense
b = "lonely sparse page line that is fairly long here"     # 1 line (sparse)
c = "\n".join(f"again dense {i}" for i in range(6))         # dense
detect_breakpoints("\f".join([a, b, c]),
                   profile="balanced", rule_k_sparse_island=True, detailed=True)
# {'breakpoints': [78, 127],
#  'details': [{'index': 78,  'confidence': 0.45, 'rule': 'K:sparse_island'},
#              {'index': 127, 'confidence': 0.45, 'rule': 'K:sparse_island'}]}
```

**Continuous text — rules stay inert.** With no page breaks the text is a single
page, so layout rules emit nothing even when enabled (orthographic Rules A–H
still run as usual).

```python
detect_breakpoints("just one\nblock of\ntext with no page breaks",
                   profile="balanced",
                   rule_i_empty_page=True, rule_j_sparse_tail=True)
# {'breakpoints': []}
```

The same four cases run from the CLI by writing the text to a file and passing
the matching flags, e.g.:

```bash
outline-detect detect volume.txt --rule-j-sparse-tail --detailed --pretty
```

These rules are **not** part of any profile preset (like Rule C, they are
net-neutral on the orthographic snippet benchmark, which is **not**
page-structured). Evaluating their effect needs a page-aligned eval set; a small
regression fixture lives at
[`data/samples/page_layout_cases.txt`](../data/samples/page_layout_cases.txt).

## Profile presets

| Setting | recall | balanced | precision |
|---------|--------|----------|-----------|
| `min_confidence` | 0.30 | 0.40 | 0.50 |
| `merge_window` | 25 | 20 | 15 |
| Rule B | yes | yes | no |
| Rules C, D, E | no | no | no |
| Unguarded G / bare C | no | no | no |

## Evaluation notes (unique corpus, ±15 chars)

On `data/breakpoints_context_snippets_unique.json` (31,591 snippets), after disabling net-negative rules D/C/E:

- **Balanced** reaches ~**60.1% F1** (~63% precision, ~57.5% recall).
- Primary contributors: **A** (yig mgo) and **G** (༈).
- **B** is slightly net-negative per-rule but retained on recall/balanced because it recovers boundaries A/G miss.
- **D** was the largest source of false alarms (~5.9k wrong vs ~570 correct on balanced before disable).

Regenerate full tables:

```bash
outline-detect evaluate data/breakpoints_context_snippets_unique.json --all-profiles --tolerance 15
```

Output: `reports/evaluations/rule_based_evaluation_unique.md`

## Re-enabling rules for experiments

```python
from outline_detection import RuleBasedDetector

det = RuleBasedDetector(profile="balanced")
det.use_rule_d = True   # opening formulas (high false-alarm rate)
det.use_rule_c = True   # page numbers
```

Or pass a custom profile dict by editing `PROFILE_PRESETS` in `src/outline_detection/detector.py`.

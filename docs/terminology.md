# Terminology

## Text boundaries

A **boundary** is the character position where one Tibetan text ends and another begins (e.g. end of one work in a collected volume, start of the next).

In training and evaluation data, boundaries are marked with an HTML-style tag:

- `</b>` — end of text A / start of text B (the detector predicts this offset)
- `<b>` — alternate opening marker (also accepted when loading data)

Each **snippet** is a short window of characters around one annotated boundary, with context on both sides.

## Tibetan boundary signals

| Term | Tibetan | Role |
|------|---------|------|
| **Yig mgo** | ༄༅ | Strong opener; often marks the start of a new text after a boundary |
| **Sbrul shad** | ༈ | Section / chapter opener; common at boundaries when guarded by prior closing |
| **Shad** | ། | Sentence/clause ender |
| **Nyis shad** | ༎ | Emphasis shad |
| **Tsheg** | ་ | Syllable separator |
| **Gter tsheg** | ༔ | Terma punctuation |

### Closing vs opening formulas

- **Closing formulas** appear *before* the boundary (end of text A): e.g. མངྒ་ལཾ, དགེའོ, རྫོགས་སོ, Sanskrit blessings.
- **Opening formulas** appear *after* the boundary (start of text B): e.g. བཞུགས་སོ, ན་མོ, རྒྱ་གར་སྐད་དུ, ༄༅.

Rule D targeted opening formulas without yig mgo; it was disabled after evaluation showed very low precision (~9%).

## Junction context

Rules anchor openers with **JUNCTION_BEFORE**:

```text
(?:^|\n|[།༎༔]\s{0,3})
```

This allows a match after a newline, start of string, or shad/tsheg with optional space—not only after a hard line break (~21% of openers follow `།།` on the same line).

## Page layout terms

Used by the optional page-layout rules (I–L) for OCR input. See [rules.md](rules.md).

| Term | Meaning |
|------|---------|
| **Page** | A slice of the source text between two page delimiters. Represented by the `Page` dataclass in [`page_layout.py`](../src/outline_detection/page_layout.py) with its character `start`/`end` offsets. |
| **Page delimiter** | The marker that separates pages. Default is the form feed `\f`; a blank-line fallback (`blank` / `blankN`) or a custom regex can be used instead. |
| **`nb_lines`** | The number of **non-empty** lines on a page after optional page-number filtering — the figure Rules J/K compare against `line_threshold` (`T`, default 4). |
| **Page-number line** | A line containing only digits (Latin or Tibetan ༠–༩), whitespace, and dashes (e.g. `449`, `1-70`, `-`). Skipped from `nb_lines` when `ignore_page_number_lines` is on. |
| **Empty page** | A page with `nb_lines == 0`; a strong text-break signal (Rule I). |

## Evaluation metrics

| Metric | Meaning |
|--------|---------|
| **Precision** | Of all predicted boundaries, how many are within tolerance of a true boundary? |
| **Recall** | Of all true boundaries, how many have a matching prediction nearby? |
| **F1** | Harmonic mean of precision and recall |
| **Tolerance** | Default ±15 characters: a prediction counts as correct if it falls within this distance of the annotation |

### Profiles

| Profile | Intent |
|---------|--------|
| **recall** | Lower confidence threshold, wider merge window; more guesses |
| **balanced** | Default for most use cases |
| **precision** | Higher threshold, narrower merge; fewer rules (B disabled) |

See [rules.md](rules.md) for per-profile flags.

# Workflow

Install the package first (from the repository root):

```bash
pip install -e ".[crf]"
```

Commands write reports under `./reports/` relative to your current directory, so run them from the repository root to keep outputs together.

## 1. Detect boundaries in text

The primary use case — text in, boundary start indices out:

```bash
outline-detect detect mytext.txt
# {"breakpoints": [0, 152, 410]}
```

Or from Python:

```python
from outline_detection import detect_breakpoints
detect_breakpoints(open("mytext.txt", encoding="utf-8").read())
```

Read from stdin with `-`, or pass inline text with `--text`. Add `--detailed` to include each boundary's confidence and firing rule, `--pretty` to indent JSON, and `-o out.json` to write to a file.

## 2. Pattern analysis (optional)

Discover boundary statistics on annotated data before tuning rules:

```bash
outline-detect analyze data/breakpoints_context_snippets_unique.json --window 200
```

Writes `reports/analysis/boundary_report_unique.md` and `.json`.

## 3. Evaluate the rule-based detector

Recommended default (deduplicated corpus):

```bash
outline-detect evaluate data/breakpoints_context_snippets_unique.json --profile balanced --tolerance 15
```

Compare all profiles and write a combined report:

```bash
outline-detect evaluate data/breakpoints_context_snippets_unique.json --all-profiles --tolerance 15
```

Outputs:

| Path | Content |
|------|---------|
| `reports/evaluations/rule_based_evaluation_unique.md` | Precision, recall, F1, per-rule breakdown |
| `reports/diagnostics/false_negatives.json` | Missed boundaries (last single-profile run) |

Custom report path: add `--report reports/evaluations/my_run.md`.

## 4. Inspect false negatives

Open `reports/diagnostics/false_negatives.json` after a single-profile evaluate run. Each entry has `sample_idx`, `position`, and left/right context strings. Use it to decide whether to adjust rules or train a CRF.

## 5. Annotate new text

```bash
outline-detect predict data/samples/INPUT.txt --profile balanced --output reports/predicted_boundaries.txt
```

This inserts `<b>` markers at predicted positions (use `detect` instead if you just want the index list).

## 6. CRF baseline (optional, needs [crf] extra)

```bash
outline-detect crf train data/breakpoints_context_snippets_unique.json --folds 5 --tolerance 15
outline-detect crf train data/breakpoints_context_snippets_unique.json --save-model
outline-detect crf predict data/samples/INPUT.txt --model reports/models/boundary_crf.pkl
```

## Data files

| File | Snippets | Notes |
|------|----------|-------|
| `data/breakpoints_context_snippets.json` | 82,560 | Full annotated corpus |
| `data/breakpoints_context_snippets_unique.json` | 31,591 | Deduplicated; faster iteration |
| `data/samples/` | — | Optional raw text for `detect` / `predict` |

## Input formats

Annotated data for `evaluate`, `analyze`, and `crf train`:

- **JSON**: list of strings with `<b>` or `</b>` markers
- **JSONL**: one JSON string per line
- **TXT**: one snippet per paragraph, or blocks separated by `---` / `===` / `***`

# Tibetan Text Boundary Detection

`outline_detection` detects where one Tibetan text ends and another begins, using pattern rules (yig mgo ༄༅, section mark ༈, closing phrases) and an optional CRF sequence labeler. Built from analysis of tens of thousands of annotated boundary snippets.

## Install

```bash
pip install -e .            # core (rule-based detection + evaluation)
pip install -e ".[crf]"     # also install CRF extras (scikit-learn, sklearn-crfsuite)
```

Requires Python 3.9+. (`pip install -r requirements.txt` does the editable `[crf]` install.)

## Python API

```python
from outline_detection import detect_breakpoints

text = "...རྫོགས་སོ།། ༄༅། །next text..."
detect_breakpoints(text)
# {"breakpoints": [0, 152, 410, ...]}
```

`detect_breakpoints` returns a dict with key `breakpoints` whose value is the list of boundary **start indices** (character offsets) found by the rule-based detector.

Options:

```python
detect_breakpoints(text, profile="precision")          # recall | balanced | precision
detect_breakpoints(text, min_confidence=0.5)            # override threshold
detect_breakpoints(text, detailed=True)                 # adds per-boundary confidence + rule
```

## CLI

The install provides an `outline-detect` command.

**Detect** boundaries (text in -> JSON out):

```bash
outline-detect detect mytext.txt
# {"breakpoints": [0, 152, 410]}

echo "..." | outline-detect detect -            # read from stdin
outline-detect detect --text "རྫོགས་སོ།། ༄༅། །next" --pretty
outline-detect detect mytext.txt -o result.json
```

**Evaluate** against annotated data:

```bash
outline-detect evaluate data/breakpoints_context_snippets_unique.json --profile balanced --tolerance 15
outline-detect evaluate data/breakpoints_context_snippets_unique.json --all-profiles --tolerance 15
```

**Analyze** boundary patterns:

```bash
outline-detect analyze data/breakpoints_context_snippets_unique.json
```

**Annotate** a raw file with boundary markers:

```bash
outline-detect predict data/samples/INPUT.txt --profile balanced
```

**CRF** (requires the `[crf]` extra):

```bash
# Full-corpus train with feature cache and post-train eval
outline-detect crf train data/breakpoints_context_snippets.json \
  --save-model --features-cache reports/models/crf_features.pkl \
  --eval-file data/breakpoints_context_snippets_unique.json

# Evaluate a saved model
outline-detect crf evaluate data/breakpoints_context_snippets_unique.json \
  --model reports/models/boundary_crf.pkl --tolerance 15

outline-detect crf predict data/samples/INPUT.txt --model reports/models/boundary_crf.pkl
```

## Data

| File | Size |
|------|------|
| `data/breakpoints_context_snippets.json` | 82,560 annotated snippets |
| `data/breakpoints_context_snippets_unique.json` | 31,591 deduplicated snippets |
| `data/samples/` | Optional raw `.txt` files for prediction demos |

Boundaries in annotated JSON are marked with `</b>` (or `<b>`).

**Hugging Face Hub:**

| Resource | Repo |
|----------|------|
| Full snippets (82,560) | [ganga4364/tibetan-outline-boundary-snippets-full](https://huggingface.co/datasets/ganga4364/tibetan-outline-boundary-snippets-full) |
| Unique benchmark (31,591) | [ganga4364/tibetan-outline-boundary-snippets-unique](https://huggingface.co/datasets/ganga4364/tibetan-outline-boundary-snippets-unique) |
| CRF full (production) | [ganga4364/tibetan-outline-boundary-crf-full](https://huggingface.co/ganga4364/tibetan-outline-boundary-crf-full) |
| CRF unbiased (honest eval) | [ganga4364/tibetan-outline-boundary-crf-unbiased](https://huggingface.co/ganga4364/tibetan-outline-boundary-crf-unbiased) |

```bash
hf download ganga4364/tibetan-outline-boundary-snippets-unique --repo-type dataset
hf download ganga4364/tibetan-outline-boundary-crf-unbiased boundary_crf.pkl --local-dir ./reports/models
```

## Outputs

`evaluate`, `analyze`, `predict`, and `crf` write under `./reports/` (relative to where you run the command; gitignored except `.gitkeep`):

| Directory | Contents |
|-----------|----------|
| `reports/evaluations/` | `rule_based_evaluation_*.md` |
| `reports/analysis/` | `boundary_report_*.md` / `.json` |
| `reports/diagnostics/` | `false_negatives.json` |
| `reports/models/` | CRF `.pkl` models |
| `reports/` | `predicted_boundaries.txt`, `crf_predicted.txt` |

## Results (unique corpus, ±15 chars)

| Method | F1 |
|--------|-----|
| Rule-based (balanced) | **0.601** |
| CRF full | 0.571 |
| CRF unbiased | 0.555 |

Rule-based **balanced** reaches ~63% precision and ~57.5% recall. Primary active rules: **A** (yig mgo) and **G** (༈). See [docs/evaluation.md](docs/evaluation.md) for full comparison and regeneration commands.

## Documentation

- [docs/terminology.md](docs/terminology.md) — Tibetan signals, markup, metrics
- [docs/rules.md](docs/rules.md) — Rules A–H and profile presets
- [docs/workflow.md](docs/workflow.md) — Full step-by-step workflow
- [docs/huggingface.md](docs/huggingface.md) — Hub datasets and models
- [docs/evaluation.md](docs/evaluation.md) — Benchmark results
- [docs/README.md](docs/README.md) — Doc index
- [CHANGELOG.md](CHANGELOG.md) — Release notes

## Repository layout

```
├── pyproject.toml
├── requirements.txt
├── src/
│   └── outline_detection/   # api, cli, detector, evaluation, analyzer, crf, utils, paths
├── data/                    # Annotated JSON corpora and samples/
├── docs/                    # Static reference
├── scripts/                 # Training, comparison, and Hub upload helpers
└── reports/                 # Generated outputs (gitignored)
```

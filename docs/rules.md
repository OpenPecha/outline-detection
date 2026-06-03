# Detection rules (A–H)

Rules are implemented in [`src/outline_detection/detector.py`](../src/outline_detection/detector.py). Formula lists and `JUNCTION_BEFORE` live in [`src/outline_detection/utils.py`](../src/outline_detection/utils.py).

## Rule summary

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

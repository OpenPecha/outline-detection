# Evaluation results

Benchmark: **unique** corpus (`breakpoints_context_snippets_unique.json`, 31,591 snippets).  
Tolerance: **±15 characters**.

## Rule-based vs CRF (F1)

| Method | Variant | Precision | Recall | F1 |
|--------|---------|-----------|--------|-----|
| Rule-based | balanced | 0.630 | 0.575 | **0.601** |
| CRF | full model | 0.718 | 0.475 | 0.571 |
| CRF | unbiased model | 0.752 | 0.440 | 0.555 |

On this benchmark, **rule-based detection (balanced) beats both CRF models on F1**. CRF full trades recall for precision; CRF unbiased is the honest eval variant (unique strings held out during training).

## Regenerate reports

```bash
# Rule-based (all profiles)
outline-detect evaluate data/breakpoints_context_snippets_unique.json --all-profiles --tolerance 15

# CRF full model
outline-detect crf evaluate data/breakpoints_context_snippets_unique.json \
  --model reports/models/boundary_crf.pkl --tolerance 15

# CRF unbiased model
outline-detect crf evaluate data/breakpoints_context_snippets_unique.json \
  --model reports/models/unbiased/boundary_crf.pkl --tolerance 15

# Combined comparison markdown
python scripts/compare_rule_crf.py
```

Outputs land in `reports/evaluations/` (gitignored).

## Sample prediction (INPUT.txt)

| Method | Boundaries found |
|--------|-----------------|
| Rule (balanced) | 10 |
| CRF (full) | 2 |

Regenerate with `outline-detect predict` and `outline-detect crf predict`.

## Models and data

Download corpora and models from Hugging Face — see [huggingface.md](huggingface.md).

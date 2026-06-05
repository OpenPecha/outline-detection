# Hugging Face Hub

Annotated corpora and trained CRF models are published as **four separate repositories** under `ganga4364/`.

## Dataset repos

| Repo | Rows | Use |
|------|------|-----|
| [tibetan-outline-boundary-snippets-full](https://huggingface.co/datasets/ganga4364/tibetan-outline-boundary-snippets-full) | 82,560 | Full training corpus (duplicates allowed) |
| [tibetan-outline-boundary-snippets-unique](https://huggingface.co/datasets/ganga4364/tibetan-outline-boundary-snippets-unique) | 31,591 | Deduplicated benchmark for evaluation |

Download to `data/`:

```bash
hf download ganga4364/tibetan-outline-boundary-snippets-full breakpoints_context_snippets.json --repo-type dataset --local-dir ./data
hf download ganga4364/tibetan-outline-boundary-snippets-unique breakpoints_context_snippets_unique.json --repo-type dataset --local-dir ./data
```

Load with the `datasets` library:

```python
from datasets import load_dataset

unique = load_dataset("ganga4364/tibetan-outline-boundary-snippets-unique", split="train")
snippets = unique["snippet"]
```

## Model repos

| Repo | Training data | When to use |
|------|---------------|-------------|
| [tibetan-outline-boundary-crf-full](https://huggingface.co/ganga4364/tibetan-outline-boundary-crf-full) | Full 82,560 snippets | Production — best recall |
| [tibetan-outline-boundary-crf-unbiased](https://huggingface.co/ganga4364/tibetan-outline-boundary-crf-unbiased) | Full minus unique (~50,880) | Honest benchmark eval (no leakage) |

Both repos ship the model as `boundary_crf.pkl`:

```bash
hf download ganga4364/tibetan-outline-boundary-crf-full boundary_crf.pkl --local-dir ./reports/models
hf download ganga4364/tibetan-outline-boundary-crf-unbiased boundary_crf.pkl --local-dir ./reports/models/unbiased
```

## Re-uploading

To refresh Hub assets after retraining:

```bash
HF_TOKEN=... python scripts/push_hub_repos.py
```

Requires `pip install datasets huggingface_hub` and a write token in the `HF_TOKEN` environment variable.

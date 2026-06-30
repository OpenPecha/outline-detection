# Data

Large annotated corpora are **not stored in git** (GitHub file-size limits). Download from Hugging Face Hub:

```bash
# Full training corpus (82,560 snippets)
hf download ganga4364/tibetan-outline-boundary-snippets-full \
  breakpoints_context_snippets.json --repo-type dataset --local-dir .

# Unique benchmark (31,591 snippets)
hf download ganga4364/tibetan-outline-boundary-snippets-unique \
  breakpoints_context_snippets_unique.json --repo-type dataset --local-dir .
```

Place the JSON files in this directory (`data/`). Each file is a JSON array of strings with `</b>` boundary markers.

**Sample raw text** for `detect` / `predict` demos: `samples/INPUT.txt`, `samples/INPUT1.txt`.

See [docs/huggingface.md](../docs/huggingface.md) for model downloads and [docs/workflow.md](../docs/workflow.md) for usage.

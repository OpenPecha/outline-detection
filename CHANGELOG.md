# Changelog

## Unreleased

### Added

- CRF **feature cache**: save/load prepared sequences via `--features-cache` to skip re-extraction on large corpora.
- `outline-detect crf evaluate` subcommand for evaluating saved `.pkl` models.
- Post-train evaluation on a held-out file via `--eval-file` / `--eval-report` on `crf train`.
- Helper scripts under `scripts/` for training, rule-vs-CRF comparison, and Hugging Face Hub upload.
- Documentation: [docs/huggingface.md](docs/huggingface.md), [docs/evaluation.md](docs/evaluation.md).

### Published

- Four separate Hugging Face repos (full/unique datasets; full/unbiased CRF models) under `ganga4364/`.

### Changed

- README and workflow docs updated with Hub download commands and CRF training workflow.

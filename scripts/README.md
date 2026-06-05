# Scripts

Helper scripts for CRF training, evaluation comparison, and Hugging Face Hub uploads. Run from the repository root unless noted.

| Script | Purpose |
|--------|---------|
| `run_crf_train.sh` | Linux/macOS wrapper for full CRF training |
| `run_crf_train.ps1` | Windows wrapper for full CRF training |
| `fetch_crf_artifacts_vastai2.ps1` | Download trained models and logs from a vast.ai host |
| `compare_rule_crf.py` | Build `reports/evaluations/rule_vs_crf_unique.md` from eval reports |
| `push_hub_repos.py` | Upload 4 separate HF dataset/model repos (requires `HF_TOKEN`) |

## Examples

```bash
# Compare rule-based vs CRF evaluation reports
python scripts/compare_rule_crf.py

# Upload datasets and models to Hugging Face Hub
HF_TOKEN=... python scripts/push_hub_repos.py
```

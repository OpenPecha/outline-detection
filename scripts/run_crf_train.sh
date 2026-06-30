#!/usr/bin/env bash
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p reports/logs reports/models
LOG="reports/logs/crf_train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG"
echo "Started at: $(date -Iseconds)" | tee -a "$LOG"
"$ROOT/.venv/Scripts/outline-detect.exe" crf train \
  data/breakpoints_context_snippets.json \
  --folds 5 \
  --tolerance 15 \
  --save-model 2>&1 | tee -a "$LOG"
echo "DONE at: $(date -Iseconds)" | tee -a "$LOG"

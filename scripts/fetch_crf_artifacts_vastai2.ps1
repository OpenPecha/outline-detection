$ErrorActionPreference = "Stop"
$root = "c:\Users\GANGA GYATSO\work\Outline_detection"
Set-Location $root

New-Item -ItemType Directory -Force -Path "$root\reports\models" | Out-Null
New-Item -ItemType Directory -Force -Path "$root\reports\logs" | Out-Null

scp vastai2:~/crf_training/reports/models/boundary_crf.pkl "$root\reports\models\boundary_crf.pkl"
scp vastai2:~/crf_training/reports/logs/crf_train_full.log "$root\reports\logs\crf_train_full.log"
scp vastai2:~/crf_training/reports/crf_eval_breakpoints_context_snippets_unique.md "$root\reports\" -ErrorAction SilentlyContinue

Write-Host "Artifacts copied to reports/"

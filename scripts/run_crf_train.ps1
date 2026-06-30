$ErrorActionPreference = "Continue"
$root = "c:\Users\GANGA GYATSO\work\Outline_detection"
Set-Location $root

$ts  = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "$root\reports\logs\crf_train_$ts.log"

New-Item -ItemType Directory -Force -Path "$root\reports\logs"  | Out-Null
New-Item -ItemType Directory -Force -Path "$root\reports\models" | Out-Null

$exe = "$root\.venv\Scripts\outline-detect.exe"
$data = "$root\data\breakpoints_context_snippets.json"

"========================================" | Tee-Object -FilePath $log
"CRF Training Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $log -Append
"Log: $log" | Tee-Object -FilePath $log -Append
"========================================" | Tee-Object -FilePath $log -Append

& $exe crf train $data --folds 5 --tolerance 15 --save-model 2>&1 |
    Tee-Object -FilePath $log -Append

"========================================" | Tee-Object -FilePath $log -Append
"CRF Training DONE: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $log -Append
"========================================" | Tee-Object -FilePath $log -Append

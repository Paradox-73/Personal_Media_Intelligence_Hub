# Full pipeline run (ingest + retrain + 50-fold experiments). Run detached:
#   Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\run_full_pipeline.ps1"
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START full pipeline" | Out-File full_run.log -Encoding utf8
& $py -u run_pipeline.py --full 2>&1 | Out-File full_run.out.log -Encoding utf8
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') pipeline exit $LASTEXITCODE" | Out-File full_run.log -Append -Encoding utf8

# Full retrain (movies + TV) after the 2026-10-02 fixes, then the train/serve
# parity audit. Run detached:
#   Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\run_retrain_and_audit.ps1"
$env:PYTHONUTF8 = "1"
$py = ".venv\Scripts\python.exe"
"=== $(Get-Date -Format 'HH:mm:ss') START pipeline" | Out-File retrain.log -Encoding utf8
& $py run_pipeline.py --no-ingest --domains "movies,shows" *> retrain.out.log
"=== $(Get-Date -Format 'HH:mm:ss') pipeline exit $LASTEXITCODE" | Out-File retrain.log -Append -Encoding utf8
& $py scripts\audit_inference_parity.py *> audit.out.log
"=== $(Get-Date -Format 'HH:mm:ss') audit exit $LASTEXITCODE; ALL DONE" | Out-File retrain.log -Append -Encoding utf8

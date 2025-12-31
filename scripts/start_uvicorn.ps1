$p = Start-Process -FilePath python -ArgumentList '-m uvicorn main:app --host 127.0.0.1 --port 8000' -PassThru
$p.Id | Out-File -FilePath uvicorn.pid -Encoding ascii
Write-Output "UVICORN_STARTED PID=$($p.Id)"

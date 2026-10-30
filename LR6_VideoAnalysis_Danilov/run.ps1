Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path results | Out-Null

& {
    python .\pipeline.py
    ""
    python .\topics.py
    ""
    python .\report.py
} 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable runLog

$runLog | Out-File -FilePath results\run_log.txt -Encoding utf8

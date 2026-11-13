Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path results | Out-Null

python .\recsys.py --dataset movielens --model all --k 5 10 20 --evaluate --eda --output results 2>&1 |
    ForEach-Object { "$_" } | Tee-Object -Variable runLog

$runLog | Out-File -FilePath results\run_log.txt -Encoding utf8

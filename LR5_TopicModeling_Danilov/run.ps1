Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path results | Out-Null

& {
    python .\topic_modeling.py --dataset news --method both --topics auto --output news_topics
    ""
    python .\topic_modeling.py --dataset products --method nmf --topics 15 --output product_categories
    ""
    python .\topic_modeling.py --dataset mixed --method both --topics 10 --compare --output mixed_compare
    ""
    python .\cities_clustering.py
} 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable runLog

$runLog | Out-File -FilePath results\run_log.txt -Encoding utf8

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$inputPath = Join-Path $root "outputs\01a0225a-43b2-7991-a2c3-00e9e4064c5d\aggregation_dedicated_daily_outcomes_last30d.csv"

if (-not (Test-Path -LiteralPath $inputPath)) {
    throw "缺少运行输入：$inputPath。请先按 README 放入最新节点—业务日明细。"
}

Push-Location $root
try {
    python score_recent7d_businesses_v4.py
}
finally {
    Pop-Location
}

# Wrapper: download orchestrator-on-edge assets (see docs/ASSETS.md).
# Usage: .\scripts\fetch_assets.ps1 --all
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
    & $py (Join-Path $PSScriptRoot "fetch_assets.py") @Args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

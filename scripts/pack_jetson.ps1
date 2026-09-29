# Build a ready-to-extract Jetson (arm64) deployment zip.
# Usage: .\scripts\pack_jetson.ps1 --out dist --name orchestrator-on-edge
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    $py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
    & $py (Join-Path $PSScriptRoot "pack_jetson.py") @Args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

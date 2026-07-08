param(
    [string]$ImageDir = ".\workflow\images",
    [string]$PageMdRoot = ".\workflow\stage3_page_md",
    [string]$Stage3Out = ".\workflow\stage3_out"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage1.ps1 -ImageDir $ImageDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage2.ps1 -ImageDir $ImageDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_stage3.ps1 -PagesRoot ".\workflow\stage2_match" -PageMdRoot $PageMdRoot -OutRoot $Stage3Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

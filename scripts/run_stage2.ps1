param(
    [string]$ImageDir = ".\workflow\images",
    [string]$RfdetrJsonl = ".\workflow\stage1_rfdetr\rfdetr_infer_results.jsonl",
    [string]$DoclayoutJsonDir = ".\workflow\stage1_doclayout\json",
    [string]$OutDir = ".\workflow\stage2_match"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".conda\messtoclean\python.exe"
Set-Location $root

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (Test-Path $OutDir) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $python -m match.match `
  --rfdetr-jsonl $RfdetrJsonl `
  --doclayout-json-dir $DoclayoutJsonDir `
  --pages-root $ImageDir `
  --output-dir $OutDir `
  --match-algo v2 `
  --match-backend flow `
  --ro-overlap-mode keep_large `
  --match-overlap-mode keep_large `
  --save-viz `
  --draw-edges `
  --max-edges-per-question 2 `
  --max-fig-per-question 2 `
  --q-pad-ratio 0.02
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

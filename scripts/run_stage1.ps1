param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$RfdetrOut = ".\workflow\code_outputs\rfdetr",
    [string]$DoclayoutOut = ".\workflow\code_outputs\doclayout",
    [string]$Checkpoint = ".\checkpoint_best_total.pth",
    [string]$DoclayoutDevice = "cpu",
    [string]$DoclayoutModelDir = "",
    [switch]$OptimizeRfdetr
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:HOME = $root
$env:USERPROFILE = $root
$env:HF_HOME = Join-Path $root ".cache\huggingface"
$env:TRANSFORMERS_CACHE = Join-Path $root ".cache\huggingface\transformers"
$env:MODELSCOPE_CACHE = Join-Path $root ".cache\modelscope"
$env:PADDLE_HOME = Join-Path $root ".cache\paddle"
$env:PADDLEOCR_HOME = Join-Path $root ".cache\paddleocr"
$env:XDG_CACHE_HOME = Join-Path $root ".cache"
$env:LOG_LEVEL = "ERROR"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:MODELSCOPE_CACHE, $env:PADDLE_HOME, $env:PADDLEOCR_HOME | Out-Null

$python = Join-Path $root ".conda\messtoclean\python.exe"
Set-Location $root

if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
    throw "Image not found: $Image"
}

if (Test-Path $RfdetrOut) {
    Remove-Item -LiteralPath $RfdetrOut -Recurse -Force
}
if (Test-Path $DoclayoutOut) {
    Remove-Item -LiteralPath $DoclayoutOut -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $RfdetrOut, $DoclayoutOut | Out-Null

$rfdetrArgs = @(
  "-m", "match.rfdetr_infer",
  "--image-path", $Image,
  "--checkpoint", $Checkpoint,
  "--output-dir", $RfdetrOut,
  "--overwrite-jsonl",
  "--clean-output",
  "--num-classes", "4"
)
if ($OptimizeRfdetr) {
    $rfdetrArgs += "--optimize-for-inference"
}

& $python @rfdetrArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$doclayoutArgs = @(
  "-m", "match.doclayout_infer",
  "--image-path", $Image,
  "--output-dir", $DoclayoutOut,
  "--model-name", "PP-DocLayout_plus-L",
  "--device", $DoclayoutDevice
)
if ($DoclayoutModelDir) {
    $doclayoutArgs += @("--model-dir", $DoclayoutModelDir)
}

& $python @doclayoutArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

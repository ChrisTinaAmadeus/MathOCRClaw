param(
    [string]$ImageDir = ".\workflow\images",
    [string]$RfdetrOut = ".\workflow\stage1_rfdetr",
    [string]$DoclayoutOut = ".\workflow\stage1_doclayout",
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

New-Item -ItemType Directory -Force -Path $ImageDir | Out-Null

if (Test-Path $RfdetrOut) {
    Remove-Item -LiteralPath $RfdetrOut -Recurse -Force
}
if (Test-Path $DoclayoutOut) {
    Remove-Item -LiteralPath $DoclayoutOut -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $RfdetrOut, $DoclayoutOut | Out-Null

$rfdetrArgs = @(
  "rfdetr_infer.py",
  "--image-dir", $ImageDir,
  "--checkpoint", $Checkpoint,
  "--output-dir", $RfdetrOut,
  "--overwrite-jsonl",
  "--clean-output",
  "--num-classes", "3"
)
if ($OptimizeRfdetr) {
    $rfdetrArgs += "--optimize-for-inference"
}

& $python @rfdetrArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$doclayoutArgs = @(
  "doclayout_infer.py",
  "--image-dir", $ImageDir,
  "--output-dir", $DoclayoutOut,
  "--model-name", "PP-DocLayout_plus-L",
  "--device", $DoclayoutDevice
)
if ($DoclayoutModelDir) {
    $doclayoutArgs += @("--model-dir", $DoclayoutModelDir)
}

& $python @doclayoutArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

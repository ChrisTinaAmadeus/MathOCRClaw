param(
    [string]$PagesRoot = ".\workflow\stage2_match",
    [string]$PageMdRoot = ".\workflow\stage3_page_md",
    [string]$OutRoot = ".\workflow\stage3_out",
    [switch]$Full
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
$env:VLM_DEBUG = "1"
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:MODELSCOPE_CACHE, $env:PADDLE_HOME, $env:PADDLEOCR_HOME | Out-Null

$python = Join-Path $root ".conda\messtoclean\python.exe"
Set-Location $root

$argsList = @(
  "proofread_page_v6_6.py",
  "--pages-root", $PagesRoot,
  "--page-md-root", $PageMdRoot,
  "--out-root", $OutRoot,
  "--find-pattern", "*",
  "--skip-partial",
  "--partial-main-min-hfrac", "0.12",
  "--temperature", "0.0",
  "--top-p", "0.7",
  "--use-offset-search",
  "--cache"
)

if ($Full) {
    $argsList += "--use-crop-qno"
} else {
    $argsList += @("--no-crop-qno", "--no-fig", "--no-patcher")
}

& $python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

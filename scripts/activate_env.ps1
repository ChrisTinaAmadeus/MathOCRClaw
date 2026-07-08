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
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:MODELSCOPE_CACHE, $env:PADDLE_HOME, $env:PADDLEOCR_HOME | Out-Null
$envPath = Join-Path $root ".conda\messtoclean"
if (-not (Test-Path $envPath)) {
    throw "Environment not found: $envPath"
}
& conda activate $envPath
Set-Location $root
Write-Host "Activated MessToClean env: $envPath"
Write-Host "Python: $(& python --version)"


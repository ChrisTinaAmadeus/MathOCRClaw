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
$python = Join-Path $root ".conda\messtoclean\python.exe"
Set-Location $root
& $python -c "import os; from dotenv import load_dotenv; load_dotenv('.env.local'); import requests, PIL, numpy, cv2, shapely, scipy, networkx, tqdm, rapidfuzz, jsonlines, yaml, matplotlib, skimage, sklearn, pandas, supervision, rtree, openai, torch, torchvision, rfdetr, paddle, paddleocr; print('imports ok'); print('model=' + os.environ.get('MTC_VLM_MODEL', '')); print('torch_cuda=' + str(torch.cuda.is_available())); print('paddle_cuda=' + str(paddle.device.is_compiled_with_cuda()))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }


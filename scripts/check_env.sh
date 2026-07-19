#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_env.sh"
cd -- "${MTC_ROOT}"

"${MTC_PYTHON}" - <<'PY'
import os
import platform
import sys

from dotenv import load_dotenv

load_dotenv(".env.local")

import cv2
import numpy
import openai
import paddle
import paddleocr
import PIL
import requests
import rfdetr
import supervision
import torch
import torchvision

if not sys.platform.startswith("linux"):
    raise SystemExit(f"Linux Python required, got: {sys.platform}")

print("imports=ok")
print(f"platform={platform.platform()}")
print(f"python={sys.executable}")
print(f"model={os.environ.get('MTC_VLM_MODEL', '')}")
print(f"api_key_configured={bool(os.environ.get('DASHSCOPE_API_KEY'))}")
print(f"torch_cuda={torch.cuda.is_available()}")
print(f"paddle_cuda={paddle.device.is_compiled_with_cuda()}")
PY

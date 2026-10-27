#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Run this on a supported NVIDIA GPU Linux system." >&2
  exit 1
fi

nvidia-smi -L

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv .venv --python 3.11
uv pip install --python .venv/bin/python --torch-backend=auto -r requirements.txt

# Current vLLM releases can download matched precompiled FlashInfer kernels.
# Older releases do not expose this command, so its absence is harmless.
if .venv/bin/vllm download-kernels --help >/dev/null 2>&1; then
  .venv/bin/vllm download-kernels
fi

.venv/bin/python - <<'PY'
import bitsandbytes
import torch
import transformers
import vllm

assert torch.cuda.is_available(), "PyTorch cannot see an NVIDIA CUDA GPU."
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Environment ready: $(pwd)/.venv/bin/python"

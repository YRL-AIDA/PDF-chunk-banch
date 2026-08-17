#!/usr/bin/env bash
set -euo pipefail

# Download the DeepSeek-OCR VLM snapshot to a local directory.
#
# DeepSeek-OCR is published as a full transformers repo on HuggingFace:
#   deepseek-ai/DeepSeek-OCR
# It is loaded with trust_remote_code=True, so the snapshot must include not only
# the weights (model.safetensors) and tokenizer, but also the remote code files
# (modeling_deepseekocr.py, modeling_deepseekv2.py, conversation.py,
#  configuration_deepseek_v2.py, deepencoder.py).
#
# Size: ~6.7 GB (model.safetensors) + a few MB of code/tokenizer files.

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weights"

REPO="deepseek-ai/DeepSeek-OCR"

echo "Downloading $REPO"
echo "Destination: $DEST_DIR"

mkdir -p "$DEST_DIR"

if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" --local-dir "$DEST_DIR"
else
    # Fallback: use the huggingface_hub Python API from the active interpreter.
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="$REPO", local_dir="$DEST_DIR")
print("Downloaded to", p)
PY
fi

echo "Done."

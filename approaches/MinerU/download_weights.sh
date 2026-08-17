#!/usr/bin/env bash
set -euo pipefail

# Download the MinerU VLM snapshot (layout detection) to a local directory.
#
# MinerU's VLM is a Qwen2-VL fine-tune published as
#   opendatalab/MinerU2.5-2509-1.2B
# on HuggingFace (ModelScope mirror: OpenDataLab/MinerU2.5-2509-1.2B).
# Unlike DiT / LayoutLM-v3, this model is NOT part of microsoft/unilm and is
# downloaded as a full snapshot (config.json, tokenizer, model.safetensors, ...).
#
# Size: ~2.3 GB.

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weights"

REPO="opendatalab/MinerU2.5-2509-1.2B"

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

#!/usr/bin/env bash
set -euo pipefail

# Download the pre-trained LayoutLMv3 detection weights (fine-tuned on PubLayNet)
# to a local directory. Three files are required:
#   - model_final.pth : Detectron2 checkpoint (Cascade Mask R-CNN + LayoutLMv3-base)
#   - config.json     : transformers config for the LayoutLMv3 backbone (AutoConfig)
#   - config.yaml     : full Detectron2 config used for fine-tuning
#
# The original Azure Blob links from the unilm README are no longer publicly
# accessible (HTTP 409). The same checkpoint is mirrored on HuggingFace under
# "HYPJUDY/layoutlmv3-base-finetuned-publaynet".

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weights"

REPO="HYPJUDY/layoutlmv3-base-finetuned-publaynet"
FILES=("model_final.pth" "config.json" "config.yaml")

echo "Downloading $REPO -> $DEST_DIR"
mkdir -p "$DEST_DIR"

for FILE in "${FILES[@]}"; do
    echo "  fetching $FILE ..."
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST_DIR"
    else
        python - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$DEST_DIR")
print("  ->", p)
PY
    fi
done

echo "Done."

#!/usr/bin/env bash
set -euo pipefail

# Download the pre-trained DiT weights (Mask R-CNN + DiT-base, fine-tuned on
# PubLayNet) to a local directory.
#
# The original Azure Blob links from the unilm README are no longer publicly
# accessible (HTTP 409). The same checkpoints are mirrored on HuggingFace under
# the "HYPJUDY/dit" repository, so we download from there.

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/weights"

REPO="HYPJUDY/dit"
FILE="dit-fts/publaynet_dit-b_mrcnn.pth"

echo "Downloading $REPO -> $FILE"
echo "Destination: $DEST_DIR"

mkdir -p "$DEST_DIR"

python - <<PY
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$DEST_DIR")
print("Downloaded to", p)
PY


echo "Done."

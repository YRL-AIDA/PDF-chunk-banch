import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pdf2image import convert_from_path
from PIL import Image

# =============================================================================
# Layout prompt (extracted at the top of the file on purpose).
#
# MinerU's VLM (opendatalab/MinerU2.5-2509-1.2B, a Qwen2-VL fine-tune) was
# trained on DocStructBench / MinerU's own layout schema, NOT on PubLayNet, so it
# answers with MinerU's own block-type vocabulary (text, title, doc_title,
# paragraph_title, list, list_item, table, image, chart, image_block,
# *_caption, ...). The user prompt is MinerU's trained trigger
# ("Layout Detection:"), which preserves the fine-tuned box format; the mapping
# below (`MINERU_BLOCK_TO_LABEL`) then folds that vocabulary into the five
# PubLayNet classes:
#
#   text / *_caption                    -> text    (текст, подписи)
#   title / doc_title / paragraph_title -> title   (заголовки)
#   list / list_item                    -> list    (списки)
#   table                               -> table   (таблицы)
#   image / chart / image_block         -> figure  (картинки)
#
# Blocks outside the mapping (equation, code, header/footer, ...) are dropped.
# =============================================================================
LAYOUT_SYSTEM_PROMPT = None

# MinerU's trained layout-detection trigger (kept verbatim).
LAYOUT_USER_PROMPT = "\nLayout Detection:"

# PubLayNet class order used by the other approaches (DiT, LayoutLM-v3).
CLASS_NAMES = ["text", "title", "list", "table", "figure"]

# MinerU block type -> PubLayNet label.
#
# MinerU emits its own block-type vocabulary (see mineru-vl-utils `structs.py`),
# which is wider than PubLayNet's five classes. Here we fold it into PubLayNet:
#   * captions (`*_caption` / `caption`) -> text   (подписи уходят в текст)
#   * `image` / `chart` / `image_block`  -> figure (картинки)
#   * `title` / `doc_title` / `paragraph_title` -> title (заголовки)
#   * `list` / `list_item` -> list (списки, склеиваются ниже по одной левой границе)
# Anything else (equation, code, header/footer, ...) is intentionally dropped.
MINERU_BLOCK_TO_LABEL = {
    "text": "text",
    # headings
    "title": "title",
    "doc_title": "title",
    "paragraph_title": "title",
    # lists (container blocks and individual items)
    "list": "list",
    "list_item": "list",
    # tables
    "table": "table",
    # figures (images, charts, multi-image blocks)
    "image": "figure",
    "chart": "figure",
    "image_block": "figure",
    # captions -> body text
    "caption": "text",
    "table_caption": "text",
    "image_caption": "text",
    "code_caption": "text",
}

# Consecutive list blocks with left edges aligned to within this fraction of the
# page width are merged into a single list region (same indentation = no wrap).
LIST_MERGE_LEFT_EPS = 0.02

# Image size MinerU feeds to the VLM for layout detection.
LAYOUT_IMAGE_SIZE = (1036, 1036)

# Pre-trained VLM weights. A full snapshot (config/tokenizer/safetensors) is
# downloaded by `download_weights.sh`.
HF_REPO = "opendatalab/MinerU2.5-2509-1.2B"
WEIGHTS_DIRNAME = "weights"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PDF_EXTS = {".pdf"}

# MinerU layout output uses these special tokens (registered in the tokenizer):
#   <|box_start|>x0 y0 x1 y1<|box_end|><|ref_start|>type<|ref_end|>[<|rotate_*|>]...
_LAYOUT_RE = re.compile(
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?"
    r"(.*?)(?=<\|box_start\|>|$)",
    re.DOTALL,
)


def _resolve_weights_dir(model_dir: Optional[str]) -> Path:
    """Return the directory containing the MinerU VLM snapshot."""
    if model_dir:
        base = Path(model_dir)
        if not (base / "config.json").is_file():
            raise FileNotFoundError(f"config.json not found in {base}")
        return base

    base = Path(__file__).resolve().parent / WEIGHTS_DIRNAME
    if not (base / "config.json").is_file():
        from huggingface_hub import snapshot_download

        base.mkdir(parents=True, exist_ok=True)
        base = Path(
            snapshot_download(repo_id=HF_REPO, local_dir=str(base))
        )
    if not (base / "config.json").is_file():
        raise FileNotFoundError(
            f"MinerU VLM weights not found in {base}. Run download_weights.sh "
            "or pass --model-dir."
        )
    return base


def load_model(
    model_dir: Optional[str],
    device: str,
) -> tuple:
    """Load the MinerU VLM and its processor.

    Returns:
        (model, processor)
    """
    weights_dir = _resolve_weights_dir(model_dir)

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(weights_dir),
        dtype=dtype,
    )
    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(str(weights_dir))

    return model, processor


def _prepare_layout_image(img_array: np.ndarray) -> Image.Image:
    """RGB + resize to MinerU's layout image size."""
    return Image.fromarray(img_array).convert("RGB").resize(
        LAYOUT_IMAGE_SIZE, Image.Resampling.BICUBIC
    )


def _build_messages() -> list[dict]:
    """Build the Qwen2-VL chat messages for layout detection.

    Mirrors MinerU's TransformersVlmClient: system prompt, then a user turn with
    the image before the text (text_before_image=False).
    """
    messages = []
    if LAYOUT_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": LAYOUT_SYSTEM_PROMPT})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": LAYOUT_USER_PROMPT},
            ],
        }
    )
    return messages


def _predict_layout_text(
    model, processor, image: Image.Image, device: str, max_new_tokens: int
) -> str:
    text = processor.apply_chat_template(
        _build_messages(), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        [generated_ids], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]


def _cover_ratio(inner: list[float], outer: list[float]) -> float:
    """Intersection-over-inner-area ratio for two [x, y, w, h] bboxes."""
    ix1, iy1, iw, ih = inner
    ox1, oy1, ow, oh = outer
    ix2, iy2 = ix1 + iw, iy1 + ih
    ox2, oy2 = ox1 + ow, oy1 + oh
    inter_w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(1e-9, iw * ih)
    return inter_w * inter_h / inner_area


def _merge_consecutive_lists(
    regions: list[dict],
    left_eps: float = LIST_MERGE_LEFT_EPS,
) -> list[dict]:
    """Merge consecutive `list` regions with aligned left edges into one region.

    MinerU often emits a list as many consecutive `list`/`list_item` blocks, one
    per row. Blocks that follow each other in reading order and share (almost)
    the same left x belong to the same list (same indentation = no wrap), so
    their bounding boxes are unioned into a single `list` region. Nested lists
    are indented, so their larger left x breaks the chain and keeps them
    separate.
    """
    if not regions:
        return regions

    # Reading order: top-to-bottom, then left-to-right.
    ordered = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    merged: list[dict] = []
    group: list[dict] = []

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            merged.append(group[0])
        else:
            xs = [r["bbox"][0] for r in group]
            ys = [r["bbox"][1] for r in group]
            x2s = [r["bbox"][0] + r["bbox"][2] for r in group]
            y2s = [r["bbox"][1] + r["bbox"][3] for r in group]
            x, y = min(xs), min(ys)
            w, h = max(x2s) - x, max(y2s) - y
            merged.append(
                {
                    "bbox": [round(x, 6), round(y, 6), round(w, 6), round(h, 6)],
                    "label": "list",
                }
            )
        group.clear()

    for region in ordered:
        if region["label"] != "list":
            flush()
            merged.append(region)
            continue
        if group and abs(region["bbox"][0] - group[-1]["bbox"][0]) > left_eps:
            flush()
        group.append(region)

    flush()
    return merged


def parse_layout_output(output: str) -> list[dict]:
    """Parse the MinerU VLM layout output into normalized regions.

    Each region is ``{"bbox": [x, y, w, h], "label": str}`` with coordinates
    normalized to [0, 1].
    """
    regions = []
    for match in _LAYOUT_RE.finditer(output):
        x1, y1, x2, y2, ref_type = match.group(1, 2, 3, 4, 5)
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except ValueError:
            continue
        if any(c < 0 or c > 1000 for c in (x1, y1, x2, y2)):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x1 == x2 or y1 == y2:
            continue

        label = MINERU_BLOCK_TO_LABEL.get(ref_type.lower())
        if label is None:
            continue

        nx1 = x1 / 1000.0
        ny1 = y1 / 1000.0
        nw = (x2 - x1) / 1000.0
        nh = (y2 - y1) / 1000.0
        regions.append(
            {
                "bbox": [round(nx1, 6), round(ny1, 6), round(nw, 6), round(nh, 6)],
                "label": label,
            }
        )

    # Drop text blocks fully covered by a table (same heuristic as MinerU's
    # _filter_table_internal_layout_blocks, restricted to our five classes).
    tables = [r for r in regions if r["label"] == "table"]
    if tables:
        kept = []
        for r in regions:
            if r["label"] == "text":
                covered = any(_cover_ratio(r["bbox"], t["bbox"]) >= 0.9 for t in tables)
                if covered:
                    continue
            kept.append(r)
        regions = kept

    # Merge consecutive list rows with aligned left edges into one list region.
    regions = _merge_consecutive_lists(regions)

    return regions


def predict_image(
    img_array: np.ndarray, model, processor, device: str, max_new_tokens: int
) -> dict:
    h, w = img_array.shape[:2]
    layout_image = _prepare_layout_image(img_array)
    raw = _predict_layout_text(model, processor, layout_image, device, max_new_tokens)
    regions = parse_layout_output(raw)
    return {"num": 1, "regions": regions, "metainfo": {"height": h, "width": w}}


def infer_pdf(
    pdf_path: Path, model, processor, device: str, dpi: int, max_new_tokens: int
) -> dict:
    images = convert_from_path(pdf_path, dpi=dpi)
    pages = []
    for page_idx, img in enumerate(images, start=1):
        img_array = np.array(img.convert("RGB"))
        page = predict_image(img_array, model, processor, device, max_new_tokens)
        page["num"] = page_idx
        # metainfo in PDF points (72 dpi) for consistency with the project.
        page["metainfo"] = {
            "height": round(img.height * 72.0 / dpi, 6),
            "width": round(img.width * 72.0 / dpi, 6),
        }
        pages.append(page)
    return {"pages": pages}


def infer_image_file(
    image_path: Path, model, processor, device: str, max_new_tokens: int
) -> dict:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    page = predict_image(img_array, model, processor, device, max_new_tokens)
    page["num"] = 1
    page["metainfo"] = {"height": img.height, "width": img.width}
    return {"pages": [page]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MinerU (VLM) document layout detection on PDFs and images"
    )
    parser.add_argument("input_dir", type=str, help="Directory with PDFs/images")
    parser.add_argument("results_dir", type=str, help="Directory to write result JSONs")
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Local dir with the MinerU VLM snapshot (skips HF download)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda if available)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PDF-to-image rendering (default: 200)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Max new tokens for layout generation (default: 2048)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in sorted(PDF_EXTS | IMAGE_EXTS):
        files.extend(sorted(input_dir.glob(f"*{ext}")))
    files = sorted(files)
    if not files:
        print(f"No PDF/images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model (device={args.device})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, processor = load_model(args.model_dir, args.device)

    total = len(files)
    succeeded = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        json_path = results_dir / f"{path.stem}.json"
        try:
            if path.suffix.lower() in PDF_EXTS:
                result = infer_pdf(
                    path, model, processor, args.device, args.dpi, args.max_new_tokens
                )
            else:
                result = infer_image_file(
                    path, model, processor, args.device, args.max_new_tokens
                )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[{i}/{total}] processed: {path.name} -> {json_path.name}")
            succeeded += 1
        except Exception as exc:
            print(f"[{i}/{total}] FAILED: {path.name} — {exc}", file=sys.stderr)
            failed += 1

    print(f"Done. Processed {total} files ({succeeded} succeeded, {failed} failed).")


if __name__ == "__main__":
    main()

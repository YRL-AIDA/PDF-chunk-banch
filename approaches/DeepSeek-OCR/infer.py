"""DeepSeek-OCR document layout detection (prompt-based grounding segmentation).

This approach is similar to DiT / LayoutLM-v3 (Detectron2 object detection) and
to MinerU (VLM layout detection) in that it returns a ``pages -> regions ->
{bbox, label}`` layout, but it differs in one key way:

  * DiT / LayoutLM-v3 predict the five PubLayNet classes with a detector;
  * MinerU emits its *own* block vocabulary, which is then **mapped** to the five
    PubLayNet classes via ``MINERU_BLOCK_TO_LABEL``;
  * DeepSeek-OCR has no fixed layout vocabulary at all — we **request** the
    classes directly with a natural-language prompt (``LAYOUT_PROMPT`` below),
    and the model answers in its native grounding markup:

        <|ref|>CLASS<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

    with integer coordinates in ``0..999`` ((0,0) = top-left, (999,999) =
    bottom-right). The prompt also *defines* each class in detail (what counts as
    a table, what as text, what as a list — the **whole list**, not an item — and
    what as a figure — with fragmented figures merged into one shape), so no
    post-hoc vocabulary mapping is required.

The five class names requested by the prompt are the canonical PubLayNet names
used across this project (``metrics/mAP_iou.py``): text, title, list, table,
figure.
"""

import argparse
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pdf2image import convert_from_path
from PIL import Image

# =============================================================================
# LAYOUT PROMPT — kept at the very top of the file on purpose, so it is easy to
# edit in one place.
#
# The prompt MUST start with the image placeholder "<image>" and the grounding
# trigger "<|grounding|>": the former is where the page image is injected, and
# the latter switches DeepSeek-OCR into grounding mode so it emits the
# <|ref|>/<|det|> box markup instead of plain text.
#
# The five classes below are the canonical PubLayNet names ("правильно названные
# классы"): text, title, list, table, figure.
# =============================================================================

# Canonical PubLayNet class order (same as DiT / LayoutLM-v3 / MinerU).
CLASS_NAMES = ["text", "title", "list", "table", "figure"]

LAYOUT_PROMPT = """<image>
<|grounding|>Detect the document layout and return the coordinates of the following five region classes only: text, title, list, table, figure.

For each detected region print exactly one line in this format:
<|ref|>CLASS<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

Definitions of the classes:
- text: body text. Continuous prose paragraphs, captions, footnotes, page numbers. Do not include headings, list items, table cells, or anything inside a figure.
- title: a heading or title. A short standalone line, usually larger or bolder, that names a document, section, chapter, or a caption of a table or figure.
- table: tabular data. An aligned grid of rows and columns, usually with ruled lines and cells. Return the whole table as ONE region; do not split it into cells, rows, or columns.
- list: a whole list. A group of consecutive list items (bullets, numbers, or letters) at the same indentation that form one logical list. Return the ENTIRE list as ONE region, not each item separately. A nested sub-list with larger indentation is a separate region.
- figure: an image, chart, diagram, plot, logo, or drawing. If one figure is visually fragmented or consists of several sub-images belonging to the same illustration, merge all its fragments into ONE region: the smallest box enclosing them all.

Coordinates are integers from 0 to 999: (0, 0) is the top-left corner and (999, 999) is the bottom-right corner of the page. Output nothing else."""

# Pre-trained VLM snapshot. `download_weights.sh` downloads the full repo
# (config.json, tokenizer*, model.safetensors, modeling_*.py, conversation.py).
HF_REPO = "deepseek-ai/DeepSeek-OCR"
WEIGHTS_DIRNAME = "weights"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PDF_EXTS = {".pdf"}

# DeepSeek-OCR emits grounding coordinates in 0..999 (see the official vLLM
# visualizer: x = int(x / 999 * image_width)).
COORD_MAX = 999.0

# Safety-net synonyms folded into the canonical PubLayNet names. The prompt asks
# for the exact names above, so this is only a fallback for the occasional
# paraphrase the model emits despite the instruction (e.g. "image" for a
# figure). Anything outside this table is dropped.
LABEL_ALIASES = {
    "text": "text",
    "title": "title",
    "list": "list",
    "table": "table",
    "figure": "figure",
    # occasional model synonyms
    "image": "figure",
    "chart": "figure",
    "diagram": "figure",
    "heading": "title",
}

# Grounding markup emitted by DeepSeek-OCR:
#   <|ref|>CLASS<|/ref|><|det|>[[x1,y1,x2,y2],[x1,y1,x2,y2]]<|/det|>
_GROUNDING_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)

# Extracts one [x1, y1, x2, y2] quadruple (optional single surrounding bracket).
_NUM = r"[-+]?\d+(?:\.\d+)?"
_BOX_RE = re.compile(
    rf"\[?\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]?"
)

# Distance (fraction of page width/height) within which separate `figure` boxes
# are merged into one figure region (fragments / sub-images of one illustration).
FIGURE_MERGE_GAP = 0.02

# Consecutive `list` boxes with left edges aligned to within this fraction of
# the page width are merged into a single whole-list region.
LIST_MERGE_LEFT_EPS = 0.02


def _resolve_weights_dir(model_dir: Optional[str]) -> Path:
    """Return the directory containing the DeepSeek-OCR snapshot."""
    if model_dir:
        base = Path(model_dir)
        if not (base / "config.json").is_file():
            raise FileNotFoundError(f"config.json not found in {base}")
        return base

    base = Path(__file__).resolve().parent / WEIGHTS_DIRNAME
    if not (base / "config.json").is_file():
        from huggingface_hub import snapshot_download

        base.mkdir(parents=True, exist_ok=True)
        base = Path(snapshot_download(repo_id=HF_REPO, local_dir=str(base)))
    if not (base / "config.json").is_file():
        raise FileNotFoundError(
            f"DeepSeek-OCR weights not found in {base}. Run download_weights.sh "
            "or pass --model-dir."
        )
    return base


def load_model(model_dir: Optional[str], device: str, quantize: str = "none"):
    """Load the DeepSeek-OCR VLM and its tokenizer.

    The checkpoint stores weights in bfloat16 (~6.7 GB). Loading them in fp32
    (transformers' default when ``torch_dtype`` is not given) would take
    ~13.4 GB, which does not fit an 8 GB GPU. We therefore load directly in
    bfloat16 and only then move to the device (``.to(dtype)`` *before*
    ``.to(device)``), so the fp32 copy never reaches the GPU.

    Args:
        model_dir: local snapshot dir (or None to use/download the default).
        device: "cuda" or "cpu".
        quantize: "none" (bfloat16) or "8bit" (bitsandbytes int8, CUDA only).

    Returns:
        (model, tokenizer)
    """
    weights_dir = _resolve_weights_dir(model_dir)

    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)

    if quantize == "8bit" and device == "cuda":
        # int8 weights (~3.4 GB) keep the whole model on GPU 0, which leaves
        # enough headroom to run the full Gundam resolution on 8 GB. Requires
        # `bitsandbytes` and `accelerate` in the environment.
        model = AutoModel.from_pretrained(
            str(weights_dir),
            trust_remote_code=True,
            use_safetensors=True,
            device_map={"": 0},
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.eval()
        return model, tokenizer

    model = AutoModel.from_pretrained(
        str(weights_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )
    model = model.eval().to(dtype).to(device)

    return model, tokenizer


def _predict_layout_text(
    model,
    tokenizer,
    img_array: np.ndarray,
    work_dir: Path,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
) -> str:
    """Run one page through DeepSeek-OCR and return the raw grounding text.

    Reuses the model's own ``infer(..., eval_mode=True)`` entry point, which does
    all image preprocessing (Gundam dynamic resolution: n x 640x640 crops + one
    ``base_size`` global view) and returns the decoded output (special tokens
    kept).

    Memory note: on an 8 GB GPU the default Gundam resolution (``crop_mode=True,
    base_size=1024``) produces ~1200 vision tokens and its fp32 logits spike
    (~750 MB) does not fit. The 8 GB-safe default is ``crop_mode=False,
    base_size=640`` (~111 vision tokens, ~190 MB logits spike).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    page_path = work_dir / "page.png"
    Image.fromarray(img_array).convert("RGB").save(page_path)

    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = model.infer(
        tokenizer,
        prompt=LAYOUT_PROMPT,
        image_file=str(page_path),
        output_path=str(out_dir),
        eval_mode=True,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    return raw or ""


def _normalize_label(label: str) -> str:
    """Lowercase and collapse punctuation/spaces to underscores."""
    return re.sub(r"[^a-z0-9_]+", "_", label.strip().lower()).strip("_")


def _extract_boxes(det_text: str) -> list[list[float]]:
    """Extract [x1, y1, x2, y2] quadruples from a <|det|>...</|det|> payload.

    The payload is a (nested) list of boxes, e.g. ``[[100,200,300,400]]`` or
    ``[[100,200,300,400],[500,600,700,800]]``. Each box is order-normalized and
    degenerate (zero-area) boxes are dropped.
    """
    boxes: list[list[float]] = []
    for m in _BOX_RE.finditer(det_text):
        try:
            x1, y1, x2, y2 = (float(v) for v in m.groups())
        except ValueError:
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 == x2 or y1 == y2:
            continue
        boxes.append([x1, y1, x2, y2])
    return boxes


def _clamp(v: float) -> float:
    return min(max(v, 0.0), COORD_MAX)


def _union_bbox(a: list[float], b: list[float]) -> list[float]:
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return [round(x, 6), round(y, 6), round(x2 - x, 6), round(y2 - y, 6)]


def _boxes_nearby(a: list[float], b: list[float], gap: float) -> bool:
    """True if two [x, y, w, h] boxes overlap or touch within ``gap``."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax - gap <= bx + bw
        and bx - gap <= ax + aw
        and ay - gap <= by + bh
        and by - gap <= ay + ah
    )


def _merge_figures(regions: list[dict], gap: float = FIGURE_MERGE_GAP) -> list[dict]:
    """Merge overlapping/nearby ``figure`` regions into one connected region.

    The prompt already asks the model to merge fragments of one figure, but the
    model can still emit several separate <|ref|>figure<|/ref|> blocks (sub-panels,
    logo + chart, ...). This unions every connected component of near figure
    boxes, so a single illustration becomes a single ``figure`` region.
    """
    others = [r for r in regions if r["label"] != "figure"]
    figures = [r for r in regions if r["label"] == "figure"]

    merged: list[dict] = []
    while figures:
        base = figures.pop(0)
        changed = True
        while changed:
            changed = False
            for i, f in enumerate(figures):
                if _boxes_nearby(base["bbox"], f["bbox"], gap):
                    base = {"bbox": _union_bbox(base["bbox"], f["bbox"]), "label": "figure"}
                    figures.pop(i)
                    changed = True
                    break
        merged.append(base)

    return others + merged


def _merge_lists(
    regions: list[dict], left_eps: float = LIST_MERGE_LEFT_EPS
) -> list[dict]:
    """Merge consecutive ``list`` boxes with aligned left edges into one region.

    DeepSeek-OCR may emit a list as several consecutive rows despite the prompt
    asking for the whole list. Rows that follow each other in reading order and
    share (almost) the same left x belong to the same list (same indentation =
    no wrap), so their boxes are unioned into a single ``list`` region. A nested
    list is indented, so its larger left x breaks the chain and keeps it separate.
    """
    if not regions:
        return regions

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
    """Parse DeepSeek-OCR grounding output into normalized regions.

    Each region is ``{"bbox": [x, y, w, h], "label": str}`` with coordinates
    normalized to [0, 1] (relative to the page). Multiple boxes inside one
    <|ref|> are unioned; separate figure blocks are merged into one figure;
    consecutive aligned list rows are merged into a whole list.
    """
    regions: list[dict] = []
    for m in _GROUNDING_RE.finditer(output):
        raw_label, det_text = m.group(1), m.group(2)
        label = LABEL_ALIASES.get(_normalize_label(raw_label))
        if label is None:
            continue

        boxes = _extract_boxes(det_text)
        if not boxes:
            continue

        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        nx1 = _clamp(x1) / COORD_MAX
        ny1 = _clamp(y1) / COORD_MAX
        nx2 = _clamp(x2) / COORD_MAX
        ny2 = _clamp(y2) / COORD_MAX

        regions.append(
            {
                "bbox": [
                    round(nx1, 6),
                    round(ny1, 6),
                    round(nx2 - nx1, 6),
                    round(ny2 - ny1, 6),
                ],
                "label": label,
            }
        )

    regions = _merge_figures(regions)
    regions = _merge_lists(regions)
    # Deterministic reading order: top-to-bottom, then left-to-right.
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return regions


def predict_image(
    img_array: np.ndarray,
    model,
    tokenizer,
    work_dir: Path,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
) -> dict:
    h, w = img_array.shape[:2]
    raw = _predict_layout_text(
        model, tokenizer, img_array, work_dir,
        base_size=base_size, image_size=image_size, crop_mode=crop_mode,
    )
    regions = parse_layout_output(raw)
    return {"num": 1, "regions": regions, "metainfo": {"height": h, "width": w}}


def infer_pdf(
    pdf_path: Path,
    model,
    tokenizer,
    dpi: int,
    work_dir: Path,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
) -> dict:
    images = convert_from_path(pdf_path, dpi=dpi)
    pages = []
    for page_idx, img in enumerate(images, start=1):
        img_array = np.array(img.convert("RGB"))
        page = predict_image(
            img_array, model, tokenizer, work_dir,
            base_size=base_size, image_size=image_size, crop_mode=crop_mode,
        )
        page["num"] = page_idx
        # metainfo in PDF points (72 dpi) for consistency with the project.
        page["metainfo"] = {
            "height": round(img.height * 72.0 / dpi, 6),
            "width": round(img.width * 72.0 / dpi, 6),
        }
        pages.append(page)
    return {"pages": pages}


def infer_image_file(
    image_path: Path,
    model,
    tokenizer,
    work_dir: Path,
    base_size: int = 640,
    image_size: int = 640,
    crop_mode: bool = False,
) -> dict:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    page = predict_image(
        img_array, model, tokenizer, work_dir,
        base_size=base_size, image_size=image_size, crop_mode=crop_mode,
    )
    page["num"] = 1
    page["metainfo"] = {"height": img.height, "width": img.width}
    return {"pages": [page]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR (VLM) document layout detection via a "
        "prompt-based grounding request, on PDFs and images"
    )
    parser.add_argument("input_dir", type=str, help="Directory with PDFs/images")
    parser.add_argument("results_dir", type=str, help="Directory to write result JSONs")
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Local dir with the DeepSeek-OCR snapshot (skips HF download)",
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
        "--base-size", type=int, default=640,
        help="Global-view size passed to model.infer() (default: 640; the "
        "official full-resolution value is 1024 but needs >8 GB or 8-bit)",
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Local-crop/global-view size passed to model.infer() (default: 640)",
    )
    parser.add_argument(
        "--crop-mode", action="store_true",
        help="Enable Gundam multi-crop preprocessing (default: off). Turning it "
        "on raises quality but needs >8 GB or --quantize 8bit on an 8 GB GPU.",
    )
    parser.add_argument(
        "--quantize", type=str, default="none", choices=["none", "8bit"],
        help="Weight quantization: 'none' (bfloat16) or '8bit' (bitsandbytes "
        "int8, CUDA only; requires bitsandbytes + accelerate)",
    )
    args = parser.parse_args()

    # The model's own `infer()` path builds bf16 CUDA tensors internally and the
    # ~6.7 GB MoE + SAM/CLIP encoders do not fit CPU in practical time.
    if args.device == "cpu":
        print(
            "Error: DeepSeek-OCR requires CUDA. The model's inference path uses "
            "bfloat16 CUDA tensors and is not practical on CPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.quantize == "8bit" and args.device != "cuda":
        print("Error: --quantize 8bit requires --device cuda.", file=sys.stderr)
        sys.exit(1)

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

    print(f"Loading model (device={args.device}, quantize={args.quantize})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, tokenizer = load_model(args.model_dir, args.device, args.quantize)

    total = len(files)
    succeeded = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix="deepseekocr_") as tmp:
        work_dir = Path(tmp)
        for i, path in enumerate(files, start=1):
            json_path = results_dir / f"{path.stem}.json"
            if json_path.exists():
                print(f'[{i}/{total}] exists: {json_path.name} ')
                continue
            try:
                if path.suffix.lower() in PDF_EXTS:
                    result = infer_pdf(
                        path, model, tokenizer, args.dpi, work_dir,
                        base_size=args.base_size,
                        image_size=args.image_size,
                        crop_mode=args.crop_mode,
                    )
                else:
                    result = infer_image_file(
                        path, model, tokenizer, work_dir,
                        base_size=args.base_size,
                        image_size=args.image_size,
                        crop_mode=args.crop_mode,
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

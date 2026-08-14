import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pdf2image import convert_from_path
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes

# PubLayNet class order used by the fine-tuned LayoutLMv3 checkpoint.
CLASS_NAMES = ["text", "title", "list", "table", "figure"]

# Config shipped with the unilm repository (Cascade Mask R-CNN + LayoutLMv3-base).
CONFIG_RELPATH = (
    Path("layoutlmv3") / "examples" / "object_detection" / "cascade_layoutlmv3.yaml"
)

# Pre-trained fine-tuned weights (PubLayNet), mirrored on HuggingFace since the
# original Azure Blob links in the README are no longer publicly accessible.
WEIGHTS_FILENAME = "model_final.pth"
CONFIG_JSON_FILENAME = "config.json"
HF_REPO = "HYPJUDY/layoutlmv3-base-finetuned-publaynet"
HF_WEIGHTS_PATH = "model_final.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PDF_EXTS = {".pdf"}


def _find_unilm_root() -> Path:
    """Locate the cloned microsoft/unilm repository."""
    candidates = [
        Path.cwd() / "approaches" / "unilm",
        Path(__file__).resolve().parents[1] / "unilm",  # approaches/unilm
        Path(__file__).resolve().parent / "unilm",       # approaches/LayoutLM-v3/unilm
        Path.cwd() / "unilm",
    ]
    for p in candidates:
        ditod_init = p / "layoutlmv3" / "examples" / "object_detection" / "ditod" / "__init__.py"
        if ditod_init.exists():
            return p
    raise FileNotFoundError(
        "Cannot find unilm repository. Clone it with:\n"
        "  git clone https://github.com/microsoft/unilm.git\n"
        "into approaches/ or the project root."
    )


def _register_components(unilm_root: Path) -> None:
    """Add layoutlmv3 + ditod directories to sys.path and import ditod so that
    VLGeneralizedRCNN, build_vit_fpn_backbone and add_vit_config are registered.

    LayoutLMv3 detection uses two packages from the unilm repo:
      - `layoutlmft`          (LayoutLMv3Model backbone)   under unilm/layoutlmv3
      - `ditod`               (Detectron2 components)      under unilm/layoutlmv3/examples/object_detection
    """
    layoutlmv3_path = str(unilm_root / "layoutlmv3")
    ditod_path = str(unilm_root / "layoutlmv3" / "examples" / "object_detection")
    for p in (layoutlmv3_path, ditod_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    import ditod  # noqa: F401
    from ditod import add_vit_config  # noqa: F401


def _resolve_config_path(config_file: Optional[str], unilm_root: Path) -> Path:
    if config_file:
        p = Path(config_file)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Config file not found: {config_file}")
    p = unilm_root / CONFIG_RELPATH
    if p.is_file():
        return p
    raise FileNotFoundError(f"Default config not found: {p}")


def _resolve_weights_dir(model_dir: Optional[str]) -> Path:
    """Return the directory that contains both model_final.pth and config.json.

    The LayoutLMv3 backbone builds its transformers config via
    `AutoConfig.from_pretrained(cfg.MODEL.CONFIG_PATH)`, so config.json must live
    in the same directory as the checkpoint.
    """
    if model_dir:
        base = Path(model_dir)
    else:
        base = Path(__file__).resolve().parent / "weights"

    base.mkdir(parents=True, exist_ok=True)

    weights = base / WEIGHTS_FILENAME
    config_json = base / CONFIG_JSON_FILENAME

    if not weights.is_file():
        from huggingface_hub import hf_hub_download

        weights = Path(
            hf_hub_download(repo_id=HF_REPO, filename=HF_WEIGHTS_PATH, local_dir=str(base))
        )
    if not config_json.is_file():
        from huggingface_hub import hf_hub_download

        config_json = Path(
            hf_hub_download(repo_id=HF_REPO, filename=CONFIG_JSON_FILENAME, local_dir=str(base))
        )

    if not weights.is_file():
        raise FileNotFoundError(f"{WEIGHTS_FILENAME} not found in {base}")
    if not config_json.is_file():
        raise FileNotFoundError(f"{CONFIG_JSON_FILENAME} not found in {base}")

    return base


def load_model(
    config_file: Optional[str],
    model_dir: Optional[str],
    device: str,
    score_threshold: float,
) -> DefaultPredictor:
    unilm_root = _find_unilm_root()
    _register_components(unilm_root)
    from ditod import add_vit_config

    config_path = _resolve_config_path(config_file, unilm_root)
    weights_dir = _resolve_weights_dir(model_dir)
    weights_path = weights_dir / WEIGHTS_FILENAME

    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.CONFIG_PATH = str(weights_dir)  # dir with transformers config.json
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=list(CLASS_NAMES))

    return predictor


def boxes_to_normalized_bbox(
    boxes: Boxes, img_width: int, img_height: int
) -> list[list[float]]:
    bboxes = []
    for box in boxes.tensor.tolist():
        x0, y0, x1, y1 = box
        x = round(x0 / img_width, 6)
        y = round(y0 / img_height, 6)
        w = round((x1 - x0) / img_width, 6)
        h = round((y1 - y0) / img_height, 6)
        bboxes.append([x, y, w, h])
    return bboxes


def predict_image(img_array: np.ndarray, predictor: DefaultPredictor) -> dict:
    outputs = predictor(img_array)
    instances = outputs["instances"].to("cpu")

    h, w = img_array.shape[:2]
    classes = instances.pred_classes.numpy()
    bboxes = boxes_to_normalized_bbox(instances.pred_boxes, w, h)

    regions = []
    for bbox, cls_id in zip(bboxes, classes):
        label = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "unknown"
        regions.append({"bbox": bbox, "label": label})

    return {"num": 1, "regions": regions, "metainfo": {"height": h, "width": w}}


def infer_pdf(pdf_path: Path, predictor: DefaultPredictor, dpi: int) -> dict:
    images = convert_from_path(pdf_path, dpi=dpi)
    pages = []
    for page_idx, img in enumerate(images, start=1):
        img_array = np.array(img.convert("RGB"))
        page = predict_image(img_array, predictor)
        page["num"] = page_idx
        # metainfo in PDF points (72 dpi) for consistency with the project
        page["metainfo"] = {
            "height": round(img.height * 72.0 / dpi, 6),
            "width": round(img.width * 72.0 / dpi, 6),
        }
        pages.append(page)
    return {"pages": pages}


def infer_image_file(image_path: Path, predictor: DefaultPredictor) -> dict:
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    page = predict_image(img_array, predictor)
    page["num"] = 1
    page["metainfo"] = {"height": img.height, "width": img.width}
    return {"pages": [page]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LayoutLMv3 document layout detection on PDFs and images"
    )
    parser.add_argument("input_dir", type=str, help="Directory with PDFs/images")
    parser.add_argument("results_dir", type=str, help="Directory to write result JSONs")
    parser.add_argument(
        "--config-file", type=str, default=None,
        help="Path to Detectron2 config YAML (default: unilm cascade_layoutlmv3.yaml)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Local dir containing model_final.pth + config.json (skips HF download)",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda if available)",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PDF-to-image rendering (default: 200)",
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

    print(
        f"Loading model (device={args.device}, "
        f"score_threshold={args.score_threshold})..."
    )
    predictor = load_model(
        args.config_file, args.model_dir, args.device, args.score_threshold
    )

    total = len(files)
    succeeded = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        json_path = results_dir / f"{path.stem}.json"
        try:
            if path.suffix.lower() in PDF_EXTS:
                result = infer_pdf(path, predictor, args.dpi)
            else:
                result = infer_image_file(path, predictor)
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

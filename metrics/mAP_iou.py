"""Compute COCO-style mAP@[IoU 0.5:0.95] for layout detection predictions.

Compares a folder of per-image prediction JSONs produced by a method
(e.g. ``results_DiT_publaynet/``) against a COCO ground-truth annotation file
(e.g. PubLayNet ``val.json``) using the reference ``torchmetrics.MeanAveragePrecision``
implementation (pycocotools backend), which reproduces the official COCO metric
(10 IoU thresholds 0.5..0.95, averaged over classes and thresholds).

Usage:
    env/bin/python metrics/mAP_iou.py results_DiT_publaynet \\
        /home/dataset/PubLayNet/micro_publaynet_5/publaynet/val.json \\
        --dataset publaynet [--output results_DiT_publaynet_mAP.json]

The ``--dataset`` flag selects how parser markup coordinates are interpreted.
For ``publaynet`` the parser stores normalized bboxes (fractions of the page),
so they are multiplied by the page dimensions from each page's ``metainfo``
(falling back to the GT image size when ``metainfo`` is missing).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger("mAP_iou")

# PubLayNet class order: category id (1-indexed) -> 0-indexed class index.
CLASS_NAMES: list[str] = ["text", "title", "list", "table", "figure"]
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Standard COCO IoU thresholds 0.5:0.95 step 0.05 (explicit for reproducibility).
IOU_THRESHOLDS: list[float] = [round(0.5 + 0.05 * i, 2) for i in range(10)]

# Predictions carry no confidence (already thresholded by the detector), so all
# detections are assigned the same score; see spec for rationale.
DEFAULT_SCORE: float = 1.0

#: Datasets supported via ``--dataset``. For these, the parser markup stores
#: bboxes normalized to [0,1], which must be multiplied by the page dimensions
#: (``metainfo`` width/height) to become absolute pixels.
NORMALIZED_BBOX_DATASETS: frozenset[str] = frozenset({"publaynet"})


def load_gt(gt_json: Path) -> dict[str, Any]:
    """Load a COCO ground-truth file.

    Args:
        gt_json: Path to a COCO annotation JSON.

    Returns:
        Dict with keys:
            ``images``: list of image dicts (``file_name``, ``width``, ``height``, ``id``).
            ``annotations``: list of annotation dicts (``bbox``, ``category_id``, ``image_id``).
            ``image_to_id``: {stem: image_id} where stem is ``Path(file_name).stem``.
            ``image_size``: {image_id: (width, height)}.
    """
    with open(gt_json, encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    image_to_id: dict[str, int] = {}
    image_size: dict[int, tuple[int, int]] = {}
    for img in images:
        image_id = img["id"]
        stem = Path(img["file_name"]).stem
        if stem in image_to_id:
            logger.warning("Duplicate file_name stem %r in GT; keeping last image id.", stem)
        image_to_id[stem] = image_id
        image_size[image_id] = (img["width"], img["height"])

    logger.info(
        "GT loaded: %d images, %d annotations (%s)",
        len(images), len(annotations), gt_json,
    )
    return {
        "images": images,
        "annotations": annotations,
        "image_to_id": image_to_id,
        "image_size": image_size,
    }


def load_preds(
    results_dir: Path,
    gt_stems: set[str],
    stem_to_size: dict[str, tuple[int, int]],
    dataset: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load prediction JSONs and convert bboxes to absolute pixels.

    For datasets in :data:`NORMALIZED_BBOX_DATASETS` (e.g. ``publaynet``) the
    parser markup stores bboxes normalized to [0,1] relative to the page size,
    so each bbox is multiplied by the page dimensions from that page's
    ``metainfo`` (falling back to the GT image size when ``metainfo`` is
    missing or invalid). For other datasets the bboxes are taken as already
    absolute.

    Args:
        results_dir: Directory containing one ``<stem>.json`` per image.
        gt_stems: Set of valid stems from the GT file (extra files are skipped).
        stem_to_size: {stem: (width, height)} from GT, fallback page size.
        dataset: Dataset name controlling the coordinate convention.

    Returns:
        {stem: [{"bbox": [x, y, w, h] (absolute px), "label_id": int 0..4}]}.
    """
    normalize = dataset in NORMALIZED_BBOX_DATASETS
    preds: dict[str, list[dict[str, Any]]] = {}
    skipped = 0

    for json_path in sorted(results_dir.glob("*.json")):
        stem = json_path.stem
        if stem not in gt_stems:
            skipped += 1
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable prediction file %s: %s", json_path, exc)
            skipped += 1
            continue

        gt_width, gt_height = stem_to_size.get(stem, (0, 0))
        regions: list[dict[str, Any]] = []
        for page in data.get("pages", []):
            # Page dimensions come from the parser markup itself; fall back to
            # the GT image size when missing.
            metainfo = page.get("metainfo") or {}
            page_width = float(metainfo.get("width") or 0)
            page_height = float(metainfo.get("height") or 0)
            if page_width <= 0 or page_height <= 0:
                page_width, page_height = float(gt_width), float(gt_height)

            for region in page.get("regions", []):
                label = region.get("label")
                if label not in CLASS_TO_INDEX:
                    logger.warning(
                        "Unknown label %r in %s; region skipped", label, json_path
                    )
                    continue
                x, y, w, h = region["bbox"]
                if normalize:
                    bbox = [x * page_width, y * page_height, w * page_width, h * page_height]
                else:
                    bbox = [float(x), float(y), float(w), float(h)]
                regions.append({"bbox": bbox, "label_id": CLASS_TO_INDEX[label]})
        preds[stem] = regions

    logger.info(
        "Predictions loaded: %d matched files, %d skipped (not in GT); "
        "dataset=%s (bbox %s)",
        len(preds), skipped, dataset,
        "normalized -> page size" if normalize else "assumed absolute",
    )
    return preds


def _to_tensor(values: list[Any], dtype: torch.dtype) -> torch.Tensor:
    """Build a CPU tensor of the given dtype, returning a 0-length tensor if empty."""
    if not values:
        return torch.empty((0,), dtype=dtype)
    return torch.tensor(values, dtype=dtype)


def _boxes_tensor(boxes: list[list[float]]) -> torch.Tensor:
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def build_inputs(
    gt: dict[str, Any],
    preds_by_stem: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], int]:
    """Build aligned torchmetrics inputs, one dict per GT image (deterministic order).

    Args:
        gt: Result of :func:`load_gt`.
        preds_by_stem: Result of :func:`load_preds`.

    Returns:
        (preds, targets, num_empty_pred_images):
            - ``preds``: list of {boxes, scores, labels} per image (aligned with targets).
            - ``targets``: list of {boxes, labels} per image.
            - ``num_empty_pred_images``: number of GT images with no predictions.
    """
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in gt["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    preds: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    num_empty_pred_images = 0

    # Deterministic order: sort by image id.
    for img in sorted(gt["images"], key=lambda i: i["id"]):
        image_id = img["id"]
        stem = Path(img["file_name"]).stem
        t_boxes: list[list[float]] = []
        t_labels: list[int] = []
        for ann in annotations_by_image.get(image_id, []):
            t_boxes.append(ann["bbox"])
            t_labels.append(int(ann["category_id"]) - 1)  # 1-indexed -> 0-indexed
        targets.append({"boxes": _boxes_tensor(t_boxes), "labels": _to_tensor(t_labels, torch.int64)})

        p_boxes: list[list[float]] = []
        p_labels: list[int] = []
        p_scores: list[float] = []
        for region in preds_by_stem.get(stem, []):
            p_boxes.append(region["bbox"])
            p_labels.append(region["label_id"])
            p_scores.append(DEFAULT_SCORE)
        if not p_boxes:
            num_empty_pred_images += 1
        preds.append(
            {
                "boxes": _boxes_tensor(p_boxes),
                "scores": _to_tensor(p_scores, torch.float32),
                "labels": _to_tensor(p_labels, torch.int64),
            }
        )

    logger.info(
        "Built %d aligned inputs (%d GT images without predictions)",
        len(preds), num_empty_pred_images,
    )
    return preds, targets, num_empty_pred_images


def run_map(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_metrics: bool,
) -> dict[str, Any]:
    """Run one MeanAveragePrecision pass.

    Args:
        preds: List of {boxes, scores, labels} per image (xywh, absolute px).
        targets: List of {boxes, labels} per image (xywh, absolute px).
        class_metrics: Return per-class metrics (``map_per_class``) as well.

    Returns:
        Raw ``compute()`` dict with at least ``map``, ``map_50``, ``map_75`` and,
        if ``class_metrics``, ``map_per_class``.
    """
    metric = MeanAveragePrecision(
        box_format="xywh",
        iou_type="bbox",
        iou_thresholds=IOU_THRESHOLDS,
        class_metrics=class_metrics,
        extended_summary=False,
        backend="pycocotools",
    )
    metric.update(preds, targets)
    return metric.compute()


def _safe_float(value: Any) -> float:
    """Convert a metric scalar to a clean float, mapping nan/-1/inf to 0.0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f) or f < 0.0:
        return 0.0
    return f


def compute_ap_seg(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> float:
    """Compute AP with all classes merged into a single class (label 0)."""
    merged_preds: list[dict[str, torch.Tensor]] = []
    for p in preds:
        merged_preds.append(
            {
                "boxes": p["boxes"],
                "scores": p["scores"],
                "labels": torch.zeros_like(p["labels"]),
            }
        )
    merged_targets: list[dict[str, torch.Tensor]] = []
    for t in targets:
        merged_targets.append(
            {"boxes": t["boxes"], "labels": torch.zeros_like(t["labels"])}
        )
    result = run_map(merged_preds, merged_targets, class_metrics=False)
    return _safe_float(result["map"])


def compute_report(
    results_dir: Path,
    gt_json: Path,
    dataset: str = "publaynet",
) -> dict[str, Any]:
    """Compute the full metrics report for a method's results folder.

    Args:
        results_dir: Directory with prediction JSONs.
        gt_json: Path to the COCO ground-truth file.
        dataset: Dataset name controlling the bbox coordinate convention
            (see :data:`NORMALIZED_BBOX_DATASETS`).

    Returns:
        Report dict with keys ``path``, ``mAP``, ``mAP@0.5``, ``mAP@0.75``,
        ``AP (<class>)`` for each class, and ``AP (seg)``.
    """
    gt = load_gt(gt_json)
    gt_stems = set(gt["image_to_id"].keys())
    stem_to_size = {
        Path(img["file_name"]).stem: (img["width"], img["height"])
        for img in gt["images"]
    }

    preds_by_stem = load_preds(results_dir, gt_stems, stem_to_size, dataset)
    preds, targets, _ = build_inputs(gt, preds_by_stem)
    per_class = run_map(preds, targets, class_metrics=True)

    report: dict[str, Any] = {
        "path": results_dir.name,
        "mAP": _safe_float(per_class["map"]),
        "mAP@0.5": _safe_float(per_class["map_50"]),
        "mAP@0.75": _safe_float(per_class["map_75"]),
    }

    map_per_class = per_class.get("map_per_class", None)
    # torchmetrics returns map_per_class aligned with the ``classes`` tensor
    # (the actual class ids present in the data), NOT with contiguous 0..N-1
    # indices. When a class is absent from the subset (e.g. ``list`` here), the
    # positional mapping below would shift every following class and zero out
    # the last one. Map each class id back to its position in ``classes``.
    classes = per_class.get("classes", None)
    class_id_to_pos: dict[int, int] = {}
    if classes is not None:
        class_id_to_pos = {int(c): i for i, c in enumerate(classes.tolist())}
    for i, name in enumerate(CLASS_NAMES):
        pos = class_id_to_pos.get(i)
        if map_per_class is not None and pos is not None:
            report[f"AP ({name})"] = _safe_float(map_per_class[pos])
        else:
            report[f"AP ({name})"] = -1.0

    report["AP (seg)"] = compute_ap_seg(preds, targets)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, compute the report, write JSON + print to stdout."""
    parser = argparse.ArgumentParser(
        description="Compute COCO-style mAP@[IoU 0.5:0.95] for layout predictions vs COCO GT."
    )
    parser.add_argument("results_dir", type=str, help="Directory with prediction JSONs")
    parser.add_argument("gt_json", type=str, help="Path to COCO ground-truth JSON")
    parser.add_argument(
        "--dataset", type=str, default="publaynet",
        choices=sorted(NORMALIZED_BBOX_DATASETS),
        help="Dataset name controlling bbox coordinate convention "
             "(normalized bboxes are multiplied by page dimensions); default: publaynet",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <cwd>/<results_dir_name>_mAP.json)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        logger.error("results_dir is not a directory: %s", results_dir)
        return 1

    gt_json = Path(args.gt_json)
    if not gt_json.is_file():
        logger.error("gt_json is not a file: %s", gt_json)
        return 1

    report = compute_report(results_dir, gt_json, args.dataset)

    output_path = Path(args.output) if args.output else Path.cwd() / f"{results_dir.name}_mAP.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

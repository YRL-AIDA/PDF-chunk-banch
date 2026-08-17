"""Compute COCO-style mAP@[IoU 0.5:0.95] with **line-based IoU** for layout detection.

Analogue of :mod:`metrics.mAP_iou` with a single change: the overlap between a
prediction box and a ground-truth box is measured by the number of text lines
("how many lines fit"), not by pixel area. Text lines are read from
``chunk_cache/<stem>.json`` (one ``chunk`` == one ``LTTextLine`` produced by the
same pdfminer parser as the predictions).

A line belongs to a region (among predictions, or separately among GT regions)
with the largest overlap, where overlap = ``area(line ∩ region) / area(line)``,
strictly greater than 0.5. The Line-IoU of a (pred, GT) pair is then the
cardinality Jaccard index of the two line sets:

    |lines(pred) ∩ lines(gt)| / |lines(pred) ∪ lines(gt)|

Everything else (GT loading, prediction loading, bbox de-normalization, greedy
COCO matching, 10 IoU thresholds 0.5..0.95, per-class AP, ``AP (seg)``, output
JSON format) is unchanged from :mod:`metrics.mAP_iou`.

Usage:
    env/bin/python metrics/mAP_lines.py results_DiT_publaynet \\
        /home/dataset/PubLayNet/micro_publaynet_1k/publaynet/val.json \\
        chunk_cache --dataset publaynet [--output results_DiT_publaynet_mAP_lines.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Bootstrap: make the project root importable when the script is run directly
# (`python metrics/mAP_lines.py`), since sys.path[0] is then `metrics/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.mAP_iou import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_SCORE,
    IOU_THRESHOLDS,
    NORMALIZED_BBOX_DATASETS,
    load_gt,
    load_preds,
)

logger = logging.getLogger("mAP_lines")

#: Membership threshold: overlap = area(line ∩ region) / area(line) must be
#: strictly greater than this for a line to belong to a region.
LINE_OVERLAP_THRESHOLD: float = 0.5


def load_lines(
    chunk_cache_dir: Path,
    gt_stems: set[str],
    stem_to_size: dict[str, tuple[int, int]],
) -> dict[str, list[list[float]]]:
    """Load text lines from ``chunk_cache``, bboxes converted to absolute px (xywh).

    Each ``<stem>.json`` has the form ``pages[] -> {num, chunks[], images[],
    metainfo{width,height}}``. Each ``chunk`` is an ``LTTextLine`` with
    ``bbox = [x, y, w, h]`` normalized to [0,1] relative to its page's
    ``metainfo``. Chunks of all pages are concatenated in page order (for
    PubLayNet there is a single page).

    Args:
        chunk_cache_dir: Directory with ``<stem>.json`` line files (``chunk_cache/``).
        gt_stems: Set of image stems from the GT ``val.json``.
        stem_to_size: {stem: (width, height)} from GT — fallback page size.

    Returns:
        {stem: [[x, y, w, h] absolute px, ...]} — one entry per GT stem.
        Stems without a ``chunk_cache`` file (or without chunks) map to ``[]``.
        Normalized bboxes are multiplied by the page size from ``metainfo``
        (fallback ``stem_to_size``), the same de-normalization as
        :func:`metrics.mAP_iou.load_preds`.
    """
    lines_by_stem: dict[str, list[list[float]]] = {}
    missing = 0

    for stem in sorted(gt_stems):
        json_path = chunk_cache_dir / f"{stem}.json"
        if not json_path.is_file():
            lines_by_stem[stem] = []
            missing += 1
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable chunk_cache file %s: %s", json_path, exc)
            lines_by_stem[stem] = []
            missing += 1
            continue

        fallback_w, fallback_h = stem_to_size.get(stem, (0, 0))
        lines: list[list[float]] = []
        for page in data.get("pages", []):
            metainfo = page.get("metainfo") or {}
            page_width = float(metainfo.get("width") or 0)
            page_height = float(metainfo.get("height") or 0)
            if page_width <= 0 or page_height <= 0:
                page_width, page_height = float(fallback_w), float(fallback_h)
            if page_width <= 0 or page_height <= 0:
                continue

            for chunk in page.get("chunks", []):
                x, y, w, h = chunk["bbox"]
                lines.append([x * page_width, y * page_height, w * page_width, h * page_height])

        lines_by_stem[stem] = lines

    if missing:
        logger.warning(
            "%d GT image(s) have no chunk_cache file (or no readable lines); "
            "their Line-IoU will be 0.",
            missing,
        )
    logger.info(
        "Lines loaded: %d files with lines, %d missing (out of %d GT stems) from %s",
        len(gt_stems) - missing, missing, len(gt_stems), chunk_cache_dir,
    )
    return lines_by_stem


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 4) xywh -> (N, 4) xyxy (x1, y1, x2, y2). Empty input -> (0, 4)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x, y, x + w, y + h], axis=1)


def assign_lines_to_regions(
    lines_xywh: list[list[float]],
    regions_xywh: list[list[float]],
) -> np.ndarray:
    """Assign each line to the region with the largest overlap.

    ``overlap(line, region) = area(line ∩ region) / area(line)`` — the fraction
    of the line's area inside the region. A line belongs to the region with the
    maximum overlap, provided ``overlap > LINE_OVERLAP_THRESHOLD`` (strictly);
    ties go to the lowest index. A line not exceeding the threshold with any
    region is assigned ``-1``. Fully vectorized via numpy broadcasting.

    Args:
        lines_xywh: List of lines (xywh, absolute px).
        regions_xywh: List of regions (xywh, absolute px).

    Returns:
        np.ndarray of shape ``(len(lines_xywh),)`` dtype int64: region index or -1.
    """
    lines = np.asarray(lines_xywh, dtype=np.float64)
    regions = np.asarray(regions_xywh, dtype=np.float64)
    n_lines = lines.shape[0] if lines.size else 0
    if n_lines == 0 or regions.size == 0:
        return np.full((n_lines,), -1, dtype=np.int64)

    l = _xywh_to_xyxy(lines)      # (n_lines, 4)
    r = _xywh_to_xyxy(regions)    # (n_regions, 4)

    ix1 = np.maximum(l[:, 0:1], r[:, 0])   # (n_lines, n_regions)
    iy1 = np.maximum(l[:, 1:2], r[:, 1])
    ix2 = np.minimum(l[:, 2:3], r[:, 2])
    iy2 = np.minimum(l[:, 3:4], r[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    line_area = (l[:, 2] - l[:, 0]) * (l[:, 3] - l[:, 1])   # (n_lines,)
    safe_area = np.where(line_area > 0, line_area, np.nan)
    overlap = np.nan_to_num(inter / safe_area[:, None], nan=0.0)  # zero-area line -> 0

    max_overlap = overlap.max(axis=1)
    argmax = overlap.argmax(axis=1)                          # first max on ties
    assign = np.where(max_overlap > LINE_OVERLAP_THRESHOLD, argmax, -1)
    return assign.astype(np.int64)


def line_iou_matrix(
    lines_xywh: list[list[float]],
    pred_boxes_xywh: list[list[float]],
    gt_boxes_xywh: list[list[float]],
) -> np.ndarray:
    """Line-IoU matrix between prediction and GT regions of one page.

    Lines are assigned independently: once to prediction regions and once to GT
    regions (:func:`assign_lines_to_regions`). For a pair ``(p, g)``::

        inter = |{line : assigned to p (pred side) and to g (gt side)}|
        union = |{line : assigned to p (pred side) or  to g (gt side)}|
        IoU   = inter / union   (0 when union == 0)

    Vectorized via one-hot matrices and ``np.matmul``.

    Args:
        lines_xywh: Page lines (xywh, absolute px).
        pred_boxes_xywh: Prediction regions (xywh, absolute px).
        gt_boxes_xywh: GT regions (xywh, absolute px).

    Returns:
        np.ndarray of shape ``(P, G)`` float64 — Line-IoU per (pred, GT) pair.
        Empty lines -> matrix of zeros (P×G).
    """
    lines = list(lines_xywh)
    preds = list(pred_boxes_xywh)
    gts = list(gt_boxes_xywh)
    n = len(lines)
    P = len(preds)
    G = len(gts)

    pred_assign = assign_lines_to_regions(lines, preds)   # (n,)
    gt_assign = assign_lines_to_regions(lines, gts)       # (n,)

    pred_oh = np.zeros((n, P), dtype=np.float64)
    gt_oh = np.zeros((n, G), dtype=np.float64)
    if n > 0:
        valid_p = pred_assign >= 0
        pred_oh[np.arange(n)[valid_p], pred_assign[valid_p]] = 1.0
        valid_g = gt_assign >= 0
        gt_oh[np.arange(n)[valid_g], gt_assign[valid_g]] = 1.0

    inter = pred_oh.T @ gt_oh                 # (P, G)
    count_p = pred_oh.sum(axis=0)             # (P,)
    count_g = gt_oh.sum(axis=0)               # (G,)
    union = count_p[:, None] + count_g[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return iou


def build_line_inputs(
    gt: dict[str, Any],
    preds_by_stem: dict[str, list[dict[str, Any]]],
    lines_by_stem: dict[str, list[list[float]]],
) -> list[dict[str, Any]]:
    """Assemble per-image input structures for line-mAP (deterministic order).

    For each GT image (sorted by image id) collects pred/GT regions and computes
    the Line-IoU matrix between them using the page's lines.

    Args:
        gt: Result of :func:`metrics.mAP_iou.load_gt`.
        preds_by_stem: Result of :func:`metrics.mAP_iou.load_preds`
            (``{stem: [{"bbox": [x,y,w,h], "label_id": int}]}``).
        lines_by_stem: Result of :func:`load_lines` (``{stem: [[x,y,w,h], ...]}``).

    Returns:
        Per-image dicts (one per GT image, ordered by image id):
            {"image_id", "pred_boxes" (P,4), "pred_labels" (P,),
             "pred_scores" (P,), "gt_boxes" (G,4), "gt_labels" (G,),
             "line_ious" (P,G)}.
        An image without predictions -> P=0 (correct FN). An image without
        lines -> ``line_ious = zeros((P, G))``.
    """
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in gt["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    inputs: list[dict[str, Any]] = []
    for img in sorted(gt["images"], key=lambda i: i["id"]):
        image_id = img["id"]
        stem = Path(img["file_name"]).stem

        gt_boxes: list[list[float]] = []
        gt_labels: list[int] = []
        for ann in annotations_by_image.get(image_id, []):
            gt_boxes.append(ann["bbox"])
            gt_labels.append(int(ann["category_id"]) - 1)  # 1-indexed -> 0-indexed

        pred_boxes: list[list[float]] = []
        pred_labels: list[int] = []
        pred_scores: list[float] = []
        for region in preds_by_stem.get(stem, []):
            pred_boxes.append(region["bbox"])
            pred_labels.append(region["label_id"])
            pred_scores.append(DEFAULT_SCORE)

        lines = lines_by_stem.get(stem, [])
        line_ious = line_iou_matrix(lines, pred_boxes, gt_boxes)

        inputs.append(
            {
                "image_id": image_id,
                "pred_boxes": np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4),
                "pred_labels": np.asarray(pred_labels, dtype=np.int64),
                "pred_scores": np.asarray(pred_scores, dtype=np.float32),
                "gt_boxes": np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4),
                "gt_labels": np.asarray(gt_labels, dtype=np.int64),
                "line_ious": line_ious,
            }
        )

    logger.info(
        "Built %d aligned line-inputs (%d GT images without predictions)",
        len(inputs), sum(1 for i in inputs if i["pred_boxes"].shape[0] == 0),
    )
    return inputs


def _match_detections(
    per_image: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy COCO-style matching at one IoU threshold for a single class.

    Each element of ``per_image`` is ``{"scores": (P,) float, "line_ious": (P,G)
    float}``, already filtered to one class. Per image, detections are sorted by
    descending score (stable); each detection is matched, in that order, to an
    unmatched GT with the maximum Line-IoU ``>= iou_threshold`` (a GT is used at
    most once). Per-image results are then concatenated and globally sorted by
    descending score (stable).

    Args:
        per_image: List of per-image structures of one class.
        iou_threshold: Current Line-IoU threshold (0.5..0.95).

    Returns:
        (scores_sorted (D,), tp_sorted (D,) bool, num_gt int): scores/TP flags in
        global descending-score order; num_gt is the total GT count of the class.
    """
    all_scores: list[np.ndarray] = []
    all_tp: list[np.ndarray] = []
    num_gt_total = 0

    for img in per_image:
        scores = np.asarray(img["scores"], dtype=np.float32)
        ious = np.asarray(img["line_ious"], dtype=np.float64)
        P = scores.shape[0]
        G = ious.shape[1]
        num_gt_total += G

        tp = np.zeros(P, dtype=bool)
        if P > 0 and G > 0:
            order = np.argsort(-scores, kind="stable")
            matched = np.zeros(G, dtype=bool)
            for idx in order:
                row = ious[idx]
                candidates = np.where((~matched) & (row >= iou_threshold))[0]
                if candidates.size == 0:
                    continue
                best = candidates[np.argmax(row[candidates])]
                tp[idx] = True
                matched[best] = True

        all_scores.append(scores)
        all_tp.append(tp)

    if not all_scores:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=bool), num_gt_total

    scores_cat = np.concatenate(all_scores)
    tp_cat = np.concatenate(all_tp)
    order = np.argsort(-scores_cat, kind="stable")
    return scores_cat[order], tp_cat[order], num_gt_total


def _ap_101_point(tp_sorted: np.ndarray, num_gt: int) -> float:
    """AP at one IoU threshold using 101-point interpolation (pycocotools).

    Computes cumTP/cumFP from the TP flags (already in descending-score order),
    derives recall/precision, smooths precision monotonically non-increasing
    right-to-left, then averages the interpolated precision over 101 recall
    thresholds ``linspace(0, 1, 101)``.

    Args:
        tp_sorted: (D,) bool — TP flags in descending-score order.
        num_gt: Total GT count of the class. 0 -> returns 0.0.

    Returns:
        float — average of the interpolated precision over the 101 recall points.
    """
    if num_gt <= 0 or tp_sorted.size == 0:
        return 0.0

    tp = tp_sorted.astype(np.float64)
    fp = (~tp_sorted).astype(np.float64)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recall = cum_tp / num_gt
    precision = cum_tp / (cum_tp + cum_fp)

    # Monotonic non-increasing precision from right to left.
    pr = np.maximum.accumulate(precision[::-1])[::-1]

    rec_thrs = np.linspace(0.0, 1.0, 101)
    # First index with recall >= threshold; precision at that index (pr is
    # already the running max to the right). If index == D, precision = 0.
    indices = np.searchsorted(recall, rec_thrs, side="left")
    D = recall.shape[0]
    prec_at = np.where(indices < D, pr[np.minimum(indices, D - 1)], 0.0)
    return float(np.mean(prec_at))


def run_map_lines(
    inputs: list[dict[str, Any]],
    class_metrics: bool,
) -> dict[str, Any]:
    """Full line-mAP run: per-class and summary values.

    For each class c ∈ 0..4 and each threshold t ∈ IOU_THRESHOLDS, builds the
    class-filtered per-image set (detections and GT of class c + the sub-matrix
    of Line-IoUs), then :func:`_match_detections` + :func:`_ap_101_point` give
    AP(c, t)::

        ap[c]      = mean_t AP(c, t)                      (0.0 if class has no GT)
        mAP        = mean over classes with GT > 0 of ap[c]
        mAP@0.5    = mean over classes with GT > 0 of AP(c, 0.5)
        mAP@0.75   = mean over classes with GT > 0 of AP(c, 0.75)

    Args:
        inputs: Result of :func:`build_line_inputs`.
        class_metrics: True — return ``map_per_class`` (list of 5 floats in
            CLASS_NAMES order; classes without GT -> 0.0).

    Returns:
        {"map", "map_50", "map_75"} and, if class_metrics, {"map_per_class"}.
    """
    num_gt_per_class = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for img in inputs:
        for g in img["gt_labels"]:
            num_gt_per_class[int(g)] += 1

    ap_by_class: dict[int, dict[float, float]] = {}
    for c in range(len(CLASS_NAMES)):
        per_image: list[dict[str, Any]] = []
        for img in inputs:
            p_idx = np.where(img["pred_labels"] == c)[0]
            g_idx = np.where(img["gt_labels"] == c)[0]
            sub_ious = img["line_ious"][p_idx][:, g_idx]
            sub_scores = img["pred_scores"][p_idx]
            per_image.append({"scores": sub_scores, "line_ious": sub_ious})

        if num_gt_per_class[c] == 0:
            ap_by_class[c] = {t: 0.0 for t in IOU_THRESHOLDS}
            continue

        ap_by_class[c] = {}
        for t in IOU_THRESHOLDS:
            _, tp_sorted, num_gt = _match_detections(per_image, t)
            ap_by_class[c][t] = _ap_101_point(tp_sorted, num_gt)

    per_class_ap = [
        float(np.mean([ap_by_class[c][t] for t in IOU_THRESHOLDS]))
        for c in range(len(CLASS_NAMES))
    ]

    valid_classes = [c for c in range(len(CLASS_NAMES)) if num_gt_per_class[c] > 0]
    if valid_classes:
        mAP = float(np.mean([per_class_ap[c] for c in valid_classes]))
        mAP50 = float(np.mean([ap_by_class[c][0.5] for c in valid_classes]))
        mAP75 = float(np.mean([ap_by_class[c][0.75] for c in valid_classes]))
    else:
        mAP = mAP50 = mAP75 = 0.0

    result: dict[str, Any] = {"map": mAP, "map_50": mAP50, "map_75": mAP75}
    if class_metrics:
        result["map_per_class"] = per_class_ap
    return result


def compute_ap_seg_lines(inputs: list[dict[str, Any]]) -> float:
    """AP with all classes merged into a single class (labels all 0).

    All pred and GT labels are collapsed to 0, detections concatenated across
    images, then one :func:`_match_detections` + :func:`_ap_101_point` pass per
    threshold, averaged over IOU_THRESHOLDS. Analogue of
    :func:`metrics.mAP_iou.compute_ap_seg`.
    """
    per_image: list[dict[str, Any]] = [
        {"scores": img["pred_scores"], "line_ious": img["line_ious"]} for img in inputs
    ]
    num_gt = int(sum(img["gt_boxes"].shape[0] for img in inputs))
    aps: list[float] = []
    for t in IOU_THRESHOLDS:
        _, tp_sorted, _ = _match_detections(per_image, t)
        aps.append(_ap_101_point(tp_sorted, num_gt))
    return float(np.mean(aps))


def compute_report_lines(
    results_dir: Path,
    gt_json: Path,
    chunk_cache_dir: Path,
    dataset: str = "publaynet",
) -> dict[str, Any]:
    """Compute the full line-mAP report for a method's results folder.

    Args:
        results_dir: Directory with prediction JSONs.
        gt_json: Path to the COCO ground-truth file.
        chunk_cache_dir: Directory with line files (``chunk_cache/``).
        dataset: Dataset name controlling the bbox coordinate convention
            (see :data:`metrics.mAP_iou.NORMALIZED_BBOX_DATASETS`).

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
    lines_by_stem = load_lines(chunk_cache_dir, gt_stems, stem_to_size)
    inputs = build_line_inputs(gt, preds_by_stem, lines_by_stem)
    per_class = run_map_lines(inputs, class_metrics=True)

    report: dict[str, Any] = {
        "path": results_dir.name,
        "mAP": float(per_class["map"]),
        "mAP@0.5": float(per_class["map_50"]),
        "mAP@0.75": float(per_class["map_75"]),
    }
    map_per_class = per_class["map_per_class"]
    for i, name in enumerate(CLASS_NAMES):
        report[f"AP ({name})"] = float(map_per_class[i])

    report["AP (seg)"] = compute_ap_seg_lines(inputs)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, compute the report, write JSON + print to stdout."""
    parser = argparse.ArgumentParser(
        description="Compute COCO-style mAP@[IoU 0.5:0.95] with line-based IoU "
                    "for layout predictions vs COCO GT."
    )
    parser.add_argument("results_dir", type=str, help="Directory with prediction JSONs")
    parser.add_argument("gt_json", type=str, help="Path to COCO ground-truth JSON")
    parser.add_argument("chunk_cache_dir", type=str, help="Directory with line JSONs (chunk_cache)")
    parser.add_argument(
        "--dataset", type=str, default="publaynet",
        choices=sorted(NORMALIZED_BBOX_DATASETS),
        help="Dataset name controlling bbox coordinate convention "
             "(normalized bboxes are multiplied by page dimensions); default: publaynet",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <cwd>/<results_dir_name>_mAP_lines.json)",
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

    chunk_cache_dir = Path(args.chunk_cache_dir)
    if not chunk_cache_dir.is_dir():
        logger.error("chunk_cache_dir is not a directory: %s", chunk_cache_dir)
        return 1

    report = compute_report_lines(results_dir, gt_json, chunk_cache_dir, args.dataset)

    output_path = (
        Path(args.output) if args.output else Path.cwd() / f"{results_dir.name}_mAP_lines.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Report written to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

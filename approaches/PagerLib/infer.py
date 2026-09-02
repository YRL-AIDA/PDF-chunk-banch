"""PagerLib document layout detection on PDFs.

pagerlib parses a PDF directly (no rasterization to images): ``FileInput``
loads the document, ``PDFIMGExtractor`` renders each page to an array used as
visual context, and ``Rows2Regions`` runs a GLAM graph model over the text rows
to cluster them into typed regions. The page layout is then emitted in the
unified project format (see ``approaches/README.md`` §8):

    {
      "pages": [
        {
          "num": 1,
          "regions": [{"bbox": [x, y, w, h], "label": "text"}],
          "metainfo": {"height": 794.0, "width": 596.0}
        }
      ]
    }

with ``bbox`` normalized to [0, 1] (fractions of the page width/height) and
PubLayNet class names: text, title, list, table, figure.

Region vocabulary mapping
-------------------------
``Rows2Regions`` (see its ``CLASSES`` in
``pagerlib/extractors/page_extractor/rows2regions/rows2regions.py``) emits five
labels: ``text``, ``header``, ``table``, ``figure``, ``other``. ``Image``
objects (figures detected as embedded pictures) carry no ``label`` and become
``figure``. These are mapped to the canonical PubLayNet vocabulary used across
this project (``metrics/mAP_iou.py``) via ``LABEL_ALIASES``:

    text   -> text
    header -> title   (a section heading is the closest PubLayNet class)
    table  -> table
    figure -> figure
    other  -> (dropped; outside the PubLayNet vocabulary)

pagerlib has no ``list`` class, so list items are absorbed into ``text`` by the
model — this is a limitation of the approach, not a bug of this script.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pagerlib.dtypes import Image
from pagerlib.extractors.page_extractor import PDFIMGExtractor, Rows2Regions
from pagerlib.file_input import FileInput

# Canonical PubLayNet class order used by every approach in this project.
CLASS_NAMES = ["text", "title", "list", "table", "figure"]

# pagerlib -> PubLayNet. 'header' (a section heading) maps to 'title'; 'other'
# is outside PubLayNet and is dropped downstream. pagerlib has no 'list' class.
LABEL_ALIASES = {
    "text": "text",
    "header": "title",
    "table": "table",
    "figure": "figure",
    "list": "list",
    "other": "other",  # dropped: not a PubLayNet class
}

PDF_EXTS = {".pdf"}


def load_model():
    """Build the pagerlib pipeline components once (reused across all files).

    pagerlib works directly on PDF: ``FileInput`` parses the document,
    ``PDFIMGExtractor`` renders each page to an array (used by ``Rows2Regions``
    as visual context), and ``Rows2Regions`` runs the GLAM graph model that
    clusters rows into typed regions.

    Returns:
        A ``(FileInput, PDFIMGExtractor, Rows2Regions)`` triple. ``Rows2Regions``
        instantiates its torch model internally (see ``get_load_model``), so this
        call is the expensive one.
    """
    file_input = FileInput()
    pdf_img_extractor = PDFIMGExtractor()
    rows2regions = Rows2Regions()
    return file_input, pdf_img_extractor, rows2regions


def _region_label(region: Any) -> str:
    """Return the pagerlib label of a region, mirroring ``example.py``.

    A ``'label'`` key in ``region.data`` wins; otherwise an ``Image`` region is
    a ``figure``; anything else is ``'other'``.

    Args:
        region: a pagerlib physical element (``Region`` or ``Image``).

    Returns:
        The raw pagerlib label string.
    """
    data = getattr(region, "data", None)
    if isinstance(data, dict) and "label" in data:
        return data["label"]
    return "other"


def _segment_to_normalized_bbox(
    segment: Any, page_width: float, page_height: float
) -> list[float]:
    """Convert a pagerlib ``ImageSegment`` to a normalized ``[x, y, w, h]`` box.

    ``ImageSegment.get_segment_p_size()`` returns absolute PDF-point coordinates
    ``{x_top_left, y_top_left, width, height}``; here they are divided by the
    page width/height to produce fractions in [0, 1].

    Args:
        segment: a pagerlib ``ImageSegment``.
        page_width: page width in PDF points (``page.segment.width``).
        page_height: page height in PDF points (``page.segment.height``).

    Returns:
        ``[x, y, w, h]`` rounded to 6 decimals.
    """
    p = segment.get_segment_p_size()
    x0 = p["x_top_left"]
    y0 = p["y_top_left"]
    w = p["width"]
    h = p["height"]
    return [
        round(x0 / page_width, 6),
        round(y0 / page_height, 6),
        round(w / page_width, 6),
        round(h / page_height, 6),
    ]


def predict_page(page: Any, page_idx: int) -> dict:
    """Extract regions from one pagerlib page and normalize them to the project
    layout schema.

    Following ``approaches/PagerLib/example.py``, ``page.children[0]`` is a
    service element (the page image / page frame) and is skipped; the layout
    regions are ``page.children[1:]``.

    Args:
        page: a pagerlib ``Page`` after ``Rows2Regions.extract``.
        page_idx: 1-based page number.

    Returns:
        ``{"num": page_idx, "regions": [...], "metainfo": {"height", "width"}}``
        with pagerlib sizes in PDF points (no 72/dpi conversion is needed,
        unlike the rasterized approaches, because pagerlib sizes are already in
        PDF points).
    """
    width = page.segment.width
    height = page.segment.height

    regions: list[dict] = []
    for region in page.children[1:]:
        raw_label = _region_label(region)
        label = LABEL_ALIASES.get(raw_label)
        if label is None or label not in CLASS_NAMES:
            # 'other' and anything outside the PubLayNet vocabulary is dropped.
            continue
        bbox = _segment_to_normalized_bbox(region.segment, width, height)
        regions.append({"bbox": bbox, "label": label})

    # Deterministic reading order: top-to-bottom, then left-to-right.
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    return {
        "num": page_idx,
        "regions": regions,
        "metainfo": {"height": height, "width": width},
    }


def infer_pdf(
    pdf_path: Path,
    file_input: FileInput,
    pdf_img_extractor: PDFIMGExtractor,
    rows2regions: Rows2Regions,
) -> dict:
    """Run the pagerlib pipeline on one PDF and return the project JSON layout.

    Args:
        pdf_path: path to a PDF file.
        file_input: a ``FileInput`` instance.
        pdf_img_extractor: a ``PDFIMGExtractor`` instance.
        rows2regions: a ``Rows2Regions`` instance.

    Returns:
        ``{"pages": [{"num", "regions", "metainfo"}, ...]}``.
    """
    prdf = file_input(str(pdf_path))
    pdf_img_extractor.extract(prdf)
    rows2regions.extract(prdf)

    pages = []
    for page_idx, page in enumerate(prdf.data["pages"], start=1):
        pages.append(predict_page(page, page_idx))
    return {"pages": pages}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PagerLib document layout detection on PDFs "
        "(FileInput + PDFIMGExtractor + Rows2Regions)"
    )
    parser.add_argument("input_dir", type=str, help="Directory with PDFs")
    parser.add_argument(
        "results_dir", type=str, help="Directory to write result JSONs"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # pagerlib parses PDFs directly (FileInput), so only .pdf inputs are taken.
    files = sorted(input_dir.glob("*.pdf"))
    files = sorted(files)
    if not files:
        print(f"No PDFs found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(
        "Loading pagerlib pipeline "
        "(FileInput + PDFIMGExtractor + Rows2Regions)..."
    )
    file_input, pdf_img_extractor, rows2regions = load_model()

    total = len(files)
    succeeded = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        json_path = results_dir / f"{path.stem}.json"
        if json_path.exists():
            print(f"[{i}/{total}] exists: {json_path.name} ")
            continue
        try:
            result = infer_pdf(path, file_input, pdf_img_extractor, rows2regions)
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

import json
import base64
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBox, LTImage, LTFigure, LTTextLine, LTChar
import sys

def pdf_to_chunks(pdf_path: str | Path, output_path: Optional[str | Path] = None) -> dict:
    pdf_path = Path(pdf_path)
    result = {"pages": []}
    laparams = LAParams()

    for page_layout in extract_pages(pdf_path, laparams=laparams):
        page_num = page_layout.pageid
        pw = page_layout.width
        ph = page_layout.height

        chunks = []
        images = []
        _walk(page_layout, chunks, images, pw, ph)

        result["pages"].append({
            "num": page_num,
            "chunks": chunks,
            "images": images,
            "metainfo": {"height": ph, "width": pw},
        })

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def _xywh(bbox, pw, ph):
    x0, y0, x1, y1 = bbox
    return [
        round(x0 / pw, 6),
        round((ph - y1) / ph, 6),
        round((x1 - x0) / pw, 6),
        round((y1 - y0) / ph, 6),
    ]


def _walk(element, chunks, images, pw, ph):
    for child in element:
        if isinstance(child, LTTextLine):
            text = child.get_text().strip()
            if text:
                chunks.append({"bbox": _xywh(child.bbox, pw, ph), "text": text})
        elif isinstance(child, (LTTextBox, LTFigure)):
            _walk(child, chunks, images, pw, ph)
        elif isinstance(child, LTImage):
            bbox = _xywh(child.bbox, pw, ph)
            try:
                raw = child.stream.get_rawdata()
                img_b64 = base64.b64encode(raw).decode("ascii")
                images.append({"bbox": bbox, "base64": img_b64})
            except Exception:
                pass


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python create_chunk_json.py <pdf_path> <json_path>")
        sys.exit(1)
    
    src_pdf = sys.argv[1]
    src_json = sys.argv[2]
    pdf_to_chunks(src_pdf, src_json)
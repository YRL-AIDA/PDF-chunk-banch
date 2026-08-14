import json
import sys
from pathlib import Path

from pdf2image import convert_from_path
from PIL import ImageDraw


def print_pdf(pdf_path: str | Path, json_path: str | Path, output_dir: str | Path = "output", dpi: int = 200) -> None:
    pdf_path = Path(pdf_path)
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    pages_images = convert_from_path(pdf_path, dpi=dpi)

    for page_data in data["pages"]:
        page_idx = page_data["num"] - 1
        if page_idx >= len(pages_images):
            continue

        img = pages_images[page_idx].copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        for chunk in page_data.get("chunks", []):
            x, y, w, h = chunk["bbox"]
            px, py = x * iw, y * ih
            pw, ph = w * iw, h * ih
            draw.rectangle([px, py, px + pw, py + ph], outline="#00ff00", width=2)

        for img_item in page_data.get("images", []):
            x, y, w, h = img_item["bbox"]
            px, py = x * iw, y * ih
            pw, ph = w * iw, h * ih
            draw.rectangle([px, py, px + pw, py + ph], outline="#ff0000", width=2)

        img.save(output_dir / f"page_{page_data['num']:04d}.png")

    print(f"Saved {len(pages_images)} pages to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python print.py <pdf_path> <json_path> [output_dir] [dpi]")
        sys.exit(1)

    src_pdf = sys.argv[1]
    src_json = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    print_pdf(src_pdf, src_json, out_dir, dpi)

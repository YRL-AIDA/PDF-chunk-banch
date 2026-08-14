import json
import sys
from pathlib import Path

from pdf2image import convert_from_path
from PIL import ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont


LABEL_COLORS: dict[str, str] = {
    "text":    "#00cc00",
    "title":   "#0066ff",
    "list":    "#ff8800",
    "table":   "#cc0000",
    "figure":  "#8833cc",
}
FALLBACK_COLOR = "#999999"


def _load_font(size: int = 14) -> FreeTypeFont | ImageFont.ImageFont:
    import os as _os
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    if _os.name == "nt":
        candidates.insert(0, "C:\\Windows\\Fonts\\arialbd.ttf")
    for path in candidates:
        if _os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def print_results(
    pdf_path: str | Path,
    json_path: str | Path,
    output_dir: str | Path = "output",
    dpi: int = 200,
    draw_labels: bool = True,
) -> None:
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

        for region in page_data.get("regions", []):
            x, y, w, h = region["bbox"]
            label = region.get("label", "?")
            color = LABEL_COLORS.get(label, FALLBACK_COLOR)

            px, py = x * iw, y * ih
            pw, ph = w * iw, h * ih

            draw.rectangle([px, py, px + pw, py + ph], outline=color, width=2)

            if draw_labels:
                font = _load_font()
                text_x = px
                text_y = max(0, py - 16)
                draw.rectangle(
                    [text_x, text_y, text_x + len(label) * 10, text_y + 14],
                    fill=color,
                )
                draw.text((text_x, text_y), label, fill="#ffffff", font=font)

        img.save(output_dir / f"page_{page_data['num']:04d}.png")

    print(f"Saved {len(data['pages'])} page(s) to {output_dir.resolve()}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python print_results.py <pdf_path> <json_path> [output_dir] [dpi]")
        sys.exit(1)

    src_pdf = sys.argv[1]
    src_json = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    print_results(src_pdf, src_json, out_dir, dpi)

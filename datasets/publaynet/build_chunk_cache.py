import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.create_chunk_json import pdf_to_chunks


def main():
    if len(sys.argv) != 3:
        print("Usage: python datasets/publaynet/build_chunk_cache.py <pdf_dir> <cache_dir>")
        sys.exit(1)

    pdf_dir = Path(sys.argv[1])
    cache_dir = Path(sys.argv[2])
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    total = len(pdf_files)
    succeeded = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, start=1):
        json_path = cache_dir / f"{pdf_path.stem}.json"
        try:
            pdf_to_chunks(pdf_path, json_path)
            print(f"[{i}/{total}] processed: {pdf_path.name} -> {json_path}")
            succeeded += 1
        except Exception as e:
            print(f"[{i}/{total}] FAILED: {pdf_path.name} — {e}")
            failed += 1

    print(f"Done. Processed {total} PDFs ({succeeded} succeeded, {failed} failed).")


if __name__ == "__main__":
    main()

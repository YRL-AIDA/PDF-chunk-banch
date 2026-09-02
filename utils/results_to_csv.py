"""Build a summary CSV table from results_*_mAP*.json files.

Scans a directory for files named ``results_{approach}_{dataset}_{metric}.json``
where ``metric`` is ``mAP`` or ``mAP_lines``, and merges them into a single
table: one row per approach, columns ``mAP``, ``AP(seg)``, ``mAP_lines``,
``AP_lines`` (segmentation AP of the corresponding metric variant).
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: results_{approach}_{dataset}_{metric}.json; approach may contain underscores.
FILE_RE = re.compile(
    r"^results_(?P<approach>.+)_(?P<dataset>[^_]+)_(?P<metric>mAP_lines|mAP)\.json$"
)

#: (csv column, metric file suffix, key inside the JSON).
COLUMN_SPECS: list[tuple[str, str, str]] = [
    ("mAP", "mAP", "mAP"),
    ("AP(seg)", "mAP", "AP (seg)"),
    ("mAP_lines", "mAP_lines", "mAP"),
    ("AP_lines", "mAP_lines", "AP (seg)"),
]


def collect_rows(
    results_dir: Path,
    dataset_filter: str | None = None,
) -> dict[str, dict[str, float | None]]:
    """Collect metric values per approach from results_* JSON files.

    Args:
        results_dir: Directory with ``results_*.json`` files.
        dataset_filter: If set, only files with this dataset are used.

    Returns:
        Mapping ``row name -> {column name -> value or None}``. Row name is
        the approach name, suffixed with the dataset when several datasets
        are present.
    """
    entries: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    datasets: set[str] = set()

    for path in sorted(results_dir.glob("results_*.json")):
        match = FILE_RE.match(path.name)
        if match is None:
            LOGGER.debug("Skipping unrecognized file: %s", path.name)
            continue
        approach = match.group("approach")
        dataset = match.group("dataset")
        metric = match.group("metric")

        if dataset_filter is not None and dataset != dataset_filter:
            continue

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to read %s: %s", path.name, exc)
            continue

        datasets.add(dataset)
        entries.setdefault(approach, {})[(dataset, metric)] = data

    multi_dataset = len(datasets) > 1
    rows: dict[str, dict[str, float | None]] = {}
    for approach, per_dataset_metric in entries.items():
        for (dataset, metric), data in per_dataset_metric.items():
            name = f"{approach} ({dataset})" if multi_dataset else approach
            row = rows.setdefault(name, {col: None for col, _, _ in COLUMN_SPECS})
            for column, wanted_metric, key in COLUMN_SPECS:
                if metric != wanted_metric:
                    continue
                value = data.get(key)
                if isinstance(value, (int, float)):
                    row[column] = float(value)
    return rows


def write_csv(
    rows: dict[str, dict[str, float | None]],
    output_path: Path,
    precision: int = 4,
) -> None:
    """Write collected rows to a CSV file.

    Args:
        rows: Mapping produced by :func:`collect_rows`.
        output_path: Destination CSV path.
        precision: Number of digits after the decimal point.
    """
    columns = [col for col, _, _ in COLUMN_SPECS]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["approach", *columns])
        for name in sorted(rows):
            writer.writerow(
                [name]
                + [
                    "" if rows[name][col] is None
                    else f"{rows[name][col]:.{precision}f}"
                    for col in columns
                ]
            )
    LOGGER.info("Wrote %d row(s) to %s", len(rows), output_path)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Directory with results_*.json files (default: project root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-dir>/results_summary.csv)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only include this dataset (e.g. PubLayNet)",
    )
    parser.add_argument("--precision", type=int, default=4, help="Digits after decimal point")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output = args.output or args.results_dir / "results_summary.csv"
    rows = collect_rows(args.results_dir, args.dataset)
    if not rows:
        LOGGER.error("No results_*.json files found in %s", args.results_dir)
        return 1
    write_csv(rows, output, args.precision)
    return 0


if __name__ == "__main__":
    sys.exit(main())

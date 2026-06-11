from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate PyTorch Chrome trace events into CSV summaries."
    )
    parser.add_argument(
        "trace_input",
        type=Path,
        help="Trace JSON file, .json.gz file, or directory containing traces.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/trace_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of events to keep per trace/category.",
    )
    return parser.parse_args()


def trace_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted([*path.glob("*.json"), *path.glob("*.json.gz")])


def load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    with path.open() as handle:
        return json.load(handle)


def batch_size_from_name(path: Path) -> int | None:
    stem = path.name.replace(".json.gz", "").replace(".json", "")
    marker = "bs"
    if marker not in stem:
        return None
    try:
        return int(stem.rsplit(marker, 1)[1])
    except ValueError:
        return None


def summarize_trace(path: Path, top_n: int) -> list[dict[str, object]]:
    trace = load_trace(path)
    events = trace.get("traceEvents", [])
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )

    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name") or "unknown"
        category = event.get("cat") or "uncategorized"
        duration_us = float(event.get("dur") or 0.0)
        key = (category, name)
        grouped[key]["count"] += 1
        grouped[key]["duration_us"] += duration_us

    rows: list[dict[str, object]] = []
    by_category: dict[str, list[tuple[tuple[str, str], dict[str, float]]]] = defaultdict(list)
    for item in grouped.items():
        by_category[item[0][0]].append(item)

    for category, items in by_category.items():
        ranked = sorted(
            items,
            key=lambda item: item[1]["duration_us"],
            reverse=True,
        )[:top_n]
        for (cat, name), stats in ranked:
            rows.append(
                {
                    "trace": str(path),
                    "batch_size": batch_size_from_name(path),
                    "category": cat,
                    "name": name,
                    "count": int(stats["count"]),
                    "total_duration_ms": round(stats["duration_us"] / 1000, 4),
                    "mean_duration_us": round(
                        stats["duration_us"] / max(stats["count"], 1),
                        4,
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    paths = trace_paths(args.trace_input)
    if not paths:
        raise SystemExit(f"No trace files found at {args.trace_input}")

    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(summarize_trace(path, args.top_n))

    fieldnames = [
        "trace",
        "batch_size",
        "category",
        "name",
        "count",
        "total_duration_ms",
        "mean_duration_us",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

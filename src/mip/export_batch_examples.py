from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export random and width-bucketed SROIE batch examples for the portfolio."
    )
    parser.add_argument("--dataset-id", default="priyank-m/SROIE_2019_text_recognition")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-width",
        type=int,
        default=100,
        help="Avoid selecting only very short crops for the similar-width example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("portfolio_data"),
    )
    parser.add_argument(
        "--public-output-dir",
        type=Path,
        default=Path("portfolio_git/frontend/public/inference-profiler"),
    )
    return parser.parse_args()


def row_record(dataset: Any, index: int) -> dict[str, Any]:
    image = dataset[index]["image"].convert("RGB")
    text = str(dataset[index].get("text", ""))
    return {
        "index": index,
        "image_obj": image,
        "text": text,
        "length": len(text),
        "width": image.width,
        "height": image.height,
        "aspect_ratio": round(image.width / max(image.height, 1), 3),
    }


def choose_similar_width_rows(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    min_width: int,
) -> list[dict[str, Any]]:
    candidates = sorted(
        [row for row in rows if row["width"] >= min_width],
        key=lambda row: (row["width"], row["index"]),
    )
    if len(candidates) < batch_size:
        raise SystemExit(
            f"Only {len(candidates)} rows have width >= {min_width}; need {batch_size}."
        )

    best_window: list[dict[str, Any]] | None = None
    best_score: tuple[float, float, int] | None = None

    for start in range(0, len(candidates) - batch_size + 1):
        window = candidates[start : start + batch_size]
        widths = [row["width"] for row in window]
        lengths = [row["length"] for row in window]
        width_range = max(widths) - min(widths)
        # Prefer visually similar crop widths, but avoid windows where every text
        # example is effectively identical in length.
        length_range = max(lengths) - min(lengths)
        length_penalty = 0 if length_range >= 4 else 10
        mean_width_distance = abs((sum(widths) / len(widths)) - 180)
        score = (width_range + length_penalty, mean_width_distance, start)
        if best_score is None or score < best_score:
            best_score = score
            best_window = window

    if best_window is None:
        raise SystemExit("Could not select a similar-width batch.")
    return best_window


def export_rows(
    rows: list[dict[str, Any]],
    *,
    name: str,
    output_dir: Path,
    public_output_dir: Path,
) -> list[dict[str, Any]]:
    image_dir = output_dir / "images" / "batch_examples"
    public_image_dir = public_output_dir / "images" / "batch_examples"
    image_dir.mkdir(parents=True, exist_ok=True)
    public_image_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        filename = f"{name}_{position:02d}.png"
        row["image_obj"].save(image_dir / filename)
        row["image_obj"].save(public_image_dir / filename)
        exported.append(
            {
                "index": row["index"],
                "text": row["text"],
                "length": row["length"],
                "width": row["width"],
                "height": row["height"],
                "aspect_ratio": row["aspect_ratio"],
                "image": f"images/batch_examples/{filename}",
            }
        )
    return exported


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_id, split=f"{args.dataset_split}[:{args.sample_limit}]")
    rows = [row_record(dataset, index) for index in range(len(dataset))]
    rng = random.Random(args.seed)

    random_rows = [rows[index] for index in rng.sample(range(len(rows)), args.batch_size)]
    width_rows = choose_similar_width_rows(
        rows,
        batch_size=args.batch_size,
        min_width=args.min_width,
    )

    payload = {
        "source_dataset": args.dataset_id,
        "random_seed": args.seed,
        "batch_size": args.batch_size,
        "random_batch": export_rows(
            random_rows,
            name="random",
            output_dir=args.output_dir,
            public_output_dir=args.public_output_dir,
        ),
        "similar_width_batch": export_rows(
            width_rows,
            name="similar_width",
            output_dir=args.output_dir,
            public_output_dir=args.public_output_dir,
        ),
    }

    data_dir = args.output_dir / "data"
    public_data_dir = args.public_output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    public_data_dir.mkdir(parents=True, exist_ok=True)
    for path in [data_dir / "batch_examples.json", public_data_dir / "batch_examples.json"]:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(path)

    widths = [row["width"] for row in payload["similar_width_batch"]]
    print(f"similar_width_range={min(widths)}..{max(widths)}")


if __name__ == "__main__":
    main()

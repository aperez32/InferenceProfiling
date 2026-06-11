from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

from mip.plot_results import write_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark runs and estimate cost from throughput."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RESULTS_CSV"),
        required=True,
        help="Run label and path to results.csv. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--batch-metrics",
        action="append",
        nargs=2,
        metavar=("LABEL", "BATCH_METRICS_CSV"),
        help="Run label and path to batch_metrics.csv for correlation plots.",
    )
    parser.add_argument(
        "--gpu-hour-cost",
        type=float,
        default=0.50,
        help="Assumed GPU cost in dollars per hour for cost estimates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/comparison"),
        help="Directory for generated comparison artifacts.",
    )
    return parser.parse_args()


def load_results(runs: list[list[str]], gpu_hour_cost: float) -> pd.DataFrame:
    frames = []
    for label, path in runs:
        frame = pd.read_csv(path)
        frame["run"] = label
        frame["gpu_hour_cost"] = gpu_hour_cost
        frame["cost_per_1k_images"] = (
            (1000 / frame["images_per_second"]) / 3600 * gpu_hour_cost
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_batch_metrics(batch_metrics: list[list[str]] | None) -> pd.DataFrame | None:
    if not batch_metrics:
        return None
    frames = []
    for label, path in batch_metrics:
        frame = pd.read_csv(path)
        frame["run"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.run, args.gpu_hour_cost)
    results.to_csv(args.output_dir / "comparison_results.csv", index=False)

    latency = px.line(
        results,
        x="batch_size",
        y="latency_ms_mean",
        error_y="latency_ms_stddev" if "latency_ms_stddev" in results.columns else None,
        color="run",
        markers=True,
        title="Mean Latency by Batching Strategy",
        labels={"latency_ms_mean": "Mean latency (ms)", "batch_size": "Batch size"},
    )
    write_figure(latency, args.output_dir / "latency_by_strategy.html")

    throughput = px.line(
        results,
        x="batch_size",
        y="images_per_second",
        error_y=(
            "images_per_second_stddev"
            if "images_per_second_stddev" in results.columns
            else None
        ),
        color="run",
        markers=True,
        title="Throughput by Batching Strategy",
        labels={"images_per_second": "Images / second", "batch_size": "Batch size"},
    )
    write_figure(throughput, args.output_dir / "throughput_by_strategy.html")

    cost = px.line(
        results,
        x="batch_size",
        y="cost_per_1k_images",
        color="run",
        markers=True,
        title=f"Estimated GPU Cost per 1k Images (${args.gpu_hour_cost:.2f}/GPU-hour)",
        labels={
            "cost_per_1k_images": "Cost per 1k images ($)",
            "batch_size": "Batch size",
        },
    )
    write_figure(cost, args.output_dir / "cost_per_1k_images.html")

    tail = px.line(
        results,
        x="batch_size",
        y=["latency_ms_p50", "latency_ms_p95", "latency_ms_p99"],
        color="run",
        markers=True,
        title="Latency Distribution Summary",
        labels={"value": "Latency (ms)", "batch_size": "Batch size"},
    )
    write_figure(tail, args.output_dir / "tail_latency_by_strategy.html")

    batch_df = load_batch_metrics(args.batch_metrics)
    if batch_df is not None:
        corr = px.scatter(
            batch_df,
            x="generated_tokens_max",
            y="latency_ms",
            color="run",
            facet_col="batch_size",
            facet_col_wrap=3,
            title="Batch Latency vs Max Generated Tokens",
            labels={
                "generated_tokens_max": "Max generated tokens in batch",
                "latency_ms": "Latency (ms)",
            },
        )
        write_figure(corr, args.output_dir / "latency_vs_generated_tokens.html")

        text_corr = px.scatter(
            batch_df,
            x="text_length_max",
            y="latency_ms",
            color="run",
            facet_col="batch_size",
            facet_col_wrap=3,
            title="Batch Latency vs Max Ground-Truth Text Length",
            labels={
                "text_length_max": "Max text length in batch",
                "latency_ms": "Latency (ms)",
            },
        )
        write_figure(text_corr, args.output_dir / "latency_vs_text_length.html")


if __name__ == "__main__":
    main()

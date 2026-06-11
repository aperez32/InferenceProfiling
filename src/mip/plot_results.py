from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import plotly.express as px


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HTML visualizations from benchmark CSV outputs."
    )
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--trace-summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/figures"),
        help="Directory for generated HTML charts.",
    )
    return parser.parse_args()


def write_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Wrote {path}")


def shorten_event_name(name: str, max_length: int = 72) -> str:
    name = str(name)
    compacted = re.sub(r"\s+", " ", name).strip()

    if compacted.startswith("void "):
        compacted = compacted.removeprefix("void ")
    if compacted.startswith("std::enable_if"):
        match = re.search(r"internal::([^<(]+)", compacted)
        if match:
            compacted = f"internal::{match.group(1)}"
    if "cutlass::Kernel" in compacted:
        match = re.search(r"cutlass::Kernel\d*<([^>]+)>", compacted)
        if match:
            compacted = f"cutlass::{match.group(1)}"
    if "pytorch_flash::flash_fwd_kernel" in compacted:
        match = re.search(r"Flash_fwd_kernel_traits<([^>]+)>", compacted)
        compacted = (
            f"pytorch_flash::flash_fwd_kernel<{match.group(1)}>"
            if match
            else "pytorch_flash::flash_fwd_kernel"
        )
    if "at::native::" in compacted:
        compacted = compacted.replace("at::native::(anonymous namespace)::", "at::native::")
        match = re.search(r"at::native::([^<(]+)", compacted)
        if match:
            compacted = f"at::native::{match.group(1)}"

    if len(compacted) <= max_length:
        return compacted
    return f"{compacted[: max_length - 1].rstrip()}..."


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latency = px.line(
        df,
        x="batch_size",
        y=["latency_ms_mean", "latency_ms_p50", "latency_ms_p95"],
        markers=True,
        title="TrOCR Latency by Batch Size",
        labels={"value": "Latency (ms)", "batch_size": "Batch size"},
    )
    write_figure(latency, args.output_dir / "latency.html")

    throughput = px.line(
        df,
        x="batch_size",
        y="images_per_second",
        markers=True,
        title="TrOCR Throughput by Batch Size",
        labels={"images_per_second": "Images / second", "batch_size": "Batch size"},
    )
    write_figure(throughput, args.output_dir / "throughput.html")

    memory_columns = [
        column
        for column in ["cuda_peak_allocated_mb", "cuda_peak_reserved_mb"]
        if column in df.columns
    ]
    if memory_columns:
        memory = px.line(
            df,
            x="batch_size",
            y=memory_columns,
            markers=True,
            title="CUDA Memory by Batch Size",
            labels={"value": "Memory (MiB)", "batch_size": "Batch size"},
        )
        write_figure(memory, args.output_dir / "cuda_memory.html")

    if args.trace_summary and args.trace_summary.exists():
        trace_df = pd.read_csv(args.trace_summary)
        kernel_df = trace_df[
            trace_df["category"].astype(str).str.contains(
                "kernel|cuda",
                case=False,
                na=False,
            )
        ].copy()
        if not kernel_df.empty:
            top_kernel_df = (
                kernel_df.sort_values("total_duration_ms", ascending=False)
                .groupby("batch_size")
                .head(15)
                .copy()
            )
            top_kernel_df["event_label"] = top_kernel_df["name"].map(shorten_event_name)
            kernels = px.bar(
                top_kernel_df,
                x="total_duration_ms",
                y="event_label",
                color="category",
                facet_col="batch_size",
                facet_col_wrap=2,
                orientation="h",
                title="Top CUDA/Kernel Events by Total Duration",
                custom_data=[
                    "name",
                    "category",
                    "batch_size",
                    "count",
                    "mean_duration_us",
                ],
                labels={
                    "total_duration_ms": "Total duration (ms)",
                    "event_label": "Event",
                },
            )
            kernels.update_traces(
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Category: %{customdata[1]}<br>"
                    "Batch size: %{customdata[2]}<br>"
                    "Total duration: %{x:.4f} ms<br>"
                    "Count: %{customdata[3]}<br>"
                    "Mean duration: %{customdata[4]:.4f} us"
                    "<extra></extra>"
                )
            )
            kernels.update_yaxes(matches=None, showticklabels=True)
            kernels.update_layout(margin={"l": 180, "r": 24, "t": 80, "b": 56})
            write_figure(kernels, args.output_dir / "top_kernel_events.html")


if __name__ == "__main__":
    main()

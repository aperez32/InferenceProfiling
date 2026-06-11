from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile TrOCR generation across batch sizes."
    )
    parser.add_argument(
        "--model-id",
        default="microsoft/trocr-base-printed",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to benchmark.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=1,
        help=(
            "Number of iterations to record with the PyTorch profiler. Keep this "
            "small for generation models because Chrome traces grow quickly."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="CUDA autocast dtype. CPU always uses fp32.",
    )
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument(
        "--save-sample-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the generated synthetic OCR images to the output directory.",
    )
    parser.add_argument(
        "--repeat-image",
        action="store_true",
        help="Build each batch by repeating the first synthetic image.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Optional Hugging Face dataset id for real-image profiling.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to load when --dataset-id is set.",
    )
    parser.add_argument(
        "--dataset-sample-limit",
        type=int,
        default=512,
        help="Maximum dataset rows to load for random batch sampling.",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Dataset column containing PIL-compatible images.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Dataset column containing ground-truth text, if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset batch sampling.",
    )
    parser.add_argument(
        "--batching",
        choices=["random", "sorted-text-length", "bucketed-text-length"],
        default="random",
        help="Batch sampling strategy for dataset or synthetic samples.",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=16,
        help="Number of text-length buckets for --batching bucketed-text-length.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/trocr-base-printed"),
        help="Directory for metrics, traces, and profiler tables.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU run when CUDA is unavailable.",
    )
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record tensor shapes in the profiler trace. This makes traces larger.",
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record memory events in the profiler trace. This makes traces larger.",
    )
    parser.add_argument(
        "--compress-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gzip Chrome traces after export to reduce disk usage.",
    )
    return parser.parse_args()


def make_synthetic_ocr_images(
    count: int,
    *,
    width: int,
    height: int,
) -> list[Image.Image]:
    font = ImageFont.load_default()
    samples = [
        "THE QUICK BROWN FOX",
        "INFERENCE PROFILING",
        "BATCH SIZE STUDY",
        "CUDA KERNEL TRACE",
        "MODEL BOTTLENECKS",
        "VISION ENCODER DECODER",
        "LATENCY THROUGHPUT",
        "HUGGING FACE TROCR",
    ]
    images: list[Image.Image] = []

    for idx in range(count):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        text = samples[idx % len(samples)]
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = max(8, (width - text_width) // 2)
        y = max(8, (height - text_height) // 2)
        draw.text((x, y), text, fill="black", font=font)
        images.append(image)

    return images


def autocast_context(torch_module, device: str, dtype_name: str):
    if device != "cuda" or dtype_name == "fp32":
        return nullcontext()

    dtype = {
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }[dtype_name]
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def synchronize_if_cuda(torch_module, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()


def percentile(values: Iterable[float], pct: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return math.nan
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def sample_variance(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def write_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_sample_images(
    images: Sequence[Image.Image],
    output_dir: Path,
    *,
    prefix: str = "sample",
) -> None:
    sample_dir = output_dir / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images, start=1):
        image.save(sample_dir / f"{prefix}_{idx:02d}.png")


def compress_file(path: Path) -> Path:
    compressed_path = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    path.unlink()
    return compressed_path


def load_dataset_samples(args: argparse.Namespace) -> tuple[list[Image.Image], list[str]]:
    from datasets import load_dataset

    split = args.dataset_split
    if args.dataset_sample_limit:
        split = f"{split}[:{args.dataset_sample_limit}]"

    dataset = load_dataset(args.dataset_id, split=split)
    missing_columns = [
        column
        for column in [args.image_column, args.text_column]
        if column and column not in dataset.column_names
    ]
    if missing_columns:
        raise SystemExit(
            f"Missing dataset column(s): {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )

    images: list[Image.Image] = []
    texts: list[str] = []
    for row in dataset:
        image = row[args.image_column]
        if not isinstance(image, Image.Image):
            raise SystemExit(
                f"Column {args.image_column!r} did not yield a PIL image. "
                f"Got {type(image).__name__}."
            )
        images.append(image.convert("RGB"))
        texts.append(str(row.get(args.text_column, "")) if args.text_column else "")
    return images, texts


def batch_indices(
    *,
    population_size: int,
    batch_size: int,
    iterations: int,
    rng: random.Random,
    repeat_image: bool,
    strategy: str,
    costs: Sequence[int],
    bucket_count: int,
) -> list[list[int]]:
    if repeat_image:
        return [[0] * batch_size for _ in range(iterations)]
    if batch_size > population_size:
        raise SystemExit(
            f"Batch size {batch_size} is larger than the available sample pool "
            f"({population_size}). Increase --dataset-sample-limit or reduce batch size."
        )
    if strategy == "random":
        return [rng.sample(range(population_size), batch_size) for _ in range(iterations)]

    sorted_indices = sorted(range(population_size), key=lambda index: costs[index])
    if strategy == "sorted-text-length":
        windows = [
            sorted_indices[start : start + batch_size]
            for start in range(0, population_size - batch_size + 1, batch_size)
        ]
        rng.shuffle(windows)
        return [windows[index % len(windows)] for index in range(iterations)]

    buckets: list[list[int]] = []
    bucket_count = max(1, min(bucket_count, population_size))
    for bucket_index in range(bucket_count):
        start = round(bucket_index * population_size / bucket_count)
        end = round((bucket_index + 1) * population_size / bucket_count)
        bucket = sorted_indices[start:end]
        if len(bucket) >= batch_size:
            buckets.append(bucket)
    if not buckets:
        raise SystemExit(
            f"No text-length bucket has at least {batch_size} samples. "
            "Reduce --bucket-count or batch size."
        )
    weights = [len(bucket) for bucket in buckets]
    return [
        rng.sample(rng.choices(buckets, weights=weights, k=1)[0], batch_size)
        for _ in range(iterations)
    ]


def generated_token_lengths(
    outputs,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    lengths: list[int] = []
    for output in outputs:
        tokens = output.tolist()
        generated = tokens[1:] if tokens else []
        if eos_token_id is not None and eos_token_id in generated:
            lengths.append(generated.index(eos_token_id))
        else:
            lengths.append(
                sum(1 for token in generated if pad_token_id is None or token != pad_token_id)
            )
    return lengths


def main() -> None:
    args = parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit(
            "CUDA is unavailable. Fix the NVIDIA/CUDA environment or pass --allow-cpu "
            "to validate the benchmark pipeline on CPU."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output_dir / "traces"
    table_dir = args.output_dir / "tables"
    trace_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id)
    model.eval().to(device)

    metadata = {
        "model_id": args.model_id,
        "device": device,
        "dtype": args.dtype if device == "cuda" else "fp32",
        "batch_sizes": args.batch_sizes,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "save_sample_images": args.save_sample_images,
        "repeat_image": args.repeat_image,
        "dataset_id": args.dataset_id,
        "dataset_split": args.dataset_split if args.dataset_id else None,
        "dataset_sample_limit": args.dataset_sample_limit if args.dataset_id else None,
        "image_column": args.image_column if args.dataset_id else None,
        "text_column": args.text_column if args.dataset_id else None,
        "seed": args.seed,
        "batching": args.batching,
        "bucket_count": args.bucket_count,
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
        "compress_traces": args.compress_traces,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    results_path = args.output_dir / "results.csv"
    fields = [
        "model_id",
        "device",
        "dtype",
        "batch_size",
        "batching",
        "iterations",
        "profile_iterations",
        "warmup",
        "max_new_tokens",
        "latency_ms_mean",
        "latency_ms_variance",
        "latency_ms_stddev",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_p99",
        "images_per_second",
        "images_per_second_variance",
        "images_per_second_stddev",
        "text_length_mean",
        "text_length_variance",
        "text_length_p95",
        "generated_tokens_mean",
        "generated_tokens_variance",
        "generated_tokens_p95",
        "generated_tokens_max",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "trace_path",
        "table_path",
    ]
    batch_metrics_path = args.output_dir / "batch_metrics.csv"
    batch_metric_fields = [
        "model_id",
        "dataset_id",
        "batching",
        "batch_size",
        "iteration",
        "latency_ms",
        "images_per_second",
        "text_length_mean",
        "text_length_max",
        "generated_tokens_mean",
        "generated_tokens_max",
    ]

    max_batch = max(args.batch_sizes)
    if args.dataset_id:
        images, texts = load_dataset_samples(args)
    else:
        images = make_synthetic_ocr_images(
            max_batch,
            width=args.image_width,
            height=args.image_height,
        )
        texts = [""] * len(images)

    if args.save_sample_images:
        image_prefix = "dataset_ocr" if args.dataset_id else "synthetic_ocr"
        save_sample_images(
            images[: min(len(images), 24)],
            args.output_dir,
            prefix=image_prefix,
        )

    rng = random.Random(args.seed)
    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    pad_token_id = model.generation_config.pad_token_id
    text_costs = [len(text) for text in texts]

    for batch_size in args.batch_sizes:
        index_batches = batch_indices(
            population_size=len(images),
            batch_size=batch_size,
            iterations=args.iterations,
            rng=rng,
            repeat_image=args.repeat_image,
            strategy=args.batching,
            costs=text_costs,
            bucket_count=args.bucket_count,
        )
        warmup_indices = index_batches[0]
        warmup_images = [images[index] for index in warmup_indices]
        encoded = processor(images=warmup_images, return_tensors="pt")
        warmup_pixel_values = encoded.pixel_values.to(device)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            for _ in range(args.warmup):
                with autocast_context(torch, device, args.dtype):
                    model.generate(
                        warmup_pixel_values,
                        max_new_tokens=args.max_new_tokens,
                    )
            synchronize_if_cuda(torch, device)

            activities = [ProfilerActivity.CPU]
            if device == "cuda":
                activities.append(ProfilerActivity.CUDA)

            latencies_ms: list[float] = []
            text_lengths: list[int] = []
            generated_lengths: list[int] = []
            trace_path = trace_dir / f"trocr_bs{batch_size}.json"
            table_path = table_dir / f"trocr_bs{batch_size}.txt"

            for iteration, indices in enumerate(index_batches):
                batch_images = [images[index] for index in indices]
                batch_texts = [texts[index] for index in indices]
                text_lengths.extend(len(text) for text in batch_texts)
                encoded = processor(images=batch_images, return_tensors="pt")
                pixel_values = encoded.pixel_values.to(device)
                synchronize_if_cuda(torch, device)
                started = time.perf_counter()
                with autocast_context(torch, device, args.dtype):
                    outputs = model.generate(
                        pixel_values,
                        max_new_tokens=args.max_new_tokens,
                    )
                synchronize_if_cuda(torch, device)
                latency_ms = (time.perf_counter() - started) * 1000
                latencies_ms.append(latency_ms)
                iteration_generated_lengths = generated_token_lengths(
                    outputs,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                generated_lengths.extend(iteration_generated_lengths)
                write_csv_row(
                    batch_metrics_path,
                    batch_metric_fields,
                    {
                        "model_id": args.model_id,
                        "dataset_id": args.dataset_id or "synthetic",
                        "batching": args.batching,
                        "batch_size": batch_size,
                        "iteration": iteration,
                        "latency_ms": round(latency_ms, 4),
                        "images_per_second": round((batch_size * 1000) / latency_ms, 4),
                        "text_length_mean": round(
                            sum(len(text) for text in batch_texts) / len(batch_texts), 4
                        ),
                        "text_length_max": max((len(text) for text in batch_texts), default=0),
                        "generated_tokens_mean": round(
                            sum(iteration_generated_lengths)
                            / max(len(iteration_generated_lengths), 1),
                            4,
                        ),
                        "generated_tokens_max": max(iteration_generated_lengths, default=0),
                    },
                )

            with profile(
                activities=activities,
                record_shapes=args.record_shapes,
                profile_memory=args.profile_memory,
                with_stack=False,
                with_modules=True,
            ) as prof:
                for _ in range(args.profile_iterations):
                    profile_images = [images[index] for index in index_batches[0]]
                    encoded = processor(images=profile_images, return_tensors="pt")
                    pixel_values = encoded.pixel_values.to(device)
                    synchronize_if_cuda(torch, device)
                    with record_function(f"trocr_generate_batch_{batch_size}"):
                        with autocast_context(torch, device, args.dtype):
                            model.generate(
                                pixel_values,
                                max_new_tokens=args.max_new_tokens,
                            )
                    synchronize_if_cuda(torch, device)
                    prof.step()

            prof.export_chrome_trace(str(trace_path))
            if args.compress_traces:
                trace_path = compress_file(trace_path)
            table = prof.key_averages().table(
                sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
                row_limit=80,
            )
            table_path.write_text(table)

        mean_ms = sum(latencies_ms) / len(latencies_ms)
        throughput_values = [(batch_size * 1000) / latency for latency in latencies_ms]
        mean_throughput = sum(throughput_values) / len(throughput_values)
        latency_variance = sample_variance(latencies_ms)
        throughput_variance = sample_variance(throughput_values)
        row = {
            "model_id": args.model_id,
            "device": device,
            "dtype": args.dtype if device == "cuda" else "fp32",
            "batch_size": batch_size,
            "batching": args.batching,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "latency_ms_mean": round(mean_ms, 4),
            "latency_ms_variance": round(latency_variance, 4),
            "latency_ms_stddev": round(math.sqrt(latency_variance), 4),
            "latency_ms_p50": round(percentile(latencies_ms, 0.50), 4),
            "latency_ms_p95": round(percentile(latencies_ms, 0.95), 4),
            "latency_ms_p99": round(percentile(latencies_ms, 0.99), 4),
            "images_per_second": round(mean_throughput, 4),
            "images_per_second_variance": round(throughput_variance, 4),
            "images_per_second_stddev": round(math.sqrt(throughput_variance), 4),
            "text_length_mean": round(sum(text_lengths) / max(len(text_lengths), 1), 4),
            "text_length_variance": round(sample_variance(text_lengths), 4)
            if text_lengths
            else 0,
            "text_length_p95": round(percentile(text_lengths, 0.95), 4)
            if text_lengths
            else 0,
            "generated_tokens_mean": round(
                sum(generated_lengths) / max(len(generated_lengths), 1), 4
            ),
            "generated_tokens_variance": round(sample_variance(generated_lengths), 4)
            if generated_lengths
            else 0,
            "generated_tokens_p95": round(percentile(generated_lengths, 0.95), 4)
            if generated_lengths
            else 0,
            "generated_tokens_max": max(generated_lengths, default=0),
            "cuda_peak_allocated_mb": round(
                torch.cuda.max_memory_allocated() / (1024**2), 4
            )
            if device == "cuda"
            else 0,
            "cuda_peak_reserved_mb": round(
                torch.cuda.max_memory_reserved() / (1024**2), 4
            )
            if device == "cuda"
            else 0,
            "trace_path": str(trace_path),
            "table_path": str(table_path),
        }
        write_csv_row(results_path, fields, row)
        print(
            f"batch={batch_size} mean={row['latency_ms_mean']}ms "
            f"throughput={row['images_per_second']} img/s trace={trace_path}"
        )


if __name__ == "__main__":
    main()

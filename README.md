# Model Inference Profiling

Profile inference bottlenecks for a Hugging Face OCR model, starting with TrOCR.

The project runs `microsoft/trocr-base-printed` across configurable batch sizes,
exports PyTorch profiler Chrome traces, summarizes CPU/CUDA operator time, and
creates portfolio-friendly visualizations of latency, throughput, memory, and
kernel activity.

## What This Captures

- End-to-end TrOCR generation latency across batch sizes.
- Throughput in images/second.
- Peak CUDA memory allocated and reserved.
- PyTorch profiler traces viewable in Chrome, Perfetto, or TensorBoard.
- Aggregated trace summaries for CPU ops, CUDA runtime calls, and GPU kernels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will need a working NVIDIA driver and CUDA-compatible PyTorch build for GPU
profiling. The scripts can run on CPU with `--allow-cpu`, but CPU runs are mostly
useful for validating the pipeline.

## Quick Start

```bash
python -m mip.profile_trocr \
  --batch-sizes 1 2 4 8 \
  --iterations 10 \
  --profile-iterations 1 \
  --warmup 3 \
  --dtype fp16 \
  --output-dir runs/trocr-base-printed
```

`--iterations` controls plain timing runs. `--profile-iterations` controls how
many runs are recorded in the PyTorch profiler trace. Keep profiled iterations
small for autoregressive generation models because Chrome traces can become very
large. By default, traces are gzip-compressed and tensor shape/memory event
recording is disabled to keep output manageable.

Use `--repeat-image` to build every batch from repeated copies of the first
synthetic image. This is useful for separating batch-size effects from
input-content effects in autoregressive generation.

For real OCR variation, profile random batches from a Hugging Face dataset:

```bash
python -m mip.profile_trocr \
  --dataset-id priyank-m/SROIE_2019_text_recognition \
  --dataset-split train \
  --dataset-sample-limit 512 \
  --batch-sizes 4 8 16 32 64 \
  --iterations 50 \
  --profile-iterations 1 \
  --warmup 3 \
  --dtype fp16 \
  --seed 42 \
  --output-dir runs/trocr-sroie-random-batches
```

This writes `results.csv` with aggregate batch-size statistics, including mean,
sample variance, standard deviation, and tail latencies. It also writes
`batch_metrics.csv` with one row per sampled batch.

Compare random batching against cost-aware text-length bucketing:

```bash
python -m mip.profile_trocr \
  --dataset-id priyank-m/SROIE_2019_text_recognition \
  --dataset-split train \
  --dataset-sample-limit 512 \
  --batch-sizes 4 8 16 32 64 \
  --iterations 50 \
  --profile-iterations 1 \
  --warmup 3 \
  --dtype fp16 \
  --seed 42 \
  --batching bucketed-text-length \
  --bucket-count 8 \
  --output-dir runs/trocr-sroie-bucketed-text

python -m mip.compare_runs \
  --run random runs/trocr-sroie-random-batches/results.csv \
  --run bucketed_text_length runs/trocr-sroie-bucketed-text/results.csv \
  --batch-metrics random runs/trocr-sroie-random-batches/batch_metrics.csv \
  --batch-metrics bucketed_text_length runs/trocr-sroie-bucketed-text/batch_metrics.csv \
  --gpu-hour-cost 0.50 \
  --output-dir runs/trocr-sroie-batching-comparison
```

`compare_runs` estimates cost per 1k images from the provided GPU-hour price and
generates latency, throughput, cost, and batch-difficulty correlation plots.

Then summarize traces and build plots:

```bash
python -m mip.analyze_trace runs/trocr-base-printed/traces \
  --output runs/trocr-base-printed/trace_summary.csv

python -m mip.plot_results runs/trocr-base-printed/results.csv \
  --trace-summary runs/trocr-base-printed/trace_summary.csv \
  --output-dir runs/trocr-base-printed/figures

python -m mip.plot_call_tree runs/trocr-base-printed/traces \
  --output-dir runs/trocr-base-printed/call_trees
```

Open the generated HTML files in `runs/trocr-base-printed/figures` and
`runs/trocr-base-printed/call_trees`.

## Trace Viewing

Each batch size writes a Chrome trace:

```text
runs/trocr-base-printed/traces/trocr_bs1.json
runs/trocr-base-printed/traces/trocr_bs2.json.gz
...
```

Open these with:

- `chrome://tracing`
- <https://ui.perfetto.dev>
- TensorBoard profiler tooling

## Repository Layout

```text
src/mip/profile_trocr.py   Run model inference and export profiler traces.
src/mip/analyze_trace.py   Aggregate Chrome trace events into CSV summaries.
src/mip/plot_results.py    Generate interactive HTML charts.
src/mip/plot_call_tree.py  Generate profiler-style call tree views.
src/mip/compare_runs.py    Compare batching strategies and estimated costs.
requirements.txt          Runtime dependencies.
```

## Notes

The first benchmark uses synthetic text images so that no dataset download is
required. For a portfolio version, a good next step is adding a small curated
image set and a narrative notebook that explains the bottlenecks discovered at
each batch size.

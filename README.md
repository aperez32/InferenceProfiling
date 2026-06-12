# Model Inference Profiling

Profile, explain, and visualize inference bottlenecks for a Hugging Face OCR model.
The case study in this repo uses `microsoft/trocr-base-printed` on SROIE receipt
text crops, then compares ordinary random batching against length-aware batching.

The project produces:

- benchmark CSVs for latency, throughput, CUDA memory, generated token counts, and text length
- PyTorch profiler Chrome traces for CPU ops, CUDA runtime calls, and GPU kernels
- interactive Plotly charts for batch-size scaling, batching policy, cost, and call trees
- static frontend assets used by the portfolio case study
- example OCR crops and token-by-token TrOCR decoding data for the architecture animation

## Why This Matters

TrOCR is an encoder-decoder transformer. The image encoder runs once, but the text
decoder generates autoregressively, one token at a time. In a batch, short samples
can wait on the longest generated output. The profiling scripts make that visible
in latency distributions, generated-token statistics, CUDA kernels, and call-tree
views.

The main experiment compares:

- `random`: each batch samples arbitrary SROIE crops
- `bucketed-text-length`: batches group samples with similar ground-truth text length

Ground-truth text length is used here as an offline proxy for expected decode
length. In a production system, the same idea would usually be implemented with a
cheap predictor, metadata, image crop width, historical request statistics, or
queue-time bucketing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

GPU profiling requires an NVIDIA driver and a CUDA-compatible PyTorch build. The
pipeline can be smoke-tested on CPU with `--allow-cpu`, but the profiler/cost
story is built around CUDA execution.

## Core Scripts

```text
src/mip/profile_trocr.py        Run TrOCR inference benchmarks and export traces.
src/mip/analyze_trace.py        Aggregate Chrome trace events into CSV summaries.
src/mip/plot_results.py         Create batch-size scaling charts from results CSVs.
src/mip/compare_runs.py         Compare batching strategies and estimate GPU cost.
src/mip/plot_call_tree.py       Build profiler-style CPU/GPU call-tree HTML views.
src/mip/export_trocr_flow.py    Export token-by-token decoder data for the frontend.
src/mip/export_batch_examples.py Export random/similar-width SROIE image examples.
```

## 1. Controlled Batch-Size Scaling

Use repeated copies of the same synthetic image to isolate batch-size effects from
input-content variation.

```bash
python -m mip.profile_trocr \
  --batch-sizes 4 8 16 32 48 64 80 96 128 160 192 \
  --iterations 10 \
  --profile-iterations 1 \
  --warmup 3 \
  --dtype fp16 \
  --repeat-image \
  --output-dir runs/trocr-repeat-image-scale
```

Key outputs:

```text
runs/trocr-repeat-image-scale/results.csv
runs/trocr-repeat-image-scale/batch_metrics.csv
runs/trocr-repeat-image-scale/traces/trocr_bs64.json.gz
runs/trocr-repeat-image-scale/tables/trocr_bs64.txt
runs/trocr-repeat-image-scale/sample_images/
```

`results.csv` includes mean latency, variance, standard deviation, p50/p95/p99
latency, throughput, text-length stats, generated-token stats, CUDA peak
allocated memory, and trace paths.

## 2. Random SROIE Batches

Run the same model on real OCR crops sampled from SROIE.

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
  --batching random \
  --output-dir runs/trocr-sroie-random-batches
```

This records one row per batch in `batch_metrics.csv`, which is useful for
correlating latency with the longest generated output in the batch.

## 3. Length-Aware Batching

Compare random batches against groups of similar expected decode length.

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
```

Other supported batching modes:

- `random`: arbitrary samples per batch
- `sorted-text-length`: contiguous windows after sorting by text length
- `bucketed-text-length`: sample from text-length buckets

## 4. Compare Runs and Estimate Cost

`compare_runs.py` merges benchmark CSVs, estimates cost per 1k images from a
GPU-hour price, and creates strategy-comparison plots.

```bash
python -m mip.compare_runs \
  --run random runs/trocr-sroie-random-batches/results.csv \
  --run bucketed_text_length runs/trocr-sroie-bucketed-text/results.csv \
  --batch-metrics random runs/trocr-sroie-random-batches/batch_metrics.csv \
  --batch-metrics bucketed_text_length runs/trocr-sroie-bucketed-text/batch_metrics.csv \
  --gpu-hour-cost 0.50 \
  --output-dir runs/trocr-sroie-batching-comparison
```

Outputs include:

```text
comparison_results.csv
latency_by_strategy.html
throughput_by_strategy.html
cost_per_1k_images.html
tail_latency_by_strategy.html
latency_vs_generated_tokens.html
latency_vs_text_length.html
```

## 5. Trace Summaries and Scaling Charts

The PyTorch profiler traces are Chrome trace JSON files. They can be opened in
`chrome://tracing`, Perfetto, or TensorBoard profiler tooling.

Summarize events into a compact CSV:

```bash
python -m mip.analyze_trace runs/trocr-sroie-random-batches/traces \
  --output runs/trocr-sroie-random-batches/trace_summary.csv \
  --top-n 80
```

Build interactive charts:

```bash
python -m mip.plot_results runs/trocr-sroie-random-batches/results.csv \
  --trace-summary runs/trocr-sroie-random-batches/trace_summary.csv \
  --output-dir runs/trocr-sroie-random-batches/figures
```

This writes Plotly HTML files for latency, throughput, CUDA memory, and top
CUDA/kernel events.

## 6. Profiler Call Trees

`plot_call_tree.py` converts Chrome trace events into interactive icicle/call-tree
views. It can split CPU and GPU work, rotate the chart, hide redundant thread
bars, and force a shared root duration so multiple charts use the same visual
time scale.

Random batch size 64 CPU/GPU views:

```bash
python -m mip.plot_call_tree \
  runs/trocr-sroie-random-batches/traces/trocr_bs64.json.gz \
  --output-dir portfolio_data/call_trees \
  --categories user_annotation cpu_op cuda_runtime cuda_driver \
  --orientation v \
  --title "Random bs64 CPU call tree" \
  --root-label "Random bs64 CPU" \
  --root-duration-ms 1550 \
  --hide-thread-nodes \
  --output-stem random_bs64_cpu

python -m mip.plot_call_tree \
  runs/trocr-sroie-random-batches/traces/trocr_bs64.json.gz \
  --output-dir portfolio_data/call_trees \
  --categories gpu_user_annotation kernel gpu_memcpy gpu_memset \
  --orientation v \
  --title "Random bs64 GPU call tree" \
  --root-label "Random bs64 GPU" \
  --root-duration-ms 1550 \
  --hide-thread-nodes \
  --output-stem random_bs64_gpu
```

Repeat those commands with the bucketed trace and `bucketed_bs64_cpu` /
`bucketed_bs64_gpu` stems. The portfolio uses these four views to compare random
vs bucketed execution with a common 1550 ms scale.

## 7. Frontend/Portfolio Data Exports

The portfolio page is static: all CSVs, JSON files, HTML plots, and images live
under `portfolio_data/` first, then are mirrored into
`portfolio_git/frontend/public/inference-profiler/`.

Export a TrOCR token-by-token decoding example for the architecture animation:

```bash
python -m mip.export_trocr_flow \
  --dataset-index 114 \
  --max-new-tokens 18 \
  --output-dir portfolio_data \
  --public-output-dir portfolio_git/frontend/public/inference-profiler
```

This writes:

```text
portfolio_data/data/trocr_flow_steps.json
portfolio_data/images/architecture/trocr_flow_input.png
```

Export image examples for random vs visually similar-width batches:

```bash
python -m mip.export_batch_examples \
  --sample-limit 512 \
  --batch-size 8 \
  --seed 42 \
  --min-width 100 \
  --output-dir portfolio_data \
  --public-output-dir portfolio_git/frontend/public/inference-profiler
```

This writes:

```text
portfolio_data/data/batch_examples.json
portfolio_data/images/batch_examples/*.png
```

## Portfolio Bundle

The current portfolio-ready bundle contains:

```text
portfolio_data/
  call_trees/
  content/
  data/
  images/
  plots/
  manifest.json
```

The deployed portfolio copy lives at:

```text
portfolio_git/frontend/public/inference-profiler/
```

Vite serves everything in that directory at `/inference-profiler/...`, which is
what the React case-study page reads.

## Interpreting Common Trace Events

- `aten::linear`, `aten::matmul`, `aten::addmm`: PyTorch operator dispatches for
  transformer linear layers. In a CUDA run, the CPU-side event often represents
  scheduling/bookkeeping; the heavy math usually appears in the GPU tree as
  GEMM/CUTLASS/cuBLAS kernels.
- `cudaLaunchKernel`: CPU runtime call that launches GPU work.
- `cudaStreamSynchronize`: CPU waiting for GPU work to complete. Large values
  often indicate synchronization boundaries, not CPU math.
- `gpu_user_annotation`: the GPU-side range corresponding to the profiled
  `record_function`, useful for seeing how much wall time the device spent under
  the model-generation region.
- `kernel`: individual GPU kernels, including elementwise kernels, layer norm,
  flash attention, and GEMM kernels.

## Notes

- Keep `--profile-iterations` small. Autoregressive generation traces grow very
  quickly.
- `--record-shapes` and `--profile-memory` are off by default because they make
  traces much larger.
- `--compress-traces` defaults to true and writes `.json.gz` traces.
- For statistically useful measurements on varied OCR data, prefer many batches
  (`--iterations 50` or more) and compare mean, variance, and tail latency rather
  than a single timing run.

# Portfolio Story Outline

## 1. OCR Examples

Start with receipt text crops from SROIE. Show the image crop next to the
ground-truth text. This grounds the profiling work in realistic OCR inputs
rather than synthetic benchmark data.

Data:

- `portfolio_data/data/sample_images.json`
- `portfolio_data/images/sroie_sample_*.png`

## 2. TrOCR Architecture

TrOCR is a vision-encoder-decoder OCR model. The image is converted into visual
patch features by a Vision Transformer encoder, then a Transformer decoder
generates text tokens autoregressively.

Important profiling implication: the encoder can process a batch in parallel,
but the decoder advances token by token. A single long or difficult sample can
extend the decode loop for the whole batch.

Data:

- `portfolio_data/content/trocr_architecture.md`

## 3. Batch Size Scaling

Show the controlled repeat-image scaling result first. With identical inputs,
throughput continues improving through larger batches and eventually reaches
diminishing returns. This isolates the GPU/batch-size effect from input
variation.

Plots:

- `portfolio_data/plots/repeat_image_throughput.html`
- `portfolio_data/plots/repeat_image_latency.html`

## 4. Real Data Random Batching

Switch to SROIE random batches. This introduces natural variation in text
length and generated token count. Throughput no longer scales cleanly, and tail
latency grows quickly for larger batches.

Plots:

- `portfolio_data/plots/random_batch_throughput.html`
- `portfolio_data/plots/random_batch_latency.html`

## 5. Cost-Aware Batching

Compare random batching against text-length bucketed batching. Bucketed batches
group similarly sized OCR samples together, reducing straggler effects in the
autoregressive decoder.

Main result: bucketed batching roughly doubles throughput and reduces estimated
GPU cost per 1k images by about 50-60%, using an assumed $0.50/GPU-hour.

Plots:

- `portfolio_data/plots/batching_strategy_throughput.html`
- `portfolio_data/plots/batching_strategy_cost.html`
- `portfolio_data/plots/batching_strategy_latency.html`
- `portfolio_data/plots/batching_strategy_tail_latency.html`

## 6. Difficulty Correlations

Use per-batch metrics to show why the batching strategy matters. The relevant
signals are max generated token count and max ground-truth text length inside a
batch.

Plots:

- `portfolio_data/plots/latency_vs_generated_tokens.html`
- `portfolio_data/plots/latency_vs_text_length.html`

## 7. Profiler Call Trees

Show representative profiler call trees. Use them as evidence that TrOCR
inference spends substantial time in repeated decoder operations such as linear
layers, attention, copies, CUDA launches, and GPU kernels.

Profiler views:

- `portfolio_data/call_trees/random_bs16_call_tree.html`
- `portfolio_data/call_trees/random_bs64_call_tree.html`
- `portfolio_data/call_trees/bucketed_bs32_call_tree.html`
- `portfolio_data/call_trees/bucketed_bs64_call_tree.html`

## 8. Conclusion

The core finding is not simply "use a bigger batch." For autoregressive OCR,
batch composition matters. Randomly mixed batches create stragglers, while
cost-aware batching by expected text length improves throughput and lowers GPU
cost without changing the model.

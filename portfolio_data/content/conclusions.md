# Conclusions

## Main Finding

Batch size alone is not enough to optimize TrOCR inference. On uniform inputs,
larger batches improve throughput until diminishing returns. On realistic OCR
inputs, mixed-cost batches create stragglers that reduce throughput and increase
tail latency.

## Best Observed Result

Text-length bucketed batching improved throughput at every tested batch size.
At larger batch sizes, the improvement was especially large:

- Batch 32: about 150% higher throughput than random batching.
- Batch 64: about 151% higher throughput than random batching.
- Estimated cost per 1k images fell by about 60% at batch sizes 32 and 64.

The cost estimate assumes $0.50 per GPU-hour.

## Interpretation

The model architecture explains the result. TrOCR uses an autoregressive text
decoder, so varied generated lengths can make one difficult sample hold back an
entire batch. Grouping similarly sized text samples reduces that straggler
effect.

## Portfolio Takeaway

The optimization does not require changing the model. It changes request
scheduling. That makes it a realistic inference-systems improvement: better
throughput and lower estimated GPU cost from smarter batching.

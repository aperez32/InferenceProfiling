# TrOCR Architecture Notes

TrOCR is a Transformer-based OCR model implemented as a
`VisionEncoderDecoderModel`.

## Data Flow

1. Input image crop
2. Image preprocessing through `TrOCRProcessor`
3. Vision Transformer encoder
4. Autoregressive Transformer decoder
5. Token ids
6. Decoded text string

## Encoder

The encoder converts the input image into visual patch representations. This
stage is comparatively batch-friendly because the model can process image
patches for many samples in parallel.

## Decoder

The decoder generates text one token at a time. Each step depends on the
previously generated tokens, so decoding has a sequential component. The batch
continues until generation finishes for all samples or reaches the configured
token limit.

## Profiling Implication

For fixed, repeated inputs, increasing batch size mostly improves GPU
utilization until throughput saturates. For varied OCR inputs, a single long or
difficult sample can become a straggler and keep the whole batch in the decode
loop.

## Practical Optimization

Batch examples with similar expected decode cost. In this project, ground-truth
text length is used as a simple proxy for expected cost. In production, a
similar signal could come from historical decode length, predicted text length,
image width, document type, or a lightweight routing model.

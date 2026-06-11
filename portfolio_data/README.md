# Portfolio Data Bundle

This directory contains the curated data and assets for the portfolio frontend.

## Structure

- `manifest.json`: high-level index for the frontend.
- `images/`: selected SROIE OCR image crops.
- `plots/`: interactive Plotly HTML charts.
- `call_trees/`: profiler-style interactive call tree views.
- `data/`: CSV and JSON metric files.
- `content/`: markdown narrative sections.

## Recommended Page Flow

1. OCR examples and labels
2. TrOCR architecture
3. Batch-size scaling on controlled repeated inputs
4. Random real-data batching
5. Random vs text-length bucketed batching
6. Cost per 1k images
7. Call-tree profiler views
8. Conclusions

## Key Files

- `data/sample_images.json`
- `data/portfolio_metrics.json`
- `data/batching_improvements.json`
- `data/batching_comparison_results.csv`
- `plots/batching_strategy_cost.html`
- `plots/batching_strategy_throughput.html`
- `content/story_outline.md`

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a frontend-friendly TrOCR token-by-token data-flow trace."
    )
    parser.add_argument("--model-id", default="microsoft/trocr-base-printed")
    parser.add_argument("--dataset-id", default="priyank-m/SROIE_2019_text_recognition")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=114,
        help="SROIE row to visualize. Defaults to an English receipt item.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=18)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("portfolio_data"),
        help="Portfolio data bundle to update.",
    )
    parser.add_argument(
        "--public-output-dir",
        type=Path,
        default=Path("portfolio_git/frontend/public/inference-profiler"),
        help="Optional Vite public asset bundle to mirror.",
    )
    return parser.parse_args()


def token_label(processor, token_id: int) -> str:
    tokenizer = processor.tokenizer
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        return str(token_id)
    return token.replace("Ġ", " ").replace("▁", " ")


def decoded_text(processor, token_ids: list[int]) -> str:
    return processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()


def save_crop(image, base_dir: Path) -> str:
    image_dir = base_dir / "images" / "architecture"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "trocr_flow_input.png"
    image.save(image_path)
    return image_path.relative_to(base_dir).as_posix()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_id).eval().to(device)

    dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    row = dataset[args.dataset_index]
    image = row["image"].convert("RGB")
    ground_truth = str(row.get("text", ""))

    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded.pixel_values.to(device)

    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(
            model.generation_config,
            "decoder_start_token_id",
            None,
        )
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(model.config.decoder, "bos_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = processor.tokenizer.bos_token_id
    if decoder_start_token_id is None:
        raise SystemExit("Could not determine decoder start token id.")

    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else None

    generated_ids = [int(decoder_start_token_id)]
    steps: list[dict[str, object]] = []

    with torch.inference_mode():
        encoder_outputs = model.get_encoder()(pixel_values)
        for step_index in range(args.max_new_tokens):
            decoder_input_ids = torch.tensor([generated_ids], device=device)
            outputs = model(
                pixel_values=None,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
            )
            next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            generated_ids.append(next_token_id)

            visible_ids = generated_ids[1:]
            steps.append(
                {
                    "step": step_index + 1,
                    "decoder_input_tokens": [
                        token_label(processor, token_id) for token_id in generated_ids[:-1]
                    ],
                    "next_token": token_label(processor, next_token_id),
                    "next_token_id": next_token_id,
                    "decoded_text": decoded_text(processor, visible_ids),
                    "is_eos": eos_token_id is not None and next_token_id == eos_token_id,
                }
            )
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    payload = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_split": args.dataset_split,
        "dataset_index": args.dataset_index,
        "ground_truth": ground_truth,
        "input_image": save_crop(image, args.output_dir),
        "preprocessing": [
            "crop",
            "resize",
            "normalize",
            "patchify",
            "flatten",
        ],
        "patch_preview": {
            "count": 12,
            "label": "image patches",
        },
        "encoder": {
            "repeat": "xN",
            "layers": ["multi-head self-attention", "feed-forward network"],
        },
        "decoder": {
            "repeat": "token loop",
            "layers": [
                "masked multi-head self-attention",
                "cross-attention over image features",
                "feed-forward network",
            ],
        },
        "generation": {
            "start_token": token_label(processor, int(decoder_start_token_id)),
            "steps": steps,
            "final_text": decoded_text(processor, generated_ids[1:]),
        },
    }

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "trocr_flow_steps.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.public_output_dir:
        public_data_dir = args.public_output_dir / "data"
        public_image_dir = args.public_output_dir / "images" / "architecture"
        public_data_dir.mkdir(parents=True, exist_ok=True)
        public_image_dir.mkdir(parents=True, exist_ok=True)
        (public_data_dir / "trocr_flow_steps.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        image.save(public_image_dir / "trocr_flow_input.png")

    print(output_path)
    print(payload["generation"]["final_text"])


if __name__ == "__main__":
    main()

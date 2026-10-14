"""Run an OpenCLIP model against the available benchmark dataset."""

import argparse
from typing import Any

import torch

from models.data import BenchmarkDataset, load_retrieval_set
from models.metrics import classification_topk, retrieval_metrics
from models.registry import MODEL_REGISTRY, EmbeddingModel


def _encode_batches(
    model: EmbeddingModel,
    items: list[Any],
    batch_size: int,
    encode: str,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    outputs = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        outputs.append(
            model.encode_images(batch)
            if encode == "images"
            else model.encode_texts(batch)
        )
    if not outputs:
        raise ValueError("Cannot encode an empty benchmark dataset")
    return torch.cat(outputs)


def evaluate_model(
    model: EmbeddingModel,
    dataset: BenchmarkDataset,
    batch_size: int = 32,
) -> dict[str, Any]:
    print(f"Benchmark dataset: {dataset.name} (task: {dataset.task})")
    image_embeddings = _encode_batches(
        model, dataset.images, batch_size=batch_size, encode="images"
    )

    if dataset.task == "retrieval":
        flat_captions = [
            caption for image_captions in dataset.captions
            for caption in image_captions
        ]
        caption_image_ids = [
            image_index
            for image_index, image_captions in enumerate(dataset.captions)
            for _ in image_captions
        ]
        text_embeddings = _encode_batches(
            model, flat_captions, batch_size=batch_size, encode="texts"
        )
        return retrieval_metrics(
            image_embeddings,
            text_embeddings,
            caption_image_ids,
        )

    if dataset.task == "classification":
        if dataset.class_prompts is None or dataset.labels is None:
            raise ValueError("Classification dataset lacks prompts or labels")
        class_embeddings = _encode_batches(
            model, dataset.class_prompts, batch_size=batch_size, encode="texts"
        )
        return classification_topk(
            image_embeddings, class_embeddings, dataset.labels, ks=(1, 5)
        )

    raise ValueError(f"Unsupported benchmark task: {dataset.task}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="vit_b_32_openai")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset = load_retrieval_set(limit=args.limit)
    model = EmbeddingModel.load(MODEL_REGISTRY[args.model], device=args.device)
    results = evaluate_model(model, dataset, batch_size=args.batch_size)
    print(results)


if __name__ == "__main__":
    main()

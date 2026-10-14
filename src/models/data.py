"""Benchmark datasets for image/text retrieval and classification."""

from dataclasses import dataclass
from pathlib import Path
import warnings

from PIL import Image


DATASET_ID = "nlphuji/flickr_1k_test_image_text_retrieval"
DEFAULT_CIFAR_LIMIT = 2000


@dataclass
class BenchmarkDataset:
    name: str
    task: str
    images: list[Image.Image]
    captions: list[list[str]]
    labels: list[int] | None = None
    class_prompts: list[str] | None = None


def _cache_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "data" / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_retrieval_set(limit: int | None = None) -> BenchmarkDataset:
    """Load the ordered Flickr retrieval split, falling back to CIFAR-100.

    The Hugging Face dataset card's loader defines the real fields as ``image``
    and ``caption``; each row contains one image and its caption list. Both
    datasets are cached under ``data/benchmark``.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")

    cache_root = _cache_root()
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            DATASET_ID,
            split="test",
            cache_dir=str(cache_root / "huggingface"),
            trust_remote_code=True,
        )
        required = {"image", "caption"}
        if not required.issubset(dataset.column_names):
            raise ValueError(
                "Flickr dataset fields differ from the card schema: "
                f"expected {sorted(required)}, found {dataset.column_names}"
            )
        count = len(dataset) if limit is None else min(limit, len(dataset))
        selected = dataset.select(range(count))
        images = [item.convert("RGB") for item in selected["image"]]
        captions = [
            [str(caption) for caption in row]
            for row in selected["caption"]
        ]
        if any(not row for row in captions):
            raise ValueError("Flickr dataset contains an image without captions.")
        return BenchmarkDataset(
            name=DATASET_ID,
            task="retrieval",
            images=images,
            captions=captions,
        )
    except Exception as error:
        warnings.warn(
            f"Could not load {DATASET_ID} ({error!r}); using CIFAR-100 "
            "classification fallback instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _load_cifar100(limit, cache_root)


def _load_cifar100(limit: int | None, cache_root: Path) -> BenchmarkDataset:
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(
        root=str(cache_root / "torchvision"),
        train=False,
        download=True,
    )
    default_count = min(DEFAULT_CIFAR_LIMIT, len(dataset))
    count = default_count if limit is None else min(limit, default_count, len(dataset))
    images: list[Image.Image] = []
    labels: list[int] = []
    captions: list[list[str]] = []
    for index in range(count):
        image, label = dataset[index]
        images.append(image.convert("RGB"))
        labels.append(int(label))
        captions.append([f"a photo of a {dataset.classes[label]}"])

    return BenchmarkDataset(
        name="torchvision/CIFAR-100 test (classification fallback)",
        task="classification",
        images=images,
        captions=captions,
        labels=labels,
        class_prompts=[
            f"a photo of a {class_name}" for class_name in dataset.classes
        ],
    )


__all__ = ["BenchmarkDataset", "DATASET_ID", "load_retrieval_set"]

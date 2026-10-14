"""Benchmark datasets for image/text retrieval and classification."""

import csv
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import warnings
from zipfile import ZipFile

from PIL import Image


DATASET_ID = "nlphuji/flickr_1k_test_image_text_retrieval"
FLICKR_ASSET_REPO_ID = "yonatanbitton/flickr_1k_test_image_text_retrieval"
FLICKR_IMAGE_DIRECTORY = "images_flickr_1k_test"
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
        try:
            warnings.warn(
                f"Could not load {DATASET_ID} with datasets ({error!r}); "
                "loading its published CSV and image archive directly.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _load_flickr_artifacts(limit, cache_root)
        except Exception as artifact_error:
            warnings.warn(
                f"Could not load {DATASET_ID} through datasets ({error!r}) "
                f"or its published artifacts ({artifact_error!r}); using "
                "CIFAR-100 classification fallback instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _load_cifar100(limit, cache_root)


def _load_flickr_artifacts(
    limit: int | None,
    cache_root: Path,
) -> BenchmarkDataset:
    from huggingface_hub import hf_hub_download

    cache_dir = str(cache_root / "huggingface")
    csv_path = hf_hub_download(
        repo_id=FLICKR_ASSET_REPO_ID,
        filename="test_1k_flickr.csv",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    archive_path = hf_hub_download(
        repo_id=FLICKR_ASSET_REPO_ID,
        filename=f"{FLICKR_IMAGE_DIRECTORY}.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    images: list[Image.Image] = []
    captions: list[list[str]] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_stream:
        reader = csv.DictReader(csv_stream)
        if reader.fieldnames is None or not {"filename", "raw"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "Flickr asset CSV must include filename and raw caption fields."
            )
        with ZipFile(archive_path) as archive:
            for row in reader:
                if limit is not None and len(images) >= limit:
                    break
                raw_captions = json.loads(row["raw"])
                if not isinstance(raw_captions, list) or not raw_captions:
                    raise ValueError(
                        f"Flickr row {len(images)} has no caption list."
                    )
                member_name = (
                    f"{FLICKR_IMAGE_DIRECTORY}/{row['filename']}"
                )
                with archive.open(member_name) as image_stream:
                    image = Image.open(BytesIO(image_stream.read())).convert("RGB")
                images.append(image)
                captions.append([str(caption) for caption in raw_captions])

    if limit != 0 and not images:
        raise ValueError("Flickr asset archive contained no benchmark rows.")
    return BenchmarkDataset(
        name=DATASET_ID,
        task="retrieval",
        images=images,
        captions=captions,
    )


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

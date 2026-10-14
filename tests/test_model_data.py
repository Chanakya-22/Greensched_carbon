import sys
from types import ModuleType
import csv
import json
from io import BytesIO
from zipfile import ZipFile

from PIL import Image
import pytest

from models import data as model_data
from models.data import DATASET_ID, BenchmarkDataset, load_retrieval_set


class FakeHuggingFaceDataset:
    column_names = ["image", "caption"]

    def __init__(self):
        self.rows = [
            {
                "image": Image.new("RGB", (1, 1), color=(index, 0, 0)),
                "caption": [f"caption {index}"],
            }
            for index in range(4)
        ]

    def __len__(self):
        return len(self.rows)

    def select(self, indices):
        selected = FakeHuggingFaceDataset()
        selected.rows = [self.rows[index] for index in indices]
        return selected

    def __getitem__(self, key):
        return [row[key] for row in self.rows]


def test_retrieval_loader_keeps_first_rows_in_order(monkeypatch):
    dataset = FakeHuggingFaceDataset()
    calls = {}

    def fake_load_dataset(dataset_id, **kwargs):
        calls["id"] = dataset_id
        calls.update(kwargs)
        return dataset

    fake_datasets = ModuleType("datasets")
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loaded = load_retrieval_set(limit=2)

    assert calls["id"] == DATASET_ID
    assert calls["split"] == "test"
    assert calls["trust_remote_code"] is True
    assert loaded.task == "retrieval"
    assert loaded.captions == [["caption 0"], ["caption 1"]]
    assert [image.getpixel((0, 0))[0] for image in loaded.images] == [0, 1]


def test_retrieval_loader_reports_fallback(monkeypatch):
    fake_datasets = ModuleType("datasets")

    def fail_to_load(*args, **kwargs):
        raise OSError("dataset unavailable")

    def fail_to_load_artifacts(limit, cache):
        raise OSError("assets unavailable")

    fake_datasets.load_dataset = fail_to_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr(model_data, "_load_flickr_artifacts", fail_to_load_artifacts)
    fallback = BenchmarkDataset(
        name="torchvision/CIFAR-100 test (classification fallback)",
        task="classification",
        images=[],
        captions=[],
        labels=[],
        class_prompts=[],
    )
    monkeypatch.setattr("models.data._load_cifar100", lambda limit, cache: fallback)

    with pytest.warns(RuntimeWarning, match="using CIFAR-100"):
        loaded = load_retrieval_set(limit=1)

    assert loaded is fallback
    assert loaded.task == "classification"


def test_retrieval_loader_reads_published_artifacts_in_fixed_order(
    monkeypatch, tmp_path
):
    fake_datasets = ModuleType("datasets")

    def fail_to_load(*args, **kwargs):
        raise RuntimeError("dataset scripts are unsupported")

    fake_datasets.load_dataset = fail_to_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    csv_path = tmp_path / "test_1k_flickr.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "raw"])
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "filename": f"image{index}.png",
                    "raw": json.dumps([f"caption {index}a", f"caption {index}b"]),
                }
            )

    archive_path = tmp_path / "images_flickr_1k_test.zip"
    with ZipFile(archive_path, "w") as archive:
        for index in range(3):
            image_bytes = BytesIO()
            Image.new("RGB", (2, 2), color=(index, 0, 0)).save(
                image_bytes, format="PNG"
            )
            archive.writestr(
                f"images_flickr_1k_test/image{index}.png",
                image_bytes.getvalue(),
            )

    fake_hub = ModuleType("huggingface_hub")

    def fake_hf_hub_download(*, filename, **kwargs):
        return str(csv_path if filename.endswith(".csv") else archive_path)

    fake_hub.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.warns(RuntimeWarning, match="loading its published CSV"):
        loaded = load_retrieval_set(limit=2)

    assert loaded.name == DATASET_ID
    assert loaded.task == "retrieval"
    assert loaded.captions == [
        ["caption 0a", "caption 0b"],
        ["caption 1a", "caption 1b"],
    ]
    assert [image.getpixel((0, 0))[0] for image in loaded.images] == [0, 1]

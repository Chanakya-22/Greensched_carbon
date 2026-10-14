import sys
from types import ModuleType

from PIL import Image
import pytest

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

    fake_datasets.load_dataset = fail_to_load
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
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

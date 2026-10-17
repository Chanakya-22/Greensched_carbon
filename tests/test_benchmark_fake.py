import csv
import json

import torch
from PIL import Image

from models import benchmark
from models.data import BenchmarkDataset


class FakeModel:
    device = torch.device("cpu")
    embedding_dim = 3
    parameter_count = 17
    preprocess = staticmethod(
        lambda image: torch.zeros((3, 4, 4), dtype=torch.float32)
    )

    def encode_images(self, images):
        if isinstance(images, torch.Tensor):
            return torch.zeros((images.shape[0], 3))
        output = []
        for image in images:
            image_id = image.getpixel((0, 0))[0] - 1
            vector = torch.zeros(3)
            vector[image_id] = 1.0
            output.append(vector)
        return torch.stack(output)

    def encode_texts(self, texts):
        output = []
        for text in texts:
            image_id = int(text.split()[0].removeprefix("image"))
            vector = torch.zeros(3)
            vector[image_id] = 1.0
            output.append(vector)
        return torch.stack(output)


def _fake_dataset():
    images = [
        Image.new("RGB", (4, 4), color=(index + 1, 0, 0))
        for index in range(3)
    ]
    captions = [
        [f"image{index} caption one", f"image{index} caption two"]
        for index in range(3)
    ]
    return BenchmarkDataset(
        name="synthetic benchmark fixture",
        task="retrieval",
        images=images,
        captions=captions,
    )


def test_multi_model_runner_writes_outputs_and_continues_after_model_error(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark,
        "MODEL_REGISTRY",
        {"fake-good": object(), "fake-good-2": object(), "fake-broken": object()},
    )
    monkeypatch.setattr(
        benchmark,
        "hardware_info",
        lambda: {
            "gpu_name": float("nan"),
            "driver_version": float("nan"),
            "cuda_version": float("nan"),
            "torch_version": "fake",
            "open_clip_version": "fake",
            "cpu": "fake cpu",
            "os": "fake os",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        benchmark,
        "run_with_energy_profile",
        lambda model, operation: (
            operation(),
            {
                "energy_kwh": 0.0003,
                "duration_s": 1.5,
                "avg_power_w": 0.72,
                "pue": 1.0,
                "energy_method": "estimated_from_runtime",
                "components_measured": ["wall-clock duration"],
                "components_estimated": ["CPU energy"],
            },
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "latency_profile",
        lambda model, batch_sizes: {
            size: {
                "median_ms": size / 10,
                "iqr_ms": size / 100,
                "images_per_s": 1000 / size,
            }
            for size in batch_sizes
        },
    )
    monkeypatch.setattr(benchmark, "peak_gpu_memory_mb", lambda model, batch_size: float("nan"))

    def load_model(name, device):
        if name == "fake-broken":
            raise RuntimeError("injected model failure")
        return FakeModel()

    out_dir = tmp_path / "docs"
    profile_path = tmp_path / "data" / "profiles.json"
    frame = benchmark.run_comparison(
        ["fake-good", "fake-broken", "fake-good-2"],
        out_dir=out_dir,
        dataset=_fake_dataset(),
        model_loader=load_model,
        profile_path=profile_path,
        device="cpu",
        allow_cpu=True,
    )

    assert frame["model"].tolist() == ["fake-good", "fake-broken", "fake-good-2"]
    assert frame["device"].tolist() == ["cpu", "cpu", "cpu"]
    assert frame.loc[0, "mean_recall_at_5"] == 1.0
    assert frame.loc[1, "error"] == "RuntimeError: injected model failure"
    assert frame.loc[2, "mean_recall_at_5"] == 1.0
    assert frame.loc[0, "parameter_count"] == 17
    assert frame.loc[0, "embedding_dim"] == 3
    assert frame.loc[0, "images_per_s_b64"] == 1000 / 64
    assert frame.loc[0, "energy_method"] == "estimated_from_runtime"

    csv_path = out_dir / benchmark.CSV_NAME
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 3
    assert records[1]["error"] == "RuntimeError: injected model failure"
    assert records[0]["dataset"] == "synthetic benchmark fixture"
    assert records[0]["device"] == "cpu"
    assert records[0]["energy_method"] == "estimated_from_runtime"
    assert records[0]["n_images"] == "3"
    assert (out_dir / benchmark.COMPARISON_PLOT_NAME).exists()
    assert (out_dir / benchmark.PARETO_PLOT_NAME).exists()
    assert (out_dir / benchmark.SELECTION_NAME).exists()

    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profiles["device"] == "cpu"
    assert profiles["models"]["fake-good"]["avg_power_w"] == 0.72
    assert profiles["models"]["fake-good"]["device"] == "cpu"
    assert (
        profiles["models"]["fake-good"]["energy_method"]
        == "estimated_from_runtime"
    )
    assert profiles["models"]["fake-good"]["img_per_s_at_batch_64"] == 1000 / 64
    assert profiles["hardware"]["gpu_name"] is None
    selection = (out_dir / benchmark.SELECTION_NAME).read_text(encoding="utf-8")
    assert "**Winner:** fake-good" in selection
    assert "**Runner-up:** fake-good-2" in selection
    assert "- Device: cpu" in selection
    assert "- Dataset: synthetic benchmark fixture (images: 3)" in selection
    assert "- Energy method: estimated_from_runtime (PUE: 1.0)" in selection
    assert "runtime proxy" in selection

    console = capsys.readouterr().out
    assert "Failed fake-broken: RuntimeError: injected model failure" in console
    assert "Completed fake-good-2" in console
    assert "GPU fields are empty" in console

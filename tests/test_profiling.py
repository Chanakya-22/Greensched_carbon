import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from models.profiling import (
    energy_profile,
    hardware_info,
    latency_profile,
    peak_gpu_memory_mb,
)


class FakeModel:
    device = torch.device("cpu")
    preprocess = staticmethod(lambda image: torch.zeros((3, 8, 8)))

    def encode_images(self, images):
        count = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        return torch.ones((count, 4))

    def encode_texts(self, texts):
        return torch.ones((len(texts), 4))


def test_latency_profile_uses_median_and_interquartile_range():
    timings = iter([0.0, 0.004, 1.0, 1.006, 2.0, 2.008, 3.0, 3.010])
    results = latency_profile(
        FakeModel(),
        batch_sizes=(2,),
        warmup=1,
        iters=4,
        timer=lambda: next(timings),
    )

    assert set(results) == {2}
    assert set(results[2]) == {"median_ms", "iqr_ms", "images_per_s"}
    assert results[2]["median_ms"] == pytest.approx(7.0)
    assert results[2]["iqr_ms"] == pytest.approx(3.0)
    assert results[2]["images_per_s"] == pytest.approx(2 * 1000 / 7)


def test_peak_memory_and_hardware_info_degrade_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert math.isnan(peak_gpu_memory_mb(FakeModel(), batch_size=1))
    info = hardware_info()

    assert math.isnan(info["gpu_name"])
    assert math.isnan(info["driver_version"])
    assert math.isnan(info["cuda_version"])
    assert info["torch_version"] == torch.__version__
    assert info["cpu"]
    assert info["os"]
    assert info["timestamp"]
    assert isinstance(info["open_clip_version"], str)


def test_energy_profile_output_structure_and_energy_values(monkeypatch):
    instances = []

    class FakeTracker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.final_emissions_data = SimpleNamespace(
                energy_consumed=0.0002,
                pue=1.0,
                energy_method="estimated_from_runtime",
            )
            instances.append(self)

        def start(self):
            return None

        def stop(self):
            return 0.0

    fake_codecarbon = ModuleType("codecarbon")
    fake_codecarbon.OfflineEmissionsTracker = FakeTracker
    monkeypatch.setitem(sys.modules, "codecarbon", fake_codecarbon)
    model = FakeModel()

    results = energy_profile(model, [object(), object()], ["caption"])

    assert set(results) == {
        "energy_kwh",
        "duration_s",
        "avg_power_w",
        "pue",
        "energy_method",
        "components_measured",
        "components_estimated",
    }
    assert results["energy_kwh"] == pytest.approx(0.0002)
    assert results["duration_s"] >= 0
    assert results["avg_power_w"] >= 0
    assert results["pue"] == 1.0
    assert results["energy_method"] == "estimated_from_runtime"
    assert results["components_measured"] == ["wall-clock duration"]
    assert "CPU energy" in results["components_estimated"]
    assert instances[0].kwargs["save_to_file"] is False
    assert instances[0].kwargs["pue"] == 1.0


def test_energy_profile_failure_logs_and_returns_nan(caplog, monkeypatch):
    class FailingTracker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            return None

        def stop(self):
            return None

    fake_codecarbon = ModuleType("codecarbon")
    fake_codecarbon.OfflineEmissionsTracker = FailingTracker
    monkeypatch.setitem(sys.modules, "codecarbon", fake_codecarbon)

    class FailingModel(FakeModel):
        def encode_images(self, images):
            raise RuntimeError("injected encoding failure")

    results = energy_profile(FailingModel(), [object()], ["caption"])

    assert math.isnan(results["energy_kwh"])
    assert math.isnan(results["duration_s"])
    assert math.isnan(results["avg_power_w"])
    assert "Energy profiling failed" in caplog.text

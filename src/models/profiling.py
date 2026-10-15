"""Latency, GPU memory, energy, and hardware profiling for embedding models."""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import platform
import statistics
import tempfile
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


logger = logging.getLogger(__name__)


def _device_for(model: Any) -> torch.device:
    return torch.device(getattr(model, "device", "cpu"))


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _image_batch(model: Any, batch_size: int) -> Any:
    preprocess = getattr(model, "preprocess", None)
    if preprocess is not None:
        image = preprocess(Image.new("RGB", (224, 224), color=(64, 128, 64)))
        batch = torch.stack([image] * batch_size)
        return batch.to(_device_for(model))
    return torch.zeros(
        (batch_size, 3, 224, 224),
        device=_device_for(model),
    )


def latency_profile(
    model: Any,
    batch_sizes: Sequence[int] = (1, 16, 64),
    warmup: int = 5,
    iters: int = 20,
    *,
    timer: Callable[[], float] | None = None,
) -> dict[int, dict[str, float]]:
    """Measure image-encoding latency and throughput for each batch size."""
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    now = time.perf_counter if timer is None else timer
    device = _device_for(model)
    results: dict[int, dict[str, float]] = {}

    for batch_size in batch_sizes:
        images = _image_batch(model, batch_size)
        for _ in range(warmup):
            model.encode_images(images)
        durations_ms: list[float] = []
        for _ in range(iters):
            _sync_if_cuda(device)
            start = now()
            model.encode_images(images)
            _sync_if_cuda(device)
            durations_ms.append((now() - start) * 1000.0)

        median_ms = float(statistics.median(durations_ms))
        lower_quartile, upper_quartile = (
            float(value) for value in np.percentile(durations_ms, [25, 75])
        )
        results[int(batch_size)] = {
            "median_ms": median_ms,
            "iqr_ms": upper_quartile - lower_quartile,
            "images_per_s": (
                batch_size * 1000.0 / median_ms
                if median_ms > 0
                else math.inf
            ),
        }
    return results


def peak_gpu_memory_mb(model: Any, batch_size: int = 64) -> float:
    """Return peak allocated CUDA memory during one image batch, or NaN on CPU."""
    device = _device_for(model)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device.type != "cuda" or not torch.cuda.is_available():
        return math.nan
    try:
        images = _image_batch(model, batch_size)
        torch.cuda.reset_peak_memory_stats(device)
        model.encode_images(images)
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    except Exception:
        logger.warning("Could not measure peak GPU memory.", exc_info=True)
        return math.nan


def _encode_all(model: Any, images: Sequence[Any], texts: Sequence[str]) -> None:
    if len(images) == 0 or len(texts) == 0:
        raise ValueError("Energy profiling requires non-empty images and texts.")
    batch_size = 64
    for start in range(0, len(images), batch_size):
        model.encode_images(images[start : start + batch_size])
    for start in range(0, len(texts), batch_size):
        model.encode_texts(list(texts[start : start + batch_size]))


def _nan_energy_result() -> dict[str, Any]:
    return {
        "energy_kwh": math.nan,
        "duration_s": math.nan,
        "avg_power_w": math.nan,
        "components_measured": [],
        "components_estimated": [],
    }


def run_with_energy_profile(
    model: Any,
    operation: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    """Run an operation once, returning its value and offline energy estimates."""
    try:
        from codecarbon import OfflineEmissionsTracker
    except Exception:
        logger.warning(
            "Could not initialize CodeCarbon; energy values will be NaN.",
            exc_info=True,
        )
        return operation(), _nan_energy_result()

    with tempfile.TemporaryDirectory(prefix="greensched-codecarbon-") as output_dir:
        tracker = None
        started = False
        try:
            tracker = OfflineEmissionsTracker(
                output_dir=output_dir,
                save_to_file=False,
                save_to_api=False,
                save_to_logger=False,
                log_level="error",
            )
            tracker.start()
            started = True
        except Exception:
            logger.warning(
                "Could not start CodeCarbon; energy values will be NaN.",
                exc_info=True,
            )
            return operation(), _nan_energy_result()

        started_at = time.perf_counter()
        try:
            value = operation()
        except Exception:
            try:
                tracker.stop()
            except Exception:
                logger.warning(
                    "Could not stop CodeCarbon after the profiled operation failed.",
                    exc_info=True,
                )
            raise

        duration_s = time.perf_counter() - started_at
        try:
            tracker.stop()
            data = tracker.final_emissions_data
            energy_kwh = float(data.energy_consumed)
            avg_power_w = (
                energy_kwh * 3_600_000.0 / duration_s
                if duration_s > 0
                else math.nan
            )
            estimated = ["CPU energy", "RAM energy", "facility overhead (PUE)"]
            if torch.cuda.is_available():
                estimated.append("GPU energy")
            energy = {
                "energy_kwh": energy_kwh,
                "duration_s": duration_s,
                "avg_power_w": avg_power_w,
                "components_measured": ["wall-clock duration"],
                "components_estimated": estimated,
            }
        except Exception:
            logger.warning(
                "Energy profiling failed; returning NaN energy measurements.",
                exc_info=True,
            )
            energy = _nan_energy_result()
            energy["duration_s"] = duration_s
        return value, energy


def energy_profile(
    model: Any,
    images: Sequence[Any],
    texts: Sequence[str],
) -> dict[str, Any]:
    """Estimate encoding energy offline; failures are logged and return NaNs."""
    try:
        _, energy = run_with_energy_profile(
            model, lambda: _encode_all(model, images, texts)
        )
        return energy
    except Exception:
        logger.warning(
            "Energy profiling failed; returning NaN measurements.",
            exc_info=True,
        )
        return _nan_energy_result()


def _open_clip_version() -> str:
    try:
        return version("open_clip_torch")
    except PackageNotFoundError:
        return "unavailable"


def hardware_info() -> dict[str, Any]:
    """Report host software/hardware metadata, using NaN for absent GPU fields."""
    has_cuda = False
    gpu_name: str | float = math.nan
    driver_version: str | float = math.nan
    cuda_version: str | float = math.nan
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        logger.warning("Could not query CUDA availability.", exc_info=True)

    if has_cuda:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or math.nan
        except Exception:
            logger.warning("Could not query CUDA device details.", exc_info=True)
        pynvml_module = None
        try:
            import pynvml as pynvml_module

            pynvml_module.nvmlInit()
            raw_version = pynvml_module.nvmlSystemGetDriverVersion()
            driver_version = (
                raw_version.decode("utf-8")
                if isinstance(raw_version, bytes)
                else str(raw_version)
            )
        except Exception:
            logger.warning("Could not query the NVIDIA driver version.", exc_info=True)
        finally:
            if pynvml_module is not None:
                try:
                    pynvml_module.nvmlShutdown()
                except Exception:
                    logger.debug("NVML shutdown failed.", exc_info=True)

    return {
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
        "open_clip_version": _open_clip_version(),
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "energy_profile",
    "hardware_info",
    "latency_profile",
    "peak_gpu_memory_mb",
    "run_with_energy_profile",
]

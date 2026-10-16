"""Compare registered OpenCLIP models on the available benchmark dataset."""

import argparse
import gc
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.data import BenchmarkDataset, load_retrieval_set
from models.metrics import (
    bootstrap_ci,
    recall_at_k,
    retrieval_metrics,
)
from models.profiling import (
    hardware_info,
    latency_profile,
    peak_gpu_memory_mb,
    run_with_energy_profile,
)
from models.registry import MODEL_REGISTRY, EmbeddingModel


logger = logging.getLogger(__name__)
RECALL_KS = (1, 5, 10)
LATENCY_BATCH_SIZES = (1, 16, 64)
CSV_NAME = "model_comparison.csv"
COMPARISON_PLOT_NAME = "model_comparison.png"
PARETO_PLOT_NAME = "model_pareto.png"
SELECTION_NAME = "model_selection.md"
PROFILE_PATH = Path(__file__).resolve().parents[2] / "data" / "profiles" / "gpu_model_profiles.json"


def _encode_batches(
    model: Any,
    items: list[Any],
    batch_size: int = 64,
    encode: str = "images",
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


def _text_items(dataset: BenchmarkDataset) -> tuple[list[str], list[int]]:
    if dataset.task == "retrieval":
        texts = [caption for captions in dataset.captions for caption in captions]
        owners = [
            image_index
            for image_index, captions in enumerate(dataset.captions)
            for _ in captions
        ]
        return texts, owners
    if dataset.task == "classification":
        if dataset.class_prompts is None:
            raise ValueError("Classification dataset lacks class prompts")
        return dataset.class_prompts, []
    raise ValueError(f"Unsupported benchmark task: {dataset.task}")


def _mean_direction_ci(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_image_ids: list[int],
    k: int,
    seed: int = 2026,
    n_resamples: int = 1000,
) -> tuple[float, float, float]:
    similarities = image_embeddings.float() @ text_embeddings.float().T
    image_hits = recall_at_k(
        similarities,
        list(range(image_embeddings.shape[0])),
        caption_image_ids,
        k,
    ).cpu().numpy()
    text_hits = recall_at_k(
        similarities.T,
        caption_image_ids,
        list(range(image_embeddings.shape[0])),
        k,
    ).cpu().numpy()
    estimate = float((image_hits.mean() + text_hits.mean()) / 2)
    rng = np.random.default_rng(seed + k)
    image_samples = rng.integers(
        0, image_hits.size, size=(n_resamples, image_hits.size)
    )
    text_samples = rng.integers(
        0, text_hits.size, size=(n_resamples, text_hits.size)
    )
    bootstrap_estimates = (
        image_hits[image_samples].mean(axis=1)
        + text_hits[text_samples].mean(axis=1)
    ) / 2
    lower, upper = np.quantile(bootstrap_estimates, [0.025, 0.975])
    return estimate, min(estimate, float(lower)), max(estimate, float(upper))


def _empty_result(
    model_name: str,
    dataset: BenchmarkDataset,
    hardware: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_name,
        "dataset": dataset.name,
        "task": dataset.task,
        "device": device,
        "n_images": len(dataset.images),
        "n_texts": sum(len(captions) for captions in dataset.captions)
        if dataset.task == "retrieval"
        else len(dataset.class_prompts or []),
        "parameter_count": math.nan,
        "embedding_dim": math.nan,
        "peak_gpu_memory_mb": math.nan,
        "energy_kwh": math.nan,
        "energy_per_1000_images_kwh": math.nan,
        "avg_power_w": math.nan,
        "pue": 1.0,
        "energy_method": "estimated_from_runtime",
        "components_measured": "",
        "components_estimated": "",
        "error": "",
    }
    for k in RECALL_KS:
        for direction in ("image_to_text", "text_to_image", "mean"):
            prefix = f"{direction}_recall_at_{k}"
            row[prefix] = math.nan
            row[f"{prefix}_ci95_low"] = math.nan
            row[f"{prefix}_ci95_high"] = math.nan
    row["overall_mean_recall"] = math.nan
    row["overall_mean_recall_ci95_low"] = math.nan
    row["overall_mean_recall_ci95_high"] = math.nan
    for k in (1, 5):
        row[f"top{k}"] = math.nan
        row[f"top{k}_ci95_low"] = math.nan
        row[f"top{k}_ci95_high"] = math.nan
    for batch_size in LATENCY_BATCH_SIZES:
        row[f"latency_median_ms_b{batch_size}"] = math.nan
        row[f"latency_iqr_ms_b{batch_size}"] = math.nan
        row[f"images_per_s_b{batch_size}"] = math.nan
    row.update({f"hardware_{key}": value for key, value in hardware.items()})
    return row


def _load_registered_model(name: str, device: str) -> EmbeddingModel:
    return EmbeddingModel.load(MODEL_REGISTRY[name], device=device)


def _profile_one_model(
    model_name: str,
    dataset: BenchmarkDataset,
    hardware: dict[str, Any],
    device: str,
    model_loader: Callable[[str, str], Any],
) -> dict[str, Any]:
    row = _empty_result(model_name, dataset, hardware, device)
    model = None
    try:
        print(f"Loading model: {model_name}", flush=True)
        model = model_loader(model_name, device)
        texts, caption_image_ids = _text_items(dataset)

        def encode_evaluation_set() -> tuple[torch.Tensor, torch.Tensor]:
            images = _encode_batches(model, dataset.images, 64, "images")
            text_vectors = _encode_batches(model, texts, 64, "texts")
            return images, text_vectors

        (image_embeddings, text_embeddings), energy = run_with_energy_profile(
            model, encode_evaluation_set
        )
        row["device"] = device
        row["energy_kwh"] = energy["energy_kwh"]
        row["avg_power_w"] = energy["avg_power_w"]
        row["pue"] = energy.get("pue", 1.0)
        row["energy_method"] = energy.get("energy_method", "estimated_from_runtime")
        row["components_measured"] = "; ".join(energy["components_measured"])
        row["components_estimated"] = "; ".join(energy["components_estimated"])
        if math.isfinite(row["energy_kwh"]) and row["n_images"]:
            row["energy_per_1000_images_kwh"] = (
                row["energy_kwh"] * 1000 / row["n_images"]
            )

        if dataset.task == "retrieval":
            metrics = retrieval_metrics(
                image_embeddings,
                text_embeddings,
                caption_image_ids,
                ks=RECALL_KS,
            )
            for k in RECALL_KS:
                for direction, metric_key in (
                    ("image_to_text", "image_to_text"),
                    ("text_to_image", "text_to_image"),
                ):
                    summary = metrics[metric_key][f"R@{k}"]
                    prefix = f"{direction}_recall_at_{k}"
                    row[prefix] = summary["estimate"]
                    row[f"{prefix}_ci95_low"] = summary["ci95_low"]
                    row[f"{prefix}_ci95_high"] = summary["ci95_high"]
                mean, lower, upper = _mean_direction_ci(
                    image_embeddings, text_embeddings, caption_image_ids, k
                )
                prefix = f"mean_recall_at_{k}"
                row[prefix] = mean
                row[f"{prefix}_ci95_low"] = lower
                row[f"{prefix}_ci95_high"] = upper
            overall = metrics["mean_recall"]
            row["overall_mean_recall"] = overall["estimate"]
            row["overall_mean_recall_ci95_low"] = overall["ci95_low"]
            row["overall_mean_recall_ci95_high"] = overall["ci95_high"]
        else:
            if dataset.labels is None:
                raise ValueError("Classification dataset lacks labels")
            scores = image_embeddings.float() @ text_embeddings.float().T
            for k in (1, 5):
                hits = recall_at_k(
                    scores,
                    dataset.labels,
                    list(range(text_embeddings.shape[0])),
                    k,
                )
                estimate, lower, upper = bootstrap_ci(
                    hits, seed=100 + k
                )
                row[f"top{k}"] = estimate
                row[f"top{k}_ci95_low"] = lower
                row[f"top{k}_ci95_high"] = upper

        latency = latency_profile(
            model,
            batch_sizes=LATENCY_BATCH_SIZES,
        )
        for batch_size, values in latency.items():
            row[f"latency_median_ms_b{batch_size}"] = values["median_ms"]
            row[f"latency_iqr_ms_b{batch_size}"] = values["iqr_ms"]
            row[f"images_per_s_b{batch_size}"] = values["images_per_s"]
        row["peak_gpu_memory_mb"] = peak_gpu_memory_mb(model, batch_size=64)
        if hasattr(model, "parameter_count"):
            row["parameter_count"] = model.parameter_count
        elif hasattr(model, "model"):
            row["parameter_count"] = sum(
                parameter.numel() for parameter in model.model.parameters()
            )
        row["embedding_dim"] = getattr(model, "embedding_dim", math.nan)
        print(
            f"Completed {model_name}: "
            f"mean R@5={row['mean_recall_at_5']:.4f}, "
            f"images/s@64={row['images_per_s_b64']:.2f}, "
            f"energy/1000 images={row['energy_per_1000_images_kwh']:.6g} kWh",
            flush=True,
        )
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        logger.exception("Model %s failed; continuing to the next model.", model_name)
        print(f"Failed {model_name}: {row['error']}", flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _plot_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    valid_recall = frame["mean_recall_at_5"].notna()
    recall_rows = frame.loc[valid_recall]
    x = np.arange(len(recall_rows))
    if not recall_rows.empty:
        y = recall_rows["mean_recall_at_5"].to_numpy(dtype=float)
        lower = y - recall_rows["mean_recall_at_5_ci95_low"].to_numpy(dtype=float)
        upper = recall_rows["mean_recall_at_5_ci95_high"].to_numpy(dtype=float) - y
        axes[0, 0].bar(x, y, yerr=np.vstack([lower, upper]), capsize=4)
        axes[0, 0].set_xticks(x, recall_rows["model"], rotation=30, ha="right")
    axes[0, 0].set_title("Mean recall@5 (95% bootstrap CI)")
    axes[0, 0].set_ylabel("Recall")

    for axis, column, title, ylabel in (
        (axes[0, 1], "images_per_s_b64", "Throughput at batch 64", "Images/s"),
        (axes[1, 0], "peak_gpu_memory_mb", "Peak GPU memory", "MB"),
        (
            axes[1, 1],
            "energy_per_1000_images_kwh",
            "Energy per 1,000 images",
            "kWh",
        ),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        axis.bar(frame.loc[valid, "model"], values[valid])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        if not valid.any():
            axis.text(
                0.5,
                0.5,
                "not available (CPU run)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.tick_params(axis="x", labelrotation=30)
        for label in axis.get_xticklabels():
            label.set_ha("right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_pareto(frame: pd.DataFrame, output_path: Path) -> None:
    valid = frame["mean_recall_at_5"].notna() & frame[
        "energy_per_1000_images_kwh"
    ].notna()
    figure, axis = plt.subplots(figsize=(8, 6))
    points = frame.loc[valid]
    axis.scatter(
        points["energy_per_1000_images_kwh"],
        points["mean_recall_at_5"],
    )
    for _, row in points.iterrows():
        axis.annotate(
            row["model"],
            (row["energy_per_1000_images_kwh"], row["mean_recall_at_5"]),
            xytext=(5, 4),
            textcoords="offset points",
        )
    method = ", ".join(
        sorted({str(m) for m in frame["energy_method"].dropna()})
    )
    axis.set_xlabel(f"Energy per 1,000 images (kWh; {method})")
    axis.set_ylabel("Mean recall@5")
    axis.set_title("Model retrieval-quality / energy comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _select_models(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    valid = frame[
        frame["error"].eq("")
        & frame["mean_recall_at_5"].notna()
        & frame["energy_per_1000_images_kwh"].notna()
    ].copy()
    if valid.empty:
        return None, None
    best_recall = valid["mean_recall_at_5"].max()
    valid["within_one_point"] = valid["mean_recall_at_5"] >= best_recall - 0.01
    valid = valid.sort_values(
        ["within_one_point", "energy_per_1000_images_kwh", "mean_recall_at_5"],
        ascending=[False, True, False],
        kind="stable",
    )
    names = valid["model"].tolist()
    return names[0], names[1] if len(names) > 1 else None


def _write_selection(frame: pd.DataFrame, path: Path) -> tuple[str | None, str | None]:
    winner, runner_up = _select_models(frame)
    table = frame[
        ["model", "mean_recall_at_5", "energy_per_1000_images_kwh", "error"]
    ].copy()
    table.columns = ["Model", "Mean recall@5", "Energy kWh / 1,000 images", "Error"]
    markdown_rows = [
        "| " + " | ".join(str(value) for value in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for values in table.itertuples(index=False, name=None):
        markdown_values = [
            f"{value:.6f}" if isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
            else ("" if pd.isna(value) else str(value))
            for value in values
        ]
        markdown_rows.append("| " + " | ".join(markdown_values) + " |")

    def _unique(column: str) -> str:
        if column not in frame.columns:
            return "unknown"
        values = sorted({str(value) for value in frame[column].dropna()})
        return ", ".join(values) if values else "unknown"

    estimated = "estimated_from_runtime" in _unique("energy_method")
    run_conditions = [
        "## Run conditions",
        "",
        f"- Device: {_unique('device')}",
        f"- Dataset: {_unique('dataset')} (images: {_unique('n_images')})",
        f"- Energy method: {_unique('energy_method')} (PUE: {_unique('pue')})",
        (
            "- Energy is estimated from wall-clock runtime, not read from a power "
            "sensor; treat it as a runtime proxy."
            if estimated
            else "- Energy was measured by codecarbon where components are listed as measured."
        ),
        "",
    ]
    lines = [
        "# Model selection",
        "",
        "## Comparison",
        "",
        *markdown_rows,
        "",
        *run_conditions,
        "## Selection rule",
        "",
        "Choose the model with the highest mean recall@5. Models within "
        "0.01 (one percentage point) of the best recall are tied; among those, "
        "choose the lowest energy per 1,000 images.",
        "",
    ]
    if winner is None:
        lines.extend(
            ["No winner or runner-up: no successful retrieval model has valid "
             "recall@5 and energy measurements.", ""]
        )
    else:
        lines.extend(
            [
                f"**Winner:** {winner}",
                f"**Runner-up:** {runner_up if runner_up is not None else 'None'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return winner, runner_up


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def run_comparison(
    model_names: list[str],
    limit: int | None = None,
    out_dir: str | Path = "docs",
    device: str | None = None,
    dataset: BenchmarkDataset | None = None,
    model_loader: Callable[[str, str], Any] | None = None,
    profile_path: str | Path = PROFILE_PATH,
    allow_cpu: bool = False,
) -> pd.DataFrame:
    """Evaluate each model independently and write comparison artifacts."""
    if not model_names:
        raise ValueError("At least one model must be selected")
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown and model_loader is None:
        raise ValueError(f"Unknown model names: {', '.join(unknown)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.lower() == "cpu" and not allow_cpu:
        raise ValueError(
            "CPU benchmarking is disabled. Pass --allow-cpu to run on CPU; "
            "GPU fields will be empty."
        )
    if selected_device.lower() == "cpu":
        print(
            "WARNING: Benchmarking on CPU via --allow-cpu; GPU fields are empty.",
            flush=True,
        )
    chosen_dataset = dataset or load_retrieval_set(limit=limit)
    print(
        f"Benchmark dataset: {chosen_dataset.name} "
        f"(task: {chosen_dataset.task}, N={len(chosen_dataset.images)})",
        flush=True,
    )
    hardware = hardware_info()
    loader = model_loader or _load_registered_model
    rows = [
        _profile_one_model(name, chosen_dataset, hardware, selected_device, loader)
        for name in model_names
    ]
    frame = pd.DataFrame(rows)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_NAME
    comparison_plot_path = output_dir / COMPARISON_PLOT_NAME
    pareto_plot_path = output_dir / PARETO_PLOT_NAME
    selection_path = output_dir / SELECTION_NAME
    frame.to_csv(csv_path, index=False)
    _plot_comparison(frame, comparison_plot_path)
    _plot_pareto(frame, pareto_plot_path)
    winner, runner_up = _write_selection(frame, selection_path)

    profiles: dict[str, Any] = {}
    for _, row in frame.iterrows():
        profiles[row["model"]] = {
            "device": row["device"],
            "energy_method": row["energy_method"],
            "avg_power_w": row["avg_power_w"],
            "img_per_s_at_batch_64": row["images_per_s_b64"],
            "kwh_per_1000_images": row["energy_per_1000_images_kwh"],
        }
    profile_file = Path(profile_path)
    profile_file.parent.mkdir(parents=True, exist_ok=True)
    profile_file.write_text(
        json.dumps(
            _json_safe(
                {
                    "device": selected_device,
                    "models": profiles,
                    "hardware": hardware,
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved comparison plot: {comparison_plot_path}", flush=True)
    print(f"Saved Pareto plot: {pareto_plot_path}", flush=True)
    print(f"Saved selection: {selection_path}", flush=True)
    print(f"Saved GPU profiles: {profile_file}", flush=True)
    print(f"Winner: {winner or 'None'}; runner-up: {runner_up or 'None'}", flush=True)
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--models",
        default=",".join(MODEL_REGISTRY),
        help="Comma-separated registered model names (default: all).",
    )
    parser.add_argument("--out-dir", default="docs", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the benchmark to run on CPU and record device=cpu.",
    )
    args = parser.parse_args(argv)
    if args.device.lower() == "cpu" and not args.allow_cpu:
        parser.error("CPU benchmarking requires --allow-cpu.")
    logging.basicConfig(level=logging.INFO)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    run_comparison(
        names,
        limit=args.limit,
        out_dir=args.out_dir,
        device=args.device,
        allow_cpu=args.allow_cpu,
    )


if __name__ == "__main__":
    main()

"""Retrieval and classification metrics for image/text embeddings."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def recall_at_k(
    similarities: torch.Tensor,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    k: int,
) -> torch.Tensor:
    """Return per-query recall hits, counting ties at the kth score inclusively."""
    scores = torch.as_tensor(similarities)
    if scores.ndim != 2:
        raise ValueError("similarities must be a two-dimensional matrix")
    if scores.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("ID counts must match the similarity matrix dimensions")
    if k < 1:
        raise ValueError("k must be at least 1")
    if scores.shape[1] == 0:
        raise ValueError("gallery must contain at least one item")

    gallery_ids_tensor = torch.as_tensor(gallery_ids, device=scores.device)
    query_ids_tensor = torch.as_tensor(query_ids, device=scores.device)
    top_k = min(k, scores.shape[1])
    threshold = torch.topk(scores, top_k, dim=1).values[:, -1:]
    in_tied_top_k = scores >= threshold
    relevant = query_ids_tensor[:, None] == gallery_ids_tensor[None, :]
    return (in_tied_top_k & relevant).any(dim=1)


def bootstrap_ci(
    query_hits: Sequence[bool] | torch.Tensor,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return estimate and percentile 95% CI by resampling independent queries."""
    hits = np.asarray(
        query_hits.detach().cpu().numpy()
        if isinstance(query_hits, torch.Tensor)
        else query_hits,
        dtype=np.float64,
    )
    if hits.ndim != 1 or hits.size == 0:
        raise ValueError("query_hits must be a non-empty one-dimensional sequence")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    estimate = float(hits.mean())
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, hits.size, size=(n_resamples, hits.size))
    bootstrap_estimates = hits[samples].mean(axis=1)
    lower, upper = np.quantile(bootstrap_estimates, [0.025, 0.975])
    return estimate, min(estimate, float(lower)), max(estimate, float(upper))


def _metric_summary(hits: torch.Tensor, seed: int, n_resamples: int) -> dict[str, float]:
    estimate, lower, upper = bootstrap_ci(hits, n_resamples=n_resamples, seed=seed)
    return {"estimate": estimate, "ci95_low": lower, "ci95_high": upper}


def retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_image_ids: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute image/text recall@k and mean recall with query bootstrap CIs.

    ``caption_image_ids`` contains the owning image index for each text vector,
    allowing any number of captions per image.
    """
    images = F.normalize(torch.as_tensor(image_embeddings).float(), dim=-1)
    texts = F.normalize(torch.as_tensor(text_embeddings).float(), dim=-1)
    if images.ndim != 2 or texts.ndim != 2:
        raise ValueError("image_embeddings and text_embeddings must be matrices")
    if images.shape[1] != texts.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    if images.shape[0] == 0 or texts.shape[0] == 0:
        raise ValueError("image and text embeddings must be non-empty")
    if len(caption_image_ids) != texts.shape[0]:
        raise ValueError("caption_image_ids must have one ID per text embedding")
    if any(index < 0 or index >= images.shape[0] for index in caption_image_ids):
        raise ValueError("caption_image_ids contains an invalid image index")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    caption_ids = list(caption_image_ids)
    image_ids = list(range(images.shape[0]))
    similarities = images @ texts.T
    image_hits: dict[int, torch.Tensor] = {}
    text_hits: dict[int, torch.Tensor] = {}
    image_to_text: dict[str, dict[str, float]] = {}
    text_to_image: dict[str, dict[str, float]] = {}
    all_bootstrap: list[np.ndarray] = []

    for k in ks:
        i2t_hits = recall_at_k(similarities, image_ids, caption_ids, k)
        t2i_hits = recall_at_k(similarities.T, caption_ids, image_ids, k)
        image_hits[k] = i2t_hits
        text_hits[k] = t2i_hits
        key = f"R@{k}"
        image_to_text[key] = _metric_summary(i2t_hits, seed + k, n_resamples)
        text_to_image[key] = _metric_summary(t2i_hits, seed + 1000 + k, n_resamples)
        for hits, metric_seed in (
            (i2t_hits, seed + 2000 + k),
            (t2i_hits, seed + 3000 + k),
        ):
            rng = np.random.default_rng(metric_seed)
            values = hits.detach().cpu().numpy().astype(np.float64)
            samples = rng.integers(
                0, values.size, size=(n_resamples, values.size)
            )
            all_bootstrap.append(values[samples].mean(axis=1))

    def mean_recall(hits_by_k: dict[int, torch.Tensor], offset: int) -> dict[str, float]:
        per_query = torch.stack([hits_by_k[k].float() for k in ks]).mean(dim=0)
        return _metric_summary(per_query, seed + offset, n_resamples)

    image_to_text["mean_recall"] = mean_recall(image_hits, 4000)
    text_to_image["mean_recall"] = mean_recall(text_hits, 5000)
    overall_estimate = float(
        np.mean(
            [
                image_hits[k].float().mean().item()
                for k in ks
            ]
            + [
                text_hits[k].float().mean().item()
                for k in ks
            ]
        )
    )
    overall_samples = np.stack(all_bootstrap).mean(axis=0)
    overall_lower, overall_upper = np.quantile(overall_samples, [0.025, 0.975])
    overall = {
        "estimate": overall_estimate,
        "ci95_low": min(overall_estimate, float(overall_lower)),
        "ci95_high": max(overall_estimate, float(overall_upper)),
    }
    return {
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "mean_recall": overall,
    }


def classification_topk(
    image_embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    labels: Sequence[int],
    ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    """Compute top-k classification accuracy from image and class-text vectors."""
    images = F.normalize(torch.as_tensor(image_embeddings).float(), dim=-1)
    classes = F.normalize(torch.as_tensor(class_embeddings).float(), dim=-1)
    targets = torch.as_tensor(labels, dtype=torch.long)
    if images.ndim != 2 or classes.ndim != 2 or images.shape[1] != classes.shape[1]:
        raise ValueError("image and class embeddings must be compatible matrices")
    if images.shape[0] != targets.numel() or targets.numel() == 0:
        raise ValueError("labels must contain one entry per non-empty image batch")
    scores = images @ classes.T
    class_ids = list(range(classes.shape[0]))
    return {
        f"top{k}": float(
            recall_at_k(scores, targets.tolist(), class_ids, k)
            .float()
            .mean()
            .item()
        )
        for k in ks
    }


__all__ = ["bootstrap_ci", "classification_topk", "recall_at_k", "retrieval_metrics"]

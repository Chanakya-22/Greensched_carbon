import torch

from models.metrics import (
    bootstrap_ci,
    classification_topk,
    recall_at_k,
    retrieval_metrics,
)


def test_identical_embeddings_have_perfect_recall():
    image_embeddings = torch.eye(12)
    text_embeddings = torch.repeat_interleave(image_embeddings, 2, dim=0)
    caption_image_ids = [index for index in range(12) for _ in range(2)]

    results = retrieval_metrics(
        image_embeddings,
        text_embeddings,
        caption_image_ids,
        n_resamples=100,
    )

    for direction in ("image_to_text", "text_to_image"):
        for k in (1, 5, 10):
            assert results[direction][f"R@{k}"]["estimate"] == 1.0
        assert results[direction]["mean_recall"]["estimate"] == 1.0
    assert results["mean_recall"]["estimate"] == 1.0


def test_random_embeddings_recall_is_near_random_ranking():
    generator = torch.Generator().manual_seed(2026)
    image_embeddings = torch.randn(250, 64, generator=generator)
    text_embeddings = torch.randn(250, 64, generator=generator)
    results = retrieval_metrics(
        image_embeddings,
        text_embeddings,
        list(range(250)),
        n_resamples=100,
        seed=3,
    )

    for direction in ("image_to_text", "text_to_image"):
        for k in (1, 5, 10):
            observed = results[direction][f"R@{k}"]["estimate"]
            expected = k / 250
            assert abs(observed - expected) < 0.04


def test_multi_caption_recall_and_tie_inclusive_cutoff():
    similarities = torch.tensor(
        [
            [0.8, 0.7, 0.1, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0, 0.0],
            [0.6, 0.2, 0.2, 0.2, 0.1],
        ]
    )
    caption_image_ids = [0, 0, 1, 2, 2]

    hits_at_1 = recall_at_k(
        similarities, [0, 1, 2], caption_image_ids, k=1
    )
    hits_at_2 = recall_at_k(
        similarities, [0, 1, 2], caption_image_ids, k=2
    )

    assert hits_at_1.tolist() == [True, True, False]
    assert hits_at_2.tolist() == [True, True, True]


def test_bootstrap_ci_contains_estimate_and_narrows_with_more_queries():
    small = ([True, False] * 20)
    large = ([True, False] * 400)

    small_estimate, small_low, small_high = bootstrap_ci(small, seed=11)
    large_estimate, large_low, large_high = bootstrap_ci(large, seed=11)

    assert small_low <= small_estimate <= small_high
    assert large_low <= large_estimate <= large_high
    assert large_high - large_low < small_high - small_low


def test_classification_fallback_reports_top1_and_top5():
    image_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    class_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )

    results = classification_topk(
        image_embeddings,
        class_embeddings,
        labels=[0, 5],
    )

    assert results == {"top1": 1.0, "top5": 1.0}

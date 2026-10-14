import pytest
import torch
from torch import nn

import open_clip
from models.registry import EmbeddingModel, MODEL_SPECS


def test_registry_names_are_unique():
    names = [spec.name for spec in MODEL_SPECS]
    assert len(names) == len(set(names))


def test_requested_pretrained_pairs_are_available():
    pretrained = set(open_clip.list_pretrained())
    pairs = {(spec.arch, spec.pretrained) for spec in MODEL_SPECS}
    assert pairs == {
        ("ViT-B-32", "openai"),
        ("ViT-B-32", "laion2b_s34b_b79k"),
        ("ViT-B-16-SigLIP", "webli"),
        ("convnext_base_w", "laion2b_s13b_b82k"),
    }
    assert pairs <= pretrained


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = type("Visual", (), {"output_dim": 3})()
        self.weight = nn.Parameter(torch.ones(1))

    def encode_image(self, images):
        return torch.tensor([[3.0, 4.0, 0.0]]).repeat(images.shape[0], 1)

    def encode_text(self, tokens):
        return torch.tensor([[0.0, 0.0, 2.0]]).repeat(tokens.shape[0], 1)


def test_fake_model_uses_embedding_model_interface():
    model = EmbeddingModel(
        FakeModel(),
        preprocess=lambda image: torch.as_tensor(image, dtype=torch.float32),
        tokenizer=lambda texts: torch.ones((len(texts), 1), dtype=torch.long),
        device="cpu",
    )

    image_embeddings = model.encode_images([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    text_embeddings = model.encode_texts(["green energy", "clean grid"])

    assert model.embedding_dim == 3
    assert model.parameter_count == 1
    assert image_embeddings.device.type == "cpu"
    assert image_embeddings.dtype == torch.float32
    assert text_embeddings.device.type == "cpu"
    assert text_embeddings.dtype == torch.float32
    torch.testing.assert_close(
        image_embeddings.norm(dim=-1), torch.ones(2)
    )
    torch.testing.assert_close(
        text_embeddings.norm(dim=-1), torch.ones(2)
    )


@pytest.mark.slow
def test_pretrained_model_produces_unit_norm_embeddings():
    model = EmbeddingModel.load(MODEL_SPECS[0], device="cpu")
    from PIL import Image

    image = Image.new("RGB", (224, 224), color="green")
    image_embedding = model.encode_images([image])
    text_embedding = model.encode_texts(["a green energy system"])

    torch.testing.assert_close(image_embedding.norm(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(text_embedding.norm(dim=-1), torch.ones(1), atol=1e-5, rtol=1e-5)

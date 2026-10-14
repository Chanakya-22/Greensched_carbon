"""OpenCLIP model registry and normalized embedding inference."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str
    pretrained: str
    notes: str


MODEL_SPECS = (
    ModelSpec(
        name="vit_b_32_openai",
        arch="ViT-B-32",
        pretrained="openai",
        notes="OpenAI pretrained ViT-B-32.",
    ),
    ModelSpec(
        name="vit_b_32_laion2b",
        arch="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        notes="LAION-2B pretrained ViT-B-32.",
    ),
    ModelSpec(
        name="vit_b_16_siglip_webli",
        arch="ViT-B-16-SigLIP",
        pretrained="webli",
        notes="WebLI pretrained ViT-B-16-SigLIP.",
    ),
    ModelSpec(
        name="convnext_base_w_laion2b",
        arch="convnext_base_w",
        pretrained="laion2b_s13b_b82k",
        notes="LAION-2B pretrained ConvNeXt-Base-W.",
    ),
)
MODEL_REGISTRY = {spec.name: spec for spec in MODEL_SPECS}


class EmbeddingModel:
    """Wrap an OpenCLIP model and return normalized CPU float32 embeddings."""

    def __init__(
        self,
        model: Any,
        preprocess: Any,
        tokenizer: Any,
        device: str | torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "cpu" else dtype
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.embedding_dim = self._get_embedding_dim()
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        device: str | torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> "EmbeddingModel":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.arch,
            pretrained=spec.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(spec.arch)
        return cls(model, preprocess, tokenizer, device, dtype)

    def encode_images(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            images = batch
        else:
            items = list(batch)
            if not items:
                raise ValueError("Image batch must contain at least one image.")
            processed = [
                item if isinstance(item, torch.Tensor) else self.preprocess(item)
                for item in items
            ]
            images = torch.stack(processed)

        with torch.inference_mode():
            with self._inference_context():
                embeddings = self.model.encode_image(images.to(self.device))
        return self._normalize(embeddings)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("Text batch must contain at least one string.")
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            with self._inference_context():
                embeddings = self.model.encode_text(tokens)
        return self._normalize(embeddings)

    def _inference_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings.float(), p=2, dim=-1).cpu()

    def _get_embedding_dim(self) -> int:
        visual = getattr(self.model, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if output_dim is not None:
            return int(output_dim)
        projection = getattr(self.model, "text_projection", None)
        if projection is not None and projection.ndim > 0:
            return int(projection.shape[-1])
        embed_dim = getattr(self.model, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
        raise ValueError("Unable to determine the model embedding dimension.")


__all__ = ["EmbeddingModel", "MODEL_REGISTRY", "MODEL_SPECS", "ModelSpec"]

"""
clip_tagger.py — CLIP image embedder for landmark lookup and similarity search.

Provides L2-normalized CLIP embeddings for images. Tagging is handled
by RAM++ (see ram_tagger.py); this module is only used for embedding-based
tasks like landmark lookup, duplicate detection, and "find similar".
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("clip_tagger")

DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAINED = "dfn2b_s39b"


class CLIPEmbedder:
    """CLIP image embedder for landmark lookup and similarity search."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
    ):
        import open_clip

        self.model_name = model_name
        self.pretrained = pretrained
        self.model_id = f"{model_name}/{pretrained}"

        log.info("Loading CLIP model %s ...", self.model_id)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self.model.eval()

    def embed_image(self, image_path: Path) -> np.ndarray:
        """Encode an image, return L2-normalized embedding vector."""
        import torch
        from PIL import Image as PILImage

        img = PILImage.open(str(image_path)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32).flatten()

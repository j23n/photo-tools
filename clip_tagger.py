"""
clip_tagger.py — CLIP zero-shot image tagger using Apple label taxonomy.

Uses open_clip to encode images and compare against pre-encoded text prompts
grouped by category. Each category gets its own softmax + threshold + max count.
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from taxonomy import (
    CATEGORY_CONFIG,
    SETTING_VALUES,
    TIME_OF_DAY_VALUES,
    WEATHER_VALUES,
    get_all_categories,
    get_labels_for_category,
    normalize_label,
)

log = logging.getLogger("clip_tagger")

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "laion2b_s34b_b79k"


class CLIPTagger:
    """Zero-shot image tagger using CLIP and Apple label taxonomy."""

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
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Pre-encode all text prompts by category
        self._category_embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
        self._encode_taxonomy()

        # Pre-encode small enum classifiers
        self._enum_embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
        self._encode_enums()

    def _encode_text_prompts(self, prompts: list[str]) -> np.ndarray:
        """Encode a list of text prompts, return L2-normalized embedding matrix."""
        import torch

        tokens = self.tokenizer(prompts)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)

    def _encode_taxonomy(self):
        """Encode all Apple label prompts grouped by category."""
        for category in get_all_categories():
            labels = get_labels_for_category(category)
            if not labels:
                continue
            prompts = [f"a photo of a {label.replace('_', ' ')}" for label in labels]
            embeddings = self._encode_text_prompts(prompts)
            self._category_embeddings[category] = (labels, embeddings)
            log.debug("  %s: %d labels encoded", category, len(labels))

    def _encode_enums(self):
        """Encode small enum classifiers (weather, setting, time_of_day)."""
        enum_groups = {
            "weather": (WEATHER_VALUES, "a photo taken in {value} weather"),
            "setting": (SETTING_VALUES, "a photo taken {value}"),
            "time_of_day": (TIME_OF_DAY_VALUES, "a photo taken at {value}"),
        }
        for name, (values, template) in enum_groups.items():
            prompts = [template.format(value=v) for v in values]
            embeddings = self._encode_text_prompts(prompts)
            self._enum_embeddings[name] = (values, embeddings)

    def _encode_image(self, image_path: Path) -> np.ndarray:
        """Encode an image, return L2-normalized embedding vector."""
        import torch
        from PIL import Image as PILImage

        img = PILImage.open(str(image_path)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32).flatten()

    def tag_image(self, image_path: Path) -> tuple[list[str], np.ndarray]:
        """Tag an image using CLIP zero-shot classification.

        Args:
            image_path: Path to a JPEG/PNG image file.

        Returns:
            (tags, embedding) where tags is a list of prefixed tag strings
            and embedding is the L2-normalized CLIP image embedding (reusable
            for duplicate detection and landmark lookup).
        """
        embedding = self._encode_image(image_path)
        tags = []

        # Per-category classification
        for category, config in CATEGORY_CONFIG.items():
            if category not in self._category_embeddings:
                continue
            labels, text_embeddings = self._category_embeddings[category]
            category_tags = self._classify(
                embedding, labels, text_embeddings,
                max_tags=config["max_tags"],
                threshold=config["threshold"],
                prefix=config["prefix"],
            )
            tags.extend(category_tags)

        # Enum classifiers
        for name, (values, text_embeddings) in self._enum_embeddings.items():
            prefix_map = {"weather": "weather/", "setting": "setting/", "time_of_day": "time/"}
            prefix = prefix_map[name]
            # For enums, take the top-1 result (no threshold — always pick one)
            similarities = embedding @ text_embeddings.T
            best_idx = int(np.argmax(similarities))
            tags.append(f"{prefix}{values[best_idx]}")

        return tags, embedding

    def _classify(
        self,
        image_embedding: np.ndarray,
        labels: list[str],
        text_embeddings: np.ndarray,
        max_tags: int,
        threshold: float,
        prefix: str,
    ) -> list[str]:
        """Classify image against a category's labels using softmax + threshold."""
        # Cosine similarities (embeddings are already L2-normalized)
        similarities = image_embedding @ text_embeddings.T

        # Softmax over category
        exp_sim = np.exp((similarities - similarities.max()) * 100.0)  # temperature scaling
        probs = exp_sim / exp_sim.sum()

        # Sort by probability descending
        sorted_indices = np.argsort(probs)[::-1]

        tags = []
        for idx in sorted_indices[:max_tags]:
            if probs[idx] < threshold:
                break
            label = labels[idx]
            display = normalize_label(label)
            tags.append(f"{prefix}{display}")

        return tags

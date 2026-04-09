"""
clip_tagger.py — Two-stage CLIP zero-shot image tagger.

Stage 1: classify image into broad categories (animal, food, scene, etc.)
          using a small set of category-level prompts where CLIP excels.
Stage 2: within each relevant category, classify specific labels from the
          pruned taxonomy (~60 labels max per category).

Uses raw cosine similarity with per-category thresholds — no broken
temperature-scaled softmax.
"""

import logging
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

DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAINED = "dfn2b_s39b"

# Stage 1: category-level prompts — broad descriptions CLIP is good at
CATEGORY_PROMPTS = {
    "animal":   ["a photo of an animal", "a photo of wildlife"],
    "food":     ["a photo of food", "a photo of a meal", "a photo of a drink"],
    "plant":    ["a photo of a plant", "a photo of flowers", "a photo of a tree"],
    "vehicle":  ["a photo of a vehicle", "a photo of transportation"],
    "object":   ["a photo of an object", "a photo of a thing"],
    "scene":    ["a photo of a place", "a photo of a landscape", "a photo of a room"],
    "activity": ["a photo of people doing an activity", "a photo of a sport"],
    "event":    ["a photo of an event", "a photo of a celebration"],
    "other":    ["a photo of people", "a photo of weather", "a photo of nature"],
}

# Stage 1 thresholds: minimum cosine similarity for a category to be
# considered relevant. These are intentionally low — stage 1 is a gate,
# not a classifier. We'd rather let a category through and have stage 2
# reject all its labels than miss it entirely.
CATEGORY_GATE_THRESHOLD = 0.20

# Max categories to drill into per image (avoids wasting time on
# low-relevance categories)
MAX_STAGE2_CATEGORIES = 4

# Stage 2: minimum cosine similarity for a label to be emitted.
# These replace the old softmax-based thresholds.
STAGE2_THRESHOLDS = {
    "animal":   0.24,
    "food":     0.23,
    "plant":    0.23,
    "vehicle":  0.25,
    "object":   0.20,
    "scene":    0.21,
    "activity": 0.22,
    "event":    0.24,
    "other":    0.22,
}

# Enum classifiers: only emit if top choice leads second by this margin
ENUM_CONFIDENCE_MARGIN = 0.02


class CLIPTagger:
    """Two-stage zero-shot image tagger using CLIP."""

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

        # Stage 1: category-level embeddings
        self._category_gate: dict[str, np.ndarray] = {}
        self._encode_category_prompts()

        # Stage 2: per-category label embeddings
        self._label_embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
        self._encode_taxonomy()

        # Enum classifiers
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

    def _encode_category_prompts(self):
        """Stage 1: encode broad category-level prompts."""
        for category in get_all_categories():
            prompts = CATEGORY_PROMPTS.get(category, [f"a photo related to {category}"])
            embeddings = self._encode_text_prompts(prompts)
            # Average the prompt embeddings for each category, then re-normalize
            avg = embeddings.mean(axis=0)
            avg = avg / np.linalg.norm(avg)
            self._category_gate[category] = avg
            log.debug("  stage1 %s: %d prompts averaged", category, len(prompts))

    def _encode_taxonomy(self):
        """Stage 2: encode per-category label prompts."""
        for category in get_all_categories():
            labels = get_labels_for_category(category)
            if not labels:
                continue
            # Use slightly richer prompt templates per category
            prompts = [self._label_prompt(category, label) for label in labels]
            embeddings = self._encode_text_prompts(prompts)
            self._label_embeddings[category] = (labels, embeddings)
            log.debug("  stage2 %s: %d labels encoded", category, len(labels))

    @staticmethod
    def _label_prompt(category: str, label: str) -> str:
        """Build a natural-language prompt for a label, tuned per category."""
        name = label.replace("_", " ")
        templates = {
            "animal":   "a photo of a {name}",
            "food":     "a photo of {name}",
            "plant":    "a photo of a {name}",
            "vehicle":  "a photo of a {name}",
            "object":   "a photo of a {name}",
            "scene":    "a photo of a {name}",
            "activity": "a photo of people {name}",
            "event":    "a photo of a {name}",
            "other":    "a photo featuring {name}",
        }
        template = templates.get(category, "a photo of a {name}")
        return template.format(name=name)

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
        """Tag an image using two-stage CLIP zero-shot classification.

        Returns:
            (tags, embedding) where tags is a list of prefixed tag strings
            and embedding is the L2-normalized CLIP image embedding.
        """
        embedding = self._encode_image(image_path)
        tags = []

        # ── Stage 1: find relevant categories ──
        category_scores = {}
        for category, gate_embedding in self._category_gate.items():
            score = float(embedding @ gate_embedding)
            if score >= CATEGORY_GATE_THRESHOLD:
                category_scores[category] = score

        # Take top N categories by score
        top_categories = sorted(category_scores, key=category_scores.get, reverse=True)
        top_categories = top_categories[:MAX_STAGE2_CATEGORIES]

        log.debug("Stage 1 categories: %s",
                  {c: f"{category_scores[c]:.3f}" for c in top_categories})

        # ── Stage 2: classify within each relevant category ──
        for category in top_categories:
            if category not in self._label_embeddings:
                continue
            config = CATEGORY_CONFIG[category]
            labels, text_embeddings = self._label_embeddings[category]
            threshold = STAGE2_THRESHOLDS.get(category, 0.22)

            category_tags = self._classify_cosine(
                embedding, labels, text_embeddings,
                max_tags=config["max_tags"],
                threshold=threshold,
                prefix=config["prefix"],
            )
            tags.extend(category_tags)

        # ── Enum classifiers (with confidence gating) ──
        prefix_map = {"weather": "weather/", "setting": "setting/", "time_of_day": "time/"}
        for name, (values, text_embeddings) in self._enum_embeddings.items():
            similarities = embedding @ text_embeddings.T
            sorted_idx = np.argsort(similarities)[::-1]
            best_score = similarities[sorted_idx[0]]
            second_score = similarities[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0
            # Only emit if the winner is clearly ahead
            if best_score - second_score >= ENUM_CONFIDENCE_MARGIN:
                tags.append(f"{prefix_map[name]}{values[sorted_idx[0]]}")

        return tags, embedding

    @staticmethod
    def _classify_cosine(
        image_embedding: np.ndarray,
        labels: list[str],
        text_embeddings: np.ndarray,
        max_tags: int,
        threshold: float,
        prefix: str,
    ) -> list[str]:
        """Classify using raw cosine similarity with a hard threshold."""
        similarities = image_embedding @ text_embeddings.T
        sorted_indices = np.argsort(similarities)[::-1]

        tags = []
        for idx in sorted_indices[:max_tags]:
            sim = similarities[idx]
            if sim < threshold:
                break
            label = labels[idx]
            display = normalize_label(label)
            tags.append(f"{prefix}{display}")
            log.debug("    %s%s  sim=%.3f", prefix, display, sim)

        return tags

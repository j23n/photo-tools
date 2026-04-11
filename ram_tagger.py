"""
ram_tagger.py — RAM++ (Recognize Anything Plus Plus) image tagger.

Replaces the two-stage CLIP zero-shot tagger with a purpose-built
multi-label recognition model. Uses a JSON mapping to convert RAM++
tag predictions to the hierarchical taxonomy (category/tag format).
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("ram_tagger")

DEFAULT_IMAGE_SIZE = 384
DEFAULT_MODEL_REPO = "xinyu1205/recognize-anything-plus-model"
DEFAULT_MODEL_FILENAME = "ram_plus_swin_large_14m.pth"

_MAPPING_PATH = Path(__file__).parent / "ram_tag_mapping.json"

# Per-category max tags (prevents tag spam)
CATEGORY_MAX_TAGS = {
    "animal":   2,
    "food":     3,
    "plant":    2,
    "vehicle":  2,
    "object":   3,
    "scene":    2,
    "activity": 2,
    "event":    1,
    "weather":  1,
    "setting":  1,
    "time":     1,
    "other":    2,
}


class RAMTagger:
    """Image tagger using RAM++ (Recognize Anything Plus Plus)."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE):
        import torch
        from ram import get_transform
        from ram.models import ram_plus

        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Auto-download weights from HuggingFace Hub
        weight_path = self._get_weights()

        log.info("Loading RAM++ model ...")
        self.model = ram_plus(
            pretrained=str(weight_path),
            image_size=image_size,
            vit="swin_l",
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        self.transform = get_transform(image_size=image_size)

        # Load tag mapping
        with open(_MAPPING_PATH) as f:
            self._mapping = json.load(f)
        mapped_count = sum(1 for v in self._mapping.values() if v is not None)
        log.info("RAM++ loaded: %d mapped tags, %d skipped",
                 mapped_count, len(self._mapping) - mapped_count)

    @staticmethod
    def _get_weights() -> Path:
        """Download RAM++ weights from HuggingFace Hub if not cached."""
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            repo_id=DEFAULT_MODEL_REPO,
            filename=DEFAULT_MODEL_FILENAME,
        ))

    def tag_image(self, image_path: Path) -> list[str]:
        """Tag an image using RAM++.

        Returns a list of prefixed tag strings like
        ["animal/dog", "scene/beach", "weather/sunny"].
        """
        import torch
        from PIL import Image as PILImage
        from ram import inference_ram

        img = PILImage.open(str(image_path)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tags_en, _ = inference_ram(img_tensor, self.model)

        raw_tags = [t.strip() for t in tags_en.split(" | ")]
        return self._map_tags(raw_tags)

    def _map_tags(self, raw_tags: list[str]) -> list[str]:
        """Map RAM++ tag strings to prefixed taxonomy tags."""
        # Group by category
        by_category: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for i, raw in enumerate(raw_tags):
            entry = self._mapping.get(raw)
            if entry is None:
                continue
            cat = entry["category"]
            tag = entry["tag"]
            # Use position as priority (RAM++ returns tags in confidence order)
            by_category[cat].append((tag, i))

        # Apply max_tags per category, preferring earlier (higher confidence) tags
        result = []
        for cat, tags_with_pos in by_category.items():
            max_tags = CATEGORY_MAX_TAGS.get(cat, 2)
            # Sort by position (confidence order), deduplicate
            seen = set()
            count = 0
            for tag, _ in sorted(tags_with_pos, key=lambda x: x[1]):
                if tag in seen:
                    continue
                seen.add(tag)
                result.append(f"{cat}/{tag}")
                count += 1
                if count >= max_tags:
                    break

        return result

"""
ram_tagger.py — RAM++ (Recognize Anything Plus Plus) image tagger.

Replaces the two-stage CLIP zero-shot tagger with a purpose-built
multi-label recognition model. Uses a JSON mapping to convert RAM++
tag predictions to the hierarchical taxonomy (category/tag format).
"""

import logging
from collections import defaultdict
from pathlib import Path

import yaml

from photo_tools.config import get_config
from photo_tools.taxonomy import get_max_tags

log = logging.getLogger("ram_tagger")

_MAPPING_PATH = Path(__file__).parent / "data" / "ram_tag_mapping.yaml"


class RAMTagger:
    """Image tagger using RAM++ (Recognize Anything Plus Plus)."""

    def __init__(self, image_size: int | None = None):
        import torch
        from ram import get_transform
        from ram.models import ram_plus

        cfg = get_config()
        self.image_size = image_size or cfg.ram.image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weight_path = self._get_weights()

        log.info("Loading RAM++ model ...")
        self.model = ram_plus(
            pretrained=str(weight_path),
            image_size=self.image_size,
            vit="swin_l",
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        self.transform = get_transform(image_size=self.image_size)

        # `inference_ram` drops the per-tag logits, so hook model.fc to
        # capture them and recover confidence scores after inference.
        self._last_logits = None
        self.model.fc.register_forward_hook(self._capture_logits)

        with open(_MAPPING_PATH) as f:
            self._mapping = yaml.safe_load(f)
        mapped_count = sum(1 for v in self._mapping.values() if v is not None)
        log.info("RAM++ loaded: %d mapped tags, %d skipped",
                 mapped_count, len(self._mapping) - mapped_count)

    def _capture_logits(self, _module, _inputs, output):
        self._last_logits = output

    @staticmethod
    def _get_weights() -> Path:
        """Download RAM++ weights from HuggingFace Hub if not cached."""
        from huggingface_hub import hf_hub_download

        cfg = get_config()
        return Path(hf_hub_download(
            repo_id=cfg.ram.model_repo,
            filename=cfg.ram.model_filename,
        ))

    def tag_image(
        self, image_path: Path,
    ) -> tuple[list[str], list[tuple[str, float, float]]]:
        """Tag an image using RAM++.

        Returns (mapped_tags, scored_raw_tags). scored_raw_tags is the
        RAM++ predicted-tag list as (tag, score, threshold) triples —
        sigmoid confidence and the per-tag threshold the model used to
        decide "predicted", both in [0, 1] — sorted by score descending.
        """
        import torch
        from PIL import Image as PILImage, ImageOps
        from ram import inference_ram

        img = ImageOps.exif_transpose(PILImage.open(str(image_path))).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            tags_en, _ = inference_ram(img_tensor, self.model)

        raw_tags = [t.strip() for t in tags_en.split(" | ")]

        scored_tags = self._score_tags(raw_tags)
        ordered_raw = [t for t, _, _ in scored_tags]
        return self._map_tags(ordered_raw), scored_tags

    def _score_tags(
        self, raw_tags: list[str],
    ) -> list[tuple[str, float, float]]:
        """Attach sigmoid confidence and per-tag threshold to raw_tags.

        Sorted descending by score. Uses logits captured from the last
        forward pass on `model.fc` and per-tag thresholds from
        `model.class_threshold`. Falls back to 0.0 for any tag not found
        in `model.tag_list`.
        """
        import torch

        if self._last_logits is None:
            return [(t, 0.0, 0.0) for t in raw_tags]

        probs = torch.sigmoid(self._last_logits).squeeze().cpu().numpy()
        thresholds = self.model.class_threshold.cpu().numpy()
        index = {str(name): i for i, name in enumerate(self.model.tag_list)}

        scored = []
        for t in raw_tags:
            i = index.get(t)
            if i is None:
                scored.append((t, 0.0, 0.0))
            else:
                scored.append((t, float(probs[i]), float(thresholds[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _map_tags(self, raw_tags: list[str]) -> list[str]:
        """Map RAM++ tag strings to hierarchical taxonomy tags.

        Output is `<Category>/<Tag>` (Titlecase, see docs/xmp-schema.md §2).
        Per-category `max_tags` is enforced. Per-category `min_confidence`
        from taxonomy.py is *not* yet enforced — `inference_ram` returns
        only tag names, not scores. Wiring score extraction is a follow-up.
        """
        by_category: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for i, raw in enumerate(raw_tags):
            entry = self._mapping.get(raw)
            if entry is None:
                continue
            cat = entry["category"]
            tag = entry["tag"]
            by_category[cat].append((tag, i))

        result = []
        for cat, tags_with_pos in by_category.items():
            max_tags = get_max_tags(cat)
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

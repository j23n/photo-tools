"""
taxonomy.py — Apple VNClassifyImageRequest label taxonomy.

Loads category→label mappings from apple_labels.json and provides
per-category config (max tags, threshold, prefix) plus label normalization.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-category configuration
# ---------------------------------------------------------------------------

CATEGORY_CONFIG = {
    "animal":   {"prefix": "animal/",   "max_tags": 2, "threshold": 0.40},
    "food":     {"prefix": "food/",     "max_tags": 3, "threshold": 0.30},
    "plant":    {"prefix": "plant/",    "max_tags": 2, "threshold": 0.40},
    "vehicle":  {"prefix": "vehicle/",  "max_tags": 2, "threshold": 0.45},
    "object":   {"prefix": "object/",   "max_tags": 3, "threshold": 0.25},
    "scene":    {"prefix": "scene/",    "max_tags": 2, "threshold": 0.25},
    "activity": {"prefix": "activity/", "max_tags": 2, "threshold": 0.30},
    "event":    {"prefix": "event/",    "max_tags": 1, "threshold": 0.45},
    "other":    {"prefix": "other/",    "max_tags": 2, "threshold": 0.30},
}

# ---------------------------------------------------------------------------
# Display name overrides (apple_label → display name)
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "jack_o_lantern": "halloween",
    "christmas_tree": "christmas",
    "christmas_decoration": "christmas",
    "easter_egg": "easter",
    "birthday_cake": "birthday",
    "tea_drink": "tea",
    "water_body": "pond",
    "bathroom_room": "bathroom",
    "kitchen_room": "kitchen",
    "living_room": "living-room",
    "dining_room": "dining-room",
}

# ---------------------------------------------------------------------------
# Small enum classifiers (CLIP zero-shot with ~10-20 prompts each)
# ---------------------------------------------------------------------------

WEATHER_VALUES = ["sunny", "cloudy", "overcast", "foggy", "rainy", "snowy", "stormy"]
SETTING_VALUES = ["indoor", "outdoor"]
TIME_OF_DAY_VALUES = ["dawn", "morning", "midday", "afternoon", "golden-hour", "dusk", "night"]

# ---------------------------------------------------------------------------
# Load label→category mapping from JSON
# ---------------------------------------------------------------------------

_DEFAULT_LABELS_PATH = Path(__file__).parent / "apple_labels.json"
_labels_path: Path = _DEFAULT_LABELS_PATH

# category_name → list[label]
_category_to_labels: dict[str, list[str]] = {}
# label → category_name
_label_to_category: dict[str, str] = {}


def set_labels_path(path: Path) -> None:
    """Override the taxonomy JSON path. Must be called before first use."""
    global _labels_path, _category_to_labels, _label_to_category
    _labels_path = path
    # Reset so next access reloads from the new path
    _category_to_labels = {}
    _label_to_category = {}


def _load():
    global _category_to_labels, _label_to_category
    if _label_to_category:
        return
    with open(_labels_path) as f:
        _category_to_labels = json.load(f)
    for cat, labels in _category_to_labels.items():
        for label in labels:
            _label_to_category[label] = cat


def get_category(label: str) -> str | None:
    """Return the category for an Apple label, or None if unknown."""
    _load()
    return _label_to_category.get(label)


def get_labels_for_category(category: str) -> list[str]:
    """Return all labels for a category."""
    _load()
    return list(_category_to_labels.get(category, []))


def get_all_categories() -> list[str]:
    """Return all category names."""
    return list(CATEGORY_CONFIG.keys())


def normalize_label(label: str) -> str:
    """Convert an Apple label to a display-friendly tag value.
    Applies display name overrides, then converts underscores to hyphens."""
    name = DISPLAY_NAMES.get(label, label)
    return name.replace("_", "-")

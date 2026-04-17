"""
taxonomy.py — Tag taxonomy configuration.

Defines the two RAM++ output categories used by the autotagging pipeline.
RAM++ tags are mapped to these categories via data/ram_tag_mapping.json.

See docs/xmp-schema.md §2 for the full taxonomy spec.
"""

CATEGORY_CONFIG = {
    "Objects": {"prefix": "Objects/", "max_tags": 5, "min_confidence": 0.6},
    "Scenes":  {"prefix": "Scenes/",  "max_tags": 3, "min_confidence": 0.4},
}


def get_all_categories() -> list[str]:
    return list(CATEGORY_CONFIG.keys())


def get_max_tags(category: str) -> int:
    entry = CATEGORY_CONFIG.get(category)
    return entry["max_tags"] if entry else 3


def get_min_confidence(category: str) -> float:
    entry = CATEGORY_CONFIG.get(category)
    return entry["min_confidence"] if entry else 0.5

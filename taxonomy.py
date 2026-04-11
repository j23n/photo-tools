"""
taxonomy.py — Tag taxonomy configuration.

Defines the hierarchical tag categories used by the autotagging pipeline.
RAM++ tags are mapped to these categories via ram_tag_mapping.json.
"""

# ---------------------------------------------------------------------------
# Per-category configuration
# ---------------------------------------------------------------------------

CATEGORY_CONFIG = {
    "animal":   {"prefix": "animal/",   "max_tags": 2},
    "food":     {"prefix": "food/",     "max_tags": 3},
    "plant":    {"prefix": "plant/",    "max_tags": 2},
    "vehicle":  {"prefix": "vehicle/",  "max_tags": 2},
    "object":   {"prefix": "object/",   "max_tags": 3},
    "scene":    {"prefix": "scene/",    "max_tags": 2},
    "activity": {"prefix": "activity/", "max_tags": 2},
    "event":    {"prefix": "event/",    "max_tags": 1},
    "weather":  {"prefix": "weather/",  "max_tags": 1},
    "setting":  {"prefix": "setting/",  "max_tags": 1},
    "time":     {"prefix": "time/",     "max_tags": 1},
    "other":    {"prefix": "other/",    "max_tags": 2},
}


def get_all_categories() -> list[str]:
    """Return all category names."""
    return list(CATEGORY_CONFIG.keys())

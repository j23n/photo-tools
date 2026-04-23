"""
taxonomy.py — Tag taxonomy configuration.

Defines the two RAM++ output categories used by the autotagging pipeline.
RAM++ tags are mapped to these categories via data/ram_tag_mapping.yaml.

See docs/xmp-schema.md §2 for the full taxonomy spec.
"""

CATEGORY_CONFIG = {
    "Objects": {"max_tags": 5, "min_confidence": 0.6},
    "Scenes":  {"max_tags": 3, "min_confidence": 0.4},
}

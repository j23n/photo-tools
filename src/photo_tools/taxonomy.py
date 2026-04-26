"""
taxonomy.py — Tag taxonomy configuration.

Defines the two RAM++ output categories used by the autotagging pipeline.
RAM++ tags are mapped to these categories via data/ram_tag_mapping.yaml.

See docs/xmp-schema.md §2 for the full taxonomy spec.
"""

# Multiplier applied to each RAM++ per-class threshold to gate tags:
# a prediction is kept only if its sigmoid score is at least this factor
# times the model's own per-class threshold.
THRESHOLD_MARGIN = 1.10

CATEGORY_CONFIG = {
    "Objects": {"max_tags": 8},
    "Scenes":  {"max_tags": 6},
}

"""Shared constants for photo-tools.

See docs/xmp-schema.md for the XMP/IPTC schema this tool emits.
"""

import re

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic", ".heif", ".dng",
}
VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm",
}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Tagger version. Bump on any schema change (see docs/xmp-schema.md §6).
# Stored in photo-tools:TaggerVersion; presence acts as the "already tagged"
# sentinel and a mismatched value triggers automatic re-tagging.
TAGGER_VERSION = "2026.3"

# Top-level taxonomy roots we own (digiKam:TagsList prefixes). People/ is
# digiKam-owned and intentionally absent from this list.
OUR_TAG_ROOTS = ("Places/", "Landmarks/", "Objects/", "Scenes/")

# OCR validation patterns
OCR_WORD_PATTERN = re.compile(r"^[a-zA-Z0-9À-ÿ][a-zA-Z0-9À-ÿ'.&@#%\-]{0,30}$")
OCR_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüÿ")

# Common word-starting consonant pairs in European languages
VALID_ONSETS = {
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr", "gh", "gl", "gn", "gr",
    "kh", "kl", "kn", "kr", "ph", "pl", "pr", "qu", "sc", "sh", "sk", "sl", "sm",
    "sn", "sp", "sq", "st", "str", "sw", "th", "tr", "tw", "vl", "wh", "wr", "zh",
}

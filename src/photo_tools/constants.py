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
TAGGER_VERSION = "2026.1"

# Top-level taxonomy roots we own (digiKam:TagsList prefixes). People/ is
# digiKam-owned and intentionally absent from this list.
OUR_TAG_ROOTS = ("Places/", "Objects/", "Scenes/")

# OCR text tags use a separate root for tag-tree organization.
OCR_TAG_ROOT = "Text/"

ALL_OUR_ROOTS = OUR_TAG_ROOTS + (OCR_TAG_ROOT,)

# Top-level node names corresponding to our roots (used to garbage-collect
# now-empty parent nodes in digiKam's tag tree).
OUR_TAG_ROOT_NAMES = {r.rstrip("/").lower() for r in ALL_OUR_ROOTS}

# Pre-2026.1 keyword prefixes / bare names. Only used by the migration paths
# (scripts/migrate_xmp.sh and drop_digikam_tags) to clean up files written by
# older versions. Do not extend this for new functionality.
LEGACY_PREFIXES = (
    "country/", "cc/", "region/", "city/", "neighborhood/",
    "landmark/", "scene/", "setting/",
    "object/", "animal/", "plant/", "vehicle/", "food/", "other/",
    "activity/", "event/",
    "weather/", "time/",
    "text/",
    "year/", "month/", "day/",
    "flash/",
)
LEGACY_BARE_TAGS = {"weekend", "weekday", "screenshot", "video", "ai:tagged"}
LEGACY_ROOT_NAMES = {p.rstrip("/").lower() for p in LEGACY_PREFIXES}

# OCR validation patterns
OCR_WORD_PATTERN = re.compile(r"^[a-zA-Z0-9À-ÿ][a-zA-Z0-9À-ÿ'.&@#%\-]{0,30}$")
OCR_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüÿ")

# Common word-starting consonant pairs in European languages
VALID_ONSETS = {
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr", "gh", "gl", "gn", "gr",
    "kh", "kl", "kn", "kr", "ph", "pl", "pr", "qu", "sc", "sh", "sk", "sl", "sm",
    "sn", "sp", "sq", "st", "str", "sw", "th", "tr", "tw", "vl", "wh", "wr", "zh",
}

"""Shared constants for photo-tools."""

import re

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic", ".heif", ".dng",
}
VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm",
}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

AI_TAG_MARKER = "ai:tagged"

# All known tag prefixes — used to identify and clear our tags on --force.
# Uses / separator for DigiKam hierarchical tag support.
ALL_PREFIXES = (
    "country/", "cc/", "region/", "city/", "neighborhood/",
    "landmark/", "scene/", "setting/",
    "object/", "animal/", "plant/", "vehicle/", "food/", "other/",
    "activity/", "event/",
    "weather/", "time/",
    "text/",
    "year/", "month/", "day/",
    "flash/",
)

# Bare (unprefixed) tags we manage
BARE_TAGS = {"weekend", "weekday", "screenshot", "video", AI_TAG_MARKER}

# Top-level category names derived from prefixes
CATEGORY_NAMES = {p.rstrip("/") for p in ALL_PREFIXES}

SCREENSHOT_RESOLUTIONS = {
    (1170, 2532), (1284, 2778), (1179, 2556), (1290, 2796),
    (1242, 2688), (1125, 2436), (1080, 1920), (750, 1334),
    (1242, 2208), (828, 1792),
    (2048, 2732), (1668, 2388), (1668, 2224), (1620, 2160), (1536, 2048),
    (1920, 1080), (2560, 1440), (3840, 2160), (1440, 900),
    (2560, 1600), (1680, 1050), (3024, 1964), (2880, 1800),
    (1080, 2400), (1080, 2340), (1440, 3200), (1440, 3088),
    (1080, 2280), (1440, 2960),
}

# OCR validation patterns
OCR_WORD_PATTERN = re.compile(r"^[a-zA-Z0-9À-ÿ][a-zA-Z0-9À-ÿ'.&@#%\-]{0,30}$")
OCR_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüÿ")

# Common word-starting consonant pairs in European languages
VALID_ONSETS = {
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr", "gh", "gl", "gn", "gr",
    "kh", "kl", "kn", "kr", "ph", "pl", "pr", "qu", "sc", "sh", "sk", "sl", "sm",
    "sn", "sp", "sq", "st", "str", "sw", "th", "tr", "tw", "vl", "wh", "wr", "zh",
}

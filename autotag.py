#!/usr/bin/env python3
"""
autotag.py — Auto-tag photos using CLIP zero-shot classification, GPS reverse
geocoding, Tesseract OCR, and EXIF metadata, then write IPTC/XMP keywords
via ExifTool (with DigiKam hierarchical tag support).

Usage:
    python autotag.py photo.jpg                        # Tag a single file
    python autotag.py ~/Pictures/Vacation/              # Tag a directory
    python autotag.py ~/Pictures/ -r                    # Recursive
    python autotag.py ~/Pictures/ -r --dry-run          # Preview without writing
    python autotag.py ~/Pictures/ -r --force             # Re-tag (replaces old AI tags)
    python autotag.py ~/Pictures/ --watch                # Watch for new files
    python autotag.py photo.jpg --no-geo                 # Skip reverse geocoding
    python autotag.py photo.jpg --no-clip                # Only geo + EXIF + OCR
    python autotag.py photo.jpg --no-ocr                 # Skip Tesseract OCR
    python autotag.py photo.jpg --no-exif                # Skip EXIF-derived tags

Requirements:
    pip install open_clip_torch requests
    brew install exiftool    # or: sudo apt install libimage-exiftool-perl
    brew install tesseract   # or: sudo apt install tesseract-ocr
    brew install ffmpeg      # for video support (optional)

Tag taxonomy (/ separator for hierarchy):
    GPS:    country/ cc/ region/ city/ neighborhood/
    CLIP:   animal/ food/ plant/ vehicle/ object/ scene/ activity/ event/
            weather/ setting/ time/ landmark/
    OCR:    text/
    EXIF:   year/ month/ day/
    People: people/
    Flags:  weekend weekday flash/fired screenshot video ai:tagged
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    print("Error: requests package required. Install with: uv add requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic", ".heif", ".dng",
}
VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm",
}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
AI_TAG_MARKER = "ai:tagged"

# All known prefixes — used to identify and clear our tags on --force
# Uses / separator for DigiKam hierarchical tag support
ALL_PREFIXES = (
    "country/", "cc/", "region/", "city/", "neighborhood/",
    "landmark/", "scene/", "setting/",
    "object/", "animal/", "plant/", "vehicle/", "food/",
    "activity/", "event/",
    "people/",
    "weather/", "time/",
    "text/",
    "year/", "month/", "day/",
    "flash/",
)
# Bare (unprefixed) tags we manage
BARE_TAGS = {"weekend", "weekday", "screenshot", "video", AI_TAG_MARKER}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_USER_AGENT = "autotag-photo-tagger/1.0"
_last_nominatim_call = 0.0

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

OCR_MIN_CONFIDENCE = 60       # per-word confidence threshold (0-100)
OCR_HIGH_CONFIDENCE = 80      # isolated single words need at least this
OCR_MIN_PHRASE_LENGTH = 6     # ignore phrases shorter than this (total chars)
OCR_MAX_TAGS = 10             # cap number of text: tags per image
OCR_WORD_PATTERN = re.compile(r"^[a-zA-Z0-9À-ÿ][a-zA-Z0-9À-ÿ'.&@#%\-]{0,30}$")
OCR_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüÿ")

# Common word-starting consonant pairs in European languages
_VALID_ONSETS = {
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr", "gh", "gl", "gn", "gr",
    "kh", "kl", "kn", "kr", "ph", "pl", "pr", "qu", "sc", "sh", "sk", "sl", "sm",
    "sn", "sp", "sq", "st", "str", "sw", "th", "tr", "tw", "vl", "wh", "wr", "zh",
}

def _is_plausible_word(word: str) -> bool:
    """Reject OCR fragments that aren't real words (no vowels, trailing junk, etc.)."""
    w = word.lower()
    # Must contain at least one vowel (rejects "ssssg", "dldebe", etc.)
    if not any(c in OCR_VOWELS for c in w):
        return False
    # If word starts with 2+ consonants, check they form a valid onset
    # (rejects mid-word fragments like "ndente", "ndomly", "llog")
    if len(w) >= 2 and w[0] not in OCR_VOWELS and w[1] not in OCR_VOWELS:
        if w[:3] not in _VALID_ONSETS and w[:2] not in _VALID_ONSETS:
            return False
    # Reject pure numbers or number-heavy strings
    digits = sum(1 for c in w if c.isdigit())
    if digits > len(w) / 2:
        return False
    return True

# Image sizing — downsize before sending to OCR to save memory and time
CLIP_MAX_PIXELS = 512         # max dimension for CLIP model input
OCR_MAX_PIXELS = 2000         # max dimension for Tesseract (needs a bit more detail)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autotag")


# ---------------------------------------------------------------------------
# EXIF reader
# ---------------------------------------------------------------------------

def read_exif(path: Path) -> dict:
    fields = [
        "-j",
        "-GPS:GPSLatitude", "-GPS:GPSLongitude",
        "-GPS:GPSLatitudeRef", "-GPS:GPSLongitudeRef",
        "-EXIF:Make", "-EXIF:Model",
        "-EXIF:DateTimeOriginal", "-EXIF:CreateDate",
        "-EXIF:Flash",
        "-EXIF:ImageWidth", "-EXIF:ImageHeight",
        "-File:ImageWidth", "-File:ImageHeight",
        "-IPTC:Keywords", "-XMP:Subject",
        "-n",
    ]
    try:
        result = subprocess.run(
            ["exiftool"] + fields + [str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return {}
        meta = json.loads(result.stdout)
        return meta[0] if meta else {}
    except Exception as e:
        log.warning("Could not read EXIF from %s: %s", path, e)
        return {}


def get_existing_keywords(exif: dict) -> set[str]:
    keywords = set()
    for field in ("Keywords", "Subject"):
        val = exif.get(field, [])
        if isinstance(val, str):
            keywords.add(val.lower())
        elif isinstance(val, list):
            keywords.update(v.lower() for v in val)
    return keywords


def read_keywords_batch(paths: list[Path]) -> dict[Path, set[str]]:
    """Read only IPTC:Keywords + XMP:Subject from all paths in a single exiftool call.
    Returns a dict mapping each path to its set of lowercase keywords."""
    if not paths:
        return {}

    result: dict[Path, set[str]] = {}
    # exiftool has a per-invocation arg limit on some platforms; batch in chunks
    BATCH_SIZE = 500
    for i in range(0, len(paths), BATCH_SIZE):
        batch = paths[i:i + BATCH_SIZE]
        try:
            proc = subprocess.run(
                ["exiftool", "-j", "-IPTC:Keywords", "-XMP:Subject"]
                + [str(p) for p in batch],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                # Fall back to per-file reads for this batch
                for p in batch:
                    exif = read_exif(p)
                    result[p] = get_existing_keywords(exif)
                continue
            meta_list = json.loads(proc.stdout)
        except Exception:
            for p in batch:
                exif = read_exif(p)
                result[p] = get_existing_keywords(exif)
            continue

        # exiftool echoes SourceFile as given; build a lookup to match
        # back to the original Path objects in case of minor differences
        str_to_path = {str(p): p for p in batch}
        for meta in meta_list:
            source = meta.get("SourceFile", "")
            path = str_to_path.get(source, Path(source))
            result[path] = get_existing_keywords(meta)

    return result


def is_our_tag(tag: str) -> bool:
    """Check if a tag was generated by this script (has our prefix or is a bare flag)."""
    t = tag.lower()
    if t in BARE_TAGS:
        return True
    for prefix in ALL_PREFIXES:
        if t.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Tag clearing (for --force)
# ---------------------------------------------------------------------------

def clear_our_tags(path: Path, existing: set[str], dry_run: bool = False) -> bool:
    """Remove all tags generated by this script, leaving manual tags untouched."""
    to_remove = [tag for tag in existing if is_our_tag(tag)]
    if not to_remove:
        return True

    if dry_run:
        log.info("[DRY RUN] Would remove %d old tags from %s", len(to_remove), path.name)
        return True

    args = ["exiftool", "-overwrite_original"]
    for tag in to_remove:
        args.append(f"-IPTC:Keywords-={tag}")
        args.append(f"-XMP-dc:Subject-={tag}")
        args.append(f"-XMP-lr:HierarchicalSubject-={hierarchical_subject(tag)}")
        args.append(f"-XMP-digiKam:TagsList-={tag}")
    args.append(str(path))

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.warning("Could not clear old tags from %s: %s", path.name, result.stderr.strip())
            return False
        log.debug("Cleared %d old tags from %s", len(to_remove), path.name)
        return True
    except Exception as e:
        log.warning("Error clearing tags from %s: %s", path.name, e)
        return False


def clear_all_keywords(path: Path, dry_run: bool = False) -> bool:
    """Nuclear option: remove ALL IPTC keywords and XMP subjects from a file."""
    if dry_run:
        log.info("[DRY RUN] Would clear ALL keywords from %s", path.name)
        return True

    args = [
        "exiftool", "-overwrite_original",
        "-IPTC:Keywords=", "-XMP-dc:Subject=",
        "-XMP-lr:HierarchicalSubject=", "-XMP-digiKam:TagsList=",
        str(path),
    ]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.warning("Could not clear keywords from %s: %s", path.name, result.stderr.strip())
            return False
        log.debug("Cleared ALL keywords from %s", path.name)
        return True
    except Exception as e:
        log.warning("Error clearing keywords from %s: %s", path.name, e)
        return False


# ---------------------------------------------------------------------------
# EXIF-derived tags
# ---------------------------------------------------------------------------

def tags_from_exif(exif: dict) -> list[str]:
    tags = []

    # Date
    date_str = exif.get("DateTimeOriginal") or exif.get("CreateDate")
    if date_str and isinstance(date_str, str) and len(date_str) >= 10:
        try:
            import datetime
            clean = date_str.replace("-", ":").replace("T", " ")
            parts = clean.split(":")
            if len(parts) >= 3:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2].split()[0])
                tags.append(f"year/{year}")
                month_names = [
                    "", "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                ]
                if 1 <= month <= 12:
                    tags.append(f"month/{month_names[month]}")
                try:
                    dt = datetime.date(year, month, day)
                    day_names = ["monday", "tuesday", "wednesday", "thursday",
                                 "friday", "saturday", "sunday"]
                    tags.append(f"day/{day_names[dt.weekday()]}")
                    tags.append("weekend" if dt.weekday() >= 5 else "weekday")
                except ValueError:
                    pass
        except (ValueError, IndexError):
            pass

    # Flash
    flash = exif.get("Flash")
    if flash is not None:
        try:
            if int(flash) & 1:
                tags.append("flash/fired")
        except (ValueError, TypeError):
            if "fired" in str(flash).lower() and "not" not in str(flash).lower():
                tags.append("flash/fired")

    # Screenshot detection
    width = exif.get("ImageWidth") or exif.get("File:ImageWidth")
    height = exif.get("ImageHeight") or exif.get("File:ImageHeight")
    if width and height:
        try:
            w, h = int(width), int(height)
            has_camera = bool((exif.get("Make") or "").strip())
            if not has_camera and ((w, h) in SCREENSHOT_RESOLUTIONS
                                  or (h, w) in SCREENSHOT_RESOLUTIONS):
                tags.append("screenshot")
        except (ValueError, TypeError):
            pass

    return tags


# ---------------------------------------------------------------------------
# GPS reverse geocoding
# ---------------------------------------------------------------------------

def get_gps_coords(exif: dict) -> tuple[float, float] | None:
    lat = exif.get("GPSLatitude")
    lon = exif.get("GPSLongitude")
    if lat is None or lon is None:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (ValueError, TypeError):
        return None
    if exif.get("GPSLatitudeRef") == "S" and lat > 0:
        lat = -lat
    if exif.get("GPSLongitudeRef") == "W" and lon > 0:
        lon = -lon
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return (lat, lon)
    return None


def reverse_geocode(lat: float, lon: float) -> dict:
    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"lat": lat, "lon": lon, "format": "jsonv2",
                    "addressdetails": 1, "zoom": 16, "accept-language": "en"},
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=10,
        )
        _last_nominatim_call = time.time()
        resp.raise_for_status()
        return resp.json().get("address", {})
    except Exception as e:
        log.warning("Geocoding failed for (%.4f, %.4f): %s", lat, lon, e)
        _last_nominatim_call = time.time()
        return {}


def tags_from_gps(exif: dict) -> list[str]:
    coords = get_gps_coords(exif)
    if coords is None:
        return []
    lat, lon = coords
    log.debug("GPS: %.5f, %.5f", lat, lon)
    address = reverse_geocode(lat, lon)
    if not address:
        return []

    tags = []
    country = address.get("country")
    if country:
        tags.append(f"country/{country.lower()}")
    cc = address.get("country_code")
    if cc:
        tags.append(f"cc/{cc.lower()}")
    region = (address.get("state") or address.get("region")
              or address.get("province") or address.get("county"))
    if region:
        tags.append(f"region/{region.lower()}")
    city = (address.get("city") or address.get("town")
            or address.get("village") or address.get("municipality"))
    if city:
        tags.append(f"city/{city.lower()}")
    neighborhood = (address.get("suburb") or address.get("neighbourhood")
                    or address.get("quarter") or address.get("district"))
    if neighborhood:
        tags.append(f"neighborhood/{neighborhood.lower()}")
    return tags


# ---------------------------------------------------------------------------
# Image preparation (convert + downsize)
# ---------------------------------------------------------------------------

# Try to import pillow-heif at startup for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HAVE_PILLOW_HEIF = True
except ImportError:
    _HAVE_PILLOW_HEIF = False

try:
    from PIL import Image as PILImage
    _HAVE_PILLOW = True
except ImportError:
    _HAVE_PILLOW = False


def _try_pillow(path: Path, tmp_path: str, max_pixels: int) -> bool:
    """Try converting with Pillow (+ pillow-heif for HEIC). Returns True on success."""
    if not _HAVE_PILLOW:
        return False
    if path.suffix.lower() in (".heic", ".heif") and not _HAVE_PILLOW_HEIF:
        return False
    try:
        img = PILImage.open(str(path))
        # Resize so longest edge = max_pixels, preserving aspect ratio
        img.thumbnail((max_pixels, max_pixels), PILImage.LANCZOS)
        # Convert to RGB (handles RGBA, palette, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(tmp_path, "JPEG", quality=90)
        return Path(tmp_path).stat().st_size > 0
    except Exception:
        return False


def _try_ffmpeg(path: Path, tmp_path: str, max_pixels: int) -> bool:
    """Try converting with ffmpeg. Returns True on success."""
    scale_filter = (
        f"scale='if(gt(iw,ih),{max_pixels},-2)':'if(gt(iw,ih),-2,{max_pixels})'"
    )
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-frames:v", "1",
             "-vf", scale_filter, "-q:v", "2", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and Path(tmp_path).stat().st_size > 0
    except Exception:
        return False


def _try_magick(path: Path, tmp_path: str, max_pixels: int) -> bool:
    """Try converting with ImageMagick (magick or convert). Returns True on success."""
    for cmd in ["magick", "convert"]:
        try:
            result = subprocess.run(
                [cmd, str(path), "-resize", f"{max_pixels}x{max_pixels}>",
                 "-quality", "90", tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and Path(tmp_path).stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


def prepare_image(path: Path, max_pixels: int) -> Path | None:
    """Convert any image to JPEG and downsize to max_pixels on the long edge.
    Returns path to a temp JPEG, or None on failure. Caller must delete the file.
    Tries Pillow first, then ffmpeg, then ImageMagick."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    tmp_path = tmp.name

    try:
        # Pillow + pillow-heif (best HEIC support, fast, handles resize natively)
        if _try_pillow(path, tmp_path, max_pixels):
            return Path(tmp_path)

        # ffmpeg (handles most formats + resizes in one pass)
        if _try_ffmpeg(path, tmp_path, max_pixels):
            return Path(tmp_path)

        # ImageMagick
        if _try_magick(path, tmp_path, max_pixels):
            return Path(tmp_path)

        log.warning("Could not convert %s — install pillow-heif, ffmpeg, or imagemagick",
                    path.name)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None

    except Exception as e:
        log.warning("prepare_image error for %s: %s", path.name, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------

def tags_from_ocr(path: Path) -> list[str]:
    """Run Tesseract OCR and return text: tags. Uses line-grouping: words that
    appear on a line with other words are kept at lower confidence (40%+),
    while isolated single words need higher confidence (70%+) to survive."""

    # Prepare image: convert + downsize
    prepared = prepare_image(path, OCR_MAX_PIXELS)
    ocr_path = prepared or path  # fallback to original if ffmpeg unavailable

    try:
        result = subprocess.run(
            ["tesseract", str(ocr_path), "stdout", "--psm", "11", "-l", "eng", "tsv"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
    except FileNotFoundError:
        log.warning("tesseract not found — skipping OCR. Install: brew install tesseract")
        return []
    except (subprocess.TimeoutExpired, Exception) as e:
        log.warning("Tesseract error on %s: %s", path.name, e)
        return []
    finally:
        if prepared:
            try:
                os.unlink(prepared)
            except OSError:
                pass

    # Parse TSV into structured rows, grouped by line
    # TSV columns: level, page, block, par, line, word, left, top, w, h, conf, text
    lines: dict[str, list[tuple[str, float]]] = {}  # line_key -> [(word, conf), ...]

    for row in result.stdout.strip().split("\n")[1:]:
        fields = row.split("\t")
        if len(fields) < 12:
            continue
        conf_str, word = fields[10].strip(), fields[11].strip()
        if not word or not conf_str:
            continue
        try:
            conf = float(conf_str)
        except ValueError:
            continue
        if conf < OCR_MIN_CONFIDENCE:
            continue
        if not OCR_WORD_PATTERN.match(word):
            continue
        if not _is_plausible_word(word):
            continue

        # Group by page-block-par-line
        line_key = f"{fields[1]}-{fields[2]}-{fields[3]}-{fields[4]}"
        if line_key not in lines:
            lines[line_key] = []
        lines[line_key].append((word, conf))

    # Join words on the same line into phrase tags
    seen = set()
    tags = []

    for line_key, words in lines.items():
        avg_conf = sum(c for _, c in words) / len(words)
        # Isolated single word needs higher confidence
        if len(words) == 1 and avg_conf < OCR_HIGH_CONFIDENCE:
            continue

        phrase = " ".join(w for w, _ in words).lower().strip()
        if len(phrase) < OCR_MIN_PHRASE_LENGTH:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        tags.append(f"text/{phrase}")

        if len(tags) >= OCR_MAX_TAGS:
            break

    if tags:
        log.debug("  OCR found %d text fragments in %s", len(tags), path.name)

    return tags


# ---------------------------------------------------------------------------
# File type validation
# ---------------------------------------------------------------------------

def detect_real_type(path: Path) -> str | None:
    """Check the actual file type by reading magic bytes. Returns 'image', 'video', or None."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except Exception:
        return None

    # JPEG
    if header[:2] == b"\xff\xd8":
        return "image"
    # PNG
    if header[:4] == b"\x89PNG":
        return "image"
    # TIFF
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image"
    # WEBP
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image"
    # HEIF/HEIC — ftyp box with heic/heix/mif1
    if header[4:8] == b"ftyp":
        ftyp = header[8:12].lower()
        if ftyp in (b"heic", b"heix", b"mif1", b"msf1"):
            return "image"
        # QuickTime / MP4 / MOV
        if ftyp in (b"qt  ", b"mp41", b"mp42", b"isom", b"m4v ", b"msnv", b"avc1"):
            return "video"
    # Also catch MOV/MP4 without recognized ftyp
    if header[4:8] == b"ftyp":
        return "video"
    # AVI
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "video"
    # MKV/WebM
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return "video"

    return None


def is_video(path: Path) -> bool:
    """Check if a file is a video, by extension or magic bytes."""
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return True
    real = detect_real_type(path)
    return real == "video"


# ---------------------------------------------------------------------------
# Video frame extraction (requires ffmpeg)
# ---------------------------------------------------------------------------

def extract_video_frame(path: Path) -> Path | None:
    """Extract a single representative frame from a video using ffmpeg.
    Takes a frame at 1 second in (or the first frame for very short clips).
    Returns path to a temp JPEG file, or None on failure."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", "1",           # seek to 1 second
                "-i", str(path),
                "-frames:v", "1",     # extract 1 frame
                "-q:v", "2",          # high quality JPEG
                tmp.name,
            ],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode != 0:
            # Try again at 0 seconds (very short video)
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    tmp.name,
                ],
                capture_output=True, text=True, timeout=30,
            )

        if result.returncode != 0 or not Path(tmp.name).exists() or Path(tmp.name).stat().st_size == 0:
            log.warning("ffmpeg could not extract frame from %s", path.name)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return None

        return Path(tmp.name)

    except FileNotFoundError:
        log.warning("ffmpeg not found — skipping video. Install: brew install ffmpeg")
        return None
    except Exception as e:
        log.warning("Frame extraction failed for %s: %s", path.name, e)
        return None


def deduplicate(tags: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


# ---------------------------------------------------------------------------
# ExifTool writer
# ---------------------------------------------------------------------------

def hierarchical_subject(tag: str) -> str:
    """Convert 'animal/beagle' to 'animal|beagle' for XMP-lr:HierarchicalSubject."""
    return tag.replace("/", "|")


def write_keywords(path: Path, keywords: list[str], dry_run: bool = False) -> bool:
    if not keywords:
        log.warning("No keywords to write for %s", path.name)
        return False
    if dry_run:
        log.info("[DRY RUN] Would write %d keywords to %s:", len(keywords), path.name)
        for kw in sorted(keywords):
            log.info("    %s", kw)
        return True

    # Write four fields per tag for DigiKam/Lightroom hierarchy support:
    #   IPTC:Keywords           = animal/beagle
    #   XMP-dc:Subject          = animal/beagle
    #   XMP-lr:HierarchicalSubject = animal|beagle
    #   XMP-digiKam:TagsList    = animal/beagle
    args = ["exiftool", "-overwrite_original"]
    for kw in keywords:
        args.append(f"-IPTC:Keywords+={kw}")
        args.append(f"-XMP-dc:Subject+={kw}")
        args.append(f"-XMP-lr:HierarchicalSubject+={hierarchical_subject(kw)}")
        args.append(f"-XMP-digiKam:TagsList+={kw}")
    args.append(str(path))

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.error("exiftool failed for %s: %s", path.name, result.stderr.strip())
            return False
        log.info("Wrote %d keywords to %s", len(keywords), path.name)
        return True
    except FileNotFoundError:
        log.error("exiftool not found. Install with: brew install exiftool")
        sys.exit(1)
    except Exception as e:
        log.error("Failed to write keywords to %s: %s", path.name, e)
        return False


# ---------------------------------------------------------------------------
# CLIP embedding cache (stored in XMP, reused by find-similar)
# ---------------------------------------------------------------------------

def read_clip_cache(path: Path) -> dict:
    result = subprocess.run(
        [
            "exiftool", "-j",
            "-XMP-phototools:CLIPEmbedding",
            "-XMP-phototools:CLIPModel",
            "-XMP-phototools:CLIPTimestamp",
            str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return {}
    try:
        meta = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return meta[0] if meta else {}


def read_cached_embedding(path: Path, model: str) -> "np.ndarray | None":
    cache = read_clip_cache(path)
    if cache.get("CLIPModel") != model:
        return None
    b64 = cache.get("CLIPEmbedding", "")
    if not b64:
        return None
    try:
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()
    except Exception:
        return None


def write_embedding(path: Path, vec: "np.ndarray", model: str, dry_run: bool) -> None:
    b64 = base64.b64encode(vec.astype(np.float32).tobytes()).decode()
    ts = datetime.now(timezone.utc).isoformat()
    if dry_run:
        log.info("[DRY RUN] Would cache embedding for %s", path.name)
        return
    subprocess.run(
        [
            "exiftool", "-overwrite_original",
            f"-XMP-phototools:CLIPEmbedding={b64}",
            f"-XMP-phototools:CLIPModel={model}",
            f"-XMP-phototools:CLIPTimestamp={ts}",
            str(path),
        ],
        capture_output=True, timeout=60,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_images(target: Path, recursive: bool = False) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [target]
        log.warning("Unsupported file type: %s", target)
        return []
    if not target.is_dir():
        log.error("Path does not exist: %s", target)
        return []

    images = []
    pattern = "**/*" if recursive else "*"
    for f in target.glob(pattern):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(f)

    seen = set()
    deduped = []
    for img in sorted(images):
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(img)
    return deduped


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

# Lazy-initialized singletons
_clip_tagger = None
_landmark_index = None


def _get_clip_tagger(clip_model: str = None, clip_pretrained: str = None):
    global _clip_tagger
    if _clip_tagger is None:
        from clip_tagger import CLIPTagger, DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED
        _clip_tagger = CLIPTagger(
            model_name=clip_model or DEFAULT_CLIP_MODEL,
            pretrained=clip_pretrained or DEFAULT_CLIP_PRETRAINED,
        )
    return _clip_tagger


def _get_landmark_index(landmarks_path: Path = None):
    global _landmark_index
    if _landmark_index is None:
        from landmarks import LandmarkIndex, DEFAULT_LANDMARKS_PATH
        lm_path = landmarks_path or DEFAULT_LANDMARKS_PATH
        if not lm_path.exists():
            log.debug("No landmarks database at %s, skipping landmark lookup", lm_path)
            return None
        _landmark_index = LandmarkIndex(lm_path)
    return _landmark_index


def process_single(
    path: Path,
    dry_run: bool,
    force: bool,
    clear_all: bool = False,
    enable_clip: bool = True,
    enable_geo: bool = True,
    enable_exif: bool = True,
    enable_ocr: bool = True,
    clip_model: str = None,
    clip_pretrained: str = None,
    landmarks_path: Path = None,
) -> bool:
    # Detect files where extension doesn't match content (e.g. Live Photos)
    video = is_video(path)
    if not video and path.suffix.lower() in IMAGE_EXTENSIONS:
        real_type = detect_real_type(path)
        if real_type == "video":
            log.info("Skipping %s (extension is %s but file is actually a video/Live Photo)",
                     path.name, path.suffix)
            return False

    exif = read_exif(path)
    existing = get_existing_keywords(exif)

    if AI_TAG_MARKER in existing and not force and not clear_all:
        log.info("Skipping %s (already tagged, use --force to re-tag)", path.name)
        return False

    # --clear-all: wipe every keyword, start from scratch
    if clear_all and existing:
        clear_all_keywords(path, dry_run)
        existing = set()
    # --force: remove only tags we recognize as ours
    elif force and existing:
        clear_our_tags(path, existing, dry_run)
        existing = {t for t in existing if not is_our_tag(t)}

    log.info("Processing %s ...", path.name)
    all_tags = []

    # For videos, extract a frame for CLIP and OCR
    frame_path = None
    if video:
        all_tags.append("video")
        frame_path = extract_video_frame(path)
        if frame_path is None:
            log.warning("Could not extract frame from %s, running EXIF/GPS only", path.name)

    # The image to use for CLIP and OCR (original file or extracted frame)
    visual_path = frame_path if video else path

    if enable_exif:
        t = tags_from_exif(exif)
        if t:
            log.debug("  EXIF: %s", t)
            all_tags.extend(t)

    if enable_geo:
        t = tags_from_gps(exif)
        if t:
            log.debug("  Geo:  %s", t)
            all_tags.extend(t)

    if enable_ocr and visual_path:
        t = tags_from_ocr(visual_path)
        if t:
            log.debug("  OCR:  %s", t)
            all_tags.extend(t)

    embedding = None
    if enable_clip and visual_path:
        # Prepare image for CLIP (needs JPEG)
        prepared = prepare_image(visual_path, CLIP_MAX_PIXELS)
        clip_input = prepared or visual_path
        try:
            tagger = _get_clip_tagger(clip_model, clip_pretrained)
            clip_tags, embedding = tagger.tag_image(clip_input)
            log.debug("  CLIP: %s", clip_tags)
            all_tags.extend(clip_tags)
        except Exception as e:
            log.warning("CLIP tagging failed for %s: %s", path.name, e)
        finally:
            if prepared:
                try:
                    os.unlink(prepared)
                except OSError:
                    pass

        # Landmark lookup using the CLIP embedding
        if embedding is not None:
            coords = get_gps_coords(exif)
            lat, lon = coords if coords else (None, None)
            try:
                lm_index = _get_landmark_index(landmarks_path)
                if lm_index is not None:
                    landmark = lm_index.lookup(embedding, lat=lat, lon=lon)
                    if landmark:
                        tag = f"landmark/{landmark.lower().replace(' ', '-')}"
                        log.debug("  Landmark: %s", tag)
                        all_tags.append(tag)
            except Exception as e:
                log.debug("Landmark lookup failed: %s", e)

    # Cache CLIP embedding in XMP for duplicate detection reuse
    if embedding is not None and not dry_run:
        model_id = _get_clip_tagger(clip_model, clip_pretrained).model_id
        write_embedding(path, embedding, model_id, dry_run)

    # Clean up temp frame
    if frame_path:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    all_tags = deduplicate(all_tags)

    # Don't re-add tags that already exist (manual ones that survived clearing)
    new_tags = [t for t in all_tags if t.lower() not in existing]

    if AI_TAG_MARKER not in existing:
        new_tags.append(AI_TAG_MARKER)

    if not new_tags:
        log.info("No new keywords for %s", path.name)
        return True

    return write_keywords(path, new_tags, dry_run)


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_directory(target, recursive, dry_run,
                    enable_clip, enable_geo, enable_exif, enable_ocr,
                    clip_model=None, clip_pretrained=None, landmarks_path=None):
    log.info("Watching %s for new images (Ctrl+C to stop) ...", target)
    seen: set[Path] = set()

    for img in find_images(target, recursive):
        exif = read_exif(img)
        if AI_TAG_MARKER in get_existing_keywords(exif):
            seen.add(img.resolve())
    log.info("Found %d already-tagged images, skipping those.", len(seen))

    while True:
        try:
            for img in find_images(target, recursive):
                resolved = img.resolve()
                if resolved in seen:
                    continue
                exif = read_exif(img)
                if AI_TAG_MARKER in get_existing_keywords(exif):
                    seen.add(resolved)
                    continue
                process_single(img, dry_run, force=False,
                               enable_clip=enable_clip, enable_geo=enable_geo,
                               enable_exif=enable_exif, enable_ocr=enable_ocr,
                               clip_model=clip_model, clip_pretrained=clip_pretrained,
                               landmarks_path=landmarks_path)
                seen.add(resolved)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Watch stopped.")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_tag_args(parser: argparse.ArgumentParser) -> None:
    """Add all 'tag' subcommand arguments to the given parser."""
    parser.add_argument("path", type=Path, help="Image file or directory")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Preview tags without writing")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Re-tag (clears old autogenerated tags first)")
    parser.add_argument("--clear-all", action="store_true",
                        help="Wipe ALL keywords before re-tagging (nuclear option)")
    parser.add_argument("-w", "--watch", action="store_true",
                        help="Watch directory for new images")
    parser.add_argument("--clip-model", default=None,
                        help="CLIP model name (default: ViT-B-32)")
    parser.add_argument("--clip-pretrained", default=None,
                        help="CLIP pretrained weights (default: laion2b_s34b_b79k)")
    parser.add_argument("--landmarks", type=Path, default=None,
                        help="Path to landmarks.json (default: ~/.local/share/photo-tools/landmarks.json)")
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip CLIP tagging (only geo + EXIF + OCR)")
    parser.add_argument("--no-geo", action="store_true")
    parser.add_argument("--no-exif", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")


def build_tag_parser(subparsers) -> argparse.ArgumentParser:
    """Register the 'tag' subcommand on the given subparsers object."""
    sub = subparsers.add_parser(
        "tag",
        help="Auto-tag photos using CLIP zero-shot classification, GPS geocoding, OCR, and EXIF metadata.",
    )
    _add_tag_args(sub)
    sub.set_defaults(func=run_tag)
    return sub


def run_tag(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    enable_clip = not args.no_clip
    enable_geo = not args.no_geo
    enable_exif = not args.no_exif
    enable_ocr = not args.no_ocr

    sources = []
    if enable_exif: sources.append("EXIF")
    if enable_geo:  sources.append("GPS")
    if enable_ocr:  sources.append("OCR")
    if enable_clip: sources.append(f"CLIP ({args.clip_model or 'ViT-B-32'})")
    log.info("Tag sources: %s", " + ".join(sources))

    if args.watch:
        if not args.path.is_dir():
            log.error("--watch requires a directory")
            sys.exit(1)
        watch_directory(args.path, args.recursive, args.dry_run,
                        enable_clip, enable_geo, enable_exif, enable_ocr,
                        clip_model=args.clip_model,
                        clip_pretrained=args.clip_pretrained,
                        landmarks_path=args.landmarks)
        return

    images = find_images(args.path, args.recursive)
    if not images:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Found %d image(s) to process", len(images))
    success = failed = skipped = 0

    for i, img in enumerate(images, 1):
        log.info("[%d/%d] %s", i, len(images), img)
        try:
            result = process_single(img, args.dry_run, args.force,
                                    clear_all=args.clear_all,
                                    enable_clip=enable_clip, enable_geo=enable_geo,
                                    enable_exif=enable_exif, enable_ocr=enable_ocr,
                                    clip_model=args.clip_model,
                                    clip_pretrained=args.clip_pretrained,
                                    landmarks_path=args.landmarks)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            log.error("Error processing %s: %s", img.name, e)
            failed += 1

    log.info("Done. %d tagged, %d skipped, %d failed.", success, skipped, failed)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tag photos using CLIP, GPS geocoding, OCR, and EXIF metadata.",
    )
    _add_tag_args(parser)
    args = parser.parse_args()
    run_tag(args)


if __name__ == "__main__":
    main()

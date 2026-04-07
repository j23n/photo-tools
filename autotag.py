#!/usr/bin/env python3
"""
autotag.py — Auto-tag photos using a local vision model, GPS reverse geocoding,
Tesseract OCR, and EXIF metadata, then write IPTC/XMP keywords via ExifTool.

Usage:
    python autotag.py photo.jpg                        # Tag a single file
    python autotag.py ~/Pictures/Vacation/              # Tag a directory
    python autotag.py ~/Pictures/ -r                    # Recursive
    python autotag.py ~/Pictures/ -r --dry-run          # Preview without writing
    python autotag.py ~/Pictures/ -r --force             # Re-tag (replaces old AI tags)
    python autotag.py ~/Pictures/ --watch                # Watch for new files
    python autotag.py --list-models                      # Show available models
    python autotag.py photo.jpg --no-geo                 # Skip reverse geocoding
    python autotag.py photo.jpg --no-ai                  # Only geo + EXIF + OCR
    python autotag.py photo.jpg --no-ocr                 # Skip Tesseract OCR
    python autotag.py photo.jpg --no-exif                # Skip EXIF-derived tags

Environment:
    AI_BASE_URL   Base URL of the OpenAI-compatible API  (default: http://100.64.0.4:8000/v1)
    AI_API_KEY    API key                                 (default: none)
    AI_MODEL      Model name                              (default: gemma4)

Requirements:
    pip install requests
    brew install exiftool    # or: sudo apt install libimage-exiftool-perl
    brew install tesseract   # or: sudo apt install tesseract-ocr
    brew install ffmpeg      # for video support (optional)

Tag taxonomy:
    GPS:    country: cc: region: city: neighborhood:
    AI:     landmark: architecture: scene: setting:
            object: animal: plant: vehicle: food:
            activity: event: cuisine:
            people: age:
            comp: mood: color:
            weather: season: time:
    OCR:    text:
    EXIF:   year: month: day:
    Flags:  weekend weekday flash:fired screenshot video ai:tagged
"""

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests package required. Install with: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://100.64.0.4:8000/v1"
DEFAULT_MODEL = "gemma4"
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".heic", ".heif", ".dng",
}
VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm",
}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
AI_TAG_MARKER = "ai:tagged"

# All known prefixes — used to identify and clear our tags on --force
ALL_PREFIXES = (
    "country:", "cc:", "region:", "city:", "neighborhood:",
    "landmark:", "architecture:", "scene:", "setting:",
    "object:", "animal:", "plant:", "vehicle:", "food:",
    "activity:", "event:", "cuisine:",
    "people:", "age:",
    "comp:", "mood:", "color:",
    "weather:", "season:", "time:",
    "text:",
    "year:", "month:", "day:",
    "flash:",
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

OCR_MIN_CONFIDENCE = 40       # per-word confidence threshold (0-100)
OCR_HIGH_CONFIDENCE = 70      # isolated single words need at least this
OCR_MIN_WORD_LENGTH = 2       # ignore fragments shorter than this
OCR_MAX_TAGS = 15             # cap number of text: tags per image
OCR_WORD_PATTERN = re.compile(r"^[a-zA-Z0-9À-ÿ][a-zA-Z0-9À-ÿ'.&@#%\-/ ]{0,50}$")

# Image sizing — downsize before sending to AI/OCR to save memory and time
AI_MAX_PIXELS = 1500          # max dimension (long edge) for vision model
OCR_MAX_PIXELS = 2000         # max dimension for Tesseract (needs a bit more detail)

SYSTEM_PROMPT = """\
You are an image metadata tagger for a personal photo library. Analyze the \
photograph and return a JSON object with descriptive keywords.

Rules:
- Use lowercase. No articles or filler words.
- Be SPECIFIC: "golden retriever" not "dog", "carbonara" not "food", \
"half dome" not "rock". Generic terms like "trees" or "buildings" are \
only acceptable if nothing more specific applies.
- Do NOT repeat the same concept across fields. If something is a vehicle, \
put it ONLY in "vehicle", not also in "object". If it's an animal, ONLY \
in "animal". If it's food, ONLY in "food". If it's a plant, ONLY in "plant".
- "object" is ONLY for things that don't fit vehicle/animal/plant/food.
- For landmarks, use the proper name ONLY if confident. Otherwise null.
- For people, describe by visible attributes only. Never guess identities.
- Only tag what is clearly visible. A missing tag is better than a wrong tag.
- Pick the SINGLE BEST value for each enum field. Don't hedge.
- Respect the max counts shown below. Prefer fewer, better tags over many vague ones.

Return ONLY a JSON object:
{
  "object":         ["max 5 — specific things NOT covered by vehicle/animal/plant/food"],
  "animal":         ["max 3 — specific species or breed"],
  "plant":          ["max 3 — specific species, NOT generic like 'trees' or 'foliage'"],
  "vehicle":        ["max 3 — specific vehicles"],
  "food":           ["max 3 — specific dishes or ingredients"],
  "scene":          ["max 2 — the dominant scene: street, beach, forest, kitchen, etc."],
  "setting":        "indoor | outdoor | unknown",
  "landmark":       "proper name or null",
  "architecture":   "specific style (art deco, gothic, brutalist, etc.) or null",
  "activity":       ["max 3 — what people or animals are doing"],
  "event":          "event type or null",
  "cuisine":        "cuisine type or null",
  "people":         "solo | couple | small group | crowd | none",
  "age":            ["max 2 — age groups visible: infant, child, teenager, adult, elderly"],
  "comp":           ["max 2 — notable composition ONLY: aerial, macro, silhouette, bokeh, panoramic, reflection, long exposure. Skip if nothing stands out."],
  "mood":           ["max 2 — dominant mood: serene, dramatic, cozy, chaotic, melancholy, festive, gritty"],
  "color":          ["max 2 — dominant color character: warm tones, cool tones, muted, monochrome, golden, etc."],
  "weather":        "sunny | cloudy | overcast | foggy | rainy | snowy | stormy | unknown",
  "season":         "spring | summer | autumn | winter | unknown",
  "time_of_day":    "dawn | morning | midday | afternoon | golden hour | dusk | night | unknown"
}
"""

USER_PROMPT = "Tag this photograph."


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
        args.append(f"-XMP:Subject-={tag}")
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
        "-IPTC:Keywords=", "-XMP:Subject=",
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
                tags.append(f"year:{year}")
                month_names = [
                    "", "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                ]
                if 1 <= month <= 12:
                    tags.append(f"month:{month_names[month]}")
                try:
                    dt = datetime.date(year, month, day)
                    day_names = ["monday", "tuesday", "wednesday", "thursday",
                                 "friday", "saturday", "sunday"]
                    tags.append(f"day:{day_names[dt.weekday()]}")
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
                tags.append("flash:fired")
        except (ValueError, TypeError):
            if "fired" in str(flash).lower() and "not" not in str(flash).lower():
                tags.append("flash:fired")

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
        tags.append(f"country:{country.lower()}")
    cc = address.get("country_code")
    if cc:
        tags.append(f"cc:{cc.lower()}")
    region = (address.get("state") or address.get("region")
              or address.get("province") or address.get("county"))
    if region:
        tags.append(f"region:{region.lower()}")
    city = (address.get("city") or address.get("town")
            or address.get("village") or address.get("municipality"))
    if city:
        tags.append(f"city:{city.lower()}")
    neighborhood = (address.get("suburb") or address.get("neighbourhood")
                    or address.get("quarter") or address.get("district"))
    if neighborhood:
        tags.append(f"neighborhood:{neighborhood.lower()}")
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
        if len(word) < OCR_MIN_WORD_LENGTH:
            continue
        if not OCR_WORD_PATTERN.match(word):
            continue

        # Group by page-block-par-line
        line_key = f"{fields[1]}-{fields[2]}-{fields[3]}-{fields[4]}"
        if line_key not in lines:
            lines[line_key] = []
        lines[line_key].append((word, conf))

    # Now decide which words to keep
    seen = set()
    tags = []

    for line_key, words in lines.items():
        line_has_multiple = len(words) >= 2

        for word, conf in words:
            # Isolated word on its own line: needs high confidence
            if not line_has_multiple and conf < OCR_HIGH_CONFIDENCE:
                continue

            normalized = word.lower().strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            tags.append(f"text:{normalized}")

            if len(tags) >= OCR_MAX_TAGS:
                break

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


# ---------------------------------------------------------------------------
# AI vision client
# ---------------------------------------------------------------------------

class VisionClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def list_models(self) -> list[str]:
        resp = self.session.get(f"{self.base_url}/models", timeout=10)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def tag_image(self, image_path: Path) -> dict | None:
        encoded = self._encode_image(image_path)
        if encoded is None:
            log.error("Cannot process %s for AI — no converter available", image_path.name)
            return None
        b64_data, media_type = encoded
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{b64_data}"}},
                ]},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        try:
            resp = self.session.post(f"{self.base_url}/chat/completions",
                                     json=payload, timeout=180)
            resp.raise_for_status()
        except requests.ConnectionError:
            log.error("Cannot connect to %s", self.base_url)
            return None
        except requests.Timeout:
            log.error("Timed out for %s", image_path.name)
            return None
        except requests.HTTPError as e:
            log.error("API error for %s: %s", image_path.name, e.response.text[:200])
            return None

        content = resp.json()["choices"][0]["message"]["content"].strip()
        return self._parse_json(content, image_path)

    @staticmethod
    def _encode_image(path: Path) -> tuple[str, str] | None:
        # Formats the vision API can handle natively
        NATIVE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

        prepared = prepare_image(path, AI_MAX_PIXELS)
        if prepared:
            log.debug("Prepared %s -> %s (%d bytes)",
                      path.name, prepared.name, prepared.stat().st_size)
            with open(prepared, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(prepared)
            return data, "image/jpeg"

        # Only fall back to raw for formats the model can actually read
        if path.suffix.lower() in NATIVE_FORMATS:
            log.debug("Sending raw %s (native format, prepare_image unavailable)", path.name)
            mime, _ = mimetypes.guess_type(str(path))
            if mime is None:
                mime = "image/jpeg"
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return data, mime

        # Non-native format (HEIC, DNG, TIFF, etc.) and no converter — give up
        return None

    @staticmethod
    def _parse_json(content: str, path: Path) -> dict | None:
        if "<|channel>" in content:
            parts = content.split("<channel|>")
            if len(parts) > 1:
                content = parts[-1].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass
            log.error("Could not parse JSON for %s", path.name)
            log.warning("Model response was: %s", content[:1000])
            return None


# ---------------------------------------------------------------------------
# Tag flattening with deduplication
# ---------------------------------------------------------------------------

def normalize_tag_value(v: str) -> str:
    """Normalize a tag value: lowercase, strip, collapse whitespace, use hyphens."""
    v = str(v).lower().strip()
    v = re.sub(r"\s+", "-", v)  # "eye level" -> "eye-level"
    return v


def flatten_ai_tags(parsed: dict) -> list[str]:
    tags = []
    seen_values = set()  # track raw values to prevent cross-category dupes

    # List fields with their prefix
    list_fields = [
        # Order matters: specific categories first, generic last
        ("vehicle",  "vehicle"),
        ("animal",   "animal"),
        ("plant",    "plant"),
        ("food",     "food"),
        ("object",   "object"),  # catch-all — skip if value already used above
        ("scene",    "scene"),
        ("activity", "activity"),
        ("age",      "age"),
        ("comp",     "comp"),
        ("mood",     "mood"),
        ("color",    "color"),
    ]
    for field, prefix in list_fields:
        values = parsed.get(field, [])
        if isinstance(values, list):
            for v in values:
                v = normalize_tag_value(v)
                if not v:
                    continue
                # Skip if this value was already used in a more specific category
                if field == "object" and v in seen_values:
                    continue
                if v not in seen_values:
                    seen_values.add(v)
                    tags.append(f"{prefix}:{v}")

    # Enum fields
    for field, prefix in [("setting", "setting"), ("time_of_day", "time"),
                          ("season", "season"), ("weather", "weather"),
                          ("people", "people")]:
        val = parsed.get(field, "unknown")
        if val and str(val).lower() not in ("unknown", "null", "none", ""):
            tags.append(f"{prefix}:{normalize_tag_value(val)}")

    # Nullable fields
    for field, prefix in [("landmark", "landmark"), ("cuisine", "cuisine"),
                          ("event", "event"), ("architecture", "architecture")]:
        val = parsed.get(field)
        if val and str(val).lower() not in ("null", "none", ""):
            tags.append(f"{prefix}:{normalize_tag_value(val)}")

    return tags


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

def write_keywords(path: Path, keywords: list[str], dry_run: bool = False) -> bool:
    if not keywords:
        log.warning("No keywords to write for %s", path.name)
        return False
    if dry_run:
        log.info("[DRY RUN] Would write %d keywords to %s:", len(keywords), path.name)
        for kw in sorted(keywords):
            log.info("    %s", kw)
        return True

    # Use = (replace) for IPTC and XMP to write the exact set we want.
    # We build the full keyword list and set it atomically.
    args = ["exiftool", "-overwrite_original"]
    for kw in keywords:
        args.append(f"-IPTC:Keywords+={kw}")
        args.append(f"-XMP:Subject+={kw}")
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

def process_single(
    client: VisionClient | None,
    path: Path,
    dry_run: bool,
    force: bool,
    clear_all: bool = False,
    enable_ai: bool = True,
    enable_geo: bool = True,
    enable_exif: bool = True,
    enable_ocr: bool = True,
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

    # For videos, extract a frame for AI and OCR
    frame_path = None
    if video:
        all_tags.append("video")
        frame_path = extract_video_frame(path)
        if frame_path is None:
            log.warning("Could not extract frame from %s, running EXIF/GPS only", path.name)

    # The image to use for AI and OCR (original file or extracted frame)
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

    if enable_ai and client is not None and visual_path:
        parsed = client.tag_image(visual_path)
        if parsed is not None:
            t = flatten_ai_tags(parsed)
            log.debug("  AI:   %s", t)
            all_tags.extend(t)

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

def watch_directory(client, target, recursive, dry_run,
                    enable_ai, enable_geo, enable_exif, enable_ocr):
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
                process_single(client, img, dry_run, force=False,
                               enable_ai=enable_ai, enable_geo=enable_geo,
                               enable_exif=enable_exif, enable_ocr=enable_ocr)
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
    parser.add_argument("path", type=Path, nargs="?", help="Image file or directory")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Preview tags without writing")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Re-tag (clears old autogenerated tags first)")
    parser.add_argument("--clear-all", action="store_true",
                        help="Wipe ALL keywords before re-tagging (nuclear option)")
    parser.add_argument("-w", "--watch", action="store_true",
                        help="Watch directory for new images")
    parser.add_argument("-m", "--model", default=None,
                        help="Model name (default: $AI_MODEL or gemma4)")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--no-geo", action="store_true")
    parser.add_argument("--no-exif", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")


def build_tag_parser(subparsers) -> argparse.ArgumentParser:
    """Register the 'tag' subcommand on the given subparsers object."""
    sub = subparsers.add_parser(
        "tag",
        help="Auto-tag photos using AI vision, GPS geocoding, OCR, and EXIF metadata.",
    )
    _add_tag_args(sub)
    sub.set_defaults(func=run_tag)
    return sub


def run_tag(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_url = args.base_url or os.environ.get("AI_BASE_URL", DEFAULT_BASE_URL)
    api_key = args.api_key or os.environ.get("AI_API_KEY", "")
    model = args.model or os.environ.get("AI_MODEL", DEFAULT_MODEL)

    enable_ai = not args.no_ai
    enable_geo = not args.no_geo
    enable_exif = not args.no_exif
    enable_ocr = not args.no_ocr

    client = None
    if enable_ai:
        client = VisionClient(base_url=base_url, api_key=api_key, model=model)

    if args.list_models:
        c = client or VisionClient(base_url=base_url, api_key=api_key, model=model)
        try:
            for m in sorted(c.list_models()):
                print(f"  {m}")
        except Exception as e:
            log.error("Failed to list models: %s", e)
            sys.exit(1)
        return

    if args.path is None:
        log.error("path is required (unless using --list-models)")
        sys.exit(1)

    sources = []
    if enable_exif: sources.append("EXIF")
    if enable_geo:  sources.append("GPS")
    if enable_ocr:  sources.append("OCR")
    if enable_ai:   sources.append(f"AI ({model})")
    log.info("Tag sources: %s", " + ".join(sources))

    if args.watch:
        if not args.path.is_dir():
            log.error("--watch requires a directory")
            sys.exit(1)
        watch_directory(client, args.path, args.recursive, args.dry_run,
                        enable_ai, enable_geo, enable_exif, enable_ocr)
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
            result = process_single(client, img, args.dry_run, args.force,
                                    clear_all=args.clear_all,
                                    enable_ai=enable_ai, enable_geo=enable_geo,
                                    enable_exif=enable_exif, enable_ocr=enable_ocr)
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
        description="Auto-tag photos using AI vision, GPS geocoding, OCR, and EXIF metadata.",
    )
    _add_tag_args(parser)
    args = parser.parse_args()
    run_tag(args)


if __name__ == "__main__":
    main()

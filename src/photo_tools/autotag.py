"""
autotag.py — Auto-tag photos using RAM++, CLIP landmark lookup, PaddleOCR,
GPS reverse geocoding, and EXIF metadata, then write IPTC/XMP keywords
via ExifTool (with DigiKam hierarchical tag support).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from photo_tools.config import get_config
from photo_tools.constants import (
    AI_TAG_MARKER,
    IMAGE_EXTENSIONS,
    OCR_VOWELS,
    OCR_WORD_PATTERN,
    SCREENSHOT_RESOLUTIONS,
    VALID_ONSETS,
)
from photo_tools.helpers import (
    add_tags,
    clear_all_keywords,
    deduplicate,
    detect_real_type,
    extract_video_frame,
    find_images,
    get_existing_keywords,
    is_our_tag,
    is_video,
    prepare_image,
    read_exif,
    remove_tags,
    write_embedding,
)

log = logging.getLogger("autotag")


# ---------------------------------------------------------------------------
# GPS geocoding state (module-level, rate-limited)
# ---------------------------------------------------------------------------

_last_nominatim_call = 0.0
_geocode_cache: list[tuple[float, float, dict]] = []


# ---------------------------------------------------------------------------
# OCR word validation
# ---------------------------------------------------------------------------

def _is_plausible_word(word: str) -> bool:
    """Reject OCR fragments that aren't real words."""
    w = word.lower()
    if not any(c in OCR_VOWELS for c in w):
        return False
    if len(w) >= 2 and w[0] not in OCR_VOWELS and w[1] not in OCR_VOWELS:
        if w[:3] not in VALID_ONSETS and w[:2] not in VALID_ONSETS:
            return False
    digits = sum(1 for c in w if c.isdigit())
    if digits > len(w) / 2:
        return False
    return True


# ---------------------------------------------------------------------------
# EXIF-derived tags
# ---------------------------------------------------------------------------

def tags_from_exif(exif: dict) -> list[str]:
    tags = []

    date_str = exif.get("DateTimeOriginal") or exif.get("CreateDate")
    if date_str and isinstance(date_str, str) and len(date_str) >= 10:
        try:
            import datetime as dt_mod
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
                    d = dt_mod.date(year, month, day)
                    day_names = ["monday", "tuesday", "wednesday", "thursday",
                                 "friday", "saturday", "sunday"]
                    tags.append(f"day/{day_names[d.weekday()]}")
                    tags.append("weekend" if d.weekday() >= 5 else "weekday")
                except ValueError:
                    pass
        except (ValueError, IndexError):
            pass

    flash = exif.get("Flash")
    if flash is not None:
        try:
            if int(flash) & 1:
                tags.append("flash/fired")
        except (ValueError, TypeError):
            if "fired" in str(flash).lower() and "not" not in str(flash).lower():
                tags.append("flash/fired")

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


def _parse_exif_datetime(exif: dict) -> datetime | None:
    date_str = exif.get("DateTimeOriginal") or exif.get("CreateDate")
    if not date_str or not isinstance(date_str, str) or len(date_str) < 10:
        return None
    try:
        clean = date_str.replace("-", ":").replace("T", " ")
        parts = clean.split(":")
        if len(parts) >= 5:
            year, month = int(parts[0]), int(parts[1])
            rest = parts[2].split()
            day = int(rest[0])
            hour = int(parts[3])
            minute = int(parts[4].split(".")[0].split("+")[0].split("-")[0])
            return datetime(year, month, day, hour, minute)
    except (ValueError, IndexError):
        pass
    return None


def build_gps_timeline(paths: list[Path]) -> dict[Path, tuple[float, float]]:
    """Pre-scan images and infer GPS for those missing it from nearby images."""
    if not paths:
        return {}

    cfg = get_config()
    batch_size = cfg.exiftool.batch_size
    max_gap = cfg.gps.max_gap_seconds

    gps_data: list[tuple[Path, datetime | None, tuple[float, float] | None]] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        try:
            proc = subprocess.run(
                ["exiftool", "-j", "-n",
                 "-GPS:GPSLatitude", "-GPS:GPSLongitude",
                 "-GPS:GPSLatitudeRef", "-GPS:GPSLongitudeRef",
                 "-EXIF:DateTimeOriginal", "-EXIF:CreateDate"]
                + [str(p) for p in batch],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                continue
            meta_list = json.loads(proc.stdout)
        except Exception:
            continue

        str_to_path = {str(p): p for p in batch}
        for meta in meta_list:
            source = meta.get("SourceFile", "")
            p = str_to_path.get(source, Path(source))
            coords = get_gps_coords(meta)
            dt = _parse_exif_datetime(meta)
            gps_data.append((p, dt, coords))

    result: dict[Path, tuple[float, float]] = {}
    for p, _, coords in gps_data:
        if coords:
            result[p] = coords

    timed_gps = [(dt, coords) for _, dt, coords in gps_data
                 if dt is not None and coords is not None]
    if not timed_gps:
        return result

    timed_gps.sort(key=lambda x: x[0])
    gps_times = [t for t, _ in timed_gps]
    gps_coords = [c for _, c in timed_gps]

    from bisect import bisect_left

    for p, dt, coords in gps_data:
        if coords is not None or dt is None:
            continue
        idx = bisect_left(gps_times, dt)
        best_dist = float("inf")
        best_coords = None
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(gps_times):
                gap = abs((dt - gps_times[candidate_idx]).total_seconds())
                if gap < best_dist:
                    best_dist = gap
                    best_coords = gps_coords[candidate_idx]
        if best_coords and best_dist <= max_gap:
            result[p] = best_coords
            log.debug("Inferred GPS for %s from nearby image (%.0fs away)",
                      p.name, best_dist)

    inferred = sum(1 for p, _, c in gps_data if c is None and p in result)
    if inferred:
        log.info("Inferred GPS for %d image(s) from nearby timestamps", inferred)

    return result


def reverse_geocode(lat: float, lon: float) -> dict:
    from photo_tools.landmarks import _haversine_km
    cfg = get_config()

    for clat, clon, addr in _geocode_cache:
        if _haversine_km(lat, lon, clat, clon) < cfg.gps.geocode_cache_radius_km:
            return addr

    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2",
                    "addressdetails": 1, "zoom": cfg.gps.nominatim_zoom,
                    "accept-language": "en"},
            headers={"User-Agent": "autotag-photo-tagger/1.0"},
            timeout=10,
        )
        _last_nominatim_call = time.time()
        resp.raise_for_status()
        addr = resp.json().get("address", {})
    except Exception as e:
        log.warning("Geocoding failed for (%.4f, %.4f): %s", lat, lon, e)
        _last_nominatim_call = time.time()
        return {}
    _geocode_cache.append((lat, lon, addr))
    return addr


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
# PaddleOCR
# ---------------------------------------------------------------------------

_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        import warnings
        warnings.filterwarnings("ignore", module="paddle")
        for name in ("ppocr", "paddle", "paddlex"):
            logging.getLogger(name).setLevel(logging.ERROR)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        )
    return _ocr_engine


def _get_image_dimensions(path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image as PILImage, ImageOps
        with PILImage.open(str(path)) as img:
            ImageOps.exif_transpose(img, in_place=True)
            return img.size
    except Exception:
        return None, None


def tags_from_ocr(path: Path) -> tuple[list[str], list[dict]]:
    """Run PaddleOCR and return (text_tags, regions)."""
    cfg = get_config()
    prepared = prepare_image(path, cfg.ocr.max_pixels)
    ocr_path = prepared or path

    try:
        img_w, img_h = _get_image_dimensions(ocr_path)
        if img_w is None:
            img_w, img_h = 1, 1

        ocr = _get_ocr_engine()
        results = ocr.predict(str(ocr_path))
    except Exception as e:
        log.warning("PaddleOCR error on %s: %s", path.name, e)
        return [], []
    finally:
        if prepared:
            try:
                os.unlink(prepared)
            except OSError:
                pass

    seen = set()
    tags = []
    regions = []

    for page in results:
        polys = page.get("dt_polys", [])
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])

        for poly, text, score in zip(polys, texts, scores):
            if score < cfg.ocr.min_confidence:
                continue

            raw_words = text.split()
            good_words = [
                w for w in raw_words
                if OCR_WORD_PATTERN.match(w) and _is_plausible_word(w)
            ]
            if not good_words:
                continue

            if len(good_words) == 1 and score < cfg.ocr.high_confidence:
                continue

            phrase = " ".join(good_words).lower().strip()
            if len(phrase) < cfg.ocr.min_phrase_length:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            tags.append(f"text/{phrase}")

            poly_arr = np.array(poly)
            x_min, y_min = poly_arr.min(axis=0)
            x_max, y_max = poly_arr.max(axis=0)
            regions.append({
                "text": phrase,
                "score": float(score),
                "x": float(x_min / img_w),
                "y": float(y_min / img_h),
                "w": float((x_max - x_min) / img_w),
                "h": float((y_max - y_min) / img_h),
            })

            if len(tags) >= cfg.ocr.max_tags:
                break

    if tags:
        log.debug("  OCR found %d text fragments in %s", len(tags), path.name)

    return tags, regions


# ---------------------------------------------------------------------------
# OCR region writing (IPTC ImageRegion + MWG RegionInfo)
# ---------------------------------------------------------------------------

def _read_existing_regions(path: Path) -> tuple[list[dict], list[dict], dict | None]:
    """Read existing MWG and IPTC regions, returning non-OCR regions to preserve."""
    try:
        result = subprocess.run(
            ["exiftool", "-j", "-struct",
             "-XMP-mwg-rs:RegionInfo", "-XMP-iptcExt:ImageRegion", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return [], [], None
        meta = json.loads(result.stdout)
        if not meta:
            return [], [], None
        data = meta[0]
    except Exception:
        return [], [], None

    region_info = data.get("RegionInfo") or {}
    existing_mwg = [r for r in (region_info.get("RegionList") or [])
                    if r.get("Type") != "BarCode"]
    applied_dims = region_info.get("AppliedToDimensions")

    existing_iptc = []
    for r in data.get("ImageRegion") or []:
        roles = r.get("RRole") or []
        is_ocr = any("annotatedText" in (ident or "")
                      for role in roles for ident in (role.get("Identifier") or []))
        if not is_ocr:
            existing_iptc.append(r)

    return existing_mwg, existing_iptc, applied_dims


def write_text_regions(path: Path, regions: list[dict], dry_run: bool = False) -> bool:
    """Write OCR text regions as IPTC ImageRegion + MWG Region metadata."""
    if not regions:
        return True
    if dry_run:
        log.info("[DRY RUN] Would write %d text regions to %s", len(regions), path.name)
        return True

    existing_mwg, existing_iptc, applied_dims = _read_existing_regions(path)

    iptc_regions = list(existing_iptc)
    mwg_regions = list(existing_mwg)
    for r in regions:
        iptc_regions.append({
            "Name": r["text"],
            "RRole": [{"Identifier": ["http://cv.iptc.org/newscodes/imageregionrole/annotatedText"],
                        "Name": "annotated text"}],
            "RegionBoundary": {
                "RbShape": "rectangle", "RbUnit": "relative",
                "RbX": round(r["x"], 5), "RbY": round(r["y"], 5),
                "RbW": round(r["w"], 5), "RbH": round(r["h"], 5),
            },
        })
        mwg_regions.append({
            "Name": r["text"],
            "Type": "BarCode",
            "Description": "OCR detected text",
            "Area": {
                "X": round(r["x"] + r["w"] / 2, 5),
                "Y": round(r["y"] + r["h"] / 2, 5),
                "W": round(r["w"], 5),
                "H": round(r["h"], 5),
                "Unit": "normalized",
            },
        })

    mwg_info = {"RegionList": mwg_regions}
    if applied_dims:
        mwg_info["AppliedToDimensions"] = applied_dims

    meta = {
        "SourceFile": str(path),
        "XMP-iptcExt:ImageRegion": iptc_regions,
        "XMP-mwg-rs:RegionInfo": mwg_info,
    }

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump([meta], tmp)
        tmp.close()
        result = subprocess.run(
            ["exiftool", "-overwrite_original", "-struct", f"-json={tmp.name}", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            log.error("exiftool region write failed for %s: %s", path.name, result.stderr.strip())
            return False
        log.info("Wrote %d text regions to %s", len(regions), path.name)
        return True
    except Exception as e:
        log.error("Failed to write text regions to %s: %s", path.name, e)
        return False
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Lazy-initialized ML singletons
# ---------------------------------------------------------------------------

_ram_tagger = None
_clip_embedder = None
_landmark_index = None


def _get_ram_tagger():
    global _ram_tagger
    if _ram_tagger is None:
        from photo_tools.ram_tagger import RAMTagger
        _ram_tagger = RAMTagger()
    return _ram_tagger


def _get_clip_embedder(clip_model: str = None, clip_pretrained: str = None):
    global _clip_embedder
    if _clip_embedder is None:
        from photo_tools.clip_tagger import CLIPEmbedder
        cfg = get_config()
        _clip_embedder = CLIPEmbedder(
            model_name=clip_model or cfg.clip.model,
            pretrained=clip_pretrained or cfg.clip.pretrained,
        )
    return _clip_embedder


def _get_landmark_index(landmarks_path: Path = None):
    global _landmark_index
    if _landmark_index is None:
        from photo_tools.landmarks import LandmarkIndex
        cfg = get_config()
        lm_path = landmarks_path or Path(cfg.landmarks.default_path).expanduser()
        if not lm_path.exists():
            log.debug("No landmarks database at %s, skipping landmark lookup", lm_path)
            return None
        _landmark_index = LandmarkIndex(lm_path)
    return _landmark_index


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_single(
    path: Path,
    dry_run: bool,
    force: bool,
    clear_all: bool = False,
    enable_ram: bool = True,
    enable_landmarks: bool = True,
    enable_ocr: bool = True,
    clip_model: str = None,
    clip_pretrained: str = None,
    landmarks_path: Path = None,
    gps_fallback: tuple[float, float] | None = None,
) -> bool:
    cfg = get_config()

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

    if clear_all and existing:
        clear_all_keywords(path, dry_run)
        existing = set()
    elif force and existing:
        to_remove = [tag for tag in existing if is_our_tag(tag)]
        if to_remove:
            remove_tags([path], to_remove, also_remove_leaves=True, dry_run=dry_run)
        our_tags = {t for t in existing if is_our_tag(t)}
        our_leaves = {t.rsplit("/", 1)[-1] for t in our_tags if "/" in t}
        existing = existing - our_tags - our_leaves

    log.info("Processing %s ...", path.name)
    all_tags = []

    frame_path = None
    if video:
        all_tags.append("video")
        frame_path = extract_video_frame(path)
        if frame_path is None:
            log.warning("Could not extract frame from %s, running EXIF/GPS only", path.name)

    visual_path = frame_path if video else path

    t = tags_from_exif(exif)
    if t:
        log.debug("  EXIF: %s", t)
        all_tags.extend(t)

    t = tags_from_gps(exif)
    if t:
        log.debug("  Geo:  %s", t)
        all_tags.extend(t)

    ocr_regions = []
    if enable_ocr and visual_path:
        t, ocr_regions = tags_from_ocr(visual_path)
        if t:
            log.debug("  OCR:  %s", t)
            all_tags.extend(t)

    embedding = None
    need_visual = (enable_ram or enable_landmarks) and visual_path
    prepared = prepare_image(visual_path, cfg.clip.max_pixels) if need_visual else None
    visual_input = prepared or visual_path

    if enable_ram and visual_path:
        try:
            tagger = _get_ram_tagger()
            ram_tags = tagger.tag_image(visual_input)
            log.debug("  RAM++: %s", ram_tags)
            all_tags.extend(ram_tags)
        except Exception as e:
            log.warning("RAM++ tagging failed for %s: %s", path.name, e)

    if (enable_ram or enable_landmarks) and visual_path:
        try:
            embedder = _get_clip_embedder(clip_model, clip_pretrained)
            embedding = embedder.embed_image(visual_input)
        except Exception as e:
            log.warning("CLIP embedding failed for %s: %s", path.name, e)

    if prepared:
        try:
            os.unlink(prepared)
        except OSError:
            pass

    if enable_landmarks and embedding is not None:
        coords = get_gps_coords(exif) or gps_fallback
        if coords is not None:
            lat, lon = coords
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

    if embedding is not None and not dry_run:
        model_id = _get_clip_embedder(clip_model, clip_pretrained).model_id
        write_embedding(path, embedding, model_id, dry_run)

    if frame_path:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    all_tags = deduplicate(all_tags)
    new_tags = [t for t in all_tags if t.lower() not in existing]

    if AI_TAG_MARKER not in existing:
        new_tags.append(AI_TAG_MARKER)

    if not new_tags:
        log.info("No new keywords for %s", path.name)
        return True

    ok = add_tags(path, new_tags, dry_run)
    if ok and ocr_regions:
        write_text_regions(path, ocr_regions, dry_run)
    return ok


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_directory(target, recursive, dry_run,
                    enable_ram, enable_landmarks, enable_ocr,
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
                               enable_ram=enable_ram,
                               enable_landmarks=enable_landmarks,
                               enable_ocr=enable_ocr,
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

    pipelines = parser.add_argument_group(
        "pipeline selection",
        "Choose which tagging pipelines to run. "
        "When none are specified, all pipelines run. "
        "EXIF metadata and GPS geocoding always run."
    )
    pipelines.add_argument("--ram", action="store_true",
                           help="Run RAM++ image content tagging")
    pipelines.add_argument("--landmarks", action="store_true",
                           help="Run landmark lookup (CLIP embedding + GPS)")
    pipelines.add_argument("--ocr", action="store_true",
                           help="Run OCR text detection")

    parser.add_argument("--clip-model", default=None,
                        help="CLIP model name (default: from config)")
    parser.add_argument("--clip-pretrained", default=None,
                        help="CLIP pretrained weights (default: from config)")
    parser.add_argument("--landmarks-db", type=Path, default=None,
                        dest="landmarks_db",
                        help="Path to landmarks.json")


def build_tag_parser(subparsers) -> argparse.ArgumentParser:
    sub = subparsers.add_parser(
        "tag",
        help="Auto-tag photos using RAM++, landmark lookup, OCR, GPS geocoding, and EXIF metadata.",
    )
    _add_tag_args(sub)
    sub.set_defaults(func=run_tag)
    return sub


def run_tag(args) -> None:
    cfg = get_config()

    any_selected = args.ram or args.landmarks or args.ocr
    enable_ram = args.ram or not any_selected
    enable_landmarks = args.landmarks or not any_selected
    enable_ocr = args.ocr or not any_selected

    sources = ["EXIF", "GPS"]
    if enable_ram:
        sources.append("RAM++")
    if enable_ocr:
        sources.append("OCR")
    lm_path = args.landmarks_db or Path(cfg.landmarks.default_path).expanduser()
    if enable_landmarks and lm_path.exists():
        sources.append("Landmarks")
    log.info("Tag sources: %s", " + ".join(sources))

    if args.watch:
        if not args.path.is_dir():
            log.error("--watch requires a directory")
            sys.exit(1)
        watch_directory(args.path, args.recursive, args.dry_run,
                        enable_ram, enable_landmarks, enable_ocr,
                        clip_model=args.clip_model,
                        clip_pretrained=args.clip_pretrained,
                        landmarks_path=args.landmarks_db)
        return

    images = find_images(args.path, args.recursive)
    if not images:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Found %d image(s) to process", len(images))

    gps_timeline = build_gps_timeline(images) if (enable_ram or enable_landmarks) else {}

    success = failed = skipped = 0

    for i, img in enumerate(images, 1):
        log.info("[%d/%d] %s", i, len(images), img)
        try:
            result = process_single(img, args.dry_run, args.force,
                                    clear_all=args.clear_all,
                                    enable_ram=enable_ram,
                                    enable_landmarks=enable_landmarks,
                                    enable_ocr=enable_ocr,
                                    clip_model=args.clip_model,
                                    clip_pretrained=args.clip_pretrained,
                                    landmarks_path=args.landmarks_db,
                                    gps_fallback=gps_timeline.get(img))
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            log.error("Error processing %s: %s", img.name, e)
            failed += 1

    log.info("Done. %d tagged, %d skipped, %d failed.", success, skipped, failed)

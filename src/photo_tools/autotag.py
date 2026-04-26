"""
autotag.py — Auto-tag photos using RAM++, CLIP landmark lookup, PaddleOCR,
GPS reverse geocoding, and EXIF metadata, then write IPTC/XMP keywords
via ExifTool (with DigiKam hierarchical tag support).
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from photo_tools.config import get_config
from photo_tools.constants import (
    IMAGE_EXTENSIONS,
    OCR_VOWELS,
    OCR_WORD_PATTERN,
    TAGGER_VERSION,
    VALID_ONSETS,
)
from photo_tools.helpers import (
    _run_exiftool,
    _run_exiftool_json,
    add_tags,
    clear_all_keywords,
    deduplicate,
    detect_real_type,
    extract_video_frame,
    find_images,
    get_existing_keywords,
    get_tagger_version,
    is_live_photo_motion,
    is_our_tag,
    is_video,
    leaf_of,
    prepare_image,
    read_exif,
    read_tagger_versions_batch,
    remove_tags,
    write_embedding,
)

try:
    from titlecase import titlecase as _titlecase
except ImportError:  # pragma: no cover - dependency declared in pyproject
    def _titlecase(s: str) -> str:
        return s.title()


def title(s: str) -> str:
    """Titlecase a path segment (geocoder value, tag leaf)."""
    return _titlecase(s.strip())

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
        meta_list = _run_exiftool_json(
            ["-n",
             "-GPS:GPSLatitude", "-GPS:GPSLongitude",
             "-GPS:GPSLatitudeRef", "-GPS:GPSLongitudeRef",
             "-EXIF:DateTimeOriginal", "-EXIF:CreateDate"]
            + [str(p) for p in batch],
            with_config=False, timeout=120,
        )
        if not meta_list:
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


def tags_from_gps(exif: dict) -> tuple[list[str], str | None]:
    """Return (places_tags, country_code).

    `places_tags` is a list with at most one entry — the single nested
    `Places/<Country>[/<Region>[/<City>[/<Neighborhood>]]]` path with missing
    levels collapsed. `country_code` is the ISO 3166-1 alpha-2 code (uppercase)
    or None; it is written separately to photo-tools:CountryCode.
    """
    coords = get_gps_coords(exif)
    if coords is None:
        return [], None
    lat, lon = coords
    log.debug("GPS: %.5f, %.5f", lat, lon)
    address = reverse_geocode(lat, lon)
    if not address:
        return [], None

    segments = []
    country = address.get("country")
    if country:
        segments.append(title(country))

    region = (address.get("state") or address.get("region")
              or address.get("province") or address.get("county"))
    if region:
        segments.append(title(region))

    city = (address.get("city") or address.get("town")
            or address.get("village") or address.get("municipality"))
    if city:
        segments.append(title(city))

    neighborhood = (address.get("suburb") or address.get("neighbourhood")
                    or address.get("quarter") or address.get("district"))
    if neighborhood:
        segments.append(title(neighborhood))

    places = ["Places/" + "/".join(segments)] if segments else []

    cc = address.get("country_code")
    country_code = cc.upper() if cc else None

    return places, country_code


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
        from photo_tools.helpers import open_and_rotate
        img = open_and_rotate(path)
        try:
            return img.size
        finally:
            img.close()
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

            phrase = " ".join(good_words).strip()
            if len(phrase) < cfg.ocr.min_phrase_length:
                continue
            phrase_key = phrase.lower()
            if phrase_key in seen:
                continue
            seen.add(phrase_key)
            tags.append(f"Text/{title(phrase)}")

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
        meta = _run_exiftool_json(
            ["-struct", "-XMP-mwg-rs:RegionInfo", "-XMP-iptcExt:ImageRegion", str(path)],
            with_config=False, timeout=30,
        )
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
        result = _run_exiftool(
            ["-overwrite_original", "-struct", f"-json={tmp.name}", str(path)],
            with_config=False, timeout=60,
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

    if is_live_photo_motion(path):
        log.info("Skipping %s (Live Photo motion companion)", path.name)
        return False

    video = is_video(path)
    if not video and path.suffix.lower() in IMAGE_EXTENSIONS:
        real_type = detect_real_type(path)
        if real_type == "video":
            log.info("Skipping %s (extension is %s but file is actually a video/Live Photo)",
                     path.name, path.suffix)
            return False

    exif = read_exif(path)
    existing = get_existing_keywords(exif)
    stored_version = get_tagger_version(exif)

    if (stored_version == TAGGER_VERSION and not force and not clear_all):
        log.info("Skipping %s (already tagged with %s, use --force to re-tag)",
                 path.name, stored_version)
        return False

    if clear_all and existing:
        clear_all_keywords(path, dry_run)
        existing = set()
    elif force and existing:
        to_remove = [tag for tag in existing if is_our_tag(tag)]
        if to_remove:
            remove_tags([path], to_remove, dry_run=dry_run)
        our_leaves = {leaf_of(t).lower() for t in existing if is_our_tag(t)}
        existing = existing - our_leaves

    log.info("Processing %s ...", path.name)
    all_tags: list[str] = []

    frame_path = None
    if video:
        frame_path = extract_video_frame(path)
        if frame_path is None:
            log.warning("Could not extract frame from %s, running EXIF/GPS only", path.name)

    visual_path = frame_path if video else path

    places, country_code = tags_from_gps(exif)
    if places:
        log.debug("  Geo:  %s", places)
        all_tags.extend(places)

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
            ram_tags, scored_ram_tags = tagger.tag_image(visual_input)
            if scored_ram_tags:
                log.info("  RAM++ top 5: %s",
                         ", ".join(f"{t} ({s:.3f}/thr {thr:.3f})"
                                   for t, s, thr in scored_ram_tags[:5]))
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
                    landmark, lm_top, lm_thr = lm_index.lookup(
                        embedding, lat=lat, lon=lon)
                    if lm_top:
                        log.info("  Landmarks top 3 (thr %.3f): %s", lm_thr,
                                 ", ".join(f"{n} ({s:.3f})" for n, s in lm_top[:3]))
                    if landmark:
                        tag = f"Landmarks/{title(landmark)}"
                        log.debug("  Landmark: %s", tag)
                        all_tags.append(tag)
            except Exception as e:
                log.debug("Landmark lookup failed: %s", e)

    if frame_path:
        try:
            os.unlink(frame_path)
        except OSError:
            pass

    all_tags = deduplicate(all_tags)
    new_tags = [t for t in all_tags if leaf_of(t).lower() not in existing]

    namespace_fields: dict[str, str] = {}
    if country_code:
        namespace_fields["CountryCode"] = country_code

    ok = True
    if new_tags or namespace_fields or stored_version != TAGGER_VERSION:
        ok = add_tags(path, new_tags, dry_run, namespace_fields=namespace_fields)
        if ok and ocr_regions:
            write_text_regions(path, ocr_regions, dry_run)
    else:
        log.info("No new keywords for %s", path.name)

    # Write embedding last — after all other exiftool calls that rewrite XMP.
    if embedding is not None and not dry_run:
        model_id = _get_clip_embedder(clip_model, clip_pretrained).model_id
        write_embedding(path, embedding, model_id, dry_run)

    return ok


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_directory(target, dry_run,
                    enable_ram, enable_landmarks, enable_ocr,
                    clip_model=None, clip_pretrained=None, landmarks_path=None):
    log.info("Watching %s for new images (Ctrl+C to stop) ...", target)
    seen: set[Path] = set()

    initial_images = find_images(target)
    initial_versions = read_tagger_versions_batch(initial_images)
    for img in initial_images:
        if initial_versions.get(img) == TAGGER_VERSION:
            seen.add(img.resolve())
    log.info("Found %d already-tagged images, skipping those.", len(seen))

    while True:
        try:
            all_images = find_images(target)
            new_images = [img for img in all_images if img.resolve() not in seen]
            if new_images:
                new_versions = read_tagger_versions_batch(new_images)
                for img in new_images:
                    resolved = img.resolve()
                    if new_versions.get(img) == TAGGER_VERSION:
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
    parser.add_argument("path", type=Path, nargs="+",
                        help="Image file(s) or directory (one or more)")
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
        if len(args.path) != 1 or not args.path[0].is_dir():
            log.error("--watch requires a single directory")
            sys.exit(1)
        watch_directory(args.path[0], args.dry_run,
                        enable_ram, enable_landmarks, enable_ocr,
                        clip_model=args.clip_model,
                        clip_pretrained=args.clip_pretrained,
                        landmarks_path=args.landmarks_db)
        return

    images: list[Path] = []
    seen: set[Path] = set()
    for p in args.path:
        for img in find_images(p):
            resolved = img.resolve()
            if resolved not in seen:
                seen.add(resolved)
                images.append(img)
    if not images:
        log.error("No supported images found at %s",
                  ", ".join(str(p) for p in args.path))
        sys.exit(1)

    skipped = 0
    motion_companions = [p for p in images if is_live_photo_motion(p)]
    if motion_companions:
        log.info("Skipping %d Live Photo motion companion(s)", len(motion_companions))
        for p in motion_companions:
            log.debug("Skipping %s (Live Photo motion companion)", p.name)
        images = [p for p in images if not is_live_photo_motion(p)]
        skipped += len(motion_companions)

    if not images:
        log.info("Done. 0 tagged, %d skipped, 0 failed.", skipped)
        return

    log.info("Found %d image(s) to process", len(images))

    if not args.force and not args.clear_all:
        versions = read_tagger_versions_batch(images)
        already_tagged = {p for p, v in versions.items() if v == TAGGER_VERSION}
        if already_tagged:
            log.info("Skipping %d/%d already tagged (use --force to re-tag)",
                     len(already_tagged), len(images))
            for p in already_tagged:
                log.debug("Skipping %s (already tagged with %s)",
                          p.name, TAGGER_VERSION)
            images = [p for p in images if p not in already_tagged]
            skipped += len(already_tagged)

    if not images:
        log.info("Done. 0 tagged, %d skipped, 0 failed.", skipped)
        return

    gps_timeline = build_gps_timeline(images) if (enable_ram or enable_landmarks) else {}

    success = failed = 0

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

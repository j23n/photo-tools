"""
autotag.py — Auto-tag photos using RAM++, CLIP landmark lookup, PaddleOCR,
GPS reverse geocoding, and EXIF metadata, then write IPTC/XMP keywords
via ExifTool (with DigiKam hierarchical tag support).
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from photo_tools import tui
from photo_tools.config import get_config
from photo_tools.constants import (
    IMAGE_EXTENSIONS,
    OCR_VOWELS,
    OCR_WORD_PATTERN,
    TAGGER_VERSION,
    VALID_ONSETS,
)
from photo_tools.helpers import (
    _expand_paths_with_sidecars,
    _group_metas_by_path,
    _run_exiftool_json,
    clear_all_keywords,
    deduplicate,
    detect_real_type,
    existing_non_ocr_regions,
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
    write_metadata,
)
from photo_tools.logging_setup import (
    PhotoSummary,
    get_counter,
    get_logger,
    log_run_summary,
    timed_step,
)

try:
    from titlecase import titlecase as _titlecase
except ImportError:  # pragma: no cover - dependency declared in pyproject
    def _titlecase(s: str) -> str:
        return s.title()


def title(s: str) -> str:
    """Titlecase a path segment (geocoder value, tag leaf)."""
    return _titlecase(s.strip())


# Subsystem loggers. The orchestrator owns ``tagging``; pipeline-specific
# code (geocoding, gps, ocr) gets its own child so users can silence or
# amplify each from --log.
log = get_logger("tagging")
log_gps = get_logger("gps")
log_geo = get_logger("geocoding")
log_ocr = get_logger("ocr")


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
    """Parse an EXIF timestamp ('2026:05:07 12:34:56' or ISO) to a datetime.

    Subsecond and timezone suffixes are ignored. Returns None on malformed
    input.
    """
    date_str = exif.get("DateTimeOriginal") or exif.get("CreateDate")
    if not date_str or not isinstance(date_str, str) or len(date_str) < 10:
        return None
    try:
        # Normalize ISO ('2026-05-07T12:34:56') to EXIF shape so a single
        # `:` split below can extract every field.
        clean = date_str.replace("-", ":").replace("T", " ")
        parts = clean.split(":")
        # parts = ['YYYY', 'MM', 'DD HH', 'MM', 'SS[.fff][±HH][:MM]']
        if len(parts) >= 4:
            year, month = int(parts[0]), int(parts[1])
            day_hour = parts[2].split()
            if len(day_hour) < 2:
                return None
            day = int(day_hour[0])
            hour = int(day_hour[1])
            minute = int(parts[3].split(".")[0].split("+")[0].split("-")[0])
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
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(
            ["-n",
             "-GPS:GPSLatitude", "-GPS:GPSLongitude",
             "-GPS:GPSLatitudeRef", "-GPS:GPSLongitudeRef",
             "-EXIF:DateTimeOriginal", "-EXIF:CreateDate"]
            + files,
            with_config=False, timeout=120,
        )
        if not meta_list:
            continue

        for p, meta in _group_metas_by_path(meta_list, str_to_path).items():
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
            log_gps.debug("Inferred GPS for %s from nearby image (%.0fs away)",
                          p.name, best_dist)

    inferred = sum(1 for p, _, c in gps_data if c is None and p in result)
    if inferred:
        log_gps.info("Inferred GPS for %d image(s) from nearby timestamps", inferred)

    return result


def reverse_geocode(lat: float, lon: float) -> dict:
    from photo_tools.landmarks import _haversine_km
    cfg = get_config()
    counter = get_counter("geocoding")

    for clat, clon, addr in _geocode_cache:
        if _haversine_km(lat, lon, clat, clon) < cfg.gps.geocode_cache_radius_km:
            counter.add("cache_hits")
            return addr

    counter.add("queries")
    global _last_nominatim_call
    elapsed = time.time() - _last_nominatim_call
    if elapsed < 1.1:
        wait = 1.1 - elapsed
        counter.add("rate_limit_waits")
        counter.add("rate_limit_seconds", wait)
        log_geo.debug("rate-limit cooldown %.1fs", wait)
        time.sleep(wait)
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
        log_geo.warning("Nominatim failed for (%.4f, %.4f): %s", lat, lon, e)
        counter.add("query_errors")
        _last_nominatim_call = time.time()
        return {}
    _geocode_cache.append((lat, lon, addr))
    return addr


def tags_from_gps(exif: dict) -> tuple[list[str], str | None, dict[str, str]]:
    """Return (places_tags, country_code, location_parts).

    `places_tags` is a list with at most one entry — the single nested
    `Places/<Country>[/<Region>[/<City>[/<Neighborhood>]]]` path with missing
    levels collapsed. `country_code` is the ISO 3166-1 alpha-2 code (uppercase)
    or None; it is written separately to photo-tools:CountryCode plus the
    standard IPTC country-code fields.

    `location_parts` is a dict with optional keys `Country`, `State`, `City`,
    `Sublocation`, `CountryCode` — the same components as the Places path,
    intended for the IPTC-standard structured location fields (see
    docs/xmp-schema.md §1.5).
    """
    coords = get_gps_coords(exif)
    if coords is None:
        return [], None, {}
    lat, lon = coords
    log_geo.debug("lookup (%.5f, %.5f)", lat, lon)
    address = reverse_geocode(lat, lon)
    if not address:
        return [], None, {}

    segments = []
    location_parts: dict[str, str] = {}

    country = address.get("country")
    if country:
        c = title(country)
        segments.append(c)
        location_parts["Country"] = c

    region = (address.get("state") or address.get("region")
              or address.get("province") or address.get("county"))
    if region:
        r = title(region)
        segments.append(r)
        location_parts["State"] = r

    city = (address.get("city") or address.get("town")
            or address.get("village") or address.get("municipality"))
    if city:
        ct = title(city)
        segments.append(ct)
        location_parts["City"] = ct

    neighborhood = (address.get("suburb") or address.get("neighbourhood")
                    or address.get("quarter") or address.get("district"))
    if neighborhood:
        n = title(neighborhood)
        segments.append(n)
        location_parts["Sublocation"] = n

    places = ["Places/" + "/".join(segments)] if segments else []

    cc = address.get("country_code")
    country_code = cc.upper() if cc else None
    if country_code:
        location_parts["CountryCode"] = country_code

    return places, country_code, location_parts


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
    """Run PaddleOCR and return (phrases, regions).

    Phrases are titlecased strings (no tag-root prefix). They are stored in
    ``XMP-phototools:OCRText`` and IPTC ImageRegion metadata — not as keyword
    tags — so they don't pollute the digiKam tag tree.
    """
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
        log_ocr.warning("PaddleOCR error on %s: %s", path.name, e)
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

        for poly, text, score in zip(polys, texts, scores, strict=False):
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
            tags.append(title(phrase))

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
        log_ocr.debug("found %d text fragment(s) in %s", len(tags), path.name)

    return tags, regions


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
    enable_gps: bool = True,
    clip_model: str = None,
    clip_pretrained: str = None,
    landmarks_path: Path = None,
    gps_fallback: tuple[float, float] | None = None,
    bypass_version_check: bool = False,
) -> bool:
    cfg = get_config()
    summary = PhotoSummary()
    photo_t0 = time.perf_counter()

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

    if (stored_version == TAGGER_VERSION and not force and not clear_all
            and not bypass_version_check):
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

    # ----- GPS reverse-geocoding ------------------------------------------
    places: list[str] = []
    country_code: str | None = None
    location_parts: dict[str, str] = {}
    if enable_gps:
        with timed_step("geocoding", photo=path.name, catch=True) as s:
            s.ran = True
            places, country_code, location_parts = tags_from_gps(exif)
            s.ok = True
            if places:
                log_geo.info("%s → %s", path.name, places[0])
                all_tags.extend(places)
            else:
                log_geo.debug("%s: no place tags", path.name)
        summary.record("gps", s.ran, s.ok)
    else:
        summary.skip("gps")

    # ----- OCR ------------------------------------------------------------
    ocr_phrases: list[str] = []
    ocr_regions: list[dict] = []
    if enable_ocr and visual_path:
        with timed_step("ocr", photo=path.name, catch=True) as s:
            s.ran = True
            ocr_phrases, ocr_regions = tags_from_ocr(visual_path)
            s.ok = True
            get_counter("ocr").add("text_regions", len(ocr_regions))
            if ocr_phrases:
                log_ocr.info("%s: %d region(s) — %s",
                             path.name, len(ocr_regions),
                             ", ".join(ocr_phrases[:3])
                             + (" …" if len(ocr_phrases) > 3 else ""))
            else:
                log_ocr.debug("%s: no text", path.name)
        summary.record("ocr", s.ran, s.ok)
    else:
        summary.skip("ocr")

    embedding = None
    need_visual = (enable_ram or enable_landmarks) and visual_path
    prepared = prepare_image(visual_path, cfg.clip.max_pixels) if need_visual else None
    visual_input = prepared or visual_path

    # ----- RAM++ ---------------------------------------------------------
    if enable_ram and visual_path:
        with timed_step("ram", photo=path.name, catch=True) as s:
            s.ran = True
            tagger = _get_ram_tagger()
            ram_tags, scored_ram_tags = tagger.tag_image(visual_input)
            s.ok = True
            get_counter("ram").add("tags_emitted", len(ram_tags))
            if scored_ram_tags:
                log_ram = get_logger("ram")
                log_ram.info(
                    "%s: top 5 — %s", path.name,
                    ", ".join(f"{t} ({sc:.3f}/thr {thr:.3f})"
                              for t, sc, thr in scored_ram_tags[:5]),
                )
                log_ram.debug("%s: all tags %s", path.name, ram_tags)
            all_tags.extend(ram_tags)
        summary.record("ram", s.ran, s.ok)
    else:
        summary.skip("ram")

    # ----- CLIP embedding (used by RAM++ pipeline + landmark lookup) -----
    if (enable_ram or enable_landmarks) and visual_path:
        with timed_step("clip", photo=path.name, catch=True) as s:
            s.ran = True
            embedder = _get_clip_embedder(clip_model, clip_pretrained)
            embedding = embedder.embed_image(visual_input)
            s.ok = True

    if prepared:
        try:
            os.unlink(prepared)
        except OSError:
            pass

    # ----- Landmarks -----------------------------------------------------
    landmark_ran = False
    landmark_ok = False
    if enable_landmarks and embedding is not None:
        coords = get_gps_coords(exif) or gps_fallback
        if coords is not None:
            lat, lon = coords
            with timed_step("landmarks", photo=path.name, catch=True) as s:
                lm_index = _get_landmark_index(landmarks_path)
                if lm_index is not None:
                    s.ran = True
                    landmark_ran = True
                    landmark, lm_top, lm_thr = lm_index.lookup(
                        embedding, lat=lat, lon=lon)
                    s.ok = True
                    landmark_ok = True
                    if landmark:
                        get_counter("landmarks").add("matched")
                        log_lm = get_logger("landmarks")
                        log_lm.info("%s → %s (thr %.3f)",
                                    path.name, landmark, lm_thr)
                        if lm_top:
                            log_lm.debug(
                                "%s: top 3 — %s", path.name,
                                ", ".join(f"{n} ({sc:.3f})"
                                          for n, sc in lm_top[:3]),
                            )
                        all_tags.append(f"Landmarks/{title(landmark)}")
                    else:
                        get_counter("landmarks").add("no_match")
                        log_lm = get_logger("landmarks")
                        log_lm.debug(
                            "%s: no match (thr %.3f, top %s)", path.name, lm_thr,
                            [(n, f"{sc:.3f}") for n, sc in lm_top[:3]] if lm_top else [],
                        )
    if landmark_ran:
        summary.record("landmarks", True, landmark_ok)
    else:
        summary.skip("landmarks")

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

    # Project existing People/* keywords (digiKam-owned) into the IPTC
    # PersonInImage list so non-digiKam consumers (Photo Mechanic, Mylio,
    # Lightroom IPTC panel) can surface them.
    person_in_image: list[str] | None = None
    existing_tags_list = exif.get("TagsList") or []
    if isinstance(existing_tags_list, str):
        existing_tags_list = [existing_tags_list]
    people_leaves = [
        leaf_of(t) for t in existing_tags_list
        if isinstance(t, str) and t.startswith("People/")
    ]
    if people_leaves:
        person_in_image = deduplicate(people_leaves)

    embedding_model_id = (
        _get_clip_embedder(clip_model, clip_pretrained).model_id
        if embedding is not None else None
    )
    existing_iptc_regions = (
        existing_non_ocr_regions(exif) if ocr_regions else None
    )

    result = write_metadata(
        path,
        new_keywords=new_tags,
        namespace_fields=namespace_fields,
        location_fields=location_parts or None,
        person_in_image=person_in_image,
        ocr_text=ocr_phrases if ocr_regions else None,
        new_ocr_regions=ocr_regions or None,
        existing_iptc_regions=existing_iptc_regions,
        embedding=embedding,
        embedding_model=embedding_model_id,
        # Stamp OCRRan whenever the OCR pipeline ran, so a future `tag fix`
        # knows OCR has been considered (even when nothing was detected).
        stamp_ocr_ran=bool(enable_ocr and visual_path),
        dry_run=dry_run,
    )

    elapsed = time.perf_counter() - photo_t0
    log.info("%s done in %.1fs  %s", path.name, elapsed, summary.render())
    return result


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

def build_tag_parser(subparsers, parents=None) -> argparse.ArgumentParser:
    tag_parser = subparsers.add_parser(
        "tag",
        parents=parents or [],
        help="Auto-tag photos (RAM++, landmarks, OCR, GPS, EXIF). "
             "Skips photos already at the current TaggerVersion. "
             "Pass --fix to fill in missing pipeline outputs instead.",
    )
    tag_parser.add_argument("path", type=Path, nargs="+",
                            help="Image file(s) or directory (one or more)")
    tag_parser.add_argument("-n", "--dry-run", action="store_true",
                            help="Preview tags without writing")
    tag_parser.add_argument("--fix", action="store_true",
                            help="Fill in missing pipeline outputs per-photo "
                                 "(geocoding, RAM++, landmarks, OCR). Detects "
                                 "what's missing from existing metadata.")
    tag_parser.add_argument("-f", "--force", action="store_true",
                            help="Re-tag (clears old autogenerated tags first). "
                                 "Ignored with --fix.")
    tag_parser.add_argument("--clear-all", action="store_true",
                            help="Wipe ALL keywords before re-tagging (nuclear "
                                 "option). Ignored with --fix.")
    tag_parser.add_argument("-w", "--watch", action="store_true",
                            help="Watch directory for new images. "
                                 "Not compatible with --fix.")

    pipelines = tag_parser.add_argument_group(
        "pipeline selection",
        "Choose which tagging pipelines to run. "
        "When none are specified, all pipelines run. "
        "Default mode: EXIF metadata and GPS geocoding always run. "
        "With --fix: each pipeline only runs on photos where its output "
        "is missing; --gps additionally gates the geocoding pipeline."
    )
    pipelines.add_argument("--gps", action="store_true",
                           help="(--fix only) Consider GPS reverse geocoding")
    pipelines.add_argument("--ram", action="store_true",
                           help="Run RAM++ image content tagging")
    pipelines.add_argument("--landmarks", action="store_true",
                           help="Run landmark lookup (CLIP embedding + GPS)")
    pipelines.add_argument("--ocr", action="store_true",
                           help="Run OCR text detection")

    tag_parser.add_argument("--clip-model", default=None,
                            help="CLIP model name (default: from config)")
    tag_parser.add_argument("--clip-pretrained", default=None,
                            help="CLIP pretrained weights (default: from config)")
    tag_parser.add_argument("--landmarks-db", type=Path, default=None,
                            dest="landmarks_db",
                            help="Path to landmarks.json")

    tag_parser.set_defaults(func=_dispatch_tag)
    return tag_parser


def _dispatch_tag(args) -> None:
    if args.fix:
        if args.watch:
            log.error("--watch is not compatible with --fix")
            sys.exit(1)
        if args.force or args.clear_all:
            log.warning("--force / --clear-all are ignored with --fix")
        run_tag_fix(args)
    else:
        if args.gps:
            log.warning("--gps only applies with --fix; ignored")
        run_tag(args)


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

    tui.start(total=len(images), header="tag",
              enabled=not getattr(args, "no_tui", False))
    try:
        for i, img in enumerate(images, 1):
            tui.set_photo(i, img.name)
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
    finally:
        tui.stop()

    log.info("Done. %d tagged, %d skipped, %d failed.", success, skipped, failed)
    log_run_summary()


# ---------------------------------------------------------------------------
# tag fix — per-pipeline missing-output detection
# ---------------------------------------------------------------------------

def _read_fix_metadata_batch(paths: list[Path]) -> dict[Path, dict]:
    """Batch-read keywords + OCRRan + GPS coords for `tag fix` planning.

    Returns {path: {"keywords": set[str], "ocr_ran": bool, "coords": tuple|None}}
    where keywords are full hierarchical strings (Places/..., Objects/...).
    """
    if not paths:
        return {}

    cfg = get_config()
    batch_size = cfg.exiftool.batch_size
    total = len(paths)
    n_batches = (total + batch_size - 1) // batch_size
    result: dict[Path, dict] = {}

    fields = [
        "-IPTC:Keywords", "-XMP:Subject", "-XMP-digiKam:TagsList",
        "-XMP-phototools:OCRRan",
        "-GPS:GPSLatitude", "-GPS:GPSLongitude",
        "-GPS:GPSLatitudeRef", "-GPS:GPSLongitudeRef",
        "-n",
    ]

    for batch_idx, i in enumerate(range(0, total, batch_size), 1):
        batch = paths[i:i + batch_size]
        log.info("Reading metadata batch %d/%d (%d files, %d/%d total)",
                 batch_idx, n_batches, len(batch),
                 min(i + len(batch), total), total)
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(fields + files, timeout=120)
        if not meta_list:
            for p in batch:
                result[p] = {"keywords": set(), "ocr_ran": False, "coords": None}
            continue
        for path, meta in _group_metas_by_path(meta_list, str_to_path).items():
            kw = set()
            for field in ("Keywords", "Subject", "TagsList"):
                val = meta.get(field, [])
                if isinstance(val, str):
                    val = [val]
                for v in val:
                    kw.add(str(v))
            result[path] = {
                "keywords": kw,
                "ocr_ran": bool(meta.get("OCRRan")),
                "coords": get_gps_coords(meta),
            }
    return result


def _has_keyword_with_prefix(keywords: set[str], prefix: str) -> bool:
    """True if any keyword in the set begins with `prefix` (e.g. 'Places/')."""
    return any(k.startswith(prefix) for k in keywords)


def _decide_fix_pipelines(
    meta: dict,
    consider_gps: bool,
    consider_ram: bool,
    consider_landmarks: bool,
    consider_ocr: bool,
    landmarks_db_exists: bool,
    gps_fallback: tuple[float, float] | None,
) -> tuple[bool, bool, bool, bool]:
    """Decide which pipelines should run for one photo.

    Returns (run_gps, run_ram, run_landmarks, run_ocr).
    """
    keywords = meta["keywords"]
    has_exif_coords = meta["coords"] is not None
    has_any_coords = has_exif_coords or gps_fallback is not None

    # Geocoding uses only EXIF coords — the GPS timeline fallback is consumed
    # by the landmark pipeline only (see process_single).
    run_gps = (consider_gps
               and has_exif_coords
               and not _has_keyword_with_prefix(keywords, "Places/"))

    run_ram = (consider_ram
               and not _has_keyword_with_prefix(keywords, "Objects/")
               and not _has_keyword_with_prefix(keywords, "Scenes/"))

    run_landmarks = (consider_landmarks
                     and landmarks_db_exists
                     and has_any_coords
                     and not _has_keyword_with_prefix(keywords, "Landmarks/"))

    run_ocr = consider_ocr and not meta["ocr_ran"]

    return run_gps, run_ram, run_landmarks, run_ocr


def run_tag_fix(args) -> None:
    cfg = get_config()

    any_selected = args.gps or args.ram or args.landmarks or args.ocr
    consider_gps = args.gps or not any_selected
    consider_ram = args.ram or not any_selected
    consider_landmarks = args.landmarks or not any_selected
    consider_ocr = args.ocr or not any_selected

    lm_path = args.landmarks_db or Path(cfg.landmarks.default_path).expanduser()
    landmarks_db_exists = lm_path.exists()
    if consider_landmarks and not landmarks_db_exists:
        log.warning("Landmarks DB not found at %s — landmark fixes will be skipped",
                    lm_path)

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

    motion_companions = [p for p in images if is_live_photo_motion(p)]
    if motion_companions:
        log.info("Skipping %d Live Photo motion companion(s)", len(motion_companions))
        images = [p for p in images if not is_live_photo_motion(p)]

    if not images:
        log.info("Done. 0 fixed, 0 already-complete, 0 failed.")
        return

    log.info("Considering %d image(s) for fixing", len(images))

    metadata = _read_fix_metadata_batch(images)
    gps_timeline = build_gps_timeline(images) if consider_landmarks else {}

    plans: list[tuple[Path, tuple[bool, bool, bool, bool]]] = []
    for img in images:
        meta = metadata.get(img, {"keywords": set(), "ocr_ran": False, "coords": None})
        plan = _decide_fix_pipelines(
            meta, consider_gps, consider_ram, consider_landmarks, consider_ocr,
            landmarks_db_exists, gps_timeline.get(img),
        )
        plans.append((img, plan))

    needs_work = [(p, plan) for p, plan in plans if any(plan)]
    already_complete = len(plans) - len(needs_work)
    log.info("Plan: %d need work, %d already complete",
             len(needs_work), already_complete)

    if not needs_work:
        log.info("Done. 0 fixed, %d already-complete, 0 failed.", already_complete)
        return

    counts = {"gps": 0, "ram": 0, "landmarks": 0, "ocr": 0}
    for _, (g, r, lm, o) in needs_work:
        if g:
            counts["gps"] += 1
        if r:
            counts["ram"] += 1
        if lm:
            counts["landmarks"] += 1
        if o:
            counts["ocr"] += 1
    log.info("Pipeline workload: GPS=%d, RAM++=%d, Landmarks=%d, OCR=%d",
             counts["gps"], counts["ram"], counts["landmarks"], counts["ocr"])

    success = failed = 0
    tui.start(total=len(needs_work), header="tag --fix",
              workload=counts,
              enabled=not getattr(args, "no_tui", False))
    try:
        for i, (img, (run_gps, run_ram, run_lm, run_ocr)) in enumerate(needs_work, 1):
            tui.set_photo(i, img.name)
            log.info("[%d/%d] %s  (gps=%s ram=%s landmarks=%s ocr=%s)",
                     i, len(needs_work), img,
                     run_gps, run_ram, run_lm, run_ocr)
            try:
                ok = process_single(
                    img, args.dry_run, force=False,
                    clear_all=False,
                    enable_ram=run_ram,
                    enable_landmarks=run_lm,
                    enable_ocr=run_ocr,
                    enable_gps=run_gps,
                    clip_model=args.clip_model,
                    clip_pretrained=args.clip_pretrained,
                    landmarks_path=args.landmarks_db,
                    gps_fallback=gps_timeline.get(img),
                    bypass_version_check=True,
                )
                if ok:
                    success += 1
            except Exception as e:
                log.error("Error fixing %s: %s", img.name, e)
                failed += 1
    finally:
        tui.stop()

    log.info("Done. %d fixed, %d already-complete, %d failed.",
             success, already_complete, failed)
    log_run_summary()

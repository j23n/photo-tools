"""
build_landmarks.py — Build a CLIP embedding database of notable landmarks.

Queries Wikidata for top landmarks (by sitelinks count), downloads
images per landmark from Wikidata image properties, computes CLIP
embeddings, averages them, and outputs landmarks.json for use by
LandmarkIndex.
"""

import io
import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

from photo_tools.config import get_config
from photo_tools.logging_setup import get_logger

log = get_logger("build_landmarks")

# Per-type SPARQL query — uses direct P31 only (no recursive P279* subclass traversal)
SPARQL_TYPE_QUERY = """\
SELECT ?item ?itemLabel ?lat ?lon ?image ?sitelinks WHERE {{
  ?item wdt:P31 wd:{qid} .
  ?item wdt:P625 ?coords .
  ?item wdt:P18 ?image .
  ?item wikibase:sitelinks ?sitelinks .
  BIND(geof:latitude(?coords) AS ?lat)
  BIND(geof:longitude(?coords) AS ?lon)
  FILTER(BOUND(?lat) && BOUND(?lon))
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {limit}
"""

# Geo-filtered variant: same structure but with a bounding-box constraint.
SPARQL_TYPE_GEO_QUERY = """\
SELECT ?item ?itemLabel ?lat ?lon ?image ?sitelinks WHERE {{
  ?item wdt:P31 wd:{qid} .
  ?item wdt:P625 ?coords .
  ?item wdt:P18 ?image .
  ?item wikibase:sitelinks ?sitelinks .
  BIND(geof:latitude(?coords) AS ?lat)
  BIND(geof:longitude(?coords) AS ?lon)
  FILTER(?lat >= {lat_min} && ?lat <= {lat_max} && ?lon >= {lon_min} && ?lon <= {lon_max})
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {limit}
"""

LANDMARK_TYPES = [
    # (qid, label, per-type SPARQL limit)
    # General
    ("Q570116", "tourist attraction", 2000),
    ("Q839954", "archaeological site", 1500),
    ("Q4989906", "monument", 2000),
    ("Q811979", "architectural structure", 2000),
    ("Q2319498", "landmark", 500),
    # Religious (spread across regions)
    ("Q16970", "church building", 1500),
    ("Q32815", "mosque", 1500),
    ("Q44539", "temple", 1500),
    ("Q34627", "synagogue", 500),
    ("Q160031", "pagoda", 500),
    ("Q194195", "stupa", 500),
    ("Q697295", "Shinto shrine", 1000),
    # Built structures
    ("Q751876", "castle", 1500),
    ("Q23413", "palace", 1500),
    ("Q57821", "fortification", 1000),
    ("Q3947", "house", 2000),
    ("Q12280", "bridge", 1500),
    ("Q12518", "tower", 1500),
    ("Q39715", "lighthouse", 500),
    ("Q11303", "skyscraper", 1000),
    ("Q483453", "fountain", 500),
    # Cultural
    ("Q33506", "museum", 1500),
    ("Q483110", "stadium", 1500),
    ("Q179700", "statue", 1500),
    ("Q174782", "town square", 1000),
    # Natural
    ("Q35112127", "natural landmark", 500),
    ("Q8502", "mountain", 1500),
    ("Q8072", "volcano", 500),
    ("Q23397", "lake", 1500),
    ("Q34038", "waterfall", 500),
    ("Q133056", "cave", 500),
]

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Wikidata image properties to collect per landmark (all curator-assigned)
WIKIDATA_IMAGE_PROPS = [
    "P18",    # image
    "P3451",  # nighttime view (old)
    "P8517",  # view
    "P5252",  # winter view
    "P7417",  # nighttime view
]

# File extensions to skip from image results
SKIP_EXTENSIONS = {".svg", ".ogg", ".ogv", ".webm", ".djvu", ".pdf", ".mid",
                   ".flac", ".wav", ".tiff", ".tif"}

# Geographic regions for --test mode
TEST_REGIONS = [
    ("Rome",    41.75, 42.05, 12.30, 12.70),
    ("Bologna", 44.35, 44.65, 11.15, 11.55),
]

IMAGE_CACHE_DIR = Path.home() / ".cache/photo-tools/landmark-images"

_NO_THUMB_EXTS = {".tiff", ".tif", ".djvu"}


def _sparql_get(query: str) -> list[dict]:
    """Execute a SPARQL query with retry on timeout/5xx."""
    cfg = get_config()
    max_retries = cfg.wikidata.max_retries
    retry_backoff = cfg.wikidata.retry_backoff

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                cfg.wikidata.sparql_url,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                raise
            retry_after = int(getattr(getattr(e, "response", None), "headers", {}).get("Retry-After", 0))
            wait = retry_after if retry_after else retry_backoff * attempt
            log.warning("Query failed (%s), retrying in %ds (%d/%d)", e, wait, attempt, max_retries)
            time.sleep(wait)
    return []  # unreachable, keeps type checkers happy


def _api_get(url: str, params: dict) -> dict | None:
    """GET a JSON API endpoint, retrying on 429 with Retry-After."""
    cfg = get_config()
    for _ in range(2):
        try:
            resp = requests.get(
                url, params=params,
                headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
                timeout=cfg.wikidata.fetch_timeout,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                log.warning("API rate limited — sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.debug("API request failed (%s): %s", url, e)
            return None
    log.warning("Still rate-limited after retry for %s, skipping", url)
    return None


def _filename_to_url(filename: str) -> str:
    """Convert a Wikimedia Commons filename to a Special:FilePath URL."""
    encoded = urllib.parse.quote(filename.replace(" ", "_"))
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}"


def fetch_image_urls(wikidata_id: str, target: int | None = None) -> list[dict]:
    """Collect up to *target* image URLs for a landmark.

    Each entry is ``{"url", "source"}``.

    Source: Wikidata image properties (curator-assigned).
    """
    if target is None:
        target = get_config().wikidata.target_images

    seen_filenames: set[str] = set()
    urls: list[dict] = []

    data = _api_get(WIKIDATA_API, {
        "action": "wbgetentities", "ids": wikidata_id,
        "props": "claims", "format": "json",
    })
    if not data:
        return urls

    claims = data.get("entities", {}).get(wikidata_id, {}).get("claims", {})
    for prop in WIKIDATA_IMAGE_PROPS:
        for claim in claims.get(prop, []):
            try:
                filename = claim["mainsnak"]["datavalue"]["value"]
                key = filename.replace(" ", "_").lower()
                if key in seen_filenames:
                    continue
                ext = Path(filename).suffix.lower()
                if ext in SKIP_EXTENSIONS:
                    continue
                seen_filenames.add(key)
                urls.append({
                    "url": _filename_to_url(filename),
                    "source": f"wikidata:{prop}",
                })
            except (KeyError, TypeError):
                pass
            if len(urls) >= target:
                return urls[:target]

    return urls[:target]


def query_wikidata(limit: int) -> list[dict]:
    """Query Wikidata SPARQL for notable landmarks, one type at a time.

    Each type gets an equal share of *limit* (per-type quota).  Types that
    return fewer results than their quota leave room for an overflow pass
    that fills remaining slots from the most notable leftover landmarks
    across all types.  This prevents high-sitelink European types from
    crowding out underrepresented regions.
    """
    log.info("Querying Wikidata for top %d landmarks across %d types ...", limit, len(LANDMARK_TYPES))

    seen_ids: set[str] = set()
    per_type: list[tuple[str, list[dict]]] = []

    for qid, label, type_limit in LANDMARK_TYPES:
        log.info("  Querying %s (%s, limit %d) ...", label, qid, type_limit)
        query = SPARQL_TYPE_QUERY.format(qid=qid, limit=type_limit)
        results = _sparql_get(query)
        log.info("  Got %d results for %s", len(results), label)

        type_results: list[dict] = []
        for r in results:
            wikidata_id = r["item"]["value"].rsplit("/", 1)[-1]
            if wikidata_id in seen_ids:
                continue
            seen_ids.add(wikidata_id)
            type_results.append({
                "name": r["itemLabel"]["value"],
                "wikidata_id": wikidata_id,
                "lat": float(r["lat"]["value"]),
                "lon": float(r["lon"]["value"]),
                "image_url": r["image"]["value"],
                "_sitelinks": int(r["sitelinks"]["value"]),
            })
        # Results arrive sorted by sitelinks desc from SPARQL ORDER BY
        per_type.append((label, type_results))
        time.sleep(2)

    # Per-type quotas: each type gets an equal share, then overflow fills
    # remaining slots from the most notable leftover landmarks globally
    quota = limit // len(LANDMARK_TYPES)
    landmarks: list[dict] = []
    overflow: list[dict] = []

    for label, type_results in per_type:
        landmarks.extend(type_results[:quota])
        overflow.extend(type_results[quota:])

    remaining = limit - len(landmarks)
    if remaining > 0:
        overflow.sort(key=lambda x: x["_sitelinks"], reverse=True)
        landmarks.extend(overflow[:remaining])

    for lm in landmarks:
        del lm["_sitelinks"]

    log.info("Selected %d landmarks (%d per-type quota, %d types)",
             len(landmarks), quota, len(LANDMARK_TYPES))
    return landmarks


def query_wikidata_geo(
    regions: list[tuple[str, float, float, float, float]],
    limit: int,
) -> list[dict]:
    """Query Wikidata for notable landmarks within geographic bounding boxes.

    Uses the same per-type quota strategy as :func:`query_wikidata`.
    """
    log.info("Querying Wikidata for top %d landmarks in %d regions ...",
             limit, len(regions))

    seen_ids: set[str] = set()
    per_type: dict[str, list[dict]] = {}

    for region_name, lat_min, lat_max, lon_min, lon_max in regions:
        log.info("Region: %s (lat %.2f–%.2f, lon %.2f–%.2f)",
                 region_name, lat_min, lat_max, lon_min, lon_max)
        for qid, label, type_limit in LANDMARK_TYPES:
            geo_limit = min(type_limit, get_config().wikidata.test_per_type_limit)
            query = SPARQL_TYPE_GEO_QUERY.format(
                qid=qid, limit=geo_limit,
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
            )
            results = _sparql_get(query)
            if results:
                log.info("  %s / %s: %d results", region_name, label, len(results))
            type_results = per_type.setdefault(label, [])
            for r in results:
                wikidata_id = r["item"]["value"].rsplit("/", 1)[-1]
                if wikidata_id in seen_ids:
                    continue
                seen_ids.add(wikidata_id)
                type_results.append({
                    "name": r["itemLabel"]["value"],
                    "wikidata_id": wikidata_id,
                    "lat": float(r["lat"]["value"]),
                    "lon": float(r["lon"]["value"]),
                    "image_url": r["image"]["value"],
                    "_sitelinks": int(r["sitelinks"]["value"]),
                })
            time.sleep(1)

    # Per-type quotas with overflow (same as query_wikidata)
    quota = limit // max(len(per_type), 1)
    landmarks: list[dict] = []
    overflow: list[dict] = []

    for label, type_results in per_type.items():
        type_results.sort(key=lambda x: x["_sitelinks"], reverse=True)
        landmarks.extend(type_results[:quota])
        overflow.extend(type_results[quota:])

    remaining = limit - len(landmarks)
    if remaining > 0:
        overflow.sort(key=lambda x: x["_sitelinks"], reverse=True)
        landmarks.extend(overflow[:remaining])

    for lm in landmarks:
        del lm["_sitelinks"]

    log.info("Found %d unique landmarks across regions (%d per-type quota, %d kept)",
             len(seen_ids), quota, len(landmarks))
    return landmarks


def _fetch(url: str, wikidata_id: str, filename: str) -> requests.Response | None:
    """GET *url*; retry once on 429 using Retry-After, skip on other errors."""
    cfg = get_config()
    for _ in range(2):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
                timeout=cfg.wikidata.fetch_timeout,
                stream=True,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                log.warning("Rate limited on %s — sleeping %ds", filename, retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning("Image download failed for %s (%s): %s", wikidata_id, filename, e)
            return None
    log.warning("Still rate-limited after retry for %s (%s), skipping", wikidata_id, filename)
    return None


def _save_stream(resp: requests.Response, dest: Path, wikidata_id: str, filename: str) -> bool:
    """Stream response to *dest*, returning True on success."""
    cfg = get_config()
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    size = tmp.stat().st_size
    if size < cfg.wikidata.min_image_size:
        log.warning("Downloaded image too small for %s (%s): %d bytes", wikidata_id, filename, size)
        tmp.unlink()
        return False
    tmp.rename(dest)
    return True


def download_image(url: str, wikidata_id: str, cache_key: str | None = None) -> tuple[Path | None, bool]:
    """Download a Wikimedia Commons image, resized via thumbnail API.

    For formats the thumbnail API cannot handle (TIFF, DjVu, …), the
    full original is downloaded and resized locally with Pillow.

    Images are cached in ~/.cache/photo-tools/landmark-images/.
    """
    cfg = get_config()
    max_width = cfg.wikidata.thumbnail_max_width
    min_size = cfg.wikidata.min_image_size
    jpeg_quality = cfg.wikidata.thumbnail_jpeg_quality

    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    ext = Path(filename).suffix.lower()
    needs_local_resize = ext in _NO_THUMB_EXTS

    cache_suffix = ".jpg" if needs_local_resize else (ext or ".jpg")
    key = cache_key or wikidata_id
    cached = IMAGE_CACHE_DIR / f"{key}{cache_suffix}"
    if cached.exists() and cached.stat().st_size >= min_size:
        return cached, True

    if needs_local_resize:
        resp = _fetch(url, wikidata_id, filename)
        if resp is None:
            return None, False
        try:
            img = Image.open(io.BytesIO(resp.content))
            ImageOps.exif_transpose(img, in_place=True)
            img.thumbnail((max_width, max_width))
            img = img.convert("RGB")
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            img.save(tmp, "JPEG", quality=jpeg_quality)
            size = tmp.stat().st_size
            if size < min_size:
                log.warning("Resized image too small for %s (%s): %d bytes", wikidata_id, filename, size)
                tmp.unlink()
                return None, False
            tmp.rename(cached)
            return cached, False
        except Exception as e:
            log.warning("Local resize failed for %s (%s): %s", wikidata_id, filename, e)
            return None, False
    else:
        thumb_url = f"https://commons.wikimedia.org/w/thumb.php?f={urllib.parse.quote(filename)}&w={max_width}"
        resp = _fetch(thumb_url, wikidata_id, filename)
        if resp is None:
            return None, False
        if _save_stream(resp, cached, wikidata_id, filename):
            return cached, False
        return None, False


def build_database(
    limit: int,
    output_path: Path,
    clip_model: str,
    clip_pretrained: str,
    resume: bool,
    wikidata_cache: Path | None = None,
    test: bool = False,
    images_per_landmark: int | None = None,
) -> None:
    """Build the landmark embedding database."""
    from photo_tools.clip_tagger import CLIPEmbedder

    cfg = get_config()
    if images_per_landmark is None:
        images_per_landmark = cfg.wikidata.target_images

    # Load existing data if resuming
    existing: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        for lm in data.get("landmarks", []):
            existing[lm["wikidata_id"]] = lm
        log.info("Resuming: %d landmarks already computed", len(existing))

    dump_path = output_path.with_suffix(".wikidata.json")
    if wikidata_cache and wikidata_cache.exists():
        with open(wikidata_cache) as f:
            landmarks = json.load(f)[:limit]
        log.info("Loaded %d landmarks from %s", len(landmarks), wikidata_cache)
    elif dump_path.exists():
        with open(dump_path) as f:
            landmarks = json.load(f)[:limit]
        log.info("Loaded %d landmarks from cache %s", len(landmarks), dump_path)
    else:
        if test:
            landmarks = query_wikidata_geo(TEST_REGIONS, cfg.wikidata.test_limit)
        else:
            landmarks = query_wikidata(limit)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(landmarks, f, indent=2)
        log.info("Dumped %d Wikidata entries to %s", len(landmarks), dump_path)

    candidates = [lm for lm in landmarks
                  if lm["wikidata_id"] not in existing
                  or not existing[lm["wikidata_id"]].get("embedding")]

    # Phase 0: collect image URLs
    # Cache structure: {wikidata_id: {name, entity_url, images: [...]}}
    urls_cache_path = output_path.with_suffix(".urls.json")
    landmark_urls: dict[str, dict] = {}
    if urls_cache_path.exists():
        with open(urls_cache_path) as f:
            landmark_urls = json.load(f)
        # Migrate old formats
        for wid, value in list(landmark_urls.items()):
            if isinstance(value, list):
                # Old format: list[str] or list[dict]
                images = value
                if images and isinstance(images[0], str):
                    images = [{"url": u, "source": "unknown"} for u in images]
                landmark_urls[wid] = {
                    "name": wid,
                    "entity_url": f"https://www.wikidata.org/wiki/{wid}",
                    "images": images,
                }
        log.info("Loaded cached URLs for %d landmarks from %s",
                 len(landmark_urls), urls_cache_path)

    urls_to_fetch = [lm for lm in candidates
                     if lm["wikidata_id"] not in landmark_urls]
    if urls_to_fetch:
        log.info("Collecting image URLs for %d landmarks ...", len(urls_to_fetch))
        for i, lm in enumerate(urls_to_fetch, 1):
            wid = lm["wikidata_id"]
            print(f"\r  urls [{i}/{len(urls_to_fetch)}] {lm['name']:<50}",
                  end="", flush=True, file=sys.stderr)
            images = fetch_image_urls(wid, target=images_per_landmark)
            landmark_urls[wid] = {
                "name": lm["name"],
                "entity_url": f"https://www.wikidata.org/wiki/{wid}",
                "images": images,
            }
            log.debug("  %s: %d URLs", lm["name"], len(images))
            urls_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(urls_cache_path, "w") as f:
                json.dump(landmark_urls, f)
            time.sleep(0.5)  # be kind to the APIs
        print(file=sys.stderr)
        log.info("Collected URLs for %d landmarks (total %d in cache)",
                 len(urls_to_fetch), len(landmark_urls))

    # Download + embed in batches
    batch_size = cfg.wikidata.save_interval
    embedder = None  # lazy-init on first batch that needs embedding

    results = list(existing.values())
    download_failed = 0
    downloaded_count = 0
    cached_count = 0
    embed_failed = 0
    total_landmarks = len(landmarks)

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(candidates) + batch_size - 1) // batch_size
        log.info("Batch %d/%d (%d landmarks) ...", batch_num, total_batches, len(batch))

        # Download images for this batch
        batch_downloads: list[tuple[str, int, str]] = []
        image_paths: dict[str, list[Path]] = {}
        for lm in batch:
            wid = lm["wikidata_id"]
            lm_data = landmark_urls.get(wid, {})
            images = lm_data.get("images", []) if isinstance(lm_data, dict) else lm_data
            for idx, entry in enumerate(images):
                url = entry["url"] if isinstance(entry, dict) else entry
                cache_key = f"{wid}_{idx}"
                filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
                ext = Path(filename).suffix.lower() or ".jpg"
                suffix = ".jpg" if ext in _NO_THUMB_EXTS else ext
                cached = IMAGE_CACHE_DIR / f"{cache_key}{suffix}"
                if cached.exists() and cached.stat().st_size >= cfg.wikidata.min_image_size:
                    image_paths.setdefault(wid, []).append(cached)
                    cached_count += 1
                else:
                    batch_downloads.append((wid, idx, url))

        if batch_downloads:
            done_count = 0
            total = len(batch_downloads)

            def _download(task: tuple[str, int, str]) -> tuple[str, int, Path | None, bool]:
                wid, idx, url = task
                cache_key = f"{wid}_{idx}"
                img_path, was_cached = download_image(url, wid, cache_key=cache_key)
                return wid, idx, img_path, was_cached

            with ThreadPoolExecutor(max_workers=cfg.wikidata.download_workers) as pool:
                futures = [pool.submit(_download, t) for t in batch_downloads]
                for future in as_completed(futures):
                    wid, idx, img_path, was_cached = future.result()
                    done_count += 1
                    if img_path is not None:
                        image_paths.setdefault(wid, []).append(img_path)
                        if was_cached:
                            cached_count += 1
                        else:
                            downloaded_count += 1
                    else:
                        download_failed += 1
                    if done_count % 10 == 0 or done_count == total:
                        print(f"\r  images [{done_count}/{total}] "
                              f"{downloaded_count} new, {download_failed} failed",
                              end="", flush=True, file=sys.stderr)
            print(file=sys.stderr)

        # Embed this batch
        for lm in batch:
            wid = lm["wikidata_id"]
            paths = image_paths.get(wid, [])
            if not paths:
                continue

            global_idx = batch_start + batch.index(lm) + 1
            print(f"\r  embedding [{global_idx}/{total_landmarks}] "
                  f"{lm['name']:<40} ({len(paths)} images)",
                  end="", flush=True, file=sys.stderr)

            if embedder is None:
                print(file=sys.stderr)
                embedder = CLIPEmbedder(model_name=clip_model, pretrained=clip_pretrained)

            embeddings = []
            for img_path in paths:
                try:
                    emb = embedder.embed_image(img_path)
                    embeddings.append(emb)
                except Exception as e:
                    log.debug("Failed to embed %s image %s: %s",
                              lm["name"], img_path.name, e)

            if not embeddings:
                embed_failed += 1
                continue

            avg = np.mean(embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)

            results.append({
                "name": lm["name"],
                "wikidata_id": wid,
                "lat": lm["lat"],
                "lon": lm["lon"],
                "embedding": avg.tolist(),
            })

        # Save after each batch
        print(file=sys.stderr)
        _save(output_path, clip_model, clip_pretrained, results)

    log.info("Done. %d landmarks saved, %d download failed, %d embed failed",
             len(results), download_failed, embed_failed)


def _save(output_path: Path, model: str, pretrained: str, landmarks: list[dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": f"{model}/{pretrained}",
            "landmarks": landmarks,
        }, f)
    log.info("Saved %d landmarks to %s", len(landmarks), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_landmarks_parser(subparsers):
    """Register the 'landmarks' subcommand group with generate-db + query."""
    lm_parser = subparsers.add_parser(
        "landmarks",
        help="Landmark database tools: build the DB, or query an image against it.",
    )
    lm_sub = lm_parser.add_subparsers(dest="landmarks_command", required=True)

    gen = lm_sub.add_parser(
        "generate-db",
        help="Build CLIP embedding database of notable landmarks from Wikidata.",
    )
    gen.add_argument("-o", "--output", type=Path,
                     default=Path.home() / ".local/share/photo-tools/landmarks.json",
                     help="Output path for landmarks.json")
    gen.add_argument("-l", "--limit", type=int, default=20000,
                     help="Max landmarks to fetch from Wikidata (default: 20000)")
    gen.add_argument("--clip-model", default=None,
                     help="CLIP model name (default: from config)")
    gen.add_argument("--clip-pretrained", default=None,
                     help="CLIP pretrained weights (default: from config)")
    gen.add_argument("--resume", action="store_true",
                     help="Skip landmarks already in output file")
    gen.add_argument("--wikidata-cache", type=Path, metavar="PATH",
                     help="Load landmarks from a previous .wikidata.json instead of querying")
    gen.add_argument("--test", action="store_true",
                     help="Build a small database (~200 landmarks in Rome and Bologna)")
    gen.add_argument("--images-per-landmark", type=int, default=None,
                     help="Target number of images per landmark (default: from config)")
    gen.set_defaults(func=run_generate_db)

    q = lm_sub.add_parser(
        "query",
        help="Query the landmark DB with an image and print the top matches.",
    )
    q.add_argument("image", type=Path, help="Path to an image to query")
    q.add_argument("-k", "--top-k", type=int, default=10,
                   help="Number of top matches to print (default: 10)")
    q.add_argument("--landmarks-db", type=Path, default=None, dest="landmarks_db",
                   help="Path to landmarks.json (default: from config)")
    q.add_argument("--clip-model", default=None,
                   help="CLIP model name (default: from config)")
    q.add_argument("--clip-pretrained", default=None,
                   help="CLIP pretrained weights (default: from config)")
    q.add_argument("--radius-km", type=float, default=None,
                   help="GPS radius filter (default: from config; ignored if image has no GPS)")
    q.add_argument("--no-gps", action="store_true",
                   help="Ignore image GPS and search globally (useful for debugging)")
    q.set_defaults(func=run_query)

    return lm_parser


def run_generate_db(args) -> None:
    """Execute the landmarks generate-db subcommand."""
    cfg = get_config()
    build_database(
        limit=args.limit,
        output_path=args.output,
        clip_model=args.clip_model or cfg.clip.model,
        clip_pretrained=args.clip_pretrained or cfg.clip.pretrained,
        resume=args.resume,
        wikidata_cache=args.wikidata_cache,
        test=args.test,
        images_per_landmark=args.images_per_landmark,
    )


def run_query(args) -> None:
    """Embed an image, then print the top-K nearest landmarks from the DB."""
    import os as _os

    from photo_tools.autotag import get_gps_coords
    from photo_tools.clip_tagger import CLIPEmbedder
    from photo_tools.helpers import prepare_image, read_exif
    from photo_tools.landmarks import LandmarkIndex, _haversine_km

    cfg = get_config()
    image = args.image
    if not image.is_file():
        log.error("Image not found: %s", image)
        sys.exit(1)

    lm_path = args.landmarks_db or Path(cfg.landmarks.default_path).expanduser()
    if not lm_path.exists():
        log.error("Landmarks DB not found at %s", lm_path)
        sys.exit(1)

    clip_model = args.clip_model or cfg.clip.model
    clip_pretrained = args.clip_pretrained or cfg.clip.pretrained
    embedder = CLIPEmbedder(model_name=clip_model, pretrained=clip_pretrained)

    index = LandmarkIndex(lm_path)
    if index.model_id != f"{clip_model}/{clip_pretrained}":
        log.warning("DB built with %s but querying with %s/%s — results may be poor",
                    index.model_id, clip_model, clip_pretrained)

    prepared = prepare_image(image, cfg.clip.max_pixels)
    visual_input = prepared or image
    try:
        embedding = embedder.embed_image(visual_input)
    finally:
        if prepared:
            try:
                _os.unlink(prepared)
            except OSError:
                pass

    coords = None if args.no_gps else get_gps_coords(read_exif(image))

    if coords is not None and not args.no_gps:
        radius_km = args.radius_km if args.radius_km is not None else cfg.landmarks.radius_km
        lat, lon = coords
        log.info("Image GPS: %.5f, %.5f  — searching within %.0f km", lat, lon, radius_km)
        _, top, threshold = index.lookup(
            embedding, lat=lat, lon=lon,
            radius_km=radius_km, threshold=cfg.landmarks.threshold,
        )
        if not top:
            print(f"No landmarks within {radius_km:.0f} km of the image.")
            return
        print(f"Top {min(args.top_k, len(top))} matches "
              f"(threshold {threshold:.3f}):")
        for name, score in top[:args.top_k]:
            mark = " *" if score >= threshold else "  "
            print(f"  {mark} {score:.3f}  {name}")
    else:
        # Global search: cosine similarity against every landmark, no FAISS shortcut
        # because we want a single global ranking.
        import numpy as _np
        scores = index._embeddings @ embedding.astype(_np.float32)
        order = _np.argsort(-scores)[:args.top_k]
        threshold = cfg.landmarks.threshold
        print(f"Global top {len(order)} (no GPS filter, threshold {threshold:.3f}):")
        for i in order:
            name = index.names[i]
            s = float(scores[i])
            dist = ""
            if coords is not None:
                dist_km = _haversine_km(coords[0], coords[1],
                                        index.lats[i], index.lons[i])
                dist = f"  ({dist_km:.0f} km away)"
            mark = " *" if s >= threshold else "  "
            print(f"  {mark} {s:.3f}  {name}{dist}")

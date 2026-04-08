#!/usr/bin/env python3
"""
build_landmarks.py — Build a CLIP embedding database of notable landmarks.

Queries Wikidata for top landmarks (by sitelinks count), downloads one
Wikimedia Commons image per landmark, computes CLIP embeddings, and
outputs landmarks.json for use by LandmarkIndex.

Usage:
    uv run build_landmarks.py                        # Build with defaults
    uv run build_landmarks.py --limit 5000           # Fewer landmarks
    uv run build_landmarks.py --output landmarks.json
    uv run build_landmarks.py --resume               # Skip already-computed
"""

import argparse
import io
import json
import logging
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_landmarks")

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Per-type query — uses direct P31 only (no recursive P279* subclass traversal)
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

LANDMARK_TYPES = [
    # (qid, label, per-type limit)
    ("Q570116", "tourist attraction", 2000),
    ("Q839954", "archaeological site", 1500),
    ("Q4989906", "monument", 2000),
    ("Q811979", "architectural structure", 2000),
    ("Q35112127", "natural landmark", 500),
    ("Q2319498", "landmark", 500),
    ("Q751876", "castle", 1500),
    ("Q16970", "church building", 1500),
    ("Q32815", "mosque", 1500),
    ("Q44539", "temple", 1500),
    ("Q23413", "palace", 1500),
    ("Q3947", "house", 2000),
    ("Q12280", "bridge", 1500),
    ("Q8502", "mountain", 1500),
    ("Q23397", "lake", 1500),
    ("Q34038", "waterfall", 500),
    ("Q133056", "cave", 500),
    ("Q33506", "museum", 1500),
    ("Q483110", "stadium", 1500),
]

MAX_RETRIES = 3
RETRY_BACKOFF = 30  # seconds


def _sparql_get(query: str) -> list[dict]:
    """Execute a SPARQL query with retry on timeout/5xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as e:
            if attempt == MAX_RETRIES:
                raise
            retry_after = int(getattr(getattr(e, "response", None), "headers", {}).get("Retry-After", 0))
            wait = retry_after if retry_after else RETRY_BACKOFF * attempt
            log.warning("Query failed (%s), retrying in %ds (%d/%d)", e, wait, attempt, MAX_RETRIES)
            time.sleep(wait)
    return []  # unreachable, keeps type checkers happy


def query_wikidata(limit: int) -> list[dict]:
    """Query Wikidata SPARQL for notable landmarks, one type at a time."""
    log.info("Querying Wikidata for top %d landmarks across %d types ...", limit, len(LANDMARK_TYPES))

    seen_ids: dict[str, dict] = {}
    for qid, label, type_limit in LANDMARK_TYPES:
        log.info("  Querying %s (%s, limit %d) ...", label, qid, type_limit)
        query = SPARQL_TYPE_QUERY.format(qid=qid, limit=type_limit)
        results = _sparql_get(query)
        log.info("  Got %d results for %s", len(results), label)

        for r in results:
            wikidata_id = r["item"]["value"].rsplit("/", 1)[-1]
            if wikidata_id in seen_ids:
                continue
            seen_ids[wikidata_id] = {
                "name": r["itemLabel"]["value"],
                "wikidata_id": wikidata_id,
                "lat": float(r["lat"]["value"]),
                "lon": float(r["lon"]["value"]),
                "image_url": r["image"]["value"],
                "_sitelinks": int(r["sitelinks"]["value"]),
            }
        # Be kind to the endpoint between type queries
        time.sleep(2)

    # Sort by sitelinks descending and take top `limit`
    landmarks = sorted(seen_ids.values(), key=lambda x: x["_sitelinks"], reverse=True)[:limit]
    for lm in landmarks:
        del lm["_sitelinks"]

    log.info("Deduplicated to %d unique landmarks (top %d kept)", len(landmarks), limit)
    return landmarks


IMAGE_CACHE_DIR = Path.home() / ".cache/photo-tools/landmark-images"
DOWNLOAD_WORKERS = 2


_NO_THUMB_EXTS = {".tiff", ".tif", ".djvu"}


def _fetch(url: str, wikidata_id: str, filename: str, timeout: int = 30) -> requests.Response | None:
    """GET *url*; retry once on 429 using Retry-After, skip on other errors."""
    for _ in range(2):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
                timeout=timeout,
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


def _save_stream(resp: requests.Response, dest: Path, wikidata_id: str, filename: str, min_size: int = 1000) -> bool:
    """Stream response to *dest*, returning True on success."""
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    size = tmp.stat().st_size
    if size < min_size:
        log.warning("Downloaded image too small for %s (%s): %d bytes", wikidata_id, filename, size)
        tmp.unlink()
        return False
    tmp.rename(dest)
    return True


def download_image(url: str, wikidata_id: str, max_width: int = 512) -> tuple[Path | None, bool]:
    """Download a Wikimedia Commons image, resized via thumbnail API.

    For formats the thumbnail API cannot handle (TIFF, DjVu, …), the
    full original is downloaded and resized locally with Pillow.

    Images are cached in ~/.cache/photo-tools/landmark-images/ by Wikidata ID.
    """
    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    ext = Path(filename).suffix.lower()
    needs_local_resize = ext in _NO_THUMB_EXTS

    # Cache as .jpg when we resize locally (original format not useful downstream)
    cache_suffix = ".jpg" if needs_local_resize else (ext or ".jpg")
    cached = IMAGE_CACHE_DIR / f"{wikidata_id}{cache_suffix}"
    if cached.exists() and cached.stat().st_size >= 1000:
        return cached, True

    if needs_local_resize:
        # Download the full original, then resize & convert to JPEG
        resp = _fetch(url, wikidata_id, filename, timeout=120)
        if resp is None:
            return None, False
        try:
            img = Image.open(io.BytesIO(resp.content))
            img.thumbnail((max_width, max_width))
            img = img.convert("RGB")
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            img.save(tmp, "JPEG", quality=85)
            size = tmp.stat().st_size
            if size < 1000:
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
) -> None:
    from clip_tagger import CLIPTagger

    # Load existing data if resuming
    existing: dict[str, dict] = {}
    if resume and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        for lm in data.get("landmarks", []):
            existing[lm["wikidata_id"]] = lm
        log.info("Resuming: %d landmarks already computed", len(existing))

    if wikidata_cache and wikidata_cache.exists():
        with open(wikidata_cache) as f:
            landmarks = json.load(f)[:limit]
        log.info("Loaded %d landmarks from %s", len(landmarks), wikidata_cache)
    else:
        landmarks = query_wikidata(limit)
        # Dump deduped Wikidata entries for future re-use
        dump_path = output_path.with_suffix(".wikidata.json")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(landmarks, f, indent=2)
        log.info("Dumped %d Wikidata entries to %s", len(landmarks), dump_path)

    # Phase 1: download all images (2 workers)
    candidates = [lm for lm in landmarks if lm["wikidata_id"] not in existing or not existing[lm["wikidata_id"]].get("embedding")]
    download_failed = 0
    cached_count = 0
    downloaded_count = 0
    image_paths: dict[str, Path] = {}

    # Pre-scan disk cache so only truly missing images go to the thread pool
    need_download: list[dict] = []
    for lm in candidates:
        wid = lm["wikidata_id"]
        ext = Path(urllib.parse.unquote(lm["image_url"].rsplit("/", 1)[-1])).suffix.lower() or ".jpg"
        suffix = ".jpg" if ext in _NO_THUMB_EXTS else ext
        cached = IMAGE_CACHE_DIR / f"{wid}{suffix}"
        if cached.exists() and cached.stat().st_size >= 1000:
            image_paths[wid] = cached
            cached_count += 1
        else:
            need_download.append(lm)
    if cached_count:
        log.info("Images: %d already cached on disk, %d to download", cached_count, len(need_download))

    total = len(need_download)
    done_count = 0

    def _download(lm: dict) -> tuple[str, str, Path | None, bool]:
        wid = lm["wikidata_id"]
        img_path, was_cached = download_image(lm["image_url"], wid)
        return wid, lm["name"], img_path, was_cached

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(_download, lm) for lm in need_download]
        for future in as_completed(futures):
            wid, name, img_path, was_cached = future.result()
            done_count += 1
            if img_path is not None:
                image_paths[wid] = img_path
                if was_cached:
                    # Race: another run cached it between our scan and download
                    cached_count += 1
                else:
                    downloaded_count += 1
            else:
                download_failed += 1
            print(f"\r  images [{done_count}/{total}] {downloaded_count} new, {download_failed} failed — {name:<40}", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    log.info("Images: %d downloaded, %d cached, %d failed", downloaded_count, cached_count, download_failed)

    # Phase 2: embed
    tagger = CLIPTagger(model_name=clip_model, pretrained=clip_pretrained)

    results = []
    embed_failed = 0
    embed_total = len(landmarks)

    for i, lm in enumerate(landmarks, 1):
        wid = lm["wikidata_id"]

        if wid in existing and existing[wid].get("embedding"):
            results.append(existing[wid])
            print(f"\r  embedding [{i}/{embed_total}] {lm['name']} (cached)", end="", flush=True, file=sys.stderr)
            continue

        img_path = image_paths.get(wid)
        if img_path is None:
            continue

        print(f"\r  embedding [{i}/{embed_total}] {lm['name']:<50}", end="", flush=True, file=sys.stderr)

        try:
            _, embedding = tagger.tag_image(img_path)
            results.append({
                "name": lm["name"],
                "wikidata_id": wid,
                "lat": lm["lat"],
                "lon": lm["lon"],
                "embedding": embedding.tolist(),
            })
        except Exception as e:
            log.debug("Failed to embed %s: %s", lm["name"], e)
            embed_failed += 1

        # Periodic save every 500 landmarks
        if i % 500 == 0:
            _save(output_path, clip_model, clip_pretrained, results)

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


def main():
    parser = argparse.ArgumentParser(description="Build landmark CLIP embedding database")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path.home() / ".local/share/photo-tools/landmarks.json")
    parser.add_argument("-l", "--limit", type=int, default=20000,
                        help="Max landmarks to fetch from Wikidata (default: 20000)")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--resume", action="store_true",
                        help="Skip landmarks already in output file")
    parser.add_argument("--wikidata-cache", type=Path, metavar="PATH",
                        help="Load landmarks from a previous .wikidata.json instead of querying")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    build_database(
        limit=args.limit,
        output_path=args.output,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        resume=args.resume,
        wikidata_cache=args.wikidata_cache,
    )


if __name__ == "__main__":
    main()

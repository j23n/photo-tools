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
import json
import logging
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

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
MIN_DELAY = 0.05   # seconds between requests (floor)
MAX_DELAY = 10.0   # seconds between requests (ceiling)
INITIAL_DELAY = 0.1
DOWNLOAD_WORKERS = 2
_rate_delay = INITIAL_DELAY
_rate_lock = threading.Lock()


def _adjust_rate(retry_after: float = 0) -> None:
    """Adjust download delay based on rate limit feedback."""
    global _rate_delay
    with _rate_lock:
        if retry_after:
            _rate_delay = max(retry_after, _rate_delay)
            log.warning("Rate limited — backing off to %.1fs between requests", _rate_delay)
        else:
            _rate_delay = max(_rate_delay * 0.95, MIN_DELAY)


def download_image(url: str, wikidata_id: str, max_width: int = 512) -> tuple[Path | None, bool]:
    """Download a Wikimedia Commons image, resized via thumbnail API.

    Images are cached in ~/.cache/photo-tools/landmark-images/ by Wikidata ID.
    """
    suffix = Path(urllib.parse.unquote(url.rsplit("/", 1)[-1])).suffix or ".jpg"
    cached = IMAGE_CACHE_DIR / f"{wikidata_id}{suffix}"
    if cached.exists() and cached.stat().st_size >= 1000:
        return cached, True

    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    thumb_url = f"https://commons.wikimedia.org/w/thumb.php?f={urllib.parse.quote(filename)}&w={max_width}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(_rate_delay)
            resp = requests.get(
                thumb_url,
                headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
                timeout=30,
                stream=True,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                _adjust_rate(retry_after)
                time.sleep(retry_after)
                continue
            _adjust_rate()
            resp.raise_for_status()
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            if tmp.stat().st_size < 1000:
                tmp.unlink()
                return None, False
            tmp.rename(cached)
            return cached, False
        except Exception as e:
            log.debug("Failed to download %s (attempt %d): %s", filename, attempt + 1, e)
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
        suffix = Path(urllib.parse.unquote(lm["image_url"].rsplit("/", 1)[-1])).suffix or ".jpg"
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

    for i, lm in enumerate(landmarks, 1):
        wid = lm["wikidata_id"]

        if wid in existing and existing[wid].get("embedding"):
            results.append(existing[wid])
            print(f"\r  embedding [{i}/{total}] {lm['name']} (cached)", end="", flush=True, file=sys.stderr)
            continue

        img_path = image_paths.get(wid)
        if img_path is None:
            continue

        print(f"\r  embedding [{i}/{total}] {lm['name']:<50}", end="", flush=True, file=sys.stderr)

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

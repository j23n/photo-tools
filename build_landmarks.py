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
import tempfile
import time
import urllib.parse
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

# Query for notable landmarks with coordinates and images
SPARQL_QUERY = """\
SELECT ?item ?itemLabel ?lat ?lon ?image ?sitelinks WHERE {{
  ?item wdt:P31/wdt:P279* ?type .
  VALUES ?type {{
    wd:Q570116    # tourist attraction
    wd:Q839954    # archaeological site
    wd:Q4989906   # monument
    wd:Q811979    # architectural structure
    wd:Q35112127  # natural landmark
    wd:Q2319498   # landmark
    wd:Q751876    # castle
    wd:Q16970     # church building
    wd:Q32815     # mosque
    wd:Q44539     # temple
    wd:Q23413     # palace
    wd:Q3947      # house
    wd:Q12280     # bridge
    wd:Q8502      # mountain
    wd:Q23397     # lake
    wd:Q34038     # waterfall
    wd:Q133056    # cave
    wd:Q33506     # museum
    wd:Q483110    # stadium
  }}
  ?item wdt:P625 ?coords .
  ?item wdt:P18 ?image .
  ?item wikibase:sitelinks ?sitelinks .
  BIND(geof:latitude(?coords) AS ?lat)
  BIND(geof:longitude(?coords) AS ?lon)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY DESC(?sitelinks)
LIMIT {limit}
"""


def query_wikidata(limit: int) -> list[dict]:
    """Query Wikidata SPARQL for notable landmarks."""
    log.info("Querying Wikidata for top %d landmarks ...", limit)
    query = SPARQL_QUERY.format(limit=limit)
    resp = requests.get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
        timeout=120,
    )
    resp.raise_for_status()
    results = resp.json()["results"]["bindings"]
    log.info("Got %d results from Wikidata", len(results))

    landmarks = []
    seen_ids = set()
    for r in results:
        wikidata_id = r["item"]["value"].rsplit("/", 1)[-1]
        if wikidata_id in seen_ids:
            continue
        seen_ids.add(wikidata_id)
        landmarks.append({
            "name": r["itemLabel"]["value"],
            "wikidata_id": wikidata_id,
            "lat": float(r["lat"]["value"]),
            "lon": float(r["lon"]["value"]),
            "image_url": r["image"]["value"],
        })
    log.info("Deduplicated to %d unique landmarks", len(landmarks))
    return landmarks


def download_image(url: str, max_width: int = 512) -> Path | None:
    """Download a Wikimedia Commons image, resized via thumbnail API."""
    # Use Wikimedia thumbnail API to get a resized version
    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    thumb_url = f"https://commons.wikimedia.org/w/thumb.php?f={urllib.parse.quote(filename)}&w={max_width}"

    try:
        resp = requests.get(
            thumb_url,
            headers={"User-Agent": "photo-tools-landmark-builder/1.0"},
            timeout=30,
            stream=True,
        )
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        for chunk in resp.iter_content(8192):
            tmp.write(chunk)
        tmp.close()
        if Path(tmp.name).stat().st_size < 1000:
            Path(tmp.name).unlink()
            return None
        return Path(tmp.name)
    except Exception as e:
        log.debug("Failed to download %s: %s", filename, e)
        return None


def build_database(
    limit: int,
    output_path: Path,
    clip_model: str,
    clip_pretrained: str,
    resume: bool,
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

    landmarks = query_wikidata(limit)
    tagger = CLIPTagger(model_name=clip_model, pretrained=clip_pretrained)

    results = []
    total = len(landmarks)
    failed = 0

    for i, lm in enumerate(landmarks, 1):
        wid = lm["wikidata_id"]

        # Resume: reuse existing embedding
        if wid in existing and existing[wid].get("embedding"):
            results.append(existing[wid])
            print(f"\r  [{i}/{total}] {lm['name']} (cached)", end="", flush=True, file=sys.stderr)
            continue

        print(f"\r  [{i}/{total}] {lm['name']:<50}", end="", flush=True, file=sys.stderr)

        img_path = download_image(lm["image_url"])
        if img_path is None:
            failed += 1
            continue

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
            failed += 1
        finally:
            try:
                img_path.unlink()
            except OSError:
                pass

        # Rate limit to be kind to Wikimedia
        time.sleep(0.2)

        # Periodic save every 500 landmarks
        if i % 500 == 0:
            _save(output_path, clip_model, clip_pretrained, results)

    print(file=sys.stderr)
    _save(output_path, clip_model, clip_pretrained, results)
    log.info("Done. %d landmarks saved, %d failed", len(results), failed)


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
    )


if __name__ == "__main__":
    main()

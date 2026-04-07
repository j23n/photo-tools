"""
duplicates.py — Tag deduplication and similar-image detection for photo-tools.

Subcommands:
    dedup-tags   Find and interactively merge/delete duplicate/synonym tags
    find-similar Find visually similar images using CLIP embeddings
"""

import argparse
import base64
import difflib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from autotag import (
    find_images,
    get_existing_keywords,
    prepare_image,
    read_exif,
    write_keywords,
    IMAGE_EXTENSIONS,
)

log = logging.getLogger("duplicates")

DEFAULT_EMBED_BASE_URL = os.environ.get(
    "EMBED_BASE_URL", os.environ.get("AI_BASE_URL", "http://100.64.0.4:8000/v1")
)
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "clip")
DEFAULT_EMBED_API_KEY = os.environ.get("EMBED_API_KEY", os.environ.get("AI_API_KEY", ""))


# ---------------------------------------------------------------------------
# Embed client
# ---------------------------------------------------------------------------

class EmbedClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def embed(self, path: Path) -> "np.ndarray | None":
        prepared = prepare_image(path, 512)
        if prepared is None:
            log.warning("Could not prepare image for embedding: %s", path.name)
            return None
        try:
            with open(prepared, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(prepared)
        except Exception as e:
            log.warning("Failed to read prepared image %s: %s", path.name, e)
            try:
                os.unlink(prepared)
            except OSError:
                pass
            return None

        payload = {
            "model": self.model,
            "input": f"data:image/jpeg;base64,{b64_data}",
        }
        try:
            resp = self.session.post(
                f"{self.base_url}/embeddings", json=payload, timeout=60
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            log.error("Cannot connect to %s", self.base_url)
            return None
        except requests.Timeout:
            log.error("Timed out embedding %s", path.name)
            return None
        except requests.HTTPError as e:
            log.error("API error for %s: %s", path.name, e.response.text[:200])
            return None

        data = resp.json().get("data", [])
        if not data:
            log.error("Empty embedding response for %s", path.name)
            return None

        vec = np.array(data[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ---------------------------------------------------------------------------
# XMP embedding cache
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


def embed_all(
    paths: list[Path],
    client: EmbedClient,
    model: str,
    force: bool,
    dry_run: bool,
) -> "dict[Path, np.ndarray]":
    result = {}
    total = len(paths)
    for i, path in enumerate(paths, 1):
        print(f"\r  Embedding {i}/{total} ...", end="", flush=True, file=sys.stderr)
        if not force:
            cached = read_cached_embedding(path, model)
            if cached is not None:
                result[path] = cached
                continue
        vec = client.embed(path)
        if vec is not None:
            write_embedding(path, vec, model, dry_run)
            result[path] = vec
        else:
            log.warning("Failed to embed %s", path.name)
    print(file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# dedup-tags helpers
# ---------------------------------------------------------------------------

def collect_tag_index(paths: list[Path]) -> "dict[str, list[Path]]":
    index: dict[str, list[Path]] = {}
    for path in paths:
        exif = read_exif(path)
        for tag in get_existing_keywords(exif):
            index.setdefault(tag, []).append(path)
    # Sort by file count descending
    return dict(sorted(index.items(), key=lambda kv: len(kv[1]), reverse=True))


def _tag_parts(tag: str) -> tuple[str, str]:
    """Split 'prefix:value' into ('prefix', 'value'), or ('', tag) if no prefix."""
    if ":" in tag:
        prefix, _, value = tag.partition(":")
        return prefix, value
    return "", tag


def find_string_candidates(tags: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    used = set()
    for i, a in enumerate(tags):
        if a in used:
            continue
        a_prefix, a_value = _tag_parts(a)
        group = [a]
        for b in tags[i + 1:]:
            if b in used:
                continue
            b_prefix, b_value = _tag_parts(b)
            # When both tags share the same prefix, compare only the value parts.
            # This prevents the shared prefix from inflating similarity scores
            # (e.g. setting:outdoor vs setting:indoor score 0.61 on values alone,
            # but 0.90 on the full strings — the former is correct).
            if a_prefix and a_prefix == b_prefix:
                cmp_a, cmp_b = a_value, b_value
            else:
                cmp_a, cmp_b = a, b
            ratio = difflib.SequenceMatcher(None, cmp_a, cmp_b).ratio()
            if ratio >= 0.82:
                group.append(b)
                continue
            # Plural forms
            if cmp_b == cmp_a + "s" or cmp_b == cmp_a + "es" or cmp_a == cmp_b + "s" or cmp_a == cmp_b + "es":
                group.append(b)
        if len(group) >= 2:
            for t in group[1:]:
                used.add(t)
            used.add(a)
            groups.append(group)
    return groups


def find_llm_candidates(
    tag_index: "dict[str, list[Path]]",
    base_url: str,
    api_key: str,
    model: str,
) -> list[list[str]]:
    tag_list = ", ".join(
        f"{tag} ({len(files)})" for tag, files in list(tag_index.items())[:500]
    )
    prompt = (
        "You are a photo library tag curator. Identify groups of tags that are "
        "synonyms, duplicates, or near-duplicates (including cross-prefix overlaps).\n\n"
        f"Tags with counts: {tag_list}\n\n"
        "Return ONLY a JSON array of arrays, where each inner array contains tags "
        "that should be merged. Only include groups with 2+ tags. "
        "Example: [[\"scene:beach\", \"scene:beaches\", \"scene:shore\"], [\"animal:dog\", \"animal:dogs\"]]"
    )

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    try:
        resp = session.post(f"{base_url}/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        log.error("LLM call failed: %s", e)
        return []

    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    start = content.find("[")
    if start < 0:
        return []
    try:
        parsed = json.loads(content[start:])
        return [g for g in parsed if isinstance(g, list) and len(g) >= 2]
    except json.JSONDecodeError:
        log.warning("Could not parse LLM response for tag candidates")
        return []


def merge_candidate_groups(groups_a: list[list[str]], groups_b: list[list[str]]) -> list[list[str]]:
    """Merge two lists of groups, deduplicating overlapping groups."""
    merged = list(groups_a)
    existing_tags: set[str] = {t for g in groups_a for t in g}
    for group in groups_b:
        overlap = [t for t in group if t in existing_tags]
        if overlap:
            # Extend an existing group that shares tags
            for existing in merged:
                if any(t in existing for t in group):
                    for t in group:
                        if t not in existing:
                            existing.append(t)
                    break
        else:
            merged.append(group)
            for t in group:
                existing_tags.add(t)
    return merged


def apply_tag_change(
    old: str,
    new: "str | None",
    files: list[Path],
    dry_run: bool,
) -> None:
    if not files:
        return
    if dry_run:
        action = f"rename to '{new}'" if new else "delete"
        log.info("[DRY RUN] Would %s '%s' in %d file(s)", action, old, len(files))
        return

    # Batch remove old tag from all files
    args = ["exiftool", "-overwrite_original"]
    args.append(f"-IPTC:Keywords-={old}")
    args.append(f"-XMP:Subject-={old}")
    args.extend(str(f) for f in files)
    subprocess.run(args, capture_output=True, timeout=120)

    # Add new tag if renaming
    if new:
        for path in files:
            write_keywords(path, [new], dry_run=False)


def interactive_dedup_session(
    groups: list[list[str]],
    tag_index: "dict[str, list[Path]]",
    dry_run: bool,
) -> None:
    total = len(groups)
    for i, group in enumerate(groups, 1):
        counts = {tag: len(tag_index.get(tag, [])) for tag in group}
        print(f"\nGroup {i}/{total}:")
        for j, tag in enumerate(group):
            print(f"  [{j}] {tag}  ({counts[tag]} files)")

        print("  [m] merge all into first  [d] delete all but first  [k] keep all  [s] skip")
        while True:
            choice = input("  > ").strip().lower()
            if choice in ("s", "skip", ""):
                break
            elif choice in ("k", "keep"):
                break
            elif choice in ("m", "merge"):
                canonical = group[0]
                for tag in group[1:]:
                    apply_tag_change(tag, canonical, tag_index.get(tag, []), dry_run)
                print(f"  Merged into '{canonical}'")
                break
            elif choice in ("d", "delete"):
                for tag in group[1:]:
                    apply_tag_change(tag, None, tag_index.get(tag, []), dry_run)
                print(f"  Deleted all but '{group[0]}'")
                break
            else:
                print("  Unknown choice. Use m/d/k/s")


# ---------------------------------------------------------------------------
# find-similar helpers
# ---------------------------------------------------------------------------

def cluster_similar(
    embeddings: "dict[Path, np.ndarray]",
    threshold: float,
) -> list[list[Path]]:
    paths = list(embeddings.keys())
    if not paths:
        return []
    matrix = np.stack([embeddings[p] for p in paths])
    sim = matrix @ matrix.T  # cosine similarity (vectors are L2-normalized)

    # Build adjacency using union-find
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    result = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        # Sort by file size descending (largest = likely highest quality)
        cluster_paths = sorted(
            [paths[i] for i in indices],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        result.append(cluster_paths)

    # Sort clusters by max intra-cluster similarity descending
    def max_sim(cluster):
        idxs = [paths.index(p) for p in cluster]
        sims = [sim[idxs[a], idxs[b]] for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        return max(sims) if sims else 0.0

    result.sort(key=max_sim, reverse=True)
    return result


def move_to_dest(path: Path, dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / path.name
    counter = 2
    while dest.exists():
        dest = dest_root / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    path.rename(dest)
    log.info("Moved %s -> %s", path.name, dest)


def interactive_similar_session(
    clusters: list[list[Path]],
    dest_root: Path,
    dry_run: bool,
    embeddings: "dict[Path, np.ndarray]",
) -> None:
    total = len(clusters)
    for i, cluster in enumerate(clusters, 1):
        paths_list = list(cluster)
        matrix = np.stack([embeddings[p] for p in paths_list])
        sim_matrix = matrix @ matrix.T

        print(f"\nCluster {i}/{total}  ({len(cluster)} photos)")
        for j, path in enumerate(paths_list):
            size_mb = path.stat().st_size / 1_048_576
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            if j == 0:
                print(f"  [KEEP] [{j}] {path.name}  {mtime}  {size_mb:.1f} MB")
            else:
                max_sim = max(sim_matrix[j, k] for k in range(j))
                print(f"         [{j}] {path.name}  sim={max_sim:.3f}  {size_mb:.1f} MB")

        print("  [k]eep all  [d]move all but first  [s]elect indices to keep  [skip]")
        while True:
            choice = input("  > ").strip().lower()
            if choice in ("skip", ""):
                break
            elif choice == "k":
                break
            elif choice == "d":
                for path in paths_list[1:]:
                    if dry_run:
                        log.info("[DRY RUN] Would move %s -> %s", path.name, dest_root)
                    else:
                        move_to_dest(path, dest_root)
                break
            elif choice == "s":
                raw = input("  Keep indices (space-separated): ").strip()
                try:
                    keep = {int(x) for x in raw.split()}
                except ValueError:
                    print("  Invalid input, skipping.")
                    break
                for j, path in enumerate(paths_list):
                    if j not in keep:
                        if dry_run:
                            log.info("[DRY RUN] Would move %s -> %s", path.name, dest_root)
                        else:
                            move_to_dest(path, dest_root)
                break
            else:
                print("  Unknown choice. Use k/d/s/skip")


# ---------------------------------------------------------------------------
# dedup-tags subcommand
# ---------------------------------------------------------------------------

def build_dedup_parser(subparsers) -> argparse.ArgumentParser:
    sub = subparsers.add_parser(
        "dedup-tags",
        help="Find and interactively merge/delete duplicate or synonym tags.",
    )
    sub.add_argument("path", type=Path, help="Target directory or file")
    sub.add_argument("-r", "--recursive", action="store_true")
    sub.add_argument("-n", "--dry-run", action="store_true",
                     help="Preview changes without writing")
    sub.add_argument("--no-llm", action="store_true",
                     help="Skip LLM synonym detection, use string similarity only")
    sub.add_argument("--delete-tag", metavar="TAG",
                     help="Non-interactive: delete this tag from all files")
    sub.add_argument("--rename-tag", nargs=2, metavar=("OLD", "NEW"),
                     help="Non-interactive: rename OLD tag to NEW")
    sub.add_argument("--base-url", default=None)
    sub.add_argument("--api-key", default=None)
    sub.add_argument("-m", "--model", default=None,
                     help="LLM model for synonym detection")
    sub.set_defaults(func=run_dedup_tags)
    return sub


def run_dedup_tags(args) -> None:
    paths = find_images(args.path, args.recursive)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Scanning tags in %d file(s) ...", len(paths))
    tag_index = collect_tag_index(paths)
    log.info("Found %d unique tags", len(tag_index))

    # Non-interactive shortcuts
    if args.delete_tag:
        tag = args.delete_tag.lower()
        apply_tag_change(tag, None, tag_index.get(tag, []), args.dry_run)
        return
    if args.rename_tag:
        old, new = args.rename_tag[0].lower(), args.rename_tag[1].lower()
        apply_tag_change(old, new, tag_index.get(old, []), args.dry_run)
        return

    tags = list(tag_index.keys())
    string_groups = find_string_candidates(tags)
    log.info("Found %d string-similar groups", len(string_groups))

    all_groups = string_groups

    if not args.no_llm:
        base_url = args.base_url or os.environ.get("AI_BASE_URL", "http://100.64.0.4:8000/v1")
        api_key = args.api_key or os.environ.get("AI_API_KEY", "")
        model = args.model or os.environ.get("AI_MODEL", "gemma4")
        log.info("Running LLM synonym detection ...")
        llm_groups = find_llm_candidates(tag_index, base_url, api_key, model)
        log.info("LLM found %d synonym groups", len(llm_groups))
        all_groups = merge_candidate_groups(string_groups, llm_groups)

    if not all_groups:
        log.info("No duplicate or synonym tags found.")
        return

    log.info("Starting interactive dedup session (%d groups) ...", len(all_groups))
    interactive_dedup_session(all_groups, tag_index, args.dry_run)


# ---------------------------------------------------------------------------
# find-similar subcommand
# ---------------------------------------------------------------------------

def build_similar_parser(subparsers) -> argparse.ArgumentParser:
    sub = subparsers.add_parser(
        "find-similar",
        help="Find visually similar images using CLIP embeddings.",
    )
    sub.add_argument("path", type=Path, help="Target directory or file")
    sub.add_argument("-r", "--recursive", action="store_true")
    sub.add_argument("-n", "--dry-run", action="store_true",
                     help="Preview moves without executing")
    sub.add_argument("--threshold", type=float, default=0.90,
                     help="Cosine similarity cutoff (default: 0.90)")
    sub.add_argument("--embed-only", action="store_true",
                     help="Compute and cache embeddings, then exit")
    sub.add_argument("--dest", type=Path, default=None,
                     help="Destination directory for moved duplicates")
    sub.add_argument("--force", action="store_true",
                     help="Re-embed even if a cached embedding exists")
    sub.add_argument("--base-url", default=None)
    sub.add_argument("--api-key", default=None)
    sub.add_argument("-m", "--model", default=None,
                     help="Embedding model name (default: $EMBED_MODEL or clip)")
    sub.set_defaults(func=run_find_similar)
    return sub


def run_find_similar(args) -> None:
    base_url = args.base_url or DEFAULT_EMBED_BASE_URL
    api_key = args.api_key or DEFAULT_EMBED_API_KEY
    model = args.model or DEFAULT_EMBED_MODEL

    all_files = find_images(args.path, args.recursive)
    # Filter to images only (exclude videos)
    image_paths = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Found %d image(s) to process", len(image_paths))
    client = EmbedClient(base_url=base_url, api_key=api_key, model=model)
    embeddings = embed_all(image_paths, client, model, args.force, args.dry_run)

    if args.embed_only:
        print(f"Embedded {len(embeddings)}/{len(image_paths)} images")
        return

    log.info("Clustering similar images (threshold=%.2f) ...", args.threshold)
    clusters = cluster_similar(embeddings, args.threshold)

    if not clusters:
        log.info("No similar image clusters found.")
        return

    log.info("Found %d cluster(s) of similar images", len(clusters))

    if args.dest:
        dest_root = args.dest
    else:
        # Default: _duplicates/ under the common ancestor of all images
        common = image_paths[0].parent
        for p in image_paths[1:]:
            while not str(p).startswith(str(common)):
                common = common.parent
        dest_root = common / "_duplicates"

    interactive_similar_session(clusters, dest_root, args.dry_run, embeddings)

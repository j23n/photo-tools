"""
duplicates.py — Similar-image detection and tag management for photo-tools.

Subcommands:
    find-similar Find visually similar images using CLIP embeddings cached in XMP
    tags         Tag management: list, search, delete, rename
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from autotag import (
    find_images,
    hierarchical_subject,
    prepare_image,
    read_cached_embedding,
    read_keywords_batch,
    write_embedding,
    write_keywords,
    CLIP_MAX_PIXELS,
    IMAGE_EXTENSIONS,
)

log = logging.getLogger("duplicates")


# ---------------------------------------------------------------------------
# Embedding loading (reads XMP cache, falls back to CLIPTagger)
# ---------------------------------------------------------------------------

def load_embeddings(
    paths: list[Path],
    model_id: str,
    force: bool,
    clip_model: str | None,
    clip_pretrained: str | None,
    dry_run: bool,
) -> "dict[Path, np.ndarray]":
    """Load CLIP embeddings for all paths. Reads from XMP cache first,
    falls back to computing via CLIPTagger for images without cached embeddings."""
    result = {}
    missing = []
    total = len(paths)

    # Phase 1: read cached embeddings
    for i, path in enumerate(paths, 1):
        print(f"\r  Reading embeddings {i}/{total} ...", end="", flush=True, file=sys.stderr)
        if not force:
            cached = read_cached_embedding(path, model_id)
            if cached is not None:
                result[path] = cached
                continue
        missing.append(path)
    print(file=sys.stderr)

    if not missing:
        return result

    # Phase 2: compute missing embeddings via CLIPTagger
    log.info("%d/%d images need embedding, loading CLIP model ...", len(missing), total)
    from clip_tagger import CLIPTagger, DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED
    tagger = CLIPTagger(
        model_name=clip_model or DEFAULT_CLIP_MODEL,
        pretrained=clip_pretrained or DEFAULT_CLIP_PRETRAINED,
    )

    for i, path in enumerate(missing, 1):
        print(f"\r  Embedding {i}/{len(missing)} ...", end="", flush=True, file=sys.stderr)
        prepared = prepare_image(path, CLIP_MAX_PIXELS)
        clip_input = prepared or path
        try:
            _, embedding = tagger.tag_image(clip_input)
            write_embedding(path, embedding, model_id, dry_run)
            result[path] = embedding
        except Exception as e:
            log.warning("Failed to embed %s: %s", path.name, e)
        finally:
            if prepared:
                try:
                    os.unlink(prepared)
                except OSError:
                    pass
    print(file=sys.stderr)
    return result


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
# tags subcommands (list, search, delete, rename)
# ---------------------------------------------------------------------------

def collect_tag_index(paths: list[Path]) -> "dict[str, list[Path]]":
    index: dict[str, list[Path]] = {}
    all_keywords = read_keywords_batch(paths)
    for path in paths:
        for tag in all_keywords.get(path, set()):
            index.setdefault(tag, []).append(path)
    return dict(sorted(index.items(), key=lambda kv: len(kv[1]), reverse=True))


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

    args = ["exiftool", "-overwrite_original"]
    args.append(f"-IPTC:Keywords-={old}")
    args.append(f"-XMP-dc:Subject-={old}")
    args.append(f"-XMP-lr:HierarchicalSubject-={hierarchical_subject(old)}")
    args.append(f"-XMP-digiKam:TagsList-={old}")
    args.extend(str(f) for f in files)
    subprocess.run(args, capture_output=True, timeout=120)

    if new:
        for path in files:
            write_keywords(path, [new], dry_run=False)


def bulk_delete_tags(
    tags: list[str],
    files: list[Path],
    dry_run: bool,
) -> None:
    """Delete multiple tags from multiple files in a single exiftool call per batch."""
    if not files or not tags:
        return
    if dry_run:
        log.info("[DRY RUN] Would delete %d tag(s) from %d file(s)", len(tags), len(files))
        return

    BATCH_SIZE = 500
    total = len(files)
    for i in range(0, total, BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        args = ["exiftool", "-overwrite_original"]
        for tag in tags:
            args.append(f"-IPTC:Keywords-={tag}")
            args.append(f"-XMP-dc:Subject-={tag}")
            args.append(f"-XMP-lr:HierarchicalSubject-={hierarchical_subject(tag)}")
            args.append(f"-XMP-digiKam:TagsList-={tag}")
        args.extend(str(f) for f in batch)
        print(f"\r  Deleting tags from files {i + 1}-{min(i + len(batch), total)}/{total} ...",
              end="", flush=True, file=sys.stderr)
        subprocess.run(args, capture_output=True, timeout=300)
    print(file=sys.stderr)


def run_list_tags(args) -> None:
    paths = find_images(args.path, args.recursive)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)
    if not tag_index:
        print("No tags found.")
        return

    max_tag_len = max(len(t) for t in tag_index)
    for tag, files in tag_index.items():
        print(f"  {tag:<{max_tag_len}}  {len(files):>4} files")
    print(f"\n{len(tag_index)} unique tags across {len(paths)} files")


def run_delete_tag(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not args.tag and not args.pattern:
        log.error("Provide a tag name or --pattern")
        sys.exit(1)

    paths = find_images(args.path, args.recursive)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)

    if args.pattern:
        import re
        try:
            regex = re.compile(args.pattern)
        except re.error as e:
            log.error("Invalid regex pattern: %s", e)
            sys.exit(1)
        matched = {tag: files for tag, files in tag_index.items() if regex.fullmatch(tag)}
        if not matched:
            log.error("No tags match pattern '%s'", args.pattern)
            sys.exit(1)
        matched_tags = sorted(matched.keys())
        affected_files = list({f for files in matched.values() for f in files})
        log.info("Pattern '%s' matched %d tag(s) across %d file(s):", args.pattern, len(matched_tags), len(affected_files))
        for tag in matched_tags:
            log.info("  %s  (%d files)", tag, len(matched[tag]))
        bulk_delete_tags(matched_tags, affected_files, args.dry_run)
        return

    tag = args.tag.lower()
    files = tag_index.get(tag, [])
    if not files:
        log.error("Tag '%s' not found in any files", tag)
        sys.exit(1)

    log.info("Deleting tag '%s' from %d file(s) ...", tag, len(files))
    bulk_delete_tags([tag], files, args.dry_run)


def run_rename_tag(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    paths = find_images(args.path, args.recursive)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)
    old = args.old.lower()
    new = args.new.lower()
    files = tag_index.get(old, [])
    if not files:
        log.error("Tag '%s' not found in any files", old)
        sys.exit(1)

    log.info("Renaming tag '%s' -> '%s' in %d file(s) ...", old, new, len(files))
    apply_tag_change(old, new, files, args.dry_run)


def run_search_tags(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    paths = find_images(args.path, args.recursive)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)
    query = args.tag.lower()
    files = tag_index.get(query, [])
    if not files:
        log.error("Tag '%s' not found", query)
        sys.exit(1)

    for f in sorted(files):
        print(f)
    print(f"\n{len(files)} file(s) with tag '{query}'")


# ---------------------------------------------------------------------------
# CLI parsers
# ---------------------------------------------------------------------------

def build_tags_parser(subparsers) -> None:
    """Register the 'tags' subcommand group with list/delete/rename/search."""
    tags_parser = subparsers.add_parser(
        "tags",
        help="Tag management: list, search, delete, rename.",
    )
    tags_sub = tags_parser.add_subparsers(dest="tags_command", required=True)

    # tags list
    p = tags_sub.add_parser("list", help="List all tags with file counts.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=run_list_tags)

    # tags search
    p = tags_sub.add_parser("search", help="List files that have a given tag.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("tag", help="Tag to search for")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=run_search_tags)

    # tags delete
    p = tags_sub.add_parser("delete", help="Delete a tag (or tags matching a regex pattern) from all files.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("tag", nargs="?", help="Exact tag to delete")
    p.add_argument("-p", "--pattern", help="Regex pattern to match tags for deletion")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Preview changes without writing")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=run_delete_tag)

    # tags rename
    p = tags_sub.add_parser("rename", help="Rename a tag across all files.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("old", help="Tag to rename")
    p.add_argument("new", help="New tag name")
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Preview changes without writing")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=run_rename_tag)


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
    sub.add_argument("--dest", type=Path, default=None,
                     help="Destination directory for moved duplicates")
    sub.add_argument("--force", action="store_true",
                     help="Re-embed even if a cached embedding exists")
    sub.add_argument("--clip-model", default=None,
                     help="CLIP model name (default: ViT-B-32)")
    sub.add_argument("--clip-pretrained", default=None,
                     help="CLIP pretrained weights (default: laion2b_s34b_b79k)")
    sub.add_argument("-v", "--verbose", action="store_true")
    sub.set_defaults(func=run_find_similar)
    return sub


def run_find_similar(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from clip_tagger import DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED
    clip_model = args.clip_model or DEFAULT_CLIP_MODEL
    clip_pretrained = args.clip_pretrained or DEFAULT_CLIP_PRETRAINED
    model_id = f"{clip_model}/{clip_pretrained}"

    all_files = find_images(args.path, args.recursive)
    image_paths = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Found %d image(s) to process", len(image_paths))
    embeddings = load_embeddings(
        image_paths, model_id, args.force,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dry_run=args.dry_run,
    )

    log.info("Clustering similar images (threshold=%.2f) ...", args.threshold)
    clusters = cluster_similar(embeddings, args.threshold)

    if not clusters:
        log.info("No similar image clusters found.")
        return

    log.info("Found %d cluster(s) of similar images", len(clusters))

    if args.dest:
        dest_root = args.dest
    else:
        common = image_paths[0].parent
        for p in image_paths[1:]:
            while not str(p).startswith(str(common)):
                common = common.parent
        dest_root = common / "_duplicates"

    interactive_similar_session(clusters, dest_root, args.dry_run, embeddings)

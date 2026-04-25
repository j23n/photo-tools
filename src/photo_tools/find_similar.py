"""
find_similar.py — Similar-image detection with an interactive dedup session.

Wires the `find-similar` CLI subcommand: load CLIP embeddings (cached in
XMP, fall back to computing), cluster by cosine similarity, then offer a
terminal UI for reviewing each cluster and moving non-kept images aside.
"""

import argparse
import logging
import math
import os
import sys
import tempfile
import termios
import tty
from datetime import datetime
from pathlib import Path

import numpy as np

from photo_tools.config import get_config
from photo_tools.constants import IMAGE_EXTENSIONS
from photo_tools.debug_viewer import _find_opener, _kill_viewer, _read_key, _truncate
from photo_tools.helpers import (
    find_images,
    open_and_rotate,
    prepare_image,
    read_cached_embeddings_batch,
    write_embedding,
)

log = logging.getLogger("find_similar")


# ---------------------------------------------------------------------------
# Embedding loading (reads XMP cache, falls back to CLIPEmbedder)
# ---------------------------------------------------------------------------

def load_embeddings(
    paths: list[Path],
    model_id: str,
    force: bool,
    clip_model: str | None,
    clip_pretrained: str | None,
    dry_run: bool,
) -> dict[Path, np.ndarray]:
    """Load CLIP embeddings for all paths. Reads from XMP cache first,
    falls back to computing via CLIPEmbedder for images without cached embeddings."""
    cfg = get_config()
    result: dict[Path, np.ndarray] = {}
    total = len(paths)

    if force:
        missing = list(paths)
    else:
        result.update(read_cached_embeddings_batch(paths, model_id))
        missing = [p for p in paths if p not in result]
        log.info("Loaded %d/%d embeddings from cache", len(result), total)

    if not missing:
        return result

    log.info("%d/%d images need embedding, loading CLIP model ...", len(missing), total)
    from photo_tools.clip_tagger import CLIPEmbedder
    embedder = CLIPEmbedder(
        model_name=clip_model,
        pretrained=clip_pretrained,
    )

    for i, path in enumerate(missing, 1):
        print(f"\r  Embedding {i}/{len(missing)} ...", end="", flush=True, file=sys.stderr)
        prepared = prepare_image(path, cfg.clip.max_pixels)
        clip_input = prepared or path
        try:
            embedding = embedder.embed_image(clip_input)
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
# Clustering
# ---------------------------------------------------------------------------

def cluster_similar(
    embeddings: dict[Path, np.ndarray],
    threshold: float,
) -> list[list[Path]]:
    paths = list(embeddings.keys())
    if not paths:
        return []
    matrix = np.stack([embeddings[p] for p in paths])
    sim = matrix @ matrix.T

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
        cluster_paths = sorted(
            [paths[i] for i in indices],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        result.append(cluster_paths)

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


# ---------------------------------------------------------------------------
# Interactive cluster UI
# ---------------------------------------------------------------------------

def _create_contact_sheet(paths: list[Path], thumb_size: int = 200) -> Path:
    """Create a numbered thumbnail grid and return the path to a temp JPEG."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    n = len(paths)
    cols = min(n, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    cell_w = thumb_size
    cell_h = thumb_size
    label_h = 20
    sheet_w = cols * cell_w
    sheet_h = rows * (cell_h + label_h)

    sheet = PILImage.new("RGB", (sheet_w, sheet_h), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation-mono/LiberationMono-Bold.ttf", 14)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

    for idx, path in enumerate(paths):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = row * (cell_h + label_h)

        try:
            img = open_and_rotate(path)
            img.thumbnail((cell_w - 4, cell_h - 4), PILImage.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            ox = x0 + (cell_w - img.width) // 2
            oy = y0 + (cell_h - img.height) // 2
            sheet.paste(img, (ox, oy))
        except Exception:
            draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2],
                           outline=(100, 100, 100))

        label = str(idx)
        lx = x0 + cell_w // 2
        ly = y0 + cell_h + 2
        draw.text((lx, ly), label, fill=(255, 255, 255), font=font, anchor="mt")

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    sheet.save(tmp.name, "JPEG", quality=92)
    return Path(tmp.name)


def _print_cluster_ui(
    paths: list[Path],
    keep: list[bool],
    sim_matrix: np.ndarray,
    cursor: int,
    cluster_idx: int,
    total_clusters: int,
    dry_run: bool,
    zoomed: bool,
) -> None:
    """Render the terminal UI for one cluster."""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    lines = [
        "",
        f"  Cluster {cluster_idx + 1}/{total_clusters}  ({len(paths)} photos)"
        + ("  [DRY RUN]" if dry_run else ""),
        "",
    ]
    for j, path in enumerate(paths):
        size_mb = path.stat().st_size / 1_048_576
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        marker = "KEEP" if keep[j] else "MOVE"
        pointer = "▶" if j == cursor else " "
        if j == 0:
            sim_str = "       "
        else:
            max_sim = max(sim_matrix[j, k] for k in range(j))
            sim_str = f"  {max_sim:.3f}"
        line = f"  {pointer} [{marker}] [{j}] {path.name}{sim_str}  {mtime}  {size_mb:.1f} MB"
        lines.append(line)

    lines.append("")
    if zoomed:
        lines.append("  Zoomed on image — press any key to return to overview")
    else:
        lines += [
            "  [↑↓] select  [space] toggle  [a] keep all  [d] keep first only",
            "  [←→] prev/next cluster  [enter] confirm & next  [z] zoom  [q] quit",
        ]

    sys.stdout.write("\x1b[2J\x1b[H")
    for line in lines:
        sys.stdout.write(_truncate(line, cols) + "\r\n")
    sys.stdout.flush()


def _apply_moves(
    paths: list[Path],
    keep: list[bool],
    dest_root: Path,
    dry_run: bool,
) -> int:
    """Move non-kept images to dest. Returns count of moved files."""
    moved = 0
    for j, path in enumerate(paths):
        if not keep[j]:
            if dry_run:
                log.info("[DRY RUN] Would move %s -> %s", path.name, dest_root)
            else:
                move_to_dest(path, dest_root)
            moved += 1
    return moved


def interactive_similar_session(
    clusters: list[list[Path]],
    dest_root: Path,
    dry_run: bool,
    embeddings: dict[Path, np.ndarray],
) -> None:
    import subprocess

    opener = _find_opener(geometry="+0+0")
    if opener is None:
        log.error("No image viewer found. Install feh, imv, nsxiv, or similar.")
        sys.exit(1)

    total_clusters = len(clusters)
    cluster_idx = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    viewer_proc: subprocess.Popen | None = None
    sheet_path: Path | None = None
    moved_total = 0

    try:
        tty.setraw(fd)

        while 0 <= cluster_idx < total_clusters:
            paths_list = list(clusters[cluster_idx])
            matrix = np.stack([embeddings[p] for p in paths_list])
            sim_matrix = matrix @ matrix.T
            keep = [True] * len(paths_list)
            cursor = 0
            zoomed = False

            if sheet_path:
                try:
                    os.unlink(sheet_path)
                except OSError:
                    pass
            sheet_path = _create_contact_sheet(paths_list)

            def _show_viewer(target: Path):
                nonlocal viewer_proc
                if viewer_proc is not None:
                    _kill_viewer(viewer_proc)
                viewer_proc = subprocess.Popen(
                    opener + [str(target)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            _show_viewer(sheet_path)
            _print_cluster_ui(
                paths_list, keep, sim_matrix, cursor,
                cluster_idx, total_clusters, dry_run, zoomed,
            )

            while True:
                key = _read_key()

                if key == "q":
                    moved_total += _apply_moves(paths_list, keep, dest_root, dry_run)
                    cluster_idx = -1
                    break

                if zoomed:
                    zoomed = False
                    _show_viewer(sheet_path)
                    _print_cluster_ui(
                        paths_list, keep, sim_matrix, cursor,
                        cluster_idx, total_clusters, dry_run, zoomed,
                    )
                    continue

                if key == "up":
                    cursor = (cursor - 1) % len(paths_list)
                elif key == "down":
                    cursor = (cursor + 1) % len(paths_list)
                elif key == " ":
                    keep[cursor] = not keep[cursor]
                elif key == "a":
                    keep = [True] * len(paths_list)
                elif key == "d":
                    keep = [True] + [False] * (len(paths_list) - 1)
                elif key == "z" or key == "\r":
                    if key == "z":
                        zoomed = True
                        _show_viewer(paths_list[cursor])
                    else:
                        moved_total += _apply_moves(paths_list, keep, dest_root, dry_run)
                        cluster_idx += 1
                        break
                elif key == "right":
                    cluster_idx += 1
                    break
                elif key == "left":
                    cluster_idx = max(0, cluster_idx - 1)
                    break
                else:
                    if key.isdigit():
                        idx = int(key)
                        if idx < len(paths_list):
                            cursor = idx
                            keep[cursor] = not keep[cursor]
                    continue

                _print_cluster_ui(
                    paths_list, keep, sim_matrix, cursor,
                    cluster_idx, total_clusters, dry_run, zoomed,
                )

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if viewer_proc is not None:
            _kill_viewer(viewer_proc)
        if sheet_path:
            try:
                os.unlink(sheet_path)
            except OSError:
                pass
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
        if moved_total:
            action = "Would move" if dry_run else "Moved"
            print(f"{action} {moved_total} image(s) to {dest_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_similar_parser(subparsers) -> argparse.ArgumentParser:
    sub = subparsers.add_parser(
        "find-similar",
        help="Find visually similar images using CLIP embeddings.",
    )
    sub.add_argument("path", type=Path, help="Target directory or file")
    sub.add_argument("-n", "--dry-run", action="store_true",
                     help="Preview moves without executing")
    sub.add_argument("--threshold", type=float, default=None,
                     help="Cosine similarity cutoff (default: from config)")
    sub.add_argument("--dest", type=Path, default=None,
                     help="Destination directory for moved duplicates")
    sub.add_argument("--force", action="store_true",
                     help="Re-embed even if a cached embedding exists")
    sub.add_argument("--clip-model", default=None,
                     help="CLIP model name (default: from config)")
    sub.add_argument("--clip-pretrained", default=None,
                     help="CLIP pretrained weights (default: from config)")
    sub.set_defaults(func=run_find_similar)
    return sub


def run_find_similar(args) -> None:
    cfg = get_config()
    clip_model = args.clip_model or cfg.clip.model
    clip_pretrained = args.clip_pretrained or cfg.clip.pretrained
    model_id = f"{clip_model}/{clip_pretrained}"
    threshold = args.threshold if args.threshold is not None else cfg.similarity.threshold

    all_files = find_images(args.path)
    image_paths = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and "_duplicates" not in p.parts
    ]

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

    log.info("Clustering similar images (threshold=%.2f) ...", threshold)
    clusters = cluster_similar(embeddings, threshold)

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

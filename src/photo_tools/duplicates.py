"""
duplicates.py — Visual duplicate detection with an interactive dedup session.

Wires the `duplicates` CLI subcommand: load CLIP embeddings (cached in
XMP, fall back to computing), cluster by cosine similarity, then offer a
terminal UI for reviewing each cluster and moving non-kept images aside.

The picker supports undo via a sidecar manifest written to the dest folder,
so previously-moved images can be restored across sessions.
"""

import argparse
import json
import math
import os
import sys
import tempfile
import termios
import tty
from datetime import datetime, timezone
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
from photo_tools.logging_setup import get_logger

log = get_logger("duplicates")


# Hard cap on cluster size. Above this the cluster is recursively split at a
# stricter similarity threshold so the picker never has to render an unwieldy
# contact sheet. Tuned for a 4-column thumbnail grid (4x4 fits comfortably).
MAX_CLUSTER_SIZE = 16

# Filename of the per-dest sidecar manifest used for undo. JSON Lines so we can
# append cheaply and tolerate partial writes.
MANIFEST_NAME = ".photo-tools-moves.jsonl"


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

def _cluster_at_threshold(
    paths: list[Path],
    embeddings: dict[Path, np.ndarray],
    threshold: float,
) -> list[list[Path]]:
    """Single-link union-find clustering at the given threshold. Returns clusters
    of size >= 2 sorted by max pairwise similarity descending."""
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

    grouped: dict[int, list[int]] = {}
    for i in range(n):
        grouped.setdefault(find(i), []).append(i)

    out: list[list[Path]] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        cluster_paths = sorted(
            [paths[i] for i in indices],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        out.append(cluster_paths)

    def max_sim(cluster):
        idxs = [paths.index(p) for p in cluster]
        sims = [sim[idxs[a], idxs[b]] for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        return max(sims) if sims else 0.0

    out.sort(key=max_sim, reverse=True)
    return out


def _split_oversized(
    cluster: list[Path],
    embeddings: dict[Path, np.ndarray],
    threshold: float,
    max_size: int,
    step: float = 0.02,
    threshold_cap: float = 0.99,
) -> list[list[Path]]:
    """Recursively split *cluster* until every part is <= *max_size*.

    At each step the threshold is bumped by *step* and the cluster is
    re-clustered. Any subcluster still too large is split again. Singletons
    that fall out of the chain are dropped (consistent with the top-level
    cluster_similar behaviour).

    If the threshold reaches *threshold_cap* and the cluster is still
    oversized, it is hard-capped at *max_size* (most-similar pair seeds, then
    nearest neighbours) — a graceful degradation rather than infinite splitting.
    """
    if len(cluster) <= max_size:
        return [cluster]

    next_thr = round(threshold + step, 4)
    if next_thr >= threshold_cap:
        log.warning("Cluster of %d images can't be split below %d at threshold %.2f — "
                    "truncating to top %d most-similar.",
                    len(cluster), max_size, threshold, max_size)
        return [cluster[:max_size]]

    sub = _cluster_at_threshold(cluster, embeddings, next_thr)
    if not sub:
        # Re-clustering yielded nothing (all pairs below new threshold) — fall
        # back to a hard cap rather than recursing forever.
        log.warning("Cluster of %d images fragmented entirely at threshold %.2f — "
                    "truncating to top %d.",
                    len(cluster), next_thr, max_size)
        return [cluster[:max_size]]

    out: list[list[Path]] = []
    for sc in sub:
        out.extend(_split_oversized(sc, embeddings, next_thr, max_size, step, threshold_cap))
    return out


def cluster_similar(
    embeddings: dict[Path, np.ndarray],
    threshold: float,
    max_size: int = MAX_CLUSTER_SIZE,
) -> list[list[Path]]:
    """Cluster paths by cosine similarity. Oversized clusters are split."""
    paths = list(embeddings.keys())
    base = _cluster_at_threshold(paths, embeddings, threshold)
    if not base:
        return []

    out: list[list[Path]] = []
    for cluster in base:
        out.extend(_split_oversized(cluster, embeddings, threshold, max_size))
    return out


# ---------------------------------------------------------------------------
# Move + manifest (undo support)
# ---------------------------------------------------------------------------

def move_to_dest(path: Path, dest_root: Path) -> Path:
    """Move *path* into *dest_root*, returning the actual destination path
    (suffixed if necessary to avoid collisions)."""
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / path.name
    counter = 2
    while dest.exists():
        dest = dest_root / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    path.rename(dest)
    log.info("Moved %s -> %s", path.name, dest)
    return dest


def _manifest_path(dest_root: Path) -> Path:
    return dest_root / MANIFEST_NAME


def _append_manifest(dest_root: Path, original: Path, moved_to: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    entry = {
        "original": str(original),
        "moved_to": str(moved_to),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(_manifest_path(dest_root), "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_manifest(dest_root: Path) -> list[dict]:
    """Load manifest entries; missing or malformed lines are skipped silently."""
    mp = _manifest_path(dest_root)
    if not mp.exists():
        return []
    entries: list[dict] = []
    with open(mp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _rewrite_manifest(dest_root: Path, entries: list[dict]) -> None:
    """Atomically overwrite the manifest with *entries*."""
    mp = _manifest_path(dest_root)
    if not entries:
        try:
            mp.unlink()
        except FileNotFoundError:
            pass
        return
    tmp = mp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    tmp.replace(mp)


def _build_moved_index(dest_root: Path) -> dict[str, str]:
    """Map original path string -> latest moved_to path string from the manifest."""
    idx: dict[str, str] = {}
    for entry in _load_manifest(dest_root):
        idx[entry["original"]] = entry["moved_to"]
    return idx


def _remove_manifest_entry(dest_root: Path, original: Path) -> None:
    """Drop any manifest entries whose 'original' equals *original*."""
    entries = _load_manifest(dest_root)
    kept = [e for e in entries if e.get("original") != str(original)]
    if len(kept) != len(entries):
        _rewrite_manifest(dest_root, kept)


# ---------------------------------------------------------------------------
# Interactive cluster UI
# ---------------------------------------------------------------------------

# Per-path status during a session.
PRESENT = "present"  # file exists at original path
MOVED = "moved"      # file moved to dest by an earlier run; can be undone
MISSING = "missing"  # file gone, no manifest entry — unrecoverable


def _create_contact_sheet(
    paths: list[Path],
    statuses: list[str],
    thumb_size: int = 200,
) -> Path:
    """Create a numbered thumbnail grid and return the path to a temp JPEG.

    Items whose status is not PRESENT render as a placeholder ("MOVED" or
    "MISSING") so the grid layout still matches the cluster's index numbering.
    """
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
        status = statuses[idx]

        if status == PRESENT:
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
        else:
            placeholder = "MOVED" if status == MOVED else "MISSING"
            draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 2, y0 + cell_h - 2],
                           fill=(48, 48, 48), outline=(80, 80, 80))
            draw.text((x0 + cell_w // 2, y0 + cell_h // 2),
                      placeholder, fill=(160, 160, 160), font=font, anchor="mm")

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
    statuses: list[str],
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
        status = statuses[j]
        if status == PRESENT:
            try:
                size_mb = path.stat().st_size / 1_048_576
                mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                size_mb = 0.0
                mtime = "----------------"
            marker = "KEEP" if keep[j] else "MOVE"
        else:
            size_mb = 0.0
            mtime = "----------------"
            marker = "MOVED" if status == MOVED else "MISSING"

        pointer = "▶" if j == cursor else " "
        if j == 0:
            sim_str = "       "
        else:
            max_sim = max(sim_matrix[j, k] for k in range(j))
            sim_str = f"  {max_sim:.3f}"

        # Dim non-PRESENT entries so they read as inactive even in monochrome
        line = f"  {pointer} [{marker:<7}] [{j}] {path.name}{sim_str}  {mtime}  {size_mb:.1f} MB"
        if status != PRESENT:
            line = f"\x1b[2m{line}\x1b[0m"
        lines.append(line)

    lines.append("")
    if zoomed:
        lines.append("  Zoomed on image — press any key to return to overview")
    else:
        lines += [
            "  [↑↓] select  [space] toggle  [a] keep all  [d] keep first only  [o] keep only this",
            "  [u] undo move  [←→] prev/next cluster  [enter] confirm & next  [z] zoom  [q] quit",
        ]

    sys.stdout.write("\x1b[2J\x1b[H")
    for line in lines:
        sys.stdout.write(_truncate(line, cols) + "\r\n")
    sys.stdout.flush()


def _apply_moves(
    paths: list[Path],
    statuses: list[str],
    keep: list[bool],
    dest_root: Path,
    dry_run: bool,
) -> int:
    """Move PRESENT non-kept images to dest. Records each move in the manifest.
    Returns count of moved files."""
    moved = 0
    for j, path in enumerate(paths):
        if statuses[j] != PRESENT or keep[j]:
            continue
        if dry_run:
            log.info("[DRY RUN] Would move %s -> %s", path.name, dest_root)
            moved += 1
            continue
        try:
            actual_dest = move_to_dest(path, dest_root)
            _append_manifest(dest_root, path, actual_dest)
            statuses[j] = MOVED
            moved += 1
        except OSError as e:
            log.error("Failed to move %s: %s", path, e)
    return moved


def _undo_move(
    path: Path,
    moved_to: Path,
    dest_root: Path,
) -> bool:
    """Move *moved_to* back to *path*. Returns True on success.

    Refuses if the original path now exists or the moved file is missing
    (cannot resolve safely without overwriting/guessing)."""
    if path.exists():
        log.error("Cannot restore %s — a file already exists at the original path",
                  path.name)
        return False
    if not moved_to.exists():
        log.error("Cannot restore %s — moved file is gone (%s)", path.name, moved_to)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    moved_to.rename(path)
    _remove_manifest_entry(dest_root, path)
    log.info("Restored %s -> %s", moved_to.name, path)
    return True


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

    moved_index = _build_moved_index(dest_root)

    total_clusters = len(clusters)
    cluster_idx = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    viewer_proc: subprocess.Popen | None = None
    sheet_path: Path | None = None
    moved_total = 0
    restored_total = 0

    try:
        tty.setraw(fd)

        while 0 <= cluster_idx < total_clusters:
            paths_list = list(clusters[cluster_idx])
            matrix = np.stack([embeddings[p] for p in paths_list])
            sim_matrix = matrix @ matrix.T

            # Resolve per-path status. PRESENT if file exists at original path;
            # otherwise MOVED if the manifest knows where it went, else MISSING.
            statuses: list[str] = []
            moved_to: list[Path | None] = []
            for p in paths_list:
                if p.exists():
                    statuses.append(PRESENT)
                    moved_to.append(None)
                elif str(p) in moved_index and Path(moved_index[str(p)]).exists():
                    statuses.append(MOVED)
                    moved_to.append(Path(moved_index[str(p)]))
                else:
                    statuses.append(MISSING)
                    moved_to.append(None)

            keep = [True] * len(paths_list)
            cursor = 0
            zoomed = False

            if sheet_path:
                try:
                    os.unlink(sheet_path)
                except OSError:
                    pass
            sheet_path = _create_contact_sheet(paths_list, statuses)

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

            def _refresh_sheet():
                nonlocal sheet_path
                if sheet_path:
                    try:
                        os.unlink(sheet_path)
                    except OSError:
                        pass
                sheet_path = _create_contact_sheet(paths_list, statuses)
                _show_viewer(sheet_path)

            _show_viewer(sheet_path)
            _print_cluster_ui(
                paths_list, statuses, keep, sim_matrix, cursor,
                cluster_idx, total_clusters, dry_run, zoomed,
            )

            while True:
                key = _read_key()

                if key == "q":
                    moved_total += _apply_moves(paths_list, statuses, keep, dest_root, dry_run)
                    cluster_idx = -1
                    break

                if zoomed:
                    zoomed = False
                    _show_viewer(sheet_path)
                    _print_cluster_ui(
                        paths_list, statuses, keep, sim_matrix, cursor,
                        cluster_idx, total_clusters, dry_run, zoomed,
                    )
                    continue

                if key == "up":
                    cursor = (cursor - 1) % len(paths_list)
                elif key == "down":
                    cursor = (cursor + 1) % len(paths_list)
                elif key == " ":
                    if statuses[cursor] == PRESENT:
                        keep[cursor] = not keep[cursor]
                elif key == "a":
                    keep = [True] * len(paths_list)
                elif key == "d":
                    # Keep the first PRESENT image only; everything else marked MOVE.
                    first_present = next((j for j, s in enumerate(statuses)
                                          if s == PRESENT), None)
                    keep = [j == first_present for j in range(len(paths_list))]
                elif key == "o":
                    if statuses[cursor] == PRESENT:
                        keep = [j == cursor for j in range(len(paths_list))]
                elif key == "u":
                    if statuses[cursor] == MOVED and moved_to[cursor] is not None:
                        if dry_run:
                            log.info("[DRY RUN] Would restore %s",
                                     paths_list[cursor].name)
                            statuses[cursor] = PRESENT
                            keep[cursor] = True
                            moved_to[cursor] = None
                            restored_total += 1
                            _refresh_sheet()
                        else:
                            ok = _undo_move(paths_list[cursor], moved_to[cursor], dest_root)
                            if ok:
                                statuses[cursor] = PRESENT
                                keep[cursor] = True
                                moved_to[cursor] = None
                                # Drop our in-memory mapping too
                                moved_index.pop(str(paths_list[cursor]), None)
                                restored_total += 1
                                _refresh_sheet()
                elif key == "z" or key == "\r":
                    if key == "z":
                        if statuses[cursor] == PRESENT:
                            zoomed = True
                            _show_viewer(paths_list[cursor])
                        elif statuses[cursor] == MOVED and moved_to[cursor] is not None:
                            zoomed = True
                            _show_viewer(moved_to[cursor])
                    else:
                        moved_total += _apply_moves(paths_list, statuses, keep, dest_root, dry_run)
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
                            if statuses[idx] == PRESENT:
                                keep[cursor] = not keep[cursor]
                    continue

                _print_cluster_ui(
                    paths_list, statuses, keep, sim_matrix, cursor,
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
        if restored_total:
            action = "Would restore" if dry_run else "Restored"
            print(f"{action} {restored_total} image(s) from {dest_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_duplicates_parser(subparsers) -> argparse.ArgumentParser:
    sub = subparsers.add_parser(
        "duplicates",
        help="Find visual duplicates using CLIP embeddings (interactive picker).",
    )
    sub.add_argument("path", type=Path, help="Target directory or file")
    sub.add_argument("-n", "--dry-run", action="store_true",
                     help="Preview moves without executing")
    sub.add_argument("--threshold", type=float, default=None,
                     help="Cosine similarity cutoff (default: from config)")
    sub.add_argument("--dest", type=Path, default=None,
                     help="Destination directory for moved duplicates "
                          "(default: <common>/_duplicates)")
    sub.add_argument("--force", action="store_true",
                     help="Re-embed even if a cached embedding exists")
    sub.add_argument("--max-cluster-size", type=int, default=MAX_CLUSTER_SIZE,
                     help=f"Split clusters larger than this (default: {MAX_CLUSTER_SIZE})")
    sub.add_argument("--clip-model", default=None,
                     help="CLIP model name (default: from config)")
    sub.add_argument("--clip-pretrained", default=None,
                     help="CLIP pretrained weights (default: from config)")
    sub.set_defaults(func=run_duplicates)
    return sub


def run_duplicates(args) -> None:
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

    log.info("Clustering similar images (threshold=%.2f, max-cluster=%d) ...",
             threshold, args.max_cluster_size)
    clusters = cluster_similar(embeddings, threshold, max_size=args.max_cluster_size)

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

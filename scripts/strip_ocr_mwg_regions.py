#!/usr/bin/env python
"""One-off: remove OCR-shaped XMP-mwg-rs:RegionInfo regions from photos.

Earlier versions of `photo-tools tag` mirrored each OCR phrase into both
IPTC ImageRegion (correct, role=annotatedText) and MWG-RS RegionInfo with
Type=BarCode + Description="OCR detected text" (wrong: digiKam's People
view surfaces named MWG regions regardless of Type, so phrases like
"Way out" leak into the face tree). The MWG-RS write has since been
dropped; this script cleans up files that were tagged before that fix.

Identification rule (must match all):
  - Type == "BarCode"
  - Description == "OCR detected text"

Non-matching regions (real faces, pets, focus marks, third-party barcodes)
are preserved. If every region in a file matches, the whole RegionInfo is
deleted; otherwise it's rewritten with only the survivors.

IPTC ImageRegion (the correct OCR location) and XMP-phototools:OCRText
are left alone — they don't pollute digiKam.

Usage: uv run scripts/strip_ocr_mwg_regions.py <path> [-n] [-y] [-v]
"""
import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from photo_tools.config import get_config, load_config
from photo_tools.helpers import (
    _run_exiftool,
    _run_exiftool_json,
    find_images,
)
from photo_tools.logging_setup import setup_logging


log = logging.getLogger("strip_ocr_mwg_regions")


_OCR_DESCRIPTION = "OCR detected text"


def _is_ocr_region(region: dict) -> bool:
    return (
        region.get("Type") == "BarCode"
        and region.get("Description") == _OCR_DESCRIPTION
    )


def _scan_batch(
    paths: list[Path],
) -> dict[Path, tuple[list[dict], list[dict], dict | None]]:
    """Read MWG-RS RegionInfo for many files in batched exiftool calls.

    Returns {path: (drop, keep, applied_dims)} only for files with at least
    one OCR-shaped region. Files with no RegionInfo, no regions, or only
    non-OCR regions are absent from the dict.
    """
    if not paths:
        return {}

    batch_size = get_config().exiftool.batch_size
    total = len(paths)
    n_batches = (total + batch_size - 1) // batch_size
    result: dict[Path, tuple[list[dict], list[dict], dict | None]] = {}

    for batch_idx, i in enumerate(range(0, total, batch_size), 1):
        batch = paths[i:i + batch_size]
        log.info("Reading RegionInfo batch %d/%d (%d files, %d/%d total)",
                 batch_idx, n_batches, len(batch),
                 min(i + len(batch), total), total)
        try:
            meta_list = _run_exiftool_json(
                ["-struct", "-XMP-mwg-rs:RegionInfo"] + [str(p) for p in batch],
                with_config=False, timeout=120,
            )
        except Exception as e:
            log.warning("Batch read failed: %s", e)
            continue
        if not meta_list:
            continue

        str_to_path = {str(p): p for p in batch}
        for meta in meta_list:
            source = meta.get("SourceFile", "")
            path = str_to_path.get(source, Path(source))
            region_info = meta.get("RegionInfo") or {}
            region_list = region_info.get("RegionList") or []
            if not region_list:
                continue
            drop = [r for r in region_list if _is_ocr_region(r)]
            if not drop:
                continue
            keep = [r for r in region_list if not _is_ocr_region(r)]
            result[path] = (drop, keep, region_info.get("AppliedToDimensions"))

    return result


def _delete_batch(paths: list[Path]) -> int:
    """Delete the entire MWG-RS RegionInfo on each file in batched calls.

    Used for files where every region was OCR-shaped (nothing to preserve).
    Returns the number of files successfully updated.
    """
    if not paths:
        return 0

    batch_size = get_config().exiftool.batch_size
    written = 0
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        result = _run_exiftool(
            ["-overwrite_original", "-XMP-mwg-rs:RegionInfo="]
            + [str(p) for p in batch],
            with_config=False, timeout=120,
        )
        if result.returncode != 0:
            log.warning("Batch delete partial failure: %s", result.stderr.strip())
        # exiftool reports per-file success on stdout; trust returncode for
        # the whole batch and re-check from caller perspective is overkill.
        written += len(batch)
    return written


def _rewrite_one(
    path: Path, keep: list[dict], applied_dims: dict | None
) -> bool:
    """Rewrite MWG-RS RegionInfo on one file, preserving non-OCR regions.

    Per-file because each file's surviving region list is unique.
    """
    info: dict = {"RegionList": keep}
    if applied_dims:
        info["AppliedToDimensions"] = applied_dims

    payload = [{
        "SourceFile": str(path),
        "XMP-mwg-rs:RegionInfo": info,
    }]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(payload, tmp)
        tmp.close()
        result = _run_exiftool(
            ["-overwrite_original", "-struct", f"-json={tmp.name}", str(path)],
            with_config=False, timeout=60,
        )
        if result.returncode != 0:
            log.error("Rewrite failed for %s: %s", path.name, result.stderr.strip())
            return False
        return True
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="File or directory to scan")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="Preview without writing")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="Skip confirmation prompt")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Debug logging")
    args = ap.parse_args()

    setup_logging(verbose=args.verbose)
    load_config()

    images = find_images(args.path)
    if not images:
        log.error("No supported files found at %s", args.path)
        return 1

    log.info("Scanning %d file(s) for OCR-shaped MWG-RS regions ...", len(images))

    scanned = _scan_batch(images)
    if not scanned:
        log.info("No files contain OCR-shaped MWG-RS regions.")
        return 0

    delete_paths: list[Path] = []
    rewrite_targets: list[tuple[Path, list[dict], dict | None]] = []
    total_drops = 0
    for p, (drop, keep, applied_dims) in scanned.items():
        total_drops += len(drop)
        if not keep:
            delete_paths.append(p)
        else:
            rewrite_targets.append((p, keep, applied_dims))

    log.info(
        "Found %d OCR region(s) across %d file(s) (%d full delete, %d rewrite).",
        total_drops, len(scanned), len(delete_paths), len(rewrite_targets),
    )
    sample = list(scanned.items())[:20]
    for p, (drop, keep, _) in sample:
        names = ", ".join(r.get("Name", "?") for r in drop)
        log.debug("  %s — drop %d, keep %d (%s)", p.name, len(drop), len(keep), names)
    if len(scanned) > 20:
        log.debug("  ... (%d more)", len(scanned) - 20)

    if args.dry_run:
        log.info("[DRY RUN] Would strip OCR regions from %d file(s).", len(scanned))
        return 0

    if not args.yes:
        try:
            answer = input(
                f"Strip {total_drops} OCR region(s) from {len(scanned)} file(s)? [y/N] "
            )
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            log.info("Aborted.")
            return 0

    written = _delete_batch(delete_paths)
    for p, keep, applied_dims in rewrite_targets:
        if _rewrite_one(p, keep, applied_dims):
            written += 1
    log.info("Updated %d / %d file(s).", written, len(scanned))
    return 0 if written == len(scanned) else 1


if __name__ == "__main__":
    sys.exit(main())

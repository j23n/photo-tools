#!/usr/bin/env python
"""One-off: wipe ALL tags from Apple Live Photo motion .mov files.

Selects .mov files whose stem (case-insensitive) ends in .jpg/.jpeg/.heic/
.heif and runs photo_tools.helpers.clear_all_tags on them. Both legacy
field containers (IPTC:Keywords, XMP-dc:Subject, XMP-digiKam:TagsList,
XMP-lr:HierarchicalSubject, MicrosoftPhoto:LastKeywordXMP, MediaPro:
CatalogSets, MicrosoftPhoto:CategorySet) and any legacy values inside
them are removed, along with the entire XMP-phototools namespace.

Usage: uv run scripts/strip_live_photo_motion_tags.py <path> [-n] [-y]
"""
import argparse
import logging
import sys
from pathlib import Path

from photo_tools.helpers import clear_all_tags, is_live_photo_motion
from photo_tools.logging_setup import setup_logging


log = logging.getLogger("strip_live_photo_motion_tags")


def find_motion_companions(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if is_live_photo_motion(target) else []
    if not target.is_dir():
        log.error("Path does not exist: %s", target)
        return []
    return sorted(p for p in target.rglob("*") if p.is_file() and is_live_photo_motion(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="File or directory to scan")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Preview without writing")
    ap.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = ap.parse_args()

    setup_logging(verbose=args.verbose)

    matches = find_motion_companions(args.path)
    if not matches:
        log.info("No Live Photo motion companions found under %s", args.path)
        return 0

    log.info("Found %d Live Photo motion companion(s)", len(matches))
    for p in matches:
        log.debug("  %s", p)

    if args.dry_run:
        log.info("[DRY RUN] Would clear ALL tags from %d file(s)", len(matches))
        return 0

    if not args.yes:
        try:
            answer = input(f"Clear ALL tags from {len(matches)} file(s)? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            log.info("Aborted.")
            return 0

    ok = clear_all_tags(matches, dry_run=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

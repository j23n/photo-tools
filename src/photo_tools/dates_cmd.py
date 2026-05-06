"""
dates_cmd.py — `photo-tools dates` subcommand family.

backfill: write EXIF DateTimeOriginal (and friends) for files that have
no date set, recovered from the filename (WhatsApp/Signal/Telegram/Pixel/
generic patterns) and optionally from the filesystem mtime.

A photo-tools sentinel field XMP-phototools:DateBackfilledFrom records
which source supplied the date (e.g. "filename:whatsapp", "mtime") so the
backfill is auditable and reversible later.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

from photo_tools.helpers import (
    find_images,
    read_dates_batch,
    write_dates,
)
from photo_tools.logging_setup import get_logger

log = get_logger("dates")


# ---------------------------------------------------------------------------
# Filename → datetime extraction
# ---------------------------------------------------------------------------
#
# Patterns are tried in order; the first hit wins. Each pattern is anchored
# against the filename stem (no extension). `precision` is "datetime" when the
# pattern carries hours/minutes/seconds, "date" when it carries only Y-M-D
# (the time then defaults to local noon — see _PRECISION_DEFAULT_TIME).
#
# All patterns are case-insensitive.

_DATE_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # WhatsApp images and videos: IMG-20230515-WA0001, VID-20230515-WA0001
    ("filename:whatsapp",
     re.compile(r"^(?:IMG|VID)-(?P<y>\d{4})(?P<mo>\d{2})(?P<d>\d{2})-WA\d+$", re.I),
     "date"),

    # Signal: signal-2023-05-15-12-34-56-789 (millisecond suffix optional)
    ("filename:signal",
     re.compile(
         r"^signal-(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})"
         r"-(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})", re.I),
     "datetime"),

    # Telegram: photo_2023-05-15_12-34-56
    ("filename:telegram",
     re.compile(
         r"^photo_(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})"
         r"_(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})", re.I),
     "datetime"),

    # Pixel/Google Camera: PXL_20230515_123456789 (millis suffix on time)
    ("filename:pixel",
     re.compile(
         r"^PXL_(?P<y>\d{4})(?P<mo>\d{2})(?P<d>\d{2})"
         r"_(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})", re.I),
     "datetime"),

    # Screenshot_2023-05-15-12-34-56 (Android), Screenshot_20230515-123456
    ("filename:screenshot",
     re.compile(
         r"^Screenshot[_-](?P<y>\d{4})-?(?P<mo>\d{2})-?(?P<d>\d{2})"
         r"[-_](?P<h>\d{2})-?(?P<mi>\d{2})-?(?P<s>\d{2})", re.I),
     "datetime"),

    # Generic Android / iOS-export with full timestamp:
    # IMG_20230515_123456, 20230515_123456, 2023-05-15_12-34-56
    ("filename:android",
     re.compile(
         r"^(?:IMG|VID|MVIMG)?_?(?P<y>\d{4})-?(?P<mo>\d{2})-?(?P<d>\d{2})"
         r"[_-](?P<h>\d{2})-?(?P<mi>\d{2})-?(?P<s>\d{2})", re.I),
     "datetime"),

    # Last-resort: any YYYY[-_]?MM[-_]?DD substring with a plausible year
    # (1990-2099). Surrounded by non-digits to avoid splicing through serial
    # numbers like IMG_45678901.
    ("filename:date-only",
     re.compile(
         r"(?:^|[^0-9])(?P<y>(?:19|20)\d{2})[-_]?"
         r"(?P<mo>0[1-9]|1[0-2])[-_]?"
         r"(?P<d>0[1-9]|[12]\d|3[01])(?:$|[^0-9])"),
     "date"),
]

# Local noon — picked deliberately so tagged dates don't drift across
# days under common timezone offsets when consumers re-interpret naive
# EXIF timestamps in UTC.
_NOON = (12, 0, 0)


def extract_date_from_filename(name: str) -> tuple[datetime, str] | None:
    """Return (datetime, source_label) for the first matching pattern, or None.

    `name` is the filename including extension; matching is done against the
    stem. The returned datetime is naive (no tzinfo) per EXIF convention.
    """
    stem = Path(name).stem
    for label, regex, precision in _DATE_PATTERNS:
        m = regex.search(stem) if label == "filename:date-only" else regex.match(stem)
        if not m:
            continue
        try:
            y, mo, d = int(m.group("y")), int(m.group("mo")), int(m.group("d"))
            if precision == "datetime":
                h, mi, s = int(m.group("h")), int(m.group("mi")), int(m.group("s"))
            else:
                h, mi, s = _NOON
            return datetime(y, mo, d, h, mi, s), label
        except (ValueError, IndexError):
            # Implausible date (e.g. month=13) — keep trying later patterns.
            continue
    return None


def extract_date_from_mtime(path: Path) -> tuple[datetime, str] | None:
    """Return (mtime as naive local datetime, "mtime"), or None on stat error."""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts), "mtime"


# ---------------------------------------------------------------------------
# Backfill driver
# ---------------------------------------------------------------------------

_VALID_SOURCES = ("filename", "mtime")


def _parse_sources(raw: str) -> list[str]:
    parts = [s.strip().lower() for s in raw.split(",") if s.strip()]
    bad = [s for s in parts if s not in _VALID_SOURCES]
    if bad:
        raise ValueError(
            f"unknown source(s): {bad}. Valid: {list(_VALID_SOURCES)}"
        )
    if not parts:
        raise ValueError("at least one source is required")
    return parts


def _resolve_date(
    path: Path, sources: list[str]
) -> tuple[datetime, str] | None:
    for source in sources:
        if source == "filename":
            hit = extract_date_from_filename(path.name)
        elif source == "mtime":
            hit = extract_date_from_mtime(path)
        else:
            continue
        if hit:
            return hit
    return None


def run_backfill_dates(args) -> None:
    try:
        sources = _parse_sources(args.source)
    except ValueError as e:
        log.error("%s", e)
        sys.exit(2)

    paths = find_images(args.path)
    if not paths:
        log.error("No supported files found at %s", args.path)
        sys.exit(1)

    log.info("Scanning %d file(s) for missing dates (sources: %s) ...",
             len(paths), ",".join(sources))

    existing = read_dates_batch(paths)

    needs_backfill: list[Path] = []
    for p in paths:
        meta = existing.get(p, {})
        has_date = bool(meta.get("DateTimeOriginal") or meta.get("CreateDate"))
        if has_date and not args.force:
            continue
        needs_backfill.append(p)

    log.info("%d file(s) missing a date; resolving sources ...",
             len(needs_backfill))

    resolved: list[tuple[Path, datetime, str]] = []
    unmatched: list[Path] = []
    for p in needs_backfill:
        hit = _resolve_date(p, sources)
        if hit:
            resolved.append((p, hit[0], hit[1]))
        else:
            unmatched.append(p)

    # Per-source counts for the summary.
    by_source: dict[str, int] = {}
    for _, _, source in resolved:
        by_source[source] = by_source.get(source, 0) + 1

    log.info("Resolved %d / %d (unmatched: %d)",
             len(resolved), len(needs_backfill), len(unmatched))
    for source, n in sorted(by_source.items()):
        log.info("  %s: %d", source, n)

    if args.dry_run:
        for p, dt, source in resolved[:50]:
            log.info("[DRY RUN] %s ← %s (%s)",
                     p.name, dt.strftime("%Y-%m-%d %H:%M:%S"), source)
        if len(resolved) > 50:
            log.info("[DRY RUN] ... (%d more)", len(resolved) - 50)
        if unmatched:
            log.info("Unmatched (first 20):")
            for p in unmatched[:20]:
                log.info("  %s", p.name)
        return

    written = 0
    for p, dt, source in resolved:
        if write_dates(p, dt, source, dry_run=False):
            written += 1
    log.info("Wrote dates to %d / %d file(s)", written, len(resolved))

    if unmatched:
        log.info("%d file(s) had no recoverable date; first 20:", len(unmatched))
        for p in unmatched[:20]:
            log.info("  %s", p.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_dates_parser(subparsers) -> None:
    dates_parser = subparsers.add_parser(
        "dates",
        help="Date metadata: backfill missing EXIF DateTimeOriginal.",
    )
    dates_sub = dates_parser.add_subparsers(dest="dates_command", required=True)

    p = dates_sub.add_parser(
        "backfill",
        help="Write DateTimeOriginal for files missing it, recovered from "
             "filename patterns (WhatsApp/Signal/Telegram/Pixel/etc.) and "
             "optionally from filesystem mtime.",
    )
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument(
        "--source",
        default="filename",
        help="Comma-separated source list, tried in order. Choices: filename, "
             "mtime. Default: filename. For libraries where mtime is reliable, "
             "use 'filename,mtime'.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even when DateTimeOriginal/CreateDate is already set.",
    )
    p.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Preview which dates would be written without modifying files.",
    )
    p.set_defaults(func=run_backfill_dates)

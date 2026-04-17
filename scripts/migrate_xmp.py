#!/usr/bin/env python3
"""migrate_xmp.py — one-shot cleanup of legacy photo-tools metadata.

Strips:
  - XMP-lr:HierarchicalSubject, MicrosoftPhoto:LastKeywordXMP,
    MediaPro:CatalogSets, MicrosoftPhoto:CategorySet
  - The entire photo-tools XMP namespace (any URI it was registered under)
  - Every keyword matching the pre-2026.1 legacy patterns (year/*, month/*,
    country/*, scene/*, ai:tagged, etc.)

After running this, re-run `photo-tools tag` over the same files. See
docs/xmp-schema.md §4.

Fast by default: a single persistent `exiftool -stay_open` process handles
every read and write, batched `BATCH_SIZE` files at a time for reads. This
is ~20x faster than spawning exiftool per file.

Usage:
    scripts/migrate_xmp.py [--dry-run] [-v] PATH [PATH...]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BATCH_SIZE = 100
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp",
              ".heic", ".heif", ".dng"}

# Legacy patterns (pre-2026.1). Keep in sync with photo_tools.constants
# LEGACY_PREFIXES / LEGACY_BARE_TAGS.
LEGACY_BARE = {"weekend", "weekday", "screenshot", "video", "ai:tagged"}
LEGACY_PREFIX_RE = re.compile(
    r"^(country|cc|region|city|neighborhood|landmark|scene|setting|"
    r"object|animal|plant|vehicle|food|other|activity|event|weather|"
    r"time|text|year|month|day|flash)/"
)

DEPRECATED_FIELDS = [
    "XMP-lr:HierarchicalSubject",
    "MicrosoftPhoto:LastKeywordXMP",
    "MediaPro:CatalogSets",
    "MicrosoftPhoto:CategorySet",
]


def is_legacy(keyword: str) -> bool:
    k = keyword.lower()
    return k in LEGACY_BARE or bool(LEGACY_PREFIX_RE.match(k))


def walk_images(targets: list[Path]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        if t.is_file():
            if t.suffix.lower() in IMAGE_EXTS:
                out.append(t)
            else:
                print(f"skip (not an image): {t}", file=sys.stderr)
        elif t.is_dir():
            for p in t.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    out.append(p)
        else:
            print(f"skip (not found): {t}", file=sys.stderr)
    return out


class Exiftool:
    """Persistent `exiftool -stay_open` process. One Perl startup, many commands."""

    _SENTINEL = "{ready}"

    def __init__(self, config: Path):
        self.proc = subprocess.Popen(
            ["exiftool", "-config", str(config), "-stay_open", "True", "-@", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write("-stay_open\nFalse\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def execute(self, args: list[str]) -> str:
        """Send a command and return stdout up to {ready}."""
        assert self.proc.stdin is not None and self.proc.stdout is not None
        for a in args:
            self.proc.stdin.write(a + "\n")
        self.proc.stdin.write("-execute\n")
        self.proc.stdin.flush()

        lines: list[str] = []
        for line in self.proc.stdout:
            s = line.rstrip("\n")
            if s == self._SENTINEL:
                break
            lines.append(s)
        return "\n".join(lines)


def plan_migration(meta: dict) -> dict | None:
    """Given one file's metadata dict, return a plan or None if clean.

    Plan shape: {"keyword_removals": {"Keywords": [...], "Subject": [...],
                 "TagsList": [...]}, "wipe_fields": [...]}
    """
    def as_list(v) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    legacy_kw = [k for k in as_list(meta.get("Keywords")) if is_legacy(k)]
    legacy_sj = [k for k in as_list(meta.get("Subject")) if is_legacy(k)]
    legacy_tl = [k for k in as_list(meta.get("TagsList")) if is_legacy(k)]

    wipe = []
    for f in DEPRECATED_FIELDS:
        # exiftool's JSON returns the bare field name (e.g. "HierarchicalSubject").
        name = f.split(":", 1)[1]
        if meta.get(name) not in (None, "", []):
            wipe.append(f)

    # Anything in the photo-tools namespace?
    has_photo_tools_ns = any(k for k, v in meta.items()
                             if k.startswith("phototools") or k in {
                                 "CLIPEmbedding", "CLIPModel", "CLIPTimestamp",
                                 "TaggerVersion", "TaggedAt", "CountryCode",
                             } and v not in (None, "", []))

    if not (legacy_kw or legacy_sj or legacy_tl or wipe or has_photo_tools_ns):
        return None
    return {
        "keyword_removals": {"Keywords": legacy_kw, "Subject": legacy_sj, "TagsList": legacy_tl},
        "wipe_fields": wipe,
        "wipe_phototools_ns": has_photo_tools_ns,
    }


def format_plan(path: Path, plan: dict) -> list[str]:
    out = [f"would migrate: {path}"]
    for field, kws in plan["keyword_removals"].items():
        short = {"Keywords": "IPTC:Keywords", "Subject": "dc:Subject",
                 "TagsList": "TagsList"}[field]
        for k in kws:
            out.append(f"    {short:14s} -= {k}")
    if plan["wipe_phototools_ns"]:
        out.append("    XMP-phototools:* (entire namespace)")
    for f in plan["wipe_fields"]:
        out.append(f"    {f}")
    return out


def build_write_args(path: Path, plan: dict) -> list[str]:
    args = ["-overwrite_original", "-q"]
    for k in plan["keyword_removals"]["Keywords"]:
        args.append(f"-IPTC:Keywords-={k}")
    for k in plan["keyword_removals"]["Subject"]:
        args.append(f"-XMP-dc:Subject-={k}")
    for k in plan["keyword_removals"]["TagsList"]:
        args.append(f"-XMP-digiKam:TagsList-={k}")
    if plan["wipe_phototools_ns"]:
        args.append("-XMP-phototools:all=")
    for f in plan["wipe_fields"]:
        args.append(f"-{f}=")
    args.append(str(path))
    return args


def batch_read(et: Exiftool, batch: list[Path]) -> list[dict]:
    """Read all relevant fields for a batch of files in one exiftool call."""
    args = [
        "-j", "-G1", "-n",
        "-IPTC:Keywords", "-XMP-dc:Subject", "-XMP-digiKam:TagsList",
        "-XMP-phototools:all",
        *[f"-{f}" for f in DEPRECATED_FIELDS],
        *[str(p) for p in batch],
    ]
    stdout = et.execute(args)
    if not stdout.strip():
        return [{} for _ in batch]
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return [{} for _ in batch]

    # Strip group prefixes from keys (e.g. "XMP-dc:Subject" -> "Subject")
    # and index by SourceFile.
    by_src = {}
    for m in data:
        clean = {}
        for k, v in m.items():
            bare = k.split(":", 1)[-1] if ":" in k else k
            clean[bare] = v
        by_src[m.get("SourceFile", "")] = clean
    return [by_src.get(str(p), {}) for p in batch]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report changes without writing")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Also log files that need no changes")
    args = ap.parse_args()

    config = Path(__file__).resolve().parent.parent / "src" / "photo_tools" / "exiftool_phototools.config"
    if not config.is_file():
        print(f"error: exiftool config not found at {config}", file=sys.stderr)
        return 1

    images = walk_images(args.paths)
    if not images:
        print("no images found", file=sys.stderr)
        return 0
    print(f"scanning {len(images)} file(s) ...", file=sys.stderr)

    et = Exiftool(config)
    total = 0
    changed = 0
    start = time.monotonic()

    try:
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i + BATCH_SIZE]
            metas = batch_read(et, batch)
            for path, meta in zip(batch, metas):
                total += 1
                plan = plan_migration(meta)
                if plan is None:
                    if args.verbose:
                        print(f"clean:     {path}")
                    continue
                changed += 1
                if args.dry_run:
                    print("\n".join(format_plan(path, plan)))
                else:
                    et.execute(build_write_args(path, plan))
                    print(f"migrated:  {path}")

            # Progress line
            elapsed = time.monotonic() - start
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  ... {total}/{len(images)} scanned, "
                  f"{changed} changed, {rate:.1f} img/s",
                  file=sys.stderr, end="\r")
        print(file=sys.stderr)
    finally:
        et.close()

    if args.dry_run:
        print(f"\nDry run: {changed}/{total} file(s) would be migrated.")
    else:
        print(f"\nDone: {changed}/{total} file(s) migrated.")
        if changed:
            print("Next: re-run 'photo-tools tag <PATH>' to repopulate under the new schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

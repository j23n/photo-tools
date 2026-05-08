"""Shared helpers for photo-tools.

Exiftool operations, file discovery, image preparation, and embedding cache.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from photo_tools.config import get_config
from photo_tools.constants import (
    OUR_TAG_ROOTS,
    SUPPORTED_EXTENSIONS,
    TAGGER_VERSION,
    VIDEO_EXTENSIONS,
)
from photo_tools.logging_setup import get_logger

log = get_logger("exif")

# Path to our exiftool config registering the photo-tools XMP namespace.
# See docs/xmp-schema.md §1.2 and exiftool_phototools.config.
_EXIFTOOL_CONFIG = Path(__file__).with_name("exiftool_phototools.config")


# ---------------------------------------------------------------------------
# Exiftool subprocess helpers
# ---------------------------------------------------------------------------

def _run_exiftool(
    args: list[str],
    *,
    with_config: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Invoke exiftool with consistent capture/timeout handling.

    `with_config` prepends `-config <photo-tools.config>` so the
    XMP-phototools namespace is recognized; pass `False` for calls that
    only touch standard tags. `timeout` defaults to
    `exiftool.timeout` from the config.
    """
    if timeout is None:
        timeout = get_config().exiftool.timeout
    cmd = ["exiftool"]
    if with_config:
        cmd += ["-config", str(_EXIFTOOL_CONFIG)]
    cmd += args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _run_exiftool_json(
    args: list[str],
    *,
    with_config: bool = True,
    timeout: float | None = None,
) -> list[dict]:
    """Run exiftool with `-j` and return parsed JSON, or [] on failure."""
    proc = _run_exiftool(["-j"] + args, with_config=with_config, timeout=timeout)
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# XMP sidecar handling (see docs/xmp-schema.md §1.5)
#
# Every photo-tools write goes to the image and a sibling XMP sidecar
# (IMG_1234.jpg -> IMG_1234.jpg.xmp). Videos skip the embedded write —
# XMP doesn't round-trip through most video containers, so the sidecar
# is the canonical store there. Reads consider both files and merge:
# list fields (keywords, OCR phrases) union-and-dedup; scalars take the
# sidecar value when present, falling back to the embedded value.
# ---------------------------------------------------------------------------

def xmp_sidecar_path(path: Path) -> Path:
    """IMG_1234.jpg → IMG_1234.jpg.xmp.

    Suffix is preserved (MWG / digiKam convention) so files sharing a
    stem ('IMG_1234.jpg' and 'IMG_1234.heic') get distinct sidecars.
    """
    return path.with_suffix(path.suffix + ".xmp")


def alt_xmp_sidecar_path(path: Path) -> Path | None:
    """IMG_1234.jpg → IMG_1234.xmp (Lightroom / Capture One convention).

    Returns None if the alt path would collide with the canonical path
    (i.e. the input is already a .xmp file). Used only for *reads* —
    writes always go to the canonical `IMG_1234.jpg.xmp`.
    """
    alt = path.with_suffix(".xmp")
    if alt == path:
        return None
    return alt


# Tag-write args targeting these groups can't go into a pure XMP file
# (no IPTC/EXIF/QuickTime sections in an XMP sidecar). Stripped before
# the sidecar phase of every write.
_NON_XMP_GROUPS = ("IPTC:", "EXIF:", "QuickTime:", "MakerNotes:", "File:")
_WRITE_OPS = ("+=", "-=", "<=", ">=", "=")


def _is_non_xmp_write_arg(arg: str) -> bool:
    """True if `arg` is `-Group:Tag<op>value` targeting a non-XMP group."""
    if not arg.startswith("-") or len(arg) < 2:
        return False
    body = arg[1:]
    for op in _WRITE_OPS:
        if op in body:
            tag = body.split(op, 1)[0]
            return any(tag.startswith(g) for g in _NON_XMP_GROUPS)
    return False


def _xmp_only_args(args: list[str]) -> list[str]:
    return [a for a in args if not _is_non_xmp_write_arg(a)]


def _sidecars_enabled() -> bool:
    """Whether sidecar reads/writes are active. Off by default; opt in
    via --xmp-sidecars (or `xmp.sidecars: true` in config).
    """
    return bool(getattr(getattr(get_config(), "xmp", None), "sidecars", False))


def _write_targets(path: Path) -> list[Path]:
    """Targets for a metadata write.

    With sidecars enabled: [image, sidecar] for images, [sidecar] for
    videos (XMP doesn't round-trip through most video containers).
    With sidecars disabled: [path] — write straight to the file and
    let exiftool handle the container.
    """
    if not _sidecars_enabled():
        return [path]
    sidecar = xmp_sidecar_path(path)
    if is_video(path):
        return [sidecar]
    return [path, sidecar]


def _exec_write(
    args: list[str],
    targets: list[Path],
    *,
    with_config: bool = True,
    timeout: float | None = None,
) -> tuple[bool, str]:
    """Run exiftool with `args` against `targets`, splitting image vs
    sidecar phases. Sidecar phase strips non-XMP write args. Returns
    (all_succeeded, last_stderr).
    """
    image_targets = [t for t in targets if t.suffix.lower() != ".xmp"]
    sidecar_targets = [t for t in targets if t.suffix.lower() == ".xmp"]

    success = True
    last_err = ""

    if image_targets:
        cmd = list(args) + [str(t) for t in image_targets]
        r = _run_exiftool(cmd, with_config=with_config, timeout=timeout)
        if r.returncode != 0:
            success = False
            last_err = r.stderr.strip()

    if sidecar_targets:
        cmd = _xmp_only_args(args) + [str(t) for t in sidecar_targets]
        r = _run_exiftool(cmd, with_config=with_config, timeout=timeout)
        if r.returncode != 0:
            success = False
            last_err = r.stderr.strip()

    return success, last_err


def _expand_paths_with_sidecars(
    paths: list[Path],
) -> tuple[list[str], dict[str, Path]]:
    """Expand `paths` for an exiftool read: each image plus its sidecar
    if one exists. Returns (file_strs, str_to_logical_path). Both
    embedded and sidecar SourceFile strings map back to the logical
    image path so callers can merge per-image.
    """
    files: list[str] = []
    str_to_path: dict[str, Path] = {}
    sidecars_on = _sidecars_enabled()
    for p in paths:
        files.append(str(p))
        str_to_path[str(p)] = p
        if not sidecars_on:
            continue
        sidecar = xmp_sidecar_path(p)
        if sidecar.exists():
            files.append(str(sidecar))
            str_to_path[str(sidecar)] = p
            continue
        # Fall back to the Lightroom / Capture One convention
        # (IMG_1234.xmp) if the canonical sidecar isn't there.
        alt = alt_xmp_sidecar_path(p)
        if alt is not None and alt.exists():
            files.append(str(alt))
            str_to_path[str(alt)] = p
    return files, str_to_path


_LIST_MERGE_FIELDS = {"Keywords", "Subject", "TagsList", "OCRText"}


def _merge_metas(metas: list[dict]) -> dict:
    """Merge exiftool result entries for one logical image.

    Caller orders entries embedded-first, sidecar-last. Keyword /
    Subject / TagsList / OCRText are unioned with order-preserving
    dedup; other fields take the last non-empty value (sidecar wins).
    """
    if not metas:
        return {}
    merged: dict = {}
    for meta in metas:
        for k, v in meta.items():
            if k == "SourceFile":
                continue
            if k in _LIST_MERGE_FIELDS:
                cur = merged.get(k)
                cur_list = [cur] if isinstance(cur, str) else list(cur or [])
                new_list = [v] if isinstance(v, str) else list(v or [])
                seen = set()
                out = []
                for x in cur_list + new_list:
                    if x not in seen:
                        seen.add(x)
                        out.append(x)
                if out:
                    merged[k] = out
            else:
                if v not in (None, "", [], {}):
                    merged[k] = v
    if "SourceFile" in metas[0]:
        merged["SourceFile"] = metas[0]["SourceFile"]
    return merged


def _group_metas_by_path(
    metas: list[dict], str_to_path: dict[str, Path]
) -> dict[Path, dict]:
    """Group exiftool results by logical image path, merging the
    image's embedded entry with its sidecar entry. Sidecar entries are
    placed last so they win on scalar conflicts.
    """
    by_path: dict[Path, list[dict]] = {}
    for meta in metas:
        source = meta.get("SourceFile", "")
        path = str_to_path.get(source)
        if path is None:
            continue
        by_path.setdefault(path, []).append(meta)
    out: dict[Path, dict] = {}
    for path, entries in by_path.items():
        entries.sort(
            key=lambda m: 1 if m.get("SourceFile", "").lower().endswith(".xmp") else 0
        )
        out[path] = _merge_metas(entries)
    return out


# ---------------------------------------------------------------------------
# Tag utilities
# ---------------------------------------------------------------------------

def is_our_tag(tag: str) -> bool:
    """True if `tag` is in a root we own (Places/, Objects/, Scenes/, Landmarks/)."""
    return any(tag.startswith(root) for root in OUR_TAG_ROOTS)


def deduplicate(tags: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def leaf_of(tag: str) -> str:
    """'Places/Italy/Rome' -> 'Rome'; flat tag returned unchanged."""
    return tag.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Exiftool tag operations
#
# Output fields per docs/xmp-schema.md §1.1:
#   dc:subject (= IPTC:Keywords):    leaf names only
#   digiKam:TagsList:                full hierarchical path, '/' separator
#   lr:HierarchicalSubject:          full hierarchical path, '|' separator
#                                    (Lightroom / Bridge native)
#   iptcExt:PersonInImage:           leaf only, for People/* tags
# ---------------------------------------------------------------------------

def _build_tag_args(tag: str, operator: str) -> list[str]:
    """Build exiftool args for a single hierarchical tag. operator is '+=' or '-='."""
    leaf = leaf_of(tag)
    lr_path = tag.replace("/", "|")
    args = [
        f"-IPTC:Keywords{operator}{leaf}",
        f"-XMP-dc:Subject{operator}{leaf}",
        f"-XMP-digiKam:TagsList{operator}{tag}",
        f"-XMP-lr:HierarchicalSubject{operator}{lr_path}",
    ]
    if tag.startswith("People/"):
        args.append(f"-XMP-iptcExt:PersonInImage{operator}{leaf}")
    return args


def add_tags(
    path: Path,
    keywords: list[str],
    dry_run: bool = False,
    *,
    namespace_fields: dict[str, str] | None = None,
) -> bool:
    """Add tags + photo-tools namespace fields to a single file using exiftool.

    `namespace_fields` is an optional dict of XMP-phototools field names to
    string values (e.g. {"CountryCode": "IT"}). The TaggerVersion + TaggedAt
    sentinel pair is always written.
    """
    if not keywords and not namespace_fields:
        log.warning("Nothing to write for %s", path.name)
        return False
    if dry_run:
        log.info("[DRY RUN] Would write %d keywords to %s:", len(keywords), path.name)
        for kw in sorted(keywords):
            log.info("    %s", kw)
        if namespace_fields:
            for k, v in sorted(namespace_fields.items()):
                log.info("    photo-tools:%s = %s", k, v)
        return True

    ts = datetime.now(UTC).isoformat()
    args = ["-overwrite_original"]
    for kw in keywords:
        args.extend(_build_tag_args(kw, "+="))
    args.extend([
        f"-XMP-phototools:TaggerVersion={TAGGER_VERSION}",
        f"-XMP-phototools:TaggedAt={ts}",
    ])
    for field, value in (namespace_fields or {}).items():
        args.append(f"-XMP-phototools:{field}={value}")

    try:
        ok, err = _exec_write(args, _write_targets(path))
        if not ok:
            log.error("exiftool failed for %s: %s", path.name, err)
            return False
        log.info("Wrote %d keywords to %s", len(keywords), path.name)
        return True
    except FileNotFoundError:
        log.error("exiftool not found. Install with: brew install exiftool")
        sys.exit(1)
    except Exception as e:
        log.error("Failed to write keywords to %s: %s", path.name, e)
        return False


def remove_tags(
    paths: list[Path],
    tags: list[str],
    *,
    dry_run: bool = False,
) -> bool:
    """Remove tags from one or more files using exiftool.

    For hierarchical tags ('Places/Italy/Rome') this removes both the full
    path from digiKam:TagsList and the leaf ('Rome') from dc:subject /
    IPTC:Keywords, since the new schema keeps dc:subject leaf-only.
    """
    if not paths or not tags:
        return True
    if dry_run:
        log.info("[DRY RUN] Would remove %d tag(s) from %d file(s)", len(tags), len(paths))
        return True

    batch_size = get_config().exiftool.batch_size
    success = True

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        args = ["-overwrite_original"]
        for tag in tags:
            args.extend(_build_tag_args(tag, "-="))

        targets: list[Path] = []
        for p in batch:
            targets.extend(_write_targets(p))
        # Drop sidecar targets that don't exist yet — `-=` on a missing
        # file would create it just to clear non-existent fields.
        targets = [t for t in targets if t.suffix.lower() != ".xmp" or t.exists()]

        try:
            ok, err = _exec_write(args, targets, with_config=False)
            if not ok:
                log.warning("Could not remove tags: %s", err)
                success = False
        except Exception as e:
            log.warning("Error removing tags: %s", e)
            success = False

    return success


def clear_all_keywords(path: Path, dry_run: bool = False) -> bool:
    """Nuclear option: remove ALL IPTC keywords and XMP subjects from a file."""
    if dry_run:
        log.info("[DRY RUN] Would clear ALL keywords from %s", path.name)
        return True

    args = [
        "-overwrite_original",
        "-IPTC:Keywords=", "-XMP-dc:Subject=",
        "-XMP-digiKam:TagsList=",
        "-XMP-lr:HierarchicalSubject=",
        "-XMP-iptcExt:PersonInImage=",
    ]
    targets = [t for t in _write_targets(path)
               if t.suffix.lower() != ".xmp" or t.exists()]
    try:
        ok, err = _exec_write(args, targets, with_config=False)
        if not ok:
            log.warning("Could not clear keywords from %s: %s", path.name, err)
            return False
        log.debug("Cleared ALL keywords from %s", path.name)
        return True
    except Exception as e:
        log.warning("Error clearing keywords from %s: %s", path.name, e)
        return False


# Fields wiped by clear_all_tags: every keyword/tag container plus the legacy
# sibling fields that past versions (and neighbouring tools) wrote into. See
# docs/xmp-schema.md §1.1 and §1.3.
_ALL_TAG_FIELDS = (
    "IPTC:Keywords",
    "XMP-dc:Subject",
    "XMP-digiKam:TagsList",
    "XMP-lr:HierarchicalSubject",
    "XMP-iptcExt:PersonInImage",
    "MicrosoftPhoto:LastKeywordXMP",
    "MediaPro:CatalogSets",
    "MicrosoftPhoto:CategorySet",
)


def _read_people_tags(paths: list[Path]) -> dict[Path, list[str]]:
    """Return digiKam People/* hierarchical tags for each file."""
    if not paths:
        return {}

    batch_size = get_config().exiftool.batch_size
    result: dict[Path, list[str]] = {}

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(
            ["-XMP-digiKam:TagsList"] + files,
            with_config=False, timeout=120,
        )
        if not meta_list:
            continue

        for path, meta in _group_metas_by_path(meta_list, str_to_path).items():
            tags = meta.get("TagsList", [])
            if isinstance(tags, str):
                tags = [tags]
            people = [str(t) for t in tags if str(t).startswith("People/")]
            if people:
                result[path] = people

    return result


def clear_all_tags(
    paths: list[Path],
    *,
    dry_run: bool = False,
) -> bool:
    """Wipe every photo-tools-owned keyword and namespace field from files.

    Removes all keyword containers listed in _ALL_TAG_FIELDS plus the entire
    XMP-phototools namespace (TaggerVersion, TaggedAt, CountryCode, CLIP
    cache). digiKam-owned People/* tags are read first and rewritten after
    the wipe so face-recognition bookkeeping survives. Face regions in
    XMP-mwg-rs are never touched — that namespace is outside the wipe set.
    """
    if not paths:
        return True
    if dry_run:
        log.info("[DRY RUN] Would clear ALL tags from %d file(s)", len(paths))
        return True

    batch_size = get_config().exiftool.batch_size
    success = True

    clear_args = [f"-{field}=" for field in _ALL_TAG_FIELDS]
    clear_args.append("-XMP-phototools:all=")

    people_by_path = _read_people_tags(paths)
    no_people = [p for p in paths if not people_by_path.get(p)]
    with_people = [(p, people_by_path[p]) for p in paths if people_by_path.get(p)]

    # Fast path: files with no People/* tags — batch-wipe.
    for i in range(0, len(no_people), batch_size):
        batch = no_people[i:i + batch_size]
        args = ["-overwrite_original", *clear_args]
        targets: list[Path] = []
        for p in batch:
            targets.extend(_write_targets(p))
        targets = [t for t in targets if t.suffix.lower() != ".xmp" or t.exists()]
        try:
            ok, err = _exec_write(args, targets)
            if not ok:
                log.warning("Could not clear tags: %s", err)
                success = False
        except Exception as e:
            log.warning("Error clearing tags: %s", e)
            success = False

    # Per-file path: wipe and restore People/* in a single exiftool call.
    for path, people_tags in with_people:
        args = ["-overwrite_original", *clear_args]
        for tag in people_tags:
            args.extend(_build_tag_args(tag, "+="))
        targets = [t for t in _write_targets(path)
                   if t.suffix.lower() != ".xmp" or t.exists()]
        try:
            ok, err = _exec_write(args, targets)
            if not ok:
                log.warning("Could not clear tags from %s: %s", path.name, err)
                success = False
        except Exception as e:
            log.warning("Error clearing tags from %s: %s", path.name, e)
            success = False

    return success


# ---------------------------------------------------------------------------
# EXIF reading
# ---------------------------------------------------------------------------

def read_exif(path: Path) -> dict:
    fields = [
        "-GPS:GPSLatitude", "-GPS:GPSLongitude",
        "-GPS:GPSLatitudeRef", "-GPS:GPSLongitudeRef",
        "-EXIF:Make", "-EXIF:Model",
        "-EXIF:DateTimeOriginal", "-EXIF:CreateDate",
        "-EXIF:ImageWidth", "-EXIF:ImageHeight",
        "-File:ImageWidth", "-File:ImageHeight",
        "-IPTC:Keywords", "-XMP:Subject", "-XMP-digiKam:TagsList",
        "-XMP-phototools:TaggerVersion",
        "-XMP-iptcExt:ImageRegion",
        "-struct", "-n",
    ]
    try:
        files, str_to_path = _expand_paths_with_sidecars([path])
        metas = _run_exiftool_json(fields + files, timeout=30)
        if not metas:
            return {}
        return _group_metas_by_path(metas, str_to_path).get(path, {})
    except Exception as e:
        log.warning("Could not read EXIF from %s: %s", path, e)
        return {}


def existing_non_ocr_regions(exif: dict) -> list[dict]:
    """Return IPTC ImageRegion entries whose role is NOT 'annotatedText'.

    Used to preserve face/manual regions when rewriting OCR regions.
    Operates on an `exif` dict produced by `read_exif` (which fetches
    `XMP-iptcExt:ImageRegion` with `-struct`).
    """
    out = []
    for r in exif.get("ImageRegion") or []:
        roles = r.get("RRole") or []
        is_ocr = any(
            "annotatedText" in (ident or "")
            for role in roles for ident in (role.get("Identifier") or [])
        )
        if not is_ocr:
            out.append(r)
    return out


def get_existing_keywords(exif: dict) -> set[str]:
    """Return existing keyword leaves (lowercased) from a file's metadata.

    Used to skip writing tags that are already present (case-insensitive
    leaf-name match against dc:subject / IPTC:Keywords / digiKam:TagsList).
    """
    keywords = set()
    for field in ("Keywords", "Subject", "TagsList"):
        val = exif.get(field, [])
        if isinstance(val, str):
            val = [val]
        for v in val:
            s = str(v)
            keywords.add(s.rsplit("/", 1)[-1].lower())
    return keywords


def get_tagger_version(exif: dict) -> str | None:
    """Return photo-tools:TaggerVersion if present (sentinel)."""
    v = exif.get("TaggerVersion")
    return str(v) if v else None


def read_keywords_batch(paths: list[Path]) -> dict[Path, set[str]]:
    """Read IPTC:Keywords + XMP:Subject from all paths in batched exiftool calls."""
    if not paths:
        return {}

    batch_size = get_config().exiftool.batch_size
    result: dict[Path, set[str]] = {}

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(
            ["-IPTC:Keywords", "-XMP:Subject", "-XMP-digiKam:TagsList"]
            + files,
            with_config=False, timeout=120,
        )
        if not meta_list:
            for p in batch:
                exif = read_exif(p)
                result[p] = get_existing_keywords(exif)
            continue

        for path, meta in _group_metas_by_path(meta_list, str_to_path).items():
            result[path] = get_existing_keywords(meta)

    return result


def read_dates_batch(paths: list[Path]) -> dict[Path, dict]:
    """Read EXIF/QuickTime date fields from many files in batched exiftool calls.

    Returns {path: {"DateTimeOriginal": str|None, "CreateDate": str|None}}.
    Files absent from the returned dict had no readable metadata at all.
    Used by `dates backfill` to detect which files are missing a date.
    """
    if not paths:
        return {}

    batch_size = get_config().exiftool.batch_size
    result: dict[Path, dict] = {}

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(
            ["-DateTimeOriginal", "-CreateDate"] + files,
            with_config=False, timeout=120,
        )
        if not meta_list:
            continue

        for path, meta in _group_metas_by_path(meta_list, str_to_path).items():
            result[path] = {
                "DateTimeOriginal": meta.get("DateTimeOriginal"),
                "CreateDate": meta.get("CreateDate"),
            }

    return result


def write_dates(
    path: Path,
    dt: datetime,
    source: str,
    *,
    dry_run: bool = False,
) -> bool:
    """Write DateTimeOriginal/CreateDate (and friends) to a single file.

    Picks EXIF tags for images and QuickTime tags for videos, plus XMP
    mirrors that survive container-agnostic round-trips. Records
    `XMP-phototools:DateBackfilledFrom=<source>` so the write is auditable.
    """
    stamp = dt.strftime("%Y:%m:%d %H:%M:%S")
    if dry_run:
        log.info("[DRY RUN] Would set %s on %s (source: %s)",
                 stamp, path.name, source)
        return True

    args = ["-overwrite_original"]
    if is_video(path):
        # Videos go to sidecar only (see _write_targets) — QuickTime
        # writes are skipped for videos so the container is untouched.
        pass
    else:
        args += [
            f"-EXIF:DateTimeOriginal={stamp}",
            f"-EXIF:CreateDate={stamp}",
            f"-EXIF:ModifyDate={stamp}",
            f"-XMP-photoshop:DateCreated={stamp}",
        ]
    args += [
        f"-XMP-xmp:CreateDate={stamp}",
        f"-XMP-phototools:DateBackfilledFrom={source}",
    ]

    try:
        ok, err = _exec_write(args, _write_targets(path))
        if not ok:
            log.error("exiftool failed for %s: %s", path.name, err)
            return False
        log.debug("Wrote %s to %s (source: %s)", stamp, path.name, source)
        return True
    except FileNotFoundError:
        log.error("exiftool not found. Install with: brew install exiftool")
        sys.exit(1)
    except Exception as e:
        log.error("Failed to write date to %s: %s", path.name, e)
        return False


def read_tagger_versions_batch(paths: list[Path]) -> dict[Path, str]:
    """Read XMP-phototools:TaggerVersion from many files in batched exiftool calls.

    Only returns entries where the sentinel is present; files without the
    field are absent from the dict (callers treat missing as "not yet tagged").
    """
    if not paths:
        return {}

    batch_size = get_config().exiftool.batch_size
    total = len(paths)
    n_batches = (total + batch_size - 1) // batch_size
    result: dict[Path, str] = {}

    for batch_idx, i in enumerate(range(0, total, batch_size), 1):
        batch = paths[i:i + batch_size]
        log.info("Reading TaggerVersion batch %d/%d (%d files, %d/%d total)",
                 batch_idx, n_batches, len(batch),
                 min(i + len(batch), total), total)
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(
            ["-XMP-phototools:TaggerVersion"] + files,
            timeout=120,
        )
        if not meta_list:
            continue

        for path, meta in _group_metas_by_path(meta_list, str_to_path).items():
            version = meta.get("TaggerVersion")
            if version:
                result[path] = str(version)

    return result


# ---------------------------------------------------------------------------
# Image preparation (convert + downsize)
# ---------------------------------------------------------------------------

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HAVE_PILLOW_HEIF = True
except ImportError:
    _HAVE_PILLOW_HEIF = False

try:
    from PIL import Image as PILImage
    from PIL import ImageOps
    _HAVE_PILLOW = True
except ImportError:
    _HAVE_PILLOW = False


def open_and_rotate(path: Path):
    """Open an image and apply EXIF orientation.

    Returns a fully-loaded PIL Image with the original file handle
    released. Caller is responsible for any further mode conversion
    (`.convert("RGB")`) or close().
    """
    img = PILImage.open(str(path))
    rotated = ImageOps.exif_transpose(img)
    rotated.load()
    img.close()
    return rotated


def _try_pillow(path: Path, tmp_path: str, max_pixels: int) -> bool:
    if not _HAVE_PILLOW:
        return False
    if path.suffix.lower() in (".heic", ".heif") and not _HAVE_PILLOW_HEIF:
        return False
    try:
        cfg = get_config()
        img = open_and_rotate(path)
        img.thumbnail((max_pixels, max_pixels), PILImage.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(tmp_path, "JPEG", quality=cfg.image.jpeg_quality)
        return Path(tmp_path).stat().st_size > 0
    except Exception as e:
        log.debug("Pillow downscale failed for %s: %s", path.name, e)
        return False


# EXIF Orientation (1-8) → ffmpeg filter chain to bring pixels upright.
# See https://exiftool.org/TagNames/EXIF.html#Orientation
_FFMPEG_ORIENT_FILTERS = {
    1: "",
    2: "hflip",
    3: "hflip,vflip",
    4: "vflip",
    5: "transpose=0",
    6: "transpose=1",
    7: "transpose=3",
    8: "transpose=2",
}


def _read_exif_orientation(path: Path) -> int:
    """Return EXIF Orientation (1-8), or 1 if unknown/unreadable."""
    try:
        result = _run_exiftool(
            ["-s3", "-n", "-Orientation", str(path)],
            with_config=False, timeout=10,
        )
        if result.returncode == 0:
            val = result.stdout.strip()
            if val.isdigit():
                n = int(val)
                if 1 <= n <= 8:
                    return n
    except Exception as e:
        log.debug("Failed to read EXIF orientation for %s: %s", path.name, e)
    return 1


def _try_ffmpeg(path: Path, tmp_path: str, max_pixels: int) -> bool:
    scale_filter = (
        f"scale='if(gt(iw,ih),{max_pixels},-2)':'if(gt(iw,ih),-2,{max_pixels})'"
    )
    orient_filter = _FFMPEG_ORIENT_FILTERS[_read_exif_orientation(path)]
    vf = f"{orient_filter},{scale_filter}" if orient_filter else scale_filter
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-noautorotate", "-i", str(path), "-frames:v", "1",
             "-vf", vf, "-q:v", "2", tmp_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and Path(tmp_path).stat().st_size > 0
    except Exception:
        return False


def _try_magick(path: Path, tmp_path: str, max_pixels: int) -> bool:
    cfg = get_config()
    for cmd in ["magick", "convert"]:
        try:
            result = subprocess.run(
                [cmd, str(path), "-auto-orient",
                 "-resize", f"{max_pixels}x{max_pixels}>",
                 "-quality", str(cfg.image.jpeg_quality), tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and Path(tmp_path).stat().st_size > 0:
                return True
        except Exception:
            continue
    return False


def prepare_image(path: Path, max_pixels: int) -> Path | None:
    """Convert any image to JPEG and downsize to max_pixels on the long edge.
    Returns path to a temp JPEG, or None on failure. Caller must delete the file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    tmp_path = tmp.name

    try:
        if _try_pillow(path, tmp_path, max_pixels):
            return Path(tmp_path)
        if _try_ffmpeg(path, tmp_path, max_pixels):
            return Path(tmp_path)
        if _try_magick(path, tmp_path, max_pixels):
            return Path(tmp_path)

        log.warning("Could not convert %s — install pillow-heif, ffmpeg, or imagemagick",
                    path.name)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None

    except Exception as e:
        log.warning("prepare_image error for %s: %s", path.name, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def detect_real_type(path: Path) -> str | None:
    """Check actual file type by reading magic bytes. Returns 'image', 'video', or None."""
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except Exception:
        return None

    if header[:2] == b"\xff\xd8":
        return "image"
    if header[:4] == b"\x89PNG":
        return "image"
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image"
    if header[4:8] == b"ftyp":
        ftyp = header[8:12].lower()
        if ftyp in (b"heic", b"heix", b"mif1", b"msf1"):
            return "image"
        if ftyp in (b"qt  ", b"mp41", b"mp42", b"isom", b"m4v ", b"msnv", b"avc1"):
            return "video"
    if header[4:8] == b"ftyp":
        return "video"
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "video"
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return "video"

    return None


def is_video(path: Path) -> bool:
    """Check if a file is a video, by extension or magic bytes."""
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return True
    return detect_real_type(path) == "video"


def is_live_photo_motion(path: Path) -> bool:
    """True if filename looks like an Apple Live Photo motion companion.

    Heuristic: a .mov whose stem (case-insensitive) ends in a still-image
    extension, e.g. IMG_1353.HEIC.MOV or IMG_1353.jpg.mov. The corresponding
    still file is NOT required to exist — this is purely a filename-shape
    check.
    """
    if path.suffix.lower() != ".mov":
        return False
    stem = path.stem.lower()
    return any(stem.endswith(ext) for ext in (".jpg", ".jpeg", ".heic", ".heif"))


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frame(path: Path) -> Path | None:
    """Extract a single representative frame from a video using ffmpeg."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", "1", "-i", str(path),
             "-frames:v", "1", "-q:v", "2", tmp.name],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode != 0:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(path),
                 "-frames:v", "1", "-q:v", "2", tmp.name],
                capture_output=True, text=True, timeout=30,
            )

        if result.returncode != 0 or not Path(tmp.name).exists() or Path(tmp.name).stat().st_size == 0:
            log.warning("ffmpeg could not extract frame from %s", path.name)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return None

        return Path(tmp.name)

    except FileNotFoundError:
        log.warning("ffmpeg not found — skipping video. Install: brew install ffmpeg")
        return None
    except Exception as e:
        log.warning("Frame extraction failed for %s: %s", path.name, e)
        return None


# ---------------------------------------------------------------------------
# CLIP embedding cache (stored in XMP, reused by `duplicates`)
# ---------------------------------------------------------------------------


def read_cached_embeddings_batch(
    paths: list[Path], model: str
) -> dict[Path, np.ndarray]:
    """Read CLIP embeddings for many files in batched exiftool calls.

    Only returns entries whose CLIPModel matches `model`; files with no
    cached embedding, wrong model, or decode errors are silently omitted
    (callers treat them as cache misses and fall back to computing).
    """
    if not paths:
        return {}

    batch_size = get_config().exiftool.batch_size
    total = len(paths)
    n_batches = (total + batch_size - 1) // batch_size
    result: dict[Path, np.ndarray] = {}

    for batch_idx, i in enumerate(range(0, total, batch_size), 1):
        batch = paths[i:i + batch_size]
        log.info("Reading embedding batch %d/%d (%d files, %d/%d total)",
                 batch_idx, n_batches, len(batch),
                 min(i + len(batch), total), total)
        files, str_to_path = _expand_paths_with_sidecars(batch)
        meta_list = _run_exiftool_json(
            ["-XMP-phototools:CLIPEmbedding",
             "-XMP-phototools:CLIPModel"]
            + files,
            timeout=120,
        )
        if not meta_list:
            continue

        hits = 0
        for path, meta in _group_metas_by_path(meta_list, str_to_path).items():
            if meta.get("CLIPModel") != model:
                continue
            b64 = meta.get("CLIPEmbedding", "")
            if not b64:
                continue
            try:
                result[path] = np.frombuffer(
                    base64.b64decode(b64), dtype=np.float32
                ).copy()
                hits += 1
            except Exception:
                continue
        log.debug("Batch %d/%d: %d/%d cache hits",
                  batch_idx, n_batches, hits, len(batch))

    return result


def write_embedding(path: Path, vec: np.ndarray, model: str, dry_run: bool) -> None:
    b64 = base64.b64encode(vec.astype(np.float32).tobytes()).decode()
    if dry_run:
        log.info("[DRY RUN] Would cache embedding for %s", path.name)
        return
    args = [
        "-overwrite_original",
        f"-XMP-phototools:CLIPEmbedding={b64}",
        f"-XMP-phototools:CLIPModel={model}",
    ]
    ok, err = _exec_write(args, _write_targets(path))
    if not ok:
        log.warning("Failed to cache embedding for %s: %s", path.name, err)


# IPTC role identifier marking a region as OCR-derived text. Mirrored on read
# in `existing_non_ocr_regions` so we can rewrite OCR regions without
# clobbering face/manual regions other tools have written.
_OCR_ROLE_IDENTIFIER = "http://cv.iptc.org/newscodes/imageregionrole/annotatedText"


# Map of our location-part keys to the exiftool field names they fan out to.
# The first entry in each tuple is XMP (always written), the second is the
# legacy IPTC IIM field (skipped on the sidecar phase by _xmp_only_args).
# See docs/xmp-schema.md §1.5.
_LOCATION_FIELDS = {
    "Country":     ("XMP-photoshop:Country",   "IPTC:Country-PrimaryLocationName"),
    "CountryCode": ("XMP-iptcCore:CountryCode", "IPTC:Country-PrimaryLocationCode"),
    "State":       ("XMP-photoshop:State",     "IPTC:Province-State"),
    "City":        ("XMP-photoshop:City",      "IPTC:City"),
    "Sublocation": ("XMP-iptcCore:Location",   "IPTC:Sub-location"),
}


def _location_field_args(location_fields: dict[str, str]) -> list[str]:
    """Build exiftool args to set IPTC-standard location fields.

    Each key writes to both an XMP and an IPTC IIM field for cross-app
    compatibility. Empty values clear the corresponding fields.
    """
    args: list[str] = []
    for key, value in location_fields.items():
        targets = _LOCATION_FIELDS.get(key)
        if not targets:
            continue
        for field in targets:
            args.append(f"-{field}={value}")
    return args


def write_metadata(
    path: Path,
    *,
    new_keywords: list[str] = (),
    namespace_fields: dict[str, str] | None = None,
    location_fields: dict[str, str] | None = None,
    person_in_image: list[str] | None = None,
    ocr_text: list[str] | None = None,
    new_ocr_regions: list[dict] | None = None,
    existing_iptc_regions: list[dict] | None = None,
    embedding: np.ndarray | None = None,
    embedding_model: str | None = None,
    stamp_ocr_ran: bool = False,
    dry_run: bool = False,
) -> bool:
    """Single-spawn metadata write for the autotag inner loop.

    Consolidates keyword adds, photo-tools namespace fields, IPTC location
    fields, PersonInImage projection, OCR text/regions, OCRRan stamp, CLIP
    embedding, and the TaggerVersion+TaggedAt sentinel into one exiftool
    invocation. Keywords use `+=` so non-photo-tools tags are preserved;
    structural fields (IPTC:ImageRegion, OCRText) ride along in a JSON
    sidecar passed in the same exiftool call.

    `person_in_image` is a list of person leaf names; when supplied, the
    XMP-iptcExt:PersonInImage list is replaced (not appended) so it stays
    in sync with the file's current People/* keywords.
    """
    has_anything = bool(
        new_keywords or namespace_fields or location_fields
        or person_in_image is not None or ocr_text or new_ocr_regions
        or embedding is not None or stamp_ocr_ran
    )
    if not has_anything:
        return False

    if dry_run:
        log.info("[DRY RUN] Would write to %s:", path.name)
        if new_keywords:
            log.info("  %d keyword(s):", len(new_keywords))
            for kw in sorted(new_keywords):
                log.info("    %s", kw)
        for k, v in sorted((namespace_fields or {}).items()):
            log.info("    photo-tools:%s = %s", k, v)
        for k, v in sorted((location_fields or {}).items()):
            log.info("    location:%s = %s", k, v)
        if person_in_image is not None:
            log.info("    PersonInImage = %s", ", ".join(person_in_image) or "(cleared)")
        if new_ocr_regions:
            log.info("  %d OCR region(s)", len(new_ocr_regions))
        if stamp_ocr_ran:
            log.info("    photo-tools:OCRRan")
        if embedding is not None:
            log.info("    photo-tools:CLIPEmbedding (%s)", embedding_model or "?")
        return True

    ts = datetime.now(UTC).isoformat()
    args = ["-overwrite_original"]

    for kw in new_keywords:
        args.extend(_build_tag_args(kw, "+="))

    if location_fields:
        args.extend(_location_field_args(location_fields))

    if person_in_image is not None:
        # Replace, not append: keep the list in sync with the current set
        # of People/* leaves. `=` clears, then `+=` rebuilds.
        args.append("-XMP-iptcExt:PersonInImage=")
        for name in person_in_image:
            args.append(f"-XMP-iptcExt:PersonInImage+={name}")

    args.extend([
        f"-XMP-phototools:TaggerVersion={TAGGER_VERSION}",
        f"-XMP-phototools:TaggedAt={ts}",
    ])
    for field, value in (namespace_fields or {}).items():
        args.append(f"-XMP-phototools:{field}={value}")
    if stamp_ocr_ran:
        args.append(f"-XMP-phototools:OCRRan={ts}")
    if embedding is not None:
        b64 = base64.b64encode(embedding.astype(np.float32).tobytes()).decode()
        args.append(f"-XMP-phototools:CLIPEmbedding={b64}")
        if embedding_model:
            args.append(f"-XMP-phototools:CLIPModel={embedding_model}")

    targets = _write_targets(path)

    tmp_path: str | None = None
    if new_ocr_regions:
        iptc_regions = list(existing_iptc_regions or [])
        for r in new_ocr_regions:
            iptc_regions.append({
                "Name": r["text"],
                "RRole": [{
                    "Identifier": [_OCR_ROLE_IDENTIFIER],
                    "Name": "annotated text",
                }],
                "RegionBoundary": {
                    "RbShape": "rectangle", "RbUnit": "relative",
                    "RbX": round(r["x"], 5), "RbY": round(r["y"], 5),
                    "RbW": round(r["w"], 5), "RbH": round(r["h"], 5),
                },
            })
        # One JSON entry per target so exiftool routes structures to
        # both the image and the sidecar via SourceFile matching.
        sidecar_entries = []
        for t in targets:
            entry: dict = {
                "SourceFile": str(t),
                "XMP-iptcExt:ImageRegion": iptc_regions,
            }
            if ocr_text:
                entry["XMP-phototools:OCRText"] = ocr_text
            sidecar_entries.append(entry)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(sidecar_entries, tmp)
        tmp.close()
        tmp_path = tmp.name
        args = ["-struct", f"-json={tmp_path}"] + args
    elif ocr_text:
        # OCR text without regions — clear + append on CLI to replace any prior list
        args.append("-XMP-phototools:OCRText=")
        for phrase in ocr_text:
            args.append(f"-XMP-phototools:OCRText+={phrase}")

    try:
        ok, err = _exec_write(args, targets, timeout=60)
        if not ok:
            log.error("exiftool failed for %s: %s", path.name, err)
            return False
        parts = []
        if new_keywords:
            parts.append(f"{len(new_keywords)} keyword(s)")
        if new_ocr_regions:
            parts.append(f"{len(new_ocr_regions)} OCR region(s)")
        if embedding is not None:
            parts.append("embedding")
        if parts:
            log.info("Wrote %s to %s", " + ".join(parts), path.name)
        else:
            log.debug("Updated metadata sentinel on %s", path.name)
        return True
    except FileNotFoundError:
        log.error("exiftool not found. Install with: brew install exiftool")
        sys.exit(1)
    except Exception as e:
        log.error("Failed to write metadata to %s: %s", path.name, e)
        return False
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_images(target: Path) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [target]
        log.warning("Unsupported file type: %s", target)
        return []
    if not target.is_dir():
        log.error("Path does not exist: %s", target)
        return []

    images = []
    for f in target.glob("**/*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(f)

    seen = set()
    deduped = []
    for img in sorted(images):
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(img)
    return deduped

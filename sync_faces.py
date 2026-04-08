"""
sync_faces.py — Sync face regions from a DigiKam database to image file metadata.

Reads confirmed face tags (person names + bounding boxes) from DigiKam's SQLite
database and writes them as MWG/IPTC face regions plus person/* keywords via ExifTool.

Only confirmed faces are synced (property='tagRegion' in DigiKam's ImageTagProperties).
Unconfirmed suggestions and auto-detected faces are skipped.

Usage:
    photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db
    photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r --dry-run
    photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r --force
"""

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from autotag import (
    find_images,
    get_existing_keywords,
    hierarchical_subject,
    read_exif,
    write_keywords,
    _read_existing_regions,
)

log = logging.getLogger("sync_faces")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

FACE_QUERY = """\
SELECT
    ar.specificPath || al.relativePath || '/' || im.name  AS file_path,
    t.name                                                 AS person_name,
    itp.value                                              AS region_xml,
    ii.width                                               AS img_width,
    ii.height                                              AS img_height,
    ii.orientation                                         AS exif_orientation
FROM ImageTagProperties itp
JOIN Images im            ON im.id = itp.imageid
JOIN Albums al            ON al.id = im.album
JOIN AlbumRoots ar        ON ar.id = al.albumRoot
JOIN Tags t               ON t.id = itp.tagid
JOIN TagProperties tp     ON tp.tagid = t.id AND tp.property = 'person'
JOIN ImageInformation ii  ON ii.imageid = im.id
WHERE itp.property = 'tagRegion'
  AND im.status = 1
"""


@dataclass
class FaceRecord:
    name: str
    region_xml: str
    img_w: int
    img_h: int
    orientation: int


@dataclass
class ImageFaces:
    img_w: int
    img_h: int
    orientation: int
    faces: list[FaceRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DigiKam DB reader
# ---------------------------------------------------------------------------

def read_digikam_faces(db_path: Path) -> dict[Path, ImageFaces]:
    """Read all confirmed face regions from DigiKam database.

    Returns dict mapping resolved file paths to their face data.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError as e:
        log.error("Cannot open DigiKam database %s: %s", db_path, e)
        sys.exit(1)

    try:
        rows = conn.execute(FACE_QUERY).fetchall()
    except sqlite3.OperationalError as e:
        log.error("Failed to query DigiKam database: %s", e)
        sys.exit(1)
    finally:
        conn.close()

    grouped: dict[Path, ImageFaces] = {}
    for file_path_str, person_name, region_xml, img_w, img_h, orientation in rows:
        filepath = Path(file_path_str).resolve()

        if not img_w or not img_h:
            log.warning("Skipping %s: missing image dimensions in DigiKam DB", filepath.name)
            continue

        if filepath not in grouped:
            grouped[filepath] = ImageFaces(
                img_w=img_w,
                img_h=img_h,
                orientation=orientation or 1,
            )
        grouped[filepath].faces.append(FaceRecord(
            name=person_name,
            region_xml=region_xml,
            img_w=img_w,
            img_h=img_h,
            orientation=orientation or 1,
        ))

    return grouped


# ---------------------------------------------------------------------------
# Region XML parser
# ---------------------------------------------------------------------------

def parse_region_xml(xml_str: str) -> dict | None:
    """Parse DigiKam region XML: <rect x="..." y="..." width="..." height="..."/>

    Returns dict with x, y, width, height as ints, or None on failure.
    """
    try:
        root = ET.fromstring(xml_str)
        return {
            "x": int(root.get("x")),
            "y": int(root.get("y")),
            "width": int(root.get("width")),
            "height": int(root.get("height")),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coordinate transform: DigiKam -> MWG
# ---------------------------------------------------------------------------

def transform_region(
    rect: dict, img_w: int, img_h: int, orientation: int,
) -> tuple[float, float, float, float, int, int]:
    """Transform DigiKam pixel rect to MWG normalized center-point coords.

    DigiKam stores absolute pixel coords, top-left origin, pre-EXIF-orientation.
    MWG needs normalized 0-1, center-point, post-EXIF-orientation.

    Returns (cx, cy, w, h, display_w, display_h).
    """
    # Step 1: normalize to 0-1 in pre-orientation space
    nx = rect["x"] / img_w
    ny = rect["y"] / img_h
    nw = rect["width"] / img_w
    nh = rect["height"] / img_h

    # Step 2: apply EXIF orientation rotation
    if orientation in (1, 2):
        pass  # normal (mirror not handled, very rare from cameras)
    elif orientation in (3, 4):
        # 180 degrees
        nx = 1 - nx - nw
        ny = 1 - ny - nh
    elif orientation in (6, 7):
        # 90 CW
        nx, ny, nw, nh = 1 - ny - nh, nx, nh, nw
    elif orientation in (8, 5):
        # 270 CW
        nx, ny, nw, nh = ny, 1 - nx - nw, nh, nw
    else:
        log.warning("Unknown EXIF orientation %d, treating as normal", orientation)

    # Display dimensions (swapped for 90/270 rotations)
    if orientation in (5, 6, 7, 8):
        display_w, display_h = img_h, img_w
    else:
        display_w, display_h = img_w, img_h

    # Step 3: convert top-left to center-point for MWG
    cx = nx + nw / 2
    cy = ny + nh / 2

    return cx, cy, nw, nh, display_w, display_h


# ---------------------------------------------------------------------------
# Face region writer
# ---------------------------------------------------------------------------

def _is_face_region_mwg(region: dict) -> bool:
    return region.get("Type") == "Face"


def _is_face_region_iptc(region: dict) -> bool:
    """Check if an IPTC region is a face (has PersonInImage, not OCR)."""
    if region.get("PersonInImage"):
        return True
    roles = region.get("RRole") or []
    is_ocr = any(
        "annotatedText" in (ident or "")
        for role in roles
        for ident in (role.get("Identifier") or [])
    )
    return not is_ocr and region.get("Name") and not roles


def _face_already_exists(name: str, cx: float, cy: float, existing_mwg: list[dict],
                         tolerance: float = 0.05) -> bool:
    """Check if a face with the same name and approximate position already exists."""
    for r in existing_mwg:
        if r.get("Type") != "Face" or r.get("Name") != name:
            continue
        area = r.get("Area") or {}
        ex, ey = float(area.get("X", 0)), float(area.get("Y", 0))
        if abs(ex - cx) < tolerance and abs(ey - cy) < tolerance:
            return True
    return False


def write_face_regions(
    path: Path,
    faces: list[dict],
    applied_dims: dict,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """Write face regions to file metadata (MWG + IPTC).

    faces: list of dicts with keys: name, cx, cy, w, h (all normalized 0-1)
    applied_dims: dict with W, H for AppliedToDimensions
    force: if True, replace all existing face regions with the new set
    """
    if not faces and not force:
        return True

    # Read existing regions to preserve non-face regions
    existing_mwg, existing_iptc, existing_dims = _read_existing_regions(path)

    # Separate face vs non-face regions
    nonface_mwg = [r for r in existing_mwg if not _is_face_region_mwg(r)]
    nonface_iptc = [r for r in existing_iptc if not _is_face_region_iptc(r)]

    if force:
        # Replace all face regions with new set
        new_faces = faces
    else:
        # Only add faces not already present
        new_faces = [
            f for f in faces
            if not _face_already_exists(f["name"], f["cx"], f["cy"], existing_mwg)
        ]
        if not new_faces:
            log.debug("All faces already in metadata for %s, skipping", path.name)
            return True

    if dry_run:
        for f in new_faces:
            log.info("  [DRY RUN] Would write face region: %s", f["name"])
        return True

    # Build region lists: non-face preserved + new faces
    mwg_regions = list(nonface_mwg)
    iptc_regions = list(nonface_iptc)

    for f in new_faces:
        mwg_regions.append({
            "Name": f["name"],
            "Type": "Face",
            "Area": {
                "X": round(f["cx"], 6),
                "Y": round(f["cy"], 6),
                "W": round(f["w"], 6),
                "H": round(f["h"], 6),
                "Unit": "normalized",
            },
        })
        # IPTC uses top-left coords
        iptc_regions.append({
            "Name": f["name"],
            "PersonInImage": f["name"],
            "RegionBoundary": {
                "RbShape": "rectangle",
                "RbUnit": "relative",
                "RbX": round(f["cx"] - f["w"] / 2, 6),
                "RbY": round(f["cy"] - f["h"] / 2, 6),
                "RbW": round(f["w"], 6),
                "RbH": round(f["h"], 6),
            },
        })

    mwg_info: dict = {"RegionList": mwg_regions}
    if existing_dims:
        mwg_info["AppliedToDimensions"] = existing_dims
    else:
        mwg_info["AppliedToDimensions"] = {
            "W": applied_dims["W"],
            "H": applied_dims["H"],
            "Unit": "pixel",
        }

    meta = {
        "SourceFile": str(path),
        "XMP-iptcExt:ImageRegion": iptc_regions,
        "XMP-mwg-rs:RegionInfo": mwg_info,
    }

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump([meta], tmp)
        tmp.close()
        result = subprocess.run(
            ["exiftool", "-overwrite_original", "-struct", f"-json={tmp.name}", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            log.error("exiftool face region write failed for %s: %s",
                      path.name, result.stderr.strip())
            return False
        log.info("Wrote %d face region(s) to %s", len(new_faces), path.name)
        return True
    except FileNotFoundError:
        log.error("exiftool not found. Install with: brew install exiftool")
        sys.exit(1)
    except Exception as e:
        log.error("Failed to write face regions to %s: %s", path.name, e)
        return False
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Keyword management for person tags
# ---------------------------------------------------------------------------

def _remove_person_keywords(path: Path, keywords_to_remove: list[str],
                            dry_run: bool = False) -> bool:
    """Remove specific person/* keywords from file metadata."""
    if not keywords_to_remove:
        return True
    if dry_run:
        for kw in keywords_to_remove:
            log.info("  [DRY RUN] Would remove keyword: %s", kw)
        return True

    args = ["exiftool", "-overwrite_original"]
    for kw in keywords_to_remove:
        args.append(f"-IPTC:Keywords-={kw}")
        args.append(f"-XMP-dc:Subject-={kw}")
        args.append(f"-XMP-lr:HierarchicalSubject-={hierarchical_subject(kw)}")
        args.append(f"-XMP-digiKam:TagsList-={kw}")
        # Also remove the leaf form (e.g. just "Alice" for "person/Alice")
        leaf = kw.rsplit("/", 1)[-1]
        if leaf != kw:
            args.append(f"-IPTC:Keywords-={leaf}")
            args.append(f"-XMP-dc:Subject-={leaf}")
    args.append(str(path))

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.warning("Could not remove person keywords from %s: %s",
                        path.name, result.stderr.strip())
            return False
        return True
    except Exception as e:
        log.warning("Error removing person keywords from %s: %s", path.name, e)
        return False


# ---------------------------------------------------------------------------
# Per-image sync
# ---------------------------------------------------------------------------

def _sync_single(
    path: Path, image_faces: ImageFaces, dry_run: bool, force: bool,
) -> bool:
    """Sync face data for a single image. Returns True if changes were made."""
    faces = []
    for fr in image_faces.faces:
        parsed = parse_region_xml(fr.region_xml)
        if parsed is None:
            log.warning("  Skipping unparseable region for '%s'", fr.name)
            continue
        cx, cy, w, h, dw, dh = transform_region(
            parsed, image_faces.img_w, image_faces.img_h, image_faces.orientation,
        )
        faces.append({
            "name": fr.name,
            "cx": cx, "cy": cy, "w": w, "h": h,
        })

    if not faces and not force:
        return False

    # Display dimensions (from first face or computed)
    if faces:
        parsed0 = parse_region_xml(image_faces.faces[0].region_xml)
        _, _, _, _, dw, dh = transform_region(
            parsed0, image_faces.img_w, image_faces.img_h, image_faces.orientation,
        )
    elif image_faces.orientation in (5, 6, 7, 8):
        dw, dh = image_faces.img_h, image_faces.img_w
    else:
        dw, dh = image_faces.img_w, image_faces.img_h
    applied_dims = {"W": dw, "H": dh}

    # Write face regions
    ok = write_face_regions(path, faces, applied_dims, dry_run, force)
    if not ok:
        return False

    # Handle person keywords
    exif = read_exif(path)
    existing_kw = get_existing_keywords(exif)
    db_person_keywords = list({f"person/{f['name']}" for f in faces})

    if force:
        # Remove stale person/* keywords not in the DB
        existing_person_kw = [kw for kw in existing_kw if kw.startswith("person/")]
        stale = [kw for kw in existing_person_kw if kw not in {k.lower() for k in db_person_keywords}]
        if stale:
            _remove_person_keywords(path, stale, dry_run)

    # Add new person keywords not already present
    new_keywords = [kw for kw in db_person_keywords if kw.lower() not in existing_kw]
    if new_keywords:
        if dry_run:
            for kw in new_keywords:
                log.info("  [DRY RUN] Would write keyword: %s", kw)
        else:
            write_keywords(path, new_keywords)

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_sync_faces_parser(subparsers) -> argparse.ArgumentParser:
    """Register the 'sync-faces' subcommand."""
    sub = subparsers.add_parser(
        "sync-faces",
        help="Sync face regions from DigiKam database to image file metadata.",
    )
    sub.add_argument("path", type=Path,
                     help="Image file or directory to sync")
    sub.add_argument("--db", type=Path, required=True,
                     help="Path to DigiKam SQLite database (digikam4.db)")
    sub.add_argument("-r", "--recursive", action="store_true",
                     help="Process directories recursively")
    sub.add_argument("-n", "--dry-run", action="store_true",
                     help="Preview changes without writing")
    sub.add_argument("-f", "--force", action="store_true",
                     help="Replace all existing face regions with current DB state")
    sub.add_argument("-v", "--verbose", action="store_true")
    sub.set_defaults(func=run_sync_faces)
    return sub


def run_sync_faces(args) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_path = args.db.resolve()
    if not db_path.exists():
        log.error("DigiKam database not found: %s", db_path)
        sys.exit(1)

    target = args.path.resolve()
    if not target.exists():
        log.error("Path does not exist: %s", target)
        sys.exit(1)

    log.info("Reading face data from %s ...", db_path)
    all_faces = read_digikam_faces(db_path)
    if not all_faces:
        log.info("No confirmed face data found in DigiKam database")
        return

    log.info("Found face data for %d image(s) in DigiKam database", len(all_faces))

    # Discover target files on disk
    target_files = find_images(args.path, args.recursive)
    if not target_files:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    # Match DB records to target files by resolved path
    target_set = {f.resolve(): f for f in target_files}
    matched: list[tuple[Path, ImageFaces]] = []
    for db_path_resolved, image_faces in all_faces.items():
        if db_path_resolved in target_set:
            matched.append((target_set[db_path_resolved], image_faces))

    if not matched:
        log.info("No DigiKam face data matches files under %s", args.path)
        return

    log.info("Matched %d image(s) with face data", len(matched))

    success = skipped = failed = 0
    total_faces = 0

    for i, (img_path, image_faces) in enumerate(sorted(matched, key=lambda x: x[0]), 1):
        log.info("[%d/%d] %s (%d face(s))", i, len(matched), img_path.name,
                 len(image_faces.faces))
        try:
            wrote = _sync_single(img_path, image_faces, args.dry_run, args.force)
            if wrote:
                success += 1
                total_faces += len(image_faces.faces)
            else:
                skipped += 1
        except Exception as e:
            log.error("Error processing %s: %s", img_path.name, e)
            failed += 1

    log.info("Done. %d image(s) synced (%d faces), %d skipped, %d failed.",
             success, total_faces, skipped, failed)

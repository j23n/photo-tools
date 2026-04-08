# photo-tools

Auto-tag photos using CLIP zero-shot classification, GPS reverse geocoding,
PaddleOCR, and EXIF metadata, then write IPTC/XMP keywords via ExifTool
(with DigiKam/Lightroom hierarchical tag support).

## Requirements

- Python 3.12+
- [ExifTool](https://exiftool.org/) (`brew install exiftool` / `sudo apt install libimage-exiftool-perl`)
- [FFmpeg](https://ffmpeg.org/) (optional, for video frame extraction)

## Installation

```bash
uv sync
```

## Commands

### `tag` — Auto-tag photos

Tag photos using multiple AI/metadata sources: CLIP vision model, GPS reverse
geocoding, PaddleOCR text detection, EXIF metadata, and landmark recognition.

```bash
photo-tools tag photo.jpg                        # Tag a single file
photo-tools tag ~/Pictures/Vacation/ -r          # Recursive directory
photo-tools tag ~/Pictures/ -r --dry-run         # Preview without writing
photo-tools tag ~/Pictures/ -r --force           # Re-tag (replaces old AI tags)
photo-tools tag ~/Pictures/ --watch              # Watch for new files
photo-tools tag photo.jpg --no-clip              # Only geo + EXIF + OCR
```

### `sync-faces` — Sync DigiKam face tags to file metadata

Read confirmed face regions from a DigiKam database and write them as
MWG/IPTC face regions plus `person/Name` keywords to image files.

```bash
photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r
photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r --dry-run
photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r --force
```

Options:
- `--db PATH` — Path to DigiKam SQLite database (required)
- `-r` — Process directories recursively
- `-n / --dry-run` — Preview changes without writing
- `-f / --force` — Replace all existing face regions with current DB state (handles renames/deletions)
- `-v / --verbose` — Show debug output

Without `--force`, new faces are appended and existing faces are left in place.
With `--force`, all face regions and `person/*` keywords are replaced with the
current DigiKam state, handling renamed or deleted faces.

Only confirmed faces are synced. Unconfirmed suggestions and auto-detected
faces in DigiKam are skipped.

### `find-similar` — Detect duplicate/similar images

Find visually similar images using CLIP embeddings cached in XMP metadata.

```bash
photo-tools find-similar ~/Pictures/ -r
photo-tools find-similar ~/Pictures/ -r --threshold 0.85
```

### `tags` — Tag management

List, search, delete, and rename tags across a photo collection.

```bash
photo-tools tags list ~/Pictures/ -r
photo-tools tags search ~/Pictures/ -r "animal/cat"
photo-tools tags delete ~/Pictures/ -r "text/hello"
photo-tools tags rename ~/Pictures/ -r "old/tag" "new/tag"
```

### `build-landmarks` — Build landmark database

Build a CLIP embedding database of notable landmarks from Wikidata.

```bash
photo-tools build-landmarks
photo-tools build-landmarks --limit 5000 --resume
```

## Tag Taxonomy

Tags use `/` as hierarchy separator (compatible with DigiKam and Lightroom):

| Source | Prefixes |
|--------|----------|
| GPS | `country/` `cc/` `region/` `city/` `neighborhood/` |
| CLIP | `animal/` `food/` `plant/` `vehicle/` `object/` `scene/` `activity/` `event/` |
| CLIP enums | `weather/` `setting/` `time/` `landmark/` |
| OCR | `text/` |
| EXIF | `year/` `month/` `day/` `flash/` |
| Faces | `person/` |
| Flags | `weekend` `weekday` `screenshot` `video` `ai:tagged` |

Keywords are written to four metadata fields for maximum compatibility:
- `IPTC:Keywords`
- `XMP-dc:Subject`
- `XMP-lr:HierarchicalSubject` (pipe-delimited for Lightroom)
- `XMP-digiKam:TagsList`

## DigiKam Setup for Face Sync

The `sync-faces` command reads face data from DigiKam's SQLite database. To use it:

### Finding Your Database Path

DigiKam stores its database as `digikam4.db`. To find it:

1. Open DigiKam
2. Go to **Settings > Configure digiKam > Database**
3. The database location is shown under "Database File Path"

Common locations:
- In your collection root: `~/Pictures/digikam4.db`
- Custom path configured in DigiKam settings

### Recommended Workflow

1. **Tag faces in DigiKam** using its People view — detect faces, confirm
   identities, and name people as usual
2. **Run sync-faces** to write the confirmed face data to file metadata:
   ```bash
   photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r --dry-run
   ```
3. Review the dry-run output, then run without `--dry-run` to write
4. If you later edit faces in DigiKam (rename, delete, add), re-run with
   `--force` to fully resync:
   ```bash
   photo-tools sync-faces ~/Pictures --db ~/Pictures/digikam4.db -r --force
   ```

### Safety

- The DigiKam database is opened **read-only** and is never modified
- Face regions are written alongside existing metadata (OCR regions, etc.)
- Use `--dry-run` to preview all changes before writing

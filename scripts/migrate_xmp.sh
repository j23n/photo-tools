#!/usr/bin/env bash
#
# migrate_xmp.sh — one-shot cleanup of legacy photo-tools metadata.
#
# Strips:
#   - lr:HierarchicalSubject, MicrosoftPhoto:LastKeywordXMP,
#     MediaPro:CatalogSets, MicrosoftPhoto:CategorySet
#   - The entire legacy photo-tools XMP namespace (any URI it was registered
#     under historically)
#   - Every keyword matching the pre-2026.1 patterns (year/*, month/*,
#     country/*, scene/*, ai:tagged, etc.)
#
# After running this, re-run `photo-tools tag` over the same files. Without the
# new photo-tools:TaggerVersion sentinel, every file is treated as fresh and
# rebuilt under the current schema.
#
# See docs/xmp-schema.md §4.
#
# Usage:
#   scripts/migrate_xmp.sh [--dry-run] PATH [PATH...]
#
# Options:
#   --dry-run    Show what would change without modifying files
#
# Examples:
#   scripts/migrate_xmp.sh ~/Photos
#   scripts/migrate_xmp.sh --dry-run /Volumes/photos/2024

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--dry-run] PATH [PATH...]" >&2
    exit 1
fi

# Locate the exiftool config (resolves the photo-tools namespace by name).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../src/photo_tools/exiftool_phototools.config"
if [[ ! -f "$CONFIG" ]]; then
    echo "error: exiftool config not found at $CONFIG" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the exiftool command.
#
# - We strip the photo-tools group entirely, then strip individual legacy
#   keyword patterns by sending exiftool a list of NoOps with conditional
#   removal. This is unfortunately verbose because exiftool needs an explicit
#   `-=value` per pattern; for prefixed keywords we use perl-style regex via
#   `-if + -tagsFromFile`. The simpler path is to dump keywords, filter, and
#   write back, which is what we do below per-file.
#
# We read each file's keyword list, filter out anything matching the legacy
# patterns, and write the survivors back. Per-file rather than batch so we can
# preserve user-added keywords accurately.
# ---------------------------------------------------------------------------

# Patterns that mark a keyword as photo-tools-owned (legacy).
LEGACY_BARE='^(weekend|weekday|screenshot|video|ai:tagged)$'
LEGACY_PREFIX='^(country|cc|region|city|neighborhood|landmark|scene|setting|object|animal|plant|vehicle|food|other|activity|event|weather|time|text|year|month|day|flash)/'

filter_keywords() {
    # stdin: one keyword per line; stdout: keywords that are NOT legacy.
    grep -viE "$LEGACY_BARE" | grep -viE "$LEGACY_PREFIX" || true
}

process_file() {
    local f="$1"

    # Pull current keyword/tag values as one-per-line.
    local keywords subject tagslist
    keywords=$(exiftool -s3 -f -sep $'\n' -IPTC:Keywords "$f" 2>/dev/null || true)
    subject=$(exiftool -s3 -f -sep $'\n' -XMP-dc:Subject "$f" 2>/dev/null || true)
    tagslist=$(exiftool -s3 -f -sep $'\n' -XMP-digiKam:TagsList "$f" 2>/dev/null || true)

    local kept_keywords kept_subject kept_tagslist
    kept_keywords=$(printf '%s\n' "$keywords"   | filter_keywords)
    kept_subject=$(printf '%s\n'  "$subject"    | filter_keywords)
    kept_tagslist=$(printf '%s\n' "$tagslist"   | filter_keywords)

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "=== $f ==="
        echo "  legacy fields to drop:"
        echo "    XMP-lr:HierarchicalSubject"
        echo "    MicrosoftPhoto:LastKeywordXMP"
        echo "    MediaPro:CatalogSets"
        echo "    MicrosoftPhoto:CategorySet"
        echo "    XMP-phototools:* (entire namespace)"
        echo "  keywords removed: $(diff <(printf '%s\n' "$keywords" | sort -u) <(printf '%s\n' "$kept_keywords" | sort -u) | grep '^<' | wc -l | tr -d ' ')"
        return 0
    fi

    # Write filtered values back, plus zero out the deprecated fields and the
    # entire photo-tools namespace.
    exiftool -config "$CONFIG" -overwrite_original -q \
        -IPTC:Keywords= -XMP-dc:Subject= -XMP-digiKam:TagsList= \
        -XMP-lr:HierarchicalSubject= \
        -MicrosoftPhoto:LastKeywordXMP= \
        -MediaPro:CatalogSets= \
        -MicrosoftPhoto:CategorySet= \
        -XMP-phototools:all= \
        "$f"

    # Re-add only the survivors. Empty list is fine.
    local args=()
    while IFS= read -r kw; do
        [[ -z "$kw" ]] && continue
        args+=("-IPTC:Keywords+=$kw")
    done <<< "$kept_keywords"
    while IFS= read -r kw; do
        [[ -z "$kw" ]] && continue
        args+=("-XMP-dc:Subject+=$kw")
    done <<< "$kept_subject"
    while IFS= read -r kw; do
        [[ -z "$kw" ]] && continue
        args+=("-XMP-digiKam:TagsList+=$kw")
    done <<< "$kept_tagslist"

    if [[ ${#args[@]} -gt 0 ]]; then
        exiftool -config "$CONFIG" -overwrite_original -q "${args[@]}" "$f"
    fi

    echo "migrated: $f"
}

# Walk every supplied path. Files: process directly. Dirs: recurse over
# common image extensions.
for target in "$@"; do
    if [[ -f "$target" ]]; then
        process_file "$target"
    elif [[ -d "$target" ]]; then
        # Use exiftool's own ext filter for portability.
        while IFS= read -r f; do
            process_file "$f"
        done < <(exiftool -r -if 'true' -p '$Directory/$FileName' \
                    -ext jpg -ext jpeg -ext png -ext tif -ext tiff \
                    -ext webp -ext heic -ext heif -ext dng \
                    "$target" 2>/dev/null)
    else
        echo "skip (not found): $target" >&2
    fi
done

if [[ $DRY_RUN -eq 0 ]]; then
    echo
    echo "Migration complete. Re-run 'photo-tools tag <PATH>' to repopulate"
    echo "the new schema."
fi

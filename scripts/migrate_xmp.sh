#!/usr/bin/env bash
#
# migrate_xmp.sh — one-shot cleanup of legacy photo-tools metadata.
#
# Strips:
#   - XMP-lr:HierarchicalSubject, MicrosoftPhoto:LastKeywordXMP,
#     MediaPro:CatalogSets, MicrosoftPhoto:CategorySet
#   - The entire photo-tools XMP namespace (any URI it was registered under)
#   - Every keyword matching the pre-2026.1 legacy patterns (year/*, month/*,
#     country/*, scene/*, ai:tagged, etc. — see LEGACY_PREFIX / LEGACY_BARE
#     below).
#
# After running this, re-run `photo-tools tag` over the same files. Without
# the photo-tools:TaggerVersion sentinel every file is treated as fresh and
# rebuilt under the current schema. See docs/xmp-schema.md §4.
#
# Usage:
#   scripts/migrate_xmp.sh [--dry-run] [-v|--verbose] PATH [PATH...]
#
# Options:
#   --dry-run      Report per-file changes without writing anything
#   -v, --verbose  Also log files that need no changes

set -euo pipefail

DRY_RUN=0
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -v|--verbose) VERBOSE=1; shift ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [--dry-run] [-v] PATH [PATH...]" >&2
    exit 1
fi

command -v exiftool >/dev/null 2>&1 || {
    echo "error: exiftool not found on PATH" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../src/photo_tools/exiftool_phototools.config"
if [[ ! -f "$CONFIG" ]]; then
    echo "error: exiftool config not found at $CONFIG" >&2
    exit 1
fi

# Legacy patterns (pre-2026.1). Keep in sync with constants.py
# LEGACY_PREFIXES / LEGACY_BARE_TAGS.
LEGACY_BARE_RE='^(weekend|weekday|screenshot|video|ai:tagged)$'
LEGACY_PREFIX_RE='^(country|cc|region|city|neighborhood|landmark|scene|setting|object|animal|plant|vehicle|food|other|activity|event|weather|time|text|year|month|day|flash)/'

is_legacy() {
    local k="$1"
    [[ "$k" =~ $LEGACY_BARE_RE ]] && return 0
    [[ "$k" =~ $LEGACY_PREFIX_RE ]] && return 0
    return 1
}

# Accumulator — total files scanned / changed.
TOTAL=0
CHANGED=0

process_file() {
    local f="$1"
    TOTAL=$((TOTAL + 1))

    # Pull each keyword field as one-value-per-line. Absent fields yield
    # empty strings (no -f flag → no placeholder).
    local kw_raw subj_raw tags_raw
    kw_raw=$(exiftool -s3 -sep $'\n' -IPTC:Keywords "$f" 2>/dev/null || true)
    subj_raw=$(exiftool -s3 -sep $'\n' -XMP-dc:Subject "$f" 2>/dev/null || true)
    tags_raw=$(exiftool -s3 -sep $'\n' -XMP-digiKam:TagsList "$f" 2>/dev/null || true)

    local -a legacy_kw=() legacy_subj=() legacy_tags=()
    while IFS= read -r k; do
        [[ -z "$k" ]] && continue
        is_legacy "$k" && legacy_kw+=("$k")
    done <<< "$kw_raw"
    while IFS= read -r k; do
        [[ -z "$k" ]] && continue
        is_legacy "$k" && legacy_subj+=("$k")
    done <<< "$subj_raw"
    while IFS= read -r k; do
        [[ -z "$k" ]] && continue
        is_legacy "$k" && legacy_tags+=("$k")
    done <<< "$tags_raw"

    # Probe deprecated / namespace fields. A non-empty value means the field
    # exists and should be stripped.
    local has_pt has_lr has_ms_last has_mediapro has_ms_cat
    has_pt=$(exiftool -config "$CONFIG" -s3 -XMP-phototools:all "$f" 2>/dev/null || true)
    has_lr=$(exiftool -s3 -XMP-lr:HierarchicalSubject "$f" 2>/dev/null || true)
    has_ms_last=$(exiftool -s3 -MicrosoftPhoto:LastKeywordXMP "$f" 2>/dev/null || true)
    has_mediapro=$(exiftool -s3 -MediaPro:CatalogSets "$f" 2>/dev/null || true)
    has_ms_cat=$(exiftool -s3 -MicrosoftPhoto:CategorySet "$f" 2>/dev/null || true)

    local n_kw=${#legacy_kw[@]} n_subj=${#legacy_subj[@]} n_tags=${#legacy_tags[@]}
    local keyword_total=$((n_kw + n_subj + n_tags))

    local has_any_field=0
    [[ -n "$has_pt$has_lr$has_ms_last$has_mediapro$has_ms_cat" ]] && has_any_field=1

    if [[ $keyword_total -eq 0 && $has_any_field -eq 0 ]]; then
        [[ $VERBOSE -eq 1 ]] && echo "clean:     $f"
        return 0
    fi

    CHANGED=$((CHANGED + 1))

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would migrate: $f"
        [[ $n_kw   -gt 0 ]] && printf '    IPTC:Keywords  -= %s\n' "${legacy_kw[@]}"
        [[ $n_subj -gt 0 ]] && printf '    dc:Subject     -= %s\n' "${legacy_subj[@]}"
        [[ $n_tags -gt 0 ]] && printf '    TagsList       -= %s\n' "${legacy_tags[@]}"
        [[ -n "$has_pt" ]]       && echo "    XMP-phototools:* (entire namespace)"
        [[ -n "$has_lr" ]]       && echo "    XMP-lr:HierarchicalSubject"
        [[ -n "$has_ms_last" ]]  && echo "    MicrosoftPhoto:LastKeywordXMP"
        [[ -n "$has_mediapro" ]] && echo "    MediaPro:CatalogSets"
        [[ -n "$has_ms_cat" ]]   && echo "    MicrosoftPhoto:CategorySet"
        return 0
    fi

    # Build the write command: remove each legacy value individually (keeps
    # user-added keywords intact) and zero out the deprecated fields /
    # photo-tools namespace wholesale.
    local -a args=(-config "$CONFIG" -overwrite_original -q)
    for k in "${legacy_kw[@]}";   do args+=("-IPTC:Keywords-=$k"); done
    for k in "${legacy_subj[@]}"; do args+=("-XMP-dc:Subject-=$k"); done
    for k in "${legacy_tags[@]}"; do args+=("-XMP-digiKam:TagsList-=$k"); done
    args+=(
        "-XMP-phototools:all="
        "-XMP-lr:HierarchicalSubject="
        "-MicrosoftPhoto:LastKeywordXMP="
        "-MediaPro:CatalogSets="
        "-MicrosoftPhoto:CategorySet="
        "$f"
    )

    exiftool "${args[@]}"
    echo "migrated:  $f"
}

# Walk every supplied path. Files: process directly. Dirs: find-based
# enumeration over common image extensions (NUL-separated for safety).
for target in "$@"; do
    if [[ -f "$target" ]]; then
        process_file "$target"
    elif [[ -d "$target" ]]; then
        while IFS= read -r -d '' f; do
            process_file "$f"
        done < <(find "$target" -type f \( \
              -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
           -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \
           -o -iname '*.heic' -o -iname '*.heif' -o -iname '*.dng' \
        \) -print0)
    else
        echo "skip (not found): $target" >&2
    fi
done

echo
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: $CHANGED / $TOTAL file(s) would be migrated."
else
    echo "Done: $CHANGED / $TOTAL file(s) migrated."
    [[ $CHANGED -gt 0 ]] && \
        echo "Next: re-run 'photo-tools tag <PATH>' to repopulate under the new schema."
fi

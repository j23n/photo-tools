"""
tags_cmd.py — `photo-tools tags` subcommand family.

list / search / delete / rename / clear / inspect (the inspect leaf is
defined in debug_viewer and registered here).
"""

import sys
from pathlib import Path

from photo_tools.config import get_config
from photo_tools.helpers import (
    add_tags,
    clear_all_tags,
    find_images,
    read_keywords_batch,
    remove_tags,
)
from photo_tools.logging_setup import get_logger

log = get_logger("tags")


# ---------------------------------------------------------------------------
# Tag index + bulk operations
# ---------------------------------------------------------------------------

def collect_tag_index(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    all_keywords = read_keywords_batch(paths)
    for path in paths:
        for tag in all_keywords.get(path, set()):
            index.setdefault(tag, []).append(path)
    return dict(sorted(index.items(), key=lambda kv: len(kv[1]), reverse=True))


def apply_tag_change(
    old: str,
    new: str | None,
    files: list[Path],
    dry_run: bool,
) -> None:
    if not files:
        return
    if dry_run:
        action = f"rename to '{new}'" if new else "delete"
        log.info("[DRY RUN] Would %s '%s' in %d file(s)", action, old, len(files))
        return

    remove_tags(files, [old], dry_run=dry_run)

    if new:
        for path in files:
            add_tags(path, [new], dry_run=False)


def bulk_delete_tags(
    tags: list[str],
    files: list[Path],
    dry_run: bool,
) -> None:
    """Delete multiple tags from multiple files."""
    if not files or not tags:
        return
    if dry_run:
        log.info("[DRY RUN] Would delete %d tag(s) from %d file(s)", len(tags), len(files))
        return

    batch_size = get_config().exiftool.batch_size
    total = len(files)
    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        print(f"\r  Deleting tags from files {i + 1}-{min(i + len(batch), total)}/{total} ...",
              end="", flush=True, file=sys.stderr)
        remove_tags(batch, tags, dry_run=dry_run)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def run_list_tags(args) -> None:
    paths = find_images(args.path)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)
    if not tag_index:
        print("No tags found.")
        return

    max_tag_len = max(len(t) for t in tag_index)
    for tag, files in tag_index.items():
        print(f"  {tag:<{max_tag_len}}  {len(files):>4} files")
    print(f"\n{len(tag_index)} unique tags across {len(paths)} files")


def run_delete_tag(args) -> None:
    if not args.tag and not args.pattern:
        log.error("Provide a tag name or --pattern")
        sys.exit(1)

    paths = find_images(args.path)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)

    if args.pattern:
        import re
        try:
            regex = re.compile(args.pattern)
        except re.error as e:
            log.error("Invalid regex pattern: %s", e)
            sys.exit(1)
        matched = {tag: files for tag, files in tag_index.items() if regex.fullmatch(tag)}
        if not matched:
            log.error("No tags match pattern '%s'", args.pattern)
            sys.exit(1)
        matched_tags = sorted(matched.keys())
        affected_files = list({f for files in matched.values() for f in files})
        log.info("Pattern '%s' matched %d tag(s) across %d file(s):",
                 args.pattern, len(matched_tags), len(affected_files))
        for tag in matched_tags:
            log.info("  %s  (%d files)", tag, len(matched[tag]))
        bulk_delete_tags(matched_tags, affected_files, args.dry_run)
        return

    tag = args.tag.lower()
    files = tag_index.get(tag, [])
    if not files:
        log.error("Tag '%s' not found in any files", tag)
        sys.exit(1)

    log.info("Deleting tag '%s' from %d file(s) ...", tag, len(files))
    bulk_delete_tags([tag], files, args.dry_run)


def run_rename_tag(args) -> None:
    paths = find_images(args.path)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)
    old = args.old.lower()
    new = args.new.lower()
    files = tag_index.get(old, [])
    if not files:
        log.error("Tag '%s' not found in any files", old)
        sys.exit(1)

    log.info("Renaming tag '%s' -> '%s' in %d file(s) ...", old, new, len(files))
    apply_tag_change(old, new, files, args.dry_run)


def run_clear_tags(args) -> None:
    paths = find_images(args.path)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    total = len(paths)
    if args.dry_run:
        log.info("[DRY RUN] Would clear tags from %d file(s) "
                 "(People/* and face regions preserved)", total)
        return

    try:
        answer = input(f"Are you sure you want to remove tags from {total} photo(s)? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        log.info("Aborted.")
        return

    log.info("Clearing tags from %d file(s) (People/* and face regions preserved) ...", total)
    clear_all_tags(paths, dry_run=False)


def run_search_tags(args) -> None:
    paths = find_images(args.path)
    if not paths:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    tag_index = collect_tag_index(paths)
    query = args.tag.lower()
    files = tag_index.get(query, [])
    if not files:
        log.error("Tag '%s' not found", query)
        sys.exit(1)

    for f in sorted(files):
        print(f)
    print(f"\n{len(files)} file(s) with tag '{query}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_tags_parser(subparsers) -> None:
    tags_parser = subparsers.add_parser(
        "tags",
        help="Tag management: list, search, delete, rename, clear.",
    )
    tags_sub = tags_parser.add_subparsers(dest="tags_command", required=True)

    p = tags_sub.add_parser("list", help="List all tags with file counts.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.set_defaults(func=run_list_tags)

    p = tags_sub.add_parser("search", help="List files that have a given tag.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("tag", help="Tag to search for")
    p.set_defaults(func=run_search_tags)

    p = tags_sub.add_parser("delete", help="Delete a tag (or regex pattern) from all files.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("tag", nargs="?", help="Exact tag to delete")
    p.add_argument("-p", "--pattern", help="Regex pattern to match tags for deletion")
    p.add_argument("-n", "--dry-run", action="store_true", help="Preview changes without writing")
    p.set_defaults(func=run_delete_tag)

    p = tags_sub.add_parser("rename", help="Rename a tag across all files.")
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("old", help="Tag to rename")
    p.add_argument("new", help="New tag name")
    p.add_argument("-n", "--dry-run", action="store_true", help="Preview changes without writing")
    p.set_defaults(func=run_rename_tag)

    p = tags_sub.add_parser(
        "clear",
        help="Wipe ALL tags and the photo-tools namespace, leaving only original EXIF.",
    )
    p.add_argument("path", type=Path, help="Target directory or file")
    p.add_argument("-n", "--dry-run", action="store_true", help="Preview changes without writing")
    p.set_defaults(func=run_clear_tags)

    from photo_tools.debug_viewer import add_inspect_subparser
    add_inspect_subparser(tags_sub)

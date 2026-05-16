"""CLI entry point for photo-tools.

Single entry point with all subcommands. Global flags (verbosity, log
spec, sidecar mode, config path) live on each subcommand, not on the
top-level parser — so `photo-tools tag -v PATH` works but
`photo-tools -v tag PATH` does not. This avoids the subparser-clobber
that comes from declaring the same flag at two levels with `parents=`.
"""

import argparse
import os
from pathlib import Path

from photo_tools.config import load_config
from photo_tools.logging_setup import setup_logging


def _build_global_parser() -> argparse.ArgumentParser:
    """Globals attached to every subcommand via `parents=[...]`."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable verbose (debug) logging for all subsystems")
    p.add_argument(
        "--log", dest="log_spec", default=None,
        metavar="SPEC",
        help="Per-subsystem log levels, e.g. "
             "'ocr=debug,ram=warning,geocoding=info'. "
             "Subsystems: tagging, geocoding, gps, ocr, ram, clip, "
             "landmarks, exif, dates, duplicates. "
             "Also reads PHOTOTOOLS_LOG env var.",
    )
    p.add_argument("--no-tui", dest="no_tui", action="store_true",
                   help="Disable the live status panel (auto-disabled "
                        "when stderr is not a TTY)")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to user config YAML overlay")
    p.add_argument("-s", "--xmp-sidecars", dest="xmp_sidecars",
                   action="store_true", default=False,
                   help="Mirror every metadata write into a sibling "
                        "IMG_1234.jpg.xmp sidecar (and merge it back on "
                        "read). Off by default; see docs/xmp-schema.md §1.3.")
    return p


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(prog="photo-tools")
    sub = parser.add_subparsers(dest="command", required=True)

    from photo_tools.autotag import build_tag_parser
    from photo_tools.build_landmarks import build_landmarks_parser
    from photo_tools.dates_cmd import build_dates_parser
    from photo_tools.duplicates import build_duplicates_parser
    from photo_tools.tags_cmd import build_tags_parser

    parents = [_build_global_parser()]
    build_tag_parser(sub, parents=parents)
    build_tags_parser(sub, parents=parents)
    build_dates_parser(sub, parents=parents)
    build_duplicates_parser(sub, parents=parents)
    build_landmarks_parser(sub, parents=parents)

    args = parser.parse_args()
    setup_logging(verbose=args.verbose, log_spec=args.log_spec)
    cfg = load_config(user_config_path=args.config)
    if args.xmp_sidecars:
        cfg.xmp.sidecars = True
    args.func(args)

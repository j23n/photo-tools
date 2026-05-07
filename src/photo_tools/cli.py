"""CLI entry point for photo-tools.

Single entry point with all subcommands. Top-level flags for verbosity
and config file are handled here before dispatching to subcommand handlers.
"""

import argparse
import os
from pathlib import Path

from photo_tools.config import load_config
from photo_tools.logging_setup import setup_logging


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(prog="photo-tools")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose (debug) logging for all subsystems")
    parser.add_argument(
        "--log", dest="log_spec", default=None,
        metavar="SPEC",
        help="Per-subsystem log levels, e.g. "
             "'ocr=debug,ram=warning,geocoding=info'. "
             "Subsystems: tagging, geocoding, gps, ocr, ram, clip, "
             "landmarks, exif, dates, duplicates. "
             "Also reads PHOTOTOOLS_LOG env var.",
    )
    parser.add_argument("--no-tui", dest="no_tui", action="store_true",
                        help="Disable the live status panel (auto-disabled "
                             "when stderr is not a TTY)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to user config YAML overlay")
    parser.add_argument("--xmp-sidecars", dest="xmp_sidecars",
                        action="store_true", default=None,
                        help="Mirror every metadata write into a sibling "
                             "IMG_1234.jpg.xmp sidecar (and merge it back on "
                             "read). Off by default; see docs/xmp-schema.md §1.3.")

    sub = parser.add_subparsers(dest="command", required=True)

    from photo_tools.autotag import build_tag_parser
    from photo_tools.build_landmarks import build_landmarks_parser
    from photo_tools.dates_cmd import build_dates_parser
    from photo_tools.duplicates import build_duplicates_parser
    from photo_tools.tags_cmd import build_tags_parser

    build_tag_parser(sub)
    build_tags_parser(sub)
    build_dates_parser(sub)
    build_duplicates_parser(sub)
    build_landmarks_parser(sub)

    args = parser.parse_args()
    setup_logging(verbose=args.verbose, log_spec=args.log_spec)
    cfg = load_config(user_config_path=args.config)
    if args.xmp_sidecars:
        cfg.xmp.sidecars = True
    args.func(args)

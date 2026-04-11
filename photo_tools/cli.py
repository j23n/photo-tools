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
                        help="Enable verbose (debug) logging")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to user config YAML overlay")

    sub = parser.add_subparsers(dest="command", required=True)

    from photo_tools.autotag import build_tag_parser
    from photo_tools.build_landmarks import build_landmarks_parser
    from photo_tools.drop_digikam_tags import build_drop_digikam_tags_parser
    from photo_tools.duplicates import build_similar_parser, build_tags_parser

    build_tag_parser(sub)
    build_tags_parser(sub)
    build_similar_parser(sub)
    build_landmarks_parser(sub)
    build_drop_digikam_tags_parser(sub)

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    load_config(user_config_path=args.config)
    args.func(args)

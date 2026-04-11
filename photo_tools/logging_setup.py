"""Centralized logging configuration for photo-tools."""

import logging


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("ppocr", "paddle", "paddlex"):
        logging.getLogger(name).setLevel(logging.ERROR)

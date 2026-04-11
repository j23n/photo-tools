import argparse
import sys
from pathlib import Path

import autotag
import duplicates


def build_landmarks_parser(subparsers) -> argparse.ArgumentParser:
    """Register the 'build-landmarks' subcommand."""
    sub = subparsers.add_parser(
        "build-landmarks",
        help="Build CLIP embedding database of notable landmarks from Wikidata.",
    )
    sub.add_argument("-o", "--output", type=Path,
                     default=Path.home() / ".local/share/photo-tools/landmarks.json",
                     help="Output path for landmarks.json")
    sub.add_argument("-l", "--limit", type=int, default=20000,
                     help="Max landmarks to fetch from Wikidata (default: 20000)")
    sub.add_argument("--clip-model", default="ViT-B-32",
                     help="CLIP model name (default: ViT-B-32)")
    sub.add_argument("--clip-pretrained", default="laion2b_s34b_b79k",
                     help="CLIP pretrained weights (default: laion2b_s34b_b79k)")
    sub.add_argument("--resume", action="store_true",
                     help="Skip landmarks already in output file")
    sub.add_argument("--test", action="store_true",
                     help="Build a small database (~200 landmarks in Rome and Bologna)")
    sub.add_argument("--images-per-landmark", type=int, default=10,
                     help="Target number of images per landmark (default: 10)")
    sub.add_argument("-v", "--verbose", action="store_true")
    sub.set_defaults(func=run_build_landmarks)
    return sub


def run_build_landmarks(args) -> None:
    import logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from build_landmarks import build_database
    build_database(
        limit=args.limit,
        output_path=args.output,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        resume=args.resume,
        test=args.test,
        images_per_landmark=args.images_per_landmark,
    )


def main():
    parser = argparse.ArgumentParser(prog="photo-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    autotag.build_tag_parser(sub)
    duplicates.build_tags_parser(sub)
    duplicates.build_similar_parser(sub)
    build_landmarks_parser(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

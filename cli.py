import argparse
import sys

import autotag
import duplicates


def main():
    parser = argparse.ArgumentParser(prog="photo-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    autotag.build_tag_parser(sub)
    duplicates.build_list_tags_parser(sub)
    duplicates.build_dedup_parser(sub)
    duplicates.build_similar_parser(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

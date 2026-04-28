#!/usr/bin/env python3
"""Print the tag hierarchy from data/ram_tag_mapping.yaml as a tree.

Usage:
    uv run python scripts/print_taxonomy.py
    uv run python scripts/print_taxonomy.py --category Scenes
    uv run python scripts/print_taxonomy.py --category Objects
    uv run python scripts/print_taxonomy.py --flat   # show leaf counts per node
"""
import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml not found — run: uv add pyyaml", file=sys.stderr)
    sys.exit(1)


def load_paths(yaml_path: Path) -> dict[str, set[str]]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    seen: dict[str, set[str]] = {"Objects": set(), "Scenes": set()}
    flat: dict[str, set[str]] = {"Objects": set(), "Scenes": set()}
    for mapping in data.values():
        if mapping is None:
            continue
        cat = mapping["category"]
        tag = mapping["tag"]
        seen[cat].add(tag)
        if "/" not in tag:
            flat[cat].add(tag)
    return seen, flat


def build_tree(paths: set[str]) -> dict:
    tree: dict = {}
    for path_str in sorted(paths):
        parts = path_str.split("/")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
    return tree


def print_tree(node: dict, indent: int = 0, counts: dict | None = None) -> None:
    for key in sorted(node):
        children = node[key]
        suffix = "/" if children else ""
        print("  " * indent + key + suffix)
        if children:
            print_tree(children, indent + 1, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tag hierarchy")
    parser.add_argument("--category", choices=["Objects", "Scenes"], default=None)
    parser.add_argument("--flat", action="store_true",
                        help="Also list tags that still have no hierarchy (flat leaf, no '/')")
    args = parser.parse_args()

    yaml_path = Path(__file__).parent.parent / "src/photo_tools/data/ram_tag_mapping.yaml"
    seen, flat = load_paths(yaml_path)

    categories = [args.category] if args.category else ["Objects", "Scenes"]

    for cat in categories:
        print(f"\n{'=' * 60}")
        print(f"  {cat}  ({len(seen[cat])} unique tags)")
        print("=" * 60)
        tree = build_tree(seen[cat])
        print_tree(tree)

        if args.flat and flat[cat]:
            print(f"\n  --- Still flat (no '/') ---")
            for tag in sorted(flat[cat]):
                print(f"    {tag}")


if __name__ == "__main__":
    main()

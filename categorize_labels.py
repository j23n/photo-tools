#!/usr/bin/env python3
"""
categorize_labels.py — Use the Claude API to categorize Apple VNClassifyImageRequest
labels into the photo-tools taxonomy categories.

Reads apple_labels_raw.txt, sends batches to Claude for categorization,
and writes apple_labels.json mapping category → [labels].

Usage:
    ANTHROPIC_API_KEY=sk-... uv run categorize_labels.py
    uv run categorize_labels.py --dry-run   # Print prompt without calling API
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

LABELS_FILE = Path(__file__).parent / "apple_labels_raw.txt"
OUTPUT_FILE = Path(__file__).parent / "apple_labels.json"

API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

CATEGORIES = {
    "animal": "All breeds (beagle, corgi, dalmatian...), wildlife (elephant, giraffe, fox...), insects (bee, butterfly, ant...), fish (trout, salmon, clownfish...), birds (eagle, parrot, penguin...), marine (whale, dolphin, seahorse...), reptiles/amphibians (lizard, frog, snake...). Also generic groupings: mammal, canine, feline, reptile, bird, fish, insect, arachnid, etc.",
    "food": "Dishes (biryani, ramen, pizza, sushi...), ingredients (garlic, onion, tomato...), baked goods (croissant, muffin, bagel...), desserts (tiramisu, cheesecake, brownie...), beverages (coffee, beer, smoothie, cocktail, tea_drink...), condiments, snacks. Includes prepared/raw food items.",
    "plant": "Flowers (rose, tulip, orchid, daisy, dahlia...), trees (oak_tree, palm_tree, maple_tree, sequoia, willow...), general (cactus, bonsai, ferns, moss, ivy, clover, foliage, blossom, shrub...).",
    "vehicle": "Cars (car, convertible, sportscar, suv, jeep...), trucks (truck, semi_truck, firetruck...), two-wheel (bicycle, motorcycle, scooter...), water (boat, sailboat, kayak, yacht, cruise_ship...), air (airplane, helicopter...), rail (train, streetcar, monorail...), other (atv, go_kart, sled, rickshaw...).",
    "object": "Everything NOT in another category: furniture (chair, table, sofa, bed...), electronics (laptop, phone, television, camera...), tools (hammer, wrench, scissors...), containers (bottle, cup, bowl, jar...), clothing/accessories (hat, sunglasses, backpack, shoes...), instruments (guitar, piano, drum...), household (candle, clock, lamp, pillow...), toys (doll, puppet, stuffed_animals...). This is the catch-all for physical things.",
    "scene": "Landscapes/environments: beach, canyon, cave, cityscape, cliff, creek, desert, forest, garden, glacier, harbour, hill, island, jungle, lake, mountain, ocean, orchard, park, path, pond, river, shore, trail, vineyard, volcano, waterfall, wetland. Indoor scenes: aquarium, bar, bathroom_room, bedroom, classroom, dining_room, garage, kitchen_room, library, living_room, museum, restaurant, stadium.",
    "activity": "Sports (baseball, basketball, boxing, cycling, diving, golf, gymnastics, hiking, hockey, rugby, skiing, soccer, surfing, swimming, tennis, volleyball, wrestling, yoga...), hobbies (camping, cooking, dancing, fishing, hunting, kayaking, painting, photography, sewing...), actions (juggling, karaoke, singing...).",
    "event": "wedding, graduation, carnival, parade, concert, circus, ceremony, celebration, performance, birthday_cake (event context), christmas_tree/christmas_decoration (event context), jack_o_lantern (halloween), easter_egg (easter). Small set of event-type labels.",
    "other": "ONLY for labels that genuinely don't fit any of the 8 categories above. This should be a small set. Generic parent labels like 'animal', 'food', 'mammal', 'bird' still belong in their natural category (animal, food, etc.). Use 'other' for things like: screenshot, daytime, night_sky, blue_sky, cloudy, haze, stormy, people-related meta-labels (adult, child, teen, baby, crowd, people), and truly uncategorizable labels.",
}

BATCH_SIZE = 200


def build_prompt(labels: list[str]) -> str:
    cat_descriptions = "\n".join(
        f"  - {name}: {desc}" for name, desc in CATEGORIES.items()
    )
    labels_str = ", ".join(labels)
    return f"""\
Categorize each Apple Vision label into exactly one of these categories:

{cat_descriptions}

Rules:
- Every label MUST be assigned to exactly one category. No label may be left out.
- When a label could fit multiple categories, pick the MOST SPECIFIC one. E.g. "clam" is food (not animal), "honey" is food (not object), "fishing" is activity (not scene).
- Generic/abstract labels (e.g. "animal", "food", "mammal", "fruit") still get assigned to their most natural category — we handle filtering separately.
- birthday_cake, christmas_tree, christmas_decoration, jack_o_lantern, easter_egg → event (they indicate events, not just objects).
- Dog/cat breeds (beagle, corgi, poodle, siamese, etc.) → animal.
- Prepared food items (sushi, pizza, ramen, steak, etc.) → food.
- Musical instruments (guitar, piano, drum, etc.) → object.
- Sports equipment (baseball_bat, golf_club, racquet, etc.) → object, but the sport itself (baseball, golf, tennis) → activity.

Labels to categorize:
{labels_str}

Return ONLY a JSON object mapping each label to its category. Example:
{{"beagle": "animal", "pizza": "food", "beach": "scene", "guitar": "object"}}"""


def categorize_batch(labels: list[str]) -> dict[str, str]:
    prompt = build_prompt(labels)
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())

    content = body["content"][0]["text"].strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    return json.loads(content[start:end])


def main():
    dry_run = "--dry-run" in sys.argv

    labels = [l.strip() for l in LABELS_FILE.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(labels)} labels from {LABELS_FILE}")

    if dry_run:
        print("\n--- PROMPT (first batch) ---")
        print(build_prompt(labels[:BATCH_SIZE]))
        print(f"\n--- Would send {(len(labels) + BATCH_SIZE - 1) // BATCH_SIZE} batch(es) ---")
        return

    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    all_mappings: dict[str, str] = {}

    for i in range(0, len(labels), BATCH_SIZE):
        batch = labels[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(labels) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} labels) ...", end="", flush=True)
        result = categorize_batch(batch)
        all_mappings.update(result)
        print(f" got {len(result)} mappings")

    # Check for missing labels
    missing = [l for l in labels if l not in all_mappings]
    if missing:
        print(f"\nWARNING: {len(missing)} labels not categorized: {missing[:20]}...")

    # Invert: category → [labels]
    by_category: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    for label, cat in sorted(all_mappings.items()):
        if cat not in by_category:
            print(f"  WARNING: unknown category '{cat}' for label '{label}', mapping to 'skip'")
            cat = "skip"
        by_category[cat].append(label)

    # Print summary
    print(f"\nCategorization summary:")
    for cat, cat_labels in by_category.items():
        print(f"  {cat:10s}: {len(cat_labels):4d} labels")

    OUTPUT_FILE.write_text(json.dumps(by_category, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

# photo-tools XMP / IPTC schema

This document defines the metadata fields photo-tools writes to image files,
the taxonomy used for keyword tags, and the conventions consumers should rely
on. It is the source of truth — code in `src/photo_tools/` implements this
spec, and the migration script in `scripts/` removes anything that doesn't
conform.

---

## 1. Output fields

photo-tools writes exactly **two keyword/tag fields** plus a small custom XMP
namespace for tool-private metadata. Everything else listed under "Fields we
do not write" must remain absent (the migration script strips them).

### 1.1 Keyword/tag fields

| Field | Content | Separator | Notes |
| --- | --- | --- | --- |
| `dc:subject` (mirrored to `IPTC:Keywords`) | Leaf names only | flat list | MWG-aligned; what Lightroom and most tools actually read |
| `digiKam:TagsList` | Full hierarchy paths | `/` | digiKam's native hierarchy field |

Both fields use **Titlecase** for every path segment (see §3).

For the example photo from issue #10:

```
dc:subject       = [Chiara, Johannes, Italy, Lazio, Rome, Municipio Roma I,
                    Balustrade, Rail, Building, Cityscape]
IPTC:Keywords    = (same as dc:subject)
digiKam:TagsList = [
    People/Chiara, People/Johannes,                  # written by digiKam, not us
    Places/Italy/Lazio/Rome/Municipio Roma I,
    Objects/Balustrade, Objects/Rail,
    Scenes/Building, Scenes/Cityscape,
]
```

### 1.2 photo-tools custom namespace

Namespace prefix: **`photo-tools`** (also surfaces as `XMP-phototools` via the
exiftool config).

Namespace URI: **`https://github.com/j23n/photo-tools/ns/1.0/`**

| Field | Type | Purpose |
| --- | --- | --- |
| `photo-tools:TaggerVersion` | string (`"YYYY.N"`) | Sentinel — presence means file was tagged by this tool. Older versions trigger re-tag on next run. |
| `photo-tools:TaggedAt` | ISO 8601 timestamp | When the file was last tagged. |
| `photo-tools:CountryCode` | ISO 3166-1 alpha-2 (uppercase, e.g. `IT`) | Country code from reverse geocoding. Kept out of the keyword space. |
| `photo-tools:CLIPEmbedding` | base64 of float32 vector | Cached image embedding for similarity search. |
| `photo-tools:CLIPModel` | string (e.g. `ViT-B-32/laion2b_s34b_b79k`) | Model identifier for the cached embedding. |

### 1.3 Fields we do not write

These were emitted by older versions or by other tools and are deliberately
absent from output. The migration script removes them on existing files:

- `XMP-lr:HierarchicalSubject` — redundant with `digiKam:TagsList`; we don't mirror it.
- `MicrosoftPhoto:LastKeywordXMP`
- `MediaPro:CatalogSets`
- `MicrosoftPhoto:CategorySet`

### 1.4 Sentinel mechanics

- A file is considered "already tagged" iff `photo-tools:TaggerVersion` is
  present.
- If the stored version differs from the current `TaggerVersion`, the file is
  re-tagged automatically.
- `--force` re-tags regardless. `--clear-all` wipes everything first.
- The legacy `ai:tagged` keyword sentinel is no longer written. The migration
  script removes it from existing files.

---

## 2. Taxonomy

Top-level roots in `digiKam:TagsList`:

```
People/<name>                                        ← digiKam-owned
Places/<Country>[/<Region>[/<City>[/<Neighborhood>]]]
Objects/<thing>
Scenes/<scene>
```

### 2.1 People

`People/*` is **owned by digiKam face recognition**. photo-tools never writes
or removes anything under this root. Migration leaves `People/*` untouched.

### 2.2 Places

A single nested path from reverse geocoding. Missing levels collapse — there
are no placeholders.

```
Places/Italy/Lazio/Rome/Municipio Roma I    ← all four available
Places/Italy/Rome                            ← region missing, collapse
Places/France                                ← only country known
```

The country code is written separately to `photo-tools:CountryCode` (see
§1.2) and is not part of the keyword space.

### 2.3 Objects

Concrete things detected by RAM++. Configured in `taxonomy.py`:

- `max_tags`: 5
- `min_confidence`: 0.6 *(plumbed; not yet enforced — see note below)*

### 2.4 Scenes

Scene/setting classification from RAM++. Configured in `taxonomy.py`:

- `max_tags`: 3
- `min_confidence`: 0.4 *(plumbed; not yet enforced — see note below)*

> **Note on confidence enforcement.** `inference_ram` returns predicted tag
> names without per-tag scores, so the current `RAMTagger._map_tags`
> respects only `max_tags`. Wiring the model's raw logits through to
> per-tag scores is a follow-up — see `src/photo_tools/ram_tagger.py`.

### 2.5 What's deliberately excluded

These were written by older versions and are dropped:

- **Date tags** (`Year/*`, `Month/*`, `Weekday/*`, `Weekend`, `DayOfMonth/*`).
  EXIF `DateTimeOriginal` is the canonical source; consumers derive year/month
  themselves.
- **`Other/*` umbrella** (including `other/people`). The "is there a person"
  signal duplicates digiKam face regions and pollutes search.
- **`flash/fired`**. Already in EXIF.
- **`screenshot`**. Heuristic was unreliable.
- **`cc/<code>`**. Replaced by `photo-tools:CountryCode` in the namespace.
- **Per-category `Activity/*`, `Event/*`, `Weather/*`, `Time/*`**. Folded into
  Objects/Scenes where they made sense; otherwise dropped.

---

## 3. Casing rules

Every segment of every keyword/tag path is **Titlecase**, including
geocoder-derived values.

- ASCII text, single-word: `str.title()` is sufficient (`rome` → `Rome`).
- Multi-word, mixed-language, Roman numerals: use the
  [`titlecase`](https://pypi.org/project/titlecase/) PyPI library
  (`municipio roma i` → `Municipio Roma I`, `città del vaticano` → `Città del
  Vaticano`).

The `titlecase` library handles small words ("of", "the", "del"), Roman
numerals, and acronyms out of the box.

---

## 4. Migration

Existing files written by older versions of photo-tools must be cleaned up
once before re-tagging. Scripts under `scripts/migrate_xmp.sh` do this in two
stages:

1. **Strip** the legacy `photo-tools` XMP namespace contents (any URI),
   `lr:HierarchicalSubject`, `MicrosoftPhoto:LastKeywordXMP`,
   `MediaPro:CatalogSets`, `MicrosoftPhoto:CategorySet`, and any keyword
   matching the legacy patterns:

   ```
   year/*, month/*, day/*, weekday, weekend, screenshot, flash/*,
   country/*, cc/*, region/*, city/*, neighborhood/*,
   object/*, scene/*, other/*, animal/*, food/*, plant/*, vehicle/*,
   activity/*, event/*, weather/*, time/*, setting/*, text/*,
   ai:tagged
   ```

2. **Re-run** `photo-tools tag` over the same files. Without the
   `photo-tools:TaggerVersion` sentinel, every file is treated as new and
   tagged from scratch under the new schema.

The migration is one-shot and idempotent. Re-tagging is acceptable cost since
this project is pre-alpha.

---

## 5. Interop notes

- **MWG / Lightroom / digiKam** all read leaf names from `dc:subject`. Keeping
  it leaf-only matches the convention.
- **digiKam** writes `People/*` itself (face recognition) and reads/writes
  `digiKam:TagsList` for hierarchy. Round-trip is clean.
- **Lightroom** writes hierarchy to `lr:HierarchicalSubject` (`|` separator).
  We do not mirror to it; users who want Lightroom-side hierarchy can run
  exiftool to copy `digiKam:TagsList` → `lr:HierarchicalSubject` themselves.
- **photo-tools custom namespace** is registered in
  `src/photo_tools/exiftool_phototools.config`. exiftool needs `-config
  exiftool_phototools.config` to read/write the namespace fields by name; the
  raw XMP is always preserved regardless.

---

## 6. Versioning

`TaggerVersion` follows `YYYY.N` (e.g. `2026.1`). Bump `N` when:

- Any field in §1 is added, removed, or repurposed.
- Taxonomy roots in §2 change shape.
- Casing or normalization rules in §3 change.

A version bump triggers automatic re-tagging on next `photo-tools tag` run
because the stored `TaggerVersion` no longer matches the running one.
